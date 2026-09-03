"""
End-to-end Whisper-small transcription on the Snapdragon X2 Elite NPU.

Pipeline:
  WAV → log-mel spectrogram → encoder (NPU) → greedy decoder loop (NPU) → BPE text.

Usage:
  python npu_transcribe.py path/to/audio.wav
  python npu_transcribe.py --record 5           # record 5 s from default mic

Requires:
  - onnxruntime, onnxruntime-qnn (already installed)
  - transformers (for WhisperTokenizer)
  - scipy (for .wav IO)
  - sounddevice (optional, only for --record)
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn

ROOT = os.path.dirname(os.path.abspath(__file__))
ENCODER = os.path.join(ROOT, "intermediates", "encoder_model_merged.onnx")
DECODER = os.path.join(ROOT, "intermediates", "decoder_model_merged.onnx")

# Whisper constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
CHUNK_SECONDS = 30
N_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE  # 480000
N_FRAMES = N_SAMPLES // HOP              # 3000
N_AUDIO_CTX = 1500
N_TEXT_CTX = 200
N_LAYER = 12
N_HEAD = 12
N_STATE = 768
HEAD_DIM = N_STATE // N_HEAD  # 64


# --- mel filterbank ----------------------------------------------------------
def _mel_filter_bank() -> np.ndarray:
    """80-bin mel filterbank matching openai-whisper's mel_filters.npz."""
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    fmin, fmax = 0.0, SAMPLE_RATE / 2
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), N_MELS + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((N_FFT + 1) * hz_pts / SAMPLE_RATE).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c > l:
            fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Whisper-style log-mel: STFT magnitude² → mel FB → log10 → clamp."""
    if audio.shape[0] < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - audio.shape[0]))
    else:
        audio = audio[:N_SAMPLES]
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    # Frame signal (pad with N_FFT//2 reflect to match whisper's center=True)
    pad = N_FFT // 2
    x = np.pad(audio, (pad, pad), mode="reflect")
    n_frames = 1 + (x.shape[0] - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, N_FFT),
        strides=(x.strides[0] * HOP, x.strides[0]),
    ).copy()
    frames *= window
    spec = np.fft.rfft(frames, n=N_FFT, axis=-1)
    mag = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel = (_mel_filter_bank() @ mag.T)  # (80, n_frames)
    mel = np.log10(np.maximum(mel, 1e-10))
    mel = np.maximum(mel, mel.max() - 8.0)
    mel = (mel + 4.0) / 4.0
    # Trim or pad to exactly N_FRAMES=3000
    if mel.shape[1] < N_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, N_FRAMES - mel.shape[1])))
    else:
        mel = mel[:, :N_FRAMES]
    return mel.astype(np.float32)[None, ...]  # (1, 80, 3000)


# --- audio IO ---------------------------------------------------------------
def load_wav(path: str) -> np.ndarray:
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # naive linear resample
        xp = np.arange(audio.shape[0])
        xnew = np.linspace(0, audio.shape[0] - 1, int(audio.shape[0] * SAMPLE_RATE / sr))
        audio = np.interp(xnew, xp, audio).astype(np.float32)
    return audio.astype(np.float32)


def record(seconds: float) -> np.ndarray:
    import sounddevice as sd
    print(f"Recording {seconds}s at {SAMPLE_RATE} Hz...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


# --- ORT session setup ------------------------------------------------------
_ep_registered = False
CACHE_DIR = os.path.join(ROOT, "qnn_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def build_session(onnx_path: str) -> ort.InferenceSession:
    """Build an ORT+QNN EP session. First call per-model produces a .onnx_ctx.onnx
    context-binary sidecar in qnn_cache/; subsequent calls reuse it (no JIT)."""
    global _ep_registered
    if not _ep_registered:
        ort.register_execution_provider_library(
            onnxruntime_qnn.get_ep_name(), onnxruntime_qnn.get_library_path()
        )
        _ep_registered = True
    npu = next(
        d for d in ort.get_ep_devices()
        if d.ep_name == "QNNExecutionProvider"
        and getattr(d.device, "type", None) is not None
        and "NPU" in str(d.device.type).upper()
    )
    ctx_path = os.path.join(CACHE_DIR, os.path.basename(onnx_path).replace(".onnx", "_ctx.onnx"))
    use_cache = os.path.exists(ctx_path)
    load_path = ctx_path if use_cache else onnx_path

    so = ort.SessionOptions()
    so.log_severity_level = 3
    if not use_cache:
        so.add_session_config_entry("ep.context_enable", "1")
        so.add_session_config_entry("ep.context_file_path", ctx_path)
        so.add_session_config_entry("ep.context_embed_mode", "1")
    so.add_provider_for_devices(
        [npu],
        {"backend_path": onnxruntime_qnn.get_qnn_htp_path(), "htp_performance_mode": "burst"},
    )
    print(f"  loading {'cached context' if load_path == ctx_path else 'fresh ONNX (will JIT + cache)'}: {os.path.basename(load_path)}")
    return ort.InferenceSession(load_path, sess_options=so)


# --- decoder loop -----------------------------------------------------------
def _decode_step(dec, tok_id: int, pos: int, kcc, vcc, kcs, vcs):
    x = np.array([tok_id], dtype=np.int32)
    offset = np.array([pos], dtype=np.int32)
    logits, kcs_new, vcs_new = dec.run(
        None,
        {"x": x, "offset": offset,
         "k_cache_cross": kcc, "v_cache_cross": vcc,
         "k_cache_self": kcs, "v_cache_self": vcs},
    )
    return logits, kcs_new, vcs_new


def transcribe(mel: np.ndarray, enc, dec, tokenizer) -> str:
    t0 = time.perf_counter()
    kcc, vcc = enc.run(None, {"input_features": mel})
    print(f"  encoder:  {(time.perf_counter() - t0) * 1000:.1f} ms")

    prefix = list(tokenizer.prefix_tokens)  # [50258, 50259, 50359, 50363]
    eot = tokenizer.eos_token_id
    tokens = list(prefix)
    kcs = np.zeros((N_LAYER, N_HEAD, HEAD_DIM, N_TEXT_CTX), dtype=np.float32)
    vcs = np.zeros((N_LAYER, N_HEAD, N_TEXT_CTX, HEAD_DIM), dtype=np.float32)

    t0 = time.perf_counter()
    # Prime cache by feeding every prefix token in order. The last call's logits
    # predict the first content token.
    logits = None
    for i, tok_id in enumerate(prefix):
        logits, kcs, vcs = _decode_step(dec, tok_id, i, kcc, vcc, kcs, vcs)

    # Greedy sampling
    while len(tokens) < N_TEXT_CTX:
        next_tok = int(np.argmax(logits.reshape(-1)))
        if next_tok == eot:
            break
        tokens.append(next_tok)
        logits, kcs, vcs = _decode_step(dec, next_tok, len(tokens) - 1, kcc, vcc, kcs, vcs)

    print(f"  decoder:  {(time.perf_counter() - t0) * 1000:.1f} ms ({len(tokens) - len(prefix)} tokens)")
    return tokenizer.decode(tokens[len(prefix):], skip_special_tokens=True)


# --- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="Path to WAV file")
    ap.add_argument("--record", type=float, default=None, help="Record N seconds from mic")
    args = ap.parse_args()

    if args.record is not None:
        audio = record(args.record)
    elif args.audio:
        audio = load_wav(args.audio)
    else:
        ap.error("provide WAV path or --record seconds")

    print(f"Audio: {audio.shape[0] / SAMPLE_RATE:.2f} s, {audio.shape[0]} samples")

    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="en", task="transcribe"
    )

    t0 = time.perf_counter()
    mel = log_mel(audio)
    print(f"mel shape: {mel.shape}  ({(time.perf_counter() - t0) * 1000:.1f} ms)")

    print("Loading encoder on NPU...")
    enc = build_session(ENCODER)
    print("Loading decoder on NPU...")
    dec = build_session(DECODER)

    print("\nTranscribing...")
    text = transcribe(mel, enc, dec, tokenizer)
    print("\n=== TRANSCRIPTION ===")
    print(text)


if __name__ == "__main__":
    main()
