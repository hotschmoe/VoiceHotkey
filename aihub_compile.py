"""
Submit Whisper encoder + decoder ONNX to Qualcomm AI Hub, targeting
Snapdragon X2 Elite CRD, and download QNN context binaries for V73.

Prereqs:
  pip install qai-hub
  qai-hub configure --api_token <YOUR_TOKEN>   (from aihub.qualcomm.com)

Outputs:
  VoiceHotkey/models/encoder_model_htp.bin
  VoiceHotkey/models/decoder_model_htp.bin
"""
import os
import sys
import time
import qai_hub as hub

ROOT = os.path.dirname(os.path.abspath(__file__))
ENCODER_ONNX = os.path.join(ROOT, "intermediates", "encoder_model_merged.onnx")
DECODER_ONNX = os.path.join(ROOT, "intermediates", "decoder_model_merged.onnx")
MODELS_OUT = os.path.join(ROOT, "VoiceHotkey", "models")
os.makedirs(MODELS_OUT, exist_ok=True)

DEVICE = hub.Device("Snapdragon X2 Elite CRD")
OPTIONS = "--target_runtime qnn_context_binary --compute_unit npu"  # QAIRT 2.45 default

# Whisper small dims
N_LAYER, N_HEAD, N_STATE = 12, 12, 768
N_AUDIO_CTX, N_TEXT_CTX = 1500, 200
HEAD_DIM = N_STATE // N_HEAD  # 64

ENCODER_INPUTS = {
    "input_features": ((1, 80, N_AUDIO_CTX * 2), "float32"),
}

DECODER_INPUTS = {
    "x":             ((1,), "int32"),
    "offset":        ((1,), "int32"),
    "k_cache_cross": ((N_LAYER, N_HEAD, HEAD_DIM, N_AUDIO_CTX), "float32"),
    "v_cache_cross": ((N_LAYER, N_HEAD, N_AUDIO_CTX, HEAD_DIM), "float32"),
    "k_cache_self":  ((N_LAYER, N_HEAD, HEAD_DIM, N_TEXT_CTX), "float32"),
    "v_cache_self":  ((N_LAYER, N_HEAD, N_TEXT_CTX, HEAD_DIM), "float32"),
}


def poll_until_done(job, label):
    """Poll without using job.wait() — it prints non-cp1252 chars and crashes on Windows."""
    last = None
    while True:
        status = job.get_status()
        state = status.state.name if hasattr(status.state, "name") else str(status.state)
        if state != last:
            print(f"  [{label}] status: {state}")
            last = state
        if status.finished:
            return status
        time.sleep(15)


def compile_one(name, onnx_path, input_specs, out_bin, existing_job_id=None):
    print(f"\n=== {name} ===")
    if existing_job_id:
        print(f"Reconnecting to existing job {existing_job_id}")
        job = hub.get_job(existing_job_id)
    else:
        print(f"Uploading {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)...")
        job = hub.submit_compile_job(
            model=onnx_path,
            device=DEVICE,
            options=OPTIONS,
            input_specs=input_specs,
            name=f"whisper_small_{name}_v73",
        )
        print(f"Job submitted: {job.job_id}")
        print(f"  Dashboard: {job.url}")
    status = poll_until_done(job, name)
    if status.failure:
        print(f"FAILED: {status.message}", file=sys.stderr)
        sys.exit(1)
    print(f"Downloading -> {out_bin}")
    job.download_target_model(out_bin)
    print(f"Done: {out_bin} ({os.path.getsize(out_bin) / 1e6:.1f} MB)")


def main():
    # Usage:
    #   python aihub_compile.py              -> both, fresh uploads
    #   python aihub_compile.py encoder      -> just encoder
    #   python aihub_compile.py decoder      -> just decoder
    #   python aihub_compile.py resume encoder <job_id>  -> reconnect
    args = sys.argv[1:]
    if args and args[0] == "resume":
        name = args[1]
        jid = args[2]
        out = os.path.join(MODELS_OUT, f"{name}_model_htp.bin")
        specs = ENCODER_INPUTS if name == "encoder" else DECODER_INPUTS
        onnx = ENCODER_ONNX if name == "encoder" else DECODER_ONNX
        compile_one(name, onnx, specs, out, existing_job_id=jid)
        return
    which = args[0] if args else "both"
    if which in ("encoder", "both"):
        compile_one("encoder", ENCODER_ONNX, ENCODER_INPUTS,
                    os.path.join(MODELS_OUT, "encoder_model_htp.bin"))
    if which in ("decoder", "both"):
        compile_one("decoder", DECODER_ONNX, DECODER_INPUTS,
                    os.path.join(MODELS_OUT, "decoder_model_htp.bin"))


if __name__ == "__main__":
    main()
