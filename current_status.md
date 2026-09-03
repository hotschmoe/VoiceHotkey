# VoiceHotkey Project Status

## Goal
System-wide voice-to-text on ASUS Zenbook A16 (Snapdragon X2 Elite Extreme, SC8480XP) using Qualcomm's Hexagon NPU. Hotkey (Ctrl+Space) triggers mic capture, Whisper transcription on NPU, and paste-at-cursor.

## Hardware
- **SoC**: Qualcomm SC8480XP (Snapdragon X2 Elite Extreme)
- **DSP**: Hexagon V81 (soc_model 88)
- **RAM**: 48GB unified
- **OS**: Windows 11 ARM64

## Status (2026-09-02): WORKING — automatic insertion repaired

The transcription pipeline was working, but automatic insertion was not. The
C# `INPUT` union only declared `KEYBDINPUT`, which made
`Marshal.SizeOf<INPUT>()` 32 bytes on ARM64. Native Windows expects the full
40-byte union (whose largest member is `MOUSEINPUT`), and `SendInput` rejects a
call when `cbSize` is wrong. `Program.cs` now declares all native union members,
so `KEYEVENTF_UNICODE` text injection can reach the foreground application. The
release build and Desktop shortcut were updated on 2026-09-02.

The QNN cache now uses external mode (`ep.context_embed_mode=0`) and ships each
generated `.onnx` wrapper with its companion `.bin` payload. ONNX Runtime QNN
1.24.4 has a known bug where embedded caches for multi-partition graphs are
generated without context data on all `EPContext` nodes and then fail to reload.

## Original transcription milestone (2026-04-18)

End-to-end dictation on the NPU from a plain C# console app. Sample timings observed:
- 2.2 s audio → 1276 ms transcription
- 2.8 s audio → 652 ms
- 1.3 s audio → 569 ms

Ctrl+Space toggles dictation; transcription pastes at the text cursor in any app.

## Architecture

Runtime stack (all from NuGet / pip, no VoiceAI SDK):
1. **NAudio** — mic capture, 16 kHz mono PCM16
2. **Pure-C# log-mel** — Hann window + MathNet.Numerics FFT → 80×3000 float32
3. **ONNX Runtime 1.24.4 + QNN EP plugin** — runs `encoder_model_merged.onnx` + `decoder_model_merged.onnx` on Hexagon V81
4. **Decode-only Whisper BPE in C#** — reads HuggingFace `vocab.json` + `added_tokens.json`, reverse GPT-2 byte-level encoding
5. **Win32 clipboard + SendInput** — Ctrl+V to paste at cursor

Cold launch JIT-compiles both graphs for the NPU (~30 s once) and writes small
`*_ctx.onnx` wrappers plus external `*_ctx_qnn.bin` payloads to `qnn_cache/`.
Subsequent launches load from cache in <1 s.

## Project Layout
```
voice_project/
  VoiceHotkey/
    Program.cs              — hotkey, mic, paste
    Transcriber.cs          — ORT+QNN sessions, encoder+decoder loop
    LogMel.cs               — log-mel spectrogram (MathNet FFT)
    WhisperTokenizer.cs     — decode-only BPE
    VoiceHotkey.csproj      — NAudio, Microsoft.ML.OnnxRuntime.QNN 1.24.4, MathNet.Numerics
    tokenizer/              — vocab.json, added_tokens.json (from HF openai/whisper-small)
  intermediates/            — encoder_model_merged.onnx (414 MB), decoder_model_merged.onnx (721 MB)
  npu_transcribe.py         — Python reference pipeline (identical pipeline, easy to iterate)
  ort_smoke_test.py         — minimal ORT+QNN EP load verification
  qnn_cache/                — auto-created on first run; holds JIT-compiled context binaries
  export_onnx_dynamo.py     — ONNX export from PyTorch whisper (tanh-GELU decoder variant)
  VoiceAI_ASR_Community_v2.3.0.0/  — (unused) original SDK source; kept only for the GELU patch in redefined_modules
  aihub_compile.py          — (unused) AI Hub compile driver from the abandoned raw-QNN path
```

## How we got here (short version)

1. Tried the VoiceAI WhisperComponent SDK with pre-built V81/soc_model=88 context binaries. App wouldn't load them — the SDK only shipped V73 runtime libs.
2. Swapped V73 → V81 libs from QAIRT 2.45: ran into `0x80000406 Unable to load lib` — retail Windows refuses to load unsigned Hexagon Skels, and Secure Boot + OEM lock prevent `bcdedit /set testsigning on`.
3. Tried the driver-signed 2.41 stack (`qcnspmcdm8480.inf`): FastRPC session loaded, but binary/runtime version check failed — and AI Hub doesn't compile for 2.41.
4. Tried the effectpack-signed 2.42 stack (`microsofteffectpack_extension`): signed but restricted to MS system apps, so our console exe couldn't use it.
5. **Pivoted to ORT + QNN EP** (Microsoft's sanctioned Copilot+ path). The pip `onnxruntime-qnn` / NuGet `Microsoft.ML.OnnxRuntime.QNN` packages ship a fully-signed QAIRT stack that loads on retail without any signing dance.
6. Rewrote transcription in C# (~400 lines across 4 files) using the ONNX exports we already had. Kept hotkey / mic / paste from the original Program.cs.

## Key facts / gotchas
- Retail Snapdragon X Windows will not load unsigned Hexagon Skels. Secure Boot is OEM-locked on the ASUS Zenbook A16, which blocks `testsigning on`. All signed paths except ORT's bundled one restrict access to specific apps.
- QAIRT 2.45's `qnn-onnx-converter` has a broken `ErfDummyLayoutInferer` that kills the decoder compile. Fix is to export the decoder with `nn.GELU(approximate='tanh')` (already applied in `VoiceAI_ASR_Community_v2.3.0.0/.../decoder_model_opt_fp16.py:159`).
- ORT 1.24+ plugin EP API is **different** from older providers list. Use `OrtEnv.RegisterExecutionProviderLibrary` + `GetEpDevices` + `SessionOptions.AppendExecutionProvider(env, epDevices, options)`. Don't pass `QNNExecutionProvider` as a string in the providers list — it silently falls back to CPU.
- Mel filterbank in `LogMel.cs` uses the HTK formula (`2595 * log10(1 + f/700)`) rather than openai-whisper's Slaney mel. Transcription is accurate in practice, but switching to Slaney is a future option.

## Possible follow-ups
- Precomputed QNN cache wrappers and payloads are included as build artifacts,
  avoiding the ~30 s JIT on this hardware/runtime combination.
- Streaming/VAD-driven capture instead of toggle (current app is push-to-talk-style).
- Longer-than-30-s dictation via audio chunking with prompt continuity.
- Swap greedy for beam-search decoder if accuracy needs tuning.
