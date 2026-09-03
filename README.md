# VoiceHotkey

Local, system-wide voice dictation for Windows on ARM64, built for the Qualcomm
Hexagon NPU in the Snapdragon X2 Elite Extreme.

Tap `Ctrl+Space`, speak, and either pause for 1.2 seconds or tap `Ctrl+Space`
again. Whisper transcribes locally and the text is inserted into the currently
focused field. The microphone is inactive while the app is idle.

## What works

- Native ARM64 C# application
- Whisper-small encoder and decoder on the Hexagon NPU
- ONNX Runtime QNN execution provider
- Global toggle hotkey
- Voice-activity auto-stop and a 30-second safety limit
- Unicode text insertion with a clipboard copy retained as fallback
- Precompiled QNN context loading after the first launch

The tested machine is an ASUS Zenbook A16 UX3607OA with a Snapdragon X2 Elite
Extreme (SC8480XP / Hexagon V81), 48 GB RAM, and Windows 11 ARM64 26H1.
Other Qualcomm hardware or QNN runtime versions may need to compile their own
context cache from the original ONNX models.

## Privacy

VoiceHotkey has no telemetry, account, cloud transcription, network client, or
transcription history. Recorded PCM stays in memory for the current utterance
and is discarded after transcription. Model inference is local.

NuGet restore and obtaining the model files naturally require network access,
but normal dictation does not.

## Build

Requirements:

- Windows 11 ARM64 on a supported Snapdragon NPU
- .NET 9 SDK
- The two prepared Whisper ONNX models listed below

Model files are intentionally not stored in Git because they exceed GitHub's
100 MB per-file limit. Place them here:

```text
intermediates/encoder_model_merged.onnx
intermediates/decoder_model_merged.onnx
```

Then build:

```powershell
cd VoiceHotkey
dotnet build VoiceHotkey.csproj -c Release -r win-arm64
```

The executable is written below
`VoiceHotkey/bin/Release/net9.0-windows10.0.22621.0/win-arm64/`.

For the tested ASUS/Snapdragon X2EE machine, the private
[v0.1.0 release](https://github.com/hotschmoe/VoiceHotkey/releases/tag/v0.1.0)
contains the exact working ARM64 bundle, models, dependencies, and SC8480XP
NPU cache. Its SHA-256 is
`D37F48CC11F0ABB35559A61BB7D0F41E59826F5A9DD0E610C869394A16042B09`.

On first launch, ONNX Runtime compiles the models for the NPU and writes a
small `*_ctx.onnx` wrapper plus an external `*_ctx_qnn.bin` payload under
`qnn_cache/`. Later launches load the cache in about a second. External cache
mode is required because ONNX Runtime QNN 1.24.4 produces invalid embedded
caches for multi-partition graphs.

## Usage

1. Start `VoiceHotkey.exe` and leave it running.
2. Focus a normal editable field.
3. Tap `Ctrl+Space` and speak.
4. Pause for auto-stop, or tap `Ctrl+Space` again.

Windows prevents a normal process from injecting input into an elevated
Administrator process. If an elevated target must receive text, VoiceHotkey
must run at the same integrity level.

## Repository contents

The C# application and tokenizer data are tracked. Small experimental Python
tools used during model export and QNN bring-up are retained for provenance.
Virtual environments, Qualcomm SDK/sample material, generated models, QNN
caches, native runtime binaries, logs, and build output are excluded.

See `current_status.md` for the bring-up history and implementation notes.
