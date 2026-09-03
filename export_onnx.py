"""
Export Whisper Small ONNX models using SDK's redefined modules.
Must run with python310_x64/python.exe (x64 Python 3.10).
"""
import sys
import os

# Add SDK's notebook dir to path for redefined_modules
SDK_NOTEBOOK = os.path.join(
    os.path.dirname(__file__),
    "VoiceAI_ASR_Community_v2.3.0.0", "2.3.0.0", "notebook",
    "whisper", "npu", "whisper_small_encoder_quantized_decoder_fp16"
)
sys.path.insert(0, SDK_NOTEBOOK)

import torch
import whisper

# Force legacy (TorchScript) ONNX export and upgrade opset_version from 12 to 17
_orig_export = torch.onnx.export
def patched_export(*args, **kwargs):
    kwargs['dynamo'] = False
    # Upgrade opset_version if it's too low (12 causes Slice corruption in new PyTorch)
    if kwargs.get('opset_version', 99) < 17:
        kwargs['opset_version'] = 17
    return _orig_export(*args, **kwargs)
torch.onnx.export = patched_export

# Create output directory
intermediates = os.path.join(os.path.dirname(__file__), "intermediates")
os.makedirs(intermediates, exist_ok=True)

# Load whisper small model
print("Loading Whisper 'small' model...")
model = whisper.load_model("small")
print(f"  Model loaded. dims: n_mels={model.dims.n_mels}, n_audio_ctx={model.dims.n_audio_ctx}")

# Export encoder (skip if already exists)
encoder_path = os.path.join(intermediates, "encoder_model.onnx")
if os.path.exists(encoder_path):
    print(f"\n--- Encoder ONNX already exists, skipping ---")
else:
    print("\n--- Exporting Encoder ---")
    from redefined_modules.encoder_model_opt_q8 import export_onnx as export_encoder
    export_encoder(model, encoder_path)
print(f"  Encoder: {encoder_path} ({os.path.getsize(encoder_path) / 1e6:.1f} MB)")

# Export decoder
print("\n--- Exporting Decoder ---")
from redefined_modules.decoder_model_opt_fp16 import export_onnx as export_decoder
decoder_path = os.path.join(intermediates, "decoder_model.onnx")
export_decoder(model, decoder_path)
print(f"  Decoder saved: {decoder_path} ({os.path.getsize(decoder_path) / 1e6:.1f} MB)")

print("\nDone! ONNX models exported to:", intermediates)
