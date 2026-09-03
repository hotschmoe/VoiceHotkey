"""
Export Whisper Small ONNX models using dynamo (default) export.
Produces opset 18-20 files with external data.
Then merges external data into single ONNX file for the converter.
"""
import sys
import os

SDK_NOTEBOOK = os.path.join(
    os.path.dirname(__file__),
    "VoiceAI_ASR_Community_v2.3.0.0", "2.3.0.0", "notebook",
    "whisper", "npu", "whisper_small_encoder_quantized_decoder_fp16"
)
sys.path.insert(0, SDK_NOTEBOOK)

import torch
import whisper
import onnx
from onnx.external_data_helper import convert_model_to_external_data

intermediates = os.path.join(os.path.dirname(__file__), "intermediates")
os.makedirs(intermediates, exist_ok=True)

# Load whisper small model
print("Loading Whisper 'small' model...")
model = whisper.load_model("small")
print(f"  Model loaded. dims: n_mels={model.dims.n_mels}, n_audio_ctx={model.dims.n_audio_ctx}")

# Export encoder with dynamo (default)
encoder_path = os.path.join(intermediates, "encoder_model.onnx")
print("\n--- Exporting Encoder (dynamo) ---")
from redefined_modules.encoder_model_opt_q8 import export_onnx as export_encoder
export_encoder(model, encoder_path)
enc_size = os.path.getsize(encoder_path)
enc_data = encoder_path + ".data"
if os.path.exists(enc_data):
    enc_size += os.path.getsize(enc_data)
print(f"  Encoder total size: {enc_size / 1e6:.1f} MB")

# Export decoder with dynamo (default)
decoder_path = os.path.join(intermediates, "decoder_model.onnx")
print("\n--- Exporting Decoder (dynamo) ---")
from redefined_modules.decoder_model_opt_fp16 import export_onnx as export_decoder
export_decoder(model, decoder_path)
dec_size = os.path.getsize(decoder_path)
dec_data = decoder_path + ".data"
if os.path.exists(dec_data):
    dec_size += os.path.getsize(dec_data)
print(f"  Decoder total size: {dec_size / 1e6:.1f} MB")

# Merge external data into single files for the QNN converter
print("\n--- Merging external data into single ONNX files ---")
for name, path in [("encoder", encoder_path), ("decoder", decoder_path)]:
    data_file = path + ".data"
    if os.path.exists(data_file):
        print(f"  Merging {name}...")
        m = onnx.load(path, load_external_data=True)
        merged_path = os.path.join(intermediates, f"{name}_model_merged.onnx")
        onnx.save(m, merged_path)
        print(f"  Saved: {merged_path} ({os.path.getsize(merged_path) / 1e6:.1f} MB)")
    else:
        print(f"  {name}: no external data, already self-contained")

print("\nDone!")
