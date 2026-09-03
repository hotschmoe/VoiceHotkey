"""
Smoke test: load encoder ONNX via ORT with QNN EP on Snapdragon X2 Elite.

ORT 1.24+ plugin EP API:
  1. register_execution_provider_library(name, library_path)
  2. Find the OrtEpDevice matching the NPU backend
  3. session_options.add_provider_for_devices([dev], options)
"""
import os, time
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn

ROOT = os.path.dirname(os.path.abspath(__file__))
ENCODER = os.path.join(ROOT, "intermediates", "encoder_model_merged.onnx")

ort.register_execution_provider_library(
    onnxruntime_qnn.get_ep_name(),
    onnxruntime_qnn.get_library_path(),
)
print("ORT:", ort.__version__)

# Inspect all EP devices — pick the HTP (NPU) one
print("\nEP devices:")
qnn_devs = []
for d in ort.get_ep_devices():
    dev = d.device
    print(f"  ep_name={d.ep_name!r} ep_vendor={d.ep_vendor!r} device.type={getattr(dev,'type',None)!r} device.vendor={getattr(dev,'vendor',None)!r}")
    print(f"    ep_metadata={d.ep_metadata}")
    if d.ep_name == "QNNExecutionProvider":
        qnn_devs.append(d)

# Pick the NPU device. If multiple, prefer NPU device type.
npu = next((d for d in qnn_devs if getattr(d.device, 'type', None) and 'NPU' in str(d.device.type).upper()), None)
if npu is None:
    npu = qnn_devs[0] if qnn_devs else None
print(f"\nSelected: ep_name={npu.ep_name} ep_metadata={npu.ep_metadata}")

so = ort.SessionOptions()
so.log_severity_level = 2
# backend_path is how we tell QNN EP which QnnXxx.dll to load; use the bundled HTP one
ep_options = {
    "backend_path": onnxruntime_qnn.get_qnn_htp_path(),
    "htp_performance_mode": "burst",
}
so.add_provider_for_devices([npu], ep_options)

print("\nCreating session...")
sess = ort.InferenceSession(ENCODER, sess_options=so)
print("Providers in session:", sess.get_providers())

print("\nRunning forward passes...")
mel = np.random.randn(1, 80, 3000).astype(np.float32)
t0 = time.perf_counter(); sess.run(None, {"input_features": mel}); t1 = time.perf_counter()
print(f"  cold: {(t1-t0)*1000:.1f} ms")
for i in range(3):
    t0 = time.perf_counter(); sess.run(None, {"input_features": mel}); t1 = time.perf_counter()
    print(f"  warm {i}: {(t1-t0)*1000:.1f} ms")

# NPU should give <150 ms typical; CPU likely ~500-700 ms
print("\nDONE.")
