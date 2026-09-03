"""
Convert Whisper ONNX models to QNN context binaries using QAIRT 2.45.
Runs under x64 Python emulation on ARM64 Windows.
"""
import sys
import os
import platform
import subprocess

# Force x86_64 platform detection for QAIRT's init.py
_orig_processor = platform.processor
platform.processor = lambda: "AMD64 Family emulated"

QAIRT_ROOT = r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406"
QAIRT_PY = os.path.join(QAIRT_ROOT, "lib", "python")
QAIRT_X64_LIBS = os.path.join(QAIRT_ROOT, "lib", "x86_64-windows-msvc")
QAIRT_ARM64_BIN = os.path.join(QAIRT_ROOT, "bin", "aarch64-windows-msvc")

# Add QAIRT Python to path
sys.path.insert(0, QAIRT_PY)
# Add x64 DLLs to PATH so they can be found
os.environ["PATH"] = QAIRT_X64_LIBS + ";" + os.environ.get("PATH", "")

# Now test imports
print("Testing QAIRT imports...")
try:
    from qti.aisw.converters import onnx as onnx_frontend
    from qti.aisw.converters.backend.ir_to_qnn import QnnConverterBackend
    from qti.aisw.converters.common.utils.converter_utils import log_error, log_info
    print("  QAIRT converter modules loaded OK")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

def convert_onnx_to_qnn(onnx_path, output_dir, model_name):
    """Convert ONNX model to QNN model .cpp/.bin"""
    print(f"\nConverting {onnx_path} -> QNN format...")

    args = [
        "--input_network", onnx_path,
        "--output_path", os.path.join(output_dir, model_name + ".cpp"),
        "--input_list", "",  # empty for no calibration
    ]

    # Use the converter frontend
    frontend = onnx_frontend.OnnxConverterFrontend(args=[
        "--input_network", onnx_path,
    ])
    ir_graph = frontend.convert()

    # Generate QNN model
    backend = QnnConverterBackend(args=[
        "--output_path", os.path.join(output_dir, model_name + ".cpp"),
    ])
    backend.convert(ir_graph)
    print(f"  Generated {model_name}.cpp")
    return os.path.join(output_dir, model_name + ".cpp")

if __name__ == "__main__":
    print("QAIRT ONNX -> QNN converter ready")
    print(f"QAIRT version: 2.45.40")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.machine()} (processor override: {platform.processor()})")
