"""
Run qnn-onnx-converter using QAIRT 2.45 Python modules.
Must run with python310_x64/python.exe.
Usage:
  python310_x64/python.exe run_qnn_converter.py decoder
  python310_x64/python.exe run_qnn_converter.py encoder
"""
import sys
import os
import platform
import multiprocessing
multiprocessing.freeze_support()

# Monkeypatch for QAIRT
platform.processor = lambda: "AMD64 Family emulated"

QAIRT_ROOT = r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406"
QAIRT_PY = os.path.join(QAIRT_ROOT, "lib", "python")
QAIRT_X64_LIBS = os.path.join(QAIRT_ROOT, "lib", "x86_64-windows-msvc")
PYD_DIR = os.path.join(QAIRT_PY, "qti", "aisw", "converters", "common", "windows-x86_64")

sys.path.insert(0, QAIRT_PY)
os.add_dll_directory(QAIRT_X64_LIBS)
os.add_dll_directory(PYD_DIR)
os.environ["PATH"] = QAIRT_X64_LIBS + ";" + PYD_DIR + ";" + os.environ.get("PATH", "")

from qti.aisw.converters import onnx as onnx_frontend
from qti.aisw.converters.backend.ir_to_qnn import QnnConverterBackend
from qti.aisw.converters.backend.qnn_quantizer import QnnQuantizer
from qti.aisw.converters.common.converter_ir.op_graph_optimizations import IROptimizations
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper, CustomHelpFormatter
from qti.aisw.converters.common.arch_linter.arch_linter import ArchLinter
from qti.aisw.converters.common.graph_optimizer import GraphOptimizer

class ONNXtoQNNArgParser(ArgParserWrapper):
    def __init__(self):
        super().__init__(
            formatter_class=CustomHelpFormatter,
            conflict_handler='resolve',
            parents=[
                onnx_frontend.OnnxConverterFrontend.ArgParser(),
                IROptimizations.ArgParser(),
                QnnQuantizer.ArgParser(),
                QnnConverterBackend.ArgParser(),
                ArchLinter.ArgParser(),
                GraphOptimizer.ArgParser(),
            ]
        )
        self.parser.description = "ONNX to QNN converter"

model_type = sys.argv[1] if len(sys.argv) > 1 else "decoder"

if model_type == "decoder":
    cmd_args = [
        "-i", "intermediates/decoder_model_int32.onnx",
        "-o", "intermediates/decoder_model.cpp",
        "--float_bitwidth", "16",
        "--input_encoding", "x", "other",
        "--input_encoding", "offset", "other",
        "--input_encoding", "k_cache_cross", "other",
        "--input_encoding", "v_cache_cross", "other",
        "--input_encoding", "k_cache_self", "other",
        "--input_encoding", "v_cache_self", "other",
        "--preserve_io", "layout", "x",
        "--preserve_io", "layout", "offset",
        "--preserve_io", "layout", "k_cache_cross",
        "--preserve_io", "layout", "v_cache_cross",
        "--preserve_io", "layout", "k_cache_self",
        "--preserve_io", "layout", "v_cache_self",
    ]
elif model_type == "encoder":
    cmd_args = [
        "-i", "intermediates/encoder_model_int32.onnx",
        "-o", "intermediates/encoder_model.cpp",
        "--float_bitwidth", "16",
        "--input_encoding", "input_features", "other",
        "--preserve_io", "layout", "input_features",
    ]
else:
    print(f"Unknown model type: {model_type}")
    sys.exit(1)

print(f"Converting {model_type} ONNX -> QNN format...")
print(f"Args: {' '.join(cmd_args)}")

parser = ONNXtoQNNArgParser()
args = parser.parse_args(cmd_args)

converter = onnx_frontend.OnnxConverterFrontend(args)
ir_graph = converter.convert()
print("Frontend conversion done")

# Override optimizer flags for QNN backend (same as qnn-onnx-converter)
args.perform_axes_to_spatial_first_order = True
args.squash_box_decoder = True
args.match_caffe_ssd_to_tf = True
args.adjust_nms_features_dims = True
args.extract_color_transform = True
args.preprocess_roi_pool_inputs = True
args.unroll_lstm_time_steps = True
args.expand_gru_op_structure = True

backend = QnnConverterBackend(args)
backend.convert(ir_graph)
print("Backend conversion done")

# Check outputs
for ext in [".cpp", ".bin"]:
    p = f"intermediates/{model_type}_model{ext}"
    if os.path.exists(p):
        print(f"Generated: {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
    else:
        print(f"MISSING: {p}")
