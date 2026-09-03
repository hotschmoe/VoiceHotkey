"""
Build QNN context binaries from ONNX models using QAIRT 2.45.
Pipeline: ONNX -> qnn-onnx-converter -> object-generator -> zig c++ -> qnn-context-binary-generator

Must run with python310_x64/python.exe (x64 Python 3.10 under ARM64 emulation).
"""
import sys
import os
import shutil
import subprocess
import platform
import json

# Monkeypatch platform for QAIRT
platform.processor = lambda: "AMD64 Family emulated"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QAIRT_ROOT = r"C:\Qualcomm\AIStack\QAIRT\2.45.40.260406"
QAIRT_PY = os.path.join(QAIRT_ROOT, "lib", "python")
QAIRT_X64_LIBS = os.path.join(QAIRT_ROOT, "lib", "x86_64-windows-msvc")
QAIRT_ARM64_BIN = os.path.join(QAIRT_ROOT, "bin", "aarch64-windows-msvc")
QAIRT_ARM64_LIBS = os.path.join(QAIRT_ROOT, "lib", "aarch64-windows-msvc")
QAIRT_INCLUDE = os.path.join(QAIRT_ROOT, "include", "QNN")
QAIRT_SHARE = os.path.join(QAIRT_ROOT, "share", "QNN", "converter")
PYD_DIR = os.path.join(QAIRT_PY, "qti", "aisw", "converters", "common", "windows-x86_64")

# Setup paths
sys.path.insert(0, QAIRT_PY)
os.add_dll_directory(QAIRT_X64_LIBS)
os.add_dll_directory(PYD_DIR)
os.environ["PATH"] = QAIRT_X64_LIBS + ";" + PYD_DIR + ";" + os.environ.get("PATH", "")
os.environ["QNN_SDK_ROOT"] = QAIRT_ROOT

INTERMEDIATES = os.path.join(SCRIPT_DIR, "intermediates")
OUTPUT = os.path.join(SCRIPT_DIR, "qnn_output")
os.makedirs(INTERMEDIATES, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

SDK_NOTEBOOK = os.path.join(
    SCRIPT_DIR, "VoiceAI_ASR_Community_v2.3.0.0", "2.3.0.0", "notebook",
    "whisper", "npu", "whisper_small_encoder_quantized_decoder_fp16"
)


def step1_convert_onnx_to_qnn(model_name, onnx_path, extra_args=None):
    """Convert ONNX model to QNN format using qnn-onnx-converter."""
    print(f"\n{'='*60}")
    print(f"Step 1: Converting {model_name} ONNX -> QNN")
    print(f"{'='*60}")

    cpp_path = os.path.join(INTERMEDIATES, f"{model_name}.cpp")
    bin_path = os.path.join(INTERMEDIATES, f"{model_name}.bin")

    from qti.aisw.converters import onnx as onnx_frontend
    from qti.aisw.converters.backend.ir_to_qnn import QnnConverterBackend
    from qti.aisw.converters.common.converter_ir.op_graph_optimizations import IROptimizations

    # Frontend: ONNX -> IR
    print(f"  Loading ONNX: {onnx_path}")
    converter_args = ["--input_network", onnx_path]
    frontend = onnx_frontend.OnnxConverterFrontend(args=converter_args)
    ir_graph = frontend.convert()

    # Optimize
    optimizer = IROptimizations(args=[])
    optimizer.optimize(ir_graph)

    # Backend: IR -> QNN model .cpp/.bin
    backend_args = ["--output_path", cpp_path]
    if extra_args:
        backend_args.extend(extra_args)
    backend = QnnConverterBackend(args=backend_args)
    backend.convert(ir_graph)

    if os.path.exists(cpp_path):
        print(f"  Generated: {cpp_path}")
    if os.path.exists(bin_path):
        print(f"  Generated: {bin_path} ({os.path.getsize(bin_path) / 1e6:.1f} MB)")

    return cpp_path, bin_path


def step2_build_model_dll(model_name, cpp_path, bin_path):
    """Build model DLL using object-generator + zig c++."""
    print(f"\n{'='*60}")
    print(f"Step 2: Building {model_name} DLL with Zig")
    print(f"{'='*60}")

    build_dir = os.path.join(INTERMEDIATES, f"build_{model_name}")
    os.makedirs(build_dir, exist_ok=True)

    # Copy template sources
    jni_dir = os.path.join(QAIRT_SHARE, "jni")
    for f in ["QnnModel.cpp", "QnnModel.hpp", "QnnModelPal.hpp",
              "QnnTypeMacros.hpp", "QnnWrapperUtils.cpp", "QnnWrapperUtils.hpp"]:
        shutil.copy(os.path.join(jni_dir, f), build_dir)
    shutil.copy(os.path.join(jni_dir, "windows", "QnnModelPal.cpp"), build_dir)

    # Copy model files
    shutil.copy(cpp_path, os.path.join(build_dir, "QnnNetworkModel.cpp"))
    if os.path.exists(bin_path):
        shutil.copy(bin_path, build_dir)

    # Run object-generator to convert .bin to .o files
    obj_dir = os.path.join(build_dir, "obj", "aarch64-windows")
    os.makedirs(obj_dir, exist_ok=True)

    obj_gen = os.path.join(QAIRT_ARM64_BIN, "object-generator.exe")

    # Extract raw files from bin
    raw_dir = os.path.join(build_dir, "raw_files")
    os.makedirs(raw_dir, exist_ok=True)

    local_bin = os.path.join(build_dir, os.path.basename(bin_path))
    if os.path.exists(local_bin):
        import tarfile
        try:
            with tarfile.open(local_bin, 'r') as tar:
                tar.extractall(raw_dir)
            print(f"  Extracted raw files from {os.path.basename(bin_path)}")
        except tarfile.TarError:
            # bin might not be a tar, try as-is
            print(f"  {os.path.basename(bin_path)} is not a tar, using directly")
            shutil.copy(local_bin, os.path.join(raw_dir, os.path.basename(bin_path)))

        # Run object-generator
        cmd = f'"{obj_gen}" "{raw_dir}" aarch64-windows "{obj_dir}"'
        print(f"  Running object-generator: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  object-generator stderr: {result.stderr}")
            print(f"  object-generator stdout: {result.stdout}")
            # Continue anyway - we might be able to build without it
        else:
            print(f"  Object files generated")

    # Compile with zig c++
    dll_path = os.path.join(OUTPUT, f"{model_name}.dll")
    sources = [
        os.path.join(build_dir, "QnnModel.cpp"),
        os.path.join(build_dir, "QnnWrapperUtils.cpp"),
        os.path.join(build_dir, "QnnModelPal.cpp"),
        os.path.join(build_dir, "QnnNetworkModel.cpp"),
    ]

    # Collect any .o files
    obj_files = []
    if os.path.isdir(obj_dir):
        for f in os.listdir(obj_dir):
            if f.endswith(".o"):
                obj_files.append(os.path.join(obj_dir, f))

    cmd = [
        "zig", "c++",
        "-target", "aarch64-windows-gnu",
        "-shared",
        "-DQNN_API=__declspec(dllexport)",
        f"-I{QAIRT_INCLUDE}",
        f"-I{build_dir}",
        "-o", dll_path,
    ] + sources + obj_files + [
        "-lpsapi",  # for EnumProcessModules
    ]

    print(f"  Compiling with zig c++...")
    print(f"  Sources: {len(sources)}, Objects: {len(obj_files)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COMPILATION FAILED:")
        print(f"  stderr: {result.stderr}")
        print(f"  stdout: {result.stdout}")
        return None
    else:
        print(f"  Built: {dll_path} ({os.path.getsize(dll_path) / 1e6:.1f} MB)")
        return dll_path


def step3_generate_context_binary(model_name, dll_path):
    """Generate QNN context binary using qnn-context-binary-generator."""
    print(f"\n{'='*60}")
    print(f"Step 3: Generating {model_name} context binary")
    print(f"{'='*60}")

    ctx_gen = os.path.join(QAIRT_ARM64_BIN, "qnn-context-binary-generator.exe")
    backend = os.path.join(QAIRT_ARM64_LIBS, "QnnHtp.dll")

    # Create HTP settings for SC8480XP (X2 Elite Extreme)
    htp_settings = {
        "graphs": [{
            "graph_names": [model_name],
            "vtcm_mb": 8,
            "O": 3
        }],
        "devices": [{
            "device_id": 0,
            "dsp_arch": "v73",
            "soc_model": 60,
            "pd_session": "unsigned",
            "cores": [{"perf_profile": "burst"}]
        }]
    }

    htp_context = {
        "backend_extensions": {
            "shared_library_path": os.path.join(QAIRT_ARM64_LIBS, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": os.path.join(OUTPUT, f"htp_settings_{model_name}.json")
        },
        "context_configs": {
            "context_priority": "high",
            "cache_compatibility_mode": "permissive"
        },
        "graph_configs": [{
            "graph_name": model_name,
            "graph_priority": "high"
        }]
    }

    # Write config files
    settings_path = os.path.join(OUTPUT, f"htp_settings_{model_name}.json")
    context_path = os.path.join(OUTPUT, f"htp_context_{model_name}.json")
    with open(settings_path, "w") as f:
        json.dump(htp_settings, f, indent=2)
    with open(context_path, "w") as f:
        json.dump(htp_context, f, indent=2)

    cmd = [
        ctx_gen,
        "--model", dll_path,
        "--backend", backend,
        "--binary_file", f"{model_name}_htp",
        "--output_dir", OUTPUT,
        "--config_file", context_path,
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}):")
        print(f"  stderr: {result.stderr}")
        print(f"  stdout: {result.stdout}")
        return None

    bin_path = os.path.join(OUTPUT, f"{model_name}_htp.bin")
    if os.path.exists(bin_path):
        print(f"  Generated: {bin_path} ({os.path.getsize(bin_path) / 1e6:.1f} MB)")
        return bin_path
    else:
        print(f"  Expected output not found: {bin_path}")
        print(f"  stdout: {result.stdout}")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, help="Run only a specific step (1, 2, or 3)")
    parser.add_argument("--model", choices=["encoder", "decoder", "both"], default="both")
    args = parser.parse_args()

    models = []
    if args.model in ("decoder", "both"):
        models.append(("decoder_model", {
            "extra_args": ["--float_bitwidth", "16"],
        }))
    if args.model in ("encoder", "both"):
        models.append(("encoder_model", {
            "extra_args": ["--float_bitwidth", "16"],  # FP16 fallback (no calibration data)
        }))

    for model_name, opts in models:
        onnx_path = os.path.join(INTERMEDIATES, f"{model_name}.onnx")
        cpp_path = os.path.join(INTERMEDIATES, f"{model_name}.cpp")
        bin_path = os.path.join(INTERMEDIATES, f"{model_name}.bin")

        if not args.step or args.step == 1:
            if not os.path.exists(onnx_path):
                print(f"ERROR: {onnx_path} not found. Run export_onnx.py first.")
                sys.exit(1)
            cpp_path, bin_path = step1_convert_onnx_to_qnn(
                model_name, onnx_path, opts.get("extra_args")
            )

        if not args.step or args.step == 2:
            dll_path = step2_build_model_dll(model_name, cpp_path, bin_path)
            if not dll_path:
                print(f"FAILED to build {model_name} DLL")
                sys.exit(1)

        if not args.step or args.step == 3:
            dll_path = os.path.join(OUTPUT, f"{model_name}.dll")
            if not os.path.exists(dll_path):
                print(f"ERROR: {dll_path} not found. Run step 2 first.")
                sys.exit(1)
            ctx_path = step3_generate_context_binary(model_name, dll_path)
            if ctx_path:
                print(f"\nSUCCESS: {model_name} context binary ready at {ctx_path}")
            else:
                print(f"\nFAILED: {model_name} context binary generation")
                sys.exit(1)

    print(f"\n{'='*60}")
    print("All done! Copy *_htp.bin files to VoiceHotkey/models/")
    print(f"{'='*60}")
