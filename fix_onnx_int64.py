"""
Fix ONNX models for QAIRT 2.45 x64 emulation on ARM64.

QAIRT's C++ ops misread int64 tensor values as int32 under emulation,
causing garbage values. This script converts all int64 tensors/attributes
to int32 so the converter reads correct values.

Usage:
  python310_x64/python.exe fix_onnx_int64.py
"""
import onnx
from onnx import numpy_helper, TensorProto
import numpy as np
import os

def fix_int64_model(input_path, output_path):
    print(f"Loading {input_path}...")
    model = onnx.load(input_path)
    graph = model.graph
    fixes = 0

    # Fix initializers: convert int64 tensors to int32
    for init in graph.initializer:
        if init.data_type == TensorProto.INT64:
            arr = numpy_helper.to_array(init)
            # Check if values fit in int32
            if np.all(arr >= np.iinfo(np.int32).min) and np.all(arr <= np.iinfo(np.int32).max):
                arr32 = arr.astype(np.int32)
                new_tensor = numpy_helper.from_array(arr32, name=init.name)
                init.CopyFrom(new_tensor)
                fixes += 1
            else:
                # Clamp MAX_INT64 (used as "end" in Slice) to MAX_INT32
                arr_clamped = np.clip(arr, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
                arr32 = arr_clamped.astype(np.int32)
                new_tensor = numpy_helper.from_array(arr32, name=init.name)
                init.CopyFrom(new_tensor)
                fixes += 1
                print(f"  Clamped large int64 values in {init.name}: {arr.flatten()[:5]}...")

    # Fix node attributes: convert int64 attributes to use int32-safe values
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                if attr.i > np.iinfo(np.int32).max or attr.i < np.iinfo(np.int32).min:
                    old = attr.i
                    attr.i = int(np.clip(attr.i, np.iinfo(np.int32).min, np.iinfo(np.int32).max))
                    fixes += 1
                    print(f"  Clamped attr {attr.name} in {node.name}: {old} -> {attr.i}")
            elif attr.type == onnx.AttributeProto.INTS:
                new_ints = []
                changed = False
                for v in attr.ints:
                    if v > np.iinfo(np.int32).max or v < np.iinfo(np.int32).min:
                        new_ints.append(int(np.clip(v, np.iinfo(np.int32).min, np.iinfo(np.int32).max)))
                        changed = True
                    else:
                        new_ints.append(v)
                if changed:
                    del attr.ints[:]
                    attr.ints.extend(new_ints)
                    fixes += 1
                    print(f"  Clamped ints attr {attr.name} in {node.name}")

    # Fix graph input/output type info: int64 -> int32
    for vi in list(graph.input) + list(graph.output):
        t = vi.type.tensor_type
        if t.elem_type == TensorProto.INT64:
            t.elem_type = TensorProto.INT32
            fixes += 1
            print(f"  Changed I/O type for {vi.name}: int64 -> int32")

    # Fix Constant nodes that produce int64 tensors
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.INT64:
                    arr = numpy_helper.to_array(attr.t)
                    arr_clamped = np.clip(arr, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
                    arr32 = arr_clamped.astype(np.int32)
                    new_tensor = numpy_helper.from_array(arr32)
                    attr.t.CopyFrom(new_tensor)
                    fixes += 1

    print(f"\nApplied {fixes} int64->int32 fixes")
    print(f"Saving to {output_path}...")
    onnx.save(model, output_path)
    print(f"Saved: {os.path.getsize(output_path) / 1e6:.1f} MB")

if __name__ == "__main__":
    os.makedirs("intermediates", exist_ok=True)

    fix_int64_model(
        "intermediates/encoder_model_merged.onnx",
        "intermediates/encoder_model_int32.onnx"
    )

    print("\n" + "="*60 + "\n")

    fix_int64_model(
        "intermediates/decoder_model_merged.onnx",
        "intermediates/decoder_model_int32.onnx"
    )
