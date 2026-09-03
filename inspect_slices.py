"""
Inspect Slice operations in the decoder ONNX model to understand
the garbage stride values QAIRT reports.
"""
import onnx
import numpy as np
from onnx import numpy_helper

model_path = "intermediates/decoder_model_merged.onnx"
print(f"Loading {model_path}...")
model = onnx.load(model_path)
graph = model.graph

# Build initializer lookup
inits = {}
for init in graph.initializer:
    inits[init.name] = numpy_helper.to_array(init)

# Find all Slice nodes
slice_nodes = [n for n in graph.node if n.op_type == "Slice"]
print(f"Found {len(slice_nodes)} Slice nodes\n")

# Check for problematic ones - those with missing or unusual steps
problems = []
for node in slice_nodes:
    name = node.name or node.output[0]
    inputs = list(node.input)

    # Slice inputs: data, starts, ends, [axes], [steps]
    starts_name = inputs[1] if len(inputs) > 1 else None
    ends_name = inputs[2] if len(inputs) > 2 else None
    axes_name = inputs[3] if len(inputs) > 3 else None
    steps_name = inputs[4] if len(inputs) > 4 else None

    starts = inits.get(starts_name) if starts_name else None
    ends = inits.get(ends_name) if ends_name else None
    axes = inits.get(axes_name) if axes_name else None
    steps = inits.get(steps_name) if steps_name else None

    # Check if steps input exists but is not in initializers (dynamic)
    steps_dynamic = steps_name and steps_name not in inits and steps_name != ""

    # Check for missing steps (defaults to 1)
    has_steps = len(inputs) > 4 and inputs[4] != ""

    info = {
        "name": name,
        "starts": starts,
        "ends": ends,
        "axes": axes,
        "steps": steps,
        "steps_name": steps_name,
        "steps_dynamic": steps_dynamic,
        "has_steps": has_steps,
        "num_inputs": len(inputs),
    }

    # Flag problems
    is_problem = False
    if steps_dynamic:
        is_problem = True
        info["issue"] = "steps is dynamic (not in initializers)"
    elif steps is not None and np.any(np.abs(steps) > 1000):
        is_problem = True
        info["issue"] = f"steps has unusual values: {steps}"
    elif not has_steps:
        # Missing steps - should default to 1, but maybe converter doesn't handle this
        info["issue"] = "no steps input (should default to 1)"

    if is_problem:
        problems.append(info)

# Print first 20 Slice nodes for context
print("=== First 20 Slice nodes ===")
for i, node in enumerate(slice_nodes[:20]):
    inputs = list(node.input)
    name = node.name or node.output[0]
    starts = inits.get(inputs[1]) if len(inputs) > 1 and inputs[1] in inits else "dynamic"
    ends = inits.get(inputs[2]) if len(inputs) > 2 and inputs[2] in inits else "dynamic"
    steps = inits.get(inputs[4]) if len(inputs) > 4 and inputs[4] in inits else ("dynamic" if len(inputs) > 4 and inputs[4] else "missing")
    print(f"  [{i}] {name}: starts={starts}, ends={ends}, steps={steps}, inputs={inputs}")

print(f"\n=== Problem Slice nodes: {len(problems)} ===")
for p in problems[:30]:
    print(f"  {p['name']}: {p['issue']}")
    if p['steps'] is not None:
        print(f"    steps={p['steps']}")
    if p['starts'] is not None:
        print(f"    starts={p['starts']}, ends={p['ends']}")

# Also check opset
print(f"\nModel opset: {[o.version for o in model.opset_import]}")
print(f"IR version: {model.ir_version}")
