"""Remove exporter/debug metadata (including local source paths) from ONNX.

Tensor values, graph structure, operator attributes, names, and shapes are left
unchanged. This is intended for preparing redistributable release artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message


def strip_debug_metadata(message: Message) -> None:
    """Recursively clear protobuf documentation and metadata-property fields."""
    for field in message.DESCRIPTOR.fields:
        value = getattr(message, field.name)
        if field.name == "doc_string":
            setattr(message, field.name, "")
        elif field.name == "metadata_props":
            del value[:]
        elif field.type == FieldDescriptor.TYPE_MESSAGE:
            if field.is_repeated:
                for child in value:
                    strip_debug_metadata(child)
            elif message.HasField(field.name):
                strip_debug_metadata(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=True)
    strip_debug_metadata(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
