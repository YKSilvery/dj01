#!/usr/bin/env python3
"""Utility to export CenterNet checkpoints to ONNX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Ensure we can import the CenterNet src package when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lib.models.model import create_model, load_model  # type: ignore[import]


class CenterNetWrapper(torch.nn.Module):
    """Wraps the raw model to provide deterministic tuple outputs for ONNX."""

    def __init__(self, model: torch.nn.Module, output_order: List[str]):
        super().__init__()
        self.model = model
        self.output_order = output_order

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:  # type: ignore[override]
        outputs = self.model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[-1]
        return tuple(outputs[name] for name in self.output_order)


def build_heads(num_classes: int, include_reg: bool) -> Dict[str, int]:
    heads: Dict[str, int] = {"hm": num_classes, "wh": 2}
    if include_reg:
        heads["reg"] = 2
    return heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a CenterNet checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained *.pth checkpoint")
    parser.add_argument("--onnx", required=True, help="Destination ONNX file path")
    parser.add_argument("--arch", default="res_18", help="Model architecture, e.g. res_18, res_101, dla_34")
    parser.add_argument("--num-classes", type=int, default=14, help="Number of detection classes")
    parser.add_argument("--head-conv", type=int, default=64, help="Head conv width used during training")
    parser.add_argument("--input-h", type=int, default=512, help="Model input height")
    parser.add_argument("--input-w", type=int, default=512, help="Model input width")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for export")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic axis for batch dimension")
    parser.add_argument("--no-reg", action="store_true", help="Exclude regression head from outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    onnx_path = Path(args.onnx).expanduser().resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    heads = build_heads(args.num_classes, include_reg=not args.no_reg)
    output_order = sorted(heads.keys())

    model = create_model(args.arch, heads, args.head_conv)
    model = load_model(model, str(checkpoint_path))
    model.eval()
    model.to(device)

    wrapper = CenterNetWrapper(model, output_order).to(device)

    dummy_input = torch.randn(args.batch_size, 3, args.input_h, args.input_w, device=device)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}}
        for name in output_order:
            dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_order,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
