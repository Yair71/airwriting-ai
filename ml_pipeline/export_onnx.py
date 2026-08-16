"""Export a pruned UnistrokeNet to dynamic INT8 ONNX.

Data contract: input float32 (1|B, 63, 4) -> logits (B, n_classes).
Target file size ≤ 40 KB after dynamic INT8 quantization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_pipeline.dataset import REPO_ROOT
from ml_pipeline.evaluate import load_checkpoint
from ml_pipeline.model import UnistrokeNet

DEFAULT_ONNX_INT8 = REPO_ROOT / "data" / "checkpoints" / "model_pruned_int8.onnx"
DEFAULT_ONNX_FP32 = REPO_ROOT / "data" / "checkpoints" / "model_pruned.onnx"


def export_fp32_onnx(model: UnistrokeNet, path: Path) -> Path:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, model.config.seq_len, model.config.n_features, device=next(model.parameters()).device)
    cpu = model.cpu()
    dummy = dummy.cpu()
    kwargs = dict(
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(cpu, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(cpu, dummy, str(path), **kwargs)
    return path


def quantize_dynamic_int8(fp32_path: Path, int8_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"EnableSubgraph": True},
    )
    return int8_path


@torch.inference_mode()
def onnx_accuracy(onnx_path: Path, loader: DataLoader, max_batches: int | None = None) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    correct = 0
    total = 0
    for i, (xb, yb) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = sess.run(None, {inp_name: xb.numpy()})[0]
        pred = np.argmax(logits, axis=-1)
        correct += int((pred == yb.numpy()).sum())
        total += int(yb.shape[0])
    return correct / max(total, 1)


def export_pruned_int8(
    ckpt_path: Path,
    int8_path: Path = DEFAULT_ONNX_INT8,
    loader: DataLoader | None = None,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cpu")
    model, blob = load_checkpoint(ckpt_path, torch.device("cpu"))
    fp32_path = int8_path.with_name("model_pruned.onnx")
    export_fp32_onnx(model, fp32_path)
    quantize_dynamic_int8(fp32_path, int8_path)
    acc = -1.0
    if loader is not None:
        acc = onnx_accuracy(int8_path, loader)
    size = int8_path.stat().st_size
    return {
        "path": str(int8_path),
        "fp32_path": str(fp32_path),
        "bytes": size,
        "acc": acc,
        "n_classes": int(blob["config"]["n_classes"]),
        "pytorch_val_acc": float(blob.get("val_accuracy", -1.0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export pruned checkpoint to dynamic INT8 ONNX.")
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "data" / "checkpoints" / "pruned_model.pth")
    parser.add_argument("--out", type=Path, default=DEFAULT_ONNX_INT8)
    args = parser.parse_args(argv)
    info = export_pruned_int8(args.ckpt, args.out)
    print(f"INT8 ONNX : {info['path']}")
    print(f"size      : {info['bytes']/1024:.2f} KB")
    print(f"limit 40KB: {'PASS' if info['bytes'] <= 40 * 1024 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
