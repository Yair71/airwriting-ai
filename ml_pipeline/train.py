"""Accuracy-first training: 1D-ResNet + BiGRU + attention.

AdamW + cosine annealing + label smoothing. Identity labels (no homoglyph collapse).
Saves `data/checkpoints/accurate_model.pth` and exports `accurate_model.onnx`.
Target: >98% top-1, <5 ms CPU latency. No 40 KB size cap.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ml_pipeline.dataset import REPO_ROOT, UnistrokeDataset
from ml_pipeline.evaluate import detailed_report, evaluate_loader, print_report
from ml_pipeline.export_onnx import export_fp32_onnx
from ml_pipeline.model import ModelConfig, UnistrokeNet, count_flops, count_parameters, fp32_bytes
from ml_pipeline.stroke_generator import DEFAULT_STROKE_NPZ, N_FEATURES

DEFAULT_NPZ = DEFAULT_STROKE_NPZ
DEFAULT_CKPT = REPO_ROOT / "data" / "checkpoints" / "accurate_model.pth"
DEFAULT_ONNX = REPO_ROOT / "data" / "checkpoints" / "accurate_model.onnx"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: UnistrokeNet,
    epoch: int,
    val_acc: float,
    metrics: dict,
    charset: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = np.arange(len(charset), dtype=np.int64)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "epoch": int(epoch),
        "val_accuracy": float(val_acc),
        "metrics": metrics,
        "charset": charset,
        "visual_charset": charset,
        "label_map": identity,
        "n_parameters": count_parameters(model),
        "flops": count_flops(model),
        "fp32_bytes": fp32_bytes(count_parameters(model)),
    }
    torch.save(payload, path)


def _make_loader(ds: UnistrokeDataset, batch_size: int, shuffle: bool, pin: bool) -> DataLoader:
    features = torch.from_numpy(np.ascontiguousarray(ds.features))
    labels = torch.from_numpy(np.ascontiguousarray(ds.labels.astype(np.int64)))
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin,
    )


def train(
    npz_path: Path = DEFAULT_NPZ,
    ckpt_path: Path = DEFAULT_CKPT,
    onnx_path: Path = DEFAULT_ONNX,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    weight_decay: float = 2e-2,
    label_smoothing: float = 0.1,
    hidden_dim: int = 96,
    gru_layers: int = 2,
    seed: int = 42,
    target_acc: float = 0.98,
    grad_clip: float = 1.0,
) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not Path(npz_path).is_file():
        raise FileNotFoundError(f"dataset not found: {npz_path}. Run: python -m ml_pipeline.stroke_generator")

    seed_everything(seed)
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, min(4, n_threads)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = torch.cuda.is_available()

    train_ds = UnistrokeDataset(npz_path=npz_path, split="train", seed=seed)
    val_ds = UnistrokeDataset(npz_path=npz_path, split="val", seed=seed)
    n_classes = train_ds.n_classes
    n_features = int(train_ds.n_features)
    if n_features != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {n_features}. Generate stroke_dataset.npz.")

    train_loader = _make_loader(train_ds, batch_size, shuffle=True, pin=pin)
    val_loader = _make_loader(val_ds, batch_size, shuffle=False, pin=pin)

    config = ModelConfig(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        gru_layers=gru_layers,
        gru_hidden=hidden_dim,
    )
    model = UnistrokeNet(config).to(device)
    n_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    use_amp = device.type == "cuda"
    autocast_dtype = torch.float16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device        : {device}")
    print(f"parameters    : {n_params}  ({fp32_bytes(n_params) / 1024:.1f} KB FP32)")
    print(f"flops / sample: {count_flops(model):,}")
    print(f"n_classes     : {n_classes} identity labels  (no homoglyph collapse)")
    print(f"n_features    : {n_features}")
    print(f"arch          : 1D-ResNet + BiGRU(h={hidden_dim}, layers={gru_layers}) + Attention")
    print(f"train / val   : {len(train_ds)} / {len(val_ds)}")
    print(f"batch / epochs: {batch_size} / {epochs}")
    print(f"opt           : AdamW lr={lr} wd={weight_decay}  cosine  label_smoothing={label_smoothing}")

    best_acc = -1.0
    history: list[dict] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_n = 0
        epoch_start = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:02d}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=autocast_dtype, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = int(yb.size(0))
            running_loss += float(loss.item()) * bs
            running_correct += int((logits.float().argmax(dim=-1) == yb).sum().item())
            running_n += bs
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        train_loss = running_loss / max(running_n, 1)
        train_acc = running_correct / max(running_n, 1)
        val_metrics = evaluate_loader(model, val_loader, device)
        lr_now = float(scheduler.get_last_lr()[0])
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_top3": val_metrics["top3"],
            "lr": lr_now,
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{epochs}  "
            f"train {train_loss:.4f}/{train_acc:.4f}  "
            f"val {val_metrics['loss']:.4f}/{val_metrics['acc']:.4f}  "
            f"top3 {val_metrics['top3']:.4f}  "
            f"lr {lr_now:.2e}  "
            f"{elapsed:.1f}s"
        )
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(
                ckpt_path,
                model,
                epoch,
                best_acc,
                {"history": history, "best_val_acc": best_acc},
                train_ds.charset,
            )
            print(f"  saved {ckpt_path}  val_acc={best_acc:.4f}")
        if best_acc >= target_acc and epoch >= 4:
            print(f"reached target top-1 {target_acc:.2%} at epoch {epoch}")
            break

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    val_metrics = evaluate_loader(model, val_loader, device)
    detail = detailed_report(model, val_loader, device, train_ds.charset)
    from ml_pipeline.evaluate import benchmark_latency

    latency_ms = benchmark_latency(model, device, seq_len=model.config.seq_len, n_features=model.config.n_features)
    report = {
        "checkpoint": str(ckpt_path),
        "split": "val",
        "device": str(device),
        "n_parameters": n_params,
        "fp32_kb": fp32_bytes(n_params) / 1024.0,
        "flops": count_flops(model),
        "val_accuracy_ckpt": best_acc,
        "epoch": int(blob.get("epoch", -1)),
        "loss": val_metrics["loss"],
        "top1": detail["acc"],
        "top3": val_metrics["top3"],
        "homoglyph_tolerant_top1": detail["homoglyph_tolerant_acc"],
        "latency_ms": latency_ms,
        "n": detail["n"],
        "confusion_pairs": detail["confusion_pairs"],
        "charset": train_ds.charset,
        "per_class_acc": detail["per_class_acc"],
    }
    print_report(report)

    export_fp32_onnx(model, Path(onnx_path))
    print(f"onnx           : {onnx_path}  ({Path(onnx_path).stat().st_size / 1024:.1f} KB)")

    total_s = time.perf_counter() - t0
    result = {
        "best_val_acc": best_acc,
        "val_top1": detail["acc"],
        "val_top3": val_metrics["top3"],
        "latency_ms": latency_ms,
        "epochs_run": history[-1]["epoch"] if history else 0,
        "seconds": total_s,
        "checkpoint": str(ckpt_path),
        "onnx": str(onnx_path),
        "n_parameters": n_params,
        "device": str(device),
        "confusion_pairs": [(src, dst, count) for _frac, src, dst, count in detail["confusion_pairs"]],
        "history": history,
    }
    log_path = ckpt_path.with_suffix(".log.json")
    serializable = {k: v for k, v in result.items() if k != "history"}
    serializable["history"] = history
    log_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"best val top-1 : {best_acc:.4f}")
    print(f"checkpoint     : {ckpt_path}")
    print(f"wall time      : {total_s:.1f}s")
    if best_acc < target_acc:
        print(f"WARNING: top-1 {best_acc:.4f} is below target {target_acc:.2%}")
    if latency_ms >= 5.0 and device.type == "cpu":
        print(f"WARNING: CPU latency {latency_ms:.2f} ms exceeds 5 ms budget")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train 1D-ResNet + BiGRU + attention to >98% top-1.")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-acc", type=float, default=0.98)
    args = parser.parse_args(argv)
    train(
        npz_path=args.npz,
        ckpt_path=args.ckpt,
        onnx_path=args.onnx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        seed=args.seed,
        target_acc=args.target_acc,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
