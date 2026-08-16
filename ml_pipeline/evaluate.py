"""Accuracy, confusion, FLOPs, and latency for the unistroke classifier."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_pipeline.dataset import REPO_ROOT, UnistrokeDataset
from ml_pipeline.model import ModelConfig, UnistrokeNet, count_flops, count_parameters, fp32_bytes
from ml_pipeline.stroke_generator import DEFAULT_STROKE_NPZ

DEFAULT_NPZ = DEFAULT_STROKE_NPZ
DEFAULT_CKPT = REPO_ROOT / "data" / "checkpoints" / "accurate_model.pth"

# Identical outlines in Arial/Times (same cmap glyph). 0/O, 1/I, Y/У stay separate.
VISUAL_TWINS: tuple[frozenset[str], ...] = (
    frozenset({"A", "А"}),
    frozenset({"B", "В"}),
    frozenset({"C", "С"}),
    frozenset({"E", "Е", "Ё"}),
    frozenset({"H", "Н"}),
    frozenset({"K", "К"}),
    frozenset({"M", "М"}),
    frozenset({"O", "О"}),
    frozenset({"P", "Р"}),
    frozenset({"T", "Т"}),
    frozenset({"X", "Х"}),
    frozenset({"И", "Й"}),
)
HOMOGLYPH_GROUPS = VISUAL_TWINS


def visual_class_map(charset: list[str]) -> tuple[np.ndarray, list[str]]:
    """Collapse Latin/Cyrillic twins. Canonical name is the first codepoint in charset order."""
    parent = {ch: ch for ch in charset}
    charset_set = set(charset)
    for group in VISUAL_TWINS:
        members = [ch for ch in charset if ch in group]
        if len(members) < 2:
            continue
        canon = members[0]
        for ch in members[1:]:
            parent[ch] = canon

    def root(ch: str) -> str:
        while parent[ch] != ch:
            ch = parent[ch]
        return ch

    visual_charset: list[str] = []
    index: dict[str, int] = {}
    old_to_new = np.empty(len(charset), dtype=np.int64)
    for i, ch in enumerate(charset):
        r = root(ch)
        if r not in index:
            index[r] = len(visual_charset)
            visual_charset.append(r)
        old_to_new[i] = index[r]
    if len(charset_set) != len(charset):
        raise ValueError("charset contains duplicates")
    return old_to_new, visual_charset


@torch.inference_mode()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    top3 = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss_sum += float(criterion(logits, yb).item())
        pred = logits.argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        top3 += int((logits.topk(min(3, logits.size(-1)), dim=-1).indices == yb.unsqueeze(-1)).any(dim=-1).sum().item())
        total += int(yb.numel())
    n = max(total, 1)
    return {"loss": loss_sum / n, "acc": correct / n, "top3": top3 / n, "n": float(total)}


def _homoglyph_id_sets(charset: list[str]) -> list[set[int]]:
    index = {ch: i for i, ch in enumerate(charset)}
    groups: list[set[int]] = []
    for group in HOMOGLYPH_GROUPS:
        ids = {index[ch] for ch in group if ch in index}
        if len(ids) >= 2:
            groups.append(ids)
    return groups


@torch.inference_mode()
def detailed_report(
    model: UnistrokeNet,
    loader: DataLoader,
    device: torch.device,
    charset: list[str],
) -> dict:
    model.eval()
    n_classes = len(charset)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb).argmax(dim=-1)
        for t, p in zip(yb.tolist(), pred.tolist()):
            confusion[int(t), int(p)] += 1

    support = confusion.sum(axis=1)
    correct = np.diag(confusion)
    per_class = np.divide(correct, np.maximum(support, 1), dtype=np.float64)
    total = int(confusion.sum())
    top1 = float(correct.sum() / max(total, 1))

    groups = _homoglyph_id_sets(charset)
    member_of: dict[int, set[int]] = {}
    for g in groups:
        for i in g:
            member_of[i] = g
    tolerant = 0
    for t in range(n_classes):
        allowed = member_of.get(t, {t})
        tolerant += int(confusion[t, list(allowed)].sum())
    tolerant_acc = float(tolerant / max(total, 1))

    off = confusion.copy()
    np.fill_diagonal(off, 0)
    pairs: list[tuple[float, str, str, int]] = []
    for t, p in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:12], off.shape)):
        count = int(off[t, p])
        if count == 0:
            break
        pairs.append((count / max(total, 1), charset[t], charset[p], count))

    return {
        "n": total,
        "acc": top1,
        "homoglyph_tolerant_acc": tolerant_acc,
        "per_class_acc": per_class,
        "confusion_pairs": pairs,
        "support": support,
    }


@torch.inference_mode()
def benchmark_latency(model: nn.Module, device: torch.device, seq_len: int = 63, n_features: int = 6, warmup: int = 30, runs: int = 200) -> float:
    model.eval()
    x = torch.zeros(1, seq_len, n_features, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def load_checkpoint(path: Path, device: torch.device) -> tuple[UnistrokeNet, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}. Run: python -m ml_pipeline.train")
    blob = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(blob["config"])
    model = UnistrokeNet(config)
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    return model, blob


def evaluate(
    ckpt_path: Path = DEFAULT_CKPT,
    npz_path: Path = DEFAULT_NPZ,
    split: str = "val",
    batch_size: int = 512,
    seed: int = 42,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, blob = load_checkpoint(ckpt_path, device)
    ds = UnistrokeDataset(npz_path=npz_path, split=split, seed=seed)
    features = torch.from_numpy(np.ascontiguousarray(ds.features))
    raw_labels = np.ascontiguousarray(ds.labels.astype(np.int64))
    if "label_map" in blob:
        label_map = np.asarray(blob["label_map"], dtype=np.int64)
        labels = torch.from_numpy(label_map[raw_labels])
        charset = [str(x) for x in blob.get("visual_charset", ds.charset)]
    else:
        labels = torch.from_numpy(raw_labels)
        charset = ds.charset
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False, num_workers=0)
    basic = evaluate_loader(model, loader, device)
    detail = detailed_report(model, loader, device, charset)
    latency_ms = benchmark_latency(model, device, seq_len=model.config.seq_len, n_features=model.config.n_features)
    n_params = count_parameters(model)
    report = {
        "checkpoint": str(ckpt_path),
        "split": split,
        "device": str(device),
        "n_parameters": n_params,
        "fp32_kb": fp32_bytes(n_params) / 1024.0,
        "flops": count_flops(model),
        "val_accuracy_ckpt": float(blob.get("val_accuracy", -1.0)),
        "epoch": int(blob.get("epoch", -1)),
        "loss": basic["loss"],
        "top1": detail["acc"],
        "top3": basic["top3"],
        "homoglyph_tolerant_top1": detail["homoglyph_tolerant_acc"],
        "latency_ms": latency_ms,
        "n": detail["n"],
        "confusion_pairs": detail["confusion_pairs"],
        "charset": charset,
        "per_class_acc": detail["per_class_acc"],
    }
    return report


def print_report(report: dict) -> None:
    print(f"checkpoint     : {report['checkpoint']}")
    print(f"split          : {report['split']}  n={report['n']}")
    print(f"device         : {report['device']}")
    print(f"parameters     : {report['n_parameters']}  ({report['fp32_kb']:.1f} KB FP32)")
    print(f"flops / sample : {report['flops']:,}")
    print(f"ckpt epoch/acc : {report['epoch']} / {report['val_accuracy_ckpt']:.4f}")
    print(f"top-1          : {report['top1']:.4f}")
    print(f"top-3          : {report['top3']:.4f}")
    print(f"homoglyph-tol. : {report['homoglyph_tolerant_top1']:.4f}")
    print(f"loss           : {report['loss']:.4f}")
    print(f"latency        : {report['latency_ms']:.3f} ms / sample (batch=1)")
    charset: list[str] = report["charset"]
    per_class = report["per_class_acc"]
    worst = np.argsort(per_class)[:10]
    print("worst classes  :")
    for i in worst:
        print(f"  {charset[int(i)]!r:6s}  {per_class[int(i)]:.3f}")
    print("top confusions :")
    for frac, src, dst, count in report["confusion_pairs"][:8]:
        print(f"  {src!r} -> {dst!r}  {count}  ({frac:.2%})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 2: evaluate baseline checkpoint.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = evaluate(args.ckpt, args.npz, args.split, args.batch_size, args.seed)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
