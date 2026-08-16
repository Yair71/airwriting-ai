"""MorphNet structured pruning: L1 on BatchNorm γ, then physical channel/GRU shrink.

Loss (offline, GPU/CPU training — not the PC daemon):
    L = L_CE + λ_resource * Σ_l Cost(l) * |γ_l|
Cost(l) = conv FLOPs of layer l. Channels with |γ| < threshold are removed.
GRU hidden units are ranked by L1 of their gate weights and shrunk to a target width.

Memory: pruned FP32 ~60–90 KB; INT8 ONNX target ≤ 40 KB.
CPU: prune rebuild is < 10 ms; fine-tune is the wall-time cost (~1 min/epoch on CPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ml_pipeline.dataset import DEFAULT_NPZ, REPO_ROOT, UnistrokeDataset
from ml_pipeline.evaluate import evaluate_loader, load_checkpoint, visual_class_map
from ml_pipeline.export_onnx import DEFAULT_ONNX_INT8, export_pruned_int8
from ml_pipeline.model import ModelConfig, UnistrokeNet, count_flops, count_parameters, fp32_bytes
from ml_pipeline.train import DEFAULT_CKPT, save_checkpoint, seed_everything

DEFAULT_PRUNED_CKPT = REPO_ROOT / "data" / "checkpoints" / "pruned_model.pth"
DEFAULT_PRUNE_CFG = REPO_ROOT / "configs" / "pruning_config.json"


def load_prune_config(path: Path = DEFAULT_PRUNE_CFG) -> dict:
    defaults = {
        "lambda_resource": 0.002,
        "sparsify_epochs": 3,
        "sparsify_lr": 3e-4,
        "finetune_epochs": 5,
        "finetune_lr": 5e-4,
        "warmup_epochs": 1,
        "keep_fraction": 0.5,
        "min_channels": 8,
        "gru_hidden_pruned": 16,
        "gamma_eps": 1e-12,
        "grad_clip": 1.0,
        "batch_size": 512,
    }
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        defaults.update(payload)
    return defaults


def visual_loaders(
    npz_path: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
    label_map: np.ndarray | None = None,
    visual_charset: list[str] | None = None,
    charset: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, dict]:
    train_ds = UnistrokeDataset(npz_path=npz_path, split="train", seed=seed)
    val_ds = UnistrokeDataset(npz_path=npz_path, split="val", seed=seed)
    if label_map is None or visual_charset is None:
        label_map, visual_charset = visual_class_map(train_ds.charset)
    if charset is None:
        charset = train_ds.charset
    map_t = torch.from_numpy(np.asarray(label_map, dtype=np.int64))
    pin = device.type == "cuda"

    def _make(ds: UnistrokeDataset, shuffle: bool) -> DataLoader:
        features = torch.from_numpy(np.ascontiguousarray(ds.features))
        raw = torch.from_numpy(np.ascontiguousarray(ds.labels.astype(np.int64)))
        return DataLoader(
            TensorDataset(features, map_t[raw]),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=pin,
        )

    meta = {
        "charset": charset,
        "visual_charset": list(visual_charset),
        "label_map": np.asarray(label_map, dtype=np.int64),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    return _make(train_ds, True), _make(val_ds, False), meta


def morphnet_resource_penalty(model: UnistrokeNet) -> torch.Tensor:
    """Σ Cost(l) * |γ_l| with Cost normalized by total conv FLOPs."""
    device = model.classifier.weight.device
    seq = model.config.seq_len
    in_c = model.config.n_features
    costs: list[float] = []
    gammas: list[torch.Tensor] = []
    for block, out_c, kernel in zip(model.conv, model.config.conv_channels, model.config.kernel_sizes):
        bn = next(m for m in block if isinstance(m, nn.BatchNorm1d))
        costs.append(float(2 * in_c * out_c * kernel * seq))
        gammas.append(bn.weight)
        in_c = out_c
    total_cost = sum(costs) or 1.0
    penalty = torch.zeros((), device=device)
    for cost, gamma in zip(costs, gammas):
        penalty = penalty + (cost / total_cost) * gamma.abs().sum()
    return penalty


def conv_keep_masks(model: UnistrokeNet, keep_fraction: float, min_channels: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Per-layer MorphNet keep: top keep_fraction of |γ|, at least min_channels."""
    masks: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    for bn in model.morphnet_batchnorms:
        mag = bn.weight.detach().abs()
        k = max(int(min_channels), int(round(keep_fraction * mag.numel())))
        k = min(max(k, 1), mag.numel())
        thresh = torch.topk(mag, k).values.min()
        thresholds.append(thresh)
        masks.append(mag >= thresh)
    stacked = torch.stack([t.reshape(()) for t in thresholds]).mean()
    return masks, stacked


def gru_unit_scores(weight_ih: torch.Tensor, weight_hh: torch.Tensor) -> torch.Tensor:
    hidden = int(weight_hh.shape[1])
    ih = weight_ih.reshape(3, hidden, -1).abs().mean(dim=(0, 2))
    hh = weight_hh.reshape(3, hidden, hidden).abs().mean(dim=(0, 2))
    return ih + hh


def _index(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).view(-1)


def _copy_bn(src: nn.BatchNorm1d, dst: nn.BatchNorm1d, keep: torch.Tensor) -> None:
    idx = _index(keep)
    dst.weight.data.copy_(src.weight.data[idx])
    dst.bias.data.copy_(src.bias.data[idx])
    dst.running_mean.data.copy_(src.running_mean.data[idx])
    dst.running_var.data.copy_(src.running_var.data[idx])
    dst.num_batches_tracked.data.copy_(src.num_batches_tracked.data)


def _copy_gru_direction(
    src_ih: torch.Tensor,
    src_hh: torch.Tensor,
    src_bih: torch.Tensor,
    src_bhh: torch.Tensor,
    dst_ih: torch.Tensor,
    dst_hh: torch.Tensor,
    dst_bih: torch.Tensor,
    dst_bhh: torch.Tensor,
    in_idx: torch.Tensor,
    h_idx: torch.Tensor,
) -> None:
    h_old = int(src_hh.shape[1])
    h_new = int(h_idx.numel())
    for gate in range(3):
        rows_old = gate * h_old + h_idx
        sl = slice(gate * h_new, (gate + 1) * h_new)
        dst_ih.data[sl] = src_ih[rows_old][:, in_idx]
        dst_hh.data[sl] = src_hh[rows_old][:, h_idx]
        dst_bih.data[sl] = src_bih[rows_old]
        dst_bhh.data[sl] = src_bhh[rows_old]


def rebuild_pruned(
    src: UnistrokeNet,
    conv_masks: list[torch.Tensor],
    gru_hidden_new: int,
) -> UnistrokeNet:
    if gru_hidden_new < 1:
        raise ValueError("gru_hidden_new must be >= 1")
    conv_channels = tuple(int(m.sum().item()) for m in conv_masks)
    if any(c < 1 for c in conv_channels):
        raise RuntimeError(f"refusing to prune a conv layer to 0 channels: {conv_channels}")

    cfg = ModelConfig(
        n_features=src.config.n_features,
        n_classes=src.config.n_classes,
        conv_channels=conv_channels,
        kernel_sizes=src.config.kernel_sizes,
        gru_hidden=int(gru_hidden_new),
        attn_dim=None,
        dropout=src.config.dropout,
        seq_len=src.config.seq_len,
    )
    dst = UnistrokeNet(cfg)
    in_idx = torch.arange(src.config.n_features)
    for i, mask in enumerate(conv_masks):
        src_conv = next(m for m in src.conv[i] if isinstance(m, nn.Conv1d))
        dst_conv = next(m for m in dst.conv[i] if isinstance(m, nn.Conv1d))
        src_bn = next(m for m in src.conv[i] if isinstance(m, nn.BatchNorm1d))
        dst_bn = next(m for m in dst.conv[i] if isinstance(m, nn.BatchNorm1d))
        out_idx = _index(mask)
        dst_conv.weight.data.copy_(src_conv.weight.data[out_idx][:, in_idx])
        _copy_bn(src_bn, dst_bn, mask)
        in_idx = out_idx

    h_old = src.config.gru_hidden
    h_new = int(gru_hidden_new)
    scores_f = gru_unit_scores(src.gru.weight_ih_l0, src.gru.weight_hh_l0)
    scores_r = gru_unit_scores(src.gru.weight_ih_l0_reverse, src.gru.weight_hh_l0_reverse)
    h_idx_f = torch.topk(scores_f, h_new).indices.sort().values
    h_idx_r = torch.topk(scores_r, h_new).indices.sort().values
    _copy_gru_direction(
        src.gru.weight_ih_l0,
        src.gru.weight_hh_l0,
        src.gru.bias_ih_l0,
        src.gru.bias_hh_l0,
        dst.gru.weight_ih_l0,
        dst.gru.weight_hh_l0,
        dst.gru.bias_ih_l0,
        dst.gru.bias_hh_l0,
        in_idx,
        h_idx_f,
    )
    _copy_gru_direction(
        src.gru.weight_ih_l0_reverse,
        src.gru.weight_hh_l0_reverse,
        src.gru.bias_ih_l0_reverse,
        src.gru.bias_hh_l0_reverse,
        dst.gru.weight_ih_l0_reverse,
        dst.gru.weight_hh_l0_reverse,
        dst.gru.bias_ih_l0_reverse,
        dst.gru.bias_hh_l0_reverse,
        in_idx,
        h_idx_r,
    )

    out_keep = torch.cat([h_idx_f, h_idx_r + h_old])
    n_attn = 2 * h_new
    if (
        src.attn.proj.weight.shape[0] == h_old * 2
        and src.attn.proj.weight.shape[1] == h_old * 2
        and dst.attn.proj.weight.shape == (n_attn, n_attn)
    ):
        dst.attn.proj.weight.data.copy_(src.attn.proj.weight.data[out_keep][:, out_keep])
        dst.attn.proj.bias.data.copy_(src.attn.proj.bias.data[out_keep])
        dst.attn.v.weight.data.copy_(src.attn.v.weight.data[:, out_keep])
    dst.classifier.weight.data.copy_(src.classifier.weight.data[:, out_keep])
    dst.classifier.bias.data.copy_(src.classifier.bias.data)
    return dst


def _param_groups(model: UnistrokeNet, weight_decay: float) -> list[dict]:
    bn_ids = {id(p) for bn in model.morphnet_batchnorms for p in bn.parameters()}
    decay = [p for p in model.parameters() if p.requires_grad and id(p) not in bn_ids]
    nodecay = [p for p in model.parameters() if p.requires_grad and id(p) in bn_ids]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def run_epoch(
    model: UnistrokeNet,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: AdamW | None,
    grad_clip: float,
    lambda_resource: float,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    correct = 0
    n = 0
    for xb, yb in tqdm(loader, leave=False, desc="train" if train else "eval"):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            loss = loss + float(lambda_resource) * morphnet_resource_penalty(model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = int(yb.size(0))
        loss_sum += float(loss.item()) * bs
        correct += int((logits.detach().argmax(dim=-1) == yb).sum().item())
        n += bs
    return loss_sum / max(n, 1), correct / max(n, 1)


def prune_and_finetune(
    baseline_ckpt: Path = DEFAULT_CKPT,
    pruned_ckpt: Path = DEFAULT_PRUNED_CKPT,
    npz_path: Path = DEFAULT_NPZ,
    onnx_path: Path = DEFAULT_ONNX_INT8,
    seed: int = 42,
    cfg: dict | None = None,
) -> dict:
    cfg = cfg or load_prune_config()
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, blob = load_checkpoint(baseline_ckpt, device)
    ckpt_map = np.asarray(blob["label_map"], dtype=np.int64) if "label_map" in blob else None
    ckpt_visual = [str(x) for x in blob["visual_charset"]] if "visual_charset" in blob else None
    ckpt_charset = [str(x) for x in blob["charset"]] if "charset" in blob else None
    train_loader, val_loader, meta = visual_loaders(
        npz_path,
        int(cfg["batch_size"]),
        seed,
        device,
        label_map=ckpt_map,
        visual_charset=ckpt_visual,
        charset=ckpt_charset,
    )
    baseline_acc = float(blob.get("val_accuracy", 0.0))
    live = evaluate_loader(model, val_loader, device)
    baseline_acc = float(live["acc"])
    print(f"baseline top-1 : {baseline_acc:.4f}  params={count_parameters(model)}")

    criterion = nn.CrossEntropyLoss()
    sparsify_epochs = int(cfg["sparsify_epochs"])
    if sparsify_epochs > 0:
        opt = AdamW(_param_groups(model, 2e-2), lr=float(cfg["sparsify_lr"]))
        print(f"sparsify {sparsify_epochs} epochs  λ={cfg['lambda_resource']}")
        for epoch in range(1, sparsify_epochs + 1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = run_epoch(
                model, train_loader, device, criterion, opt, float(cfg["grad_clip"]), float(cfg["lambda_resource"])
            )
            val = evaluate_loader(model, val_loader, device)
            print(
                f"  sparsify {epoch}/{sparsify_epochs}  "
                f"train {tr_loss:.4f}/{tr_acc:.4f}  val {val['acc']:.4f}  {time.perf_counter()-t0:.1f}s"
            )
            for i, bn in enumerate(model.morphnet_batchnorms):
                g = bn.weight.detach().abs()
                print(f"    conv{i} |γ| mean={float(g.mean()):.4f} min={float(g.min()):.4f} max={float(g.max()):.4f}")

    masks, threshold = conv_keep_masks(model, float(cfg["keep_fraction"]), int(cfg["min_channels"]))
    print(f"γ threshold    : {float(threshold):.6f}")
    print(f"conv keep      : {[int(m.sum().item()) for m in masks]} / {list(model.config.conv_channels)}")

    gru_new = min(int(cfg["gru_hidden_pruned"]), int(model.config.gru_hidden))
    pruned = rebuild_pruned(model, masks, gru_new).to(device)
    print(
        f"pruned params  : {count_parameters(pruned)}  "
        f"({fp32_bytes(count_parameters(pruned))/1024:.1f} KB FP32)  "
        f"flops={count_flops(pruned):,}"
    )
    pre_ft = evaluate_loader(pruned, val_loader, device)
    print(f"pre-finetune   : {pre_ft['acc']:.4f}")

    ft_epochs = int(cfg["finetune_epochs"])
    warmup_epochs = int(cfg["warmup_epochs"])
    opt = AdamW(_param_groups(pruned, 1e-2), lr=float(cfg["finetune_lr"]))
    warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_epochs))
    cosine = CosineAnnealingLR(opt, T_max=max(1, ft_epochs - warmup_epochs), eta_min=1e-5)
    sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[max(1, warmup_epochs)])

    best_acc = -1.0
    best_state = None
    print(f"fine-tune {ft_epochs} epochs  warmup={warmup_epochs}")
    for epoch in range(1, ft_epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(pruned, train_loader, device, criterion, opt, float(cfg["grad_clip"]), 0.0)
        val = evaluate_loader(pruned, val_loader, device)
        sched.step()
        print(
            f"  ft {epoch}/{ft_epochs}  train {tr_loss:.4f}/{tr_acc:.4f}  "
            f"val {val['acc']:.4f}  lr={opt.param_groups[0]['lr']:.2e}  {time.perf_counter()-t0:.1f}s"
        )
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in pruned.state_dict().items()}

    if best_state is not None:
        pruned.load_state_dict(best_state)
        pruned.to(device)

    drop = baseline_acc - best_acc
    save_checkpoint(
        pruned_ckpt,
        pruned,
        ft_epochs,
        best_acc,
        {
            "baseline_acc": baseline_acc,
            "accuracy_drop": drop,
            "gamma_threshold": float(threshold),
            "conv_channels": list(pruned.config.conv_channels),
            "gru_hidden": pruned.config.gru_hidden,
        },
        meta["charset"],
        meta["visual_charset"],
        meta["label_map"],
    )
    print(f"saved pruned   : {pruned_ckpt}  val={best_acc:.4f}  drop={drop:+.4f}")

    onnx_info = export_pruned_int8(pruned_ckpt, onnx_path, val_loader, device)
    print(f"INT8 ONNX      : {onnx_info['path']}  {onnx_info['bytes']/1024:.2f} KB  top-1={onnx_info['acc']:.4f}")
    print(f"size limit 40KB: {'PASS' if onnx_info['bytes'] <= 40 * 1024 else 'FAIL'}")
    print(f"drop < 0.5%    : {'PASS' if drop < 0.005 else 'FAIL'}  (pytorch pruned)")
    int8_drop = baseline_acc - float(onnx_info["acc"])
    print(f"INT8 drop      : {int8_drop:+.4f}  {'PASS' if int8_drop < 0.005 else 'FAIL'}")
    return {
        "baseline_acc": baseline_acc,
        "pruned_acc": best_acc,
        "drop": drop,
        "onnx": onnx_info,
        "n_parameters": count_parameters(pruned),
        "checkpoint": str(pruned_ckpt),
    }


def recover_pruned(
    pruned_ckpt: Path = DEFAULT_PRUNED_CKPT,
    baseline_ckpt: Path = DEFAULT_CKPT,
    npz_path: Path = DEFAULT_NPZ,
    onnx_path: Path = DEFAULT_ONNX_INT8,
    epochs: int = 12,
    lr: float = 1.5e-3,
    batch_size: int = 512,
    seed: int = 42,
) -> dict:
    """Continue fine-tuning a pruned checkpoint with teacher distillation."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, blob = load_checkpoint(baseline_ckpt, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student, pblob = load_checkpoint(pruned_ckpt, device)
    train_loader, val_loader, meta = visual_loaders(
        npz_path,
        batch_size,
        seed,
        device,
        label_map=np.asarray(pblob["label_map"], dtype=np.int64),
        visual_charset=[str(x) for x in pblob["visual_charset"]],
        charset=[str(x) for x in pblob["charset"]],
    )
    baseline_acc = float(evaluate_loader(teacher, val_loader, device)["acc"])
    start_acc = float(evaluate_loader(student, val_loader, device)["acc"])
    print(f"recover from {start_acc:.4f}  teacher {baseline_acc:.4f}  {epochs} epochs lr={lr}")
    criterion = nn.CrossEntropyLoss()
    opt = AdamW(_param_groups(student, 1e-2), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    best_acc = start_acc
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    temp = 2.0
    for epoch in range(1, epochs + 1):
        student.train()
        t0 = time.perf_counter()
        loss_sum = 0.0
        correct = 0
        n = 0
        for xb, yb in tqdm(train_loader, leave=False, desc=f"recover {epoch}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = student(xb)
            with torch.no_grad():
                t_logits = teacher(xb)
            ce = criterion(logits, yb)
            kl = nn.functional.kl_div(
                torch.log_softmax(logits / temp, dim=-1),
                torch.softmax(t_logits / temp, dim=-1),
                reduction="batchmean",
            ) * (temp * temp)
            loss = 0.7 * ce + 0.3 * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            bs = int(yb.size(0))
            loss_sum += float(loss.item()) * bs
            correct += int((logits.detach().argmax(dim=-1) == yb).sum().item())
            n += bs
        sched.step()
        val = evaluate_loader(student, val_loader, device)
        print(
            f"  recover {epoch}/{epochs}  train {loss_sum/max(n,1):.4f}/{correct/max(n,1):.4f}  "
            f"val {val['acc']:.4f}  lr={opt.param_groups[0]['lr']:.2e}  {time.perf_counter()-t0:.1f}s"
        )
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    student.load_state_dict(best_state)
    student.to(device)
    drop = baseline_acc - best_acc
    save_checkpoint(
        pruned_ckpt,
        student,
        epochs,
        best_acc,
        {"baseline_acc": baseline_acc, "accuracy_drop": drop, "recovered": True},
        meta["charset"],
        meta["visual_charset"],
        meta["label_map"],
    )
    onnx_info = export_pruned_int8(pruned_ckpt, onnx_path, val_loader, device)
    print(f"saved recovered: {pruned_ckpt}  val={best_acc:.4f}  drop={drop:+.4f}")
    print(f"INT8 ONNX      : {onnx_info['path']}  {onnx_info['bytes']/1024:.2f} KB  top-1={onnx_info['acc']:.4f}")
    print(f"size limit 40KB: {'PASS' if onnx_info['bytes'] <= 40 * 1024 else 'FAIL'}")
    int8_drop = baseline_acc - float(onnx_info["acc"])
    print(f"INT8 drop      : {int8_drop:+.4f}  {'PASS' if int8_drop < 0.005 else 'FAIL'}")
    return {"pruned_acc": best_acc, "drop": drop, "onnx": onnx_info}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 3: MorphNet prune + INT8 ONNX export.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_PRUNED_CKPT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_INT8)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--config", type=Path, default=DEFAULT_PRUNE_CFG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recover", action="store_true", help="fine-tune an existing pruned checkpoint")
    parser.add_argument("--recover-epochs", type=int, default=12)
    args = parser.parse_args(argv)
    if args.recover:
        recover_pruned(args.out, args.baseline, args.npz, args.onnx, epochs=args.recover_epochs, seed=args.seed)
    else:
        prune_and_finetune(args.baseline, args.out, args.npz, args.onnx, args.seed, load_prune_config(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
