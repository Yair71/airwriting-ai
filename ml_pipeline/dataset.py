"""Kinematic preprocessor, synthetic dataset writer, and PyTorch Dataset.

Data contract (must match C++ preprocessor.cpp)
-----------------------------------------------
Input:  raw polyline P = {(x, y)} inside the 5 cm box (here: font / augmented units).
Step 1: 3-point moving average.
Step 2: arc-length resample to exactly 64 points.
Step 3: centroid -> (0, 0), scale so max(|x|, |y|) = 1  => range [-1, 1].
Step 4: features float32 [63, 4] = [dx, dy, sin(theta), cos(theta)].

NPZ layout (`data/synthetic/synthetic_dataset.npz`)
    features : float32 [N, 63, 4]
    points   : float32 [N, 64, 2]
    labels   : int32   [N]
    charset  : unicode [C]
Memory: N=30_000 -> ~48 MB features + ~15 MB points.
CPU: ~0.2–0.5 ms/sample generation (extract cached; augment + preprocess dominate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ml_pipeline.augmentor import AugmentConfig, Augmentor
from ml_pipeline.font_sampler import (
    CHARSETS,
    REPO_ROOT,
    RESAMPLE_POINTS,
    build_charset,
    discover_fonts,
    extract_glyph_polyline,
    font_covers,
    resample_arc_length,
    script_of,
)

FEATURE_STEPS = RESAMPLE_POINTS - 1  # 63
N_FEATURES = 4
DEFAULT_NPZ = REPO_ROOT / "data" / "synthetic" / "synthetic_dataset.npz"
DEFAULT_PLOT = REPO_ROOT / "data" / "synthetic" / "verification.png"
DEFAULT_MAP = REPO_ROOT / "configs" / "unistroke_map.json"


def moving_average_3(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return pts.copy()
    out = pts.copy()
    out[1:-1] = (pts[:-2] + pts[1:-1] + pts[2:]) / 3.0
    return out


def normalize_xy(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).copy()
    pts -= pts.mean(axis=0, keepdims=True)
    extent = float(np.max(np.abs(pts)))
    if extent < 1e-12:
        return pts
    return pts / extent


def points_to_features(points64: np.ndarray) -> np.ndarray:
    """(64, 2) -> (63, 4) [dx, dy, sin(theta), cos(theta)]."""
    pts = np.asarray(points64, dtype=np.float64)
    if pts.shape != (RESAMPLE_POINTS, 2):
        raise ValueError(f"expected ({RESAMPLE_POINTS}, 2), got {pts.shape}")
    delta = np.diff(pts, axis=0)
    mag = np.linalg.norm(delta, axis=1)
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    zero = mag < 1e-12
    sin_t[zero] = 0.0
    cos_t[zero] = 1.0
    feats = np.stack([delta[:, 0], delta[:, 1], sin_t, cos_t], axis=1)
    return feats.astype(np.float32, copy=False)


def kinematic_preprocess(raw_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (points64 float32 [64,2], features float32 [63,4])."""
    pts = np.asarray(raw_points, dtype=np.float64)
    if pts.shape[0] < 2:
        zeros64 = np.zeros((RESAMPLE_POINTS, 2), dtype=np.float32)
        zeros_f = np.zeros((FEATURE_STEPS, N_FEATURES), dtype=np.float32)
        zeros_f[:, 3] = 1.0
        return zeros64, zeros_f
    smoothed = moving_average_3(pts)
    resampled = resample_arc_length(smoothed, RESAMPLE_POINTS)
    normalized = normalize_xy(resampled)
    features = points_to_features(normalized)
    return normalized.astype(np.float32), features


def write_unistroke_map(path: Path, charset: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    char_to_id = {ch: i for i, ch in enumerate(charset)}
    payload = {
        "n_classes": len(charset),
        "resample_points": RESAMPLE_POINTS,
        "sequence_length": FEATURE_STEPS,
        "n_features": N_FEATURES,
        "feature_order": ["dx", "dy", "sin_theta", "cos_theta"],
        "charset": list(charset),
        "char_to_id": char_to_id,
        "scripts": {name: list(chars) for name, chars in CHARSETS.items()},
        "coordinate_frame": "typographic_y_up_centroid_normalized",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class UnistrokeDataset(Dataset):
    """Loads synthetic_dataset.npz. Returns (features [63, 4] float32, label int)."""

    def __init__(
        self,
        npz_path: str | Path = DEFAULT_NPZ,
        split: str = "train",
        val_fraction: float = 0.15,
        seed: int = 42,
        load_points: bool = False,
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train|val|all, got {split!r}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        path = Path(npz_path)
        if not path.is_file():
            raise FileNotFoundError(f"dataset not found: {path}. Run: python -m ml_pipeline")

        blob = np.load(path, allow_pickle=False, mmap_mode="r")
        features = blob["features"]
        labels = blob["labels"]
        n = int(features.shape[0])
        if features.ndim != 3 or features.shape[1] != FEATURE_STEPS or features.shape[2] not in {4, 6}:
            raise ValueError(
                f"bad features shape {features.shape}, expected (N, {FEATURE_STEPS}, 4|6)"
            )

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * val_fraction)))
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        if split == "train":
            idx = train_idx
        elif split == "val":
            idx = val_idx
        else:
            idx = np.arange(n)

        self.features = np.ascontiguousarray(features[idx])
        self.labels = np.ascontiguousarray(labels[idx])
        self.points = (
            np.ascontiguousarray(blob["points"][idx]) if load_points and "points" in blob.files else None
        )
        self.charset = [str(x) for x in blob["charset"]]
        self.indices = idx
        self.split = split
        self.n_features = int(self.features.shape[-1])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.array(self.features[index], copy=True))
        y = torch.tensor(int(self.labels[index]), dtype=torch.int64)
        return x, y

    @property
    def n_classes(self) -> int:
        return len(self.charset)


def generate_synthetic_dataset(
    out_path: Path = DEFAULT_NPZ,
    variants_per_glyph: int = 32,
    seed: int = 42,
    map_path: Path = DEFAULT_MAP,
) -> dict[str, int | Path]:
    if variants_per_glyph < 0:
        raise ValueError("variants_per_glyph must be >= 0")

    charset = build_charset()
    char_to_id = {ch: i for i, ch in enumerate(charset)}
    fonts = discover_fonts()
    if not fonts:
        raise FileNotFoundError(
            "No TTF/OTF fonts found. Drop fonts into data/fonts/{latin,cyrillic,hebrew} "
            "or install Arial/Times/Segoe UI."
        )

    rng = np.random.default_rng(seed)
    augmentor = Augmentor(AugmentConfig(), rng)

    feature_rows: list[np.ndarray] = []
    point_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    skipped = 0
    used_pairs = 0
    coverage: dict[str, int] = {name: 0 for name in CHARSETS}

    for font_path in tqdm(fonts, desc="fonts", unit="font"):
        try:
            covered = font_covers(font_path, charset)
        except Exception:
            skipped += len(charset)
            continue
        dense_cache: dict[str, np.ndarray] = {}
        for char in charset:
            if not covered.get(char, False):
                skipped += 1
                continue
            try:
                dense = extract_glyph_polyline(font_path, char)
            except Exception:
                skipped += 1
                continue
            if dense.shape[0] < 2:
                skipped += 1
                continue
            dense_cache[char] = dense
            used_pairs += 1
            coverage[script_of(char)] += 1

            samples = [dense]
            for _ in range(variants_per_glyph):
                samples.append(augmentor.augment(dense))
            label = char_to_id[char]
            for sample in samples:
                points64, feats = kinematic_preprocess(sample)
                if not np.isfinite(feats).all() or not np.isfinite(points64).all():
                    continue
                feature_rows.append(feats)
                point_rows.append(points64)
                label_rows.append(label)

    if not feature_rows:
        raise RuntimeError("No glyphs could be extracted. Check font coverage for Latin/Cyrillic/Hebrew.")

    features = np.stack(feature_rows, axis=0)
    points = np.stack(point_rows, axis=0)
    labels = np.asarray(label_rows, dtype=np.int32)
    charset_arr = np.asarray(list(charset))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features,
        points=points,
        labels=labels,
        charset=charset_arr,
    )
    write_unistroke_map(map_path, charset)

    n_classes_present = int(np.unique(labels).size)
    return {
        "n_samples": int(labels.shape[0]),
        "n_classes": len(charset),
        "n_classes_present": n_classes_present,
        "n_fonts": len(fonts),
        "n_font_glyph_pairs": used_pairs,
        "n_skipped": skipped,
        "latin_pairs": coverage["latin"],
        "cyrillic_pairs": coverage["cyrillic"],
        "hebrew_pairs": coverage["hebrew"],
        "out_path": out_path,
        "map_path": map_path,
        "npz_bytes": out_path.stat().st_size,
    }


def plot_verification(npz_path: Path, plot_path: Path, n_show: int = 5, seed: int = 0, show: bool = True) -> None:
    blob = np.load(npz_path, allow_pickle=False)
    points = blob["points"]
    labels = blob["labels"]
    charset = [str(x) for x in blob["charset"]]
    rng = np.random.default_rng(seed)
    n = int(points.shape[0])
    if n == 0:
        raise RuntimeError("dataset is empty")
    unique = np.unique(labels)
    n_show = min(int(n_show), int(unique.size), n)
    chosen_labels = rng.choice(unique, size=n_show, replace=False)
    idx = [int(rng.choice(np.flatnonzero(labels == lab))) for lab in chosen_labels]

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fig, axes = plt.subplots(1, len(idx), figsize=(3.2 * len(idx), 3.4))
    if len(idx) == 1:
        axes = [axes]
    win_fonts = Path(os_win_fonts())
    for segoe in ("segoeui.ttf", "arial.ttf"):
        candidate = win_fonts / segoe
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break

    for ax, sample_i in zip(axes, idx):
        pts = points[int(sample_i)]
        ch = charset[int(labels[int(sample_i)])]
        t = np.linspace(0.0, 1.0, pts.shape[0])
        ax.scatter(pts[:, 0], pts[:, 1], c=t, cmap="viridis", s=18, zorder=3)
        ax.plot(pts[:, 0], pts[:, 1], color="#444444", linewidth=1.2, alpha=0.85)
        ax.set_title(f"{ch}  id={int(labels[int(sample_i)])}", fontsize=14)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)
    fig.suptitle("Unistroke verification (64-pt, centroid-normalized)", fontsize=12)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def os_win_fonts() -> str:
    import os

    return str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 1: generate synthetic unistroke dataset from TTF fonts.")
    parser.add_argument("--out", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--variants", type=int, default=32, help="augmented copies per (font, glyph)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="open the matplotlib window")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    stats = generate_synthetic_dataset(
        out_path=args.out,
        variants_per_glyph=args.variants,
        seed=args.seed,
        map_path=args.map,
    )
    print("synthetic dataset written")
    print(f"  samples          : {stats['n_samples']}")
    print(f"  classes          : {stats['n_classes_present']}/{stats['n_classes']}")
    print(f"  fonts            : {stats['n_fonts']}")
    print(f"  font-glyph pairs : {stats['n_font_glyph_pairs']}")
    print(f"  skipped glyphs   : {stats['n_skipped']}")
    print(f"  latin pairs      : {stats['latin_pairs']}")
    print(f"  cyrillic pairs   : {stats['cyrillic_pairs']}")
    print(f"  hebrew pairs     : {stats['hebrew_pairs']}")
    print(f"  npz              : {stats['out_path']}  ({int(stats['npz_bytes']) / 1e6:.2f} MB)")
    print(f"  label map        : {stats['map_path']}")
    print(f"  tensor           : features (N, {FEATURE_STEPS}, {N_FEATURES})  points (N, {RESAMPLE_POINTS}, 2)")

    if not args.no_plot:
        plot_verification(Path(stats["out_path"]), args.plot, n_show=5, seed=args.seed, show=args.show)
        print(f"  verification plot: {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
