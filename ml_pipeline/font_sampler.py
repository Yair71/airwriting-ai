"""TTF/OTF glyph outline extraction and arc-length resampling.

Data contract
-------------
extract_glyph_polyline(font, char) -> float64 [M, 2], M variable, font units, y-up.
resample_arc_length(points, 64)    -> float64 [64, 2] equidistant along the path.

Empty glyphs return shape (0, 2). Callers must skip those.
CPU: ~0.05–0.3 ms per glyph on a desktop CPU (FreeType decompose + Bezier flatten).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

try:
    import freetype
except ImportError as exc:  # pragma: no cover
    raise ImportError("freetype-py is required for glyph extraction") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_FONT_ROOT = REPO_ROOT / "data" / "fonts"
RESAMPLE_POINTS = 64
_BEZIER_SAMPLES = 16
_MIN_CONTOUR_LENGTH = 1e-6

# Latin A–Z + 0–9, Cyrillic А–Я + Ё, Hebrew א–ת (includes five finals).
CHARSETS: dict[str, tuple[str, ...]] = {
    "latin": tuple(chr(c) for c in range(ord("0"), ord("9") + 1))
    + tuple(chr(c) for c in range(ord("A"), ord("Z") + 1)),
    "cyrillic": tuple(chr(c) for c in range(0x0410, 0x0430)) + ("Ё",),
    "hebrew": tuple(chr(c) for c in range(0x05D0, 0x05EB)),
}

_WINDOWS_FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
_SYSTEM_FONT_FILES = (
    "arial.ttf",
    "arialbd.ttf",
    "ariali.ttf",
    "arialbi.ttf",
    "times.ttf",
    "timesbd.ttf",
    "timesi.ttf",
    "timesbi.ttf",
    "cour.ttf",
    "courbd.ttf",
    "couri.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "calibrii.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "segoeuii.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "verdana.ttf",
    "verdanab.ttf",
    "verdanai.ttf",
    "georgia.ttf",
    "georgiab.ttf",
    "georgiai.ttf",
    "comic.ttf",
    "comici.ttf",
    "david.ttf",
    "davidbd.ttf",
)

_LINUX_FONT_GLOBS = (
    "/usr/share/fonts/truetype/**/*.ttf",
    "/usr/share/fonts/truetype/**/*.otf",
    "/usr/local/share/fonts/**/*.ttf",
)


def build_charset() -> tuple[str, ...]:
    return CHARSETS["latin"] + CHARSETS["cyrillic"] + CHARSETS["hebrew"]


def script_of(char: str) -> str:
    if char in CHARSETS["latin"]:
        return "latin"
    if char in CHARSETS["cyrillic"]:
        return "cyrillic"
    if char in CHARSETS["hebrew"]:
        return "hebrew"
    raise KeyError(f"Character {char!r} is not in the unistroke charset")


def discover_fonts(extra_dirs: Sequence[Path] | None = None) -> list[Path]:
    """Return unique existing TTF/OTF paths: bundled data/fonts first, then system."""
    found: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        if not path.is_file():
            return
        if path.suffix.lower() not in {".ttf", ".otf", ".ttc"}:
            return
        key = str(path.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    search_roots = [
        BUNDLED_FONT_ROOT / "latin",
        BUNDLED_FONT_ROOT / "cyrillic",
        BUNDLED_FONT_ROOT / "hebrew",
        BUNDLED_FONT_ROOT,
    ]
    if extra_dirs:
        search_roots.extend(extra_dirs)
    for root in search_roots:
        if not root.is_dir():
            continue
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for path in sorted(root.rglob(pattern)):
                _add(path)

    if sys.platform == "win32":
        for name in _SYSTEM_FONT_FILES:
            _add(_WINDOWS_FONT_DIR / name)
    else:
        import glob

        for pattern in _LINUX_FONT_GLOBS:
            for match in glob.glob(pattern, recursive=True):
                _add(Path(match))

    return found


def _xy(point) -> np.ndarray:
    return np.array([float(point.x), float(point.y)], dtype=np.float64)


def _sample_quadratic(p0: np.ndarray, control: np.ndarray, p1: np.ndarray, n: int = _BEZIER_SAMPLES) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    mt = 1.0 - t
    return (mt * mt)[:, None] * p0 + (2.0 * mt * t)[:, None] * control + (t * t)[:, None] * p1


def _sample_cubic(
    p0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    p1: np.ndarray,
    n: int = _BEZIER_SAMPLES,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    mt = 1.0 - t
    return (
        (mt**3)[:, None] * p0
        + (3.0 * mt * mt * t)[:, None] * c1
        + (3.0 * mt * t * t)[:, None] * c2
        + (t**3)[:, None] * p1
    )


class _OutlineCollector:
    """FreeType decompose sink. Callbacks must not raise."""

    def __init__(self) -> None:
        self._contours: list[list[np.ndarray]] = []
        self._cur: list[np.ndarray] = []

    def _flush(self) -> None:
        if self._cur:
            self._contours.append(self._cur)
            self._cur = []

    def move_to(self, a, _ctx) -> None:
        self._flush()
        self._cur = [_xy(a)]

    def line_to(self, a, _ctx) -> None:
        pt = _xy(a)
        if not self._cur:
            self._cur = [pt]
            return
        self._cur.append(pt)

    def conic_to(self, control, to, _ctx) -> None:
        if not self._cur:
            self._cur = [_xy(control)]
        samples = _sample_quadratic(self._cur[-1], _xy(control), _xy(to))
        self._cur.extend(list(samples[1:]))

    def cubic_to(self, c1, c2, to, _ctx) -> None:
        if not self._cur:
            self._cur = [_xy(c1)]
        samples = _sample_cubic(self._cur[-1], _xy(c1), _xy(c2), _xy(to))
        self._cur.extend(list(samples[1:]))

    def contours(self) -> list[np.ndarray]:
        self._flush()
        out: list[np.ndarray] = []
        for contour in self._contours:
            if len(contour) < 2:
                continue
            pts = np.ascontiguousarray(np.vstack(contour), dtype=np.float64)
            pts = dedupe_consecutive(pts)
            if pts.shape[0] >= 2:
                out.append(pts)
        return out


def dedupe_consecutive(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], delta > eps])
    return points[keep]


def _contour_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def join_contours(contours: Sequence[np.ndarray], bridge_points: int = 4) -> np.ndarray:
    """Concatenate contours into one unistroke: top-to-bottom, then left-to-right."""
    kept = [c for c in contours if _contour_length(c) > _MIN_CONTOUR_LENGTH]
    if not kept:
        return np.zeros((0, 2), dtype=np.float64)

    def _key(contour: np.ndarray) -> tuple[float, float]:
        return (-float(np.max(contour[:, 1])), float(np.min(contour[:, 0])))

    ordered = sorted(kept, key=_key)
    parts: list[np.ndarray] = [ordered[0]]
    n_bridge = max(int(bridge_points), 0)
    for nxt in ordered[1:]:
        prev_end = parts[-1][-1]
        nxt_start = nxt[0]
        gap = nxt_start - prev_end
        if n_bridge > 0 and float(np.linalg.norm(gap)) > 1e-8:
            t = np.linspace(0.0, 1.0, n_bridge + 2, dtype=np.float64)[1:-1]
            parts.append(prev_end[None, :] + t[:, None] * gap[None, :])
        parts.append(nxt)
    return dedupe_consecutive(np.vstack(parts))


def resample_arc_length(points: np.ndarray, n: int = RESAMPLE_POINTS) -> np.ndarray:
    """Linear interpolation along cumulative arc-length to exactly n points."""
    if n < 2:
        raise ValueError(f"resample count must be >= 2, got {n}")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (M, 2) points, got {pts.shape}")
    if pts.shape[0] == 0:
        return np.zeros((n, 2), dtype=np.float64)
    if pts.shape[0] == 1:
        return np.repeat(pts[:1], n, axis=0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total < 1e-12:
        return np.repeat(pts[:1], n, axis=0)

    targets = np.linspace(0.0, total, n, dtype=np.float64)
    x = np.interp(targets, cumulative, pts[:, 0])
    y = np.interp(targets, cumulative, pts[:, 1])
    return np.stack([x, y], axis=1)


def extract_glyph_contours(font_path: str | Path, char: str, pixel_size: int = 1024) -> list[np.ndarray]:
    """Decompose one codepoint into flattened contour polylines. Empty list if missing."""
    if len(char) != 1:
        raise ValueError(f"char must be a single codepoint, got {char!r}")
    path = Path(font_path)
    face = freetype.Face(str(path))
    face.set_char_size(pixel_size * 64)
    index = face.get_char_index(char)
    if index == 0:
        return []
    face.load_char(char, freetype.FT_LOAD_NO_BITMAP | freetype.FT_LOAD_NO_HINTING)
    if face.glyph.outline.n_points < 2:
        return []
    collector = _OutlineCollector()
    face.glyph.outline.decompose(
        collector,
        move_to=collector.move_to,
        line_to=collector.line_to,
        conic_to=collector.conic_to,
        cubic_to=collector.cubic_to,
    )
    return collector.contours()


def extract_glyph_polyline(font_path: str | Path, char: str, pixel_size: int = 1024) -> np.ndarray:
    """Single unistroke polyline: the longest contour (outer writing path).

    Inner holes are dropped. Concatenating them turns O/8/B into multi-loop
    traces that no longer match one-finger airwriting.
    """
    contours = extract_glyph_contours(font_path, char, pixel_size=pixel_size)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)
    longest = max(contours, key=_contour_length)
    if _contour_length(longest) <= _MIN_CONTOUR_LENGTH:
        return np.zeros((0, 2), dtype=np.float64)
    return longest


def font_covers(font_path: str | Path, chars: Iterable[str], pixel_size: int = 256) -> dict[str, bool]:
    """Cheap coverage map: True if the glyph exists and has an outline."""
    face = freetype.Face(str(font_path))
    face.set_char_size(pixel_size * 64)
    flags = freetype.FT_LOAD_NO_BITMAP | freetype.FT_LOAD_NO_HINTING
    out: dict[str, bool] = {}
    for char in chars:
        index = face.get_char_index(char)
        if index == 0:
            out[char] = False
            continue
        face.load_char(char, flags)
        out[char] = face.glyph.outline.n_points >= 2
    return out
