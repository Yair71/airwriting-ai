"""Build standalone dist/AirTouch.exe via PyInstaller (preferred) or Nuitka.

Bundles ONNX model, MediaPipe hand landmarker, dictionaries, and configs.
Default: --onefile --noconsole background daemon.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DIST = REPO_ROOT / "dist"
BUILD = REPO_ROOT / "build"
NAME = "AirTouch"
VERSION = "1.0.0"
VERSION_TUPLE = (1, 0, 0, 0)


def _sep() -> str:
    return ";" if sys.platform == "win32" else ":"


def _write_version_file() -> Path:
    """Generate a PyInstaller windows version-info resource."""
    path = BUILD / "airtouch_version_info.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    major, minor, patch, build = VERSION_TUPLE
    # PyInstaller version file format (UTF-8)
    content = textwrap.dedent(
        f"""
        # UTF-8
        VSVersionInfo(
          ffi=FixedFileInfo(
            filevers=({major}, {minor}, {patch}, {build}),
            prodvers=({major}, {minor}, {patch}, {build}),
            mask=0x3f,
            flags=0x0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0)
          ),
          kids=[
            StringFileInfo(
              [
                StringTable(
                  u'040904B0',
                  [
                    StringStruct(u'CompanyName', u'AirTouch'),
                    StringStruct(u'FileDescription', u'AirTouch dual-hand air-writing daemon'),
                    StringStruct(u'FileVersion', u'{VERSION}'),
                    StringStruct(u'InternalName', u'AirTouch'),
                    StringStruct(u'LegalCopyright', u'Copyright (c) AirTouch'),
                    StringStruct(u'OriginalFilename', u'AirTouch.exe'),
                    StringStruct(u'ProductName', u'AirTouch'),
                    StringStruct(u'ProductVersion', u'{VERSION}'),
                  ]
                )
              ]
            ),
            VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
          ]
        )
        """
    ).lstrip()
    path.write_text(content, encoding="utf-8")
    return path


def _data_args() -> list[str]:
    sep = _sep()
    pairs: list[tuple[Path, str]] = [
        (REPO_ROOT / "data" / "checkpoints" / "accurate_model.onnx", "data/checkpoints"),
        (REPO_ROOT / "data" / "models" / "hand_landmarker.task", "data/models"),
        (REPO_ROOT / "data" / "dictionaries", "data/dictionaries"),
        (REPO_ROOT / "configs", "configs"),
        (REPO_ROOT / "src", "src"),
    ]
    # Optional extras if present
    for extra in (
        REPO_ROOT / "data" / "checkpoints" / "accurate_model.pth",
    ):
        if extra.is_file():
            pairs.append((extra, "data/checkpoints"))

    args: list[str] = []
    missing: list[str] = []
    required = {
        "accurate_model.onnx",
        "hand_landmarker.task",
        "dictionaries",
        "configs",
    }
    for src, dest in pairs:
        if src.exists():
            args.extend(["--add-data", f"{src}{sep}{dest}"])
        else:
            name = src.name
            if name in required or dest.rstrip("/").split("/")[-1] in required:
                missing.append(str(src))
    if missing:
        print("WARNING: missing bundle inputs:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
    return args


def _hidden_imports() -> list[str]:
    mods = [
        "mediapipe",
        "onnxruntime",
        "pystray",
        "pynput",
        "PIL",
        "PIL.Image",
        "PIL.ImageDraw",
        "PIL.ImageFont",
        "cv2",
        "numpy",
        "src",
        "src.paths",
        "src.vision.dual_tracker",
        "src.vision.threaded_camera",
        "src.vision.mouse_controller",
        "src.vision.gesture_recognizer",
        "src.vision.stroke_collector",
        "src.vision.hand_calibrator",
        "src.vision.one_euro",
        "src.recognition.stroke_classifier",
        "src.autocompletion.trie_engine",
        "src.platform.win_injector",
        "src.platform.native_cursor",
        "src.platform.focus_detector",
        "src.platform.keyboard_layout",
        "src.ui.ghost_overlay",
        "src.ui.stop_panel",
    ]
    out: list[str] = []
    for m in mods:
        out.extend(["--hidden-import", m])
    return out


def build_pyinstaller(onefile: bool = True, noconsole: bool = True) -> int:
    import importlib.util

    if importlib.util.find_spec("PyInstaller") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0.0"])

    version_file = _write_version_file()
    DIST.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        NAME,
        "--clean",
        "--noconfirm",
        f"--distpath={DIST}",
        f"--workpath={BUILD / 'pyinstaller'}",
        f"--specpath={BUILD}",
        f"--version-file={version_file}",
    ]
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    if noconsole:
        cmd.append("--noconsole")
    else:
        cmd.append("--console")

    cmd.extend(_data_args())
    cmd.extend(_hidden_imports())
    cmd.extend(
        [
            "--collect-all",
            "mediapipe",
            "--collect-all",
            "onnxruntime",
            "--collect-submodules",
            "src",
            str(REPO_ROOT / "main.py"),
        ]
    )
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def build_nuitka() -> int:
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        "--onefile",
        "--windows-console-mode=disable",
        f"--output-filename={NAME}.exe",
        f"--output-dir={DIST}",
        "--include-package=src",
        "--include-data-dir=data/dictionaries=data/dictionaries",
        "--include-data-dir=data/models=data/models",
        "--include-data-dir=data/checkpoints=data/checkpoints",
        "--include-data-dir=configs=configs",
        f"--product-name={NAME}",
        f"--file-version={VERSION}",
        f"--product-version={VERSION}",
        "--file-description=AirTouch dual-hand air-writing daemon",
        str(REPO_ROOT / "main.py"),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile AirTouch into dist/AirTouch.exe")
    parser.add_argument("--backend", choices=("pyinstaller", "nuitka"), default="pyinstaller")
    parser.add_argument("--onedir", action="store_true")
    parser.add_argument("--console", action="store_true", help="Keep console window")
    args = parser.parse_args(argv)
    DIST.mkdir(parents=True, exist_ok=True)
    if args.backend == "nuitka":
        code = build_nuitka()
    else:
        code = build_pyinstaller(onefile=not args.onedir, noconsole=not args.console)
    exe = DIST / f"{NAME}.exe"
    if exe.is_file():
        print(f"OK: {exe}  ({exe.stat().st_size / 1e6:.1f} MB)")
    else:
        alt = DIST / NAME / f"{NAME}.exe"
        if alt.is_file():
            print(f"OK: {alt}  ({alt.stat().st_size / 1e6:.1f} MB)")
        else:
            print("Build finished but exe not found under dist/", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
