# convert_png_to_grid.py
# Usage:
#   python scripts/mazes/convert_png_to_grid.py --png assets/mazes/maze_001.png --rows 60 --cols 60 --out data/mazes/maze_001/maze_001
#
# Outputs:
#   data/mazes/maze_001/maze_001_grid.npy
#   data/mazes/maze_001/maze_001_grid.txt
#   data/mazes/maze_001/maze_001_meta.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
import numpy as np


def load_grayscale(png_path: Path) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read PNG: {png_path}")

        if img.ndim == 2:
            return img

        if img.shape[2] == 4:
            # Composite over white to avoid transparent borders turning black.
            b, g, r, a = cv2.split(img)
            alpha = a.astype(np.float32) / 255.0
            bg = np.full_like(alpha, 1.0)
            r = (r.astype(np.float32) / 255.0) * alpha + bg * (1.0 - alpha)
            g = (g.astype(np.float32) / 255.0) * alpha + bg * (1.0 - alpha)
            b = (b.astype(np.float32) / 255.0) * alpha + bg * (1.0 - alpha)
            rgb = (np.stack([b, g, r], axis=2) * 255.0).astype(np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    from PIL import Image

    img = Image.open(png_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(bg, img).convert("L")
    return np.array(composed)


def trim_to_content(gray: np.ndarray, tol: int = 5) -> tuple[np.ndarray, Optional[tuple[int, int, int, int]]]:
    h, w = gray.shape
    border = np.concatenate(
        [gray[0, :], gray[h - 1, :], gray[:, 0], gray[:, w - 1]]
    )
    bg = int(np.median(border))
    mask = np.abs(gray.astype(np.int16) - bg) > tol
    if not mask.any():
        return gray, None
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return gray[y0:y1, x0:x1], (x0, y0, x1, y1)


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return int(threshold)


def threshold_image(gray: np.ndarray, threshold: int) -> np.ndarray:
    t = otsu_threshold(gray) if threshold < 0 else threshold
    bw = np.where(gray > t, 255, 0).astype(np.uint8)
    return bw


def invert_binary(bw: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.bitwise_not(bw)
    return (255 - bw).astype(np.uint8)


def png_to_grid_patchvote(
    png_path: Path,
    rows: int,
    cols: int,
    threshold: int = 128,
    open_ratio: float = 0.55,
    wall_ratio: float = -1.0,
    invert: bool = False,
    auto_invert: bool = True,
    trim: bool = True,
    trim_tol: int = 5,
) -> tuple[np.ndarray, Optional[tuple[int, int, int, int]]]:
    """
    Convert a black/white maze PNG into a grid:
      0 = open (walkable)
      1 = wall

    This uses patch voting (counts white pixels in each cell) which is more robust
    than sampling a single pixel.
    """
    img = load_grayscale(png_path)
    trim_bbox = None
    if trim:
        img, trim_bbox = trim_to_content(img, tol=trim_tol)

    # Binary mask: white corridors = 255, black walls/background = 0
    bw = threshold_image(img, threshold)

    if auto_invert:
        white_frac = (bw == 255).mean()
        if white_frac < 0.5:
            bw = invert_binary(bw)
    if invert:
        bw = invert_binary(bw)

    h, w = bw.shape
    cell_h = h / rows
    cell_w = w / cols

    grid = np.ones((rows, cols), dtype=np.uint8)  # default wall

    for r in range(rows):
        y0 = int(r * cell_h)
        y1 = int((r + 1) * cell_h)
        for c in range(cols):
            x0 = int(c * cell_w)
            x1 = int((c + 1) * cell_w)

            patch = bw[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            white_frac = (patch == 255).mean()
            if wall_ratio >= 0:
                black_frac = 1.0 - white_frac
                grid[r, c] = 1 if black_frac >= wall_ratio else 0
            else:
                grid[r, c] = 0 if white_frac >= open_ratio else 1

    return grid, trim_bbox


def png_to_grid_resize(
    png_path: Path,
    rows: int,
    cols: int,
    threshold: int = 128,
    invert: bool = False,
    auto_invert: bool = True,
    trim: bool = True,
    trim_tol: int = 5,
) -> tuple[np.ndarray, Optional[tuple[int, int, int, int]]]:
    img = load_grayscale(png_path)
    trim_bbox = None
    if trim:
        img, trim_bbox = trim_to_content(img, tol=trim_tol)

    bw = threshold_image(img, threshold)

    if auto_invert:
        white_frac = (bw == 255).mean()
        if white_frac < 0.5:
            bw = invert_binary(bw)
    if invert:
        bw = invert_binary(bw)

    if cv2 is not None:
        resized = cv2.resize(bw, (cols, rows), interpolation=cv2.INTER_AREA)
    else:
        from PIL import Image

        resample = getattr(getattr(Image, "Resampling", Image), "BOX", None)
        if resample is None:
            resample = getattr(Image, "BOX", None)
        if resample is None:
            resample = getattr(Image, "BILINEAR", 2)
        resized = np.array(Image.fromarray(bw).resize((cols, rows), resample=resample))
    grid = np.where(resized >= 128, 0, 1).astype(np.uint8)
    return grid, trim_bbox


def grid_to_ascii(grid: np.ndarray, start: tuple[int, int] | None, goal: tuple[int, int] | None) -> str:
    """
    Render grid as ASCII:
      # = wall
      . = open
      S = start
      G = goal
    """
    rows, cols = grid.shape
    lines = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            ch = "." if grid[r, c] == 0 else "#"
            if start is not None and (r, c) == start:
                ch = "S"
            if goal is not None and (r, c) == goal:
                ch = "G"
            row_chars.append(ch)
        lines.append("".join(row_chars))
    return "\n".join(lines)


def grid_to_xo(grid: np.ndarray, start: tuple[int, int] | None, goal: tuple[int, int] | None) -> str:
    """
    Render grid as X/O:
      X = wall
      O = open
      S = start
      G = goal
    """
    rows, cols = grid.shape
    lines = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            ch = "O" if grid[r, c] == 0 else "X"
            if start is not None and (r, c) == start:
                ch = "S"
            if goal is not None and (r, c) == goal:
                ch = "G"
            row_chars.append(ch)
        lines.append("".join(row_chars))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert maze PNG to training grid (0=open, 1=wall).")
    ap.add_argument("--png", required=True, help="Path to maze PNG (black background, white corridors).")
    ap.add_argument("--rows", type=int, default=60, help="Grid rows (training resolution).")
    ap.add_argument("--cols", type=int, default=60, help="Grid cols (training resolution).")
    ap.add_argument(
        "--threshold",
        type=int,
        default=-1,
        help="0-255 threshold for binarization (-1 for Otsu).",
    )
    ap.add_argument(
        "--open_ratio",
        type=float,
        default=0.55,
        help="Fraction of white pixels needed for a cell to be considered open (0-1).",
    )
    ap.add_argument(
        "--wall_ratio",
        type=float,
        default=-1.0,
        help="If >=0, use black pixel fraction for walls (0-1). Overrides open_ratio.",
    )
    ap.add_argument(
        "--method",
        choices=["patch", "resize"],
        default="patch",
        help="Grid extraction method (patch voting is more robust; resize is faster).",
    )
    ap.add_argument(
        "--invert",
        action="store_true",
        help="Invert the binary mask after thresholding.",
    )
    ap.add_argument(
        "--no-auto-invert",
        action="store_true",
        help="Disable automatic inversion when corridors are darker than walls.",
    )
    ap.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable trimming away uniform borders.",
    )
    ap.add_argument(
        "--trim-tol",
        type=int,
        default=5,
        help="Tolerance for detecting border background when trimming.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output prefix path, e.g. data/mazes/maze_001/maze_001 (creates *_grid.npy, etc.)",
    )
    ap.add_argument(
        "--config",
        default="",
        help="Optional JSON config with overrides (rows, cols, threshold, open_ratio, method, invert, auto_invert, trim, trim_tol, start, goal).",
    )
    ap.add_argument("--start", default="", help="Optional start as r,c (e.g. 1,1).")
    ap.add_argument("--goal", default="", help="Optional goal as r,c (e.g. 58,58).")

    args = ap.parse_args()

    png_path = Path(args.png)
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.config:
        config_path = Path(args.config)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    def parse_rc(s: str) -> tuple[int, int] | None:
        s = s.strip()
        if not s:
            return None
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected r,c but got: {s}")
        return int(parts[0]), int(parts[1])

    start = parse_rc(args.start)
    goal = parse_rc(args.goal)

    auto_invert = not args.no_auto_invert
    trim = not args.no_trim

    if args.method == "resize":
        grid, trim_bbox = png_to_grid_resize(
            png_path=png_path,
            rows=args.rows,
            cols=args.cols,
            threshold=args.threshold,
            invert=args.invert,
            auto_invert=auto_invert,
            trim=trim,
            trim_tol=args.trim_tol,
        )
    else:
        grid, trim_bbox = png_to_grid_patchvote(
            png_path=png_path,
            rows=args.rows,
            cols=args.cols,
            threshold=args.threshold,
            open_ratio=args.open_ratio,
            wall_ratio=args.wall_ratio,
            invert=args.invert,
            auto_invert=auto_invert,
            trim=trim,
            trim_tol=args.trim_tol,
        )

    # Save .npy (fast training load)
    npy_path = out_prefix.with_name(out_prefix.name + "_grid.npy")
    np.save(npy_path, grid)

    # Save ASCII preview
    txt_path = out_prefix.with_name(out_prefix.name + "_grid.txt")
    txt_path.write_text(grid_to_ascii(grid, start=start, goal=goal), encoding="utf-8")

    # Save X/O preview
    xo_path = out_prefix.with_name(out_prefix.name + "_grid_xo.txt")
    xo_path.write_text(grid_to_xo(grid, start=start, goal=goal), encoding="utf-8")

    # Save metadata
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.json")
    meta = {
        "png": str(png_path).replace("\\", "/"),
        "rows": int(args.rows),
        "cols": int(args.cols),
        "threshold": int(args.threshold),
        "open_ratio": float(args.open_ratio),
        "wall_ratio": float(args.wall_ratio),
        "method": args.method,
        "invert": bool(args.invert),
        "auto_invert": bool(auto_invert),
        "trim": bool(trim),
        "trim_tol": int(args.trim_tol),
        "trim_bbox": [int(v) for v in trim_bbox] if trim_bbox else None,
        "start": list(start) if start else None,
        "goal": list(goal) if goal else None,
        "encoding": {"open": 0, "wall": 1},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {npy_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {xo_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
