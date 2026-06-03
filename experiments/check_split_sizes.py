"""Measure the real GT box-size distribution per split (val/test), in input-image px.

Answers concretely: how many boxes fall in each size bin, and do large objects
(>64px, >128px) actually exist in val/test? No assumptions about image size — reads
each image's real dimensions.

    .venv\\Scripts\\python.exe experiments\\check_split_sizes.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"D:\Project\Bosch\2026\small_object\dataset")
SPLITS = ["val", "test", "train"]
BINS = [0, 8, 16, 32, 64, 128, 1e9]
BIN_LABELS = ["<8", "8-16", "16-32", "32-64", "64-128", ">128"]


def split_sizes(split: str):
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split
    sizes = []
    n_imgs = 0
    n_bg = 0
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        n_imgs += 1
        W, H = Image.open(img).size
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists() or not lbl.read_text().strip():
            n_bg += 1
            continue
        for row in lbl.read_text().splitlines():
            p = row.split()
            if len(p) >= 5:
                w_px = float(p[3]) * W
                h_px = float(p[4]) * H
                sizes.append((w_px * h_px) ** 0.5)
    return np.array(sizes), n_imgs, n_bg


def main():
    for split in SPLITS:
        if not (ROOT / "images" / split).exists():
            print(f"[{split}] MISSING")
            continue
        s, n_imgs, n_bg = split_sizes(split)
        print(f"\n===== {split}: {n_imgs} images, {n_bg} background ({100*n_bg/max(n_imgs,1):.0f}%), {len(s)} boxes =====")
        if len(s) == 0:
            continue
        print(f"  sqrt(w*h) px: median={np.median(s):.1f}  mean={np.mean(s):.1f}  "
              f"min={s.min():.1f}  max={s.max():.1f}  p90={np.percentile(s,90):.1f}  p99={np.percentile(s,99):.1f}")
        hist, _ = np.histogram(s, bins=BINS)
        for lab, c in zip(BIN_LABELS, hist):
            print(f"    {lab:>7} px : {c:4d}  ({100*c/len(s):4.1f}%)")
        n_large = int((s > 64).sum())
        n_xlarge = int((s > 128).sum())
        print(f"  >64px: {n_large} boxes ({100*n_large/len(s):.1f}%) | >128px: {n_xlarge} boxes ({100*n_xlarge/len(s):.1f}%)")


if __name__ == "__main__":
    main()
