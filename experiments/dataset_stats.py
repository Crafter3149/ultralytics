"""Dataset pre-flight stats for the small-object YOLO26 experiments.

Computes, per split (train/val/test):
  - image / background(no-object) / with-object counts  -> positive/negative ratio
  - total boxes and class distribution                  -> validates nc / names
  - box size sqrt(w*h) in pixels                         -> suggested NWD constant C

Run:  python stats.py
Deps: pyyaml + pillow (both ship with ultralytics).
"""

from pathlib import Path
import math
import sys
import yaml
from PIL import Image

# data.yaml path: pass as first CLI arg, else use this default (edit for your machine)
DATA_YAML = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\YannWorkspace\data\dataset\data.yaml")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
root = Path(cfg.get("path", DATA_YAML.parent))
if not root.is_absolute():
    root = (DATA_YAML.parent / root).resolve()

print("data.yaml:", DATA_YAML)
print("root     :", root)
print("nc =", cfg.get("nc"), "| names =", cfg.get("names"))
print("=" * 64)

global_sqrt = []
class_count = {}

for split in ("train", "val", "test"):
    rel = cfg.get(split)
    if not rel:
        continue
    img_dir = (root / rel).resolve()
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    if not img_dir.exists():
        print(f"[{split}] MISSING image dir: {img_dir}")
        continue

    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    n_bg = n_pos = n_boxes = 0
    s_list = []
    sizes_seen = set()

    for img in imgs:
        lbl = lbl_dir / (img.stem + ".txt")
        rows = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
        if not rows:
            n_bg += 1
            continue
        n_pos += 1
        try:
            W, H = Image.open(img).size
            sizes_seen.add((W, H))
        except Exception:
            W = H = None
        for r in rows:
            n_boxes += 1
            if len(r) >= 5:
                class_count[r[0]] = class_count.get(r[0], 0) + 1
                if W:
                    w_px, h_px = float(r[3]) * W, float(r[4]) * H
                    s_list.append(math.sqrt(w_px * h_px))

    global_sqrt += s_list
    print(f"[{split}] images={len(imgs)}  with_obj={n_pos}  background={n_bg}  boxes={n_boxes}")
    if n_pos + n_bg:
        print(f"        background ratio = {n_bg / (n_pos + n_bg):.1%}")
    print(f"        image sizes seen: {sorted(sizes_seen)[:5]}{' ...' if len(sizes_seen) > 5 else ''}")
    if s_list:
        print(
            f"        box sqrt(w*h) px: mean={sum(s_list) / len(s_list):.2f}  "
            f"median={pct(s_list, 0.5):.2f}  p10={pct(s_list, 0.1):.2f}  "
            f"p90={pct(s_list, 0.9):.2f}  min={min(s_list):.2f}  max={max(s_list):.2f}"
        )

print("=" * 64)
print("class distribution (class_id: count):", class_count)
if global_sqrt:
    C = sum(global_sqrt) / len(global_sqrt)
    print(f"GLOBAL box sqrt(w*h) px: mean={C:.3f}  median={pct(global_sqrt, 0.5):.3f}  (n={len(global_sqrt)})")
    print(f">>> Suggested NWD C ~= {C:.1f}")
else:
    print("No boxes parsed -- check label format / paths.")
