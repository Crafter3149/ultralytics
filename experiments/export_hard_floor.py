"""Export the test GT objects that NO trained model detects (the ~5% hard floor).

For each such GT, save into experiments/hard_floor/:
  - NN_<stem>_gt<j>_<size>px_crop.png : zoomed context crop, GT box in red
  - NN_<stem>_gt<j>_<size>px_full.png : full original image, GT box red + green locator ring
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import ultralytics  # noqa: F401  (prints version for the record)
from ultralytics import YOLO

ROOT = Path(r"D:\Project\Bosch\2026\small_object\dataset")
OUT = Path(r"D:\Project\my\ultralytics\experiments\hard_floor")
IMGSZ, IOU_THR, SPLIT = 1408, 0.1, "test"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
H = r"D:\Project\my\ultralytics\experiments\history"
MODELS = {
    "E0": rf"{H}\E0_baseline\weights\best.pt",
    "E_NWD": rf"{H}\small_object\E_NWD\weights\best.pt",
    "E_DFL": rf"{H}\small_object\E_DFL\weights\best.pt",
    "E_FOCAL": rf"{H}\small_object\E_FOCAL\weights\best.pt",
    "E1": rf"{H}\small_object\E1\weights\best.pt",
}


def load_gt(lbl, W, Hh):
    boxes, sizes = [], []
    if lbl.exists():
        for row in lbl.read_text().splitlines():
            p = row.split()
            if len(p) >= 5:
                cx, cy, w, h = (float(x) for x in p[1:5])
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * Hh, (cx + w / 2) * W, (cy + h / 2) * Hh])
                sizes.append((w * W * h * Hh) ** 0.5)
    return (np.array(boxes) if boxes else np.zeros((0, 4))), sizes


def iou_mat(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p, g = pred[:, None, :], gt[None, :, :]
    x1 = np.maximum(p[..., 0], g[..., 0]); y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2]); y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    return inter / np.maximum(ap[:, None] + ag[None, :] - inter, 1e-9)


def found_set(model_path):
    m = YOLO(model_path)
    res = m.predict(source=str(ROOT / "images" / SPLIT), imgsz=IMGSZ, conf=0.001, iou=0.7,
                    max_det=300, verbose=False, stream=True)
    found = set()
    lbl_dir = ROOT / "labels" / SPLIT
    for r in res:
        stem = Path(r.path).stem; Hh, Ww = r.orig_shape
        gt, _ = load_gt(lbl_dir / (stem + ".txt"), Ww, Hh)
        pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        pc = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        if len(gt) == 0:
            continue
        ious = iou_mat(pb, gt); claimed = set()
        for i in np.argsort(-pc):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= IOU_THR and j not in claimed:
                claimed.add(j)
        found |= {(stem, j) for j in claimed}
    return found


def main():
    print(f"ultralytics {ultralytics.__version__}")
    img_dir, lbl_dir = ROOT / "images" / SPLIT, ROOT / "labels" / SPLIT
    gt_info = {}  # (stem, j) -> (img_path, box, size)
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        W, Hh = Image.open(img).size
        boxes, sizes = load_gt(lbl_dir / (img.stem + ".txt"), W, Hh)
        for j in range(len(boxes)):
            gt_info[(img.stem, j)] = (img, boxes[j], sizes[j])

    union = set().union(*(found_set(p) for p in MODELS.values()))
    never = sorted(set(gt_info) - union, key=lambda k: gt_info[k][2])
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"total GT={len(gt_info)}  found-by-some={len(union)}  NEVER-found={len(never)}")
    print(f"writing to {OUT}")

    for i, key in enumerate(never, 1):
        img_path, box, size = gt_info[key]
        stem, j = key
        im = Image.open(img_path).convert("RGB"); W, Hh = im.size
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = min(max(max(x2 - x1, y2 - y1) * 12, 120), 420)
        cl = (max(0, int(cx - side / 2)), max(0, int(cy - side / 2)),
              min(W, int(cx + side / 2)), min(Hh, int(cy + side / 2)))
        crop = im.crop(cl)
        sc = 480 / max(crop.size)
        crop = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))), Image.NEAREST)
        d = ImageDraw.Draw(crop)
        d.rectangle([(x1 - cl[0]) * sc, (y1 - cl[1]) * sc, (x2 - cl[0]) * sc, (y2 - cl[1]) * sc],
                    outline=(255, 0, 0), width=2)
        base = f"{i:02d}_{stem}_gt{j}_{size:.0f}px"
        crop.save(OUT / f"{base}_crop.png")

        full = im.copy(); d2 = ImageDraw.Draw(full)
        d2.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        d2.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], outline=(0, 255, 0), width=3)
        fs = 1500 / max(full.size)
        full = full.resize((int(full.width * fs), int(full.height * fs)))
        full.save(OUT / f"{base}_full.png")
        print(f"  {base}  @({cx:.0f},{cy:.0f}) in {img_path.name}")


if __name__ == "__main__":
    main()
