"""Size-stratified detection check: did E1's finer P2 anchor grid (stride 4) help
the SMALL objects it was meant to help?

Per GT size bin (sqrt(w*h) px), at conf=0.001 (detection ceiling), report:
  - n     : GT count in the bin
  - ceil  : max-recall = fraction of those GT found (one-to-one, IoU>=0.1)
  - conf  : mean confidence of the matched (TP) detections in the bin

If E1's P2 anchors paid off, E1 ceil should beat E0 in the small bins (<8, 8-16).
"""

from pathlib import Path

import numpy as np

import ultralytics
from ultralytics import YOLO

ROOT = Path(r"D:\Project\Bosch\2026\small_object\dataset")
IMGSZ = 1408
IOU_THR = 0.1
BINS = [0, 8, 16, 32, 64, 1e9]
LABELS = ["<8", "8-16", "16-32", "32-64", ">64"]
H = r"D:\Project\my\ultralytics\experiments\history"
MODELS = {"E0": rf"{H}\E0_baseline\weights\best.pt", "E1": rf"{H}\small_object\E1\weights\best.pt"}


def load_gt(lbl, W, Hh):
    if not lbl.exists():
        return np.zeros((0, 4)), np.zeros(0)
    boxes, sizes = [], []
    for row in lbl.read_text().splitlines():
        p = row.split()
        if len(p) >= 5:
            cx, cy, w, h = (float(x) for x in p[1:5])
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * Hh, (cx + w / 2) * W, (cy + h / 2) * Hh])
            sizes.append((w * W * h * Hh) ** 0.5)
    return (np.array(boxes), np.array(sizes)) if boxes else (np.zeros((0, 4)), np.zeros(0))


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


def collect(model_path, splits):
    """Return per-GT (size, found, tp_conf) across the given splits."""
    m = YOLO(model_path)
    sizes, found, confs = [], [], []
    for split in splits:
        img_dir = ROOT / "images" / split
        lbl_dir = ROOT / "labels" / split
        res = m.predict(source=str(img_dir), imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, verbose=False, stream=True)
        for r in res:
            img = Path(r.path); Hh, Ww = r.orig_shape
            gt, gsz = load_gt(lbl_dir / (img.stem + ".txt"), Ww, Hh)
            pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            pc = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
            if len(gt) == 0:
                continue
            ious = iou_mat(pb, gt)
            matched_conf = {}
            claimed = set()
            for i in np.argsort(-pc):
                if ious.shape[1] == 0:
                    break
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= IOU_THR and j not in claimed:
                    claimed.add(j); matched_conf[j] = float(pc[i])
            for j in range(len(gt)):
                sizes.append(gsz[j]); found.append(j in claimed); confs.append(matched_conf.get(j, np.nan))
    return np.array(sizes), np.array(found), np.array(confs)


def main():
    print(f"ultralytics {ultralytics.__version__}  (val+test combined, IoU>={IOU_THR}, conf=0.001)")
    data = {name: collect(p, ["val", "test"]) for name, p in MODELS.items()}
    bin_idx_e0 = np.digitize(data["E0"][0], BINS) - 1
    header = f"{'bin':>7} | {'n':>4} | {'E0 ceil':>8} {'E1 ceil':>8}  Δ      | {'E0 conf':>8} {'E1 conf':>8}"
    print(header); print("-" * len(header))
    for b, lab in enumerate(LABELS):
        e0s, e0f, e0c = data["E0"]; e1s, e1f, e1c = data["E1"]
        m0 = (np.digitize(e0s, BINS) - 1) == b
        m1 = (np.digitize(e1s, BINS) - 1) == b
        n = int(m0.sum())
        if n == 0:
            continue
        c0, c1 = e0f[m0].mean(), e1f[m1].mean()
        cf0 = np.nanmean(e0c[m0]) if np.isfinite(e0c[m0]).any() else float("nan")
        cf1 = np.nanmean(e1c[m1]) if np.isfinite(e1c[m1]).any() else float("nan")
        print(f"{lab:>7} | {n:>4} | {c0:8.3f} {c1:8.3f}  {c1-c0:+.3f} | {cf0:8.3f} {cf1:8.3f}")


if __name__ == "__main__":
    main()
