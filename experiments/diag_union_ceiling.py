"""Cross-model UNION detection ceiling: is the ~7% test floor breakable, and does
E1's P2 architecture find objects the P3-architecture models (E0/E_NWD/E_DFL/E_FOCAL)
cannot?

Per GT (keyed by image+index), record which models detect it (conf=0.001, IoU>=0.1).
Then report individual recall, the UNION (found by >=1 model), whether E1 adds unique
finds on top of the P3 models, and the irreducible floor (GT found by nobody).
"""

from pathlib import Path

import numpy as np

import ultralytics
from ultralytics import YOLO

ROOT = Path(r"D:\Project\Bosch\2026\small_object\dataset")
IMGSZ = 1408
IOU_THR = 0.1
H = r"D:\Project\my\ultralytics\experiments\history"
MODELS = {
    "E0": rf"{H}\E0_baseline\weights\best.pt",
    "E_NWD": rf"{H}\small_object\E_NWD\weights\best.pt",
    "E_DFL": rf"{H}\small_object\E_DFL\weights\best.pt",
    "E_FOCAL": rf"{H}\small_object\E_FOCAL\weights\best.pt",
    "E1": rf"{H}\small_object\E1\weights\best.pt",
}
P3_MODELS = ["E0", "E_NWD", "E_DFL", "E_FOCAL"]
SPLIT = "test"


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
    """Return set of (img_stem, gt_idx) this model detects on SPLIT."""
    m = YOLO(model_path)
    img_dir = ROOT / "images" / SPLIT
    lbl_dir = ROOT / "labels" / SPLIT
    res = m.predict(source=str(img_dir), imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=1000, verbose=False, stream=True)
    found = set()
    for r in res:
        img = Path(r.path); Hh, Ww = r.orig_shape
        gt, _ = load_gt(lbl_dir / (img.stem + ".txt"), Ww, Hh)
        pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        pc = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        if len(gt) == 0:
            continue
        ious = iou_mat(pb, gt); claimed = set()
        for i in np.argsort(-pc):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= IOU_THR and j not in claimed:
                claimed.add(j)
        for j in claimed:
            found.add((img.stem, j))
    return found


def all_gt_keys():
    keys, sizes = {}, {}
    lbl_dir = ROOT / "labels" / SPLIT
    img_dir = ROOT / "images" / SPLIT
    from PIL import Image
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        W, Hh = Image.open(img).size
        _, sz = load_gt(lbl_dir / (img.stem + ".txt"), W, Hh)
        for j, s in enumerate(sz):
            keys[(img.stem, j)] = True; sizes[(img.stem, j)] = s
    return set(keys), sizes


def main():
    print(f"ultralytics {ultralytics.__version__}  split={SPLIT}  (conf=0.001, IoU>={IOU_THR})")
    all_gt, sizes = all_gt_keys()
    N = len(all_gt)
    fs = {name: found_set(p) for name, p in MODELS.items()}
    print(f"total GT = {N}\n--- individual ceilings ---")
    for name in MODELS:
        print(f"  {name:8s}: recall={len(fs[name]) / N:.3f}  (found {len(fs[name])})")

    p3_union = set().union(*(fs[n] for n in P3_MODELS))
    all_union = set().union(*fs.values())
    print("--- unions ---")
    print(f"  P3 models (E0/E_NWD/E_DFL/E_FOCAL) union : {len(p3_union) / N:.3f}  (found {len(p3_union)})")
    print(f"  + E1 (P2 arch) added                     : {len(all_union) / N:.3f}  (found {len(all_union)})")
    print(f"  => E1 unique finds beyond P3 union       : {len(all_union - p3_union)}")
    print(f"  E1 finds among E_FOCAL's misses          : {len(fs['E1'] - fs['E_FOCAL'])} "
          f"(of {N - len(fs['E_FOCAL'])} E_FOCAL misses)")

    never = all_gt - all_union
    print(f"--- irreducible floor (found by NO model) : {len(never)}  ({len(never) / N:.3f}) ---")
    if never:
        nsz = sorted(round(sizes[k], 1) for k in never)
        print(f"  their sizes (px): {nsz}")


if __name__ == "__main__":
    main()
