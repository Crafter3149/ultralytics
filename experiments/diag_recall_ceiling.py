"""Diagnose E1's low recall: detection ceiling (H2) vs confidence suppression (H1).

At conf=0.001 (every plausible box survives), greedy one-to-one match preds to GT
at IoU>=0.1. Reports per split:
  - max_recall : fraction of GT with ANY matching pred = the detection ceiling
  - missed     : GT with no pred at IoU>=0.1 even at conf 0.001
  - tp_conf    : confidence distribution of the matched (TP) preds

Reading:
  E1 max_recall ~= E0 but lower tp_conf  => confidence suppression (cls imbalance, H1):
      the particles ARE proposed, just scored low -> focal should help.
  E1 max_recall <  E0                    => detection/feature failure (H2):
      the P2 head / reinit lost the ability to propose boxes -> focal won't fix it.

    <gpu-venv-python> experiments\\diag_recall_ceiling.py   (with PYTHONPATH=fork)
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
}


def load_gt(lbl: Path, W: int, H_: int) -> np.ndarray:
    if not lbl.exists():
        return np.zeros((0, 4))
    out = []
    for row in lbl.read_text().splitlines():
        p = row.split()
        if len(p) >= 5:
            cx, cy, w, h = (float(x) for x in p[1:5])
            out.append([(cx - w / 2) * W, (cy - h / 2) * H_, (cx + w / 2) * W, (cy + h / 2) * H_])
    return np.array(out) if out else np.zeros((0, 4))


def iou_mat(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p, g = pred[:, None, :], gt[None, :, :]
    x1 = np.maximum(p[..., 0], g[..., 0]); y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2]); y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    return inter / np.maximum(ap[:, None] + ag[None, :] - inter, 1e-9)


def run(model_path: str, split: str) -> None:
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split
    m = YOLO(model_path)
    res = m.predict(source=str(img_dir), imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, verbose=False, stream=True)
    n_gt = n_tp = 0
    tp_conf = []
    for r in res:
        img = Path(r.path); Hh, Ww = r.orig_shape
        gt = load_gt(lbl_dir / (img.stem + ".txt"), Ww, Hh)
        pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        pc = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        n_gt += len(gt)
        if len(gt) == 0:
            continue
        ious = iou_mat(pb, gt)
        claimed = set()
        for i in np.argsort(-pc):
            if ious.shape[1] == 0:
                break
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= IOU_THR and j not in claimed:
                claimed.add(j); n_tp += 1; tp_conf.append(float(pc[i]))
    tp_conf = np.array(tp_conf) if tp_conf else np.zeros(0)
    print(f"  {split:4s}: GT={n_gt:4d}  max_recall={n_tp / max(n_gt, 1):.3f}  missed={n_gt - n_tp:3d}  "
          f"| TP conf mean={tp_conf.mean():.3f} median={np.median(tp_conf):.3f} "
          f"<0.25={int((tp_conf < 0.25).sum())}/{len(tp_conf)}")


def main():
    print(f"ultralytics {ultralytics.__version__}  (IoU>={IOU_THR}, conf=0.001, max_det=300)")
    for name, p in MODELS.items():
        print(name)
        for split in ("val", "test"):
            run(p, split)


if __name__ == "__main__":
    main()
