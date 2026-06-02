"""Low-IoU ("collision") detection eval for tiny objects.

Matching rule: a prediction is a TP if its IoU with an unclaimed GT >= IOU_THR
(one-to-one, greedy by confidence). Box tightness beyond that is NOT scored.
Reports, per IoU_THR: AP (area under P-R), and the best-F1 operating point
(conf, precision, recall, F1).

    <venv>\\Scripts\\python.exe eval_lowiou.py [model.pt] [data.yaml] [imgsz]
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import yaml
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else r"D:\Project\my\ultralytics\experiments\history\E0_baseline\weights\best.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else r"D:\Project\Bosch\2026\small_object\dataset\data.yaml"
IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 1408
IOU_THRS = [0.5, 0.1]  # 0.5 = standard, 0.1 = collision-level
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iou_matrix(pred, gt):
    """pred (N,4), gt (M,4) xyxy -> (N,M) IoU."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p = pred[:, None, :]
    g = gt[None, :, :]
    x1 = np.maximum(p[..., 0], g[..., 0]); y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2]); y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = ap[:, None] + ag[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def load_gt(lbl_path, W, H):
    if not lbl_path.exists():
        return np.zeros((0, 4))
    rows = [r.split() for r in lbl_path.read_text().splitlines() if r.strip()]
    out = []
    for r in rows:
        if len(r) >= 5:
            cx, cy, w, h = (float(x) for x in r[1:5])
            out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(out) if out else np.zeros((0, 4))


def compute_ap(confs, tps, n_gt):
    if len(confs) == 0 or n_gt == 0:
        return 0.0
    order = np.argsort(-confs)
    tps = np.asarray(tps)[order]
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(1 - tps)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    p_env = np.maximum.accumulate(precision[::-1])[::-1]  # monotonic from right
    levels = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, levels, side="left")
    ap = np.mean([p_env[i] if i < len(p_env) else 0.0 for i in idx])
    return ap, precision, recall


def best_f1(confs, tps, n_gt):
    order = np.argsort(-confs)
    c = np.asarray(confs)[order]
    tps = np.asarray(tps)[order]
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(1 - tps)
    rec = tp_cum / max(n_gt, 1)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(np.argmax(f1))
    return c[k], prec[k], rec[k], f1[k]


def main():
    cfg = yaml.safe_load(Path(DATA).read_text(encoding="utf-8"))
    root = Path(cfg.get("path", Path(DATA).parent))
    img_dir = (root / cfg["test"]).resolve()
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    print(f"model: {MODEL}")
    print(f"test : {img_dir}  ({len(imgs)} images, imgsz={IMGSZ})")

    model = YOLO(MODEL)
    results = model.predict(source=str(img_dir), imgsz=IMGSZ, conf=0.001,
                            iou=0.7, max_det=300, verbose=False, stream=True)

    n_gt = 0
    per_thr = {t: [] for t in IOU_THRS}  # list of (conf, tp)
    for r in results:
        img = Path(r.path)
        H, W = r.orig_shape
        gt = load_gt(lbl_dir / (img.stem + ".txt"), W, H)
        pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        pc = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        n_gt += len(gt)
        ious = iou_matrix(pb, gt)
        order = np.argsort(-pc)
        for thr in IOU_THRS:
            claimed = set()
            for i in order:
                if len(gt):
                    j = int(np.argmax(ious[i]))
                    if ious[i, j] >= thr and j not in claimed:
                        per_thr[thr].append((pc[i], 1)); claimed.add(j); continue
                per_thr[thr].append((pc[i], 0))

    print(f"total GT = {n_gt}\n" + "=" * 56)
    for thr in IOU_THRS:
        arr = np.array(per_thr[thr]) if per_thr[thr] else np.zeros((0, 2))
        confs, tps = (arr[:, 0], arr[:, 1]) if len(arr) else (np.zeros(0), np.zeros(0))
        ap, _, _ = compute_ap(confs, tps, n_gt)
        c, p, rcl, f1 = best_f1(confs, tps, n_gt)
        tag = "collision" if thr <= 0.1 else "standard "
        print(f"IoU>={thr:<4} ({tag}) | AP={ap:.4f} | best-F1: P={p:.3f} R={rcl:.3f} F1={f1:.3f} @conf={c:.3f}")


if __name__ == "__main__":
    main()
