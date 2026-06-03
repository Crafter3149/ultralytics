"""Probe the two suspicious 'never-found' images (2_5 and 10_1, 28-36px objects):
is there REALLY no predicted box near the GT, or did my IoU>=0.1 cut / max_det=300
cap hide it? Raise max_det to 1000 and list the preds nearest the GT.
"""

from pathlib import Path

import numpy as np

import ultralytics  # noqa: F401
from ultralytics import YOLO

ROOT = Path(r"D:\Project\Bosch\2026\small_object\dataset")
IMGSZ = 1408
IMAGES = ["5d935064-2_5_4", "7a3d413b-10_1_3"]
MODELS = {
    "E0": r"D:\Project\my\ultralytics\experiments\history\E0_baseline\weights\best.pt",
    "E_FOCAL": r"D:\Project\my\ultralytics\experiments\history\small_object\E_FOCAL\weights\best.pt",
}


def load_gt(lbl, W, Hh):
    out = []
    for row in lbl.read_text().splitlines():
        p = row.split()
        if len(p) >= 5:
            cx, cy, w, h = (float(x) for x in p[1:5])
            out.append([(cx - w / 2) * W, (cy - h / 2) * Hh, (cx + w / 2) * W, (cy + h / 2) * Hh])
    return np.array(out) if out else np.zeros((0, 4))


def iou1(b, g):
    x1 = max(b[0], g[0]); y1 = max(b[1], g[1]); x2 = min(b[2], g[2]); y2 = min(b[3], g[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ab = (b[2] - b[0]) * (b[3] - b[1]); ag = (g[2] - g[0]) * (g[3] - g[1])
    return inter / max(ab + ag - inter, 1e-9)


def main():
    print(f"ultralytics {ultralytics.__version__}  (max_det=1000, conf=0.001)")
    for stem in IMAGES:
        img = next(p for p in (ROOT / "images" / "test").glob(stem + ".*"))
        from PIL import Image
        im = Image.open(img)
        W, Hh = im.size
        lum = np.asarray(im.convert("L"))
        gts = load_gt(ROOT / "labels" / "test" / (stem + ".txt"), W, Hh)
        print(f"\n===== {img.name}  ({W}x{Hh}px)  mode={im.mode}  brightness mean={lum.mean():.1f} "
              f"min={lum.min()} max={lum.max()}  GT count={len(gts)} =====")
        for j, g in enumerate(gts):
            print(f"  GT{j}: xyxy=({g[0]:.0f},{g[1]:.0f},{g[2]:.0f},{g[3]:.0f})  "
                  f"wh=({g[2]-g[0]:.0f}x{g[3]-g[1]:.0f})  sqrt={((g[2]-g[0])*(g[3]-g[1]))**0.5:.1f}px")
        for name, mp in MODELS.items():
            m = YOLO(mp)
            r = m.predict(source=str(img), imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=1000, verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy(); pc = r.boxes.conf.cpu().numpy()
            if len(pb) == 0:
                print(f"  [{name}] total preds=0  *** MODEL OUTPUT NOTHING ON THIS IMAGE ***")
                continue
            print(f"  [{name}] total preds={len(pb)}  conf range [{pc.min():.4f}, {pc.max():.4f}]")
            for j, g in enumerate(gts):
                gcx, gcy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
                ious = np.array([iou1(b, g) for b in pb])
                dist = np.sqrt((((pb[:, 0] + pb[:, 2]) / 2 - gcx)) ** 2 + (((pb[:, 1] + pb[:, 3]) / 2 - gcy)) ** 2)
                # best by IoU
                k = int(np.argmax(ious)) if len(ious) else -1
                # nearest by center distance
                n = int(np.argmin(dist)) if len(dist) else -1
                if k >= 0:
                    bb = pb[k]
                    print(f"    GT{j}: best-IoU pred IoU={ious[k]:.3f} conf={pc[k]:.4f} "
                          f"xyxy=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f})")
                    nb = pb[n]
                    print(f"          nearest-center pred dist={dist[n]:.0f}px IoU={ious[n]:.3f} conf={pc[n]:.4f} "
                          f"xyxy=({nb[0]:.0f},{nb[1]:.0f},{nb[2]:.0f},{nb[3]:.0f})  (#preds within 50px={int((dist<50).sum())})")


if __name__ == "__main__":
    main()
