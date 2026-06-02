"""Run eval_iou-style AP at multiple IoU thresholds for ALL A-group models on one split.

Reuses ultralytics' own matching/NMS/AP (only overrides iouv), so AP@0.50 must
reproduce the standard mAP50 as a correctness check. Prints one combined table.

    .venv\\Scripts\\python.exe experiments\\eval_grid.py [split]
"""

import sys

import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

DATA = r"D:\Project\Bosch\2026\small_object\dataset\data.yaml"
IMGSZ = 1408
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
THRS = [0.10, 0.30, 0.50, 0.75]

H = r"D:\Project\my\ultralytics\experiments\history"
MODELS = {
    "E0":      rf"{H}\E0_baseline\weights\best.pt",
    "E_NWD":   rf"{H}\small_object\E_NWD\weights\best.pt",
    "E_DFL":   rf"{H}\small_object\E_DFL\weights\best.pt",
    "E_FOCAL": rf"{H}\small_object\E_FOCAL\weights\best.pt",
}


class MultiThrValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.iouv = torch.tensor(THRS)
        self.niou = self.iouv.numel()


def main():
    print(f"ultralytics {ultralytics.__version__}  ({ultralytics.__file__})")
    print(f"split={SPLIT}  imgsz={IMGSZ}  thresholds={THRS}")
    header = f"{'model':10s} | {'mP':>6s} {'mR':>6s} | " + " ".join(f"AP@{t:.2f}" for t in THRS)
    print(header)
    print("-" * len(header))
    for name, w in MODELS.items():
        m = YOLO(w).val(validator=MultiThrValidator, data=DATA, split=SPLIT, imgsz=IMGSZ,
                        workers=0, plots=False, verbose=False)
        ap = m.box.all_ap.mean(0)  # mean over classes -> (niou,)
        row = f"{name:10s} | {m.box.mp:6.3f} {m.box.mr:6.3f} | " + " ".join(f"{ap[k]:6.3f}" for k in range(len(THRS)))
        print(row, flush=True)
    print("\n^ AP@0.50 should match each run's standard val mAP50 (correctness check)")


if __name__ == "__main__":
    main()
