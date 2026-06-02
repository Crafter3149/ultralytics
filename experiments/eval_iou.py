"""Compute AP at custom IoU thresholds by reusing ultralytics' OWN matching+AP.

We only override `iouv` (the IoU-threshold vector) on the DetectionValidator, so
the matching, NMS, and AP integration are byte-identical to the standard metric.
=> The AP@0.50 column MUST reproduce ultralytics' reported mAP50; if it does, the
   AP@0.10 column (collision-level) is trustworthy by construction.

    <venv>\\Scripts\\python.exe eval_iou.py [model.pt] [data.yaml] [imgsz]
"""

import sys

import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

MODEL = sys.argv[1] if len(sys.argv) > 1 else r"D:\Project\my\ultralytics\experiments\history\E0_baseline\weights\best.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else r"D:\Project\Bosch\2026\small_object\dataset\data.yaml"
IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 1408
SPLIT = sys.argv[4] if len(sys.argv) > 4 else "test"
THRS = [0.10, 0.30, 0.50]


class LowIoUValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.iouv = torch.tensor(THRS)  # match_predictions() does .cpu() internally
        self.niou = self.iouv.numel()


def main():
    model = YOLO(MODEL)
    m = model.val(validator=LowIoUValidator, data=DATA, split=SPLIT, imgsz=IMGSZ,
                  workers=0, plots=False, verbose=False)
    ap = m.box.all_ap  # (nc, len(THRS))
    print(f"\nultralytics {ultralytics.__version__}  ({ultralytics.__file__})")
    print(f"model: {MODEL}")
    print(f"mp(mean precision)={m.box.mp:.4f}  mr(mean recall)={m.box.mr:.4f}  (at best-F1 conf)")
    for k, t in enumerate(THRS):
        print(f"  AP@{t:.2f} = {ap[:, k].mean():.4f}")
    print("  ^ AP@0.50 must match ultralytics standard mAP50 (0.746) as a correctness check")


if __name__ == "__main__":
    main()
