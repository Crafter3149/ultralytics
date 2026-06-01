"""Evaluate a trained YOLO model on a dataset's TEST split.

Reports AP@0.5:0.95 (AP), AP@0.5, AP@0.75, precision, recall.
Run with the venv that trained the model so the matching ultralytics is used:
    <venv>\\Scripts\\python.exe eval_test.py [model.pt] [data.yaml] [imgsz]
"""

import sys
import numpy as np
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else r"D:\Project\Bosch\infer\model\task3_2026\particle_p2_v6\weights\best.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else r"D:\Project\Bosch\2026\small_object\dataset\data.yaml"
IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 960


def main():
    print(f"model : {MODEL}")
    print(f"data  : {DATA}  (split=test, imgsz={IMGSZ})")
    print("-" * 60)

    model = YOLO(MODEL)
    res = model.val(data=DATA, split="test", imgsz=IMGSZ, workers=0, plots=False, verbose=True)
    box = res.box

    ap75 = float(np.mean(box.all_ap[:, 5])) if getattr(box, "all_ap", None) is not None and box.all_ap.size else float("nan")

    print("\n================= TEST METRICS =================")
    print(f"AP@0.5:0.95 (mAP, primary) = {box.map:.4f}")
    print(f"AP@0.5      (mAP50)        = {box.map50:.4f}")
    print(f"AP@0.75     (mAP75)        = {ap75:.4f}")
    print(f"Precision (mean)           = {box.mp:.4f}")
    print(f"Recall    (mean)           = {box.mr:.4f}")
    print("================================================")


if __name__ == "__main__":
    main()
