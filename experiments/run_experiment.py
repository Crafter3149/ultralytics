"""Run one small-object A-group experiment with a single command.

    python run_experiment.py E0       # baseline yolo26n
    python run_experiment.py E_NWD    # box loss CIoU -> NWD (C=13)
    python run_experiment.py E_DFL    # reg_max 1 -> 4 (DFL on)
    python run_experiment.py E_FOCAL  # cls BCE -> focal (a=0.25, g=2.0)

E_NWD / E_FOCAL are toggled via env vars read inside ultralytics/utils/loss.py
(so the modified loss.py must be installed). E_DFL changes reg_max in the model
config. All share the same baseline hyperparameters and dataset, so the runs are
directly comparable. Edit the MACHINE CONFIG block for the 3090 box.
"""

import os
import sys
from pathlib import Path

import yaml

# ============== MACHINE CONFIG (edit for the 3090 box) ==============
DATA = r"D:\YannWorkspace\data\dataset\data.yaml"
PROJECT = r"D:\YannWorkspace\runs\small_object"
PRETRAINED = "yolo26n.pt"          # auto-downloaded if missing; partial-loads for E_DFL
WORKERS = 8                         # set 0 if Windows multiprocessing errors

# Baseline hyperparameters (from EXPERIMENTS.md) — identical across experiments
HYP = dict(
    imgsz=1408,
    batch=2,
    epochs=200,
    scale=0.1,
    close_mosaic=20,
)
# ====================================================================

EXP = (sys.argv[1] if len(sys.argv) > 1 else "E0").upper()


def reg4_cfg() -> str:
    """Write a yolo26n config with reg_max=4 (for E_DFL); 'yolo26n' keeps scale=n."""
    import ultralytics

    base = Path(ultralytics.__file__).parent / "cfg" / "models" / "26" / "yolo26.yaml"
    d = yaml.safe_load(base.read_text(encoding="utf-8"))
    d["reg_max"] = 4
    out = Path(__file__).parent / "yolo26n-reg4.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return str(out)


def configure(exp: str) -> str:
    """Set experiment env flags and return the model config path."""
    for k in ("EXP_NWD", "EXP_FOCAL"):  # start clean
        os.environ.pop(k, None)
    cfg = "yolo26n.yaml"
    if exp == "E0":
        pass
    elif exp == "E_NWD":
        os.environ["EXP_NWD"] = "1"
        os.environ["EXP_NWD_C"] = "13.0"        # median object px @ imgsz~1408; rescale if imgsz changes
        os.environ["EXP_NWD_RATIO"] = "1.0"     # 1.0 = pure NWD; <1.0 blends CIoU
    elif exp == "E_DFL":
        cfg = reg4_cfg()
    elif exp == "E_FOCAL":
        os.environ["EXP_FOCAL"] = "1"
        os.environ["EXP_FOCAL_ALPHA"] = "0.25"
        os.environ["EXP_FOCAL_GAMMA"] = "2.0"
    else:
        raise SystemExit(f"unknown experiment {exp!r}; use E0 / E_NWD / E_DFL / E_FOCAL")
    return cfg


def main():
    cfg = configure(EXP)
    print(f"[run] experiment = {EXP}")
    print(f"[run] model cfg  = {cfg}")
    print(f"[run] flags      = NWD={os.environ.get('EXP_NWD', '0')} "
          f"FOCAL={os.environ.get('EXP_FOCAL', '0')} C={os.environ.get('EXP_NWD_C', '-')}")

    from ultralytics import YOLO

    model = YOLO(cfg)
    results = model.train(
        data=DATA,
        name=EXP,
        project=PROJECT,
        exist_ok=True,
        pretrained=PRETRAINED,   # intersect_dicts partial-loads when reg_max changes (E_DFL)
        workers=WORKERS,
        **HYP,
    )
    print(f"[done] {EXP}: best weights under {PROJECT}\\{EXP}\\weights\\best.pt")


if __name__ == "__main__":
    main()
