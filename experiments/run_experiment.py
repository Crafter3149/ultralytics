"""Run one small-object experiment with a single command.

    A-group (loss/channel ablations on the baseline head):
    python run_experiment.py E0       # baseline yolo26n (Detect P3/P4/P5)
    python run_experiment.py E_NWD    # box loss CIoU -> NWD (C=13)
    python run_experiment.py E_DFL    # reg_max 1 -> 4 (DFL on)
    python run_experiment.py E_FOCAL  # cls BCE -> focal (a=0.25, g=2.0)

    B-group (Track B head architecture; A-group all rejected so E0+ = E0, no loss flags):
    python run_experiment.py E1       # +P2 head -> Detect(P2,P3,P4,P5)  [yolo26-p2.yaml]
    python run_experiment.py E2       # P1/P2/P3 head, no P4/P5          [yolo26-p1.yaml]
    python run_experiment.py E2 1     # optional 2nd arg = batch override (see VRAM note)
    python run_experiment.py E1_FOCAL # E1's P2 head + focal cls (2nd-round, for the cls imbalance E1 showed)
    python run_experiment.py E1_NWD       # E1 head + box NWD
    python run_experiment.py E1_FOCAL_NWD # E1 head + focal + NWD
    python run_experiment.py E0_RERUN     # E0 baseline, different seed (probe E0 training variance)

E_NWD / E_FOCAL are toggled via env vars read inside ultralytics/utils/loss.py
(so the modified loss.py must be installed). E_DFL / E1 / E2 change the model
config; their new head layers partial-load (backbone loads 1:1 from yolo26n.pt,
head is retrained via intersect_dicts). All share the same baseline hyperparameters
and dataset, so runs are directly comparable. Edit MACHINE CONFIG for the 3090 box.

VRAM: P2 (E1) and especially P1 (E2) add high-res feature maps at imgsz=1408
(P2 stride4 = 352x352 grid, P1 stride2 = 704x704 grid), so activation memory and
the per-anchor loss blow up vs baseline. Baseline batch=2 already sits at the 24GB
limit. If E1/E2 OOM, lower batch via the 2nd arg (e.g. `run_experiment.py E2 1`).
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
    elif exp == "E1":
        cfg = "yolo26n-p2.yaml"   # Track B: add P2 head -> Detect(P2,P3,P4,P5)
    elif exp == "E2":
        cfg = "yolo26n-p1.yaml"   # Track B: P1/P2/P3 head, no P4/P5 -> Detect(P1,P2,P3)
    elif exp == "E1_FOCAL":
        cfg = "yolo26n-p2.yaml"   # 2nd-round: E1's P2 head + focal cls (rescue the cls imbalance E1 showed)
        os.environ["EXP_FOCAL"] = "1"
        os.environ["EXP_FOCAL_ALPHA"] = "0.25"
        os.environ["EXP_FOCAL_GAMMA"] = "2.0"
    elif exp == "E1_NWD":
        cfg = "yolo26n-p2.yaml"   # E1 head + box NWD
        os.environ["EXP_NWD"] = "1"
        os.environ["EXP_NWD_C"] = "13.0"
        os.environ["EXP_NWD_RATIO"] = "1.0"
    elif exp == "E1_FOCAL_NWD":
        cfg = "yolo26n-p2.yaml"   # E1 head + focal cls + box NWD (both A-group losses on the P2 head)
        os.environ["EXP_FOCAL"] = "1"
        os.environ["EXP_FOCAL_ALPHA"] = "0.25"
        os.environ["EXP_FOCAL_GAMMA"] = "2.0"
        os.environ["EXP_NWD"] = "1"
        os.environ["EXP_NWD_C"] = "13.0"
        os.environ["EXP_NWD_RATIO"] = "1.0"
    elif exp == "E0_RERUN":
        pass                      # baseline config; main() sets a different seed to probe E0 training variance
    else:
        raise SystemExit(f"unknown experiment {exp!r}; use E0 / E_NWD / E_DFL / E_FOCAL / E1 / E2 / E1_FOCAL / E1_NWD / E1_FOCAL_NWD / E0_RERUN")
    return cfg


def main():
    cfg = configure(EXP)
    hyp = dict(HYP)
    if len(sys.argv) > 2:        # optional batch override: `run_experiment.py E2 1` (P1/P2 may OOM at batch=2)
        hyp["batch"] = int(sys.argv[2])
    if EXP == "E0_RERUN":
        hyp["seed"] = 1          # != default seed 0: probe whether E0's win is training-seed luck
    print(f"[run] experiment = {EXP}")
    print(f"[run] model cfg  = {cfg}")
    print(f"[run] batch      = {hyp['batch']}  imgsz={hyp['imgsz']}  epochs={hyp['epochs']}")
    print(f"[run] flags      = NWD={os.environ.get('EXP_NWD', '0')} "
          f"FOCAL={os.environ.get('EXP_FOCAL', '0')} C={os.environ.get('EXP_NWD_C', '-')}")

    from ultralytics import YOLO

    model = YOLO(cfg)
    results = model.train(
        data=DATA,
        name=EXP,
        project=PROJECT,
        exist_ok=True,
        pretrained=PRETRAINED,   # intersect_dicts partial-loads when the head changes (E_DFL / E1 / E2)
        workers=WORKERS,
        **hyp,
    )
    print(f"[done] {EXP}: best weights under {PROJECT}\\{EXP}\\weights\\best.pt")


if __name__ == "__main__":
    main()
