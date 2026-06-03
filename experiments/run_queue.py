"""Run a queue of experiments sequentially, unattended (e.g. overnight).

    python experiments/run_queue.py                # default: E_NWD E_DFL E_FOCAL
    python experiments/run_queue.py E_NWD E_DFL    # custom subset / order

Each experiment runs as a SEPARATE process (clean CUDA + env per run). If one
crashes it is logged as FAILED and the queue CONTINUES to the next, so an
overnight run never dies on a single failure. A summary is printed at the end.
Tip: tee to a file ->  python experiments\\run_queue.py > experiments\\history\\queue.log 2>&1
"""

import subprocess
import sys
import time
from pathlib import Path

QUEUE = [a.upper() for a in sys.argv[1:]] or ["E_NWDTAL", "E0_RERUN", "E_NWDTAL_FOCAL"]
RUNNER = Path(__file__).resolve().parent / "run_experiment.py"
PY = sys.executable


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    print(f"[queue] {ts()}  {len(QUEUE)} experiments: {' -> '.join(QUEUE)}", flush=True)
    results = []
    for i, exp in enumerate(QUEUE, 1):
        print(f"\n{'=' * 64}\n[queue {i}/{len(QUEUE)}] {ts()}  START {exp}\n{'=' * 64}", flush=True)
        t0 = time.time()
        rc = -1
        try:
            rc = subprocess.call([PY, str(RUNNER), exp])
        except Exception as e:  # noqa: BLE001 - keep the queue alive no matter what
            print(f"[queue] {exp} raised: {e}", flush=True)
        ok = rc == 0
        dt = (time.time() - t0) / 60
        results.append((exp, ok, rc, dt))
        print(f"[queue] {ts()}  {exp} -> {'OK' if ok else f'FAILED (rc={rc})'}  ({dt:.1f} min)", flush=True)

    print(f"\n{'=' * 64}\n[queue] SUMMARY  {ts()}\n{'=' * 64}")
    for exp, ok, rc, dt in results:
        print(f"  {exp:10s} {'OK' if ok else 'FAILED':6s}  {dt:6.1f} min")
    print(r"[queue] done. per-run results under runs\small_object\<EXP>")


if __name__ == "__main__":
    main()
