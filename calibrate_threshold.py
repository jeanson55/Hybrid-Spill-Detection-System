"""
Offline Threshold Calibrator Training Script
=============================================

Fits the learned-threshold calibrator (PINNValidator item #2) from
labelled residual data and saves it to disk so the live pipeline can
load it via `threshold_calibrator_path=`.

Where the labels come from
---------------------------
`residual` and `uncertainty` for each historical detection are already
logged for you in `spill_captures/capture_log.json` (written by
SpillImageSaver) for every CONFIRMED alert, and can also be pulled from
your own logging of PENDING/REJECTED detections if you log those too.
`label` is NOT automatic — it requires a human to review each event and
mark it:
    1 = confirmed real spill (true positive)
    0 = confirmed false positive (e.g. reflection, shadow, wet floor
        that never actually spread)

Expected input CSV columns: residual,uncertainty,label
    residual     - PINNResult.residual (float)
    uncertainty  - PINNResult.uncertainty, MC-Dropout std (float; use 0
                   if you only logged single-pass residuals)
    label        - 1 or 0, from manual review

Usage
-----
    python scripts/calibrate_threshold.py \
        --csv labelled_events.csv \
        --pinn-weights spill_best_pinn.pt \
        --out threshold_calibrator.pt

If you don't have a labelled CSV yet, run with --demo to see the script
work end-to-end on synthetic data first.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.pinn_validator import PINNValidator


def load_csv(path: str):
    residuals, uncertainties, labels = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"residual", "uncertainty", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must have columns {required}, got {reader.fieldnames}"
            )
        for row in reader:
            residuals.append(float(row["residual"]))
            uncertainties.append(float(row["uncertainty"]))
            labels.append(int(row["label"]))
    return residuals, uncertainties, labels


def make_demo_data(pinn: PINNValidator, n_true=25, n_false=25):
    """
    Generates synthetic labelled examples by running MC-dropout residual
    checks over plausible-looking boxes (label 1) and clearly-invalid
    boxes/timings (label 0), purely so you can see the calibration flow
    run end-to-end before you have real field labels.
    """
    residuals, uncertainties, labels = [], [], []
    for i in range(n_true):
        bbox = (100 + i, 100, 260 + i, 220)
        r = pinn.validate(bbox, frame_time=1.0 + i * 0.05, fluid_class="light_hydrocarbon")
        residuals.append(r.residual)
        uncertainties.append(r.uncertainty)
        labels.append(1)
    for i in range(n_false):
        bbox = (5 + i * 3, 5, 20 + i * 3, 20)  # tiny, jittery boxes
        r = pinn.validate(bbox, frame_time=100.0 + i * 3.0, fluid_class="unknown")
        residuals.append(r.residual)
        uncertainties.append(r.uncertainty)
        labels.append(0)
    return residuals, uncertainties, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=str, default=None, help="Labelled residual/uncertainty/label CSV")
    ap.add_argument("--pinn-weights", type=str, default="pinn_adapted.pt",
                     help="Path to trained ThinFilmPINN weights (default: pinn_adapted.pt)")
    ap.add_argument("--out", type=str, default="threshold_calibrator.pt", help="Output calibrator path")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--demo", action="store_true", help="Use synthetic demo data instead of --csv")
    args = ap.parse_args()

    pinn = PINNValidator(model_path=args.pinn_weights, device=args.device, use_mc_dropout=True)

    if args.demo:
        print("[calibrate] Using synthetic demo data (NOT real calibration data).")
        residuals, uncertainties, labels = make_demo_data(pinn)
    elif args.csv:
        residuals, uncertainties, labels = load_csv(args.csv)
        print(f"[calibrate] Loaded {len(residuals)} labelled examples from {args.csv}")
    else:
        ap.error("Provide --csv path/to/labelled_events.csv, or --demo to test the flow.")
        return

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"[calibrate] Label balance: {n_pos} plausible / {n_neg} false-positive")
    if min(n_pos, n_neg) < 4:
        print("[calibrate] WARNING: very few examples of one class — "
              "the fitted calibrator may not generalise. Collect more labelled events.")

    report = pinn.calibrate_threshold(
        residuals, uncertainties, labels,
        save_path=args.out,
    )
    print(f"[calibrate] Done. Train accuracy={report['train_accuracy']:.3f}, "
          f"final loss={report['final_loss']:.4f}")
    print(f"[calibrate] Calibrator saved to {args.out}")
    print(f"[calibrate] Load it in the pipeline with: "
          f"threshold_calibrator_path='{args.out}'")


if __name__ == "__main__":
    main()
