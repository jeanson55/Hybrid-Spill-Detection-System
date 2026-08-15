"""
Interactive Labeling Tool for Threshold Calibration
====================================================

Walks through your saved spill_captures and lets you mark each one as a
real spill (1) or false positive (0), then writes a CSV ready to hand
straight to calibrate_threshold.py --csv.

Why this exists: ThresholdCalibrator needs labelled (residual,
uncertainty, label) data, and there's no way around a human looking at
each event to provide that. SpillImageSaver only logs CONFIRMED alerts
(things that passed geometric + physics + temporal confirmation), so
"false positive" here specifically means "this got through every filter
but still isn't a real spill" — e.g. a reflection or a puddle that
persisted across frames without ever behaving like a spreading fluid.
Only you can make that call for your site.

Usage
-----
    python scripts/label_captures.py --captures-dir spill_captures --out labelled_events.csv

Labels save incrementally to <captures-dir>/labels.json as you go, so
you can quit partway (press 'q' or Ctrl+C) and resume later without
re-labeling anything already done.

Controls: y = real spill   n = false positive   s = skip (unsure)   q = quit and write CSV
"""

import argparse
import csv
import json
import os
import platform
import subprocess
from pathlib import Path


def open_image(path: Path):
    """Best-effort open in the OS's default image viewer; falls back to
    just printing the path if that fails for any reason."""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path))  # noqa: only exists on Windows
        elif system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print(f"[label] Could not open an image viewer ({e}) — open this manually: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures-dir", type=str, default="spill_captures",
                     help="Directory written by SpillImageSaver (default: spill_captures)")
    ap.add_argument("--out", type=str, default="labelled_events.csv",
                     help="Output CSV path for calibrate_threshold.py --csv")
    args = ap.parse_args()

    captures_dir = Path(args.captures_dir)
    log_path = captures_dir / "capture_log.json"
    if not log_path.exists():
        print(f"[label] No capture log found at {log_path}.")
        print("        Nothing to label yet — entries only appear here once the "
              "live pipeline has actually confirmed at least one alert.")
        return

    entries = json.loads(log_path.read_text())
    labels_path = captures_dir / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    unlabeled = [e for e in entries if e["alert_id"] not in labels]
    print(f"[label] {len(entries)} total captures, {len(labels)} already labelled, "
          f"{len(unlabeled)} left to review.\n")

    for i, entry in enumerate(unlabeled):
        crop_path = captures_dir / entry["files"]["crop"]
        print(f"[{i + 1}/{len(unlabeled)}] {entry['alert_id']}  "
              f"residual={entry['avg_residual']}  "
              f"fluid={entry.get('detected_fluid_class', '?')}  "
              f"conf={entry['avg_confidence']}")
        open_image(crop_path)

        choice = None
        while choice not in ("y", "n", "s", "q"):
            choice = input("Real spill? [y/n/s=skip/q=quit]: ").strip().lower()

        if choice == "q":
            print("[label] Quitting — progress saved.")
            break
        if choice == "s":
            continue

        labels[entry["alert_id"]] = 1 if choice == "y" else 0
        labels_path.write_text(json.dumps(labels, indent=2))  # persist after every label

    # Build the CSV calibrate_threshold.py expects, from everything labelled so far
    # (including labels from earlier sessions, not just this run).
    rows = []
    for entry in entries:
        if entry["alert_id"] in labels:
            rows.append({
                "residual": entry["avg_residual"],
                "uncertainty": entry.get("uncertainty") or 0.0,
                "label": labels[entry["alert_id"]],
            })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["residual", "uncertainty", "label"])
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(r["label"] for r in rows)
    print(f"\n[label] Wrote {len(rows)} labelled rows to {args.out} "
          f"({n_pos} real spill / {len(rows) - n_pos} false positive).")
    if len(rows) < 8:
        print("[label] calibrate_threshold.py needs at least 8 labelled examples — "
              "keep labeling more captures (or wait for more confirmed alerts) before running it.")
    else:
        print(f"[label] Ready: python calibrate_threshold.py --csv {args.out} --pinn-weights pinn_adapted.pt")


if __name__ == "__main__":
    main()
