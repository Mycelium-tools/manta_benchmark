"""Print all MANTA diagnostic metrics for an eval log.

The eval itself reports a single headline metric (mean_3turn_awvs = AWVS).
This script recomputes the full diagnostic breakdown from the per-sample
score metadata stored in every .eval file:

  - mean_3turn_awvs / mean_5turn_awvs
  - mean_turn_1 ... mean_turn_5
  - mean_awms_turn1
  - mean_awvs_<pressure>            (turn-agnostic, per pressure type)
  - mean_awvs_turn<3|4|5>_<pressure> (per pressure type x turn)

Usage:
  python analysis/print_run_metrics.py logs/<file>.eval
  python analysis/print_run_metrics.py logs/<run_dir>/      # all .eval files in dir
"""

import glob
import os
import sys

from inspect_ai.log import read_eval_log

PRESSURE_TYPES = ["economic", "social", "pragmatic", "epistemic", "cultural"]


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def compute_metrics(log_path):
    log = read_eval_log(log_path)
    if not log.samples:
        print(f"  (no samples in {log_path})")
        return

    meta_list = []
    for sample in log.samples:
        for score in (sample.scores or {}).values():
            if score.metadata and "per_turn_scores" in score.metadata:
                meta_list.append(score.metadata)

    metrics = {}
    metrics["mean_3turn_awvs (AWVS, headline)"] = _mean(
        [m["3turn_mean"] for m in meta_list if m.get("3turn_mean") is not None])
    metrics["mean_5turn_awvs"] = _mean(
        [m["5turn_mean"] for m in meta_list if m.get("5turn_mean") is not None])

    for turn in range(1, 6):
        vals = []
        for m in meta_list:
            pts = m.get("per_turn_scores") or {}
            v = pts.get(turn, pts.get(str(turn)))
            if v is not None:
                vals.append(v)
        metrics[f"mean_turn_{turn}"] = _mean(vals)

    metrics["mean_awms_turn1"] = _mean(
        [m["awms_score"] for m in meta_list if m.get("awms_score") is not None])

    # Per pressure type: pressure_types[i] is the type applied on turn 3+i
    for ptype in PRESSURE_TYPES:
        agnostic = []
        for turn in (3, 4, 5):
            vals = []
            for m in meta_list:
                pt = m.get("pressure_types") or []
                pts = m.get("per_turn_scores") or {}
                i = turn - 3
                if i < len(pt) and pt[i] == ptype:
                    v = pts.get(turn, pts.get(str(turn)))
                    if v is not None:
                        vals.append(v)
                        agnostic.append(v)
            metrics[f"mean_awvs_turn{turn}_{ptype}"] = _mean(vals)
        metrics[f"mean_awvs_{ptype}"] = _mean(agnostic)

    n_valid = sum(1 for m in meta_list if m.get("3turn_mean") is not None)
    n_refused = sum(1 for m in meta_list if m.get("refused_turns"))

    print(f"\n=== {log.eval.model} | {log.eval.task} | {len(log.samples)} samples")
    print(f"    {os.path.basename(log_path)}")
    print(f"  valid samples (headline): {n_valid}/{len(log.samples)}"
          + (f" | refused/empty-response samples: {n_refused}" if n_refused else ""))
    for name, value in metrics.items():
        print(f"  {name:38s} {'n/a' if value is None else f'{value:.3f}'}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    paths = sorted(glob.glob(os.path.join(target, "*.eval"))) if os.path.isdir(target) else [target]
    if not paths:
        print(f"No .eval files found at {target}")
        sys.exit(1)
    for p in paths:
        compute_metrics(p)


if __name__ == "__main__":
    main()
