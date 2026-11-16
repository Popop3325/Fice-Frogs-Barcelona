"""
Error analysis of the dead-ball (D) rule on the public 5-minute window.

Works with any baseline that writes a per-frame cue file with the columns
frame, label, slow, missing, edge, tag, player_speed_smooth
(baseline_v2.py -> data/cues_v2.csv, baseline_v3.py -> data/cues_v3.csv).

It prints:
  1. every predicted D run: is it a true dead ball, and which cues fired
  2. every true D run: how much of it was found, and what was predicted instead
  3. for every combination of cues: how many frames, and what share are truly D

NOTE: the window has only 4 true D segments. This is for understanding failures,
not for tuning thresholds against the public score.

Usage (from the repository root):
    python error_analysis.py data/cues_v3.csv > error_analysis_v3.txt
"""
import sys

import numpy as np
import pandas as pd

GT = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
CUES = sys.argv[1] if len(sys.argv) > 1 else "data/cues_v3.csv"
FPS = 25

gt = pd.read_csv(GT)
gt = gt[gt["label"] != "X"].set_index("frame")
cu = pd.read_csv(CUES).set_index("frame").loc[gt.index]

truth = gt["label"]
pred = cu["label"]
frames = cu.index.to_numpy()
pred_d = (pred == "D").to_numpy()
true_d = (truth == "D").to_numpy()


def runs(mask):
    """Return (starts, ends_exclusive, values) of the runs of a boolean array."""
    m = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(m.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(m)]
    return starts, ends, m[starts]


print(f"cue file: {CUES}")
print(f"graded frames: {len(cu)} | true D: {int(true_d.sum())} | predicted D: {int(pred_d.sum())} "
      f"| both: {int((pred_d & true_d).sum())}")
print()

# --- 1. predicted D runs ---------------------------------------------------------------
print("== Predicted D runs ==")
print("(truly D = share of the run that is D in the ground truth)")
for a, b, val in zip(*runs(pred_d)):
    if not val:
        continue
    seg_truth = truth.iloc[a:b]
    c = cu.iloc[a:b]
    print(f"  frames {frames[a]}-{frames[b - 1]}  {(b - a) / FPS:5.1f} s  "
          f"truly D {seg_truth.eq('D').mean():4.0%}  "
          f"(GT live: H {seg_truth.eq('H').mean():4.0%} A {seg_truth.eq('A').mean():4.0%})  "
          f"| cues: slow {c['slow'].mean():4.0%} missing {c['missing'].mean():4.0%} "
          f"edge {c['edge'].mean():4.0%} tag {c['tag'].mean():4.0%}  "
          f"| smoothed player speed {c['player_speed_smooth'].median():4.2f}")
print()

# --- 2. true D runs ---------------------------------------------------------------------
print("== True D runs ==")
for a, b, val in zip(*runs(true_d)):
    if not val:
        continue
    seg_pred = pred.iloc[a:b]
    c = cu.iloc[a:b]
    print(f"  frames {frames[a]}-{frames[b - 1]}  {(b - a) / FPS:5.1f} s  "
          f"found as D {seg_pred.eq('D').mean():4.0%}  "
          f"(predicted instead: H {seg_pred.eq('H').mean():4.0%} A {seg_pred.eq('A').mean():4.0%})  "
          f"| cues: slow {c['slow'].mean():4.0%} missing {c['missing'].mean():4.0%} "
          f"edge {c['edge'].mean():4.0%} tag {c['tag'].mean():4.0%}")
print()

# --- 3. cue combinations ------------------------------------------------------------------
print("== Cue combinations over all graded frames ==")
print("combo digits = slow, missing, edge, tag (1 = cue active)")
combo = cu[["slow", "missing", "edge", "tag"]].astype(int).astype(str).agg("".join, axis=1)
tab = pd.DataFrame({"combo": combo.to_numpy(), "true_D": true_d, "pred_D": pred_d})
g = tab.groupby("combo").agg(frames=("true_D", "size"),
                              true_D=("true_D", "sum"),
                              pred_D=("pred_D", "sum"))
g["share_true_D"] = g["true_D"] / g["frames"]
g = g.sort_values("frames", ascending=False)
print(g.to_string(float_format=lambda v: f"{v:5.2f}"))
