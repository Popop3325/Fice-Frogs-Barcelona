"""
Possession-level error analysis on the public 5-minute window.

Compares the possessions of a submission with the true ones. A possession is an unbroken
run of H or of A (a dead-ball stretch splits possessions).

It prints:
  1. every true possession: best IoU with a predicted one, how many predicted pieces
     cover it, and what was predicted inside it (H / A / D)
  2. every predicted possession: best IoU with a true one, and a category:
       "in GT dead ball"  - at least half of its frames are dead ball in the ground truth
       "right team"       - the majority of its frames have the same team in the ground truth
       "wrong team"       - the majority of its frames have the other team in the ground truth
  3. a summary of the predicted possessions by category and by whether they match (IoU >= 0.3)

NOTE: only 17 true possessions. Use it to understand failure modes, not to tune thresholds.

Usage (from the repository root):
    python possession_analysis.py evaluation/results/my_v3.csv > possession_analysis_v3.txt
"""
import sys

import numpy as np
import pandas as pd

GT = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
SUB = sys.argv[1] if len(sys.argv) > 1 else "evaluation/results/my_v3.csv"
FPS = 25

gt = pd.read_csv(GT)
gt = gt[gt["label"] != "X"].set_index("frame")
sub = pd.read_csv(SUB).set_index("frame").loc[gt.index]

truth = gt["label"].to_numpy()
pred = sub["label"].to_numpy()
frames = gt.index.to_numpy()


def poss_runs(lab):
    """Possessions = unbroken runs of H or A. Returns a list of (start, end_exclusive, team)."""
    out = []
    i, n = 0, len(lab)
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        if lab[i] in ("H", "A"):
            out.append((i, j, lab[i]))
        i = j
    return out


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    inter = overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


G = poss_runs(truth)
P = poss_runs(pred)
print(f"submission: {SUB}")
print(f"true possessions: {len(G)} | predicted possessions: {len(P)}")
print()

# --- 1. true possessions ------------------------------------------------------------------
print("== True possessions ==")
for k, g in enumerate(G):
    s, e, t = g
    pieces = [p for p in P if overlap(g, p) > 0]
    best = max((iou(g, p) for p in P), default=0.0)
    seg = pred[s:e]
    print(f"  GT #{k:02d} {t}  frames {frames[s]}-{frames[e - 1]}  {(e - s) / FPS:5.1f} s  "
          f"best IoU {best:4.2f}  predicted pieces {len(pieces)}  "
          f"| predicted inside: H {np.mean(seg == 'H'):4.0%} A {np.mean(seg == 'A'):4.0%} "
          f"D {np.mean(seg == 'D'):4.0%}")
print()

# --- 2. predicted possessions ---------------------------------------------------------------
print("== Predicted possessions ==")
rows = []
for k, p in enumerate(P):
    s, e, t = p
    seg = truth[s:e]
    sh_d = np.mean(seg == "D")
    sh_h = np.mean(seg == "H")
    sh_a = np.mean(seg == "A")
    if sh_d >= 0.5:
        cat = "in GT dead ball"
    else:
        majority = "H" if sh_h >= sh_a else "A"
        cat = "right team" if majority == t else "wrong team"
    best = max((iou(p, g) for g in G), default=0.0)
    rows.append((cat, best))
    print(f"  P #{k:02d} {t}  frames {frames[s]}-{frames[e - 1]}  {(e - s) / FPS:5.1f} s  "
          f"best IoU {best:4.2f}  {cat:<16} "
          f"| GT inside: H {sh_h:4.0%} A {sh_a:4.0%} D {sh_d:4.0%}")
print()

# --- 3. summary ----------------------------------------------------------------------------------
print("== Summary of predicted possessions ==")
summary = pd.DataFrame(rows, columns=["category", "best_iou"])
summary["matched (IoU >= 0.3)"] = summary["best_iou"] >= 0.3
tab = summary.groupby("category").agg(predicted=("best_iou", "size"),
                                       matched=("matched (IoU >= 0.3)", "sum"))
tab["unmatched"] = tab["predicted"] - tab["matched"]
print(tab.to_string())
