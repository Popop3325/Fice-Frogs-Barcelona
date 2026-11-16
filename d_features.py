"""
Diagnostic v2: how do dead-ball frames (D) differ from live-play frames (H/A)
in the public 5-minute ground truth?

For each feature it prints the median and the p10..p90 range separately for D and
for live play, and at the end the precision/recall of a few simple cues for D.

NOTE: the window contains only 4 D segments. Use this to form hypotheses,
not to tune thresholds.

Usage (from the repository root):
    python d_features.py > d_features.txt
"""
import numpy as np
import pandas as pd
from kloppy import metrica, sportscode

from games import asset_paths

GT = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
FPS = 25
L, W = 105.3, 68.0  # pitch length and width in metres

# --- ground truth and window bounds -------------------------------------------
gt = pd.read_csv(GT)
gt = gt[gt["label"] != "X"].set_index("frame")
lo, hi = int(gt.index.min()), int(gt.index.max())

# --- tracking (loaded 1 second early so the speed feature has a look-back) -----
paths = asset_paths()
ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
df = ds.to_df()
df["frame"] = df["frame_id"] - 1  # ATD frame_id starts at 1, video frames at 0
df = df.set_index("frame").loc[lo - FPS:hi]

bx = df["ball_x"] * L
by = df["ball_y"] * W
missing = bx.isna()

feats = pd.DataFrame(index=df.index)
feats["missing"] = missing.astype(float)

# length of the unbroken no-ball stretch that contains the frame (0 if the ball is visible)
grp = (missing != missing.shift()).cumsum()
feats["missing_run"] = missing.groupby(grp).transform("size").where(missing, 0)

# ball speed over 1 second (m/s), only where the ball is visible in both frames
feats["ball_speed"] = np.hypot(bx - bx.shift(FPS), by - by.shift(FPS)) * FPS / FPS

# distance from the last known ball position to the pitch boundary, metres (< 0 = outside)
bxf, byf = bx.ffill(), by.ffill()
feats["edge_dist"] = np.minimum.reduce([bxf, L - bxf, byf, W - byf])

# mean speed of all tracked players (m/s)
s_cols = [c for c in df.columns if c.endswith("_s") and not c.startswith("ball")]
feats["player_speed"] = df[s_cols].mean(axis=1)

# distance from the ball to the nearest tracked player, metres
x_cols = [c for c in df.columns if c.endswith("_x") and not c.startswith("ball")]
y_cols = [c[:-2] + "_y" for c in x_cols]
px = df[x_cols].to_numpy() * L
py = df[y_cols].to_numpy() * W
d = np.hypot(px - bx.to_numpy()[:, None], py - by.to_numpy()[:, None])
d = np.where(np.isnan(d), np.inf, d)
nearest = d.min(axis=1)
feats["nearest_player"] = np.where(np.isfinite(nearest), nearest, np.nan)

# is a Smart Tagging SET PIECES tag active on this frame?
ev = sportscode.load(paths["smart_tagging"]).to_df()
ev = ev[ev["code"].str.upper() == "SET PIECES"]
sp = pd.Series(False, index=feats.index)
for _, e in ev.iterrows():
    f0 = int(round(e["timestamp"].total_seconds() * FPS))
    f1 = int(round(e["end_timestamp"].total_seconds() * FPS))
    sp.loc[(sp.index >= f0) & (sp.index <= f1)] = True
feats["set_piece_tag"] = sp.astype(float)

# keep only the labelled window
feats = feats.loc[lo:hi]
dead = gt.loc[feats.index, "label"].eq("D")
print(f"frames in window: {len(feats)} | dead (D): {int(dead.sum())} "
      f"({dead.mean():.0%}) | live play: {int((~dead).sum())}")
print()

# --- feature comparison ----------------------------------------------------------
print("Feature (median [p10 .. p90])".ljust(44), "D".ljust(30), "live play (H/A)")
titles = [
    ("missing", "share of frames without the ball"),
    ("missing_run", "no-ball stretch length, frames"),
    ("ball_speed", "ball speed over 1 s, m/s"),
    ("edge_dist", "ball distance to pitch edge, m"),
    ("player_speed", "mean player speed, m/s"),
    ("nearest_player", "nearest player to ball, m"),
    ("set_piece_tag", "SET PIECES tag active"),
]


def summ(s):
    s = s.dropna()
    if s.empty:
        return "no data"
    return f"{s.median():7.2f} [{s.quantile(.1):6.2f} .. {s.quantile(.9):6.2f}]"


for col, title in titles:
    print(title.ljust(44), summ(feats.loc[dead, col]).ljust(30), summ(feats.loc[~dead, col]))
print()

# --- simple cues for D -------------------------------------------------------------
print("Simple cues for D (precision = share of flagged frames that are truly D;")
print("recall = share of true D frames that the cue finds):")


def pr(name, cue):
    cue = cue.fillna(False).astype(bool)
    tp = int((cue & dead).sum())
    n = int(cue.sum())
    prec = tp / n if n else float("nan")
    rec = tp / int(dead.sum())
    print(f"  {name:<52} frames {n:5d}   precision {prec:5.2f}   recall {rec:5.2f}")


pr("ball missing for >= 2 s in a row", feats["missing_run"] >= 2 * FPS)
pr("ball missing for >= 4 s in a row", feats["missing_run"] >= 4 * FPS)
pr("SET PIECES tag active", feats["set_piece_tag"] > 0)
pr("ball within 3 m of the edge or outside", feats["edge_dist"] <= 3.0)
pr("ball missing >= 2 s OR SET PIECES tag",
   (feats["missing_run"] >= 2 * FPS) | (feats["set_piece_tag"] > 0))
print()

# --- per D segment -----------------------------------------------------------------
print("Per D segment (medians):")
seg = (dead != dead.shift()).cumsum()
for _, idx in feats[dead].groupby(seg[dead]).groups.items():
    f = feats.loc[idx]
    print(f"  frames {idx[0]}-{idx[-1]}  no ball {f['missing'].mean():4.0%}  "
          f"ball speed {f['ball_speed'].median():5.2f}  "
          f"player speed {f['player_speed'].median():5.2f}  "
          f"edge dist {f['edge_dist'].median():6.2f} m  "
          f"SET PIECES {f['set_piece_tag'].mean():4.0%}")
