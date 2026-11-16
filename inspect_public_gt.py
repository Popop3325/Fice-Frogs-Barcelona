"""
Diagnostic: what happens in the public 5-minute ground truth.

For every segment (H / A / D) it prints:
  - start and end frames, duration in seconds
  - share of frames where the ball is missing from the tracking
  - ball position (x, y in the range 0..1) at the start and at the end
  - which Smart Tagging events overlap the segment (+-2 seconds)

Goal: see what a dead ball (D) looks like in the data so a rule for D can be designed.

Usage (from the repository root):
    python inspect_public_gt.py
"""
import numpy as np
import pandas as pd
from kloppy import metrica, sportscode

from games import asset_paths

GT = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
FPS = 25
PAD = 2 * FPS

# --- ground truth (graded frames only) ----------------------------------------
gt = pd.read_csv(GT)
gt = gt[gt["label"] != "X"].reset_index(drop=True)
seg_id = (gt["label"] != gt["label"].shift()).cumsum()
segs = gt.groupby(seg_id).agg(label=("label", "first"),
                              start=("frame", "min"),
                              end=("frame", "max")).reset_index(drop=True)

# --- ball from the tracking data ----------------------------------------------
paths = asset_paths()
ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
df = ds.to_df()
df["frame"] = df["frame_id"] - 1  # ATD frame_id starts at 1, video frames at 0
ball = df.set_index("frame")[["ball_x", "ball_y"]]

# --- Smart Tagging ----------------------------------------------------------------
ev = sportscode.load(paths["smart_tagging"]).to_df()
ev["f0"] = (ev["timestamp"].dt.total_seconds() * FPS).round().astype(int)
ev["f1"] = (ev["end_timestamp"].dt.total_seconds() * FPS).round().astype(int)
print("Smart Tagging columns:", list(ev.columns))
print()


def fmt_xy(row):
    if row is None:
        return "no ball"
    return f"({row.ball_x:.3f}, {row.ball_y:.3f})"


def first_valid(b):
    b = b.dropna()
    return None if b.empty else b.iloc[0]


def last_valid(b):
    b = b.dropna()
    return None if b.empty else b.iloc[-1]


for i, s in segs.iterrows():
    b = ball.loc[s.start:s.end]
    nan_share = b["ball_x"].isna().mean()
    dur = (s.end - s.start + 1) / FPS
    print(f"#{i:02d} {s.label}  frames {s.start}-{s.end}  {dur:5.1f} s  "
          f"no ball {nan_share:4.0%}  "
          f"ball at start {fmt_xy(first_valid(b))}  at end {fmt_xy(last_valid(b))}")

    near = ev[(ev["f1"] >= s.start - PAD) & (ev["f0"] <= s.end + PAD)]
    for _, e in near.iterrows():
        extra = []
        for col in ("Type", "Team"):
            if col in ev.columns and pd.notna(e[col]):
                extra.append(f"{col}={e[col]}")
        print(f"       tag: {e['code']}  frames {e['f0']}-{e['f1']}  {' '.join(extra)}")
