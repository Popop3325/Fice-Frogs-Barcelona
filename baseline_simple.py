"""
Simplest possession algorithm (v1).

Idea:
  1. In every frame, find the tracked player nearest to the ball.
  2. That player's team is the possessor (H or A).
  3. If the ball is not tracked, keep the last known label (within the same half).
  4. Smooth: majority vote over a WINDOW-frame window (25 frames = 1 second).
  5. Outside play (before kick-off, half-time, after the final whistle) write X.

What is NOT here yet: the D label (dead ball). That is the next step.

Usage (from the repository root, next to games.py):
    python baseline_simple.py
Output: evaluation/results/my_simple.csv
"""
import numpy as np
import pandas as pd
from kloppy import metrica

from games import asset_paths, half_video_blocks

N_FRAMES = 199319
WINDOW = 50                 # smoothing window length, in frames
PITCH_L, PITCH_W = 105.3, 68.0
OUT = "evaluation/results/my_simple.csv"

# --- 1. Load the tracking data ----------------------------------------------
paths = asset_paths()
ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
df = ds.to_df()
print("frames in tracking:", len(df), "| columns:", df.shape[1])

# which track slot belongs to which team
team_of = {}
for team in ds.metadata.teams:
    side = "H" if "home" in str(team.ground).lower() else "A"
    for p in team.players:
        team_of[str(p.player_id)] = side

x_cols = [c for c in df.columns if c.endswith("_x") and not c.startswith("ball")]
slot_ids = [c[:-2] for c in x_cols]
y_cols = [f"{s}_y" for s in slot_ids]
is_home = np.array([team_of.get(s) == "H" for s in slot_ids])
known = np.array([s in team_of for s in slot_ids])
print("player slots:", len(slot_ids), "| with a known team:", int(known.sum()))

# --- 2. Nearest player to the ball in every frame ---------------------------
bx = df["ball_x"].to_numpy() * PITCH_L
by = df["ball_y"].to_numpy() * PITCH_W
px = df[x_cols].to_numpy() * PITCH_L
py = df[y_cols].to_numpy() * PITCH_W

d = np.hypot(px - bx[:, None], py - by[:, None])
d[:, ~known] = np.nan
d = np.where(np.isnan(d), np.inf, d)

nearest = d.argmin(axis=1)
has_answer = np.isfinite(d.min(axis=1))        # ball visible and at least one player

raw = np.full(len(df), np.nan)                 # 1 = H, 0 = A, NaN = unknown
raw[has_answer] = is_home[nearest[has_answer]].astype(float)

frame_idx = (df["frame_id"].to_numpy() - 1).astype(int)   # frame_id starts at 1

# --- 3. Fill gaps + smooth within each half ----------------------------------
labels = np.full(N_FRAMES, "X", dtype=object)
for half, start, end in half_video_blocks("BAR-ATH"):
    mask = (frame_idx >= start) & (frame_idx <= end)
    seg = pd.Series(raw[mask], index=frame_idx[mask])
    seg = seg.ffill().bfill()                              # no ball -> previous label
    smooth = seg.rolling(WINDOW, center=True, min_periods=1).mean()
    labels[seg.index.to_numpy()] = np.where(smooth >= 0.5, "H", "A")

# --- 4. Save ------------------------------------------------------------------
out = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels})
out.to_csv(OUT, index=False)
print("written:", OUT, "| rows:", len(out))
print(out["label"].value_counts().to_string())
