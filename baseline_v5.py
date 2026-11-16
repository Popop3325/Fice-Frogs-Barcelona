"""
Possession baseline v5: window sensitivity sweep.

Same logic as baseline_v3.py (nearest-player owner signal + hysteresis, rule-based
dead ball), but the owner smoothing window is a parameter and several windows are
written in one run, so the tracking data is read only once.

Why: on the public 5-minute window every difference between our variants is inside the
bootstrap noise (17 possessions), so the window should be chosen by structure and by the
full-match behaviour, not by the public score. The sweep documents the sensitivity.

Sanity check: --windows 50 must reproduce evaluation/results/my_v3.csv exactly:
    cmp evaluation/results/my_v3.csv evaluation/results/my_v5_w50.csv

Usage (from the repository root, next to games.py):
    python baseline_v5.py                      # windows 50 75 100 125
    python baseline_v5.py --windows 100        # a single window
Output: evaluation/results/my_v5_w<WINDOW>.csv for every window
"""
import argparse

import numpy as np
import pandas as pd
from kloppy import metrica, sportscode

from games import asset_paths, half_video_blocks

parser = argparse.ArgumentParser(description="Possession baseline v5 (window sweep)")
parser.add_argument("--windows", type=int, nargs="+", default=[50, 75, 100, 125],
                    help="owner smoothing windows in frames (25 frames = 1 s)")
args = parser.parse_args()

N_FRAMES = 199319
FPS = 25
L, W = 105.3, 68.0  # pitch length and width in metres

# ---- owner (H/A) parameters ---------------------------------------------------
HI, LO = 0.65, 0.35  # hysteresis thresholds: switch to H above HI, to A below LO

# ---- dead-ball (D) parameters (identical to v3) -------------------------------
SPEED_WIN = 5 * FPS        # window for the smoothed mean player speed
SLOW_SPEED = 1.5           # m/s, below this the players are "walking"
MISSING_RUN = 2 * FPS      # ball missing for at least this many frames in a row
EDGE_M = 3.0               # ball within this distance of the pitch edge (or outside)
MAX_GAP = 2 * FPS          # close gaps shorter than this between D runs
MIN_D = 4 * FPS            # drop D runs shorter than this

blocks = list(half_video_blocks("BAR-ATH"))  # [(half, first_frame, last_frame), ...]

# ---- 1. load tracking ------------------------------------------------------------
paths = asset_paths()
ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
df = ds.to_df()
df["frame"] = df["frame_id"] - 1  # ATD frame_id starts at 1, video frames at 0
df = df.set_index("frame").reindex(np.arange(N_FRAMES))
print("frames in tracking:", int(df["ball_x"].shape[0]), "| columns:", df.shape[1])

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

# ---- 2. owner signal: team of the nearest player to the ball ---------------------
bx_s = df["ball_x"] * L
by_s = df["ball_y"] * W
bx = bx_s.to_numpy()
by = by_s.to_numpy()
px = df[x_cols].to_numpy() * L
py = df[y_cols].to_numpy() * W

d = np.hypot(px - bx[:, None], py - by[:, None])
d[:, ~known] = np.nan
d = np.where(np.isnan(d), np.inf, d)
nearest = d.argmin(axis=1)
has_answer = np.isfinite(d.min(axis=1))

raw = np.full(N_FRAMES, np.nan)  # 1 = H, 0 = A, NaN = unknown
raw[has_answer] = is_home[nearest[has_answer]].astype(float)

# ---- 3. dead-ball cues (identical to v3) -----------------------------------------
# (a) players walking
s_cols = [c for c in df.columns if c.endswith("_s") and not c.startswith("ball")]
pspeed = df[s_cols].mean(axis=1).ffill().bfill()
pspeed_smooth = pspeed.rolling(SPEED_WIN, center=True, min_periods=1).median()
cue_slow = (pspeed_smooth < SLOW_SPEED).to_numpy()

# (b) ball missing for a long stretch
missing = bx_s.isna()
grp = (missing != missing.shift()).cumsum()
run_len = missing.groupby(grp).transform("size").where(missing, 0)
cue_missing = (run_len >= MISSING_RUN).to_numpy()

# (c) last known ball position near the pitch edge (or outside)
bxf, byf = bx_s.ffill(), by_s.ffill()
edge_dist = np.asarray(np.minimum.reduce([bxf, L - bxf, byf, W - byf]), dtype=float)
cue_edge = edge_dist <= EDGE_M  # NaN compares as False

# (d) Smart Tagging SET PIECES interval active
cue_tag = np.zeros(N_FRAMES, dtype=bool)
st = sportscode.load(paths["smart_tagging"]).to_df()
st = st[st["code"].str.upper() == "SET PIECES"]
for _, e in st.iterrows():
    f0 = max(int(round(e["timestamp"].total_seconds() * FPS)), 0)
    f1 = min(int(round(e["end_timestamp"].total_seconds() * FPS)), N_FRAMES - 1)
    cue_tag[f0:f1 + 1] = True

# D rule: walking players need corroboration; set piece + ball at the edge is a restart
corroborated = cue_missing | cue_edge | cue_tag
dead_raw = (cue_slow & corroborated) | (cue_tag & cue_edge)


# ---- helpers ----------------------------------------------------------------------
def runs(mask):
    """Return (starts, ends_exclusive, values) of the runs of a boolean array."""
    m = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(m.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(m)]
    return starts, ends, m[starts]


def clean_dead(mask):
    """Close short gaps between D runs, then drop short D runs."""
    m = np.asarray(mask, dtype=bool).copy()
    starts, ends, vals = runs(m)
    for s, e, v in zip(starts, ends, vals):
        if (not v) and (e - s) <= MAX_GAP and s > 0 and e < len(m):
            m[s:e] = True
    starts, ends, vals = runs(m)
    for s, e, v in zip(starts, ends, vals):
        if v and (e - s) < MIN_D:
            m[s:e] = False
    return m


def hysteresis(share):
    """Binary state that flips to 1 above HI and to 0 below LO, otherwise keeps its value."""
    out = np.empty(len(share))
    state = 1.0 if share[0] >= 0.5 else 0.0
    for i, v in enumerate(share):
        if v >= HI:
            state = 1.0
        elif v <= LO:
            state = 0.0
        out[i] = state
    return out


def possession_stats(lab):
    """Number and mean duration (s) of possessions: unbroken runs of H or of A."""
    s = pd.Series(lab)
    grp = (s != s.shift()).cumsum()
    lens = s.groupby(grp).size()
    kinds = s.groupby(grp).first()
    poss = lens[kinds.isin(["H", "A"])]
    return len(poss), (poss.mean() / FPS if len(poss) else float("nan"))


# ---- 4. one output file per window -----------------------------------------------------
dead_by_half = {half: clean_dead(dead_raw[start:end + 1]) for half, start, end in blocks}

for window in args.windows:
    labels = np.full(N_FRAMES, "X", dtype=object)
    print()
    print(f"== owner window {window} frames ({window / FPS:.1f} s) ==")

    for half, start, end in blocks:
        sl = slice(start, end + 1)

        filled = pd.Series(raw[sl]).ffill().bfill()   # no ball -> keep previous label
        share = filled.rolling(window, center=True, min_periods=1).mean().to_numpy()
        owner = hysteresis(share)

        dead = dead_by_half[half]
        lab = np.where(owner == 1.0, "H", "A").astype(object)
        lab[dead] = "D"
        labels[sl] = lab

        n_poss, mean_poss = possession_stats(lab)
        print(f"  half {half}: D share {dead.mean():5.1%} | "
              f"possessions {n_poss} (mean {mean_poss:.1f} s)")

    out_path = f"evaluation/results/my_v5_w{window}.csv"
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(out_path, index=False)
    print(f"  written: {out_path}")
