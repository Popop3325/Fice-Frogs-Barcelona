"""
Ball possession for every video frame of BAR-ATH (Barcelona vs Athletic Club).

Output: one CSV, one row per frame of the tactical video, labels
    H = home team (Barcelona) in possession, ball alive
    A = away team (Athletic) in possession, ball alive
    D = dead ball
    X = no answer (outside play: before kick-off, half-time, after the final whistle)

Method (details in the technical report):
  1. Owner (H/A). In every frame the team of the tracked player nearest to the ball is a
     vote. Frames without a ball give no vote and keep the previous label. The votes are
     averaged over a rolling window (default 100 frames = 4 s) and passed through a
     hysteresis (switch to H above 0.65, to A below 0.35). Possession is therefore a
     property of a few seconds of play, not of a single touch.
  2. Dead ball (D). Rule-based voting between four cues computed from the tracking data
     and the Smart Tagging file:
         slow     median mean-player-speed over 5 s below 1.5 m/s (players walking)
         missing  ball absent from the tracking for >= 2 s in a row
         edge     last known ball position within 3 m of the pitch edge (or outside)
         tag      a SET PIECES interval of the Smart Tagging file is active
     D = (slow AND (missing OR edge OR tag)) OR (tag AND edge). Gaps shorter than 2 s
     between D runs are closed and D runs shorter than 4 s are dropped.
  3. D overrides H/A. Frames outside the two halves are X.

Usage (from the repository root, next to games.py, data unpacked in ./data):
    python possession.py --out evaluation/results/final_w100.csv

No ground truth of any kind is read by this script.
"""
import argparse

import numpy as np
import pandas as pd
from kloppy import metrica, sportscode

from games import asset_paths, half_video_blocks

N_FRAMES = 199319
FPS = 25
L, W = 105.3, 68.0  # pitch length and width in metres

# ---- owner (H/A) parameters ---------------------------------------------------
DEFAULT_WINDOW = 100  # rolling window of the owner votes, in frames (25 frames = 1 s)
HI, LO = 0.65, 0.35   # hysteresis thresholds: switch to H above HI, to A below LO

# ---- dead-ball (D) parameters -------------------------------------------------
SPEED_WIN = 5 * FPS        # window for the smoothed mean player speed
SLOW_SPEED = 1.5           # m/s, below this the players are "walking"
MISSING_RUN = 2 * FPS      # ball missing for at least this many frames in a row
EDGE_M = 3.0               # ball within this distance of the pitch edge (or outside)
MAX_GAP = 2 * FPS          # close gaps shorter than this between D runs
MIN_D = 4 * FPS            # drop D runs shorter than this


# ---- data loading -------------------------------------------------------------------
def load_tracking():
    """Read the automatic tracking data as one row per video frame (0 .. N_FRAMES-1)."""
    paths = asset_paths()
    ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
    df = ds.to_df()
    df["frame"] = df["frame_id"] - 1  # ATD frame_id starts at 1, video frames at 0
    df = df.set_index("frame").reindex(np.arange(N_FRAMES))
    print("frames in tracking:", len(df), "| columns:", df.shape[1])
    return paths, ds, df


# ---- owner signal -------------------------------------------------------------------
def owner_vote(ds, df):
    """Per frame: 1.0 if the nearest tracked player to the ball is a home player,
    0.0 if an away player, NaN if the ball or all players are missing."""
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

    bx = (df["ball_x"] * L).to_numpy()
    by = (df["ball_y"] * W).to_numpy()
    px = df[x_cols].to_numpy() * L
    py = df[y_cols].to_numpy() * W

    d = np.hypot(px - bx[:, None], py - by[:, None])
    d[:, ~known] = np.nan
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.argmin(axis=1)
    has_answer = np.isfinite(d.min(axis=1))

    raw = np.full(N_FRAMES, np.nan)
    raw[has_answer] = is_home[nearest[has_answer]].astype(float)
    return raw


# ---- dead-ball candidates -------------------------------------------------------------
def dead_ball_candidates(df, paths):
    """Per frame boolean: the dead-ball rule fires (before the clean-up of short runs)."""
    bx_s = df["ball_x"] * L
    by_s = df["ball_y"] * W

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

    corroborated = cue_missing | cue_edge | cue_tag
    return (cue_slow & corroborated) | (cue_tag & cue_edge)


# ---- helpers --------------------------------------------------------------------------
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


# ---- labels ---------------------------------------------------------------------------
def build_labels(raw, dead_raw, window):
    """Combine the owner votes and the dead-ball candidates into one label per frame."""
    labels = np.full(N_FRAMES, "X", dtype=object)
    for half, start, end in half_video_blocks("BAR-ATH"):
        sl = slice(start, end + 1)

        filled = pd.Series(raw[sl]).ffill().bfill()   # no ball -> keep previous label
        share = filled.rolling(window, center=True, min_periods=1).mean().to_numpy()
        owner = hysteresis(share)

        dead = clean_dead(dead_raw[sl])
        lab = np.where(owner == 1.0, "H", "A").astype(object)
        lab[dead] = "D"
        labels[sl] = lab

        n_poss, mean_poss = possession_stats(lab)
        print(f"half {half}: D share {dead.mean():5.1%} | "
              f"possessions {n_poss} (mean {mean_poss:.1f} s)")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Ball possession per video frame")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="owner smoothing window in frames (default %(default)s)")
    parser.add_argument("--out", default="evaluation/results/submission.csv",
                        help="output CSV (default %(default)s)")
    args = parser.parse_args()

    paths, ds, df = load_tracking()
    raw = owner_vote(ds, df)
    dead_raw = dead_ball_candidates(df, paths)
    labels = build_labels(raw, dead_raw, args.window)

    out = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels})
    out.to_csv(args.out, index=False)
    print("written:", args.out, "| rows:", len(out))
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
