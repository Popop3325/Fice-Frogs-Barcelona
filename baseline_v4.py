"""
Possession baseline v4.

Difference compared with baseline_v3.py: the OWNER (H/A) signal is based on control,
not on "nearest player, whatever the distance".

v1-v3 took the team of the nearest tracked player to the ball in every frame, even if
that player was 15 m away (ball in the air, long pass) or if a player of each team was
equally close (50/50 duel). The possession rules say that in loose-ball phases the team
that last had possession keeps it, and that a change needs a player to have the ball
under control.

v4 votes only when the situation is clear:
  - a player of team X is within CTRL_R metres of the ball, AND
  - the nearest player of the other team is at least CTRL_MARGIN metres farther.
Otherwise the frame gives no vote and the previous label is kept. The vote series is
then smoothed and passed through a hysteresis, exactly as in v2/v3.

The dead-ball (D) rule is unchanged from v3, so a comparison between my_v3 and my_v4
isolates the effect of the owner logic.
Do not tune CTRL_R / CTRL_MARGIN against the public score (17 possessions only).

Usage (from the repository root, next to games.py):
    python baseline_v4.py
Output: evaluation/results/my_v4.csv  (+ data/cues_v4.csv for error_analysis.py)
"""
import numpy as np
import pandas as pd
from kloppy import metrica, sportscode

from games import asset_paths, half_video_blocks

N_FRAMES = 199319
FPS = 25
L, W = 105.3, 68.0  # pitch length and width in metres
OUT = "evaluation/results/my_v4.csv"
CUES_OUT = "data/cues_v4.csv"

# ---- owner (H/A) parameters ---------------------------------------------------
CTRL_R = 3.0         # a player must be within this distance of the ball (m) to count
CTRL_MARGIN = 1.0    # ... and the nearest opponent must be this much farther (m)
WINDOW = 50          # rolling window for the share of "home controls", frames
HI, LO = 0.65, 0.35  # hysteresis thresholds: switch to H above HI, to A below LO

# ---- dead-ball (D) parameters (unchanged from v3) -----------------------------
SPEED_WIN = 5 * FPS        # window for the smoothed mean player speed
SLOW_SPEED = 1.5           # m/s, below this the players are "walking"
MISSING_RUN = 2 * FPS      # ball missing for at least this many frames in a row
EDGE_M = 3.0               # ball within this distance of the pitch edge (or outside)
MAX_GAP = 2 * FPS          # close gaps shorter than this between D runs
MIN_D = 4 * FPS            # drop D runs shorter than this

blocks = list(half_video_blocks("BAR-ATH"))  # [(half, first_frame, last_frame), ...]
graded = np.zeros(N_FRAMES, dtype=bool)
for _, b0, b1 in blocks:
    graded[b0:b1 + 1] = True

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

# ---- 2. owner signal: who clearly controls the ball ------------------------------
bx_s = df["ball_x"] * L
by_s = df["ball_y"] * W
bx = bx_s.to_numpy()
by = by_s.to_numpy()
px = df[x_cols].to_numpy() * L
py = df[y_cols].to_numpy() * W

d = np.hypot(px - bx[:, None], py - by[:, None])
d[:, ~known] = np.nan
d = np.where(np.isnan(d), np.inf, d)

home_slots = known & is_home
away_slots = known & ~is_home
d_home = d[:, home_slots].min(axis=1)  # distance of the nearest home player (inf if none)
d_away = d[:, away_slots].min(axis=1)  # distance of the nearest away player (inf if none)

both = np.isfinite(d_home) & np.isfinite(d_away)
with np.errstate(invalid="ignore"):
    margin = np.where(both, d_away - d_home, np.nan)  # > 0: a home player is closer

home_ctrl = both & (d_home <= CTRL_R) & (margin >= CTRL_MARGIN)
away_ctrl = both & (d_away <= CTRL_R) & (margin <= -CTRL_MARGIN)

raw = np.full(N_FRAMES, np.nan)  # 1 = H controls, 0 = A controls, NaN = no clear vote
raw[home_ctrl] = 1.0
raw[away_ctrl] = 0.0

visible = np.isfinite(bx)
decisive = home_ctrl | away_ctrl
print()
print("owner votes over graded frames:")
print(f"  ball missing                    {np.mean(~visible[graded]):5.1%}")
print(f"  ball visible, clear control     {np.mean(decisive[graded]):5.1%}")
print(f"  ball visible, loose / contested {np.mean((visible & ~decisive)[graded]):5.1%}")

# ---- 3. dead-ball cues (unchanged from v3) ---------------------------------------
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


# ---- 4. build labels half by half ----------------------------------------------------
print()
labels = np.full(N_FRAMES, "X", dtype=object)

for half, start, end in blocks:
    sl = slice(start, end + 1)

    filled = pd.Series(raw[sl]).ffill().bfill()   # no clear vote -> keep previous label
    share = filled.rolling(WINDOW, center=True, min_periods=1).mean().to_numpy()
    owner = hysteresis(share)

    dead = clean_dead(dead_raw[sl])

    lab = np.where(owner == 1.0, "H", "A").astype(object)
    lab[dead] = "D"
    labels[sl] = lab

    n_poss, mean_poss = possession_stats(lab)
    d_starts, d_ends, d_vals = runs(dead)
    d_lens = (d_ends - d_starts)[d_vals] / FPS
    print(f"half {half}: D share {dead.mean():5.1%} | D segments {len(d_lens)} "
          f"(median {np.median(d_lens) if len(d_lens) else float('nan'):.1f} s) | "
          f"possessions {n_poss} (mean {mean_poss:.1f} s)")

# ---- 5. save ---------------------------------------------------------------------------
out = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels})
out.to_csv(OUT, index=False)
print()
print("written:", OUT, "| rows:", len(out))
print(out["label"].value_counts().to_string())

# ---- 6. dump per-frame cues for error analysis (data/ is git-ignored) -----------------
cues = pd.DataFrame({
    "frame": np.arange(N_FRAMES),
    "label": labels,
    "slow": cue_slow,
    "missing": cue_missing,
    "edge": cue_edge,
    "tag": cue_tag,
    "player_speed_smooth": pspeed_smooth.to_numpy(),
})
cues.to_csv(CUES_OUT, index=False)
print(f"written: {CUES_OUT} (per-frame cues for error_analysis.py)")
