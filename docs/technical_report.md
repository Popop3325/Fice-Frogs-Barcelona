# Technical report — Ball possession per frame, BAR-ATH

Team: **Fice Frogs** — Lytvyn Oleksii, Telefus Artem, Khmeliuk Arina, Tkachuk Oleksii 
**Task:** for each of the 199,319 frames of the tactical video, label `H` / `A` / `D` / `X`.
**Repository:** https://github.com/Popop3325/Fice-Frogs-Barcelona

## 1. Summary

We built a transparent, rule-based pipeline on the provided automatic tracking data (ATD)
and the Smart Tagging file. It has two independent parts: an *owner* signal (which team
controls the ball) and a *dead-ball* detector, combined so that dead ball overrides owner.
The design is driven by the ranking metric, `spellF1`, which scores whole possessions: a
label that flickers between teams, or a spurious dead-ball inside a possession, splits one
true possession into several and costs far more than a handful of wrong frames.

On the public 5-minute window the final system reaches `spellF1` 0.53 and `macroF1` 0.75
(the organisers' reference `baseline_smooth50` scores 0.33 / 0.62 on the same window; its published full-match `spellF1` is 0.2433). We stress that
this window contains only 17 possessions, and that every difference between our variants
is inside the bootstrap noise reported by the scorer (section 6). The final choices were
therefore justified by structure and by checks on the whole match that need no ground
truth, not by the public score.

## 2. Data used

| asset | used | how |
|---|---|---|
| tactical ATD, positions | yes | ball and 46 player slots, converted to metres (105.3 × 68) |
| tactical ATD, player speeds | yes | mean speed of all tracked players, as a "players walking" cue |
| ATD team membership | yes | which slot belongs to the home or the away team |
| Smart Tagging, `SET PIECES` | yes | one of the four dead-ball cues |
| Smart Tagging, other codes and the `Team` tags | no | phases are not possessions, and `Team` is `N/A` for every throw-in; in the public window the phase teams were not consistent with the ground truth possession |
| tactical homography, tactical video | no | the task is solvable from the tracking; we did not run our own detector |
| broadcast assets | no | not synchronised to the tactical timeline |

The 5 minutes of public ground truth were used only to understand the behaviour of the
data and to check that the pipeline is correct. No hand annotation was made and no private
ground truth was used.

## 3. Method

### 3.1 Owner signal (H/A)

1. In each frame find the tracked player nearest to the ball; the team of that player is a
   vote (home = 1, away = 0). If the ball (41% of graded frames) or all players are
   missing, there is no vote.
2. Within each half, a missing vote keeps the previous one (forward fill, back fill at the
   start). This implements the rule that in loose-ball phases the team that last had
   possession keeps it.
3. The votes are averaged over a centred rolling window of 100 frames (4 s).
4. The averaged share goes through a hysteresis: the owner switches to home when the share
   exceeds 0.65 and to away when it drops below 0.35, otherwise it keeps its state.

Effect: a team must dominate roughly two thirds of a 4 s window to take possession, so a
single touch, a deflected pass or a short aerial duel does not change the owner. Short
true possessions (below about 2 s) cannot be recovered by construction.

### 3.2 Dead ball (D)

Four boolean cues are computed for every frame:

| cue | definition |
|---|---|
| `slow` | median over 5 s of the mean speed of all tracked players is below 1.5 m/s |
| `missing` | the ball is absent from the tracking for at least 2 s in a row |
| `edge` | the last known ball position is within 3 m of the pitch boundary (or outside it) |
| `tag` | a Smart Tagging `SET PIECES` interval is active |

Rule: **D = (`slow` AND (`missing` OR `edge` OR `tag`)) OR (`tag` AND `edge`)**.
Afterwards, gaps shorter than 2 s between D runs are closed and D runs shorter than 4 s
are dropped (the shortest dead-ball stretch in the public ground truth is 5.5 s).

The rule was designed from the public-window diagnostics (section 5): no single cue is a
reliable dead-ball detector (best precision about 0.56), but combinations that include
walking players are almost always dead ball, whereas "walking players" alone (48%) and
"ball missing near the edge while players run" (0%) are not.

### 3.3 Combination

Within each half, a frame is `D` if the dead-ball rule fires, otherwise `H` or `A` by the
owner signal. Frames outside the two halves are `X`. The half boundaries are the ones
published in the data description (first half frames 11185–90651, second half
110808–190197).

## 4. Possession rules implemented

* Dead ball is a stoppage, never a possession: it overrides the owner signal. A stretch
  `H D H` is therefore two possessions.
* While the ball is alive exactly one team is in possession; there is no neutral state.
* Loose ball or missing ball: the previous owner is kept.
* A change of possession needs sustained evidence (about two thirds of a 4 s window), which
  is our operational notion of "control". Single touches and deflections are ignored.
* No explicit model of restarts: after a dead ball the owner is decided by the same
  signal; we do not infer who takes the throw-in or the kick.

## 5. Experiments and what we dropped

All numbers are on the public 5-minute window (17 possessions) and are indicative only.

| version | owner | dead ball | spellF1 | macroF1 | possessions (true 17) |
|---|---|---|---|---|---|
| v1 | nearest player, majority vote 2 s | none | 0.314 | 0.504 | 34 |
| v2 | + hysteresis | `slow` alone is enough (score ≥ 2) | 0.244 | 0.739 | 29 |
| v3 | window 50 frames | `slow` needs a second cue | 0.340 | 0.723 | 30 |
| v4 | "clear control" votes only (≤ 3 m, ≥ 1 m ahead of the opponent) | as v3 | 0.339 | 0.705 | 29 |
| **final** | **window 100 frames** | **as v3** | **0.526** | **0.749** | **18** |

* **v1 → v2.** Adding a dead-ball detector raised `macroF1` from 0.50 to 0.74 but lowered
  `spellF1`. The error analysis showed the cause: 972 frames of live play were labelled `D`
  (precision 0.58), each one splitting a possession. On the whole match the rule labelled
  53% of the time as dead ball, against roughly 40% that we expect for a professional
  match (our own prior, not a measured value), and the "players walking"
  cue was active on 47.5% of graded frames (25% in the opening minutes): the pace of the
  game drops after the first minutes, so an absolute speed threshold alone does not
  transfer.
* **v2 → v3.** Requiring corroboration by a second cue raised dead-ball precision to 0.81
  (recall fell from 0.90 to 0.46) and brought the dead-ball share on the whole match to
  37.4% and 42.8% per half.
* **v4 (dropped).** Voting only when a player is clearly in control (≤ 3 m and ≥ 1 m ahead
  of the nearest opponent) is closer to the wording of the rules, but it decides on only
  33% of graded frames and shifts the possession boundaries; `spellF1` and the frame-level
  team accuracy did not improve (0.339 vs 0.340 and 0.827 vs 0.856).
* **Window sweep.** The possession analysis of v3 showed that 8 of 30 predicted possessions
  were wrong-team flickers of 0.2–3.9 s inside long true possessions. A longer window
  removes them. With the same dead-ball rule:

| window | possessions (public) | mean duration (true 14.2 s) | spellF1 | macroF1 | possessions (whole match) |
|---|---|---|---|---|---|
| 50 frames (2 s) | 30 | 8.9 s | 0.340 | 0.723 | 527 |
| 75 | 22 | 12.1 s | 0.451 | 0.739 | 411 |
| **100 (4 s)** | **18** | **14.8 s** | **0.526** | 0.749 | **361** |
| 125 | 16 | 16.7 s | 0.521 | 0.752 | 319 |

  We chose 100 frames because it matches the number and duration of possessions of the
  public ground truth and, extrapolating the public rate, the roughly 360 possessions
  expected over the whole match, while keeping the boundary shift smaller than at 125
  (mean latency +3 vs +6 frames). The 100 and 125 results are not distinguishable.

## 6. Results and the noise

| system, public 5 minutes | spellF1 | macroF1 |
|---|---|---|
| `baseline_raw` | 0.038 | 0.597 |
| `baseline_smooth25` | 0.287 | 0.620 |
| `baseline_smooth50` (organisers' reference) | 0.326 | 0.621 |
| `baseline_smooth125` | 0.424 | 0.581 |
| **final (window 100)** | **0.526** | **0.749** |

Final system in detail: dead-ball precision 0.81, recall 0.46, F1 0.58; 18 predicted
possessions against 17 true ones, 10 matched at IoU 0.5 (8 spurious, 7 missed); frame-level
team accuracy on frames where both ground truth and prediction are alive: 0.895.

The scorer's bootstrap (1000 draws over the 17 true possessions) states that gaps below
about 0.34 in `spellF1` are within noise on this window. The 5–95% interval of the final
system is 0.38–0.67 for `spellF1` and 0.53–0.86 for `macroF1`; for comparison, the
intervals of the window-50 system and of the organisers' `baseline_smooth50` are 0.23–0.44
and 0.21–0.45. The intervals overlap, so we do not claim a measured improvement over the
baselines on the full match; the public score served only to check that the pipeline is
correct.

## 7. Limitations

* **Dead-ball recall is 0.46.** Stoppages in which the ball stays visible and the only
  evidence is walking players are missed. In the public window this produced one 17 s
  spurious possession in place of a true dead ball. A rule "walking for 10 s or more is
  dead ball" would catch it, but late in the match the median player speed falls to
  1.2–1.3 m/s and walking is active more than half of the time, so we expect many false
  dead balls; we could not validate it and did not include it.
* **Absolute speed threshold.** 1.5 m/s was read from the opening minutes; the pace changes
  over the match. A threshold relative to the local pace is a natural improvement.
* **Dead-ball boundaries** are imprecise: the end of a dead ball lags the restart by a few
  seconds, and a dead-ball run can have a hole when the walking cue switches off.
* **Short true possessions** (about 1–3 s) are not recovered; they are about a third of the
  true possessions in the public window.
* **Tracking quality.** The ball is missing on 41% of graded frames; there are no player
  identities, so the analysis is collective.
* **Validation.** Only 5 minutes of ground truth were available, from the opening minutes
  of the first half; behaviour in other phases of the match is unverified.

## 8. Reproduction

```bash
python3 -m venv venv && source venv/bin/activate
python -m pip install -r requirements-final.txt
# unpack the match assets into ./data (not included in this repository)
python possession.py --out evaluation/results/submission.csv
python evaluation/validate_submission.py evaluation/results/submission.csv
```

The pipeline is deterministic. It was tested with Python 3.14.6, numpy 2.5.4, pandas 3.1.1
and kloppy 3.19.1 (pinned in `requirements-final.txt`), and the file it writes was
checked to be byte-identical (`cmp`) to the one used for the experiments in section 5.
The numbers in sections 5–6 are regenerated with `baseline_simple.py`, `baseline_v2.py`,
`baseline_v3.py`, `baseline_v4.py`, `baseline_v5.py`, `possession_analysis.py` and
`python evaluation/run_evaluation.py`.

## 9. Compliance

The labels are produced entirely by the algorithm in `possession.py`. No label was
annotated by hand. Only the published 5 minutes of ground truth were used, for diagnostics.
The match data is not included in the repository.
