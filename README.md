# Ball possession per frame — BAR-ATH (More than a Hack 2027, Phase 1)

Team: **Fice Frogs** — Lytvyn Oleksii, Telefus Artem, Khmeliuk Arina, Tkachuk Oleksii 

For every video frame of the tactical recording of FC Barcelona vs Athletic Club,
`possession.py` writes one label:

| label | meaning |
|---|---|
| `H` | home team (Barcelona) in possession, ball alive |
| `A` | away team (Athletic) in possession, ball alive |
| `D` | dead ball |
| `X` | outside play (before kick-off, half-time, after the final whistle) |

The system is rule-based and transparent. It uses the provided automatic tracking data
(ATD) and the Smart Tagging file, and it reads no ground truth of any kind.
The method, the experiments and the limitations are described in
[`docs/technical_report.md`](docs/technical_report.md).

> This repository started as a copy of the organisers' challenge repository
> (`metrica-sports/more-than-a-hack-2027`, MIT licence). Their original README is kept as
> [`README_challenge.md`](README_challenge.md). `games.py` and the `evaluation/` scorer are
> theirs, unchanged.

## Results at a glance

Scored on the **public 5 minutes** of ground truth (17 possessions) with the organisers'
scorer. These numbers are indicative only: the scorer's own bootstrap puts differences
below about 0.34 `spellF1` inside the noise on this window, so we make no claim about the
full match.

| system | spellF1 (ranking metric) | macroF1 |
|---|---|---|
| organisers' `baseline_smooth50` | 0.326 | 0.621 |
| **this submission** (`possession.py`) | **0.526** | **0.749** |

## Method in short

* **Owner (H/A).** In every frame the team of the tracked player nearest to the ball is a
  vote. Frames without a ball give no vote and keep the previous label. The votes are
  averaged over a 4 s rolling window and passed through a hysteresis (switch to home above
  0.65, to away below 0.35), so a single touch or a deflection does not change possession.
* **Dead ball (D).** Voting between four cues: players walking, ball missing for 2 s or
  more, ball at the pitch edge, and a Smart Tagging `SET PIECES` interval. Walking players
  alone are not enough; they need a second cue.
* **Outside the two halves:** `X`.

## Reproduce the submission file

Tested with Python 3.14.6 and the versions pinned in `requirements-final.txt`
(numpy 2.5.4, pandas 3.1.1, kloppy 3.19.1). Use a recent Python (3.11 or newer).

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements-final.txt
```

### Data

The match assets are **not** part of this repository and must not be redistributed.
Unpack the bundle you received at enrolment into `data/` (git-ignored), or point
`MTAH_DATA_DIR` at the folder that contains `tactical/`. Only these files are read:

```
tactical/BAR-ATH_tactical_atd.xml
tactical/BAR-ATH_tactical_atd.txt
tactical/BAR-ATH_tactical_smart_tagging.xml
```

### Run

```bash
python possession.py --out evaluation/results/submission.csv
python evaluation/validate_submission.py evaluation/results/submission.csv
python evaluation/run_evaluation.py        # scores against the public 5 minutes
```

The run is deterministic: the same data gives the same file. The output has the header
`frame,label` and 199,319 rows (frames 0 to 199318). It is uploaded to the competition
form as `<team_name>.csv`.

## Repository contents

| file | purpose |
|---|---|
| `possession.py` | **the pipeline**; produces the submission file |
| `docs/technical_report.md` | technical document |
| `requirements-final.txt` | pinned dependencies |
| `baseline_simple.py` | v1: nearest player and majority vote, no dead ball |
| `baseline_v2.py`, `baseline_v3.py` | dead-ball rule, first without and then with corroboration |
| `baseline_v4.py` | owner by "clear control" (dropped, see the report) |
| `baseline_v5.py` | sensitivity sweep of the owner window (50/75/100/125 frames) |
| `inspect_public_gt.py`, `d_features.py` | diagnostics on the public 5-minute ground truth |
| `error_analysis.py`, `possession_analysis.py` | error analysis of the dead-ball rule and of the possessions |
| `games.py`, `evaluation/`, `starter/` | organisers' code (unchanged) |

The experiment scripts are kept so that the numbers in the technical report can be
regenerated; the submission file comes from `possession.py` only. Run every script from
the repository root, next to `games.py`.

## Rules compliance

* The labels are produced entirely by the algorithm; nothing was annotated by hand.
* Only the published 5 minutes of ground truth were used, and only for diagnostics.
* No match data and no derived data (tracking exports, clips) are included here.

## Licence

The code is released under the MIT licence (see [`LICENSE`](LICENSE)). The match data is
licensed separately by the organisers, for this competition only.
