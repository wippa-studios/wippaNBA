# Usage

> Full commands for both pipelines and the Flask app. Part of the [Legit NBA Predictions README](../README.md).

---

## A. NBA-data models

```bash
# 1. Download data
python scripts/download_nba_data.py                 # core, 11 seasons
python scripts/download_nba_data.py --stages core,extras   # + tracking, synergy, shot charts

# 2. Run the models (each is standalone)
python -m src.Models.baselines       # the bar everything else must beat
python -m src.Models.logreg_rolling
python -m src.Models.margin_total
python -m src.Models.gbm

# 3. Measure the things that actually matter (each prints a paired bootstrap
#    interval, never a bare point estimate)
python scripts/closing_line.py             # does any of this beat the closing line?
python scripts/availability.py             # is the availability effect attainable honestly?
python scripts/feature_block_ablation.py   # does each feature block earn its place?
python scripts/recency_followup.py         # does recency weighting help?
python scripts/edge_betting.py             # bet only where the model beats the price
```

## B. Original betting pipeline

**Quick start**

```bash
python3 main.py -xgb -odds=fanduel
```

| flag | effect |
|---|---|
| `-xgb` | run the XGBoost models |
| `-nn` | run the neural-network models |
| `-A` | run all models |
| `-kc` | add Kelly Criterion bankroll fractions |
| `-odds=<book>` | fetch odds (`fanduel`, `draftkings`, `betmgm`, `pointsbet`, `caesars`, `wynn`, `bet_rivers_ny`) |

> ⚠️ The committed model files predate every fix in this fork and must be retrained
> before use. See [Limitations](../README.md#limitations) and
> [How models were retrained and added](models_and_training.md#how-models-were-retrained-and-added).

**Data pipeline and training** (run from the repository root)

```bash
python -m src.Process-Data.Get_Data          # scrape daily team stats
python -m src.Process-Data.Get_Odds_Data      # scrape sportsbook odds + results
python -m src.Process-Data.Create_Games       # build the training dataset

python -m src.Train-Models.XGBoost_Model_ML   --dataset dataset_2012-26 --trials 15 --splits 5
python -m src.Train-Models.XGBoost_Model_UO   --dataset dataset_2012-26 --trials 15 --splits 5
```

Fill gaps with `--backfill` (optionally `--season 2025-26`).

`--trials` defaults to **15**, not 100. On this dataset the standard deviation of the
cross-validation score *across configurations* is several times the entire true
spread of held-out performance, so a bigger search ranks noise — and at 100 trials
the expected winner's curse is 2.50 standard errors. Every trainer now prints that
correction next to its own `Best val log loss`, so the number cannot be read without
it. See [ROADMAP.md](../ROADMAP.md) Level 6.2.

> **Note on directory names.** `Process-Data` and `Train-Models` contain hyphens and
> so are not importable as packages. All three of these work and are tested:
> `python -m src.Train-Models.XGBoost_Model_ML` ·
> `python src/Train-Models/XGBoost_Model_ML.py` ·
> `cd src/Train-Models && python -m XGBoost_Model_ML`.
> Renaming the directories is tracked as follow-up work.

## Flask app

```bash
export RAPIDAPI_KEY="<your rapidapi key>"   # optional; team/player endpoints 503 without it
cd Flask && flask run
```

Do **not** run `flask --debug run` on any reachable host: debug mode serves
application source, including any credentials, to unauthenticated visitors.

### `/games` — the NBA slate

```bash
curl 'http://127.0.0.1:5000/games'
curl 'http://127.0.0.1:5000/games?date=2026-10-21&tz=America/New_York'
```

Server-rendered from `ScoreboardV3`, **one upstream call per 60-second window**
shared across all viewers — the endpoint returns 403 under burst, so the page never
polls from the browser. `?date=` defaults to today; `?tz=` takes any IANA zone and
falls back to a `viewer_tz` cookie, then to UTC.

Three upstream traps this route absorbs, all confirmed against live payloads: nothing
on `ScoreboardV3` says which team is home (only `gameCode`, `YYYYMMDD/AWAYHOME`,
carries it, and `LineScore` orders rows **home first**); `gameEt` is Eastern
wall-clock with a `Z` on it, so only `gameTimeUTC` is used; and `no_games` is a true
answer, not a failure, so a 403, a dead upstream and a partial payload each get their
own status and their own sentence. **Known gap:** the in-progress branch
(`gameStatus == 2`) is modelled from the schema and was **not observed live**. 65
tests cover the route without touching the network; the full write-up is in
[docs/flask_games_route.md](flask_games_route.md).
