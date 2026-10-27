# Module map: `src/` to `src/wippa_nba`

This document records the package reorganisation performed on branch
`reorg/takeover`, based on commit `d7082842db7e3b2a86051d16ee27774f544eff31`.

The source root is still `src/`. It is **not** a Python package: `src/__init__.py`
was deleted, and the only importable top-level package under it is `wippa_nba`.
`Tests/`, `scripts/`, `results/`, and `docs/` were not moved.

## Final layout

```
src/
└── wippa_nba/
    ├── betting/          Kelly sizing, expected value
    ├── data/             data access, providers, feature builders, dictionaries
    ├── evaluation/       leakage gate, metrics, validation statistics
    ├── features/         the shared feature contract (drop lists and order)
    ├── legacy/
    │   ├── process_data/ the historical scrape/build pipeline
    │   └── train_models/ the historical trainers and training helpers
    ├── models/           model implementations used by analysis and inference
    ├── predict/          inference runners invoked by main.py
    └── web/              the Flask application, templates, and static assets
```

Every directory above has an `__init__.py`. The package has no import-time side
effects beyond the module-level setup each file already performed.

## Old path to new path

One row per moved file. Every file below was moved with `git mv`, preserving
history.

| Old path | New path |
|---|---|
| `Flask/app.py` | `src/wippa_nba/web/app.py` |
| `Flask/static/app.js` | `src/wippa_nba/web/static/app.js` |
| `Flask/static/styles.css` | `src/wippa_nba/web/static/styles.css` |
| `Flask/templates/games.html` | `src/wippa_nba/web/templates/games.html` |
| `Flask/templates/index.html` | `src/wippa_nba/web/templates/index.html` |
| `Flask/templates/player_modal.html` | `src/wippa_nba/web/templates/player_modal.html` |
| `Flask/templates/team_modal.html` | `src/wippa_nba/web/templates/team_modal.html` |
| `src/DataProviders/SbrOddsProvider.py` | `src/wippa_nba/data/SbrOddsProvider.py` |
| `src/DataProviders/feat_squad_continuity.py` | `src/wippa_nba/data/feat_squad_continuity.py` |
| `src/DataProviders/feat_squad_quality.py` | `src/wippa_nba/data/feat_squad_quality.py` |
| `src/DataProviders/feat_team_elo.py` | `src/wippa_nba/data/feat_team_elo.py` |
| `src/DataProviders/injury_features.py` | `src/wippa_nba/data/injury_features.py` |
| `src/DataProviders/nba_data.py` | `src/wippa_nba/data/nba_data.py` |
| `src/Models/__init__.py` | `src/wippa_nba/models/__init__.py` |
| `src/Models/baselines.py` | `src/wippa_nba/models/baselines.py` |
| `src/Models/dynamic_ratings.py` | `src/wippa_nba/models/dynamic_ratings.py` |
| `src/Models/ensemble.py` | `src/wippa_nba/models/ensemble.py` |
| `src/Models/gbm.py` | `src/wippa_nba/models/gbm.py` |
| `src/Models/hierarchical_ratings.py` | `src/wippa_nba/models/hierarchical_ratings.py` |
| `src/Models/logreg_rolling.py` | `src/wippa_nba/models/logreg_rolling.py` |
| `src/Models/margin_total.py` | `src/wippa_nba/models/margin_total.py` |
| `src/Models/measurement.py` | `src/wippa_nba/models/measurement.py` |
| `src/Models/nn_rolling.py` | `src/wippa_nba/models/nn_rolling.py` |
| `src/Models/pace_first.py` | `src/wippa_nba/models/pace_first.py` |
| `src/Models/player_ratings.py` | `src/wippa_nba/models/player_ratings.py` |
| `src/Models/rapm.py` | `src/wippa_nba/models/rapm.py` |
| `src/Models/ratings.py` | `src/wippa_nba/models/ratings.py` |
| `src/Predict/NN_Runner.py` | `src/wippa_nba/predict/NN_Runner.py` |
| `src/Predict/XGBoost_Runner.py` | `src/wippa_nba/predict/XGBoost_Runner.py` |
| `src/Process-Data/Add_Days_Rest.py` | `src/wippa_nba/legacy/process_data/Add_Days_Rest.py` |
| `src/Process-Data/Create_Games.py` | `src/wippa_nba/legacy/process_data/Create_Games.py` |
| `src/Process-Data/Create_Recency_Features.py` | `src/wippa_nba/legacy/process_data/Create_Recency_Features.py` |
| `src/Process-Data/Fix_Odds_Date_Format.py` | `src/wippa_nba/legacy/process_data/Fix_Odds_Date_Format.py` |
| `src/Process-Data/Get_Data.py` | `src/wippa_nba/legacy/process_data/Get_Data.py` |
| `src/Process-Data/Get_Injury_Data.py` | `src/wippa_nba/legacy/process_data/Get_Injury_Data.py` |
| `src/Process-Data/Get_Odds_Data.py` | `src/wippa_nba/legacy/process_data/Get_Odds_Data.py` |
| `src/Process-Data/Get_Odds_Line_Movement.py` | `src/wippa_nba/legacy/process_data/Get_Odds_Line_Movement.py` |
| `src/Process-Data/__init__.py` | `src/wippa_nba/legacy/process_data/__init__.py` |
| `src/Train-Models/Logistic_Regression_ML.py` | `src/wippa_nba/legacy/train_models/Logistic_Regression_ML.py` |
| `src/Train-Models/Logistic_Regression_UO.py` | `src/wippa_nba/legacy/train_models/Logistic_Regression_UO.py` |
| `src/Train-Models/NN_Model_ML.py` | `src/wippa_nba/legacy/train_models/NN_Model_ML.py` |
| `src/Train-Models/NN_Model_UO.py` | `src/wippa_nba/legacy/train_models/NN_Model_UO.py` |
| `src/Train-Models/XGBoost_Model_ML.py` | `src/wippa_nba/legacy/train_models/XGBoost_Model_ML.py` |
| `src/Train-Models/XGBoost_Model_UO.py` | `src/wippa_nba/legacy/train_models/XGBoost_Model_UO.py` |
| `src/Train-Models/__init__.py` | `src/wippa_nba/legacy/train_models/__init__.py` |
| `src/Train-Models/boosting_training.py` | `src/wippa_nba/legacy/train_models/boosting_training.py` |
| `src/Train-Models/logistic_training.py` | `src/wippa_nba/legacy/train_models/logistic_training.py` |
| `src/Train-Models/neural_training.py` | `src/wippa_nba/legacy/train_models/neural_training.py` |
| `src/Train-Models/training_common.py` | `src/wippa_nba/legacy/train_models/training_common.py` |
| `src/Train-Models/training_reporting.py` | `src/wippa_nba/legacy/train_models/training_reporting.py` |
| `src/Utils/Dictionaries.py` | `src/wippa_nba/data/Dictionaries.py` |
| `src/Utils/Expected_Value.py` | `src/wippa_nba/betting/Expected_Value.py` |
| `src/Utils/Kelly_Criterion.py` | `src/wippa_nba/betting/kelly.py` |
| `src/Utils/features.py` | `src/wippa_nba/features/contract.py` |
| `src/Utils/leakage_gate.py` | `src/wippa_nba/evaluation/leakage_gate.py` |
| `src/Utils/metrics.py` | `src/wippa_nba/evaluation/metrics.py` |
| `src/Utils/nba_client.py` | `src/wippa_nba/data/nba_client.py` |
| `src/Utils/tools.py` | `src/wippa_nba/data/tools.py` |
| `src/Utils/validation.py` | `src/wippa_nba/evaluation/validation.py` |

### Renames that are not a simple directory move

- `src/Utils/features.py` -> `src/wippa_nba/features/contract.py`. The module is
  the feature **contract** (drop lists, order validation), not a feature
  builder; the leaf name states that.
- `src/Utils/Kelly_Criterion.py` -> `src/wippa_nba/betting/kelly.py`. The module
  keeps the same public names; importers that used `Kelly_Criterion` now import
  `kelly`, aliased where the old name was used.
- `src/Utils/Expected_Value.py` -> `src/wippa_nba/betting/Expected_Value.py`.
  The file name is kept because tests and callers refer to the module by that
  name.
- `src/Utils/Dictionaries.py`, `src/Utils/tools.py`, `src/Utils/nba_client.py`
  -> `src/wippa_nba/data/`. They are data-access helpers, and their names are
  kept.
- `src/Predict/*` -> `src/wippa_nba/predict/*`. `Predict/` was not in the
  requested mapping; these are inference entry points used by `main.py` and the
  Flask app, not part of the historical build pipeline, so they form their own
  package rather than joining `legacy/`.
- `src/Process-Data/*` -> `src/wippa_nba/legacy/process_data/*` and
  `src/Train-Models/*` -> `src/wippa_nba/legacy/train_models/*`. The hyphens are
  gone, so both directories are ordinary importable packages now.
- `Flask/app.py`, `Flask/templates/*`, `Flask/static/*` ->
  `src/wippa_nba/web/*`. The Flask app still launches `main.py` as a subprocess;
  only its repository-root calculation changed (two `dirname` calls became four).

## Placement decisions

The requested mapping was followed exactly where it named a destination:

| Source | Requested destination | Result |
|---|---|---|
| `Utils/leakage_gate.py`, `Utils/metrics.py`, `Utils/validation.py` | `evaluation/` | as requested |
| `Utils/features.py` | `features/<chosen>` | `features/contract.py` (see above) |
| `Utils/Kelly_Criterion.py` | `betting/kelly.py` | as requested |
| `Models/*` | `models/*` | as requested |
| `DataProviders/*` | `data/*` | as requested |
| `Process-Data/*` | `legacy/process_data/*` | as requested |
| `Train-Models/*` | `legacy/train_models/*` | as requested |
| `Flask/app.py` + templates/static | `web/` | as requested |

Modules the mapping did not name were placed by dependency and role:
`Expected_Value.py` -> `betting/`; `Dictionaries.py`, `tools.py`,
`nba_client.py` -> `data/`; `Predict/*` -> `predict/`.

## Dependency direction

The resulting internal dependency graph is acyclic:

```
data            (no internal imports except Dictionaries)
models        -> data
evaluation      (standalone)
features      -> legacy.train_models -> evaluation
predict       -> evaluation, betting, features, legacy.train_models
web           -> subprocess only (main.py); no Python import of the package
```

`features/contract.py` imports `drop_columns_for` from
`legacy.train_models.training_common`. This is deliberate: the contract must be
the trainers' own drop list, not a copy. It is also the one inverted dependency
in the tree -- a shared contract package depends on a legacy trainer module.
Splitting `drop_columns_for` into a neutral module would remove that edge and is
left as follow-up work.

## Import and bootstrap changes

The migration rewrote, across `src/`, `Tests/`, `scripts/`, `conftest.py`,
`main.py`, and the moved Flask app:

- `from src.X import ...` and `import src.X` forms;
- dotted module paths inside strings, docstrings, `importlib`, `runpy`, and
  subprocess snippets;
- bare `from DataProviders ...` / `from Models ...` imports used by tests and
  scripts that put `src/` on `sys.path`;
- flat `import training_common` / `from training_reporting import ...` imports
  used by the trainers and their tests;
- `Path` expressions and `__file__`-relative depth calculations.

Per-rule substitution counts are in the migration's `.migration_report.json`
(not committed). The counts that matter for review:

| Rule | Substitutions |
|---|---|
| `from src.Utils import NAME` | 19 |
| dotted module path rewrites | 145 |
| bare `DataProviders` / `Models` imports | 100 |
| flat `training_*` imports | 48 |
| path-expression rewrites | 39 |
| `__file__` depth rewrites | 19 |
| explicit `sys.path` bootstraps | 17 |
| quoted-string / subprocess fixes | 5 |

Because `wippa_nba` now lives one level below the old modules, every entry point
that used to add the repository root adds `src/` instead:

- `conftest.py` inserts both `src/` and the repository root; `pyproject.toml`
  mirrors that in `pythonpath`.
- `main.py` inserts `src/` before importing the package, because Flask launches
  it by absolute path.
- The six trainer entry points insert `src/` (three levels above
  `legacy/train_models`).
- `training_common.py` inserts `src/` rather than the repository root.
- The Process-Data scripts insert `BASE_DIR / "src"`; `Create_Games.py`, which
  had no bootstrap, gained one.
- `scripts/check_feature_alignment.py` and `scripts/evaluate.py` insert `src/`;
  the former also keeps the repository root for `import main`.
- `src/wippa_nba/web/app.py` computes the repository root with four `dirname`
  calls instead of two.

`pyproject.toml` now declares a src layout:

```
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

The distribution name, dependencies, and entry points are unchanged. Flask
templates and static files are declared as package data for `wippa_nba.web`.

## Verification

Run from the repository root with the project environment:

```
python -m compileall -q src
python -m pytest tests/ -q
```

Results, both measured with `Data/OddsData.sqlite` absent because that zero-byte
file is an ignored test artefact whose presence changes the `test_closing_line`
skips and one failure:

| Run | Failed | Passed | Skipped | Subtests | Collected | Exit |
|---|---|---|---|---|---|---|
| Before the move | 9 | 1368 | 175 | 92 | 1552 | 1 |
| After the move | 9 | 1368 | 175 | 92 | 1552 | 1 |

The nine failures are the same test IDs in both runs, and all are pre-existing
missing-data failures (`test_edge_search`, `test_odds_table_resolution`, and
`test_training_layer::TestDatasetTableAgreement`). No previously passing test
fails after the move, no test was deleted, and the skip count did not change.
`python -m compileall -q src` exits 0.

`setuptools.find_packages(where="src")` returns 11 packages: `wippa_nba` and its
ten subpackages, including both `legacy.*` packages.

## Deliberately out of scope

- No test was deleted, skipped, or rewritten to hide a move-induced failure.
- No dependency, version, distribution name, or public entry point changed.
- `README.md` upstream credit was not touched.
- Stale comments in a few modules still describe the old hyphenated layout; the
  code and this map are the source of truth.
