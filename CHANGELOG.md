# Changelog

All notable changes to this repository are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file was added during the 2026 repository takeover. The project's history
predates this file, so earlier changes are recorded in `git log` only — this
file does not attempt to reconstruct them.

## [Unreleased]

### Changed

- Repository history migrated to a code-only history: datasets and trained
  model artifacts are no longer committed. `Data/` now contains only
  `manifest.json`, which records the provenance of the dataset a contributor is
  expected to build locally.
- Application code moved into a single installable package at
  `src/wippa_nba/` — models, data providers, evaluation, features, betting, the
  legacy pipeline and the Flask web app. The old `src/Process-Data`,
  `src/Train-Models`, `src/Utils` and `src/Predict` layouts are gone.
- Test suite reorganised into `tests/unit`, `tests/integration` and
  `tests/regression`, with dataset-dependent tests skipping explicitly when
  data is absent.

### Fixed

- Published test counts corrected: `docs/test_suite.md` reported 1,104 passing;
  the suite as measured on this branch is 1,368 passed and 184 skipped
  (`python -m pytest tests/ -q`, 0 failed).

---

This repository is a fork of
[kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)
by Kyle Skompinski and contributors.
