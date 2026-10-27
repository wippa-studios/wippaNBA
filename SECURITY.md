# Security policy

This repository is a fork of
[kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)
by Kyle Skompinski and contributors. This policy covers this fork only.

## In scope

- The package code under `src/wippa_nba/`, including the Flask app in
  `src/wippa_nba/web/` (server-side template handling, deserialisation of model
  artifacts, any file or path handling).
- The root entry point `main.py` and the utility scripts under `scripts/`.

## Out of scope

- Third-party data sources the pipeline reads (NBA stats APIs, odds sites) and
  the content they serve.
- The archived `Legit-NBA-Predictions` repository — a separate project; report
  anything found there to that repository's own maintainers.
- Betting outcomes and the accuracy or profitability of the models themselves;
  those are research questions, not security vulnerabilities.

## How to report

Use [GitHub's private vulnerability reporting](https://github.com/wippa-studios/wippaNBA/security/advisories/new)
(Security → Report a vulnerability). Please do not open a public issue for
something you believe is exploitable. Include the affected path, the exact
invocation that triggers it, and your Python version.

## What to expect

This repository stores no credentials, no datasets and no personal data:
`Data/` contains only a provenance manifest, and secrets belong in the
gitignored `.env` / `local_settings.py` local-override files. The expected
severity surface is therefore small. Reports are handled on a best-effort basis
by the maintainers; there is no guaranteed response time or SLA.
