"""A reusable leakage gate: does a feature know something it could not know yet?

WHY THIS MODULE EXISTS
----------------------
This repository's most instructive result is a **leak that truncation could not
see**. A player-Elo block scored |z| = 5.18 -- the largest effect ever measured
here -- by summing over the players who *actually played*. The weights were
honest trailing averages, so rebuilding with the future deleted left them
unchanged. The leak was in **membership**: who was in the room that night.

A gate built only on "drop the future and check nothing moves" cannot find that
class of bug. This module keeps the truncation probe, but separates three
questions that are usually conflated:

======================  ==================================================  ==========================
probe                   rebuild uses                                        what a change means
======================  ==================================================  ==========================
``as_of``               every record dated **on or before** the event    **a real leak.** The feature
                                                                        reads a later-dated record.
``blind``               records dated **strictly before** the event, plus  **a same-date dependency.**
                        the event's own row and any record on that date     Possibly a leak, possibly
                        the caller marks ``known_before``                  legitimate. Investigate.
``self`` (optional)     the same as ``blind`` but **without** the event's  the feature reads its own
                        own row                                            row. Expected for causal,
                                                                           self-referential features;
                                                                           diagnostic only.
======================  ==================================================  ==========================

The split matters. An earlier design used a single "blind" probe that removed
the probe event's own row while keeping everything earlier. That flags *every*
legitimate self-referential feature -- a trailing ``rolling(5).mean()`` loses its
last input by construction -- and trains its users to ignore it. Keeping the
event's own row in ``blind`` is the fairness fix: a feature that reads only
itself and earlier records is untouched, while a feature that reads *other*
records sharing its date (tonight's box scores, a same-day snapshot, the other
half of a doubleheader) still moves.

The report keeps the questions apart in the API, not only in prose:
``LeakageReport.leaks`` contains only ``as_of`` findings,
``LeakageReport.membership_dependencies`` contains ``blind``-only findings, and
``LeakageReport.self_dependencies`` contains the optional diagnostic. Do not
promote a membership dependency to a leak without reading it.

THE CONTRACT
------------
``build_features(frame)`` receives a pandas frame and returns a numeric feature
frame indexed by ``row_key``. The gate calls it on the full frame, then on one
truncated rebuild per probe event. Nothing else is required: no path, no NBA
data, no repository layout. It imports only numpy and pandas, so the same file
can be copied into another project or pointed at another pipeline.

The frame must carry:

* ``date_col`` -- the event date. Rows after the probe date are removed.
* ``row_key`` -- a unique identifier per row across the whole frame.
* ``known_before_col`` (optional) -- a boolean column marking records dated on
  the event date that were genuinely available before the first event that day.
  Without it, the blind rebuild keeps only the probe event's own row. With it,
  the caller can say "the schedule for tonight was known this morning" while
  leaving tonight's box scores and snapshots out.

The feature frame must be numeric (booleans are fine) and indexed by row keys
that exist in the input frame. A feature that is ``NaN`` in one rebuild and not
the other counts as a change: a NaN pattern that backfills is a finding, not a
rounding artefact.

WHAT IT CANNOT SEE
------------------
If a post-event outcome is stored as a **column on the event's own row**, the
blind probe keeps that row and cannot tell the column from a pre-game input. The
honest fixes are to model the outcome as a separate dated record (a snapshot or
box-score row), which is how the three known leaks here were built, or to use
``include_self_probe=True`` and treat the self finding as a question about which
of the row's own columns are knowable at prediction time. Do not describe a
clean gate as proof that a frame is leak-free: it answers the three questions
above, and no more.

Run the fixtures: ``pytest Tests/test_leakage_gate_artifact.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

__all__ = [
    "ColumnFinding",
    "LeakageReport",
    "ProbeExample",
    "audit_future_dependence",
]

VARIANTS = ("as_of", "blind", "self")

#: A tolerance for float noise. A real leak moves a feature by whole points;
#: the summation-order noise measured in this repository is ~1e-12. The default
#: sits three orders of magnitude above the noise and far below any real move.
DEFAULT_TOL = 1e-9


@dataclass(frozen=True)
class ProbeExample:
    """One probe event where a feature changed.

    ``full`` is the value built from the whole frame; ``probe`` is the value
    from the truncated rebuild. ``probe is None`` means the feature row itself
    was absent from the rebuild (the builder dropped the event entirely).
    """

    row_key: str
    date: str
    full: float | None
    probe: float | None


@dataclass(frozen=True)
class ColumnFinding:
    """What one feature did under one variant.

    ``variant`` is the only thing that decides how to read a finding. The
    ``is_leak`` property is deliberately true for ``as_of`` only; a ``blind``
    finding is a dependency to investigate and a ``self`` finding is a
    diagnostic, not a verdict.
    """

    feature: str
    variant: str
    probes: int
    changed: int
    max_abs_delta: float | None
    examples: tuple[ProbeExample, ...] = ()

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; expected one of {VARIANTS}")

    @property
    def is_leak(self) -> bool:
        """True only for the sound discriminator: a later-dated dependency."""
        return self.variant == "as_of"

    def describe(self) -> str:
        ratio = f"{self.changed}/{self.probes}"
        delta = "NaN transition" if self.max_abs_delta is None else f"{self.max_abs_delta:.6g}"
        return (f"{self.feature}: changed on {ratio} probes, "
                f"max |delta| {delta}")


@dataclass(frozen=True)
class LeakageReport:
    """The gate's answer, with the three questions kept apart.

    ``leaks`` are ``as_of`` findings: a feature moved when records after the
    event were removed, so it demonstrably reads the future. Trust these.

    ``membership_dependencies`` are ``blind``-only findings: the feature did not
    move under ``as_of``, but it moved when *other records sharing its date*
    were removed. That is the membership shape (tonight's actual squad), and it
    is also the shape of a same-day snapshot. It may be a leak or a legitimate
    pre-game input; investigate, do not conclude.

    ``self_dependencies`` are optional ``self`` findings: the feature moved when
    the event's own row was removed. A trailing average does this by
    construction. Diagnostic only.
    """

    leaks: tuple[ColumnFinding, ...]
    membership_dependencies: tuple[ColumnFinding, ...]
    self_dependencies: tuple[ColumnFinding, ...]
    probes: int
    features: int
    tol: float
    skipped: tuple[str, ...] = ()
    rows_available: int = 0

    @property
    def clean(self) -> bool:
        """No confirmed leak and no membership dependency to investigate.

        A blind-only finding is not proof of leakage, but it is not a clean bill
        of health either: the report is clean only when neither collection has
        anything in it. Use ``future_dependence_free`` for the narrower claim
        that nothing reads a later-dated record.
        """
        return (self.probes > 0
                and not self.leaks
                and not self.membership_dependencies)

    @property
    def future_dependence_free(self) -> bool:
        """The sound discriminator passed: no feature reads a later record."""
        return self.probes > 0 and not self.leaks

    @property
    def inconclusive(self) -> bool:
        """No probe could be evaluated, so the report proves nothing."""
        return self.probes == 0

    @property
    def needs_investigation(self) -> bool:
        return bool(self.membership_dependencies)

    @property
    def fully_audited(self) -> bool:
        return not self.skipped

    def summary(self) -> str:
        """A human-readable verdict that names the distinction.

        The summary is part of the API: #48's complaint was that a report
        presenting ``as_of`` and ``blind`` findings as equal sends the reader to
        the wrong conclusion. This text never calls a blind finding a leak.
        """
        lines = [
            f"Leakage gate: {self.rows_available} feature rows, "
            f"{self.probes} probe(s) over {self.probes} date(s), "
            f"{self.features} feature column(s), tol={self.tol:g}",
        ]

        if self.inconclusive:
            lines.append("Verdict: INCONCLUSIVE -- no probe could be rebuilt, "
                         "so nothing is proved.")
            for reason in self.skipped[:5]:
                lines.append(f"  skipped: {reason}")
            return "\n".join(lines)

        if self.leaks:
            lines.append(f"  as_of  (later records)       LEAKS: {len(self.leaks)}")
            for finding in self.leaks:
                lines.append(f"    - {finding.describe()}")
                for example in finding.examples[:2]:
                    lines.append(
                        f"        e.g. {example.row_key} ({example.date}): "
                        f"{_fmt(example.full)} -> {_fmt(example.probe)}"
                    )
        else:
            lines.append("  as_of  (later records)       leaks: none")

        if self.membership_dependencies:
            lines.append(
                "  blind  (same-date records)   membership dependencies: "
                f"{len(self.membership_dependencies)}"
            )
            for finding in self.membership_dependencies:
                lines.append(f"    - {finding.describe()}")
                for example in finding.examples[:2]:
                    lines.append(
                        f"        e.g. {example.row_key} ({example.date}): "
                        f"{_fmt(example.full)} -> {_fmt(example.probe)}"
                    )
        else:
            lines.append("  blind  (same-date records)   membership dependencies: none")

        if self.self_dependencies:
            lines.append(
                "  self   (own event row)       diagnostics: "
                f"{len(self.self_dependencies)} (expected for self-referential features)"
            )
            for finding in self.self_dependencies[:5]:
                lines.append(f"    - {finding.describe()}")

        if self.leaks:
            lines.append(
                "Verdict: LEAK CONFIRMED -- a feature moves when later-dated "
                "records are removed. Do not trust scores built on this frame."
            )
        elif self.membership_dependencies:
            lines.append(
                "Verdict: NO CONFIRMED LEAK, but "
                f"{len(self.membership_dependencies)} membership dependency(ies) "
                "to investigate. A blind finding is not proof of leakage."
            )
        else:
            lines.append(
                "Verdict: PASS on both probes -- no feature depends on later "
                "records or on other records sharing its date."
            )

        if self.skipped:
            lines.append(f"  {len(self.skipped)} probe(s) skipped; see report.skipped.")
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "absent"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.6g}"


def _validate_inputs(frame: pd.DataFrame, date_col: str, row_key: str,
                     known_before_col: str | None) -> tuple[pd.Series, pd.Series | None]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if row_key not in frame.columns:
        raise KeyError(f"row_key {row_key!r} is not a column of the frame")
    if date_col not in frame.columns:
        raise KeyError(f"date_col {date_col!r} is not a column of the frame")
    if known_before_col is not None and known_before_col not in frame.columns:
        raise KeyError(f"known_before_col {known_before_col!r} is not a column")

    keys = frame[row_key]
    if keys.isna().any():
        raise ValueError(f"row_key {row_key!r} contains missing values")
    if keys.duplicated().any():
        duplicates = keys[keys.duplicated()].astype(str).unique()[:3]
        raise ValueError(f"row_key {row_key!r} must be unique; duplicates: {list(duplicates)}")
    if frame.empty:
        raise ValueError("frame is empty, so there is nothing to audit")
    if not frame.index.is_unique:
        raise ValueError(
            "frame index must be unique: the gate aligns date masks by index, and "
            "a duplicated index would compare the wrong rows. Call reset_index()."
        )

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if dates.isna().any():
        bad = frame.loc[dates.isna(), row_key].astype(str).unique()[:3]
        raise ValueError(f"date_col {date_col!r} has unparseable dates for: {list(bad)}")

    known_before = None
    if known_before_col is not None:
        known_before = frame[known_before_col].fillna(False).astype(bool)
    return dates, known_before


def _validate_features(features: pd.DataFrame, frame: pd.DataFrame, row_key: str,
                       date_col: str, feature_cols: Sequence[str] | None) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("build_features must return a pandas DataFrame")
    if features.index.has_duplicates:
        duplicates = features.index[features.index.duplicated()].astype(str).unique()[:3]
        raise ValueError(f"build_features returned a duplicate index: {list(duplicates)} "
                         "the index must be the row_key")
    if features.empty:
        raise ValueError("build_features returned no rows; there is nothing to compare")

    known_keys = set(frame[row_key].tolist())
    unknown = [key for key in features.index if key not in known_keys]
    if unknown:
        raise ValueError(
            "build_features returned keys that are not in the input frame, so they "
            f"cannot be dated: {unknown[:3]}"
        )

    # Identifiers are not features. Drop the row key (when it is also a column)
    # and the date, then insist the rest is numeric: silently skipping an object
    # column would be the same class of guard that cannot tell "absent" from
    # "failed to look".
    drop = {row_key, date_col}
    candidates = [c for c in features.columns if c not in drop]

    if feature_cols is None:
        columns = candidates
    else:
        columns = list(feature_cols)
        missing = [c for c in columns if c not in features.columns]
        if missing:
            raise KeyError(f"feature_cols not found in build_features output: {missing}")
        if not columns:
            raise ValueError("feature_cols is empty")

    for column in columns:
        dtype = features[column].dtype
        if not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)):
            raise TypeError(
                f"feature {column!r} is {dtype}, not numeric. The gate compares "
                "numeric feature values; encode categories before auditing."
            )
    return features, columns


def _probe_keys(features: pd.DataFrame, dates_by_key: dict, n_probe: int, seed: int) -> list[str]:
    """One probe per distinct feature date, sampled deterministically.

    Probing every row is wasteful when a date carries a whole slate, and probing
    only one row per date still exercises the date cut. Rows are sorted by key
    before the first is taken, so the selection does not depend on frame order.
    """
    rows = pd.DataFrame({
        "key": list(features.index),
        "date": [dates_by_key[key] for key in features.index],
    })
    unique_dates = sorted(pd.to_datetime(rows["date"]).unique())
    n_probe = max(1, int(n_probe))
    rng = np.random.default_rng(seed)
    if n_probe < len(unique_dates):
        picked = rng.choice(np.asarray(unique_dates, dtype="datetime64[ns]"),
                            size=n_probe, replace=False)
        picked = sorted(pd.to_datetime(picked))
    else:
        picked = unique_dates

    keys: list[str] = []
    for date in picked:
        on_date = rows[pd.to_datetime(rows["date"]) == date].sort_values("key")
        keys.append(str(on_date.iloc[0]["key"]))
    return keys


def _changed_features(full_row: pd.Series, probe_row: pd.Series | None,
                      columns: Iterable[str], tol: float
                      ) -> list[tuple[str, float | None, float | None, float | None]]:
    """Compare one probe's feature vector, with the NaN pattern exact.

    Returns ``(feature, full_value, probe_value, abs_delta)`` for every change.
    ``probe_row is None`` means the builder dropped the event; every non-NaN
    feature then counts as changed, because the feature it produced depended on
    records the rebuild removed.
    """
    changed = []
    for column in columns:
        full = full_row[column]
        full_missing = bool(pd.isna(full))
        if probe_row is None:
            if not full_missing:
                changed.append((column, _as_float(full), None, None))
            continue
        probe = probe_row[column]
        probe_missing = bool(pd.isna(probe))
        if full_missing and probe_missing:
            continue
        if full_missing != probe_missing:
            changed.append((column, _as_float(full), _as_float(probe), None))
            continue
        delta = abs(float(full) - float(probe))
        if delta > tol:
            changed.append((column, _as_float(full), _as_float(probe), delta))
    return changed


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate(changes: list[tuple[str, str, str, float | None, float | None, float | None]],
               columns: Sequence[str], evaluated: int, variant: str) -> tuple[ColumnFinding, ...]:
    """Collapse per-probe changes into one finding per feature."""
    by_feature: dict[str, list[tuple[str, str, float | None, float | None, float | None]]] = {
        column: [] for column in columns
    }
    for feature, key, date, full, probe, delta in changes:
        by_feature[feature].append((key, date, full, probe, delta))

    findings = []
    for feature in columns:
        rows = by_feature[feature]
        if not rows:
            continue
        deltas = [row[4] for row in rows if row[4] is not None]
        findings.append(ColumnFinding(
            feature=feature,
            variant=variant,
            probes=evaluated,
            changed=len(rows),
            max_abs_delta=max(deltas) if deltas else None,
            examples=tuple(
                ProbeExample(row_key=key, date=date, full=full, probe=probe)
                for key, date, full, probe, _delta in rows[:3]
            ),
        ))
    return tuple(sorted(findings, key=lambda finding: finding.feature))


def audit_future_dependence(
    build_features: Callable[[pd.DataFrame], pd.DataFrame],
    frame: pd.DataFrame,
    date_col: str,
    row_key: str,
    n_probe: int = 200,
    *,
    known_before_col: str | None = None,
    feature_cols: Sequence[str] | None = None,
    tol: float = DEFAULT_TOL,
    seed: int = 0,
    include_self_probe: bool = False,
    self_keep_cols: Sequence[str] = (),
    on_error: str = "skip",
) -> LeakageReport:
    """Probe a feature builder for information it could not have known.

    Parameters
    ----------
    build_features:
        ``build_features(frame) -> DataFrame``. Must return one numeric row per
        event, indexed by ``row_key``. It is called on the full frame and then
        on one truncated frame per probe event.
    frame:
        The raw record universe. Any columns; rows are removed by date.
    date_col, row_key:
        The event date and a unique row identifier. ``row_key`` must also be the
        index of the feature frame.
    n_probe:
        Maximum number of distinct event dates to probe. One row per date is
        sampled; all dates are probed when there are fewer than ``n_probe``.
    known_before_col:
        Optional boolean column. Rows dated on the probe date with ``True`` are
        retained by the blind rebuild. Without it, only the probe event's own
        row survives that date. Mark here what was genuinely available before
        the first event of the day -- never tonight's box scores or snapshots.
    feature_cols:
        Restrict the audit to these columns. By default every numeric column the
        builder returns, excluding the row key and date columns.
    tol:
        Absolute tolerance for float noise. The NaN pattern is always compared
        exactly.
    seed:
        Sampling seed for the probe dates. The same frame and seed produce the
        same probes.
    include_self_probe:
        Also run the diagnostic that replaces the event's own row with a stub
        (all columns null except the keys and ``self_keep_cols``). Findings land
        in ``report.self_dependencies`` and are never called leaks.
    self_keep_cols:
        Columns the stub keeps, typically the identifiers a builder needs to
        recognise the event (``home``, ``away``, ``record_type``). Everything
        else on the probe row is nulled.
    on_error:
        ``"skip"`` records a probe whose rebuild raised and continues; ``"raise"``
        propagates. Skipped probes are named in ``report.skipped`` and in the
        summary, and an all-skipped audit is ``INCONCLUSIVE`` rather than clean.

    Returns
    -------
    LeakageReport
        Read ``.leaks`` first. ``.membership_dependencies`` is a separate
        question, and ``.summary()`` says so in words.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError("on_error must be 'skip' or 'raise'")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    missing_keep = [c for c in self_keep_cols if c not in frame.columns]
    if missing_keep:
        raise KeyError(f"self_keep_cols not found in the frame: {missing_keep}")

    dates, known_before = _validate_inputs(frame, date_col, row_key, known_before_col)

    work = frame.copy(deep=False)
    full = build_features(frame.copy())
    full, columns = _validate_features(full, frame, row_key, date_col, feature_cols)

    dates_by_key = dict(zip(frame[row_key].tolist(), dates.tolist()))
    probes = _probe_keys(full, dates_by_key, n_probe, seed)

    changes: dict[str, list] = {"as_of": [], "blind": [], "self": []}
    evaluated = 0
    skipped: list[str] = []

    def _evaluate(variant: str, key: str, date: pd.Timestamp,
                  subset: pd.DataFrame) -> list[tuple]:
        rebuilt = build_features(subset)
        _, rebuilt_columns = _validate_features(
            rebuilt, subset, row_key, date_col, feature_cols
        )
        if list(rebuilt_columns) != list(columns):
            raise ValueError(
                f"build_features changed its feature columns on a truncated rebuild "
                f"for {key!r}: {rebuilt_columns!r} != {columns!r}"
            )
        probe_row = None
        if key in rebuilt.index:
            probe_row = rebuilt.loc[key]
            if isinstance(probe_row, pd.DataFrame):
                raise ValueError(f"build_features returned duplicate rows for {key!r}")
        return [
            (feature, key, date.date().isoformat(), full_value, probe_value, delta)
            for feature, full_value, probe_value, delta in _changed_features(
                full.loc[key], probe_row, columns, tol
            )
        ]

    for key in probes:
        date = pd.Timestamp(dates_by_key[key])
        same_date = dates == date
        earlier = dates < date
        probe_row_mask = work[row_key] == key

        as_of = work.loc[earlier | same_date]
        blind_mask = earlier | probe_row_mask
        if known_before is not None:
            blind_mask = blind_mask | (same_date & known_before)
        blind = work.loc[blind_mask]

        frames = [("as_of", as_of), ("blind", blind)]
        if include_self_probe:
            # A stub, not a deletion: deleting the row makes every builder that
            # emits one row per event look dependent on its own row. The stub
            # keeps the event recognisable and nulls what the feature might read,
            # so a change means the feature really did read one of those columns.
            base_mask = earlier
            if known_before is not None:
                base_mask = base_mask | (same_date & known_before)
            stub = work.loc[probe_row_mask].copy()
            keep = {row_key, date_col}
            if known_before_col is not None:
                keep.add(known_before_col)
            keep.update(self_keep_cols)
            for column in stub.columns:
                if column not in keep:
                    stub[column] = np.nan
            self_frame = pd.concat([work.loc[base_mask], stub], axis=0)
            frames.append(("self", self_frame))

        probe_changes: list[tuple[str, list[tuple]]] = []
        for variant, subset in frames:
            try:
                probe_changes.append((variant, _evaluate(variant, key, date, subset)))
            except Exception as exc:  # noqa: BLE001 - the choice is explicit below
                if on_error == "raise":
                    raise
                skipped.append(f"{variant} probe {key!r} ({date.date()}): "
                               f"{type(exc).__name__}: {exc}")
                probe_changes = []
                break

        # Commit only when every requested variant for this probe built. A probe
        # that failed halfway must not contribute half a finding to a count that
        # says it was never evaluated.
        if probe_changes:
            for variant, rows in probe_changes:
                changes[variant].extend(rows)
            evaluated += 1

    leaks = _aggregate(changes["as_of"], columns, evaluated, "as_of")
    blind = _aggregate(changes["blind"], columns, evaluated, "blind")
    self_findings = _aggregate(changes["self"], columns, evaluated, "self")

    leak_features = {finding.feature for finding in leaks}
    membership = tuple(f for f in blind if f.feature not in leak_features)
    membership_features = {finding.feature for finding in membership}
    self_only = tuple(
        f for f in self_findings
        if f.feature not in leak_features and f.feature not in membership_features
    )

    return LeakageReport(
        leaks=leaks,
        membership_dependencies=membership,
        self_dependencies=self_only,
        probes=evaluated,
        features=len(columns),
        tol=float(tol),
        skipped=tuple(skipped),
        rows_available=len(full),
    )
