"""Validation-discipline utilities for the NBA outcome models.

This module exists to make two methodological defects auditable rather than
implicit:

1. ``TimeSeriesSplit(n_splits=5)`` is used with no embargo. Consecutive NBA games
   share team state, so a validation fold's rows are near-duplicates of the
   immediately preceding training rows and the fold is not independent.
   ``purged_walk_forward_split`` fixes this by purging any training row that
   shares a calendar date with a test row, and additionally embargoing the rows
   immediately before the test window.

2. Every trainer runs a hyperparameter search (100 trials for XGBoost) and
   reports the BEST trial's score on one held-out slice. That is a winner's-curse
   statistic: the reported number grows with the number of trials tried even when
   nothing improves. ``selection_bias_estimate``, ``deflated_sharpe_ratio`` and
   ``report_selection_risk`` quantify how much of a headline number can be
   selection noise.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not fix the trainers. Nothing here is wired into
``src/Train-Models/*``; those files are owned elsewhere. This module is
standalone and currently unused, which is expected.

ASSUMPTIONS THAT ARE MADE (and the ones that are NOT)
-----------------------------------------------------
MADE:
  * Rows are exchangeable within a fold but NOT across adjacent days. This is the
    entire justification for the embargo, and it is an assumption about the data,
    not a fact that this module can verify.
  * Same-day games are not independent. Two games on one calendar date are
    treated as one correlated event for purging purposes.
  * The reported metric is compared against an estimate whose standard error is
    ``standard_error``; the selection statistics are expressed in those units.
  * Under the null hypothesis, each configuration's score is approximately
    Gaussian and INDEPENDENT of the others.

NOT MADE (these are real limitations, not oversights):
  * Independence of trials is assumed and almost never true. Trials evaluated on
    the same held-out slice share that slice's noise, so the correction below is
    optimistic when trials are correlated. More trials than are genuinely
    independent means the true inflation is SMALLER than estimated here.
  * The variance of the per-trial score under the null is unknown. Everything
    here assumes it is exactly 1.0 in standard-error units, which is only right
    if the caller passes a metric whose null distribution is standard normal.
    Passing an unusual metric (a heavy-tailed ROI, say) silently mis-scales the
    correction.
  * Nothing here is a guarantee or an exact result. Every statistic in this
    module is an APPROXIMATION whose error is quantified in the docstrings and
    in ``Tests/test_validation.py``. An approximation that reads like a proof is
    worse than no correction at all.

STYLE
-----
Plain functions, no type annotations, no dataclasses, no logging -- matching the
rest of this repository. ``print()`` is used only by ``report_selection_risk``,
whose entire job is to produce a human-readable summary.
"""

import numpy as np
import pandas as pd

# Euler-Mascheroni constant, used by the expected-maximum-of-N-normals
# approximation in Bailey & Lopez de Prado's deflated Sharpe ratio.
EULER_MASCHERONI = 0.5772156649015329

SECONDS_PER_DAY = 86400.0


def normal_quantile(p):
    """Inverse standard-normal CDF (the ``Phi^-1`` used throughout this module).

    Implemented with Wichura's algorithm AS241 (the double-precision rational
    approximation used by the R ``qnorm`` implementation) rather than
    ``scipy.stats.norm.ppf``. That is deliberate: this module is required to add
    no new third-party dependencies, and scipy is not one of them.

    ASSUMPTIONS / ACCURACY:
      * Accurate to roughly 1e-15 relative in the central region. It is exact
        enough that every statistic below is limited by its own modelling
        assumptions, not by this function.
      * ``p == 0`` returns ``-inf`` and ``p == 1`` returns ``+inf``, which is the
        correct limit rather than an error.

    Raises ValueError for ``p`` outside [0, 1] or non-finite input.
    """
    p_array = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p_array)) and not np.all(
        np.isinf(p_array) & ((p_array == -np.inf) | (p_array == np.inf))
    ):
        raise ValueError("normal_quantile: p must be finite and within [0, 1]")
    if np.any(p_array < 0.0) or np.any(p_array > 1.0):
        raise ValueError("normal_quantile: p must be within [0, 1]")

    p_flat = p_array.ravel()
    out = np.empty(p_flat.shape, dtype=float)
    out[:] = np.nan

    out[p_flat == 0.0] = -np.inf
    out[p_flat == 1.0] = np.inf

    # AS241 rational-approximation coefficients.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    # Rational approximations are used below the 0.02425 and above the 0.97575
    # tails, where the polynomial form loses accuracy. p == 0 and p == 1 are
    # EXCLUDED from both masks: they were already assigned the exact +/-inf
    # limits above, and feeding 0 into log() would overwrite that with NaN.
    low_cut = 0.02425
    high_cut = 1.0 - low_cut
    interior = (p_flat > 0.0) & (p_flat < 1.0)
    low = interior & (p_flat < low_cut)
    high = interior & (p_flat > high_cut)
    middle = interior & ~(low | high)

    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p_flat[low]))
        out[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p_flat[high]))
        out[high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    if np.any(middle):
        q = p_flat[middle] - 0.5
        r = q * q
        out[middle] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    return out.reshape(p_array.shape)


def normal_cdf(x):
    """Standard-normal CDF, used to turn a z-score into a probability.

    ASSUMPTION: only ``deflated_sharpe_ratio`` calls this, and only to evaluate
    ``Phi(z)``. ``math.erf`` is accurate to machine precision over the range that
    matters here, so this function contributes no approximation error of its own --
    the approximations are upstream of it, and are labelled as such.

    Accepts a scalar or an array; returns a float or an array of floats.
    """
    import math

    x_array = np.asarray(x, dtype=float)
    erf_values = np.vectorize(math.erf, otypes=[float])(x_array / np.sqrt(2.0))
    cdf = 0.5 * (1.0 + erf_values)
    if np.ndim(x) == 0:
        return float(cdf)
    return cdf


def _as_datetime_series(dates):
    """Coerce ``dates`` to a pandas datetime Series, or raise with a clear reason.

    ACCEPTS: anything ``pandas.to_datetime`` accepts -- a ``pd.Series``, a
    ``np.ndarray`` of ``datetime64``, a ``pd.Index``, a list of ISO strings, a
    list of ``datetime.date``, and so on.

    ASSUMPTION / LIMITATION: the returned series is NOT normalised to midnight.
    Time-of-day is preserved until the caller needs calendar-day comparisons,
    because the caller may legitimately care about ordering within a day. Purging
    compares calendar days; embargo compares instants. The distinction is
    documented where each is used.
    """
    if dates is None:
        raise ValueError(
            "dates is None. Pass a datetime-like series (pd.Series, numpy "
            "datetime64 array, ISO string list, or datetime.date list)."
        )

    try:
        series = pd.Series(dates)
    except TypeError:
        # Generators and other one-shot iterables.
        series = pd.Series(list(dates))

    converted = pd.to_datetime(series)
    if len(converted) == 0:
        raise ValueError(
            "dates is empty. A purged split needs at least one row per fold."
        )
    if bool(converted.isna().any()):
        missing = int(converted.isna().sum())
        raise ValueError(
            f"dates contains {missing} unparseable or missing value(s). "
            "Rows with no date cannot be purged or embargoed safely, so they "
            "are rejected rather than silently kept."
        )
    return converted


def embargo_days(dates, embargo_pct):
    """Convert an embargo fraction into a whole number of DAYS.

    ``embargo_pct`` is a fraction of the total calendar span of ``dates``:
    ``embargo_pct * (max(dates) - min(dates))``. The result is floored to a whole
    day and is therefore never negative.

    ASSUMPTIONS:
      * The relevant time axis is the calendar date, so a partial day is not a
        meaningful embargo -- flooring is the conservative reading and never
        over-embargoes.
      * The caller passes the FULL date series, not just a training fold. Using a
        fold's own span would make the embargo depend on the fold, which is
        exactly the kind of hidden assumption this module exists to remove.
      * ``dates`` may be unsorted and in any pandas-coercible form; the span is
        the same either way.

    NOT ASSUMED: nothing about game density, injury news, or any other channel
    through which information could leak across the boundary. The embargo is a
    blanket time buffer, not a modelled leakage process.

    The default of ``0.01`` applied to this repo's roughly 1.2-season test span
    is a handful of days -- which is a guess, and a small one. It is exposed here
    so the number can be printed and argued with rather than buried in a call.
    """
    pct = float(embargo_pct)
    if not np.isfinite(pct):
        raise ValueError("embargo_pct must be a finite number")
    if pct < 0.0:
        raise ValueError(
            f"embargo_pct must be >= 0, got {pct}. A negative embargo would "
            "pull training rows closer to the test window, which is the opposite "
            "of what this function is for."
        )

    converted = _as_datetime_series(dates)
    span_seconds = (converted.max() - converted.min()).total_seconds()
    span_days = span_seconds / SECONDS_PER_DAY
    return int(np.floor(pct * span_days))


def _split_setup(dates, n_splits, embargo_pct, gap):
    """Internal: shared setup for the walk-forward splitter.

    Kept separate from ``purged_walk_forward_split`` so that ALL argument
    validation happens EAGERLY, at call time, rather than being deferred to the
    first ``next()`` on the generator. A validation utility that only complains
    once you start iterating is a validation utility people work around.

    Returns the tuple the fold generator needs, or raises ValueError naming the
    specific problem.
    """
    # --- eager argument validation -------------------------------------------
    try:
        n_splits_int = int(n_splits)
    except (TypeError, ValueError):
        raise ValueError(f"n_splits must be an integer, got {n_splits!r}")
    if n_splits_int != n_splits or n_splits_int < 1:
        raise ValueError(f"n_splits must be a positive integer, got {n_splits!r}")

    try:
        gap_int = int(gap)
    except (TypeError, ValueError):
        raise ValueError(f"gap must be an integer number of days, got {gap!r}")
    if gap_int != gap or gap_int < 0:
        raise ValueError(f"gap must be a non-negative integer, got {gap!r}")

    converted = _as_datetime_series(dates)
    n_rows = len(converted)

    test_size = n_rows // (n_splits_int + 1)
    if test_size < 1:
        raise ValueError(
            f"{n_rows} rows cannot be split into {n_splits_int} folds with a "
            "non-empty test set each. Reduce n_splits or supply more rows."
        )

    embargo = embargo_days(converted, embargo_pct)
    buffer_days = embargo + gap_int

    # Stable sort so that rows sharing a timestamp keep their relative input
    # order, which makes the output deterministic across runs and platforms.
    order = np.argsort(converted.to_numpy(), kind="mergesort")
    sorted_dates = converted.iloc[order]
    day_values = sorted_dates.dt.normalize().to_numpy()
    instant_values = sorted_dates.to_numpy()

    first_test_start = n_rows - n_splits_int * test_size
    return order, day_values, instant_values, test_size, first_test_start, buffer_days


def _walk_forward_bounds(n_rows, test_size, first_test_start):
    """Internal: the ``(start, stop)`` bounds of each test block.

    Split SHAPE matches ``sklearn.model_selection.TimeSeriesSplit``: fixed-size
    test blocks at the tail of the series, expanding training window in front.
    Only the purging and the embargoing differ, and those are applied by the
    caller.

    The bounds are over positions in the date-SORTED series, not in the caller's
    original order.
    """
    bounds = []
    start = first_test_start
    while start < n_rows:
        stop = min(start + test_size, n_rows)
        bounds.append((start, stop))
        start = stop
    return bounds


def purged_walk_forward_split(dates, n_splits=5, embargo_pct=0.01, gap=0):
    """Yield ``(train_idx, test_idx)`` for a purged, embargoed walk-forward split.

    Returns a generator of pairs of numpy integer arrays. Each array holds
    POSITIONAL indices into the ORIGINAL ``dates`` sequence as it was passed in
    (use ``X.iloc[train_idx]``), sorted ascending. The internal sort does not leak
    into the indices, and the caller cannot accidentally validate on unsorted data
    because the sort happens here.

    THREE REMOVALS APPLY TO EVERY TRAINING SET:

    1. WINDOW -- only rows strictly BEFORE the test window train the model. This
       is walk-forward: no training row is ever later than the test row it is
       scored against. Nothing after the test window is used at all.

    2. PURGE -- any training row sharing the same CALENDAR DATE as any test row is
       dropped. ASSUMPTION: same-day games are not independent observations. A
       game and its same-day neighbour share a date, a slate and usually an
       injury report, so counting them as separate evidence double-counts one
       event. Purging compares calendar days (``dt.normalize()``), NOT instants,
       so two games kicked off at 19:00 and 21:30 on the same date are purged
       together. This is stricter than an instant-based rule and is the point.

    3. EMBARGO -- additionally drop training rows within ``embargo_pct`` of the
       total calendar span, expressed in whole days, ending immediately before the
       test window's start. This is DISTINCT from purging: purging removes rows
       ON the boundary, the embargo removes rows just BEFORE it, where serial
       correlation (rest days, form streaks, a team playing three games in five
       nights) means information is still shared. If the caller also passes
       ``gap``, that many extra DAYS are removed on top of the embargo.

       ``gap`` IS IN DAYS, not in samples. This deliberately differs from
       ``sklearn``'s ``TimeSeriesSplit(gap=...)``, which counts rows. Counting
       samples makes the leak depend on how many games happened to fall in a
       window; days is the unit the assumption is actually about.

    ASSUMPTIONS NOT MADE:
      * That a time buffer fully removes leakage. It removes the leakage that
        travels through time. Anything else -- a shared opponent, a league-wide
        rule change, a feature that was computed with future information -- is
        NOT addressed here.
      * That adjacent folds are independent of each other. They are not; only
        each fold is internally disjoint.

    RAISES ValueError if argument validation fails (eagerly, at call time), or if
    any fold would have an EMPTY training set -- raised when that fold is
    produced, with the fold number, the test window and the buffer size, so the
    cause is never silent.
    """
    order, day_values, instant_values, test_size, first_test_start, buffer_days = \
        _split_setup(dates, n_splits, embargo_pct, gap)

    buffer = pd.Timedelta(days=buffer_days)
    bounds = _walk_forward_bounds(len(day_values), test_size, first_test_start)

    def generate():
        for fold_number, (start, stop) in enumerate(bounds):
            test_idx = np.sort(order[start:stop])

            test_days = day_values[start:stop]
            test_window_start = pd.Timestamp(test_days[0])

            # (2) PURGE: drop training rows sharing a calendar date with a test row.
            candidate_mask = np.zeros(len(day_values), dtype=bool)
            candidate_mask[:start] = True
            purged = candidate_mask & ~np.isin(day_values, test_days)

            # (3) EMBARGO: drop rows within the buffer immediately before the
            # test window. Compared on INSTANTS here, so the cutoff is exactly
            # the instant `test_window_start - buffer`.
            cutoff = np.datetime64(test_window_start - buffer)
            embargoed = purged & ~(instant_values >= cutoff)

            train_idx = np.sort(order[embargoed])

            if train_idx.size == 0:
                raise ValueError(
                    f"fold {fold_number}: purging + embargoing left an EMPTY "
                    f"training set (test window "
                    f"{test_window_start.date()} -> "
                    f"{pd.Timestamp(test_days[-1]).date()}, "
                    f"{stop - start} test rows, {buffer_days}-day buffer). "
                    "Either the buffer is far too large for this date range or "
                    "there is not enough history to train on. Lower "
                    "embargo_pct/gap rather than silently training on nothing."
                )

            yield train_idx, test_idx

    return generate()


def selection_bias_estimate(n_configs_tried, n_test_rows):
    """Expected selection effect of trying ``n_configs_tried`` configurations.

    Returns the expected inflation, IN STANDARD ERRORS, of a metric reported as
    the best of N independently drawn scores. If each configuration's score is
    ``N(mu, SE^2)`` then the maximum of N of them is not centred on ``mu``: it is
    E[max of N standard normals] standard errors ABOVE it. That gap is the
    winner's curse, and it is what makes "best of 100 trials" look better than
    "best of 1 trial" even when every trial is worthless.

    METHOD: the Blom approximation to the expected maximum order statistic of N
    standard normals,

        E[max(Z_1..Z_N)] ~= Phi^-1( (N - 0.375) / (N + 0.25) )

    which for N = 1 reduces exactly to ``Phi^-1(0.5) = 0`` and tracks the exact
    value closely in the moderate-N range (Monte-Carlo agreement is asserted in
    ``Tests/test_validation.py``). THIS IS AN APPROXIMATION, NOT A GUARANTEE. For
    very large N it degrades: at N = 10^6 the exact expected maximum is about
    4.86 while this returns about 4.75.

    FINITE-SAMPLE CORRECTION: the standard error of a headline metric is itself
    estimated from ``n_test_rows`` observations, so the plug-in SE is itself
    uncertain. The result is scaled by ``sqrt(T / (T - 2))``, a first-order
    widening that stands in for that uncertainty. HONESTY NOTE: this factor is
    essentially cosmetic at this repo's scale (at T = 1697 it is 1.00059, i.e.
    0.06%) and is not a rigorous correction -- it is a monotone, auditable
    placeholder. It is also exactly zero when ``n_configs_tried == 1``, so it
    cannot manufacture a correction out of nothing.

    ASSUMPTIONS (all unverified -- the function cannot check any of them):
      * INDEPENDENCE. The N scores are treated as independent draws. In practice
        every trial in this repo is scored on the SAME held-out slice, so they
        share that slice's noise and are positively correlated. Correlated trials
        produce a SMALLER true selection effect than this function reports. The
        correction is therefore optimistic -- the right direction to be wrong in
        for a bias that inflates results.
      * APPROXIMATE NORMALITY of the per-trial score distribution.
      * The null variance of the per-trial score is exactly 1 standard error, i.e.
        the caller passes a standard error for the same metric that is being
        reported. A mismatch here silently mis-scales the whole answer.
      * Trials were scored on data not used to fit them. If the search itself
        peeked at the holdout, no correction here is sufficient.

    NOT ASSUMED: that the reported number was selected on this holdout, or that
    the metric is a proportion. Both are the caller's business.
    """
    n_configs = int(n_configs_tried)
    if n_configs != n_configs_tried or n_configs < 1:
        raise ValueError(
            f"n_configs_tried must be a positive integer, got {n_configs_tried!r}. "
            "Zero configurations tried means no selection happened."
        )

    n_rows = int(n_test_rows)
    if n_rows != n_test_rows or n_rows < 2:
        raise ValueError(
            f"n_test_rows must be an integer >= 2, got {n_test_rows!r}. With "
            "fewer than 2 rows no standard error is defined."
        )

    if n_configs == 1:
        # No selection: the single score is unbiased for the mean of one draw.
        return 0.0

    expected_max = float(normal_quantile((n_configs - 0.375) / (n_configs + 0.25)))
    finite_sample = np.sqrt(n_rows / (n_rows - 2.0))
    return expected_max * finite_sample


def _sample_moments(returns):
    """Internal: (mean, std, skewness, kurtosis) of a returns series.

    ASSUMPTION / CONVENTION: the KURTOSIS returned is the PEARSON kurtosis
    (3.0 for a Gaussian), NOT pandas' excess kurtosis. The published
    deflated-Sharpe formula is written in Pearson terms and silently produces a
    wrong correction if that distinction is lost.
    """
    series = pd.Series(np.asarray(returns, dtype=float)).dropna()
    if len(series) < 3:
        raise ValueError(
            f"need at least 3 returns to estimate skewness and kurtosis, got "
            f"{len(series)}. Fewer observations and the deflated Sharpe ratio "
            "is not defined -- report nothing rather than a confident number."
        )
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    # A CONSTANT series does not produce an exactly-zero standard deviation --
    # floating point leaves a residue around 1e-18 -- so an equality test against
    # zero would let a degenerate series through and report a Sharpe of ~1e16.
    # The tolerance is relative to the data's own scale.
    degeneracy_tolerance = 1e-12 * (1.0 + abs(mean))
    if not np.isfinite(std) or std <= degeneracy_tolerance:
        raise ValueError(
            f"returns have a degenerate (effectively zero) standard deviation "
            f"({std!r}), so no Sharpe ratio exists. Check for a constant or "
            "near-constant series."
        )
    skew = float(series.skew())
    kurtosis_pearson = float(series.kurt()) + 3.0
    return mean, std, skew, kurtosis_pearson


def _sharpe(returns):
    """Internal: non-annualised (per-observation) Sharpe ratio of a series."""
    mean, std, _, _ = _sample_moments(returns)
    return mean / std


def deflated_sharpe_ratio(returns, n_configs_tried, benchmark=None):
    """Bailey & Lopez de Prado deflated Sharpe ratio, computed with numpy only.

    The Probability/Deflated Sharpe Ratio answers: "given that I tried
    ``n_configs_tried`` configurations, how likely is it that the Sharpe I am
    holding is positive by SKILL rather than by picking the luckiest of several
    tries?" It returns a probability in [0, 1].

    THE PUBLISHED FORMULATION, in full:

        DSR = Phi[ (SR - SR*) * sqrt(T - 1)
                   / sqrt(1 - gamma3 * SR + ((gamma4 - 1) / 4) * SR^2) ]

        SR* = sqrt(Var(SR)) * [ (1 - gamma) * Phi^-1(1 - 1/N)
                                + gamma * Phi^-1(1 - 1/(N * e)) ]

    where N is the number of configurations tried, gamma is the
    Euler-Mascheroni constant, gamma3 is the SKEWNESS and gamma4 is the PEARSON
    KURTOSIS of the returns, T is the number of observations, and SR* -- the
    expected BEST Sharpe across N independent trials -- is the deflation target.
    The denominator is the non-Gaussian correction: skewed and fat-tailed returns
    make a Sharpe ratio less certain than the naive sqrt(T) scaling assumes.

    A ``benchmark`` Sharpe SR0 is composed by SHIFTING THE HURDLE rather than the
    estimate: the test becomes ``SR - (SR0 + SR*)`` instead of ``SR - SR*``. With
    the default ``benchmark=None`` that is exactly the published formula, and a
    benchmark shifts the bar rather than inflating the result.

    SCALE, AND WHY IT MATTERS: every Sharpe here is NON-ANNUALISED. That is not a
    shortcut, it is the self-consistent choice. Annualising multiplies SR and
    SR* by sqrt(252) but leaves the non-Gaussian denominator untouched, so mixing
    an annualised SR into that denominator -- which is easy to do by accident --
    inflates the result without bound. Keeping every term per-observation makes
    the whole expression scale-consistent and is the version the derivation
    actually supports.

    ASSUMPTIONS:
      * ``Var(SR) = 1``, i.e. each trial's non-annualised Sharpe is treated as a
        standard normal under the null. THIS IS THE BIG ONE. The true deflation
        target depends on the standard deviation of the Sharpe ratio ACROSS the N
        trials, and this function only ever receives one returns series, so it
        cannot estimate it. If the across-trial SD is 0.5, the deflation should
        be HALF of what this returns. Set ``n_configs_tried`` accordingly, or
        treat the result as an upper bound on how much the search inflated it.
      * The N trials are INDEPENDENT. Trials scored on one shared holdout are not.
      * Returns are serially uncorrelated. For daily sports-betting returns that
        is optimistic; autocorrelation makes a Sharpe ratio look better than it is
        and is NOT corrected for here.
      * IID sampling with finite moments; the skew/kurtosis correction is a
        second-order adjustment, not an exact Edgeworth expansion.

    ``benchmark``: the alternative the strategy must beat. ``None`` (the default)
    means a zero-Sharpe benchmark. A SEQUENCE is treated as benchmark RETURNS and
    its non-annualised Sharpe is used. A plain NUMBER is taken to BE that
    Sharpe directly -- this shorthand exists because passing benchmark returns is
    the common case but passing an already-computed benchmark Sharpe is common
    too, and guessing wrong silently inverts the test.

    NOT ASSUMED: that the returns series is the best of N. If the caller passes a
    series that was NOT selected on, the deflation is a harmless conservatism; if
    the caller passes a series that WAS selected on something bigger than
    ``n_configs_tried`` trials, the deflation is too small and the number is
    meaningless.
    """
    n_configs = int(n_configs_tried)
    if n_configs != n_configs_tried or n_configs < 1:
        raise ValueError(
            f"n_configs_tried must be a positive integer, got {n_configs_tried!r}"
        )

    mean, std, skew, kurtosis = _sample_moments(returns)
    sharpe = mean / std
    n_obs = int(len(pd.Series(np.asarray(returns, dtype=float)).dropna()))

    if benchmark is None:
        benchmark_sharpe = 0.0
    elif np.isscalar(benchmark) and not isinstance(benchmark, str):
        benchmark_sharpe = float(benchmark)
        if not np.isfinite(benchmark_sharpe):
            raise ValueError("benchmark Sharpe must be finite")
    else:
        benchmark_sharpe = _sharpe(benchmark)

    # Deflation target: expected maximum of N independent standard normals.
    # Var(SR) is ASSUMED to be 1 (see the docstring); the caller cannot supply it.
    assumed_variance = 1.0
    if n_configs == 1:
        expected_max_sharpe = 0.0
    else:
        q1 = float(normal_quantile(1.0 - 1.0 / n_configs))
        q2 = float(normal_quantile(1.0 - 1.0 / (n_configs * np.e)))
        expected_max_sharpe = np.sqrt(assumed_variance) * (
            (1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2
        )

    correction = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * (sharpe ** 2)
    if not np.isfinite(correction) or correction <= 0.0:
        raise ValueError(
            f"the non-Gaussian correction term evaluated to {correction!r}, which "
            "is not positive. With these skew/kurtosis values the published "
            "deflated-Sharpe denominator is meaningless -- report that instead of "
            "a number."
        )

    z = (sharpe - (benchmark_sharpe + expected_max_sharpe)) \
        * np.sqrt(n_obs - 1.0) / np.sqrt(correction)
    return float(normal_cdf(z))


def report_selection_risk(n_configs_tried, metric_name, n_test_rows, standard_error):
    """Print and return a plain-language summary of the selection risk.

    This is the function a human will actually read next to a headline number,
    so it is written to be quoted in a README without editing. It prints the
    summary AND returns it as a string, so tests can assert on the exact wording
    instead of scraping stdout.

    ``standard_error`` must be in the SAME UNITS as the reported metric. If the
    metric is a proportion (accuracy), pass the standard error as a proportion --
    passing percentage points here would misstate the effect by 100x, which is
    why the raw value and its x100 form are both printed.

    The summary states, in as many words, the thing the number actually means:
    a result smaller than the selection effect is not evidence of skill. That
    sentence is emitted verbatim and deliberately.

    ASSUMPTIONS: whatever ``selection_bias_estimate`` assumes -- the numbers here
    are that estimate rendered as text. It is an APPROXIMATION of the selection
    effect, not a bound on it, and a small one at that. In particular it is
    optimistic (it assumes independent trials); correlated trials give a smaller
    real effect.
    """
    se = float(standard_error)
    if not np.isfinite(se) or se < 0.0:
        raise ValueError(
            f"standard_error must be a finite non-negative number, got "
            f"{standard_error!r}. A metric with no standard error cannot be "
            "compared against a selection effect."
        )

    n_configs = int(n_configs_tried)
    n_rows = int(n_test_rows)

    bias_se = selection_bias_estimate(n_configs, n_rows)
    bias_units = bias_se * se

    lines = [
        "=" * 62,
        " SELECTION RISK",
        "=" * 62,
        f" metric:                  {metric_name}",
        f" configurations tried:    {n_configs}",
        f" test rows:               {n_rows}",
        f" standard error:          {se:.6g}  (x100 = {se * 100:.6g})",
        f" expected selection gain: {bias_se:.3f} SE "
        f"({bias_units:.6g}, x100 = {bias_units * 100:.6g})",
        "",
        " The line above is an APPROXIMATION of the winner's curse: the",
        " expected distance from the average trial to the best of N trials,",
        " assuming independent, approximately Gaussian scores.",
        " It is not a bound and not a guarantee.",
        " Trials scored on one shared holdout are correlated in",
        " reality, which makes the true effect SMALLER.",
        "",
        " Read it this way: a headline number less than this gap is not",
        " distinguishable from the gap itself. Run more trials and the number",
        " goes up. That is the search, not the model.",
        "",
        " a result this small relative to the selection effect is not"
        " evidence of skill.",
        "=" * 62,
    ]
    summary = "\n".join(lines)
    print(summary)
    return summary