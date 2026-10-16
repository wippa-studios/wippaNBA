"""Probability-quality metrics: log loss, Brier score, and calibration.

WHY THIS MODULE EXISTS
======================
Accuracy has been retired as this project's headline metric, and this module is
where that replacement actually lives. A model can be *accurate* -- it picks the
right side more often than a coin flip -- and still be useless for the only job
this project has, which is turning a probability into a bet. What a bet needs is
not a label, it is a number it can be priced against:

    expected value = P(win) * decimal_odds - (1 - P(win))

Nothing in that expression rounds. There is no accuracy term in it and no
threshold. So a model that is right 58% of the time while reporting 51% on the
games it wins is not 58% good at betting; it is off by seven points on every
single recommendation, and the Kelly size that follows is wrong by the same
seven points.

THE CONSEQUENCE, SPELLED OUT
---------------------------
Expected value in this repo (``src/Utils/Expected_Value.py``) is computed
*directly* off the probability this module scores. Therefore a calibration error
of 3 points is a 3-point error in every bet recommendation that model ever
produces -- on the bets that get taken and, just as damagingly, on the ones that
get skipped. A 6-point under-prediction is not a rounding issue; it is the
difference between flagging a bet and hiding it.

The defect that produced this module was exactly that. Inverse-frequency class
weighting pushed the model's mean P(home) to 0.5079 against a true base rate of
0.5694 -- 6.15 points low -- while the sigmoid calibrator was fit on *unweighted*
labels, so it was correcting a distortion it had never been validated against.
The accuracy number barely moved, which is precisely why nobody noticed. Under
this module's metrics it is the loudest thing in the report:

    expected_calibration_error -> 0.0615
    probability_summary        -> "under-confident by 6.15 points"

WHAT IS IN HERE
===============
``log_loss_score``            strictly proper score, the headline number.
``brier_score``               mean squared probability error, multi-class aware.
``calibration_curve_data``    THE binning implementation. Everything below is a
                              view over this one function, so the ECE a caller
                              prints and the curve a caller plots cannot disagree.
``reliability_curve``         the plot/print-ready unpack of the above.
``expected_calibration_error``sample-weighted mean absolute bin gap.
``maximum_calibration_error``worst single bin gap, a stricter headline.
``probability_summary``       the printable report a human actually reads.

Dependencies are numpy only (pandas Series and numpy arrays both work, because
everything is funnelled through ``np.asarray``). scikit-learn is not imported at
all: these definitions are written out longhand so they can be read and audited
against scikit-learn's documented formulas, and so a scikit-learn upgrade cannot
silently change a number that feeds a bet. Calibrator *fitting* lives in
``src/Train-Models/training_reporting.py`` and is deliberately not here -- this
module scores probabilities, it does not produce them.

CONVENTIONS
===========
* Metrics are losses or gaps: LOWER IS BETTER, except ``counts``, which is not a
  score. There is no "accuracy" function here on purpose.
* 1-D ``y_prob`` is read as P(positive class). 2-D ``y_prob`` is the full
  (n_rows, n_classes) matrix with one column per class.
* All of these return plain floats or numpy arrays. Nothing prints implicitly;
  ``probability_summary`` RETURNS its report as a string and the caller prints
  it, so it stays composable and testable.
* Nothing raises on a merely *degenerate* input, and nothing silently repairs a
  merely *broken* one. The line between those two is spelled out under EDGE CASES.
"""

import numpy as np

# Float64 machine epsilon. sklearn's log loss clips probabilities to this same
# value, which is what keeps a hard 0 or 1 from producing an infinite loss.
PROBABILITY_EPSILON = np.finfo(np.float64).eps

# How far a probability row may sit from summing to 1 before it is called a bug
# rather than floating-point noise. Deliberately loose enough for predict_proba
# output, tight enough to catch an unnormalised score vector.
ROW_SUM_TOLERANCE = 1e-6

# The two supported binning layouts for the reliability table.
BINNING_STRATEGIES = ("uniform", "quantile")


# ---------------------------------------------------------------------------
# Input handling. Every public metric funnels through here so that no two of
# them can disagree about what the labels mean.
# ---------------------------------------------------------------------------


def _to_float_array(values, name):
    """Coerce an array-like to a float64 array, or fail with a usable message."""
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a numeric array-like; a "
            f"{type(values).__name__} could not be converted to float"
        ) from error


def _check_probabilities(y_prob):
    """Validate the shape and the value range of a probability array.

    NaN and inf are rejected rather than masked. A missing or diverged
    probability is a broken model, and a metric that quietly dropped those rows
    would be hiding exactly the failure this module exists to surface.
    """
    probabilities = _to_float_array(y_prob, "y_prob")
    if probabilities.ndim not in (1, 2):
        raise ValueError(
            "y_prob must be 1-D (P of the positive class) or 2-D "
            f"(rows by classes); got shape {probabilities.shape}"
        )
    if probabilities.size == 0:
        return probabilities
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "y_prob contains NaN or inf. A probability is never missing or "
            "infinite; fix the model output instead of dropping the row"
        )
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError(
            f"y_prob must lie in [0, 1]; it spans "
            f"[{probabilities.min():.6g}, {probabilities.max():.6g}]"
        )
    return probabilities


def _check_row_sums(probabilities):
    """Refuse a 2-D matrix whose rows are not probability distributions.

    Not renormalised -- a silently "fixed" matrix is how a column-order bug
    survives for seasons. It is an error, and it is an error here.
    """
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        return
    gaps = np.abs(probabilities.sum(axis=1) - 1.0)
    worst = float(gaps.max())
    if worst > ROW_SUM_TOLERANCE:
        raise ValueError(
            "every row of a 2-D y_prob must sum to 1; the worst row is off by "
            f"{worst:.6g}, past the {ROW_SUM_TOLERANCE:g} tolerance. Pass a real "
            "probability matrix (or pass one class column as 1-D)"
        )


def _class_indices(y_true, labels, n_columns):
    """Map y_true onto column indices of the probability matrix.

    Returns ``(indices, classes)``. Column 0 is ``classes[0]``, and for a 1-D
    probability vector column 1 is by definition the positive class.
    """
    if labels is None:
        if n_columns == 2:
            # Fixed even when y_true holds one class only: a 1-D vector is
            # always P(class 1), so the layout must not depend on what happened
            # to show up in the labels.
            classes = [0, 1]
        elif y_true.size == 0:
            classes = list(range(n_columns))
        else:
            classes = np.unique(y_true).tolist()
    else:
        classes = list(labels)

    if len(classes) != n_columns:
        if labels is not None:
            raise ValueError(
                f"labels describes {len(classes)} classes ({classes}) but y_prob "
                f"has {n_columns} probability column(s)"
            )
        raise ValueError(
            f"y_true shows {len(classes)} class(es) ({classes}) but y_prob has "
            f"{n_columns} probability column(s). Pass labels=... if the columns "
            "are not in sorted order, or if a class never occurred in y_true "
            "(the totals model's push, for instance)"
        )

    lookup = {value: position for position, value in enumerate(classes)}
    indices = np.empty(y_true.shape[0], dtype=int)
    for position, value in enumerate(y_true):
        if value not in lookup:
            hint = ""
            if n_columns == 2:
                hint = (
                    ". A 1-D y_prob describes a binary problem; a three-class "
                    "target such as the totals model's under/over/push needs its "
                    "full (n, 3) probability matrix"
                )
            raise ValueError(
                f"y_true contains the label {value!r}, which has no column in "
                f"labels={classes}{hint}"
            )
        indices[position] = lookup[value]
    return indices, classes


def _validate_inputs(y_true, y_prob, labels):
    """Shared validation. Returns ``(indices, classes, probabilities)``."""
    truth = np.asarray(y_true)
    if truth.ndim == 2 and truth.shape[1] == 1:
        truth = truth.ravel()
    if truth.ndim != 1:
        raise ValueError(
            f"y_true must be a 1-D array of class labels; got shape {truth.shape}"
        )

    probabilities = _check_probabilities(y_prob)
    if probabilities.ndim == 2 and probabilities.shape[1] == 1:
        # predict_proba()[:, [1]] is a common shape and means the same thing.
        probabilities = probabilities.ravel()
    _check_row_sums(probabilities)

    if probabilities.shape[0] != truth.shape[0]:
        raise ValueError(
            f"y_true has {truth.shape[0]} rows but y_prob has "
            f"{probabilities.shape[0]}; they must be the same games in the same "
            "order"
        )

    n_columns = 2 if probabilities.ndim == 1 else probabilities.shape[1]
    indices, classes = _class_indices(truth, labels, n_columns)
    return indices, classes, probabilities


def _validate_bins(n_bins):
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise ValueError(f"n_bins must be a positive integer; got {n_bins!r}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer; got {n_bins!r}")
    return int(n_bins)


def _validate_strategy(strategy):
    if strategy not in BINNING_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {BINNING_STRATEGIES}; got {strategy!r}"
        )
    return strategy


# ---------------------------------------------------------------------------
# The one binning implementation.
# ---------------------------------------------------------------------------


def _uniform_edges(n_bins):
    """Equal-width bin edges over [0, 1].

    Built as ``arange / n_bins`` rather than ``linspace``. linspace computes
    ``start + step * i``, and 3 * (1/10) is 0.30000000000000004 -- a hair above
    the literal 0.3 -- so a prediction of exactly 0.3 would be filed in
    ``[0.200, 0.300)`` and the printed label would contradict the number in it.
    Dividing instead rounds 3/10 to the same double as the literal 0.3, and the
    table then reads the way a human reads it.
    """
    return np.arange(n_bins + 1) / n_bins


def _quantile_edges(sorted_confidence, assignments, n_bins):
    """Value edges describing the equal-mass bins, for display only.

    Equal-mass binning is decided by RANK, not by a probability cut, so a tie
    sitting on a boundary is split across two bins rather than forcing both of
    them empty. That is why the edges below are reconstructed from the observed
    bin ranges instead of from percentiles. They are non-decreasing and are only
    ever used to label the table; the bin a value lands in is already decided.

    The consequence, which has to be known before reading a table: a split tie
    is divided in INPUT ORDER. Feeding a label-sorted array into equal-mass
    binning therefore manufactures a calibration gap out of nothing, because
    the first half of a tied block is all wins. Interleave equal probabilities
    before binning them.
    """
    counts = np.bincount(assignments, minlength=n_bins)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    edges = np.empty(n_bins + 1)
    current = float(sorted_confidence[0])
    edges[0] = current
    for position in range(n_bins):
        if counts[position] > 0:
            current = float(sorted_confidence[starts[position] + counts[position] - 1])
        edges[position + 1] = current
    edges[n_bins] = float(sorted_confidence[-1])
    return edges


def _bin_assignments(confidence, n_bins, strategy):
    """Assign each confidence value to a bin, and return the bin edges.

    Returns ``(edges, assignments)``. Edges always has ``n_bins + 1`` entries so
    that a caller can label every bin, including the empty ones.
    """
    if strategy == "uniform":
        edges = _uniform_edges(n_bins)
        # side="right" puts a value on the lower edge into the higher bin, which
        # matches numpy histogram convention. The clip puts a hard 1.0 into the
        # last bin instead of off the end of the array.
        assignments = np.searchsorted(edges, confidence, side="right") - 1
        return edges, np.clip(assignments, 0, n_bins - 1)

    order = np.argsort(confidence, kind="stable")
    ranks = np.empty(confidence.shape[0], dtype=int)
    ranks[order] = np.arange(confidence.shape[0])
    assignments = (ranks * n_bins) // confidence.shape[0]
    assignments = np.minimum(assignments, n_bins - 1)
    return _quantile_edges(confidence[order], assignments, n_bins), assignments


def _confidence_and_correct(indices, probabilities):
    """Flatten probabilities into ``(confidence, correct)`` pairs.

    For a 1-D vector that is ``(P(positive), did the positive class happen)`` --
    n pairs. For an (n, C) matrix every class contributes its own n pairs,
    one-vs-rest, so a three-class totals model is scored the same way a binary
    one is and no class is quietly left out.
    """
    if probabilities.ndim == 1:
        return probabilities, (indices == 1).astype(float)

    n_columns = probabilities.shape[1]
    correct = np.zeros(probabilities.size, dtype=float)
    correct[np.arange(indices.shape[0]) * n_columns + indices] = 1.0
    return probabilities.reshape(-1), correct


def calibration_curve_data(y_true, y_prob, n_bins=10, strategy="uniform", labels=None):
    """Bin the predictions and return what each bin actually showed.

    This is the single binning implementation in the module.
    ``expected_calibration_error``, ``maximum_calibration_error`` and
    ``reliability_curve`` are all views over its output, so a calibration error
    and the curve drawn beside it are computed from one set of numbers and
    cannot disagree.

    Returns a dict with:

    ``bin_edges``          (n_bins + 1,) float array, for labelling the plot.
    ``mean_predicted``     (n_bins,) mean predicted probability per bin.
    ``observed_frequency`` (n_bins,) fraction of outcomes that actually happened.
    ``counts``             (n_bins,) int array, how many pairs fell in each bin.
    ``total``              how many pairs there were overall.

    Empty bins get ``nan`` means and a count of 0. ``nan`` rather than 0 on
    purpose: an empty bin has no accuracy, and printing a 0 there would read as
    "the model was confidently wrong", which is a different and much worse claim.

    Why this and not accuracy: the mean predicted probability per bin IS the
    bet-sizing input. If a bin says 0.70 and 0.62 of those games were actually
    won, then every bet sized off that bin is oversized by 8 points. Accuracy
    pools the whole table into one number and cannot see it.

    ``strategy``:
      ``"uniform"``  equal-width bins over [0, 1]. Easy to read, but on an
                     imbalanced problem most games land in two bins and a bin
                     holding three games gets equal weight in the ECE to one
                     holding three hundred.
      ``"quantile"`` equal-mass bins. Every bin holds about the same number of
                     games, so each bin's gap is estimated from a comparable
                     sample. Usually the better default here, because P(home)
                     around 0.55 puts most of a season inside one uniform bin.
                     Read ``_quantile_edges`` first if your predictions contain
                     repeated values: a tie is split in input order.

    Also handles NaN-free inputs only: see the module docstring's edge-case list.
    """
    n_bins = _validate_bins(n_bins)
    strategy = _validate_strategy(strategy)
    indices, classes, probabilities = _validate_inputs(y_true, y_prob, labels)
    confidence, correct = _confidence_and_correct(indices, probabilities)

    if confidence.size == 0:
        return {
            "bin_edges": _uniform_edges(n_bins),
            "mean_predicted": np.full(n_bins, np.nan),
            "observed_frequency": np.full(n_bins, np.nan),
            "counts": np.zeros(n_bins, dtype=int),
            "total": 0,
        }

    edges, assignments = _bin_assignments(confidence, n_bins, strategy)
    counts = np.bincount(assignments, minlength=n_bins)
    predicted_totals = np.bincount(assignments, weights=confidence, minlength=n_bins)
    observed_totals = np.bincount(assignments, weights=correct, minlength=n_bins)

    filled = counts > 0
    mean_predicted = np.full(n_bins, np.nan)
    observed_frequency = np.full(n_bins, np.nan)
    mean_predicted[filled] = predicted_totals[filled] / counts[filled]
    observed_frequency[filled] = observed_totals[filled] / counts[filled]

    return {
        "bin_edges": edges,
        "mean_predicted": mean_predicted,
        "observed_frequency": observed_frequency,
        "counts": counts,
        "total": int(confidence.size),
    }


def _bin_gaps(curve):
    """Absolute gap per populated bin, plus the counts that weight them."""
    counts = curve["counts"]
    filled = counts > 0
    gaps = np.abs(curve["observed_frequency"][filled] - curve["mean_predicted"][filled])
    return gaps, counts[filled]


# ---------------------------------------------------------------------------
# The metrics.
# ---------------------------------------------------------------------------


def log_loss_score(y_true, y_prob, labels=None):
    """Mean negative log likelihood of the true outcomes. Lower is better.

    For a 1-D ``y_prob`` (P of the positive class) that is
    ``mean(-(y*log(p) + (1-y)*log(1-p)))``. For a 2-D matrix it is
    ``mean(-log(p[i, true_class_i]))``. Zero is perfect, ``log(2)`` is what a
    coin flip scores, and the loss rises without bound as a confident model is
    proved wrong -- which is the behaviour a bettor wants: being loudly wrong
    costs more than being vaguely wrong.

    Probabilities of exactly 0 or exactly 1 are clipped to the float64 epsilon
    first, so a saturated model produces a large finite number instead of
    ``inf`` and the rest of the report still prints. That is the only adjustment
    made to the inputs; nothing here is renormalised.

    This is the metric to lead with over accuracy. A model can convert most of
    its picks correctly and still score badly here purely because its
    probabilities are wrong, and that second failure -- not the first -- is what
    misprices a bet.

    Returns ``nan`` for empty input rather than raising: a metrics table in a
    training run should degrade, not crash the run that produced it.

    ``labels`` gives the class order of a 2-D ``y_prob``; without it the classes
    are inferred as the sorted unique values of ``y_true``. For a 1-D ``y_prob``
    the classes are always ``[0, 1]`` unless ``labels`` says otherwise, so a
    single-class ``y_true`` still scores against the right column layout. Note
    that a 1-D vector is ALWAYS the probability of the LAST class in ``labels``,
    matching scikit-learn: under the default that is the positive class, and
    ``labels=[1, 0]`` makes it P(label 0) rather than P(label 1).
    """
    indices, _classes, probabilities = _validate_inputs(y_true, y_prob, labels)
    if probabilities.shape[0] == 0:
        return float("nan")

    if probabilities.ndim == 1:
        # One row per game, one column per class: column 0 is P(negative).
        matrix = np.column_stack([1.0 - probabilities, probabilities])
    else:
        matrix = probabilities
    matrix = np.clip(matrix, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)

    rows = np.arange(indices.shape[0])
    return float(-np.mean(np.log(matrix[rows, indices])))


def brier_score(y_true, y_prob, labels=None):
    """Mean squared error between the predicted and the actual probability.

    Two shapes, two definitions, and the difference is a factor of two:

    * 1-D ``y_prob`` (P of the positive class) gives the binary form,
      ``mean((p - y)^2)``, which lives in [0, 1] with 0 perfect. This is the
      form that compares directly against a bet: it is the expected squared
      error in the number that sizes the bet.
    * 2-D ``y_prob`` gives the multi-class form,
      ``mean_i(sum_c((p_ic - y_ic)^2))``, summed over all N columns and
      averaged over rows, so it lives in [0, 2]. This is the definition
      scikit-learn uses and the right one for the totals model, where
      over/under/push really are three classes: pass the full ``(n, 3)``
      matrix, and scoring only the over column would silently ignore the two
      ways the prediction can be wrong.

    The N columns are SUMMED, not averaged. So a two-column matrix is exactly
    twice the 1-D binary answer on the same rows, which is deliberate: the shape
    of ``y_prob`` picks the definition rather than the numbers deciding it.

    Note for anyone diffing against scikit-learn: scikit-learn's Brier is
    version-dependent in a way that has bitten this repo before.
    ``brier_score_loss`` multiplies the binary answer by 0.5 to put it on the
    [0, 1] scale (that is, the two one-hot columns are summed and then halved
    -- algebraically identical to the binary form, but a different number from
    the multi-class sum). So:
      * ``brier_score_loss(y, p_1d)``                == this function;
      * ``brier_score_loss(y, p_1d, scale_by_half=False)`` == TWICE this;
      * a 3+-column matrix needs ``scale_by_half=False`` to match, and the
        default already matches.
    Every metric here uses the unscaled definitions above, so a cross-check
    against scikit-learn must state which convention it is comparing.

    Like log loss, this is blind to whether the model ranked well. That is what
    makes it complementary rather than a substitute: a model can have a good Brier
    score and still be 6 points off on the one bin a bettor bets from.

    Returns ``nan`` for empty input.
    """
    indices, _classes, probabilities = _validate_inputs(y_true, y_prob, labels)
    if probabilities.shape[0] == 0:
        return float("nan")

    if probabilities.ndim == 1:
        return float(np.mean((probabilities - indices) ** 2))

    one_hot = np.zeros(probabilities.shape)
    one_hot[np.arange(indices.shape[0]), indices] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform", labels=None):
    """Sample-weighted mean absolute gap between confidence and reality.

    For every bin, ``abs(observed frequency - mean predicted probability)``,
    weighted by how many predictions landed in it and averaged over all
    predictions. Empty bins are skipped. Zero is perfect.

    This is the headline calibration number, and it is the one that speaks
    directly to the money: it is the average size, in probability points, of the
    error in the number an EV calculation multiplies by. An ECE of 0.03 means
    bets are systematically priced 3 points wrong. An accuracy number says
    nothing about that -- it is the one place the error is measured instead of
    averaged away.

    The weighting is what makes it trustworthy rather than flattering: a bin
    holding two games cannot move the ECE nearly as far as a bin holding two
    thousand, which is what keeps a tiny bin's noise from reading as a large
    systematic error. Pass ``strategy="quantile"`` so that the bins themselves
    hold comparable numbers of games.
    """
    curve = calibration_curve_data(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, labels=labels
    )
    if curve["total"] == 0:
        return float("nan")
    gaps, counts = _bin_gaps(curve)
    return float(np.sum(counts * gaps) / curve["total"])


def maximum_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform", labels=None):
    """The single worst bin gap. Lower is better.

    The same measurement as ``expected_calibration_error`` without the
    averaging, so it cannot be improved by spreading a bad error thinly across
    many bins. Useful as a stricter headline when the ECE looks acceptable: an
    ECE of 0.02 is tolerable; a worst-bin gap of 0.15 means one corner of the
    probability range -- often exactly the confident tail a bettor bets from --
    is 15 points wrong.

    Returned in the same units and on the same ``[0, 1]`` scale as the ECE, and
    is always greater than or equal to it (the ECE is a weighted mean of the same
    set of gaps). Empty bins are skipped, so this never reports a gap for a bin
    that holds nothing.
    """
    curve = calibration_curve_data(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, labels=labels
    )
    if curve["total"] == 0:
        return float("nan")
    gaps, counts = _bin_gaps(curve)
    return float(np.max(gaps))


def reliability_curve(y_true, y_prob, n_bins=10, strategy="uniform", labels=None):
    """Bin edges, mean predicted, observed frequency and counts, unpacked.

    Exactly ``calibration_curve_data`` as a four-tuple, for callers that want to
    plot or print it directly::

        edges, predicted, observed, counts = reliability_curve(y_true, y_prob)

    ``edges`` has ``n_bins + 1`` entries, one more than the other three, so bin
    ``i`` is ``[edges[i], edges[i + 1])``. Empty bins come back as ``nan`` means
    with a count of 0; ``np.nan`` drops the point out of a plot instead of
    drawing it at zero.

    Prefer this over reading the dict when all four are wanted; prefer the dict
    when only the counts are. There is one binning underneath either way.
    """
    curve = calibration_curve_data(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, labels=labels
    )
    return (
        curve["bin_edges"],
        curve["mean_predicted"],
        curve["observed_frequency"],
        curve["counts"],
    )


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------


def _class_rates(indices, probabilities):
    """Observed frequency and mean predicted probability for each class column."""
    if probabilities.ndim == 1:
        matrix = np.column_stack([1.0 - probabilities, probabilities])
    else:
        matrix = probabilities
    n_columns = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.full(n_columns, np.nan), np.full(n_columns, np.nan)
    observed = np.array(
        [float(np.mean(indices == column)) for column in range(n_columns)]
    )
    return observed, matrix.mean(axis=0)


def _format_bin_label(edges, position):
    """One row's bin range.

    Equal-mass binning splits tied values, so a bin can end up holding one
    repeated probability. That is printed as ``= 0.600`` rather than as an
    empty interval, which would read as a bug.
    """
    left = edges[position]
    right = edges[position + 1]
    if right <= left:
        return f"= {left:.3f}"
    return f"[{left:.3f}, {right:.3f})"


def probability_summary(y_true, y_prob, n_bins=10, strategy="uniform", labels=None):
    """A printable calibration report, returned as a string. Nothing prints.

    This is the function a human actually reads, and it leads with the two
    numbers that decide whether the probabilities can be bet:

        base rate            what fraction of these games the outcome happened
        mean predicted       what the model said, on average
        calibration gap      the difference, in points, with its direction

    A gap of -0.06 is the motivating defect made legible: the model is
    under-confident by 6 points, so every bet it recommends is priced 6 points
    wrong and genuinely positive-EV bets can fall below the threshold. The
    report then gives the ECE, the worst bin, log loss, the Brier score, and the
    full reliability table with counts.

    RETURNED, NOT PRINTED, on purpose. A metrics function with a print inside it
    cannot be composed into a larger report or asserted on, so callers do
    ``print(probability_summary(...))``.

    Degenerate inputs do not raise here; they are annotated instead:
      * no rows -> every figure reads ``nan`` and a note explains why;
      * a single class in ``y_true`` -> a note explains that the observed
        frequencies are 0 or 1, which is arithmetically fine and informationally
        thin.
    """
    n_bins = _validate_bins(n_bins)
    strategy = _validate_strategy(strategy)
    indices, classes, probabilities = _validate_inputs(y_true, y_prob, labels)
    curve = calibration_curve_data(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, labels=labels
    )
    observed_by_class, predicted_by_class = _class_rates(indices, probabilities)

    n_rows = probabilities.shape[0]
    n_columns = 2 if probabilities.ndim == 1 else probabilities.shape[1]
    # How many classes actually turned up in y_true, which is not the same
    # question as how many columns the probability matrix has: a 1-D vector is
    # always scored against [0, 1] even if only one of those ever happened.
    n_present = int(np.unique(indices).size)

    lines = ["Calibration summary"]
    lines.append(f"  rows                      {n_rows}")
    lines.append(f"  class columns             {n_columns}")
    lines.append(f"  binning                   {strategy}, {n_bins} bins")

    if n_columns == 2:
        base_rate = float(observed_by_class[1])
        mean_predicted = float(predicted_by_class[1])
        gap = mean_predicted - base_rate
        if np.isnan(gap):
            lines.append("  base rate                 nan")
            lines.append("  mean predicted            nan")
            lines.append("  calibration gap           nan")
        else:
            direction = (
                "over-confident" if gap > 0 else "under-confident"
            ) if gap != 0 else "unbiased"
            lines.append(f"  base rate                 {base_rate:.4f}")
            lines.append(f"  mean predicted            {mean_predicted:.4f}")
            lines.append(
                f"  calibration gap           {gap:+.4f}  "
                f"{direction} by {abs(gap) * 100:.2f} points"
            )
    else:
        lines.append("  per-class base rate and mean predicted probability")
        for column in range(n_columns):
            label = classes[column] if column < len(classes) else column
            observed = float(observed_by_class[column])
            predicted = float(predicted_by_class[column])
            difference = predicted - observed
            lines.append(
                f"    class {label!s:<6} observed {observed:.4f}   "
                f"mean predicted {predicted:.4f}   gap {difference:+.4f}"
            )

    lines.append(
        f"  expected calibration err  "
        f"{expected_calibration_error(y_true, y_prob, n_bins, strategy, labels):.4f}"
    )
    lines.append(
        f"  maximum calibration err   "
        f"{maximum_calibration_error(y_true, y_prob, n_bins, strategy, labels):.4f}"
    )
    lines.append(
        f"  log loss                  "
        f"{log_loss_score(y_true, y_prob, labels):.4f}"
    )
    lines.append(f"  brier score               {brier_score(y_true, y_prob, labels):.4f}")

    lines.append("")
    lines.append("  reliability table")
    lines.append(
        f"  {'bin':<18} {'n':>6}  {'mean predicted':>15}  {'observed':>9}  {'gap':>8}"
    )
    edges = curve["bin_edges"]
    mean_predicted = curve["mean_predicted"]
    observed_frequency = curve["observed_frequency"]
    counts = curve["counts"]
    for position in range(n_bins):
        label = _format_bin_label(edges, position)
        if counts[position] == 0:
            lines.append(f"  {label:<18} {0:>6}  {'-':>15}  {'-':>9}  {'-':>8}")
            continue
        bin_gap = (
            float(observed_frequency[position]) - float(mean_predicted[position])
        )
        lines.append(
            f"  {label:<18} {int(counts[position]):>6}  "
            f"{float(mean_predicted[position]):>15.4f}  "
            f"{float(observed_frequency[position]):>9.4f}  {bin_gap:>+8.4f}"
        )
    lines.append(f"  {'total':<18} {int(curve['total']):>6}")

    if n_rows == 0:
        lines.append("")
        lines.append("  NOTE: no rows to score; every figure above is nan.")
    elif n_present < 2:
        lines.append("")
        lines.append(
            "  NOTE: only one class is present in y_true. The figures above are "
            "correct,\n        but the observed frequencies are degenerate "
            "(all 0 or all 1), so the\n        reliability table says nothing "
            "about how the probabilities rank."
        )
    elif strategy == "uniform":
        lines.append("")
        lines.append(
            f"  NOTE: {n_rows} rows over {n_bins} equal-width bins. Most games "
            "share one\n        bin; consider strategy=\"quantile\" for a "
            "table with comparable\n        bin sizes."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EDGE CASES
# ---------------------------------------------------------------------------
# The line between these two groups is deliberate and worth stating plainly,
# because it is the difference between a metric module and a laundering machine.
#
# DEGENERATE -- a legal input that simply carries little information. Measured
# and reported, never repaired, never an exception:
#   * empty input           -> nan from every score, empty curve from
#                               reliability_curve, a report that says so.
#   * one class in y_true   -> scored normally; log loss and Brier are perfectly
#                               well defined. The reliability table is
#                               informationally thin (every observed frequency is
#                               0 or 1) and probability_summary says so. The
#                               layout does NOT change: a 1-D y_prob is always
#                               read against classes [0, 1].
#   * constant predictions  -> scored normally. Every bin shares one predicted
#                               value, so the ECE collapses to the closed form
#                               abs(base rate - that constant) for both binning
#                               strategies. That is the cheapest possible sanity
#                               check on a model that has collapsed, and it is
#                               why it is not special-cased.
#   * empty bins            -> nan means, count 0. Never divided by.
#
# BROKEN -- an input that is a bug in the caller or in the model. Raising is the
# honest answer, because every alternative (mask, clip, renormalise) converts a
# visible failure into an invisible one, and invisible failures are exactly what
# let a 6-point calibration error ship:
#   * NaN or inf in y_prob  -> ValueError. A probability is never missing.
#   * y_prob outside [0, 1] -> ValueError.
#   * 2-D rows not summing
#     to 1 within 1e-6      -> ValueError. Not renormalised. A column-order
#                               bug that gets silently rescaled is a column-order
#                               bug that ships.
#   * length mismatch       -> ValueError.
#   * a label with no column-> ValueError, including the case of a three-class
#     y_true scored against 1-D probabilities, which is named explicitly because
#                               the totals model does it by accident easily.
#   * bad n_bins/strategy   -> ValueError.
#
# NOT IN SCOPE: fitting a calibrator. scikit-learn's CalibratedClassifierCV owns
# that, and this module deliberately neither wraps nor reimplements it.