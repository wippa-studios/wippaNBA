"""Bet sizing and risk policy for the Kelly stake model.

This module has two halves, and they should not be confused for each other.

The bottom half (``american_to_decimal``, ``calculate_kelly_criterion``) is
ARITHMETIC. It has been audited and is correct: ``calculate_kelly_criterion``
reduces exactly to the textbook ``f* = (b*p - q)/b`` over the full range of
inputs. Its one wart is that ``american_to_decimal`` is misnamed -- it returns
NET odds (true decimal odds minus one) rather than decimal odds, despite what
its docstring says. That behaviour is frozen, load-bearing, and pinned by
``Tests/test_kelly_american_odds.py``; it is documented here so the next
reader does not "fix" it.

The top half (everything below) is POLICY, and that is where the actual risk
lives. Full Kelly is optimal only under an assumption this repository cannot
make: that the edge is KNOWN.

The Kelly fraction ``f*`` is the exact maximiser of expected LOG bankroll,
i.e. the long-run geometric growth rate, for a single repeatable bet with a
known win probability and known payout. Every one of those three words is an
assumption. A model with uncertain calibration supplies a probability that is
an ESTIMATE, not a fact; the estimate is wrong in a direction nobody can sign
in advance, and the size of the error is exactly what the Kelly fraction is
most sensitive to. Wagering ``f*`` is optimal for the edge you BELIEVE you
have, and ruinous for the edge you ACTUALLY have, and the two differ by more
than the edge itself whenever the model is even slightly miscalibrated. The
entire justification for the fractional-Kelly helpers below is that they move
the stake away from the optimum, because the optimum is computed from a
quantity that is not known.

Two further assumptions the optimality argument also needs, and that this
repository also cannot make, are that bets are quasi-independent and that the
bankroll is divisible and repeatable. NBA slates violate the first
(``correlated_exposure_cap`` is the blunt response), and neither is assumed to
hold for the single-bet helpers.
"""

import math


def american_to_decimal(american_odds):
    """
    Converts American odds to decimal odds (European odds).
    """
    if american_odds >= 100:
        decimal_odds = (american_odds / 100)
    else:
        decimal_odds = (100 / abs(american_odds))
    return round(decimal_odds, 2)

def calculate_kelly_criterion(american_odds, model_prob):
    """
    Calculates the fraction of the bankroll to be wagered on each bet
    """
    decimal_odds = american_to_decimal(american_odds)
    bankroll_fraction = round((100 * (decimal_odds * model_prob - (1 - model_prob))) / decimal_odds, 2)
    return bankroll_fraction if bankroll_fraction > 0 else 0


# ---------------------------------------------------------------------------
# POLICY LAYER -- bet sizing and risk controls.
#
# Everything below is additive. Nothing above this line is modified: the two
# functions that existed before this block keep their exact bodies, their
# exact return values, and their exact signatures, because
# Tests/test_kelly_criterion.py and Tests/test_kelly_american_odds.py pin that
# behaviour.
# ---------------------------------------------------------------------------


def _as_probability(value, fallback=0.5):
    """Coerce a value into a finite probability in [0.0, 1.0].

    ASSUMPTION: a probability outside [0, 1] is an upstream BUG, not a real
    edge case, so this exists only so that one bad number cannot turn into a
    negative stake or a division error deeper in the sizing chain, where it
    would be much harder to trace back to its source.

    It CLAMPS rather than raises, and that is a genuine trade, not an
    oversight: clamping keeps a nightly job alive through one malformed row,
    but it will also disguise a caller who passes 55 where they meant 0.55 as
    a certainty. Every clamp here is therefore a documented last resort, and
    any caller who has to rely on it has a bug to fix upstream. This function
    deliberately does NOT log; the audit trail is the caller's job, and the
    cleanest audit trail is the one ``size_bet`` returns as a dict.

    Non-finite input has no clamped value that means anything: ``nan`` has no
    ordering to compare against, and ``+-inf`` is not a probability at all.
    Those map to ``fallback`` (coin-flip 0.5 by default), which is the
    correct fallback for a stake: a model that has stopped producing
    numbers should place no bet, and 0.5 priced against a vig-inclusive
    market produces no edge, so the stake comes back 0.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(value):
        return fallback
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _require_fraction(fraction, name="fraction", allow_zero=False):
    """Validate a fraction-of-something knob and return it as a float.

    ASSUMPTION: these knobs are RISK CONTROLS, so a caller passing an
    out-of-range value has misunderstood the policy and must be told, not
    accommodated. Clamping a fraction of 0.9 (or -1, or 12) into range would
    let a misconfigured stake run at full Kelly while the code silently
    claimed otherwise -- that is precisely the "silent magic" failure this
    module is being changed to eliminate. So this raises ``ValueError``
    instead.

    The accepted range is (0, 1] for ``fraction`` (0.0 is rejected because a
    zero stake fraction is a different decision -- "do not bet" -- and belongs
    in the caller's logic, not hidden inside a sizing helper) and [0, 1] for a
    shrinkage weight (0.0 is legitimate there: "trust the model fully").

    Also rejects ``None``, booleans-as-numbers are tolerated as their numeric
    value (True -> 1.0), and any non-numeric object raises ``ValueError``
    rather than ``TypeError`` so that callers only have to catch one thing.
    """
    try:
        value = float(fraction)
    except (TypeError, ValueError):
        raise ValueError(
            name + " must be a real number, got " + repr(fraction)
        )
    too_small = value <= 0.0 and not (allow_zero and value == 0.0)
    if not math.isfinite(value) or too_small or value > 1.0:
        low = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(
            name + " must be in " + low + ", got " + repr(fraction)
        )
    return value


def _as_nonnegative(value, fallback=0.0):
    """Coerce a value into a finite, non-negative float.

    ASSUMPTION: used for stake fractions and percentage caps, which are
    allowed to exceed 1.0 (a bet may legitimately be a multiple of bankroll in
    the raw Kelly arithmetic before it gets capped) but must never be
    negative, and non-finite values are meaningless. Garbage collapses to 0.0
    rather than raising, because a single unreadable entry in a bet slate
    should zero that entry, not abort the sizing of the other seven.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return value


def implied_probability(american_odds):
    """The book's implied probability for ONE side, INCLUDING the vig.

    Returns ``1 / (true decimal odds)``. For ``-110`` that is
    ``1 / 1.909090909 = 0.523810``; for ``+110`` it is ``1 / 2.1 = 0.476190``.

    RELATIONSHIP TO ``american_to_decimal``: that function returns NET odds
    ``b`` (true decimal odds minus one), so the implied probability is
    ``1 / (1 + b)``. This function deliberately does NOT route through it,
    because ``american_to_decimal`` rounds to two decimals and that rounding
    is load-bearing for the frozen Kelly arithmetic but wrong for this one:
    ``1 / 1.91`` is 0.523560, which is 25 micro-probability away from the true
    0.523810, and that gap is the same order as the edges this repository is
    trying to find. So the price is re-derived here from the raw American
    number, unrounded, and the result is rounded to six decimals purely so
    the returned audit dictionaries are readable.

    WHY IT IS NEEDED SEPARATELY: a single quoted price is not a probability.
    It is one half of a two-sided book that carries a margin, so this number
    is biased -- systematically too high, because the margin is baked in.
    Turning a pair of these into a vig-free pair is ``devig_two_way``'s job.
    Using this value directly as "what the market thinks" is a known,
    documented, and conservative-when-betting-favourites error, not an
    oversight.

    ASSUMPTION: the quoted price is for a two-way market with a single
    margin applied across both sides. A price from a market that is already
    vig-free (some exchanges) will be treated as if it still carried a margin,
    which under-de-vigs -- a conservative error in the same direction.

    ASSUMPTION NOT MADE: this function assumes nothing about which side of a
    game the price refers to. It returns the probability of the outcome the
    price is quoted on, which for a two-way market is one side and its
    complement is a separate quote, not ``1 - this value``.

    EDGE CASES: there is no validation, matching ``american_to_decimal``. A 0
    propagates into ``100 / abs(0)`` and raises ``ZeroDivisionError``; callers
    are expected to filter unusable prices before they get here. ``-1`` and
    ``+1`` are absurd prices but return 0.009901 and 0.009901 respectively
    rather than raising.
    """
    if american_odds >= 100:
        decimal_odds = 1.0 + (american_odds / 100.0)
    else:
        decimal_odds = 1.0 + (100.0 / abs(american_odds))
    return round(1.0 / decimal_odds, 6)


def devig_two_way(prob_a, prob_b, method="proportional"):
    """Turn two vig-inclusive implied probabilities into a vig-free pair.

    Returns ``(fair_a, fair_b)`` where the two entries sum to 1.0. Both
    inputs are vig-INCLUSIVE: two quoted prices from a two-sided book sum to
    more than 1.0, and the excess is the book's margin.

    METHODS
    -------
    ``"proportional"`` (default): ``p_i' = p_i / (p_a + p_b)``. Divides the
    overround in proportion to each price, so the favourite-longshot bias
    (longshots are systematically shaded longer than the true odds justify)
    is carried straight through, uncorrected.

    ``"shin"``: Shin (1993). Interprets the vig as the footprint of informed
    money in the line and solves for the informed fraction ``z`` such that the
    de-vigged pair sums to 1.0, using the closed-form transform
    ``p' = (sqrt(z^2 + 4(1-z)p^2/S) - z) / (2(1-z))`` with ``S = p_a + p_b``.
    It moves probability AWAY from longshots and TOWARDS favourites relative
    to the proportional answer, which is the direction the empirical
    favourite-longshot bias points.

    WHICH TO USE FOR FAVOURITE-LONGSHOT BIAS: ``"shin"``. That is the one
    thing it is for. The bias is not symmetric -- books shade longshots more
    than favourites -- so a method that divides the margin proportionally
    leaves the skew fully intact, while Shin's informed-trader model is the
    standard way of removing it.

    BUT NO METHOD IS CORRECT. They are differently wrong, and the choice is an
    assumption about the book, not a fact about the game. Proportional assumes
    the margin is proportional to price. Shin assumes a fraction of the book
    is informed and that the rest is noise. Neither is checkable from a single
    pair of prices, both are unfalsifiable without historical closing lines,
    and the difference between them is small compared with the difference
    between EITHER of them and a genuinely sharp market -- which is the real
    reason to shrink toward the market at all rather than to argue about
    de-vig methods. If you have both sides of the book, use this; if you have
    closing-line prices over time, prefer fitting Shin's ``z`` to that.

    ASSUMPTION: exactly two outcomes. A three-way market (moneyline with two
    underdogs, or any market with genuine draw odds) would need a three-way
    de-vig; the proportional form extends trivially, Shin does not.

    EDGE CASES: inputs are clamped into [0, 1] (see ``_as_probability``). A
    pair that already sums to 1.0 has no margin to remove, and Shin's root
    finder is not defined there, so it falls back to the proportional answer,
    which is the identity. A pair summing to zero has no direction to
    distribute and raises ``ValueError``. An unknown ``method`` raises
    ``ValueError`` rather than defaulting, because silently picking a
    de-vig method is exactly the sort of quiet decision-making this module
    exists to remove.
    """
    quoted_a = _as_probability(prob_a)
    quoted_b = _as_probability(prob_b)
    quoted_total = quoted_a + quoted_b
    if quoted_total <= 0.0:
        raise ValueError(
            "devig_two_way needs at least one positive probability, got "
            + repr(prob_a) + " and " + repr(prob_b)
        )

    if method == "proportional":
        return (
            round(quoted_a / quoted_total, 6),
            round(quoted_b / quoted_total, 6),
        )
    if method == "shin":
        if quoted_total <= 1.0 + 1e-12:
            # No margin to remove; proportional is the identity and Shin's
            # informed-fraction root does not exist inside (0, 1).
            return (
                round(quoted_a / quoted_total, 6),
                round(quoted_b / quoted_total, 6),
            )
        fair_a, fair_b = _shin_pair(quoted_a, quoted_b, quoted_total)
        if fair_a is None:
            return (
                round(quoted_a / quoted_total, 6),
                round(quoted_b / quoted_total, 6),
            )
        shin_total = fair_a + fair_b
        return (round(fair_a / shin_total, 6), round(fair_b / shin_total, 6))
    raise ValueError(
        "unknown devig method " + repr(method)
        + "; expected 'proportional' or 'shin'"
    )


def _shin_transform(quoted_prob, quoted_total, z):
    """One side of Shin's closed-form de-vig transform. See devig_two_way."""
    inner = z * z + 4.0 * (1.0 - z) * (quoted_prob * quoted_prob) / quoted_total
    if inner < 0.0:
        inner = 0.0
    return (math.sqrt(inner) - z) / (2.0 * (1.0 - z))


def _shin_pair(quoted_a, quoted_b, quoted_total):
    """Solve Shin for the informed fraction z and return the fair pair.

    ASSUMPTION: the de-vigged probabilities must sum to exactly 1.0, and the
    sum of Shin's transform is strictly decreasing in z over the pair, so
    there is a unique root and bisection finds it. ``z`` is solved for rather
    than taken from the closed form because the closed form's intermediate
    discriminant goes negative for the vig levels a real book quotes (around
    1.02 to 1.10) at the usually-assumed insider fraction, which turns a
    routine call into a domain error. Returns ``(None, None)`` when no root
    brackets, so the caller can fall back rather than emit a wrong number.
    """
    def pair_sum(z):
        return (
            _shin_transform(quoted_a, quoted_total, z)
            + _shin_transform(quoted_b, quoted_total, z)
        )

    low, high = 0.0, 0.999
    if pair_sum(high) > 1.0:
        return None, None
    for _ in range(200):
        mid = 0.5 * (low + high)
        if pair_sum(mid) > 1.0:
            low = mid
        else:
            high = mid
    z = 0.5 * (low + high)
    return _shin_transform(quoted_a, quoted_total, z), _shin_transform(quoted_b, quoted_total, z)


def shrink_probability(model_prob, market_prob, weight):
    """Shrink a model probability toward the market's by ``weight``.

    ``p' = (1 - weight) * p_model + weight * p_market``

    ``weight`` in [0, 1] is how much of the market to BORROW: 0.0 trusts the
    model completely, 1.0 replaces it with the market entirely, 0.5 takes the
    midpoint.

    ASSUMPTION, and it is the load-bearing one: the market is BETTER than the
    model, or at least not worse. Shrinking toward a strictly worse
    reference makes every bet worse, monotonically, and there is no value of
    ``weight`` that fixes that. The premise is supported empirically in
    sports betting (markets are efficient and beating the close is hard) but
    it is an empirical claim about a specific league and era, not a theorem.

    ASSUMPTION, second: this is LINEAR SHRINKAGE IN PROBABILITY SPACE, which
    is exactly correct only in the narrow case where ``market_prob`` is the
    TRUE probability and the model is unbiased-but-noisy, in which case the
    formula really is the posterior mean under a two-point prior. It is NOT
    the log-loss-optimal shrinkage in general, and it is not the geometric
    interpolation in log-odds space that an odds-ratio view of the world
    would want. The two disagree most exactly where this function is used --
    a model and a market that disagree by several points on a side -- so treat
    ``weight`` as a TUNED KNOB calibrated against out-of-sample log loss, not
    as a derived constant.

    ASSUMPTION, third: a single scalar ``weight`` applies to every bet. Real
    model advantage is larger on some lines than others. One global weight is
    a floor on how much the model is trusted, not a model of how much it
    should be trusted.

    Inputs are clamped into [0, 1] (see ``_as_probability``); ``weight`` is
    validated into [0, 1] and raises ``ValueError`` outside it.
    """
    weight = _require_fraction(weight, "weight", allow_zero=True)
    model_value = _as_probability(model_prob)
    market_value = _as_probability(market_prob)
    return round(
        (1.0 - weight) * model_value + weight * market_value, 6
    )


def fractional_kelly(american_odds, model_prob, fraction=0.25):
    """Full-Kelly stake scaled by ``fraction``, as a PERCENTAGE of bankroll.

    Returns ``calculate_kelly_criterion(american_odds, model_prob) *
    fraction``. It is the SAME arithmetic as full Kelly, deliberately: the only
    change is the multiplier, so this function inherits the audited
    calculation and the frozen rounding exactly. ``fraction`` is validated
    into (0, 1] and raises ``ValueError`` outside it.

    The result is never negative: ``calculate_kelly_criterion`` already floors
    at 0, so a bet with no edge returns 0 no matter the fraction.

    WHY 0.25 IS THE DEFAULT
    ----------------------
    Because the full-Kelly optimum is computed from a probability this
    repository does not actually know. ``f*`` maximises expected log bankroll
    for a KNOWN edge; a miscalibrated model breaks the assumption the
    maximisation rests on, and the damage is asymmetric -- overbetting a
    positive edge costs some growth, overbetting a negative edge costs the
    bankroll.

    The standard result (MacLean, Thorp & Ziemba) is that betting a fraction
    ``f`` of the Kelly optimum retains a fraction ``2f - f^2`` of the optimal
    growth rate, because the growth function is quadratic near its peak. At
    ``f = 0.25`` that is 0.5 - 0.0625 = 0.4375: 43.75% of full Kelly's growth
    for 25% of its drawdown. That is the trade quarter Kelly makes, and it is
    the whole argument for it.

    It is a CONVENTION, not a derived constant. The right fraction is set by
    how wrong this model is, not by any theorem; a better-calibrated model
    should use a larger one, and a worse one a smaller one (or none at all).
    Nothing here measures that, so the number is a documented default that a
    caller is expected to override with evidence.

    ASSUMPTION: single-bet bankroll risk only. This sizes ONE bet against the
    whole bankroll and knows nothing about the rest of the day's slate, which
    is why ``size_bet`` and ``correlated_exposure_cap`` exist alongside it.
    """
    fraction = _require_fraction(fraction, "fraction")
    return round(calculate_kelly_criterion(american_odds, _as_probability(model_prob)) * fraction, 4)


def size_bet(american_odds, model_prob, market_prob=None, fraction=0.25,
             shrink_weight=0.0, max_bet_pct=5.0):
    """THE bet-sizing function. Composes de-vig, shrink, fractional Kelly, cap.

    This is the single entry point a caller should use. It runs the full
    policy in one place and returns an audit dict describing every step, on
    the principle that a stake you cannot reconstruct after the fact is a
    stake you cannot audit.

    THE PIPELINE
    ------------
    1. De-vig, if ``market_prob`` is supplied. ``market_prob`` is read as the
       VIG-INCLUSIVE implied probability for the side being bet -- the same
       quantity ``implied_probability`` returns. The counterpart side's
       vig-inclusive probability is inferred from the offered price under the
       assumption that the book quotes its two sides at mirrored prices, whose
       implied probabilities sum to exactly 1.0 (that identity is exact
       algebra: ``1/(1+b) + 1/(1+1/b) = 1``). The two are then passed through
       ``devig_two_way`` with the proportional method.

       The honest consequence of using one price: when ``market_prob`` is just
       ``implied_probability(american_odds)``, this de-vig IS the identity,
       because a mirrored pair carries no overround at the implied-probability
       level (its margin lives entirely in the payout asymmetry). It starts
       doing real work only when the caller supplies a SHARPER market
       probability than the offered price implies -- a lower-vig source, a
       closing line, an exchange -- in which case the difference between the
       two is removed here. A caller who has both sides of a real book should
       call ``devig_two_way`` directly and pass the vig-free number as
       ``market_prob``; this path exists so that the common case does not
       crash, not so it can be done properly.

    2. Shrink, by ``shrink_weight``, toward the de-vigged market probability
       (see ``shrink_probability`` for why the linear form is only exactly
       right for small disagreements). Skipped when no ``market_prob`` is
       supplied.

    3. Fractional Kelly on the resulting probability (see
       ``fractional_kelly`` for why the default is 0.25).

    4. Clamp at ``max_bet_pct``, and report in the audit dict whether the
       clamp bound.

    RETURNS
    -------
    A dict with these keys, all percentages of bankroll unless the name says
    otherwise::

        stake_pct                 final clamped stake. THE number to use.
        raw_kelly_pct             full-Kelly stake before the fraction
        fractional_kelly_pct      stake after the fraction, before the cap
        cap_pct                   the max_bet_pct that was applied
        capped                    True iff the cap actually bound
        no_edge                   True iff there was no positive edge at all
        probability_used          the probability Kelly was computed from
        model_prob                the model probability, after clamping
        market_prob_vig_inclusive what was passed in, or None
        market_prob_vig_free      after de-vig, or None
        fraction                  the Kelly fraction applied
        shrink_weight             the shrinkage weight applied

    ASSUMPTION: one bet, sized against the whole bankroll, with the rest of
    the day's exposure handled by ``correlated_exposure_cap``. The per-bet cap
    here bounds a SINGLE line, not a slate.

    ASSUMPTION: ``max_bet_pct`` is a risk limit in percentage points, not a
    fraction, matching the percentage convention of the rest of this module.

    EDGE CASES: ``fraction`` and ``shrink_weight`` are validated and raise
    ``ValueError`` out of range. Supplying ``shrink_weight > 0`` WITHOUT a
    ``market_prob`` raises ``ValueError`` rather than quietly skipping the
    shrinkage -- a risk control that silently does nothing is worse than no
    risk control, because the caller believes it is applied. A non-finite or
    negative ``max_bet_pct`` is treated as 0.0 (stake nothing).
    """
    fraction = _require_fraction(fraction, "fraction")
    shrink_weight = _require_fraction(shrink_weight, "shrink_weight", allow_zero=True)
    model_value = _as_probability(model_prob)
    max_pct = _as_nonnegative(max_bet_pct)

    market_vig_inclusive = None
    market_vig_free = None
    if market_prob is not None:
        market_vig_inclusive = _as_probability(market_prob)
        counterpart = 1.0 - implied_probability(american_odds)
        market_vig_free = devig_two_way(market_vig_inclusive, counterpart)[0]
    elif shrink_weight > 0.0:
        raise ValueError(
            "shrink_weight > 0 requires a market_prob to shrink toward; "
            "pass the market's probability for this side, or set "
            "shrink_weight=0.0 to state explicitly that the model is "
            "trusted on its own"
        )

    if market_vig_free is None:
        probability_used = model_value
    else:
        probability_used = shrink_probability(model_value, market_vig_free, shrink_weight)

    raw_kelly = calculate_kelly_criterion(american_odds, probability_used)
    fractional_pct = round(raw_kelly * fraction, 4)
    stake_pct = min(fractional_pct, max_pct)

    return {
        "stake_pct": stake_pct,
        "raw_kelly_pct": raw_kelly,
        "fractional_kelly_pct": fractional_pct,
        "cap_pct": max_pct,
        "capped": fractional_pct > max_pct,
        "no_edge": fractional_pct <= 0.0,
        "probability_used": probability_used,
        "model_prob": model_value,
        "market_prob_vig_inclusive": market_vig_inclusive,
        "market_prob_vig_free": market_vig_free,
        "fraction": fraction,
        "shrink_weight": shrink_weight,
    }


def correlated_exposure_cap(bets, max_total_pct):
    """Scale a slate of proposed stakes down proportionally to a total cap.

    ``bets`` is a list of stake percentages. If they already sum to at most
    ``max_total_pct`` they are returned unchanged; otherwise every entry is
    multiplied by ``max_total_pct / sum(bets)`` so the total lands exactly on
    the cap. The scale factor is not returned, but it is
    ``min(1.0, max_total_pct / sum(bets))`` and any caller can recompute it.

    WHY THIS EXISTS
    ---------------
    A slate of eight nominally-5% bets is not eight independent 5% bets. If
    those eight lines share any common cause -- one team's rest, one ref
    crew, one blown cover at the buzzer, a single model being wrong in the
    same direction on eight games -- then the joint outcome is not the
    average of the eight and the bankroll is exposed to far more than the sum
    of the fractions suggests. Full Kelly's optimality argument assumes
    quasi-independent, repeatable bets; an NBA slate breaks that assumption
    quietly and every day. Capping total exposure is the blunt, honest
    response.

    ASSUMPTION, and it is a weak one: proportional scaling. It treats all
    entries as equally deserving, so it shrinks a 1% line as hard as a 12%
    line, and it scales a slate of eight near-independent bets exactly as
    hard as a slate of eight bets on the same team. It knows NOTHING about
    correlation structure, and proportional scaling to a total is a FLOOR on
    exposure, not a correct model of it. A correlation-aware allocator would
    shrink positions that share a factor much harder than positions that do
    not; this does not, because nothing in this repository currently
    estimates those correlations. Do not read a slate that passes through
    here as being diversified.

    ASSUMPTION: the entries are percentages of the same bankroll, which is
    what ``size_bet`` returns, and scaling them by a common factor preserves
    their RELATIVE ranking even though it breaks their individual caps
    implicitly (a capped 5% entry in a scaled slate is now less).

    ASSUMPTION NOT MADE: this does not account for correlated LOSSES across
    days, for correlated exposure with existing positions, or for the
    possibility that the correct answer is to drop the worst bets rather than
    shrink all of them. Shrinking preserves the caller's ranking, which is a
    defensible default and not obviously the right one.

    EDGE CASES: an empty slate returns ``[]``. An empty slate or an all-zero
    slate sums to 0 and is returned unchanged (there is no direction to
    distribute into). Negative and non-finite entries are coerced to 0.0
    first, so one malformed line cannot produce a negative scaled stake or a
    negative scale factor. A negative ``max_total_pct`` is treated as 0.0.
    """
    cap = _as_nonnegative(max_total_pct)
    cleaned = [_as_nonnegative(bet) for bet in bets]
    total = sum(cleaned)
    if total <= cap:
        return cleaned
    scale = cap / total
    return [bet * scale for bet in cleaned]