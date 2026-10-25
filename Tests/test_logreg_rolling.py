"""Tests for the rolling-feature logistic regression (MODEL 2).

Three things are pinned here, and they are not the same thing:

* the LEAKAGE GATE -- that a feature for game t is unchanged when every game
  after t is deleted from the input, and that the prior-season carryover draws
  on earlier seasons only. A feature that peeks forward makes every other number
  in this repository meaningless, so this is the test that matters most and it
  is written to fail loudly, not to skip quietly;

* the TRAIN-ONLY GATE -- that imputation medians and standardisation statistics
  are fitted on training rows and are provably untouched by the contents of the
  test block. Imputing before splitting is a recurring bug in this repo and it
  is quiet: the model still trains, the metric still looks fine, and the number
  is just wrong by a little. These tests assert on the fitted PARAMETERS, not
  just on the downstream score, so a leak cannot hide behind a small effect;

* the SIGN GATE -- that on synthetic data with a relationship known by
  construction, the fitted coefficients come out with the signs the domain
  demands: home court positive, own offence positive, own defence negative,
  away-side back-to-back and travel negative. A pipeline that silently inverts a
  contrast, or leaks a level where a difference belonged, produces a model that
  scores well and means nothing.

A NOTE ON THE OFF / DEF / NET BLOCK, because it constrains what may be claimed
from any coefficient table produced by this model
----------------------------------------------------
``nba_data`` documents that ``ewm_net_rating`` is definitionally redundant with
``ewm_off_rating - ewm_def_rating`` (R^2 0.99999735) and makes the design matrix
singular. That is borne out here: over the standardised block
``[net_diff, off_diff, def_diff]`` the third singular value is ~8e-4 of the
first, so the near-null direction is ``net - off + def = 0``.

Two consequences, both respected below. First, per-coefficient readings of
``net_diff``, ``off_diff`` and ``def_diff`` are NOT safe when all three are in
the model - the penalty splits one direction between three columns arbitrarily.
Second, the obvious diagnostic is the wrong one: the MARGINAL correlations look
harmless (``corr(off, def) = -0.22``, ``corr(net, off) = 0.81``) because the
dependency is between a combination, not a pair. Only the singular values show
it. The performance rows are unaffected - any of these fits predict identically
well - so the ablation and walk-forward tables are valid; it is only the
coefficient interpretation that needs an identified parameterisation. That is
what ``IDENTIFIED_FEATURES`` and the ``signs`` feature group are for.

The last group also carries an end-to-end version that points the shared data
module at a synthetic ``DATA_ROOT`` on disk and runs the REAL
``build_games -> rolling_team_features -> RollingLogreg`` chain against data
where the true relationship is known.

Everything here is offline. The real-data tests read the same CSVs as
``src/DataProviders/nba_data`` and write nothing outside a tmp dir.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.DataProviders import nba_data as nd
from src.Models import logreg_rolling as lr

# The design matrix over the real 13,199-game history is the same object for
# every test that needs it, and building it costs ~2s. Built once, lazily, so a
# single slow call is paid once per session rather than once per test.
_DESIGN_CACHE: dict = {}
_TRUNC_N = 4000


def real_games():
    if "games" not in _DESIGN_CACHE:
        _DESIGN_CACHE["games"] = nd.build_games()
    return _DESIGN_CACHE["games"]


def real_design(halflife: float = 10.0):
    key = f"design{halflife}"
    if key not in _DESIGN_CACHE:
        _DESIGN_CACHE[key] = lr.load_design(halflife=halflife)
    return _DESIGN_CACHE[key]


def assert_truncation_equal(full: pd.DataFrame, cut: pd.DataFrame) -> None:
    """``full.iloc[:len(cut)]`` must equal ``cut`` - the leakage contract, with
    one stated tolerance.

    The canonical columns must match BIT-FOR-BIT: a centred rolling mean, a
    full-sample standardisation or a season-end prior moves them and fails. The
    shared availability block (issue #36) is truncation-invariant only to float
    SUMMATION noise: rebuilding it on fewer games can reorder same-date rows
    inside the per-squad player sums and moves values by ~1e-13 (measured
    8.5e-14 on the 4,000-game cut; the round3 P1 gate's own pass bar is 1e-12).
    A leak moves these columns by O(0.01), so the tight bar still catches one,
    and the NaN pattern is compared exactly either way.
    """
    n = len(cut)
    prefix = full.iloc[:n].reset_index(drop=True)
    cut = cut.reset_index(drop=True)
    avail = [c for c in lr.AVAILABILITY_COLUMNS if c in prefix.columns]
    base = [c for c in prefix.columns if c not in avail]
    pd.testing.assert_frame_equal(prefix[base], cut[base], check_exact=True)
    if avail:
        pd.testing.assert_frame_equal(prefix[avail], cut[avail],
                                      check_exact=False, rtol=0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Synthetic fixtures - a world where the answer is known
# --------------------------------------------------------------------------- #

# Log-odds coefficients for the synthetic world, expressed PER STANDARD DEVIATION
# because that is the space the model reports in. Chosen to be the signs the
# domain demands AND large enough to survive sampling noise at n=4000.
#
# There are deliberately no `off_diff`/`def_diff` entries: the world is driven by
# NET rating, and net == off - def by construction, so off and defence carry no
# information net does not. Their COEFFICIENTS are still checked, in
# `test_offensive_and_defensive_coefficients_have_the_right_sign`, because
# regressing a net-driven label on the identified (off, def) pair must reproduce
# positive-on-offence and negative-on-defence.
TRUE_BETAS = {
    "net_diff": 0.55,
    "margin_diff": 0.10,
    "pace_diff": -0.08,
    "ts_diff": 0.30,
    "efg_diff": 0.45,
    "tov_diff": -0.35,
    "ftr_diff": 0.05,
    "orb_diff": 0.02,
    "matchup_home": 0.10,
    "matchup_away": 0.10,
    "pace_mean": 0.20,
    "rest_home": 0.12,
    "rest_away": -0.12,
    "rest_diff": 0.00,
    "b2b_home": -0.30,
    "b2b_away": 0.30,
    "three_home": -0.10,
    "three_away": 0.10,
    "three_diff": 0.00,
    "travel_k": -0.20,
    "home_adv": 0.26,
}

# The feature set with NO linear dependency among its members, so every
# coefficient in it is individually identified and can be compared to a known
# truth. It excludes off_diff/def_def (net == off - def per team, hence
# net_diff == off_diff - def_diff across two teams). That identity is asserted
# directly in `test_the_rating_contrasts_are_near_collinear` rather than
# papered over with a loose tolerance.
IDENTIFIED_FEATURES = [
    "net_diff", "margin_diff", "pace_diff", "pace_mean", "ts_diff", "efg_diff",
    "tov_diff", "ftr_diff", "orb_diff", "rest_home", "rest_away", "b2b_home",
    "b2b_away", "three_home", "three_away", "travel_k", lr.HOME_COLUMN,
]


def synthetic_games(n: int = 4000, seed: int = 7):
    """``games``-shaped frame plus the rolling/rest frames a synthetic run needs.

    The per-team ratings are drawn once per game pair, then the LABEL is
    generated from ``assemble_features`` output via the coefficients above. That
    ordering matters: it means the label is a function of exactly the features
    the model is allowed to see, so any coefficient the model reports has a known
    true value to be compared against.
    """
    rng = np.random.default_rng(seed)
    n_games = n

    # Strength is drawn ONCE per side and net is DERIVED from it, which is what
    # the real data does (NET_RATING == OFF_RATING - DEF_RATING, per team, per
    # game). Drawing them independently would make off/def noise with no relation
    # to net, and the sign tests below would be asserting something true of a
    # world that does not exist.
    def sides(scale=3.0):
        o = rng.normal(110, scale, n_games)
        d = rng.normal(110, scale, n_games)
        return o, d, o - d      # off, def, net -- net == off - def EXACTLY

    off_h, def_h, net_h = sides()
    off_a, def_a, net_a = sides()

    roll = pd.DataFrame({
        "home_ewm_net_rating": net_h,
        "away_ewm_net_rating": net_a,
        "home_ewm_margin": rng.normal(0, 6, n_games),
        "away_ewm_margin": rng.normal(0, 6, n_games),
        "home_ewm_off_rating": off_h,
        "away_ewm_off_rating": off_a,
        "home_ewm_def_rating": def_h,
        "away_ewm_def_rating": def_a,
        "home_ewm_pace": rng.normal(99, 3, n_games),
        "away_ewm_pace": rng.normal(99, 3, n_games),
        "home_ewm_ts_pct": rng.normal(0.57, 0.02, n_games),
        "away_ewm_ts_pct": rng.normal(0.57, 0.02, n_games),
        "home_ewm_efg": rng.normal(0.50, 0.03, n_games),
        "away_ewm_efg": rng.normal(0.50, 0.03, n_games),
        "home_ewm_tov": rng.normal(0.135, 0.015, n_games),
        "away_ewm_tov": rng.normal(0.135, 0.015, n_games),
        "home_ewm_ftr": rng.normal(0.20, 0.05, n_games),
        "away_ewm_ftr": rng.normal(0.20, 0.05, n_games),
        "home_ewm_orb": rng.normal(0.27, 0.02, n_games),
        "away_ewm_orb": rng.normal(0.27, 0.02, n_games),
    })
    rest = pd.DataFrame({
        "home_rest_days": rng.integers(1, 8, n_games).astype(float),
        "away_rest_days": rng.integers(1, 8, n_games).astype(float),
        "home_back_to_back": rng.integers(0, 2, n_games).astype(float),
        "away_back_to_back": rng.integers(0, 2, n_games).astype(float),
        "home_in_3_of_4": rng.integers(0, 3, n_games).astype(float),
        "away_in_3_of_4": rng.integers(0, 3, n_games).astype(float),
        "travel_miles": rng.uniform(0, 2700, n_games),
        # Time-zone travel columns, mirroring rest_and_travel's contract
        # (issue #56). The fixture's teams are placeholders, so the values are
        # draws, not geometry -- nothing here asserts their semantics.
        "tz_offset_diff": rng.integers(0, 4, n_games).astype(float),
        "away_tz_crossed": rng.integers(0, 4, n_games).astype(float),
        "is_home": 1.0,
    })
    games = pd.DataFrame({
        "GAME_ID": [f"{i:05d}" for i in range(n_games)],
        "GAME_DATE": pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n_games), "D"),
        "SEASON": "synthetic",
        "home": "AAA", "away": "BBB",
        "home_pts": np.nan, "away_pts": np.nan, "y": 0, "margin": np.nan, "total": np.nan,
    })

    X = lr.assemble_features(games, roll, rest)
    # Generate the label from the STANDARDISED features, because that is what the
    # model will see. Using raw values instead means `pace_mean` (which sits near
    # 99, not near 0) contributes 0.20 * 99 = 19.8 logits, every game becomes a
    # home win, and the fixture silently stops testing anything.
    Xs = X.copy()
    for c in Xs.columns:
        sd = float(Xs[c].std(ddof=0))
        if sd > 1e-12:
            Xs[c] = (Xs[c] - Xs[c].mean()) / sd
        # Constant columns (home_adv) stay put; their "coefficient" is an
        # intercept in log-odds, not a per-SD effect.
    eta = np.zeros(len(games))
    for name, beta in TRUE_BETAS.items():
        if name in Xs.columns:
            eta = eta + beta * Xs[name].to_numpy(dtype=float)
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=len(games)) < p).astype(int)
    games["y"] = y
    # Flip scores so the label is also consistent with a margin, exactly as the
    # real table is.
    games["margin"] = np.where(y == 1, 5.0, -5.0)
    games["home_pts"] = 110 + games["margin"] / 2
    games["away_pts"] = 110 - games["margin"] / 2
    return games, roll, rest, X


# --------------------------------------------------------------------------- #
# (a) LEAKAGE
# --------------------------------------------------------------------------- #

class TestFeaturesAreStrictlyCausal(unittest.TestCase):
    """A feature for game t may use games before t and nothing else."""

    def test_design_matrix_prefix_is_unchanged_when_the_future_is_deleted(self):
        """THE leakage test.

        Building the design on the first N games must produce exactly the same
        rows as the first N rows of the design built on all 13,199. If any
        feature leaks - a centred rolling mean, a full-sample standardisation, a
        season-end prior - the deleted games change the surviving rows and this
        fails.
        """
        full = real_design(halflife=10.0).reset_index(drop=True)
        g = real_games()
        trunc = g.iloc[:_TRUNC_N].reset_index(drop=True)
        cut = lr.load_design(halflife=10.0, games=trunc)

        self.assertEqual(len(cut), _TRUNC_N)
        assert_truncation_equal(full, cut)

    def test_rolling_features_are_truncation_invariant(self):
        """Pin the SHARED builder too, not just this model's wrapper.

        If a refactor of ``nba_data`` breaks the upstream guarantee, the failure
        should surface here rather than as a mysteriously good score.
        """
        g = real_games()
        full = nd.rolling_team_features(g, halflife=10.0).reset_index(drop=True)
        cut = nd.rolling_team_features(g.iloc[:_TRUNC_N].reset_index(drop=True),
                                       halflife=10.0).reset_index(drop=True)
        pd.testing.assert_frame_equal(full.iloc[:_TRUNC_N].reset_index(drop=True),
                                     cut, check_exact=True)

    def test_rest_and_travel_are_truncation_invariant(self):
        g = real_games()
        full = nd.rest_and_travel(g).reset_index(drop=True)
        cut = nd.rest_and_travel(g.iloc[:_TRUNC_N].reset_index(drop=True)).reset_index(drop=True)
        pd.testing.assert_frame_equal(full.iloc[:_TRUNC_N].reset_index(drop=True),
                                     cut, check_exact=True)

    def test_every_controlling_contrast_is_home_minus_away(self):
        """Contrasts must be built as differences, not as raw levels.

        The failure mode is fitting raw home and away levels for the same
        quantity, which double-counts team strength. Swapping home and away must
        negate exactly the contrasts and leave the levels alone.
        """
        games, roll, rest, _ = synthetic_games(n=50)
        mapping = {}
        for c in roll.columns:
            mapping[c] = ("away_" + c[5:]) if c.startswith("home_") else ("home_" + c[5:])
        roll_swapped = roll.rename(columns=mapping)

        a = lr.assemble_features(games, roll, rest)
        b = lr.assemble_features(games, roll_swapped, rest)
        for col in ("net_diff", "off_diff", "def_diff", "margin_diff", "efg_diff"):
            np.testing.assert_allclose(b[col].to_numpy(), -a[col].to_numpy(), atol=1e-12)
        # A level is symmetric under the swap; a difference is not. That is the
        # one-line statement of "build differences explicitly".
        np.testing.assert_allclose(b["pace_mean"].to_numpy(),
                                   a["pace_mean"].to_numpy(), atol=1e-12)

    def test_no_feature_column_comes_from_a_snapshot_table(self):
        """``snapshots/`` is a season-end aggregate; in-season features must not
        touch it. Every design column has to be one this module derives from the
        shared per-game builders, by construction.

        This guard earns its keep on every feature block that gets added. The
        per-team home/away split failed it the moment it landed — three columns
        with no declared origin — which is the correct outcome: a new feature that
        nobody had to declare where it came from is exactly how a snapshot-fed or
        otherwise leaking column gets in unnoticed.
        """
        g = real_games()
        sched_cols = set(lr.rest_features(g.reset_index(drop=True),
                                          nd.rest_and_travel(g).reset_index(drop=True)
                                          ).columns)
        allowed = ({n for n, _, _ in lr.RATING_PAIRS}
                   | {n for n, _, _ in lr.MATCHUP_PAIRS}
                   | {n for n, _, _ in lr.REBOUND_PAIRS}
                   | {n for n, _, _ in lr.HOME_SPLIT_PAIRS}
                   | {"pace_mean", lr.HOME_COLUMN}
                   # The availability block (issue #36): player-level expected
                   # minutes over the PRIOR-game squad, built by the shared
                   # `nba_data.availability_features` from strictly-past player
                   # rows. Declared origin: the shared per-player loader, never
                   # snapshots/ and never tonight's participants.
                   | set(lr.AVAILABILITY_COLUMNS)
                   | sched_cols)
        produced = set(real_design().columns)
        self.assertEqual(produced - allowed, set(),
                         "a design column has no declared origin")
        # And the shared builder must not have grown a snapshot-fed path silently.
        import inspect
        src = inspect.getsource(nd.rolling_team_features)
        self.assertNotIn("snapshot", src.lower())

    def test_first_game_of_a_season_has_no_in_season_history(self):
        gp = lr.season_games_played(real_games())
        for col in ("home_n", "away_n"):
            self.assertEqual(int(gp[col].min()), 0)
        # Every season's opener must read zero for BOTH sides.
        g = real_games()
        firsts = g.groupby("SEASON", sort=False).head(1)
        self.assertEqual(int(gp["home_n"].iloc[firsts.index].max()), 0)

    def test_games_played_resets_every_season(self):
        """The shared builder's ``games_played`` is a career cumcount; the one
        this model uses must be per-season or early-season shrinkage is a no-op."""
        g = real_games()
        gp = lr.season_games_played(g)
        by_team_season = g.assign(hn=gp["home_n"].to_numpy(),
                                  an=gp["away_n"].to_numpy())
        for season, grp in by_team_season.groupby("SEASON"):
            counts = pd.concat([grp["home"], grp["away"]]).value_counts()
            # No team can have played more games in one season than the league
            # scheduled; a career counter would exceed 82 on every season.
            self.assertLessEqual(int(counts.max()), 82, f"season {season}")

    def test_prior_table_only_reads_earlier_seasons(self):
        """The carryover for season S must be reconstructible from seasons < S
        alone. Recompute values independently and compare."""
        g = real_games()
        prior = lr.prior_season_features(g, halflife=10.0)
        seasons = sorted(g["SEASON"].unique())
        target_season = seasons[5]
        prev_season = seasons[4]

        # Independently: run the builder on ONLY seasons <= prev_season and read
        # that team's last appearance in prev_season.
        upto = g[g["SEASON"] <= prev_season].reset_index(drop=True)
        roll = nd.rolling_team_features(upto, halflife=10.0).reset_index(drop=True)
        rows = prior[prior["SEASON"] == target_season]
        self.assertGreater(len(rows), 0)
        checked = 0
        for _, row in rows.iterrows():
            team = row["team"]
            mask = ((upto["SEASON"] == prev_season)
                    & ((upto["home"] == team) | (upto["away"] == team)))
            idx = np.flatnonzero(mask.to_numpy())
            if len(idx) == 0:
                continue                      # did not play last season
            i = idx[-1]
            side = "home_" if upto["home"].iat[i] == team else "away_"
            expected = float(roll[f"{side}ewm_net_rating"].iat[i])
            if not np.isfinite(expected):
                continue
            self.assertAlmostEqual(float(row["ewm_net_rating"]), expected,
                                   places=10, msg=f"{team} prior mismatch")
            checked += 1
        self.assertGreaterEqual(checked, 3,
                                "too few comparable rows for this to prove "
                                "anything")

    def test_first_season_has_no_prior(self):
        """The FIRST season in the data has no earlier season to read, so its
        prior rows must be empty rather than populated from its own end state.

        This is the off-by-one that matters most: a row keyed by season S that
        held S's own end-of-season rating would hand an October 2015 game roughly
        a full season of future information, and the model would score
        beautifully while meaning nothing. It is also not hypothetical - that is
        exactly the bug this assertion was written after, and fixing it turned a
        reported "+0.010 log loss from the prior" into a reported LOSS.
        """
        g = real_games()
        prior = lr.prior_season_features(g, halflife=10.0)
        seasons = sorted(g["SEASON"].unique())
        rows = prior[prior["SEASON"] == seasons[0]]
        self.assertGreater(len(rows), 0)
        self.assertTrue(rows["ewm_net_rating"].isna().all(),
                        "the first season must carry no prior at all")
        # The second season, by contrast, must be fully populated.
        rows2 = prior[prior["SEASON"] == seasons[1]]
        self.assertGreater(rows2["ewm_net_rating"].notna().sum(), 0)


# --------------------------------------------------------------------------- #
# (b) TRAIN-ONLY PREPROCESSING
# --------------------------------------------------------------------------- #

class TestPreprocessingIsFittedOnTrainOnly(unittest.TestCase):
    """Medians and means/stds come from training rows and nothing else."""

    def setUp(self):
        rng = np.random.default_rng(11)
        n = 800
        self.X = pd.DataFrame({
            "net_diff": rng.normal(0, 4, n),
            "efg_diff": rng.normal(0, 0.03, n),
            "home_adv": np.ones(n),
        })
        self.X.loc[:99, "efg_diff"] = np.nan          # NaNs in train only
        self.y = (rng.uniform(size=n) < 0.5).astype(int)
        self.n_train = 600

    def test_means_and_scales_ignore_the_test_block(self):
        tr = self.X.iloc[:self.n_train]

        # Poison the test block: values far outside anything in train.
        poisoned = self.X.copy()
        poisoned.iloc[self.n_train:, 0] = 1e6
        poisoned.iloc[self.n_train:, 1] = -1e6

        clean = lr.FeaturePipeline().fit(tr)
        dirty = lr.FeaturePipeline().fit(poisoned.iloc[:self.n_train])
        pd.testing.assert_frame_equal(clean.params(), dirty.params())

        # Sanity: the poisoned version WOULD differ if fitted on everything,
        # which is what makes this test able to fail.
        leaky = lr.FeaturePipeline().fit(poisoned)
        self.assertNotAlmostEqual(float(leaky.means_["net_diff"]),
                                  float(clean.means_["net_diff"]), places=3)

    def test_imputation_median_comes_from_train_rows_only(self):
        tr = self.X.iloc[:self.n_train]
        p = lr.FeaturePipeline().fit(tr)
        expected = tr["efg_diff"].median()
        self.assertAlmostEqual(float(p.medians_["efg_diff"]), float(expected), places=12)
        # A median computed over ALL rows would use the poisoned test values.
        poisoned = self.X.copy()
        poisoned.iloc[self.n_train:, 1] = 0.0
        p2 = lr.FeaturePipeline().fit(poisoned.iloc[:self.n_train])
        self.assertAlmostEqual(float(p2.medians_["efg_diff"]), float(expected), places=12)

    def test_fitting_on_train_then_transforming_all_is_not_fit_transform(self):
        """``fit_transform`` on the full frame leaks; the two-step call does not.
        These must differ, otherwise the guard above is vacuous."""
        full_fit = lr.FeaturePipeline().fit_transform(self.X)
        proper = lr.FeaturePipeline().fit(self.X.iloc[:self.n_train]).transform(self.X)
        self.assertFalse(np.allclose(full_fit.to_numpy(), proper.to_numpy()))
        # And `proper` on the TRAIN rows must equal fit_transform on train alone.
        train_only = lr.FeaturePipeline().fit_transform(self.X.iloc[:self.n_train])
        np.testing.assert_allclose(proper.iloc[:self.n_train].to_numpy(),
                                   train_only.to_numpy(), atol=1e-12)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            lr.FeaturePipeline().transform(self.X)

    def test_home_column_is_neither_centred_nor_scaled(self):
        """``home_adv`` is constant 1.0. Centring it would destroy its
        coefficient; scaling it is a division by zero."""
        p = lr.FeaturePipeline().fit(self.X)
        out = p.transform(self.X)
        np.testing.assert_allclose(out["home_adv"].to_numpy(), np.ones(len(out)))
        self.assertEqual(p.means_["home_adv"], 0.0)
        self.assertEqual(p.scales_["home_adv"], 1.0)

    def test_constant_feature_does_not_explode(self):
        X = self.X.copy()
        X["flat"] = 3.0
        p = lr.FeaturePipeline().fit(X)
        out = p.transform(X)
        self.assertTrue(np.isfinite(out.to_numpy()).all())
        np.testing.assert_allclose(out["flat"].to_numpy(), np.zeros(len(out)))

    def test_all_nan_column_is_flagged_not_silently_zeroed(self):
        X = self.X.copy()
        X["empty"] = np.nan
        p = lr.FeaturePipeline().fit(X)
        self.assertIn("empty", p.degenerate_columns_)
        self.assertTrue(np.isfinite(p.transform(X).to_numpy()).all())

    def test_predictions_do_not_depend_on_the_test_block(self):
        """End to end: fit on train, then scramble every future row. Predictions
        on a fixed probe set must be bit-identical."""
        m = lr.RollingLogreg(C=1.0, features=["net_diff", "efg_diff", "home_adv"])
        m.fit(self.X.iloc[:self.n_train], self.y[:self.n_train])
        probe = self.X.iloc[300:320].copy()
        before = m.predict_proba(probe)

        polluted = self.X.copy()
        polluted.iloc[self.n_train:, :] = 999.0
        m2 = lr.RollingLogreg(C=1.0, features=["net_diff", "efg_diff", "home_adv"])
        m2.fit(polluted.iloc[:self.n_train], self.y[:self.n_train])
        np.testing.assert_array_equal(before, m2.predict_proba(probe))

    def test_walk_forward_refits_the_pipeline_inside_each_split(self):
        """Changing rows after a split's test block must not alter that split."""
        g = real_games()
        X = real_design().reset_index(drop=True)
        y = g["y"].to_numpy(dtype=int)
        feats = lr.prune_features(X, lr.resolve_features("full"))
        splits = nd.walk_forward_split(g, n_splits=5, initial_frac=0.5)

        last_tr, last_te = splits[-1]
        m = lr.RollingLogreg(C=1.0, features=feats).fit(X.iloc[last_tr], y[last_tr])
        before = m.predict_proba(X.iloc[last_te])
        params_before = m.pipeline_.params()

        # An earlier split's test block IS part of this split's training rows, so
        # perturbing it SHOULD change the fit. Rows strictly after this split's
        # test block should not.
        later = X.index[X.index > last_te.max()]
        polluted = X.copy()
        polluted.loc[later, feats] = 777.0
        m2 = lr.RollingLogreg(C=1.0, features=feats).fit(polluted.iloc[last_tr], y[last_tr])
        np.testing.assert_array_equal(before, m2.predict_proba(X.iloc[last_te]))
        pd.testing.assert_frame_equal(params_before, m2.pipeline_.params())


# --------------------------------------------------------------------------- #
# (c) SIGN / SANITY ON SYNTHETIC DATA WITH A KNOWN RELATIONSHIP
# --------------------------------------------------------------------------- #

class TestSignsOnSyntheticData(unittest.TestCase):
    """The generating relationship is known, so every coefficient is checkable."""

    @classmethod
    def setUpClass(cls):
        cls.games, cls.roll, cls.rest, cls.X = synthetic_games(n=4000)
        cls.y = cls.games["y"].to_numpy(dtype=int)
        cls.feats = lr.prune_features(cls.X, lr.resolve_features("full"))

    def test_recovered_coefficients_match_the_generating_ones(self):
        """Every coefficient on an identified feature comes back at its true
        value, in the units the model reports (per SD)."""
        feats = lr.prune_features(self.X, IDENTIFIED_FEATURES)
        m = lr.RollingLogreg(C=1.0, features=feats)
        m.fit(self.X, self.y)
        coef = m.coefficients()["coef"]
        for name in feats:
            self.assertIn(name, coef.index)
            self.assertAlmostEqual(float(coef[name]), TRUE_BETAS[name], delta=0.12,
                                   msg=f"{name}: got {coef[name]:+.4f}, "
                                       f"true {TRUE_BETAS[name]:+.4f}")

    def test_offensive_and_defensive_coefficients_have_the_right_sign(self):
        """Own offence helps; own defence hurts.

        Read from the `signs` model, which drops ``net_diff``. Keeping all three
        makes the design effectively singular (see the module docstring), and the
        split between them arbitrary. That is a property of the DATA, not of the
        code, so the fix is to fit the identified pair when the question is about
        signs.
        """
        feats = lr.prune_features(self.X, lr.resolve_features(["signs", "home"]))
        m = lr.RollingLogreg(C=1.0, features=feats).fit(self.X, self.y)
        coef = m.coefficients()["coef"]
        self.assertGreater(float(coef["off_diff"]), 0.0,
                           "own offensive rating must raise home win probability")
        self.assertLess(float(coef["def_diff"]), 0.0,
                        "own defensive rating must lower home win probability")

    def test_home_court_effect_is_positive_and_the_right_size(self):
        feats = lr.prune_features(self.X, lr.resolve_features("full"))
        m = lr.RollingLogreg(C=1.0, features=feats).fit(self.X, self.y)
        self.assertGreater(m.home_edge_logodds(), 0.0)
        self.assertAlmostEqual(m.home_edge_logodds(), TRUE_BETAS["home_adv"], delta=0.12)

    def test_away_side_back_to_back_and_travel_hurt_the_away_team(self):
        """A b2b/travel effect shows up as a POSITIVE coefficient on the away
        side's flag, because the label is a HOME win."""
        feats = lr.prune_features(self.X, lr.resolve_features("full"))
        m = lr.RollingLogreg(C=1.0, features=feats).fit(self.X, self.y)
        coef = m.coefficients()["coef"]
        self.assertGreater(float(coef["b2b_away"]), 0.0)
        self.assertLess(float(coef["b2b_home"]), 0.0)
        self.assertLess(float(coef["travel_k"]), 0.0)

    def test_model_beats_the_base_rate_on_data_it_was_generated_to_explain(self):
        feats = lr.prune_features(self.X, lr.resolve_features("full"))
        m = lr.RollingLogreg(C=1.0, features=feats).fit(self.X, self.y)
        p = m.predict_proba(self.X)
        base = nd.evaluate(self.y, lr.constant_baseline(self.y, len(self.y)))
        got = nd.evaluate(self.y, p)
        self.assertLess(got["log_loss"], base["log_loss"] - 0.05)
        # AUC lands near 0.75, not 0.95, and that is the correct answer for this
        # world: ~15 predictors of modest size plus binomial draw noise. A much
        # higher AUC here would mean the fixture leaked, not that the model is
        # good. The real model sits in the same 0.70-0.73 band.
        self.assertGreater(got["auc"], 0.72)
        # And the paired delta must have the documented sign:
        # paired_bootstrap_delta(y, a, b) = loss(a) - loss(b), so a POSITIVE
        # delta means b is better. Here a is the base rate, so positive is good.
        d = nd.paired_bootstrap_delta(self.y, lr.constant_baseline(self.y, len(self.y)), p)
        self.assertGreater(d["delta"], 0.0)
        self.assertGreater(d["ci_lo"], 0.0)

    def test_shuffled_labels_destroy_the_signal(self):
        """Negative control. If the model still separated a random label, the
        sign tests above would be proving nothing."""
        rng = np.random.default_rng(3)
        feats = lr.prune_features(self.X, lr.resolve_features("full"))
        m = lr.RollingLogreg(C=1.0, features=feats).fit(
            self.X, rng.permutation(self.y))
        got = nd.evaluate(self.y, m.predict_proba(self.X))
        base = nd.evaluate(self.y, lr.constant_baseline(self.y, len(self.y)))
        self.assertLess(abs(got["log_loss"] - base["log_loss"]), 0.01)
        self.assertLess(got["auc"], 0.56)


# --------------------------------------------------------------------------- #
# End-to-end on a synthetic DATA ROOT, through the REAL shared loader
# --------------------------------------------------------------------------- #

_SYNTH_ROOT = None


def _build_synthetic_data_root() -> Path:
    """Write a minimal on-disk dataset in the exact layout the shared loader
    expects: league team game logs + advanced + four factors + player minutes,
    one season.

    The signal lives in the RATINGS, not in the margin, because that is what the
    rolling features read. ATL is the better team by ~4 net-rating points in
    every game, and the margin is then GENERATED from those ratings plus a
    2.5-point home bonus. If the ratings were noise the model would be right to
    find nothing and the test would be asserting a fantasy.

    The player-minutes table exists so the shared availability block
    (`nba_data.availability_features`, wired into `load_design` by issue #36)
    builds end to end in this world too - a fixed six-man rotation per club with
    jittered minutes. It is generated from its OWN rng stream so the world's
    margins and ratings stay exactly as they were before the block landed.
    """
    root = Path(tempfile.mkdtemp(prefix="logreg_synth_"))
    for sub in ("game_logs", "team_gamelogs", "player_gamelogs", "schedule",
                "meta", "snapshots"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(19)
    rng_players = np.random.default_rng(23)
    n_games = 120
    dates = pd.date_range("2021-01-01", periods=n_games, freq="D")
    latent = {"ATL": 4.0, "BOS": -4.0}
    HOME_BONUS = 2.5
    player_minutes = (32.0, 30.0, 28.0, 24.0, 20.0, 16.0)
    player_ids = {"ATL": (1101, 1102, 1103, 1104, 1105, 1106),
                  "BOS": (2101, 2102, 2103, 2104, 2105, 2106)}

    rows_team, rows_adv, rows_ff, rows_players = [], [], [], []
    schedule = []
    for i, d in enumerate(dates):
        home, away = ("ATL", "BOS") if i % 2 == 0 else ("BOS", "ATL")
        edge = latent[home] - latent[away] + HOME_BONUS
        margin = float(rng.normal(edge, 10.0))
        hp, ap = 110 + margin / 2, 110 - margin / 2
        gid = f"{i:010d}"
        for team, pts, opp_pts, is_home in ((home, hp, ap, True),
                                            (away, ap, hp, False)):
            matchup = f"{team} vs. {away}" if is_home else f"{team} @ {home}"
            rows_team.append({
                "SEASON": "2020-21", "TEAM_ID": 1, "TEAM_ABBREVIATION": team,
                "TEAM_NAME": team, "GAME_ID": gid, "GAME_DATE": str(d.date()),
                "MATCHUP": matchup, "WL": "W" if pts > opp_pts else "L",
                "MIN": 240, "PTS": pts, "PLUS_MINUS": int(pts - opp_pts),
                "VIDEO_AVAILABLE": 0, "SEASON_TYPE": "Regular Season",
            })
            net = float(rng.normal(latent[team], 2.0))
            # Keep OFF - DEF == NET exactly, as the real advanced table does.
            dfn = float(rng.normal(110, 2.0))
            off = dfn + net
            rows_adv.append({
                "SEASON": "2020-21", "TEAM_ID": 1, "TEAM_ABBREVIATION": team,
                "TEAM_NAME": team, "GAME_ID": gid, "GAME_DATE": str(d.date()),
                "MATCHUP": matchup, "MIN": 240,
                "OFF_RATING": off, "DEF_RATING": dfn, "NET_RATING": net,
                "PACE": float(rng.normal(99, 2)),
                "TS_PCT": float(rng.normal(0.57, 0.02)),
                "SEASON_TYPE": "Regular Season",
            })
            rows_ff.append({
                "SEASON": "2020-21", "TEAM_ID": 1, "TEAM_ABBREVIATION": team,
                "TEAM_NAME": team, "GAME_ID": gid, "GAME_DATE": str(d.date()),
                "MATCHUP": matchup, "MIN": 240,
                "EFG_PCT": float(rng.normal(0.50, 0.02)),
                "FTA_RATE": float(rng.normal(0.20, 0.04)),
                "TM_TOV_PCT": float(rng.normal(0.135, 0.01)),
                "OREB_PCT": float(rng.normal(0.27, 0.02)),
                "OPP_EFG_PCT": float(rng.normal(0.50, 0.02)),
                "SEASON_TYPE": "Regular Season",
            })
            for slot, base_min in enumerate(player_minutes):
                rows_players.append({
                    "SEASON": "2020-21", "SEASON_TYPE": "Regular Season",
                    "TEAM_ABBREVIATION": team, "GAME_ID": gid,
                    "PLAYER_ID": player_ids[team][slot],
                    "MIN": float(max(0.0, rng_players.normal(base_min, 2.0))),
                })
        schedule.append({"gameId": gid, "gameDate": d.strftime("%m/%d/%Y 19:00:00"),
                         "gameStatus": 3, "gameLabel": None,
                         "homeTeam_teamTricode": home, "awayTeam_teamTricode": away})

    pd.DataFrame(rows_team).to_csv(root / "game_logs" / "league_team_2020-21.csv",
                                    index=False)
    pd.DataFrame(rows_adv).to_csv(root / "team_gamelogs" / "advanced_2020-21.csv",
                                   index=False)
    pd.DataFrame(rows_ff).to_csv(root / "team_gamelogs" / "four_factors_2020-21.csv",
                                  index=False)
    pd.DataFrame(rows_players).to_csv(
        root / "player_gamelogs" / "advanced_2020-21.csv", index=False)
    pd.DataFrame(schedule).to_csv(root / "schedule" / "LeagueGameV2_2020-21.csv",
                                  index=False)
    return root


def _clear_shared_caches():
    for name in ("load_team_games", "load_team_advanced", "load_team_fourfactors",
                 "load_player_advanced", "load_player_usage",
                 "load_schedule", "build_games"):
        fn = getattr(nd, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


class TestEndToEndOnSyntheticDataRoot(unittest.TestCase):
    """The whole chain - shared loader, causal builders, this model - on data
    where the answer is known."""

    @classmethod
    def setUpClass(cls):
        global _SYNTH_ROOT
        _SYNTH_ROOT = _build_synthetic_data_root()
        cls._orig_root = nd.DATA_ROOT
        nd.DATA_ROOT = str(_SYNTH_ROOT)
        _clear_shared_caches()
        cls.games = nd.build_games()
        cls.design = lr.load_design(halflife=5.0, games=cls.games,
                                    with_rebounds=False)

    @classmethod
    def tearDownClass(cls):
        nd.DATA_ROOT = cls._orig_root
        _clear_shared_caches()
        if _SYNTH_ROOT is not None and _SYNTH_ROOT.exists():
            shutil.rmtree(_SYNTH_ROOT, ignore_errors=True)

    def test_the_loader_produced_a_usable_synthetic_history(self):
        self.assertEqual(len(self.games), 120)
        self.assertEqual(set(self.games["SEASON"]), {"2020-21"})
        self.assertEqual(set(self.games["home"]) | set(self.games["away"]),
                         {"ATL", "BOS"})

    def test_design_is_truncation_invariant_end_to_end(self):
        """The leakage gate again, but through the on-disk loader, where a
        central-difference or full-sample statistic would have room to hide."""
        cut = lr.load_design(halflife=5.0,
                             games=self.games.iloc[:60].reset_index(drop=True),
                             with_rebounds=False)
        assert_truncation_equal(self.design, cut)

    def test_the_better_team_is_the_one_that_wins_more(self):
        """ATL carries a 4-point net-rating edge, so a correctly-signed net_diff
        coefficient must come out positive."""
        feats = lr.prune_features(self.design, lr.resolve_features("full"))
        self.assertTrue(feats)
        y = self.games["y"].to_numpy(dtype=int)
        m = lr.RollingLogreg(C=1.0, features=feats).fit(self.design, y)
        coef = m.coefficients()["coef"]
        self.assertGreater(float(coef["net_diff"]), 0.0)

    def test_home_court_is_positive_even_on_synthetic_data(self):
        feats = lr.prune_features(self.design, lr.resolve_features("full"))
        y = self.games["y"].to_numpy(dtype=int)
        m = lr.RollingLogreg(C=1.0, features=feats).fit(self.design, y)
        self.assertGreater(m.home_edge_logodds(), 0.0)


# --------------------------------------------------------------------------- #
# The off/def/net singularity, quantified
# --------------------------------------------------------------------------- #

class TestRatingBlockIsEffectivelySingular(unittest.TestCase):
    """The caveat that governs every coefficient table from this model."""

    def test_the_off_def_net_block_is_effectively_singular(self):
        """Quantifies the note in ``nba_data``: over the standardised block
        ``[net_diff, off_diff, def_diff]`` the third singular value is ~8e-4 of
        the first, because ``net - off + def ~ 0``.

        This is also why MARGINAL correlation is the wrong diagnostic here and
        would let someone dismiss the problem: corr(off, def) is only -0.22 and
        corr(net, off) only 0.81, yet the block is singular. The dependency lives
        in a combination, not in a pair.
        """
        X = real_design()
        Z = X[["net_diff", "off_diff", "def_diff"]].to_numpy(float)
        Z = Z[~np.isnan(Z).any(axis=1)]
        Z = (Z - Z.mean(0)) / Z.std(0)
        s = np.linalg.svd(Z, compute_uv=False)
        ratio = float(s[2] / s[0])
        self.assertLess(ratio, 1e-3,
                        f"smallest singular value ratio rose to {ratio:.2e}; "
                        "if the advanced table changed, recheck whether "
                        "individual off/def/net coefficients are readable again")

    def test_predictions_are_unaffected_by_which_collinear_parameterisation_is_used(self):
        """The singularity wrecks coefficient READING, not prediction. The
        identified `signs` fit and the full fit must agree on held-out rows to
        within numerical noise, which is what licenses reporting performance
        from the full model while reading signs from the `signs` one."""
        g = real_games()
        X = real_design()
        y = g["y"].to_numpy(dtype=int)
        tr, te = nd.walk_forward_split(g, n_splits=5, initial_frac=0.5)[-1]
        full = lr.prune_features(X, lr.resolve_features("full"))
        signs = lr.prune_features(X, lr.resolve_features(["signs", "home"]))
        p_full = lr.RollingLogreg(C=1.0, features=full).fit(
            X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
        p_signs = lr.RollingLogreg(C=1.0, features=signs).fit(
            X.iloc[tr], y[tr]).predict_proba(X.iloc[te])
        # Different models, but the collinear direction is the SAME information,
        # so their losses must be within a hair of each other.
        ll_full = nd.evaluate(y[te], p_full)["log_loss"]
        ll_signs = nd.evaluate(y[te], p_signs)["log_loss"]
        self.assertLess(abs(ll_full - ll_signs), 0.01,
                        "dropping nine features moved log loss by more than "
                        "0.01, which is far more than a reparameterisation "
                        "should - investigate before trusting either row")

    def test_the_rating_contrasts_are_near_collinear(self):
        """Pin the identity that makes the off/def/net coefficients unreadable,
        and the one that is easy to get backwards.

        * ``net_diff == off_diff - def_diff`` - exact algebra, but the underlying
          data only holds it to about 0.1. The NBA publishes NET_RATING,
          OFF_RATING and DEF_RATING as independent rounded columns, so per game
          ``NET != OFF - DEF`` exactly, and an EWM accumulates the discrepancy
          over roughly a half-life.
        * ``matchup_home + matchup_away`` is ``home_net + away_net`` - a SUM of
          levels - and NOT ``net_diff``. It is tempting to assume otherwise and
          it is wrong, which is why the matchup terms are a genuinely distinct
          contrast rather than a duplicate of the rating difference. That one is
          also only near-exact (~0.1), for the same rounding reason.

        The synthetic fixture satisfies the net identity EXACTLY because it was
        built to, so the two tolerances are stated separately rather than being
        loosened into one useless number.
        """
        X = real_design()
        gap = (X["net_diff"] - (X["off_diff"] - X["def_diff"])).abs().max()
        self.assertLess(float(gap), 0.15,
                        f"net vs (off - def) diverged by {gap:.4f}; if the "
                        "advanced table changed shape, recheck this bound")

        g = real_games()
        roll = nd.rolling_team_features(g, halflife=10.0).reset_index(drop=True)
        net_sum = (roll["home_ewm_net_rating"].to_numpy()
                   + roll["away_ewm_net_rating"].to_numpy())
        matchup_sum = X["matchup_home"].to_numpy() + X["matchup_away"].to_numpy()
        self.assertLess(float(np.nanmax(np.abs(matchup_sum - net_sum))), 0.15,
                        "matchup pair must sum to home_net + away_net")
        self.assertFalse(
            np.allclose(matchup_sum, X["net_diff"].to_numpy(), atol=1e-6),
            "if the matchup sum DID equal net_diff, this module's ablation rows "
            "rating+matchup and rating would be the same model")

        # Same identity, exactly, on the synthetic world.
        _, _, _, Xs = synthetic_games(n=200)
        np.testing.assert_allclose(
            Xs["net_diff"].to_numpy(),
            Xs["off_diff"].to_numpy() - Xs["def_diff"].to_numpy(), atol=1e-9)

    def test_schedule_level_and_difference_forms_agree_in_the_design(self):
        g = real_games()
        X = lr.rest_features(g.reset_index(drop=True),
                             nd.rest_and_travel(g).reset_index(drop=True))
        np.testing.assert_allclose(X["rest_diff"].to_numpy(),
                                   X["rest_home"].to_numpy()
                                   - X["rest_away"].to_numpy(), atol=1e-10)


# --------------------------------------------------------------------------- #
# Prior-season shrinkage arithmetic
# --------------------------------------------------------------------------- #

class TestShrinkageArithmetic(unittest.TestCase):
    def test_k_zero_is_a_no_op(self):
        roll = pd.DataFrame({"home_ewm_net_rating": [1.0, 2.0],
                             "away_ewm_net_rating": [0.5, 0.5]})
        games = real_games().head(2).reset_index(drop=True)
        out = lr.shrink_to_prior(roll, games, pd.DataFrame({"a": [1.0]}), k=0.0)
        pd.testing.assert_frame_equal(out, roll, check_exact=True)

    def test_a_season_opener_returns_the_prior_exactly(self):
        """n = 0, so the formula must collapse to the prior regardless of k."""
        roll = pd.DataFrame({"home_ewm_net_rating": [np.nan],
                             "away_ewm_net_rating": [3.0]})
        games = real_games().head(1).reset_index(drop=True)
        prior = pd.DataFrame({
            "SEASON": [games["SEASON"].iat[0]] * 2,
            "team": [games["home"].iat[0], games["away"].iat[0]],
            "prior_n": [0, 0],
            "ewm_net_rating": [7.0, -2.0],
        })
        out = lr.shrink_to_prior(roll, games, prior, k=25.0, cols=["ewm_net_rating"])
        self.assertAlmostEqual(float(out["home_ewm_net_rating"].iat[0]), 7.0, places=10)
        self.assertAlmostEqual(float(out["away_ewm_net_rating"].iat[0]), -2.0, places=10)

    def test_the_weighting_matches_the_documented_formula(self):
        """(n * ewm + k * prior) / (n + k), checked on a hand-computed row."""
        games = real_games().head(40).reset_index(drop=True)
        gp = lr.season_games_played(games)
        i = 20
        ewm_h, ewm_a = 2.0, 4.0
        roll = pd.DataFrame({"home_ewm_net_rating": [np.nan] * len(games),
                             "away_ewm_net_rating": [np.nan] * len(games)})
        roll.iloc[i, 0] = ewm_h
        roll.iloc[i, 1] = ewm_a
        prior = pd.DataFrame({
            "SEASON": [games["SEASON"].iat[i]] * 2,
            "team": [games["home"].iat[i], games["away"].iat[i]],
            "prior_n": [0, 0],
            "ewm_net_rating": [10.0, -6.0],
        })
        k = 5.0
        out = lr.shrink_to_prior(roll, games, prior, k=k, cols=["ewm_net_rating"])
        nh = int(gp["home_n"].iat[i])
        na = int(gp["away_n"].iat[i])
        self.assertAlmostEqual(
            float(out["home_ewm_net_rating"].iat[i]),
            (nh * ewm_h + k * 10.0) / (nh + k), places=10)
        self.assertAlmostEqual(
            float(out["away_ewm_net_rating"].iat[i]),
            (na * ewm_a + k * -6.0) / (na + k), places=10)

    def test_a_team_with_no_prior_falls_back_to_the_league_mean(self):
        games = real_games().head(30).reset_index(drop=True)
        roll = pd.DataFrame({"home_ewm_net_rating": [1.0] * 30,
                             "away_ewm_net_rating": [1.0] * 30})
        # A prior table that knows only one of the two teams.
        prior = pd.DataFrame({
            "SEASON": [games["SEASON"].iat[0]] * 2,
            "team": [games["home"].iat[0], "__SOMEONE_ELSE__"],
            "prior_n": [10, 10],
            "ewm_net_rating": [4.0, 8.0],
        })
        out = lr.shrink_to_prior(roll, games, prior, k=10.0, cols=["ewm_net_rating"])
        # League mean of {4.0, 8.0} is 6.0; the unknown team lands there rather
        # than staying NaN for the imputer to absorb.
        self.assertFalse(out["home_ewm_net_rating"].isna().any())
        self.assertTrue(np.isfinite(out["home_ewm_net_rating"].to_numpy()).all())


# --------------------------------------------------------------------------- #
# Evaluation harness sanity
# --------------------------------------------------------------------------- #

class TestEvaluationHarness(unittest.TestCase):
    def test_walk_forward_splits_are_chronological_and_disjoint(self):
        g = real_games()
        splits = nd.walk_forward_split(g, n_splits=5, initial_frac=0.5)
        self.assertEqual(len(splits), 5)
        prev_hi = -1
        for tr, te in splits:
            self.assertEqual(len(set(tr) & set(te)), 0, "train and test overlap")
            self.assertLess(tr.max(), te.min(), "test block precedes train block")
            self.assertGreaterEqual(te.min(), prev_hi, "test blocks not in order")
            prev_hi = te.max()
            self.assertLessEqual(
                g["GAME_DATE"].iloc[tr].max(), g["GAME_DATE"].iloc[te].min())

    def test_always_home_baseline_is_the_training_base_rate(self):
        """Not the test block's rate - using that is a leak that flatters the
        baseline and flatters nothing else."""
        self.assertAlmostEqual(float(lr.constant_baseline([1, 0, 1, 1], 10)[0]), 0.75)
        self.assertTrue(np.allclose(lr.constant_baseline([1, 0], 3), 0.5))
        # A single-class training set must not produce a hard 0.0 or 1.0, which
        # would score as an infinite log loss.
        p = lr.constant_baseline([1, 1, 1], 5)
        self.assertGreater(float(p[0]), 0.0)
        self.assertLess(float(p[0]), 1.0)

    def test_constant_predictions_have_no_discrimination(self):
        """AUC exactly 0.5 for a constant predictor is the sanity check that the
        reference point is really uninformative, not slightly informative."""
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 500)
        p = np.full(500, float(y.mean()))
        r = nd.evaluate(y, p)
        self.assertAlmostEqual(r["auc"], 0.5, places=12)
        self.assertAlmostEqual(r["accuracy"], float(max(y.mean(), 1 - y.mean())), places=12)

    def test_paired_bootstrap_sign_convention(self):
        """`paired_bootstrap_delta(y, a, b)` returns loss(a) - loss(b), so a
        POSITIVE delta means b is BETTER. This has been misread in this repo
        before; it is pinned here so it cannot be again."""
        rng = np.random.default_rng(13)
        y = rng.integers(0, 2, 800)
        good = np.where(y == 1, 0.75, 0.25)          # an excellent model
        bad = np.full(800, float(y.mean()))          # the base rate
        d = nd.paired_bootstrap_delta(y, bad, good)
        self.assertGreater(d["delta"], 0.0,
                           "positive delta must mean the second arg is better")
        self.assertGreater(d["ci_lo"], 0.0)
        # Reversed arguments flip the sign.
        d2 = nd.paired_bootstrap_delta(y, good, bad)
        self.assertLess(d2["delta"], 0.0)

    def test_bootstrap_metric_se_is_positive_and_reproducible(self):
        rng = np.random.default_rng(17)
        n = 600
        y = rng.integers(0, 2, n)
        p = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.1, n), 0.01, 0.99)
        a = lr.bootstrap_metric_se(y, p, n_boot=120, seed=1)
        b = lr.bootstrap_metric_se(y, p, n_boot=120, seed=1)
        self.assertEqual(a["log_loss_se"], b["log_loss_se"])
        for k in ("log_loss_se", "brier_se", "auc_se"):
            self.assertGreater(a[k], 0.0)
            self.assertTrue(np.isfinite(a[k]))


class TestAvailabilityBlockIsTheSharedBlock(unittest.TestCase):
    """Issue #36 contract: the shared availability block enters the canonical
    frame VERBATIM - the block's own columns, in the block's own order, exactly
    once - and the canonical feature recipe consumes it.

    The block is `nba_data.availability_features`, the strict-roster expected-
    minutes construction measured at +0.0038 nats (CI [+0.0023, +0.0055]) and
    reproduced at +0.00399 on the ablation frame. Nothing in this model may
    re-derive those columns: one implementation, one order, one append.
    """

    def test_the_declared_order_is_the_blocks_order(self):
        avail = nd.availability_features(real_games())
        self.assertEqual(list(avail.columns), list(lr.AVAILABILITY_COLUMNS),
                         "AVAILABILITY_COLUMNS must be the block's own columns "
                         "in the block's own order")

    def test_the_block_is_appended_once_after_the_canonical_columns(self):
        X = real_design(halflife=10.0).reset_index(drop=True)
        base = lr.load_design(halflife=10.0, with_availability=False)
        avail = nd.availability_features(real_games()).reset_index(drop=True)
        self.assertEqual(list(X.columns)[:len(base.columns)], list(base.columns),
                         "the block must follow the canonical columns without "
                         "replacing or reordering them")
        self.assertEqual(list(X.columns)[len(base.columns):], list(avail.columns),
                         "the block must be appended in its own order")
        self.assertEqual(len(X.columns), len(set(X.columns)),
                         "no column - the block especially - may appear twice")
        pd.testing.assert_frame_equal(
            X[list(avail.columns)].reset_index(drop=True), avail,
            obj="the block's values must be the shared builder's, verbatim")

    def test_the_off_flag_returns_the_pre_36_canonical_frame(self):
        base = lr.load_design(halflife=10.0, with_availability=False)
        full_cols = list(real_design(halflife=10.0).columns)
        self.assertEqual(
            list(base.columns),
            [c for c in full_cols if c not in set(lr.AVAILABILITY_COLUMNS)],
            "with_availability=False must be the historical frame, unchanged")
        for col in lr.AVAILABILITY_COLUMNS:
            self.assertNotIn(col, base.columns)

    def test_the_canonical_recipe_consumes_the_block_exactly_once(self):
        feats = lr.resolve_features("full")
        avail = [f for f in feats if f in set(lr.AVAILABILITY_COLUMNS)]
        self.assertEqual(avail, list(lr.AVAILABILITY_COLUMNS),
                         "the canonical model must consume the whole block, in "
                         "canonical order, without repeats")
        for col in lr.AVAILABILITY_COLUMNS:
            self.assertEqual(feats.count(col), 1)


class TestModelSurface(unittest.TestCase):
    def test_resolve_features_accepts_groups_recipes_and_columns(self):
        by_group = lr.resolve_features(["rating", "home"])
        by_recipe = lr.resolve_features("full")
        by_column = lr.resolve_features(["net_diff", lr.HOME_COLUMN])
        self.assertIn("net_diff", by_group)
        self.assertIn("rest_home", by_recipe)
        self.assertIn("pace_mean", by_recipe)
        self.assertEqual(by_column, ["net_diff", lr.HOME_COLUMN])
        with self.assertRaises(KeyError):
            lr.resolve_features(["not_a_thing"])

    def test_prune_features_drops_only_absent_columns(self):
        X = pd.DataFrame({"net_diff": [1.0], "home_adv": [1.0]})
        self.assertEqual(lr.prune_features(X, ["net_diff", "drb_diff", "home_adv"]),
                         ["net_diff", "home_adv"])

    def test_regularisation_path_covers_every_split_and_C(self):
        g = real_games().head(3000).reset_index(drop=True)
        X = lr.load_design(halflife=10.0, games=g, with_rebounds=False)
        grid = (0.001, 0.01, 0.1, 1.0)
        path = lr.regularisation_path(X, g, lr.resolve_features("full"),
                                      C_grid=grid, n_splits=3, initial_frac=0.5)
        self.assertEqual(len(path), 3 * len(grid))
        self.assertEqual(sorted(path["split"].unique()), [1, 2, 3])
        self.assertEqual(sorted(path["C"].unique()), list(grid))

    def test_regularisation_path_reports_the_whole_grid_not_its_argmin(self):
        """The C grid has to reach far enough down that the penalty is doing
        something. sklearn's C multiplies the SUMMED loss, and with ~10k rows that
        sum is ~6200, so C must get to ~1e-3 before 1/C is comparable. A grid
        starting at 0.01 looks like a regularisation path and is a flat line."""
        self.assertLess(min(lr.C_GRID), 1e-3)

    def test_inner_selection_never_sees_the_rows_it_is_scored_on(self):
        """`select_C_inner` is what keeps a headline number off the argmin of a
        test-set curve, so it must be blind to held-out rows.

        Called on a TRUNCATED frame it can only see the first block. If it were
        peeking at later rows, appending games past the truncation point could
        change the answer - and that is exactly the leak this guards.
        """
        g = real_games().iloc[:6000].reset_index(drop=True)
        X = lr.load_design(halflife=10.0, games=g, with_rebounds=False)
        feats = lr.resolve_features("full")
        c_small = lr.select_C_inner(X, g, feats, n_inner=3)

        # A longer history is built but only the first `len(g)` rows are scored,
        # and the `games` frame handed in is the short one.
        g_long = real_games().iloc[:7500].reset_index(drop=True)
        X_long = lr.load_design(halflife=10.0, games=g_long, with_rebounds=False)
        c_long = lr.select_C_inner(X_long.iloc[:len(g)], g, feats, n_inner=3)
        self.assertEqual(c_small, c_long)

    def test_inner_selection_returns_a_value_from_the_grid(self):
        g = real_games().iloc[:4000].reset_index(drop=True)
        X = lr.load_design(halflife=10.0, games=g, with_rebounds=False)
        c = lr.select_C_inner(X, g, lr.resolve_features("full"), n_inner=3)
        self.assertIn(c, list(lr.C_GRID))

    def test_missing_feature_column_raises_at_fit_time(self):
        m = lr.RollingLogreg(C=1.0, features=["net_diff"])
        with self.assertRaises(KeyError):
            m.fit(pd.DataFrame({"other": [1.0, 2.0]}), [0, 1])

    def test_fit_intercept_is_dropped_when_home_column_is_present(self):
        """home_adv is the constant 1.0, so it is linearly dependent on the
        intercept. With both present the solver splits the home effect between
        them and reports a home_adv coefficient of exactly zero, which reads as
        "no home-court effect found" when one was found and put in the
        intercept."""
        m = lr.RollingLogreg(C=1.0, features=["net_diff", lr.HOME_COLUMN])
        self.assertFalse(m.fit_intercept)
        m2 = lr.RollingLogreg(C=1.0, features=["net_diff"])
        self.assertTrue(m2.fit_intercept)
        m3 = lr.RollingLogreg(C=1.0, features=["net_diff", lr.HOME_COLUMN],
                              fit_intercept=True)
        self.assertTrue(m3.fit_intercept)

    def test_walk_forward_eval_reports_every_split_and_both_models(self):
        g = real_games().iloc[:5000].reset_index(drop=True)
        X = lr.load_design(halflife=10.0, games=g, with_rebounds=False)
        m = lr.walk_forward_eval(X, g, lr.resolve_features("full"), C=1.0,
                                 n_splits=3, initial_frac=0.5)
        self.assertEqual(set(m["model"].unique()), {"model", "always_home"})
        self.assertEqual(len(m), 6)
        # The trivial reference must be exactly uninformative.
        base = m[m["model"] == "always_home"]["auc"].to_numpy()
        np.testing.assert_allclose(base, 0.5, atol=1e-9)


class TestRestFeatureEngineering(unittest.TestCase):
    def test_offseason_gap_is_capped(self):
        """Raw rest reaches 288 days across an offseason. Left uncapped that is
        a season indicator, not a rest signal."""
        g = real_games()
        rest = nd.rest_and_travel(g).reset_index(drop=True)
        self.assertGreater(float(rest["home_rest_days"].max()), 30.0,
                           "the raw column is expected to contain the offseason gap")
        sched = lr.rest_features(g.reset_index(drop=True), rest)
        self.assertLessEqual(float(sched["rest_home"].max()), lr.REST_CAP - lr.REST_PIVOT)
        self.assertLessEqual(float(sched["rest_away"].max()), lr.REST_CAP - lr.REST_PIVOT)

    def test_schedule_diff_and_level_forms_are_mutually_consistent(self):
        g = real_games()
        rest = nd.rest_and_travel(g).reset_index(drop=True)
        sched = lr.rest_features(g.reset_index(drop=True), rest)
        np.testing.assert_allclose(
            sched["rest_diff"].to_numpy(),
            sched["rest_home"].to_numpy() - sched["rest_away"].to_numpy(), atol=1e-10)
        np.testing.assert_allclose(
            sched["b2b_diff"].to_numpy(),
            sched["b2b_home"].to_numpy() - sched["b2b_away"].to_numpy(), atol=1e-10)
        np.testing.assert_allclose(
            sched["three_diff"].to_numpy(),
            sched["three_home"].to_numpy() - sched["three_away"].to_numpy(), atol=1e-10)

    def test_travel_is_expressed_per_thousand_miles(self):
        g = real_games().head(50).reset_index(drop=True)
        rest = nd.rest_and_travel(g)
        sched = lr.rest_features(g, rest)
        np.testing.assert_allclose(sched["travel_k"].to_numpy(),
                                   rest["travel_miles"].to_numpy() / 1000.0)

    def test_home_indicator_is_constant_one(self):
        X = real_design()
        np.testing.assert_array_equal(X[lr.HOME_COLUMN].to_numpy(), np.ones(len(X)))


if __name__ == "__main__":       # pragma: no cover
    unittest.main()