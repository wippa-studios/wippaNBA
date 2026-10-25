"""True RAPM from ``GameRotation`` stints, for the season that has data.

This is the model issue #39 asks for: player coefficients fitted on **stint
plus-minus**, not on a trailing box-score proxy.  The design follows directly
from the definition of a stint: for a player's stint ``[t0, t1]``,

    PT_DIFF = integral over [t0, t1] of
              (home lineup strength - away lineup strength + home edge) dt

so the design row for that stint is the time each player on the floor overlaps
it, signed ``+`` for the home side and ``-`` for the away side, plus elapsed
time for the home indicator.  Response is the recorded ``PT_DIFF``.  Coefficients
are points per 48 minutes.  This is algebraically the same estimating equation
as the usual stint regression, but it uses the recorded player-stint rows
directly instead of first solving an ill-conditioned interval decomposition.

Only games whose every interval has exactly five players on each side are
usable.  The 2020-21 file has 728 games (46,330 stint rows); 724 pass that
check.  The remaining 352 regular-season games were never downloaded, and four
downloaded games have partial rows / missing ``PT_DIFF``.  Those numbers are
reported rather than papered over.

Prediction uses the **expected** roster from
:mod:`src.Models.player_ratings` (players who appeared in the team's previous
ten games), weighted by expected minutes.  Actual participants are never an
input; they are only the label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.DataProviders import nba_data as nd
from src.Models import player_ratings as pr

__all__ = [
    "RapmConfig",
    "RapmModel",
    "build_design",
    "fit_rapm",
    "load_rotation",
    "predict_margins",
    "run_walk_forward",
]


@dataclass(frozen=True)
class RapmConfig:
    """Fixed settings, declared before the run."""

    alpha_grid: tuple[float, ...] = (30.0, 100.0, 300.0, 1000.0, 3000.0,
                                     10000.0, 30000.0, 100000.0, 300000.0,
                                     1000000.0)
    inner_initial_frac: float = 0.8
    calibration_sigma: float = 12.0
    seconds_per_game: float = 28800.0
    min_weight: float = 0.0
    pool_window: int = 10
    halflife: float = 8.0


@dataclass
class RapmModel:
    player_ids: np.ndarray
    theta: np.ndarray
    home_points_per_48: float
    alpha: float
    config: RapmConfig

    def player_series(self) -> pd.Series:
        return pd.Series(self.theta, index=self.player_ids).sort_values(ascending=False)


def _normalise_stints(rot: pd.DataFrame) -> pd.DataFrame:
    """Normalise the two clock columns and drop rows with no plus-minus.

    Some rows in the checked-in CSVs have ``OUT_TIME_REAL < IN_TIME_REAL``.
    That is a data-collection artifact, not a negative stint: the player was on
    the floor between the two clock values.  Taking min/max recovers the
    interval and turns 728 mostly-complete games into 724 fully usable ones.
    Rows with a missing ``PT_DIFF`` cannot enter the response and are dropped.
    """
    out = rot.copy()
    out["GAME_ID"] = out["GAME_ID"].astype(np.int64)
    out["start"] = out[["IN_TIME_REAL", "OUT_TIME_REAL"]].min(axis=1)
    out["end"] = out[["IN_TIME_REAL", "OUT_TIME_REAL"]].max(axis=1)
    out = out[out["start"].notna() & out["end"].notna()]
    out = out[out["end"] > out["start"]]
    out = out[out["PT_DIFF"].notna()]
    if not pd.api.types.is_integer_dtype(out["PERSON_ID"]):
        out["PERSON_ID"] = out["PERSON_ID"].astype(np.int64)
    return out


def load_rotation(season: str = "2020-21",
                  data_root: str | Path | None = None) -> pd.DataFrame:
    path = (Path(data_root) if data_root is not None
            else Path(__file__).resolve().parents[2] / "data" / "nba")
    f = path / "rotations" / f"rotation_{season}.csv"
    if not f.exists():
        raise FileNotFoundError(f"no rotation file at {f}")
    rot = pd.read_csv(f, dtype={"GAME_ID": str})
    return _normalise_stints(rot)


def _game_id_key(game_ids: pd.Series) -> np.ndarray:
    """Repo GAME_ID ('0022000001') and rotation GAME_ID (22000001) to int."""
    return game_ids.astype(str).str.lstrip("0").astype(np.int64).to_numpy()


def build_design(rotation: pd.DataFrame, games: pd.DataFrame,
                 config: RapmConfig | None = None) -> dict:
    """Build the stint-level RAPM design.

    Returns a dict with sparse ``X``, response ``y``, weights ``w``, the row's
    ``game_id`` (repo string), ``player_ids``, and diagnostics.  Games with any
    interval that does not have exactly five players per side are skipped; the
    caller is told how many and why.
    """
    cfg = config or RapmConfig()
    rot = _normalise_stints(rotation)
    g = games.copy()
    g["_gid"] = _game_id_key(g["GAME_ID"])
    rot = rot[rot["GAME_ID"].isin(set(g["_gid"]))]

    player_ids = np.sort(rot["PERSON_ID"].unique()).astype(np.int64)
    pidx = {int(p): i for i, p in enumerate(player_ids)}
    home_col = len(player_ids)
    n_cols = len(player_ids) + 1

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    y: list[float] = []
    w: list[float] = []
    row_game: list[str] = []
    n_intervals = 0
    skipped_partial: list[str] = []
    skipped_empty: list[str] = []
    n_games_used = 0

    game_by_id = g.set_index("_gid")
    for gid, game in game_by_id.iterrows():
        stints = rot[rot["GAME_ID"] == gid]
        if stints.empty:
            skipped_empty.append(str(game["GAME_ID"]))
            continue
        times = np.sort(np.unique(np.r_[stints["start"].to_numpy(),
                                        stints["end"].to_numpy()]))
        intervals: list[tuple[float, float, list[int], list[int]]] = []
        ok = True
        for a, b in zip(times[:-1], times[1:]):
            if b <= a:
                continue
            home = stints[(stints["TEAM_SIDE"] == "HOME")
                          & (stints["start"] <= a) & (stints["end"] >= b)]
            away = stints[(stints["TEAM_SIDE"] == "AWAY")
                          & (stints["start"] <= a) & (stints["end"] >= b)]
            if len(home) != 5 or len(away) != 5:
                ok = False
                break
            intervals.append((float(a), float(b),
                              home["PERSON_ID"].astype(int).tolist(),
                              away["PERSON_ID"].astype(int).tolist()))
        if not ok or not intervals:
            skipped_partial.append(str(game["GAME_ID"]))
            continue
        n_games_used += 1
        n_intervals += len(intervals)

        for _, stint in stints.iterrows():
            dur = float(stint["end"] - stint["start"])
            if dur <= cfg.min_weight:
                continue
            sign = 1.0 if stint["TEAM_SIDE"] == "HOME" else -1.0
            row = len(y)
            acc: dict[int, float] = {}
            for a, b, hp, ap in intervals:
                overlap = min(b, float(stint["end"])) - max(a, float(stint["start"]))
                if overlap <= 0:
                    continue
                share = overlap / cfg.seconds_per_game
                for p in hp:
                    acc[pidx[p]] = acc.get(pidx[p], 0.0) + share
                for p in ap:
                    acc[pidx[p]] = acc.get(pidx[p], 0.0) - share
                acc[home_col] = acc.get(home_col, 0.0) + share
            for c, v in acc.items():
                rows.append(row)
                cols.append(c)
                vals.append(v)
            y.append(sign * float(stint["PT_DIFF"]))
            w.append(dur)
            row_game.append(str(game["GAME_ID"]))

    if not y:
        raise ValueError("no usable stints; check the rotation/game join")
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(y), n_cols))
    return {
        "X": X, "y": np.asarray(y, dtype=float), "w": np.asarray(w, dtype=float),
        "game_id": np.asarray(row_game), "player_ids": player_ids,
        "home_col": home_col,
        "diagnostics": {
            "games_in_frame": int(len(g)),
            "games_used": int(n_games_used),
            "games_skipped_partial": len(skipped_partial),
            "games_skipped_empty": len(skipped_empty),
            "partial_game_examples": skipped_partial[:5],
            "empty_game_examples": skipped_empty[:5],
            "stint_rows_used": int(len(y)),
            "intervals": int(n_intervals),
            "players": int(len(player_ids)),
        },
    }


def fit_rapm(X, y, w, player_ids, alpha: float,
             config: RapmConfig | None = None) -> RapmModel:
    """Weighted ridge fit of the stint design.

    The home indicator is the last column and is **not** penalised: the home
    edge is a league constant, not a team strength that should be shrunk toward
    zero with the players.
    """
    cfg = config or RapmConfig()
    X = sparse.csr_matrix(X)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    XW = X.multiply(w[:, None])
    gram = (X.T @ XW).toarray()
    pen = np.ones(gram.shape[0], dtype=float)
    pen[-1] = 0.0
    gram.flat[:: gram.shape[0] + 1] += float(alpha) * pen
    rhs = X.T @ (y * w)
    coef = np.linalg.solve(gram, rhs)
    return RapmModel(player_ids=np.asarray(player_ids, dtype=np.int64),
                     theta=coef[:-1].copy(),
                     home_points_per_48=float(coef[-1]),
                     alpha=float(alpha), config=cfg)


def _candidate_team_net(panel_rows: pd.DataFrame, em: pr.ExpectedMinutes,
                        model: RapmModel, game_ids: Sequence[str]
                        ) -> tuple[pd.Series, pd.Series]:
    """Expected-minutes team strength for each game, home and away."""
    want = set(game_ids)
    sub = panel_rows[panel_rows["GAME_ID"].isin(want)].copy()
    if sub.empty:
        raise ValueError("no candidate rows for the requested games")
    w = np.clip(em.predict(sub), 0.0, 48.0) / 240.0
    code = {int(p): i for i, p in enumerate(model.player_ids)}
    ratings = np.asarray([model.theta[code[int(p)]]
                          if int(p) in code else 0.0
                          for p in sub["PLAYER_ID"].to_numpy()], dtype=float)
    sub["_net"] = w * ratings
    agg = sub.groupby(["GAME_ID", "team"], sort=False)["_net"].sum()
    side = sub[["GAME_ID", "team", "is_home"]].drop_duplicates(["GAME_ID", "team"])
    side = side.set_index(["GAME_ID", "team"])
    agg = agg.reindex(side.index)
    side["_net"] = agg.to_numpy()
    home = side[side["is_home"] > 0.5].reset_index().set_index("GAME_ID")["_net"]
    away = side[side["is_home"] < 0.5].reset_index().set_index("GAME_ID")["_net"]
    order = list(game_ids)
    return home.reindex(order), away.reindex(order)


def predict_margins(panel_rows: pd.DataFrame, em: pr.ExpectedMinutes,
                    model: RapmModel, game_ids: Sequence[str]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    home, away = _candidate_team_net(panel_rows, em, model, game_ids)
    h = home.to_numpy(dtype=float)
    a = away.to_numpy(dtype=float)
    return h - a + model.home_points_per_48, h, a


def _calibration(margin: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(margin)
    if ok.sum() < 30:
        return 0.11, 0.0
    return pr._calibrate(margin[ok], y[ok])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def run_walk_forward(games: pd.DataFrame | None = None,
                     rotation: pd.DataFrame | None = None,
                     config: RapmConfig | None = None,
                     n_splits: int = 5,
                     panel: pr.PlayerPanel | None = None) -> dict:
    """Walk-forward RAPM within one season, with expected-roster scoring.

    The model is fitted on the training games' stints only.  ``alpha`` is
    selected on a time-ordered inner validation block inside the training
    block; the outer test block never enters the fit, the shrinkage choice or
    the calibration.
    """
    cfg = config or RapmConfig()
    g = nd.build_games() if games is None else games.reset_index(drop=True)
    rot = load_rotation() if rotation is None else rotation
    season = g[g["SEASON"] == "2020-21"].reset_index(drop=True)
    design = build_design(rot, season, cfg)
    used_games = set(design["game_id"])
    season = season[season["GAME_ID"].isin(used_games)].reset_index(drop=True)
    # `build_design` is aligned to the games frame's order for the rows it
    # keeps; the split must use the same subset.
    if panel is None:
        panel = pr.build_panel(season, pool_window=cfg.pool_window,
                               halflife=cfg.halflife)
    rows = panel.rows

    p_all, y_all, idx_all = [], [], []
    split_table = []
    for k, (tr, te) in enumerate(nd.walk_forward_split(season, n_splits=n_splits)):
        tr_ids = list(season.loc[tr, "GAME_ID"])
        te_ids = list(season.loc[te, "GAME_ID"])
        ytr = season.loc[tr, "y"].to_numpy(dtype=int)
        yte = season.loc[te, "y"].to_numpy(dtype=int)
        tr_set, te_set = set(tr_ids), set(te_ids)

        # design rows belonging to each split
        mask_tr = np.isin(design["game_id"], list(tr_set))
        mask_te = np.isin(design["game_id"], list(te_set))

        # ---- inner time-ordered validation inside the training block ----
        inner_g = season[season["GAME_ID"].isin(tr_set)].reset_index(drop=True)
        inner = nd.walk_forward_split(inner_g, n_splits=1,
                                      initial_frac=cfg.inner_initial_frac)[0]
        itr_ids = list(inner_g.loc[inner[0], "GAME_ID"])
        ite_ids = list(inner_g.loc[inner[1], "GAME_ID"])
        inner_tr_mask = np.isin(design["game_id"], itr_ids)
        em_inner = pr.ExpectedMinutes().fit(
            rows[rows["GAME_ID"].isin(set(itr_ids))])
        y_val = inner_g.loc[inner[1], "y"].to_numpy(dtype=int)
        grid = []
        best = None
        for alpha in cfg.alpha_grid:
            model = fit_rapm(design["X"][inner_tr_mask],
                             design["y"][inner_tr_mask],
                             design["w"][inner_tr_mask],
                             design["player_ids"], alpha, cfg)
            m_val, _, _ = predict_margins(rows, em_inner, model, ite_ids)
            ok = np.isfinite(m_val)
            # Selection uses the pre-registered 12-point scale, not a
            # calibration fitted on the rows it is selecting on.
            ll = float(nd.evaluate(
                y_val[ok], _sigmoid(m_val[ok] / cfg.calibration_sigma))["log_loss"])
            grid.append({"alpha": float(alpha), "inner_log_loss": ll,
                         "n_inner_validation": int(ok.sum())})
            if best is None or ll < best["inner_log_loss"]:
                best = {"alpha": alpha, "inner_log_loss": ll,
                        "m_val": m_val, "y_val": y_val, "model": model}
        # The final probability mapping is fitted on the inner-validation
        # margins, which are out-of-sample relative to the inner fit and still
        # strictly inside the outer training block.
        a, b = _calibration(best["m_val"], best["y_val"])

        # ---- refit on the full training block and score the outer test ----
        em = pr.ExpectedMinutes().fit(rows[rows["GAME_ID"].isin(tr_set)])
        model = fit_rapm(design["X"][mask_tr], design["y"][mask_tr],
                         design["w"][mask_tr], design["player_ids"],
                         best["alpha"], cfg)
        m_te, nh, na = predict_margins(rows, em, model, te_ids)
        p = _sigmoid(a * m_te + b)
        ok = np.isfinite(p)
        p_all.append(p[ok]); y_all.append(yte[ok]); idx_all.append(te[ok])
        split_table.append({
            "split": k + 1, "n_train": int(len(tr)), "n_test": int(ok.sum()),
            "selected_alpha": float(best["alpha"]),
            "inner_log_loss": float(best["inner_log_loss"]),
            "calibration_a": float(a), "calibration_b": float(b),
            "alpha_grid": grid,
            "test_log_loss": float(nd.evaluate(yte[ok], p[ok])["log_loss"]),
            "home_points_per_48": float(model.home_points_per_48),
            "n_players": int(len(model.player_ids)),
            "theta_sd": float(np.std(model.theta)),
            "theta_min": float(np.min(model.theta)),
            "theta_max": float(np.max(model.theta)),
            "n_stints_train": int(mask_tr.sum()),
        })

    return {
        "p": np.concatenate(p_all), "y": np.concatenate(y_all),
        "game_index": np.concatenate(idx_all),
        "season": season, "design_diagnostics": design["diagnostics"],
        "split_table": split_table, "n": int(len(np.concatenate(y_all))),
        "alpha_grid": list(cfg.alpha_grid),
        "config": cfg,
    }
