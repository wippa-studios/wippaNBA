"""Hierarchical player-to-team ratings with partial pooling.

The model is deliberately the smallest one that answers issue #41's question::

    team_net_per_100[j] = mu + sum_i A[j, i] * theta_i + eps_j
    theta_i ~ N(0, tau^2)

``A[j, i]`` is player *i*'s **expected** minutes share for team-game *j*
(``E[MIN] / 240``), never his actual minutes.  ``theta_i`` is a player skill in
net-rating points per 100 possessions.  The prior variance ``tau^2`` is what
makes this hierarchical: a player with few expected minutes is pulled hard
toward the league mean (``theta=0``), while a heavily-used player is pulled
less.  The equivalent ridge parameter is ``alpha = sigma^2 / tau^2``.

There is no hand-set shrinkage.  For each outer walk-forward fold, ``alpha`` is
chosen from a fixed grid by a time-ordered inner validation block carved out of
the **training** rows only.  The test block is never touched by the selection,
by the expected-minutes model, or by the calibration.

The roster is the one in :mod:`src.Models.player_ratings`'s candidate panel:
players who appeared for the team in the previous ten team-games.  That is a
pre-game approximation, and it is the honest side of the availability gap.
Actual participants are used only as the **label** (the game's final score),
never as an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.DataProviders import nba_data as nd
from src.Models import player_ratings as pr

__all__ = [
    "HierarchicalConfig",
    "HierarchicalRatings",
    "fit_player_skills",
    "predict_team_net",
    "run_walk_forward",
]


@dataclass(frozen=True)
class HierarchicalConfig:
    """Fixed choices, declared before the run.

    ``alpha_grid`` is the shrinkage grid; each alpha is tried inside every
    training fold.  ``inner_initial_frac`` is the size of the inner training
    block as a share of the outer training block.  ``calibration_sigma`` is
    used only to rank alphas on the inner validation block; the final
    probability uses a two-parameter logistic calibration fitted on those same
    inner-validation margins.
    """

    alpha_grid: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
    inner_initial_frac: float = 0.8
    calibration_sigma: float = 11.0
    min_minutes: float = 0.0
    halflife: float = 8.0
    pool_window: int = 10
    hca_points: float = 2.2


@dataclass
class HierarchicalRatings:
    """A fitted random-effects player model."""

    players: np.ndarray
    intercept: float
    theta: np.ndarray
    em: pr.ExpectedMinutes
    alpha: float
    config: HierarchicalConfig
    n_players_seen: int = 0
    n_groups_fit: int = 0

    def player_ratings(self) -> pd.Series:
        return pd.Series(self.theta, index=self.players).sort_values(ascending=False)


def _design(rows: pd.DataFrame, em: pr.ExpectedMinutes,
            player_codes: dict | None = None,
            n_players: int | None = None,
            min_minutes: float = 0.0
            ) -> tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray,
                       np.ndarray, np.ndarray]:
    """Build the sparse expected-minutes design and its group keys.

    Returns ``(keys, A, player_ids, codes, weights)``.  ``codes`` is -1 for a
    player unseen during fitting; those rows are dropped because the model's
    prior for an unseen player is the league mean, i.e. exactly zero after the
    intercept.
    """
    r = rows.copy()
    w = em.predict(r)
    r["_w"] = np.where(w > float(min_minutes), w, 0.0) / 240.0
    r = r[r["_w"] > 0].copy()
    if r.empty:
        raise ValueError("no expected-minute rows survive the design build")

    mi = pd.MultiIndex.from_frame(r[["GAME_ID", "team"]])
    gi, uniq = pd.factorize(mi, sort=True)
    keys = pd.DataFrame({"GAME_ID": [u[0] for u in uniq],
                         "team": [u[1] for u in uniq]})

    if player_codes is None:
        pids, codes = np.unique(r["PLAYER_ID"].to_numpy(), return_inverse=True)
        pids = pids.astype(np.int64)
        player_codes = {int(p): int(i) for i, p in enumerate(pids)}
        n_players = len(pids)
    else:
        if n_players is None:
            n_players = len(player_codes)
        if int(n_players) != len(player_codes):
            raise ValueError("n_players must equal the size of player_codes")
        pids = np.full(len(player_codes), -1, dtype=np.int64)
        for pid, code in player_codes.items():
            pids[int(code)] = int(pid)
        if (pids < 0).any():
            raise ValueError("player_codes must be a contiguous 0..n-1 index")
        codes = np.asarray([player_codes.get(int(p), -1)
                            for p in r["PLAYER_ID"].to_numpy()], dtype=int)
    keep = codes >= 0
    if not keep.any():
        raise ValueError("no player in the scoring rows was seen during fitting")
    A = sparse.csr_matrix(
        (r["_w"].to_numpy()[keep], (gi[keep], codes[keep])),
        shape=(len(keys), int(n_players)),
    )
    return keys, A, np.asarray(pids, dtype=np.int64), codes, r["_w"].to_numpy()


def _target_for(keys: pd.DataFrame, tgt: pd.DataFrame) -> np.ndarray:
    idx = pd.MultiIndex.from_frame(keys[["GAME_ID", "team"]])
    values = (tgt.set_index(["GAME_ID", "team"])["net_per_100"]
              .reindex(idx).to_numpy(dtype=float))
    if not np.isfinite(values).all():
        raise ValueError("a team-game in the design has no finite net target")
    return values


def fit_player_skills(rows: pd.DataFrame, tgt: pd.DataFrame,
                      em: pr.ExpectedMinutes, alpha: float,
                      config: HierarchicalConfig | None = None
                      ) -> HierarchicalRatings:
    """Fit the random-effects model at a fixed ridge parameter.

    ``alpha`` is ``sigma^2 / tau^2``.  The intercept is unpenalised, so the
    model can sit at the league mean instead of shrinking the whole league to
    zero.  The solve is a 1,000-ish dimensional normal equation, which is why
    the grid can be evaluated inside every fold.
    """
    cfg = config or HierarchicalConfig()
    keys, A, pids, _, _ = _design(rows, em, min_minutes=cfg.min_minutes)
    y = _target_for(keys, tgt)

    A1 = sparse.hstack([np.ones((A.shape[0], 1), dtype=float), A], format="csr")
    gram = (A1.T @ A1).toarray()
    gram[1:, 1:] += float(alpha) * np.eye(A.shape[1])
    rhs = A1.T @ y
    coef = np.linalg.solve(gram, rhs)
    return HierarchicalRatings(
        players=pids.astype(np.int64),
        intercept=float(coef[0]),
        theta=coef[1:].copy(),
        em=em,
        alpha=float(alpha),
        config=cfg,
        n_players_seen=int(A.shape[1]),
        n_groups_fit=int(A.shape[0]),
    )


def predict_team_net(rows: pd.DataFrame, game_ids: Sequence,
                     model: HierarchicalRatings
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict ``(margin, home_net, away_net)`` for the requested games.

    The design uses expected minutes and the previously fitted player index.
    A player not seen in training contributes zero (the prior mean).  Home
    court is added back after the HCA-stripped target was modelled.
    """
    want = list(game_ids)
    sub = rows[rows["GAME_ID"].isin(want)]
    if sub.empty:
        raise ValueError("no candidate rows for the requested games")
    keys, A, _, _, _ = _design(
        sub, model.em,
        player_codes={int(p): int(i) for i, p in enumerate(model.players)},
        n_players=len(model.players),
        min_minutes=model.config.min_minutes,
    )
    net = model.intercept + A @ model.theta
    keyed = keys.assign(_net=net)
    # `team` is a tricode; the candidate rows carry the home/away flag.
    games = pd.DataFrame({"GAME_ID": want})
    gmeta = sub[["GAME_ID", "team", "is_home"]].drop_duplicates(["GAME_ID", "team"])
    keyed = keyed.merge(gmeta, on=["GAME_ID", "team"], how="left")
    home = keyed[keyed["is_home"] > 0.5].set_index("GAME_ID")["_net"]
    away = keyed[keyed["is_home"] < 0.5].set_index("GAME_ID")["_net"]
    h = home.reindex(games["GAME_ID"]).to_numpy(dtype=float)
    a = away.reindex(games["GAME_ID"]).to_numpy(dtype=float)
    margin = h - a + model.config.hca_points
    return margin, h, a


def _calibration(margin: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(margin)
    if ok.sum() < 30:
        return 0.11, 0.0
    return pr._calibrate(margin[ok], y[ok])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def run_walk_forward(games: pd.DataFrame | None = None,
                     config: HierarchicalConfig | None = None,
                     n_splits: int = 5,
                     panel: pr.PlayerPanel | None = None,
                     tgt: pd.DataFrame | None = None) -> dict:
    """Walk-forward fit, inner-fold shrinkage selection, and OOF predictions."""
    cfg = config or HierarchicalConfig()
    g = nd.build_games() if games is None else games.reset_index(drop=True)
    if panel is None:
        panel = pr.build_panel(g, pool_window=cfg.pool_window,
                               halflife=cfg.halflife)
    rows = panel.rows
    targets = pr.team_targets(g) if tgt is None else tgt

    all_p, all_y, all_idx = [], [], []
    splits_table: list[dict] = []
    for k, (tr, te) in enumerate(nd.walk_forward_split(g, n_splits=n_splits)):
        tr_ids = set(g.loc[tr, "GAME_ID"])
        te_ids = list(g.loc[te, "GAME_ID"])
        ytr = g.loc[tr, "y"].to_numpy(dtype=int)
        yte = g.loc[te, "y"].to_numpy(dtype=int)

        # ---- inner time-ordered validation, inside the training block ----
        inner_g = g.iloc[tr].reset_index(drop=True)
        inner = nd.walk_forward_split(inner_g, n_splits=1,
                                      initial_frac=cfg.inner_initial_frac)
        itr, ite = inner[0]
        itr_ids = set(inner_g.loc[itr, "GAME_ID"])
        ite_ids = list(inner_g.loc[ite, "GAME_ID"])
        em_inner = pr.ExpectedMinutes().fit(rows[rows["GAME_ID"].isin(itr_ids)])
        grid_rows = []
        best = None
        for alpha in cfg.alpha_grid:
            model = fit_player_skills(
                rows[rows["GAME_ID"].isin(itr_ids)], targets, em_inner, alpha, cfg)
            m_val, _, _ = predict_team_net(
                rows[rows["GAME_ID"].isin(ite_ids)], ite_ids, model)
            y_val = inner_g.loc[ite, "y"].to_numpy(dtype=int)
            ok = np.isfinite(m_val)
            ll = float(nd.evaluate(
                y_val[ok], _sigmoid(m_val[ok] / cfg.calibration_sigma))["log_loss"])
            grid_rows.append({"alpha": float(alpha), "inner_log_loss": ll,
                              "n_inner_validation": int(ok.sum())})
            if best is None or ll < best["inner_log_loss"]:
                best = {"alpha": float(alpha), "inner_log_loss": ll,
                        "model": model, "m_val": m_val, "y_val": y_val}
        a, b = _calibration(best["m_val"], best["y_val"])

        # ---- refit on the full training block and score the test block ----
        em = pr.ExpectedMinutes().fit(rows[rows["GAME_ID"].isin(tr_ids)])
        model = fit_player_skills(rows[rows["GAME_ID"].isin(tr_ids)],
                                  targets, em, best["alpha"], cfg)
        m_te, nh, na = predict_team_net(
            rows[rows["GAME_ID"].isin(te_ids)], te_ids, model)
        p = _sigmoid(a * m_te + b)
        ok = np.isfinite(p)
        all_p.append(p[ok])
        all_y.append(yte[ok])
        all_idx.append(te[ok])
        splits_table.append({
            "split": k + 1,
            "n_train": int(len(tr)),
            "n_test": int(ok.sum()),
            "selected_alpha": best["alpha"],
            "inner_log_loss": best["inner_log_loss"],
            "calibration_a": float(a),
            "calibration_b": float(b),
            "alpha_grid": grid_rows,
            "n_players": int(len(model.players)),
            "theta_sd": float(np.std(model.theta)),
            "intercept": float(model.intercept),
            "test_log_loss": float(nd.evaluate(yte[ok], p[ok])["log_loss"]),
        })

    p = np.concatenate(all_p)
    y = np.concatenate(all_y)
    idx = np.concatenate(all_idx)
    return {
        "p": p, "y": y, "game_index": idx,
        "split_table": splits_table,
        "n": int(len(y)),
        "alpha_grid": list(cfg.alpha_grid),
        "config": cfg,
        "panel_rows": int(len(rows)),
    }
