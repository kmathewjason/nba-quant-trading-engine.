"""
sgp_engine.py — Same Game Parlay (SGP) Correlation Engine
==========================================================
Sportsbooks discount SGP payouts because legs are correlated — when
LeBron James dishes 12 assists, Anthony Davis typically gets more
easy buckets. This module:

  1. Builds a per-game-correlation matrix from historical raw box-scores,
     mapping every (player_A_stat, player_B_stat) pair that shares a GAME_ID.

  2. Applies the Gaussian-copula approximation to adjust two independent
     marginal probabilities (P_A, P_B) into a joint over probability
     P(A_over ∩ B_over) that respects the empirical Pearson correlation.

  3. Compares that adjusted joint probability to the sportsbook's SGP
     payout odds to compute Expected Value and a quarter-Kelly stake.

Key formula
-----------
The joint probability for two marginally-normal variables with correlation ρ:

    P(X_A > t_A, X_B > t_B)  ≈  Φ₂(z_A, z_B; ρ)

where
    z_A  = Φ⁻¹(P_A)   (probit of the marginal over-probability)
    z_B  = Φ⁻¹(P_B)
    Φ₂   = bivariate standard-normal CDF

For multi-leg parlays (≥3 legs) we chain the pairwise adjustments using
the geometric-mean correlation of all unique pairs as a conservative
approximation.

Notes
-----
*  The normality assumption is approximate; NBA counting stats (PTS, AST)
   are right-skewed.  The Gaussian copula is industry-standard for sports
   SGP modelling because (a) it is analytically tractable and (b) it
   consistently produces smaller joint probabilities than the independence
   assumption, which is the correct directional correction.

*  Only SAME-TEAM props within the SAME GAME are meaningfully correlated.
   Cross-team props (e.g. PG1_PTS vs. PG2_PTS on opposing teams) can be
   negatively correlated due to pace constraints; the engine handles both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

logger = logging.getLogger(__name__)

_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Stats available in the raw player game-log
STAT_COLS = ["PTS", "AST", "REB", "STL", "BLK", "MIN", "FGA", "TOV"]

# Minimum shared games required to report a correlation (otherwise NaN)
MIN_SHARED_GAMES = 10

# Conservative Kelly multiplier (matches optimizer.py)
_KELLY_MULTIPLIER = 0.25


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PropLeg:
    """A single leg of a Same Game Parlay."""
    player_name: str
    stat:        str    # one of STAT_COLS, e.g. 'PTS', 'AST'
    line:        float
    over_prob:   float  # model's P(stat > line) for this leg


@dataclass
class SGPResult:
    """Full evaluation of a 2-leg (or n-leg) SGP."""
    legs:               list[PropLeg]
    correlation:        float          # Pearson ρ between the two stat series
    independent_prob:   float          # P_A × P_B (no adjustment)
    adjusted_prob:      float          # Gaussian-copula joint probability
    book_sgp_odds:      int            # American odds for the parlay
    book_implied_prob:  float          # no-vig implied from book odds
    edge:               float          # adjusted_prob − book_implied_prob
    ev_per_unit:        float          # expected value per $1 wagered
    kelly_full:         float
    kelly_qtr:          float          # quarter-Kelly stake fraction
    stake_dollars:      float          # kelly_qtr × bankroll
    recommendation:     str            # 'BET' or 'PASS'


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Correlation matrix builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_correlation_matrix(
    player_logs: pd.DataFrame | None = None,
    season: str = "2024-25",
    same_team_only: bool = False,
) -> pd.DataFrame:
    """
    Build a pairwise stat-correlation matrix from historical game logs.

    The matrix has a MultiIndex: (player_name, stat) on both axes.
    Each cell is the Pearson correlation between player A's *stat_A* series
    and player B's *stat_B* series across their shared game IDs.

    Parameters
    ----------
    player_logs : pd.DataFrame or None
        Raw player game logs. If None, loaded from data/raw/.
    season : str
        Used to locate the cached Parquet if player_logs is None.
    same_team_only : bool
        If True, only compute correlations for same-team player pairs.
        Faster, and the results are more actionable for SGP purposes.

    Returns
    -------
    pd.DataFrame with MultiIndex (PLAYER_NAME, STAT) × (PLAYER_NAME, STAT).
    """
    if player_logs is None:
        path = _RAW_DIR / f"player_game_logs_{season.replace('-','_')}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Raw logs not found at {path}. Run ingestion first."
            )
        player_logs = pd.read_parquet(path)

    import warnings  # noqa: PLC0415

    df = player_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # Drop DNPs and zero-variance bench rows (would produce NaN correlations)
    df = df[df["MIN"].fillna(0) > 5]

    stat_cols_present = [c for c in STAT_COLS if c in df.columns]
    logger.info("Building correlation matrix (%s) ...",
                "same-team pairs" if same_team_only else "all pairs")

    # Build valid same-team (player_a, player_b) pairs once
    if same_team_only:
        valid_pairs: set[tuple[str, str]] = set()
        for (_, _team), grp in df.groupby(["GAME_ID", "TEAM_ID"]):
            ps = grp["PLAYER_NAME"].tolist()
            for ii in range(len(ps)):
                for jj in range(ii + 1, len(ps)):
                    valid_pairs.add((ps[ii], ps[jj]))
    else:
        valid_pairs = None  # type: ignore[assignment]

    records: list[dict] = []

    # ── Same-stat correlations via vectorised pandas corr() ──────────────────
    for stat in stat_cols_present:
        stat_wide = df.pivot_table(
            index="GAME_ID", columns="PLAYER_NAME",
            values=stat, aggfunc="mean",
        )
        # Drop zero-variance columns (suppress the numpy divide warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stat_wide = stat_wide.loc[:, stat_wide.std() > 0]
            corr_m = stat_wide.corr(min_periods=MIN_SHARED_GAMES)

        notna  = stat_wide.notna().astype(int)
        count_m = notna.T.dot(notna)
        players = corr_m.columns.tolist()

        for ii, pa in enumerate(players):
            for pb in players[ii + 1:]:
                if valid_pairs is not None \
                        and (pa, pb) not in valid_pairs \
                        and (pb, pa) not in valid_pairs:
                    continue
                r = corr_m.loc[pa, pb]
                if pd.isna(r):
                    continue
                n = int(count_m.loc[pa, pb])
                if n < MIN_SHARED_GAMES:
                    continue
                records.append({
                    "player_a": pa, "stat_a": stat,
                    "player_b": pb, "stat_b": stat,
                    "n_games":  n,  "pearson_r": round(r, 4),
                })

    # ── Cross-stat: AST→PTS, REB→PTS (feed-finish relationships) ─────────────
    cross_pairs = [("AST", "PTS"), ("REB", "PTS"), ("MIN", "PTS")]
    for stat_a, stat_b in cross_pairs:
        if stat_a not in stat_cols_present or stat_b not in stat_cols_present:
            continue
        wa = df.pivot_table(index="GAME_ID", columns="PLAYER_NAME",
                            values=stat_a, aggfunc="mean")
        wb = df.pivot_table(index="GAME_ID", columns="PLAYER_NAME",
                            values=stat_b, aggfunc="mean")
        common = wa.index.intersection(wb.index)
        wa, wb = wa.loc[common], wb.loc[common]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wa = wa.loc[:, wa.std() > 0]
            wb = wb.loc[:, wb.std() > 0]

        for pa in wa.columns:
            for pb in wb.columns:
                if pa == pb:
                    continue
                if valid_pairs is not None \
                        and (pa, pb) not in valid_pairs \
                        and (pb, pa) not in valid_pairs:
                    continue
                paired = pd.concat(
                    [wa[pa].rename("a"), wb[pb].rename("b")], axis=1
                ).dropna()
                if len(paired) < MIN_SHARED_GAMES:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    r = paired["a"].corr(paired["b"])
                if pd.isna(r):
                    continue
                records.append({
                    "player_a": pa, "stat_a": stat_a,
                    "player_b": pb, "stat_b": stat_b,
                    "n_games":  len(paired), "pearson_r": round(r, 4),
                })

    corr_df = pd.DataFrame(records)
    logger.info("Correlation matrix: %d pairs computed.", len(corr_df))
    return corr_df


def lookup_correlation(
    corr_df: pd.DataFrame,
    player_a: str,
    stat_a: str,
    player_b: str,
    stat_b: str,
) -> float:
    """
    Look up the Pearson r for a (player_a.stat_a, player_b.stat_b) pair.
    Handles both orderings. Returns 0.0 if the pair is not found.
    """
    mask = (
        ((corr_df["player_a"] == player_a) & (corr_df["stat_a"] == stat_a) &
         (corr_df["player_b"] == player_b) & (corr_df["stat_b"] == stat_b))
        |
        ((corr_df["player_a"] == player_b) & (corr_df["stat_a"] == stat_b) &
         (corr_df["player_b"] == player_a) & (corr_df["stat_b"] == stat_a))
    )
    hits = corr_df[mask]
    if hits.empty:
        logger.debug(
            "No correlation found for (%s.%s, %s.%s) — assuming ρ=0.0",
            player_a, stat_a, player_b, stat_b,
        )
        return 0.0
    return float(hits.iloc[0]["pearson_r"])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Gaussian-copula joint probability
# ═══════════════════════════════════════════════════════════════════════════════

def joint_prob_two_legs(
    p_a: float,
    p_b: float,
    rho: float,
) -> float:
    """
    Estimate P(A_over ∩ B_over) using a Gaussian copula.

    Steps
    -----
    1. Convert each marginal probability to a standard-normal quantile
       (probit transform): z = Φ⁻¹(p).
    2. Evaluate the bivariate standard-normal CDF at (z_A, z_B; ρ)
       using scipy's multivariate_normal.

    This is equivalent to assuming the two marginals are connected by
    a Gaussian copula with correlation ρ.

    Parameters
    ----------
    p_a, p_b : float  — marginal P(over) for each leg from the model
    rho : float        — Pearson correlation between the two stat series

    Returns
    -------
    float — joint P(A_over AND B_over)
    """
    # Clip to avoid ±inf in the probit transform
    p_a  = np.clip(p_a,  1e-6, 1 - 1e-6)
    p_b  = np.clip(p_b,  1e-6, 1 - 1e-6)
    rho  = np.clip(rho, -0.999, 0.999)

    z_a = norm.ppf(p_a)
    z_b = norm.ppf(p_b)

    cov = [[1.0, rho],
           [rho, 1.0]]
    joint = stats.multivariate_normal.cdf(
        x=[z_a, z_b],
        mean=[0.0, 0.0],
        cov=cov,
    )
    return float(np.clip(joint, 1e-9, 1.0))


def joint_prob_n_legs(
    legs: list[PropLeg],
    corr_df: pd.DataFrame,
) -> tuple[float, float]:
    """
    Compute independent and correlation-adjusted joint probability for
    n ≥ 2 legs using pairwise Gaussian-copula adjustments.

    For n > 2 the adjustment is:
      P_adjusted  =  P_independent × ∏_{all pairs} (P_pair_adjusted / P_pair_independent)

    The product of pair-adjustments multiplicatively scales the
    independence assumption by how much each pair deviates from independence.

    Returns
    -------
    (independent_prob, adjusted_prob)
    """
    p_independent = 1.0
    for leg in legs:
        p_independent *= leg.over_prob

    if len(legs) == 1:
        return p_independent, p_independent

    # Collect pairwise adjustments
    adjustment = 1.0
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            la, lb = legs[i], legs[j]
            rho = lookup_correlation(
                corr_df, la.player_name, la.stat, lb.player_name, lb.stat
            )
            p_pair_indep   = la.over_prob * lb.over_prob
            p_pair_adj     = joint_prob_two_legs(la.over_prob, lb.over_prob, rho)
            if p_pair_indep > 1e-9:
                adjustment *= p_pair_adj / p_pair_indep

    p_adjusted = np.clip(p_independent * adjustment, 1e-9, 1.0)
    return p_independent, float(p_adjusted)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SGP EV evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _american_to_decimal(american: int) -> float:
    if american > 0:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


def evaluate_sgp(
    legs: list[PropLeg],
    corr_df: pd.DataFrame,
    book_sgp_odds: int,
    bankroll: float = 1_000.0,
    min_edge: float = 0.02,
    kelly_multiplier: float = _KELLY_MULTIPLIER,
) -> SGPResult:
    """
    Evaluate a Same Game Parlay for Expected Value.

    Parameters
    ----------
    legs : list[PropLeg]
        2–6 prop legs. Each leg must already carry a model over_prob.
    corr_df : pd.DataFrame
        Correlation table from build_correlation_matrix().
    book_sgp_odds : int
        Sportsbook's American moneyline for the full parlay.
    bankroll : float
        Bankroll for Kelly stake sizing.
    min_edge : float
        Minimum edge to recommend a bet.
    kelly_multiplier : float
        Fraction of full Kelly to use (default 0.25).

    Returns
    -------
    SGPResult
    """
    if len(legs) < 2:
        raise ValueError("An SGP requires at least 2 legs.")

    p_independent, p_adjusted = joint_prob_n_legs(legs, corr_df)

    # Pairwise ρ — for a 2-leg SGP this is the single correlation;
    # for multi-leg we report the mean of all pairwise correlations.
    rhos: list[float] = []
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            rhos.append(lookup_correlation(
                corr_df,
                legs[i].player_name, legs[i].stat,
                legs[j].player_name, legs[j].stat,
            ))
    mean_rho = float(np.mean(rhos)) if rhos else 0.0

    # Book implied probability (single side — no vig removal needed for
    # a parlay; the book sets the payout, not two-sided lines)
    dec_odds  = _american_to_decimal(book_sgp_odds)
    book_impl = 1.0 / dec_odds        # raw implied — parlay has no "other side"

    edge       = p_adjusted - book_impl
    b          = dec_odds - 1.0
    ev         = p_adjusted * b - (1.0 - p_adjusted)   # = p*b - q
    kf         = max((p_adjusted * b - (1.0 - p_adjusted)) / b, 0.0) if b > 0 else 0.0
    kqf        = kf * kelly_multiplier
    stake      = kqf * bankroll
    rec        = "BET" if edge >= min_edge and ev > 0 else "PASS"

    logger.info(
        "SGP [%s] | ρ=%.3f | P_indep=%.4f → P_adj=%.4f | "
        "book_impl=%.4f | edge=%+.4f | EV=%+.4f | kelly_qtr=%.4f → $%.2f | %s",
        " + ".join(f"{l.player_name} {l.stat}>{l.line}" for l in legs),
        mean_rho, p_independent, p_adjusted,
        book_impl, edge, ev, kqf, stake, rec,
    )

    return SGPResult(
        legs=legs,
        correlation=round(mean_rho, 4),
        independent_prob=round(p_independent, 6),
        adjusted_prob=round(p_adjusted, 6),
        book_sgp_odds=book_sgp_odds,
        book_implied_prob=round(book_impl, 4),
        edge=round(edge, 4),
        ev_per_unit=round(ev, 4),
        kelly_full=round(kf, 4),
        kelly_qtr=round(kqf, 4),
        stake_dollars=round(stake, 2),
        recommendation=rec,
    )


def evaluate_sgp_slate(
    parlays: list[tuple[list[PropLeg], int]],
    corr_df: pd.DataFrame,
    bankroll: float = 1_000.0,
    min_edge: float = 0.02,
    kelly_multiplier: float = _KELLY_MULTIPLIER,
) -> pd.DataFrame:
    """
    Evaluate multiple SGP opportunities and return a ranked DataFrame.

    Parameters
    ----------
    parlays : list of (legs, book_sgp_odds) tuples
    corr_df : correlation table from build_correlation_matrix()
    bankroll, min_edge, kelly_multiplier : passed through to evaluate_sgp()

    Returns
    -------
    pd.DataFrame sorted by EDGE descending.
    """
    rows: list[dict] = []
    for legs, odds in parlays:
        r = evaluate_sgp(legs, corr_df, odds, bankroll, min_edge, kelly_multiplier)
        rows.append({
            "LEGS":             " + ".join(
                                    f"{l.player_name} {l.stat}>{l.line}"
                                    for l in r.legs),
            "CORRELATION":      r.correlation,
            "P_INDEPENDENT":    r.independent_prob,
            "P_ADJUSTED":       r.adjusted_prob,
            "BOOK_SGP_ODDS":    r.book_sgp_odds,
            "BOOK_IMPLIED":     r.book_implied_prob,
            "EDGE":             r.edge,
            "EV_PER_UNIT":      r.ev_per_unit,
            "KELLY_FULL":       r.kelly_full,
            "KELLY_QTR":        r.kelly_qtr,
            "STAKE_DOLLARS":    r.stake_dollars,
            "RECOMMENDATION":   r.recommendation,
        })

    df = pd.DataFrame(rows).sort_values("EDGE", ascending=False).reset_index(drop=True)
    n_bet = (df["RECOMMENDATION"] == "BET").sum()
    logger.info(
        "SGP slate: %d parlays evaluated | %d BET | %d PASS",
        len(df), n_bet, len(df) - n_bet,
    )
    return df
