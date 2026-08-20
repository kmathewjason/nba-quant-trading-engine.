"""
optimizer.py — Expected Value (EV) & Kelly Criterion
======================================================
Two surfaces:

  1. Game-level (original) — evaluate_opportunity / evaluate_slate
     Takes BetOpportunity objects (team win-probability vs. spread odds).

  2. Prop-level (new) — evaluate_props
     Takes the daily prediction DataFrame from predict_daily.run_daily()
     plus sportsbook American odds, and returns a ranked bet sheet with
     EV and quarter-Kelly stake for every prop.

Kelly Criterion
---------------
    f* = (b·p − q) / b

    where:
        p  = model's probability of the over hitting
        q  = 1 − p  (probability of under)
        b  = decimal_odds − 1  (net profit per $1 wagered)

Risk management: the final recommended stake uses a 0.25× Kelly
multiplier ("quarter-Kelly") — a standard conservative scaling used
in sports-betting quant finance to reduce variance without abandoning
the Kelly optimality property.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared maths helpers
# ═══════════════════════════════════════════════════════════════════════════════

def american_to_decimal(american: int) -> float:
    """Convert American moneyline odds to decimal (European) format."""
    if american > 0:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


def decimal_to_implied_prob(decimal: float) -> float:
    """Raw implied probability (vig NOT removed)."""
    return 1.0 / decimal


def remove_vig(p1: float, p2: float) -> tuple[float, float]:
    """
    Remove bookmaker overround using the additive (proportional) method.
    Divides each raw implied probability by the total overround.
    """
    total = p1 + p2
    return p1 / total, p2 / total


def calc_ev(model_prob: float, decimal_odds: float) -> float:
    """
    Expected Value per $1 wagered.

        EV = p × (decimal_odds − 1) − (1 − p)
           = p × decimal_odds − 1
    """
    return model_prob * decimal_odds - 1.0


def calc_kelly(model_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly Criterion fraction.

        f* = (b·p − q) / b
             where b = decimal_odds − 1, q = 1 − p

    Floored at 0 — never bet a negative-EV side.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    return max((model_prob * b - q) / b, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Surface 1 — Game-level (team win-probability)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BetOpportunity:
    """A single game-level wagering opportunity."""
    game_id:            str
    home_team:          str
    away_team:          str
    model_prob_home:    float   # P(home wins) from ensemble
    american_odds_home: int     # e.g. -110, +150
    american_odds_away: int
    bankroll:           float = 1_000.0


@dataclass
class BetRecommendation:
    """Output of evaluate_opportunity."""
    game_id:             str
    side:                str     # 'home' or 'away'
    model_prob:          float
    implied_prob:        float
    edge:                float   # model_prob − no-vig implied prob
    ev_per_unit:         float   # EV per $1 wagered
    kelly_fraction:      float   # full Kelly
    half_kelly_fraction: float   # half-Kelly
    stake_dollars:       float   # half-Kelly × bankroll
    recommendation:      str     # 'BET' or 'PASS'


def evaluate_opportunity(
    opp: BetOpportunity,
    min_edge: float = 0.02,
) -> BetRecommendation:
    """
    Evaluate one game-level opportunity.

    Parameters
    ----------
    opp : BetOpportunity
    min_edge : float
        Minimum edge required to flip the recommendation to 'BET'.

    Returns
    -------
    BetRecommendation
    """
    dec_home = american_to_decimal(opp.american_odds_home)
    dec_away = american_to_decimal(opp.american_odds_away)

    raw_home = decimal_to_implied_prob(dec_home)
    raw_away = decimal_to_implied_prob(dec_away)
    imp_home, imp_away = remove_vig(raw_home, raw_away)

    p_away = 1.0 - opp.model_prob_home
    edge_home = opp.model_prob_home - imp_home
    edge_away = p_away - imp_away

    if edge_home >= edge_away:
        side, p, imp, dec = "home", opp.model_prob_home, imp_home, dec_home
    else:
        side, p, imp, dec = "away", p_away, imp_away, dec_away

    ev  = calc_ev(p, dec)
    kf  = calc_kelly(p, dec)
    hkf = kf / 2.0

    return BetRecommendation(
        game_id=opp.game_id,
        side=side,
        model_prob=round(p, 4),
        implied_prob=round(imp, 4),
        edge=round(p - imp, 4),
        ev_per_unit=round(ev, 4),
        kelly_fraction=round(kf, 4),
        half_kelly_fraction=round(hkf, 4),
        stake_dollars=round(hkf * opp.bankroll, 2),
        recommendation="BET" if (p - imp) >= min_edge and ev > 0 else "PASS",
    )


def evaluate_slate(
    opportunities: list[BetOpportunity],
    min_edge: float = 0.02,
) -> pd.DataFrame:
    """
    Evaluate a full game slate and return a ranked DataFrame.

    Returns
    -------
    pd.DataFrame sorted by edge descending.
    """
    results = [evaluate_opportunity(o, min_edge) for o in opportunities]
    df = pd.DataFrame([r.__dict__ for r in results])
    df = df.sort_values("edge", ascending=False).reset_index(drop=True)
    logger.info("Slate evaluated: %d games, %d BET recommendations",
                len(df), (df["recommendation"] == "BET").sum())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Surface 2 — Prop-level (player points props)
# ═══════════════════════════════════════════════════════════════════════════════

# Default odds assumed when no book odds are provided (-110 is the standard
# US sportsbook juice on both sides of a prop)
_DEFAULT_OVER_ODDS  = -110
_DEFAULT_UNDER_ODDS = -110

# Conservative Kelly multiplier — industry standard for managing variance
_KELLY_MULTIPLIER = 0.25

# Maximum fraction of bankroll that may be staked across ALL concurrent bets
# in a single day. Prevents Kelly blowout when many props are bet simultaneously.
_MAX_DAILY_EXPOSURE = 0.15   # 15 %


def evaluate_props(
    predictions: pd.DataFrame,
    odds: pd.DataFrame | None = None,
    bankroll: float = 1_000.0,
    min_edge: float = 0.04,
    min_ev: float = 0.05,
    kelly_multiplier: float = _KELLY_MULTIPLIER,
    max_daily_exposure: float = _MAX_DAILY_EXPOSURE,
) -> pd.DataFrame:
    """
    Evaluate player prop betting opportunities.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of predict_daily.run_daily(). Must contain:
            PLAYER_NAME, TEAM, OPPONENT, IS_HOME, PROP_TYPE,
            LINE, ENSEMBLE_PROB
    odds : pd.DataFrame or None
        Optional sportsbook odds with columns:
            PLAYER_NAME, PROP_TYPE, LINE, OVER_ODDS, UNDER_ODDS
        All values are American moneyline integers.
        If None, defaults to -110 / -110 for all props.
    bankroll : float
        Total bankroll in dollars used for stake sizing.
    min_edge : float
        Minimum edge (model_prob − no-vig implied_prob) required for a
        'BET' recommendation. Default 4 % — ensures the model advantage
        meaningfully overcomes the sportsbook's vig (~2.38 % at -110).
    min_ev : float
        Minimum expected value per $1 wagered required for a 'BET'
        recommendation. Default $0.05. Eliminates micro-bets where the
        edge is real but too small to be worth the execution friction.
    kelly_multiplier : float
        Fraction of full Kelly to use. Default 0.25 (quarter-Kelly).
    max_daily_exposure : float
        Hard cap on the sum of all concurrent stakes as a fraction of
        bankroll. Default 0.15 (15 %). Prevents Kelly blowout when many
        props fire simultaneously — each BET stake is scaled down
        proportionally so the total never exceeds this limit.

    Returns
    -------
    pd.DataFrame with one row per prop, sorted by EDGE descending.

    Columns
    -------
    PLAYER_NAME, TEAM, OPPONENT, IS_HOME, PROP_TYPE, LINE,
    ENSEMBLE_PROB,       — model's P(over)
    BOOK_OVER_ODDS,      — American odds on the over
    BOOK_UNDER_ODDS,     — American odds on the under
    OVER_IMPLIED,        — raw (vigged) book implied P(over); 52.38% at -110
    EDGE,                — ENSEMBLE_PROB − OVER_IMPLIED (must clear the vig)
    EV_PER_UNIT,         — expected value per $1 wagered on the over
    KELLY_FULL,          — full Kelly fraction
    KELLY_QTR,           — quarter-Kelly fraction (recommended)
    STAKE_DOLLARS,       — portfolio-capped stake in dollars
    BET_SIDE,            — 'OVER' or 'UNDER' (whichever has +edge)
    RECOMMENDATION       — 'BET' or 'PASS'
    """
    df = predictions.copy()

    # ── Merge odds ────────────────────────────────────────────────────────────
    if odds is not None:
        df = df.merge(
            odds[["PLAYER_NAME", "PROP_TYPE", "LINE", "OVER_ODDS", "UNDER_ODDS"]],
            on=["PLAYER_NAME", "PROP_TYPE", "LINE"],
            how="left",
        )
        df["OVER_ODDS"]  = df["OVER_ODDS"].fillna(_DEFAULT_OVER_ODDS).astype(int)
        df["UNDER_ODDS"] = df["UNDER_ODDS"].fillna(_DEFAULT_UNDER_ODDS).astype(int)
    else:
        df["OVER_ODDS"]  = _DEFAULT_OVER_ODDS
        df["UNDER_ODDS"] = _DEFAULT_UNDER_ODDS

    # ── Vectorised EV & Kelly ─────────────────────────────────────────────────
    dec_over  = df["OVER_ODDS"].apply(american_to_decimal)
    dec_under = df["UNDER_ODDS"].apply(american_to_decimal)

    # Raw (vigged) implied probabilities — the true sportsbook breakeven.
    # At -110 this is 1/1.9091 = 52.38%. This is what the model must BEAT
    # to have a positive EV bet, and what is displayed in the UI as "Implied %".
    raw_over  = 1.0 / dec_over
    raw_under = 1.0 / dec_under

    # No-vig fair prices (used only for Kelly denominator consistency)
    total_vig = raw_over + raw_under
    nv_over   = raw_over  / total_vig
    nv_under  = raw_under / total_vig

    p_over  = df["ENSEMBLE_PROB"]
    p_under = 1.0 - p_over

    # Edge = model probability minus the RAW (vigged) book implied probability.
    # This is the correct definition: the model must clear the vig hurdle,
    # not merely beat the no-vig 50/50 fair line.
    edge_over  = p_over  - raw_over
    edge_under = p_under - raw_under

    # Bet the side with the larger edge (vs vigged implied, not no-vig)
    bet_over  = edge_over >= edge_under
    bet_prob  = np.where(bet_over, p_over,  p_under)
    bet_dec   = np.where(bet_over, dec_over, dec_under)
    bet_edge  = np.where(bet_over, edge_over, edge_under)

    ev  = bet_prob * bet_dec - 1.0
    b   = bet_dec - 1.0
    q   = 1.0 - bet_prob
    kf  = np.where(b > 0,
                   np.maximum((bet_prob * b - q) / b, 0.0),
                   0.0)
    kqf = kf * kelly_multiplier

    df["BOOK_OVER_ODDS"]  = df["OVER_ODDS"]
    df["BOOK_UNDER_ODDS"] = df["UNDER_ODDS"]
    # Store the RAW (vigged) book implied probability so the UI displays
    # the real sportsbook breakeven (52.38% at -110), not the no-vig 50.0%.
    df["OVER_IMPLIED"]    = raw_over.round(4)
    df["EDGE"]            = bet_edge.round(4)
    df["EV_PER_UNIT"]     = ev.round(4)
    df["KELLY_FULL"]      = kf.round(4)
    df["KELLY_QTR"]       = kqf.round(4)
    df["STAKE_DOLLARS"]   = (kqf * bankroll).round(2)
    df["BET_SIDE"]        = np.where(bet_over, "OVER", "UNDER")
    df["RECOMMENDATION"]  = np.where(
        (bet_edge >= min_edge) & (ev >= min_ev), "BET", "PASS"
    )

    # ── Select and rank output ────────────────────────────────────────────────
    out_cols = [
        "PLAYER_NAME", "TEAM", "OPPONENT", "IS_HOME",
        "PROP_TYPE", "LINE",
        "ENSEMBLE_PROB", "BOOK_OVER_ODDS", "BOOK_UNDER_ODDS",
        "OVER_IMPLIED", "EDGE", "EV_PER_UNIT",
        "KELLY_FULL", "KELLY_QTR", "STAKE_DOLLARS",
        "BET_SIDE", "RECOMMENDATION",
    ]
    result = df[out_cols].sort_values("EDGE", ascending=False).reset_index(drop=True)

    # ── Portfolio cap: scale BET stakes so total ≤ max_daily_exposure ─────────
    # Kelly fractions are derived independently per bet and cannot be summed
    # linearly for concurrent wagers. We proportionally rescale so the total
    # exposure never exceeds the daily cap.
    bet_mask = result["RECOMMENDATION"] == "BET"
    if bet_mask.any():
        raw_total_frac = result.loc[bet_mask, "KELLY_QTR"].sum()
        budget_frac    = max_daily_exposure              # e.g. 0.15
        if raw_total_frac > budget_frac:
            scale = budget_frac / raw_total_frac         # < 1.0
            result.loc[bet_mask, "KELLY_QTR"]      = (
                result.loc[bet_mask, "KELLY_QTR"] * scale
            ).round(4)
            result.loc[bet_mask, "STAKE_DOLLARS"]  = (
                result.loc[bet_mask, "KELLY_QTR"] * bankroll
            ).round(2)

    n_bet       = bet_mask.sum()
    total_stake = result.loc[bet_mask, "STAKE_DOLLARS"].sum()
    logger.info(
        "Props evaluated: %d total | %d BET (%d PASS) | "
        "Total recommended stake: $%.2f (cap=%.0f%%)",
        len(result), n_bet, len(result) - n_bet,
        total_stake, max_daily_exposure * 100,
    )
    return result
