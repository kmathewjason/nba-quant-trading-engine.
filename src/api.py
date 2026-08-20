"""
api.py — FastAPI backend for the NBA Prop Prediction Dashboard
==============================================================
Serves daily player prop predictions and Kelly-Criterion bet sizing
as JSON, consumed by the React frontend (nba-dashboard).

Endpoints
---------
GET /api/predictions/daily
    Returns today's (or a given date's) predictions + optimizer output.

GET /api/predictions/sgp
    Evaluates a Same Game Parlay submitted as a JSON body.

GET /health
    Simple liveness check.

Run locally
-----------
    uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.live_odds import fetch_player_props, fetch_today_event_ids, merge_with_predictions
from src.optimizer import evaluate_props
from src.predict_daily import run_daily

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

# ── Application lifespan (pre-warm artefacts on startup) ─────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the model artefacts once so the first request is fast."""
    logger.info("API starting up — pre-warming model artefacts …")
    try:
        _get_predictions.cache_clear()   # ensure fresh on restart
    except Exception:
        pass
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="NBA Prop Prediction API",
    version="1.0.0",
    description="Daily player prop predictions with EV and Kelly sizing.",
    lifespan=lifespan,
)

# ── CORS — allow the local Vite dev server (port 5173) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",   # fallback
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Response models
# ═══════════════════════════════════════════════════════════════════════════════

class PropPrediction(BaseModel):
    player_name:    str
    team:           str
    opponent:       str
    is_home:        int
    prop_type:      str
    line:           float
    lr_prob:        float
    xgb_prob:       float
    mlp_prob:       float
    ensemble_prob:  float
    book_over_odds: int
    book_under_odds: int
    over_implied:   float
    edge:           float
    ev_per_unit:    float
    kelly_full:     float
    kelly_qtr:      float      # quarter-Kelly fraction — multiply by bankroll for stake
    bet_side:       str
    recommendation: str


class DailyResponse(BaseModel):
    game_date:       str
    prop_line:       float
    total_props:     int
    bet_count:       int
    predictions:     list[PropPrediction]


# ── SGP request / response ────────────────────────────────────────────────────

class SGPLeg(BaseModel):
    player_name: str
    stat:        str
    line:        float
    over_prob:   float


class SGPRequest(BaseModel):
    legs:          list[SGPLeg]
    book_sgp_odds: int
    bankroll:      float = 1_000.0
    min_edge:      float = 0.02


class SGPResponse(BaseModel):
    legs:             list[str]
    correlation:      float
    independent_prob: float
    adjusted_prob:    float
    book_sgp_odds:    int
    book_implied_prob: float
    edge:             float
    ev_per_unit:      float
    kelly_full:       float
    kelly_qtr:        float
    stake_dollars:    float
    recommendation:   str


# ═══════════════════════════════════════════════════════════════════════════════
# Cached prediction runner
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=8)
def _get_predictions(
    game_date: str,
    season: str,
    prop_line: float,
    over_odds: int,
    under_odds: int,
    min_line: float,
) -> pd.DataFrame:
    """
    Run the full pipeline (cached per unique argument combination).
    Cache is invalidated on server restart.

    Live odds flow
    --------------
    1. Run model inference to get per-player ENSEMBLE_PROB predictions.
    2. Try to fetch live prop lines + American odds from The-Odds-API.
    3. Merge live odds onto predictions by player name.
    4. For players with no live match, fall back to the model-derived season-avg
       line and the user-supplied (or default -110/-110) odds.
    5. Pass the combined odds DataFrame to evaluate_props().
    """
    preds = run_daily(game_date=game_date, season=season,
                      prop_line=prop_line, min_line=min_line)
    if preds.empty:
        return preds

    # ── 1. Attempt live odds fetch ─────────────────────────────────────────────
    live_odds_df = pd.DataFrame()
    odds_source  = "model-derived (no API key or fetch failed)"
    if os.environ.get("ODDS_API_KEY", "").strip():
        try:
            event_ids    = fetch_today_event_ids(game_date)
            live_odds_df = fetch_player_props(event_ids)
            if not live_odds_df.empty:
                odds_source = f"live ({live_odds_df['BOOK'].value_counts().idxmax()})"
        except Exception:
            logger.exception("Live odds fetch failed — falling back to model lines.")

    # ── 2. Build the odds DataFrame passed to evaluate_props() ────────────────
    if not live_odds_df.empty:
        # Merge live odds onto predictions; unmatched players get fallback odds
        matched_odds, unmatched_preds = merge_with_predictions(preds, live_odds_df)

        # Fallback odds for unmatched players: use model line + user-supplied odds
        if not unmatched_preds.empty:
            fallback_odds = pd.DataFrame({
                "PLAYER_NAME": unmatched_preds["PLAYER_NAME"],
                "PROP_TYPE":   unmatched_preds["PROP_TYPE"],
                "LINE":        unmatched_preds["LINE"],
                "OVER_ODDS":   over_odds,
                "UNDER_ODDS":  under_odds,
            })
            odds_df = pd.concat([matched_odds, fallback_odds], ignore_index=True)
        else:
            odds_df = matched_odds
    else:
        # No live odds available — use model lines and user/default odds uniformly
        odds_df = pd.DataFrame({
            "PLAYER_NAME": preds["PLAYER_NAME"],
            "PROP_TYPE":   preds["PROP_TYPE"],
            "LINE":        preds["LINE"],
            "OVER_ODDS":   over_odds,
            "UNDER_ODDS":  under_odds,
        })

    logger.info("Odds source: %s", odds_source)

    # bankroll is supplied per-request — KELLY_QTR is a fraction, stake is
    # computed client-side by multiplying kelly_qtr × bankroll.
    result = evaluate_props(preds, odds=odds_df, bankroll=1_000.0)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/api/predictions/daily",
    response_model=DailyResponse,
    summary="Daily player prop predictions",
    description=(
        "Returns ensemble model probabilities, EV, and Kelly stakes for "
        "every active player on today's (or a specified) game slate."
    ),
)
async def get_daily_predictions(
    game_date: str | None = Query(
        default=None,
        description="Date in YYYY-MM-DD format. Defaults to today.",
        example="2025-04-13",
    ),
    season: str = Query(
        default="2024-25",
        description="NBA season string, e.g. '2024-25'.",
    ),
    prop_line: float = Query(
        default=19.5,
        description="Fallback points line for players with no season history.",
    ),
    over_odds: int = Query(
        default=-110,
        description="American moneyline for the OVER side.",
    ),
    under_odds: int = Query(
        default=-110,
        description="American moneyline for the UNDER side.",
    ),
    min_line: float = Query(
        default=5.5,
        description=(
            "Minimum player line to include. Players below this threshold "
            "are excluded — their props would be voided (DNP) by sportsbooks."
        ),
    ),
) -> DailyResponse:
    target_date = game_date or date.today().strftime("%Y-%m-%d")

    try:
        df = _get_predictions(
            game_date=target_date,
            season=season,
            prop_line=prop_line,
            over_odds=over_odds,
            under_odds=under_odds,
            min_line=min_line,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction pipeline failed")
        raise HTTPException(status_code=500,
                            detail="Prediction pipeline error.") from exc

    if df.empty:
        return DailyResponse(
            game_date=target_date,
            prop_line=prop_line,
            total_props=0,
            bet_count=0,
            predictions=[],
        )

    predictions = [
        PropPrediction(
            player_name=    row["PLAYER_NAME"],
            team=           row["TEAM"],
            opponent=       row["OPPONENT"],
            is_home=        int(row["IS_HOME"]),
            prop_type=      row["PROP_TYPE"],
            line=           float(row["LINE"]),
            lr_prob=        float(row.get("LR_PROB",   row["ENSEMBLE_PROB"])),
            xgb_prob=       float(row.get("XGB_PROB",  row["ENSEMBLE_PROB"])),
            mlp_prob=       float(row.get("MLP_PROB",  row["ENSEMBLE_PROB"])),
            ensemble_prob=  float(row["ENSEMBLE_PROB"]),
            book_over_odds= int(row["BOOK_OVER_ODDS"]),
            book_under_odds=int(row["BOOK_UNDER_ODDS"]),
            over_implied=   float(row["OVER_IMPLIED"]),
            edge=           float(row["EDGE"]),
            ev_per_unit=    float(row["EV_PER_UNIT"]),
            kelly_full=     float(row["KELLY_FULL"]),
            kelly_qtr=      float(row["KELLY_QTR"]),
            bet_side=       row["BET_SIDE"],
            recommendation= row["RECOMMENDATION"],
        )
        for row in df.to_dict("records")
    ]

    return DailyResponse(
        game_date=target_date,
        prop_line=prop_line,
        total_props=len(predictions),
        bet_count=sum(1 for p in predictions if p.recommendation == "BET"),
        predictions=predictions,
    )


@app.post(
    "/api/predictions/sgp",
    response_model=SGPResponse,
    summary="Evaluate a Same Game Parlay",
)
async def evaluate_sgp_endpoint(body: SGPRequest) -> SGPResponse:
    if len(body.legs) < 2:
        raise HTTPException(status_code=422, detail="SGP requires at least 2 legs.")

    try:
        from src.sgp_engine import (  # noqa: PLC0415
            PropLeg, build_correlation_matrix, evaluate_sgp,
        )
        import pandas as pd  # noqa: PLC0415

        raw = pd.read_parquet(
            "data/raw/player_game_logs_2024_25.parquet"
        )
        corr_df = build_correlation_matrix(raw, same_team_only=True)
        legs = [
            PropLeg(
                player_name=l.player_name,
                stat=l.stat,
                line=l.line,
                over_prob=l.over_prob,
            )
            for l in body.legs
        ]
        result = evaluate_sgp(legs, corr_df, body.book_sgp_odds,
                              bankroll=body.bankroll, min_edge=body.min_edge)
    except Exception as exc:
        logger.exception("SGP evaluation failed")
        raise HTTPException(status_code=500, detail="SGP evaluation error.") from exc

    return SGPResponse(
        legs=[f"{l.player_name} {l.stat}>{l.line}" for l in result.legs],
        correlation=result.correlation,
        independent_prob=result.independent_prob,
        adjusted_prob=result.adjusted_prob,
        book_sgp_odds=result.book_sgp_odds,
        book_implied_prob=result.book_implied_prob,
        edge=result.edge,
        ev_per_unit=result.ev_per_unit,
        kelly_full=result.kelly_full,
        kelly_qtr=result.kelly_qtr,
        stake_dollars=result.stake_dollars,
        recommendation=result.recommendation,
    )
