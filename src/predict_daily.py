"""
predict_daily.py — Daily player prop inference
===============================================
Run every morning to produce an ensemble probability for each active
player crossing a given points prop line.

Usage
-----
    python -m src.predict_daily                        # today, default line
    python -m src.predict_daily --date 2025-04-13
    python -m src.predict_daily --date 2025-04-13 --line 24.5 --season 2024-25

Output
------
A DataFrame with columns:
    PLAYER_NAME, TEAM, OPPONENT, IS_HOME, PROP_TYPE, LINE,
    ENSEMBLE_PROB, XGB_PROB, MLP_PROB, LR_PROB,
    MODEL_EDGE  (ensemble − book_implied, filled when --odds supplied)

Also saved to data/predictions/predictions_YYYY-MM-DD.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import (
    PlayerGameLogs,
    ScoreboardV2,
    TeamGameLogs,
)
from nba_api.stats.static import teams as nba_teams_static

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent
_MODELS_DIR = _ROOT / "models"
_RAW_DIR    = _ROOT / "data" / "raw"
_PRED_DIR   = _ROOT / "data" / "predictions"
_PRED_DIR.mkdir(parents=True, exist_ok=True)

_REQUEST_DELAY = 1.2   # seconds between nba_api calls

# Feature columns — must match training exactly
_FEATURE_COLS = [
    "DAYS_REST", "IS_HOME",
    "OPP_DEFRTG_ROLL5", "OPP_DEFRTG_ROLL10",
    "TEAM_PACE_ROLL5",  "TEAM_PACE_ROLL10",
    "PTS_ROLL3",  "PTS_ROLL5",  "PTS_ROLL10",
    "MIN_ROLL3",  "MIN_ROLL5",  "MIN_ROLL10",
    "USG_ROLL3",  "USG_ROLL5",  "USG_ROLL10",
    "TS_ROLL3",   "TS_ROLL5",   "TS_ROLL10",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Artefact loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_artefacts() -> dict:
    """Load all saved prop model artefacts from models/."""
    def _pkl(name: str):
        p = _MODELS_DIR / name
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artefact: {p}. Run validate_train.py first."
            )
        with open(p, "rb") as fh:
            return pickle.load(fh)

    meta = json.loads((_MODELS_DIR / "prop_meta.json").read_text())
    return {
        "lr":      _pkl("prop_lr.pkl"),
        "xgb":     _pkl("prop_xgb.pkl"),
        "mlp":     _pkl("prop_mlp.pkl"),
        "imputer": _pkl("prop_imputer.pkl"),
        "scaler":  _pkl("prop_scaler.pkl"),
        "meta":    meta,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Schedule & roster helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_today_games(game_date: str) -> pd.DataFrame:
    """
    Return a DataFrame of today's matchups with columns:
        GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID, HOME_ABB, AWAY_ABB
    Returns empty DataFrame if no games scheduled.
    """
    logger.info("Fetching schedule for %s …", game_date)
    sb = ScoreboardV2(game_date=game_date)
    time.sleep(_REQUEST_DELAY)
    games = sb.get_data_frames()[0]

    if games.empty:
        logger.warning("No games found for %s.", game_date)
        return pd.DataFrame()

    team_map = {t["id"]: t["abbreviation"]
                for t in nba_teams_static.get_teams()}

    games["HOME_ABB"] = games["HOME_TEAM_ID"].map(team_map)
    games["AWAY_ABB"] = games["VISITOR_TEAM_ID"].map(team_map)
    logger.info("Found %d games: %s", len(games),
                list(zip(games["AWAY_ABB"], games["HOME_ABB"])))
    return games[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID",
                  "HOME_ABB", "AWAY_ABB"]]


# ═══════════════════════════════════════════════════════════════════════════════
# Rolling feature construction
# ═══════════════════════════════════════════════════════════════════════════════

def _rolling_player(df: pd.DataFrame, cols: list[str], window: int,
                    alias: str | None = None) -> pd.DataFrame:
    """shift(1) rolling mean per PLAYER_ID — no leakage."""
    alias = alias or cols[0]
    rolled = (
        df.sort_values(["PLAYER_ID", "GAME_DATE"])
        .groupby("PLAYER_ID")[cols]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    rolled.columns = [f"{c.replace(cols[0], alias) if len(cols)==1 else c}_ROLL{window}"
                      for c in rolled.columns]
    return rolled


def _pace_defrtg(team_logs: pd.DataFrame, game_ids: list[str]) -> pd.DataFrame:
    """
    Compute rolling pace & opponent def-rtg for the teams in today's games.
    Mirrors features.build_player_prop_features logic exactly.
    """
    tl = team_logs[["TEAM_ID", "GAME_ID", "GAME_DATE",
                    "FGA", "FTA", "OREB", "TOV", "PTS"]].copy()
    tl["GAME_DATE"] = pd.to_datetime(tl["GAME_DATE"])
    tl["PACE_EST"]  = tl["FGA"] - tl["OREB"] + tl["TOV"] + 0.44 * tl["FTA"]

    opp = tl[["GAME_ID", "TEAM_ID", "PTS", "PACE_EST"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS",
                 "PACE_EST": "OPP_PACE"})
    tl = tl.merge(opp, on="GAME_ID", how="left")
    tl = tl[tl["TEAM_ID"] != tl["OPP_TEAM_ID"]]
    tl["DEFRTG_EST"] = (tl["OPP_PTS"] / tl["PACE_EST"].clip(lower=1)) * 100
    tl = tl.sort_values(["TEAM_ID", "GAME_DATE"])

    for w in (5, 10):
        for col, out in [("PACE_EST",   f"TEAM_PACE_ROLL{w}"),
                         ("DEFRTG_EST", f"OPP_DEFRTG_ROLL{w}")]:
            tl[out] = (
                tl.groupby("TEAM_ID")[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    ctx_cols = ["TEAM_ID", "TEAM_PACE_ROLL5", "TEAM_PACE_ROLL10",
                "OPP_DEFRTG_ROLL5", "OPP_DEFRTG_ROLL10"]
    # Return the most recent row per team (= last game's rolling value)
    latest = (
        tl[tl["GAME_ID"].isin(game_ids)]
        [ctx_cols]
        .drop_duplicates(subset=["TEAM_ID"], keep="last")
    )
    return latest


def build_inference_features(
    player_logs: pd.DataFrame,
    team_logs: pd.DataFrame,
    today_games: pd.DataFrame,
    game_date: str,
    prop_line: float = 19.5,
) -> pd.DataFrame:
    """
    Construct the exact same 18 feature columns used at training time,
    but for today's active players using only their historical logs.

    Parameters
    ----------
    player_logs  : raw season player logs
    team_logs    : raw season team logs
    today_games  : output of _fetch_today_games()
    game_date    : 'YYYY-MM-DD' string for today
    prop_line    : points threshold used as reference

    Returns
    -------
    DataFrame with one row per player-game, _FEATURE_COLS filled,
    plus metadata: PLAYER_ID, PLAYER_NAME, TEAM_ID, GAME_ID,
                   HOME_ABB, AWAY_ABB, IS_HOME, DAYS_REST.
    """
    target_date = pd.Timestamp(game_date)

    # ── Player history up to (not including) today ────────────────────────────
    pl = player_logs.copy()
    pl["GAME_DATE"] = pd.to_datetime(pl["GAME_DATE"])
    pl = pl[(pl["GAME_DATE"] < target_date) & (pl["MIN"].fillna(0) > 0)]
    pl = pl.sort_values(["PLAYER_ID", "GAME_DATE"])

    # Derived per-game cols (mirrors features.py)
    pl["USG_APPROX"] = (pl["FGA"] + 0.44 * pl["FTA"] + pl["TOV"]) / pl["MIN"].clip(lower=1) * 48
    denom = 2.0 * (pl["FGA"] + 0.44 * pl["FTA"])
    pl["TS_PCT"]     = pl["PTS"] / denom.replace(0, np.nan)

    # Rolling windows (shift already applied inside)
    roll_src   = ["PTS", "MIN", "USG_APPROX", "TS_PCT"]
    roll_alias = ["PTS", "MIN", "USG", "TS"]
    for w in (3, 5, 10):
        rolled = (
            pl.groupby("PLAYER_ID")[roll_src]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        rolled.columns = [f"{a}_ROLL{w}" for a in roll_alias]
        pl = pd.concat([pl, rolled], axis=1)

    # Days of rest
    pl["DAYS_REST"] = (
        pl.groupby("PLAYER_ID")["GAME_DATE"]
        .diff().dt.days.clip(upper=10).fillna(3)
    )

    # ── Per-player season scoring average (used as the prop line) ─────────────
    # Round to nearest 0.5 — mirrors real sportsbook line granularity
    season_avg = (
        pl.groupby("PLAYER_ID")["PTS"]
        .mean()
        .round(0)
        .sub(0.5)          # e.g. avg 22.0 → line 21.5
        .clip(lower=0.5)
        .rename("SEASON_PTS_AVG")
    )

    # ── One synthetic "today" row per active player ───────────────────────────
    # Latest historical row per player gives us the freshest rolling values
    latest = pl.groupby("PLAYER_ID").tail(1).copy()
    latest = latest.join(season_avg, on="PLAYER_ID")

    # Merge today's game info
    team_ids_today = pd.concat([
        today_games[["GAME_ID", "HOME_TEAM_ID", "HOME_ABB"]].rename(
            columns={"HOME_TEAM_ID": "TEAM_ID", "HOME_ABB": "TEAM_ABB"}),
        today_games[["GAME_ID", "VISITOR_TEAM_ID", "AWAY_ABB"]].rename(
            columns={"VISITOR_TEAM_ID": "TEAM_ID", "AWAY_ABB": "TEAM_ABB"}),
    ])
    home_ids = set(today_games["HOME_TEAM_ID"])
    team_ids_today["IS_HOME"] = team_ids_today["TEAM_ID"].isin(home_ids).astype(int)

    # Add opponent abbreviation
    home_side = today_games[["HOME_TEAM_ID", "AWAY_ABB", "HOME_ABB"]].rename(
        columns={"HOME_TEAM_ID": "TEAM_ID", "AWAY_ABB": "OPP_ABB"})
    away_side = today_games[["VISITOR_TEAM_ID", "HOME_ABB", "AWAY_ABB"]].rename(
        columns={"VISITOR_TEAM_ID": "TEAM_ID", "HOME_ABB": "OPP_ABB"})
    opp_map = pd.concat([home_side[["TEAM_ID", "OPP_ABB"]],
                         away_side[["TEAM_ID", "OPP_ABB"]]])

    rows = latest.merge(team_ids_today, on="TEAM_ID", how="inner")
    rows = rows.merge(opp_map, on="TEAM_ID", how="left")

    # Override game-context columns with today's values
    last_game_date = rows["GAME_DATE"]
    rows["DAYS_REST"] = (target_date - last_game_date).dt.days.clip(upper=10)
    rows["GAME_DATE"] = target_date

    # ── Team pace / def-rtg context (from historical team logs) ──────────────
    # Use IDs of the most recent completed games to get rolling values
    all_game_ids = team_logs["GAME_ID"].tolist()
    ctx = _pace_defrtg(team_logs, all_game_ids)
    rows = rows.merge(ctx, on="TEAM_ID", how="left",
                      suffixes=("_old", ""))
    # Drop old duplicates if suffix collision
    for c in ["OPP_DEFRTG_ROLL5", "OPP_DEFRTG_ROLL10",
              "TEAM_PACE_ROLL5",  "TEAM_PACE_ROLL10"]:
        old = c + "_old"
        if old in rows.columns:
            rows[c] = rows[c].fillna(rows[old])
            rows.drop(columns=[old], inplace=True)

    # Player-specific line: season avg rounded to nearest 0.5
    # Fall back to the global prop_line for players with no history
    rows["prop_line"] = rows["SEASON_PTS_AVG"].fillna(prop_line)

    return rows.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

def predict(
    feature_rows: pd.DataFrame,
    artefacts: dict,
) -> pd.DataFrame:
    """
    Run ensemble inference and return the results DataFrame.

    Returns
    -------
    DataFrame with columns:
        PLAYER_NAME, TEAM, OPPONENT, IS_HOME, PROP_TYPE, LINE,
        LR_PROB, XGB_PROB, MLP_PROB, ENSEMBLE_PROB
    sorted by ENSEMBLE_PROB descending.
    """
    feat_cols = artefacts["meta"]["feature_cols"]
    weights   = artefacts["meta"]["ensemble_weights"]

    # Impute + scale
    X = feature_rows[feat_cols].values
    X = artefacts["imputer"].transform(X)
    X = artefacts["scaler"].transform(X)

    lr_proba  = artefacts["lr"].predict_proba(X)[:, 1]
    xgb_proba = artefacts["xgb"].predict_proba(X)[:, 1]
    mlp_proba = artefacts["mlp"].predict_proba(X)[:, 1]
    ens_proba = weights["xgb"] * xgb_proba + weights["mlp"] * mlp_proba

    out = pd.DataFrame({
        "PLAYER_NAME":   feature_rows["PLAYER_NAME"].values,
        "TEAM":          feature_rows["TEAM_ABB"].values,
        "OPPONENT":      feature_rows["OPP_ABB"].values,
        "IS_HOME":       feature_rows["IS_HOME"].values,
        "PROP_TYPE":     "PTS",
        "LINE":          feature_rows["prop_line"].values,
        "LR_PROB":       np.round(lr_proba, 4),
        "XGB_PROB":      np.round(xgb_proba, 4),
        "MLP_PROB":      np.round(mlp_proba, 4),
        "ENSEMBLE_PROB": np.round(ens_proba, 4),
    })

    out = out.sort_values("ENSEMBLE_PROB", ascending=False).reset_index(drop=True)
    logger.info("Generated predictions for %d players", len(out))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_daily(
    game_date: str | None = None,
    season: str = "2024-25",
    prop_line: float = 19.5,
    min_line: float = 5.5,
) -> pd.DataFrame:
    """
    Full daily inference pipeline.

    Parameters
    ----------
    game_date : str or None
        'YYYY-MM-DD'. Defaults to today.
    season : str
        NBA season string for fetching logs ('2024-25').
    prop_line : float
        Fallback points threshold for players with no season history.
    min_line : float
        Drop players whose computed season-average line is below this
        threshold. Default 5.5 — removes DNP-risk bench players whose
        props sportsbooks would void on a no-play rather than grade Under.

    Returns
    -------
    pd.DataFrame of predictions, also saved to data/predictions/.
    """
    if game_date is None:
        game_date = date.today().strftime("%Y-%m-%d")

    logger.info("=== Daily prop inference for %s (line=%.1f, min_line=%.1f) ===",
                game_date, prop_line, min_line)

    # ── Load artefacts ────────────────────────────────────────────────────────
    artefacts = _load_artefacts()
    logger.info("Artefacts loaded. Ensemble weights: %s",
                artefacts["meta"]["ensemble_weights"])

    # ── Fetch today's schedule ────────────────────────────────────────────────
    # ScoreboardV2 wants MM/DD/YYYY
    api_date = datetime.strptime(game_date, "%Y-%m-%d").strftime("%m/%d/%Y")
    today_games = _fetch_today_games(api_date)
    if today_games.empty:
        logger.warning("No games today — nothing to predict.")
        return pd.DataFrame()

    # ── Load historical logs (from already-cached raw Parquets if available) ──
    player_log_path = _RAW_DIR / f"player_game_logs_{season.replace('-','_')}.parquet"
    team_log_path   = _RAW_DIR / f"team_game_logs_{season.replace('-','_')}.parquet"

    if player_log_path.exists() and team_log_path.exists():
        logger.info("Loading cached raw logs from %s", _RAW_DIR)
        player_logs = pd.read_parquet(player_log_path)
        team_logs   = pd.read_parquet(team_log_path)
    else:
        logger.info("Cached logs not found — fetching from nba_api …")
        from src.ingestion import fetch_player_game_logs, fetch_team_game_logs
        player_logs = fetch_player_game_logs(season)
        team_logs   = fetch_team_game_logs(season)

    # ── Build inference features ──────────────────────────────────────────────
    feature_rows = build_inference_features(
        player_logs, team_logs, today_games,
        game_date=game_date, prop_line=prop_line,
    )

    if feature_rows.empty:
        logger.warning("No active players found for today's teams.")
        return pd.DataFrame()

    # ── DNP filter: drop bench players below the minimum line threshold ───────
    before = len(feature_rows)
    feature_rows = feature_rows[feature_rows["prop_line"] >= min_line].reset_index(drop=True)
    logger.info(
        "Built features for %d player-rows (%d dropped: line < %.1f)",
        len(feature_rows), before - len(feature_rows), min_line,
    )

    # ── Run ensemble inference ────────────────────────────────────────────────
    predictions = predict(feature_rows, artefacts)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = _PRED_DIR / f"predictions_{game_date}.csv"
    predictions.to_csv(out_path, index=False)
    logger.info("Predictions saved → %s", out_path)

    return predictions


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NBA daily player prop predictor")
    p.add_argument("--date",   default=None,
                   help="Game date YYYY-MM-DD (default: today)")
    p.add_argument("--season", default="2024-25",
                   help="NBA season string e.g. 2024-25")
    p.add_argument("--line",   type=float, default=19.5,
                   help="Points prop line (default: 19.5)")
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = _parse_args()
    df = run_daily(game_date=args.date, season=args.season, prop_line=args.line)
    if not df.empty:
        print("\n" + df.to_string(index=False))
