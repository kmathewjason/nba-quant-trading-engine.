"""
live_odds.py — The-Odds-API v4 live prop lines ingestion
=========================================================
Fetches real-time NBA player_points Over/Under lines and American odds from
The-Odds-API, normalises them into a DataFrame that matches the column contract
expected by optimizer.evaluate_props(), and merges with the model predictions
produced by predict_daily.run_daily().

Authentication
--------------
Set the environment variable ODDS_API_KEY before starting the server:

    export ODDS_API_KEY="your_key_here"          # terminal / shell profile
    echo "ODDS_API_KEY=your_key_here" >> .env    # .env file (never commit)

The key is read at call time via os.environ — it is never stored in code.

The-Odds-API v4 endpoint used
------------------------------
GET /v4/sports/basketball_nba/events/{event_id}/odds
    ?regions=us
    &markets=player_points
    &oddsFormat=american
    &bookmakers=draftkings,fanduel

Fallback behaviour
------------------
If the API key is missing, the network call fails, or no props are returned,
every function returns an empty DataFrame and logs a warning. The caller
(api.py) falls back to the model-derived lines and default -110/-110 odds.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_BASE_URL       = "https://api.the-odds-api.com/v4"
_SPORT          = "basketball_nba"
_MARKET         = "player_points"
_ODDS_FORMAT    = "american"
_REGIONS        = "us"

# Preferred book order — first book found per player is used
_BOOK_PRIORITY  = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"]

_REQUEST_TIMEOUT = 10   # seconds per HTTP call
_RETRY_DELAY     = 1.5  # seconds between retries on transient errors
_MAX_RETRIES     = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _api_key() -> str | None:
    """Read the API key from the environment. Returns None if not set."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    return key if key else None


def _get(url: str, params: dict) -> dict | list | None:
    """
    HTTP GET with retry on transient errors (5xx / connection timeout).
    Returns None on any failure; logs the reason.
    """
    key = _api_key()
    if not key:
        logger.warning(
            "ODDS_API_KEY is not set. Live odds unavailable. "
            "Set the environment variable and restart the server."
        )
        return None

    params = {**params, "apiKey": key}

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.debug("Odds API %s — status %s | remaining credits: %s",
                         url, resp.status_code, remaining)

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                logger.error("Odds API: invalid or missing API key (401).")
                return None
            if resp.status_code == 422:
                logger.error("Odds API: unprocessable request (422) — %s", resp.text)
                return None
            if resp.status_code == 429:
                logger.warning("Odds API: rate limit hit (429). Retrying…")
            else:
                logger.warning("Odds API: unexpected status %s", resp.status_code)

        except requests.exceptions.Timeout:
            logger.warning("Odds API: request timed out (attempt %d/%d).",
                           attempt, _MAX_RETRIES)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Odds API: connection error — %s (attempt %d/%d).",
                           exc, attempt, _MAX_RETRIES)

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_today_event_ids(game_date: str | None = None) -> list[str]:
    """
    Return The-Odds-API event IDs for NBA games on the given date.

    Parameters
    ----------
    game_date : str or None
        'YYYY-MM-DD'. Defaults to today if None.

    Returns
    -------
    list of event_id strings (empty list on failure).
    """
    from datetime import date as _date, datetime, timezone

    if game_date is None:
        game_date = _date.today().isoformat()

    # Build ISO 8601 window covering the full calendar day in UTC
    day_start = f"{game_date}T00:00:00Z"
    day_end   = f"{game_date}T23:59:59Z"

    url  = f"{_BASE_URL}/sports/{_SPORT}/events"
    data = _get(url, {
        "dateFormat":       "iso",
        "commenceTimeFrom": day_start,
        "commenceTimeTo":   day_end,
    })

    if not data:
        return []

    ids = [evt["id"] for evt in data if isinstance(evt, dict) and "id" in evt]
    logger.info("Found %d Odds-API event IDs for %s", len(ids), game_date)
    return ids


def fetch_player_props(
    event_ids: list[str],
    book_priority: list[str] = _BOOK_PRIORITY,
) -> pd.DataFrame:
    """
    Fetch player_points Over/Under lines for each event and return a
    normalised DataFrame.

    Parameters
    ----------
    event_ids : list[str]
        Event IDs from fetch_today_event_ids().
    book_priority : list[str]
        Bookmaker keys in preference order. The first book that has a line
        for a given player is used.

    Returns
    -------
    pd.DataFrame with columns:
        PLAYER_NAME   – normalised title-case name
        PROP_TYPE     – always "PTS"
        LINE          – the Over/Under points line (float, e.g. 24.5)
        OVER_ODDS     – American integer odds for the Over
        UNDER_ODDS    – American integer odds for the Under
        BOOK          – bookmaker key that supplied this line
    Empty DataFrame if no data available.
    """
    if not event_ids:
        return pd.DataFrame()

    rows: list[dict] = []

    for event_id in event_ids:
        url  = f"{_BASE_URL}/sports/{_SPORT}/events/{event_id}/odds"
        data = _get(url, {
            "regions":     _REGIONS,
            "markets":     _MARKET,
            "oddsFormat":  _ODDS_FORMAT,
            "bookmakers":  ",".join(book_priority),
        })

        if not data:
            continue

        bookmakers: list[dict] = data.get("bookmakers", [])

        # Build a priority-ordered index: book_key → market outcomes
        book_map: dict[str, list[dict]] = {}
        for bm in bookmakers:
            bk = bm.get("key", "")
            for mkt in bm.get("markets", []):
                if mkt.get("key") == _MARKET:
                    book_map[bk] = mkt.get("outcomes", [])

        # For each player, pick the highest-priority book that has both sides
        player_seen: set[str] = set()
        for book_key in book_priority:
            outcomes = book_map.get(book_key, [])
            if not outcomes:
                continue

            # Group outcomes by player name
            by_player: dict[str, dict] = {}
            for outcome in outcomes:
                name  = outcome.get("description", "").strip().title()
                side  = outcome.get("name", "").upper()   # "Over" / "Under"
                price = outcome.get("price")
                point = outcome.get("point")
                if not name or side not in ("OVER", "UNDER") or price is None:
                    continue
                if name not in by_player:
                    by_player[name] = {}
                by_player[name][side] = {"price": int(price), "point": float(point)}

            for name, sides in by_player.items():
                if name in player_seen:
                    continue
                if "OVER" not in sides or "UNDER" not in sides:
                    continue
                # Lines should match across sides — use the Over point
                line = sides["OVER"]["point"]
                rows.append({
                    "PLAYER_NAME": name,
                    "PROP_TYPE":   "PTS",
                    "LINE":        line,
                    "OVER_ODDS":   sides["OVER"]["price"],
                    "UNDER_ODDS":  sides["UNDER"]["price"],
                    "BOOK":        book_key,
                })
                player_seen.add(name)

        time.sleep(0.3)  # polite pacing between event requests

    if not rows:
        logger.warning("No player_points props returned from Odds API.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["PLAYER_NAME"]).reset_index(drop=True)
    logger.info(
        "Live odds fetched: %d player props from %s",
        len(df), df["BOOK"].value_counts().to_dict(),
    )
    return df


def merge_with_predictions(
    predictions: pd.DataFrame,
    live_odds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Left-join live odds onto model predictions by player name.

    Name matching is best-effort: both sides are lowercased and stripped
    before joining, then the original casing is restored.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of predict_daily.run_daily() — must have PLAYER_NAME column.
    live_odds : pd.DataFrame
        Output of fetch_player_props() — must have PLAYER_NAME column.

    Returns
    -------
    (merged_odds_df, unmatched_predictions_df)
        merged_odds_df        — odds DataFrame re-keyed to match predictions
        unmatched_predictions — rows in predictions with no live odds match
    """
    if live_odds.empty:
        return pd.DataFrame(), predictions.copy()

    # Normalise join keys
    pred_key = predictions["PLAYER_NAME"].str.strip().str.lower()
    odds_key = live_odds["PLAYER_NAME"].str.strip().str.lower()

    live_keyed = live_odds.copy()
    live_keyed["_join_key"] = odds_key

    pred_keyed = predictions[["PLAYER_NAME", "PROP_TYPE"]].copy()
    pred_keyed["_join_key"] = pred_key

    merged = pred_keyed.merge(
        live_keyed[["_join_key", "LINE", "OVER_ODDS", "UNDER_ODDS", "BOOK"]],
        on="_join_key",
        how="left",
    )

    matched_mask   = merged["LINE"].notna()
    unmatched_preds = predictions[~matched_mask.values].copy()

    # Build the odds DataFrame in the shape evaluate_props() expects
    odds_out = merged[matched_mask].copy()
    odds_out["PLAYER_NAME"] = predictions["PLAYER_NAME"].values[matched_mask.values]
    odds_out["OVER_ODDS"]   = odds_out["OVER_ODDS"].astype(int)
    odds_out["UNDER_ODDS"]  = odds_out["UNDER_ODDS"].astype(int)
    odds_out = odds_out[["PLAYER_NAME", "PROP_TYPE", "LINE",
                          "OVER_ODDS", "UNDER_ODDS"]].reset_index(drop=True)

    n_matched = len(odds_out)
    n_total   = len(predictions)
    logger.info(
        "Odds matched: %d/%d players (%.0f%%). %d unmatched → default -110 odds.",
        n_matched, n_total, 100 * n_matched / max(n_total, 1),
        len(unmatched_preds),
    )
    return odds_out, unmatched_preds
