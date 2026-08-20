"""
ingestion.py — NBA API fetching
Pulls game logs, team stats, and player stats from the nba_api package
and persists raw data to data/raw/ as Parquet files.
"""

import os
import logging
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelogs,
    playergamelogs,
)
from nba_api.stats.static import teams as nba_teams

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# nba_api rate-limit: ~1 req/s is safe
_REQUEST_DELAY = 1.0


def _save(df: pd.DataFrame, name: str) -> Path:
    """Persist a DataFrame to Parquet and return the file path."""
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows → %s", len(df), path)
    return path


def fetch_league_games(season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch all regular-season game records for a given season.

    Parameters
    ----------
    season : str
        NBA season string, e.g. '2024-25'.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Fetching league games for season %s", season)
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",          # NBA
        season_type_nullable="Regular Season",
    )
    time.sleep(_REQUEST_DELAY)
    df = finder.get_data_frames()[0]
    _save(df, f"league_games_{season.replace('-', '_')}")
    return df


def fetch_team_game_logs(season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch per-game team stats for every team in the given season.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Fetching team game logs for season %s", season)
    logs = teamgamelogs.TeamGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    time.sleep(_REQUEST_DELAY)
    df = logs.get_data_frames()[0]
    _save(df, f"team_game_logs_{season.replace('-', '_')}")
    return df


def fetch_player_game_logs(season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch per-game player stats for the given season.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Fetching player game logs for season %s", season)
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    time.sleep(_REQUEST_DELAY)
    df = logs.get_data_frames()[0]
    _save(df, f"player_game_logs_{season.replace('-', '_')}")
    return df


def fetch_all(season: str = "2024-25") -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper: fetch league games, team logs, and player logs.

    Returns
    -------
    dict with keys 'league_games', 'team_logs', 'player_logs'
    """
    return {
        "league_games": fetch_league_games(season),
        "team_logs": fetch_team_game_logs(season),
        "player_logs": fetch_player_game_logs(season),
    }
