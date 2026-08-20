"""
features.py — Rolling stats & preprocessing

Two independent pipelines:
  1. build_team_features / build_matchup_features  — team win-probability model
  2. build_player_prop_features                    — player points prop model

Both write Parquet files to data/processed/.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _player_rolling(
    df: pd.DataFrame,
    cols: list[str],
    window: int,
    group_col: str = "PLAYER_ID",
    date_col: str = "GAME_DATE",
) -> pd.DataFrame:
    """
    Compute shift(1) rolling mean for *cols* within each player group.
    shift(1) ensures only prior games feed the feature — no data leakage.
    Returns a new DataFrame of rolled columns (same index as *df*).
    """
    rolled = (
        df.sort_values([group_col, date_col])
        .groupby(group_col)[cols]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    rolled.columns = [f"{c}_ROLL{window}" for c in rolled.columns]
    return rolled


def _team_rolling(
    df: pd.DataFrame,
    cols: list[str],
    window: int,
) -> pd.DataFrame:
    """Rolling mean for team-level columns (grouped by TEAM_ID)."""
    rolled = (
        df.sort_values(["TEAM_ID", "GAME_DATE"])
        .groupby("TEAM_ID")[cols]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    rolled.columns = [f"{c}_ROLL{window}" for c in rolled.columns]
    return rolled


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline 1 — Team win-probability features  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

TEAM_STAT_COLS = [
    "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
    "REB", "AST", "TOV", "STL", "BLK",
    "PLUS_MINUS",
]


def _rolling_mean(df: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    """Rolling mean for team pipeline (adds columns in-place style)."""
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    rolled = (
        df.groupby("TEAM_ID")[cols]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    rolled.columns = [f"{c}_ROLL{window}" for c in rolled.columns]
    return pd.concat([df, rolled], axis=1)


def build_team_features(
    team_logs: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build rolling-window team features from raw team game logs.

    Parameters
    ----------
    team_logs : pd.DataFrame
        Output of ingestion.fetch_team_game_logs().
    windows : list[int]
        Rolling window sizes in games. Defaults to [5, 10].

    Returns
    -------
    pd.DataFrame — one row per team-game with rolling features appended.
    """
    if windows is None:
        windows = [5, 10]

    df = team_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    for w in windows:
        df = _rolling_mean(df, TEAM_STAT_COLS, w)

    path = PROCESSED_DIR / "team_features.parquet"
    df.to_parquet(path, index=False)
    logger.info("Team features saved → %s  (%d rows)", path, len(df))
    return df


def build_matchup_features(team_features: pd.DataFrame) -> pd.DataFrame:
    """
    Join home and away team rolling features on GAME_ID to create
    a single matchup row suitable for model training.

    Returns
    -------
    pd.DataFrame — one row per game with home_* / away_* columns.
    """
    roll_cols = [c for c in team_features.columns if "_ROLL" in c]
    base_cols = ["GAME_ID", "GAME_DATE", "TEAM_ID", "WL"] + roll_cols

    df = team_features[base_cols].copy()

    home_mask = team_features["MATCHUP"].str.contains("vs\\.", na=False)
    home = df[home_mask.values].add_prefix("home_").rename(
        columns={"home_GAME_ID": "GAME_ID", "home_GAME_DATE": "GAME_DATE"}
    )
    away = df[~home_mask.values].add_prefix("away_").rename(
        columns={"away_GAME_ID": "GAME_ID"}
    )

    matchup = home.merge(away, on="GAME_ID", how="inner")
    matchup["target"] = (matchup["home_WL"] == "W").astype(int)

    path = PROCESSED_DIR / "matchup_features.parquet"
    matchup.to_parquet(path, index=False)
    logger.info("Matchup features saved → %s  (%d rows)", path, len(matchup))
    return matchup


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline 2 — Player points-prop features
# ═══════════════════════════════════════════════════════════════════════════════

# Columns for per-player rolling windows
_PLAYER_ROLL_COLS = ["PTS", "MIN", "FGA", "FTA", "FG3A"]

# Columns used to derive pace / defensive efficiency at the team level
# Pace ≈ (FGA + 0.44·FTA - OREB + TOV) per game
# DefRtg ≈ points allowed per 100 possessions, approximated per game log
_TEAM_PACE_COLS   = ["FGA", "FTA", "OREB", "TOV", "PTS", "MIN"]


def _compute_usage_rate(df: pd.DataFrame) -> pd.Series:
    """
    Approximate usage rate per row:
        USG% = 100 × (FGA + 0.44·FTA + TOV) / (team_MIN/5 × (FGA+0.44·FTA+TOV)/MIN)

    Simplified to the individual-possession share formula that only requires
    per-player box-score columns:
        USG% ≈ (FGA + 0.44·FTA + TOV) / (MIN / 48 * constant)

    We use the orthodox per-minute approximation available from a single-game log:
        USG% = (FGA + 0.44·FTA + TOV) / MAX(MIN, 1) × 48
    """
    numerator = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    minutes = df["MIN"].clip(lower=1)
    return (numerator / minutes * 48).rename("USG_APPROX")


def _compute_true_shooting(df: pd.DataFrame) -> pd.Series:
    """
    True Shooting % = PTS / (2 × (FGA + 0.44 × FTA))
    Returns NaN when denominator is 0 (no shot attempts).
    """
    denom = 2.0 * (df["FGA"] + 0.44 * df["FTA"])
    return (df["PTS"] / denom.replace(0, np.nan)).rename("TS_PCT")


def _build_team_pace_defrtg(team_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Derive rolling team pace and defensive efficiency from team game logs.

    Pace (possessions) ≈ FGA - OREB + TOV + 0.44·FTA
    DefRtg (pts allowed per 100 poss) is approximated by joining each game's
    opponent PTS and dividing by the team's own pace estimate.

    Returns a DataFrame indexed by (TEAM_ID, GAME_ID) with:
        OPP_DEFRTG_ROLL5, OPP_DEFRTG_ROLL10,
        TEAM_PACE_ROLL5,  TEAM_PACE_ROLL10
    to be joined onto the player log.
    """
    tl = team_logs[["TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP",
                    "FGA", "FTA", "OREB", "TOV", "PTS", "MIN"]].copy()
    tl["GAME_DATE"] = pd.to_datetime(tl["GAME_DATE"])

    # Raw pace estimate (possessions)
    tl["PACE_EST"] = tl["FGA"] - tl["OREB"] + tl["TOV"] + 0.44 * tl["FTA"]

    # Join opponent PTS for defensive rating
    # Each game appears twice (home + away); self-join on GAME_ID
    opp = tl[["GAME_ID", "TEAM_ID", "PTS", "PACE_EST"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS", "PACE_EST": "OPP_PACE"}
    )
    tl = tl.merge(opp, on="GAME_ID", how="left")
    tl = tl[tl["TEAM_ID"] != tl["OPP_TEAM_ID"]]

    # DefRtg: points the opponent scored per 100 of *our* possessions
    tl["DEFRTG_EST"] = (tl["OPP_PTS"] / tl["PACE_EST"].clip(lower=1)) * 100

    tl = tl.sort_values(["TEAM_ID", "GAME_DATE"])

    for w in (5, 10):
        for col, out in [("PACE_EST", f"TEAM_PACE_ROLL{w}"),
                         ("DEFRTG_EST", f"OPP_DEFRTG_ROLL{w}")]:
            tl[out] = (
                tl.groupby("TEAM_ID")[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    keep_cols = ["TEAM_ID", "GAME_ID",
                 "TEAM_PACE_ROLL5", "TEAM_PACE_ROLL10",
                 "OPP_DEFRTG_ROLL5", "OPP_DEFRTG_ROLL10"]
    return tl[keep_cols].drop_duplicates(subset=["TEAM_ID", "GAME_ID"])


def build_player_prop_features(
    player_logs: pd.DataFrame,
    team_logs: pd.DataFrame,
    prop_line: float = 19.5,
    roll_windows: list[int] | None = None,
    output_name: str = "processed_features",
) -> pd.DataFrame:
    """
    Build a player points-prop feature set with strict chronological ordering.

    Features produced
    -----------------
    Rolling (windows 3, 5, 10) per player — shift(1), no leakage:
        PTS_ROLLn, MIN_ROLLn, USG_ROLLn, TS_ROLLn

    Contextual:
        DAYS_REST       — calendar days since player's last game (capped at 10)
        IS_HOME         — 1 if player's team is home, 0 if away
        OPP_DEFRTG_ROLL5/10  — opponent's rolling defensive rating
        TEAM_PACE_ROLL5/10   — player's team rolling pace

    Target:
        PTS_OVER_LINE   — 1 if PTS > prop_line, else 0

    Metadata columns retained:
        PLAYER_ID, PLAYER_NAME, TEAM_ID, GAME_ID, GAME_DATE, PTS, prop_line

    Parameters
    ----------
    player_logs : pd.DataFrame
        Raw player game logs (output of ingestion.fetch_player_game_logs).
    team_logs : pd.DataFrame
        Raw team game logs (output of ingestion.fetch_team_game_logs).
    prop_line : float
        The points threshold for the binary target.  Default 19.5.
    roll_windows : list[int]
        Windows for rolling averages.  Defaults to [3, 5, 10].
    output_name : str
        Stem of the output Parquet file written to data/processed/.

    Returns
    -------
    pd.DataFrame — one row per player-game, sorted by GAME_DATE ascending.
    """
    if roll_windows is None:
        roll_windows = [3, 5, 10]

    # ── 1. Clean player logs ──────────────────────────────────────────────────
    df = player_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # Drop DNP / inactive rows (MIN == 0 or NaN)
    df = df[df["MIN"].fillna(0) > 0].copy()

    # ── 2. Derived per-game columns ───────────────────────────────────────────
    df["USG_APPROX"] = _compute_usage_rate(df)
    df["TS_PCT"]     = _compute_true_shooting(df)
    df["IS_HOME"]    = df["MATCHUP"].str.contains("vs\\.", na=False).astype(int)

    # Days of rest: diff between consecutive game dates per player, capped at 10
    df["DAYS_REST"] = (
        df.groupby("PLAYER_ID")["GAME_DATE"]
        .diff()
        .dt.days
        .clip(upper=10)
        # First game of the season → assume median rest (3 days)
        .fillna(3)
    )

    # ── 3. Rolling player features ────────────────────────────────────────────
    roll_src_cols = ["PTS", "MIN", "USG_APPROX", "TS_PCT"]
    roll_out_names = ["PTS", "MIN", "USG", "TS"]

    for w in roll_windows:
        rolled = _player_rolling(df, roll_src_cols, w)
        # Rename to friendlier aliases
        rename_map = {
            f"{src}_ROLL{w}": f"{alias}_ROLL{w}"
            for src, alias in zip(roll_src_cols, roll_out_names)
        }
        rolled = rolled.rename(columns=rename_map)
        df = pd.concat([df, rolled], axis=1)

    # ── 4. Team pace & opponent defensive efficiency ───────────────────────────
    team_ctx = _build_team_pace_defrtg(team_logs)
    df = df.merge(team_ctx, on=["TEAM_ID", "GAME_ID"], how="left")

    # ── 5. Binary target ──────────────────────────────────────────────────────
    df["prop_line"]      = prop_line
    df["PTS_OVER_LINE"]  = (df["PTS"] > prop_line).astype(int)

    # ── 6. Select & order output columns ─────────────────────────────────────
    meta_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID",
                 "GAME_DATE", "PTS", "prop_line"]
    ctx_cols  = ["DAYS_REST", "IS_HOME",
                 "OPP_DEFRTG_ROLL5", "OPP_DEFRTG_ROLL10",
                 "TEAM_PACE_ROLL5",  "TEAM_PACE_ROLL10"]
    roll_cols = [
        f"{alias}_ROLL{w}"
        for alias in roll_out_names
        for w in roll_windows
    ]
    target_col = ["PTS_OVER_LINE"]

    out_cols = meta_cols + ctx_cols + roll_cols + target_col
    # Keep only columns that exist (guards against any optional merges)
    out_cols = [c for c in out_cols if c in df.columns]
    df = df[out_cols].sort_values("GAME_DATE").reset_index(drop=True)

    # ── 7. Persist ────────────────────────────────────────────────────────────
    path = PROCESSED_DIR / f"{output_name}.parquet"
    df.to_parquet(path, index=False)
    logger.info(
        "Player prop features saved → %s  (%d rows, %d cols, prop_line=%.1f)",
        path, len(df), len(df.columns), prop_line,
    )
    logger.info(
        "Target distribution — OVER: %d (%.1f%%)  UNDER: %d (%.1f%%)",
        df["PTS_OVER_LINE"].sum(),
        100 * df["PTS_OVER_LINE"].mean(),
        (df["PTS_OVER_LINE"] == 0).sum(),
        100 * (1 - df["PTS_OVER_LINE"].mean()),
    )
    return df


def chronological_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    date_col: str = "GAME_DATE",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict time-series split — no shuffling, no random seed required.
    All test rows are chronologically *after* all train rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted or sortable by *date_col*.
    test_fraction : float
        Fraction of rows (by count) assigned to the test set.

    Returns
    -------
    (train_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    train = df.iloc[:cut].copy()
    test  = df.iloc[cut:].copy()
    logger.info(
        "Chronological split → train: %d rows (up to %s) | test: %d rows (from %s)",
        len(train), train[date_col].max().date(),
        len(test),  test[date_col].min().date(),
    )
    return train, test


# ── Utility ────────────────────────────────────────────────────────────────────

def load_processed(name: str) -> pd.DataFrame:
    """Load a processed Parquet file by name (without extension)."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path)
