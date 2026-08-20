"""
main.py — End-to-end NBA prop prediction pipeline runner

Usage:
    python main.py [--season 2024-25] [--bankroll 1000] [--min-edge 0.02]
                   [--prop-line 19.5] [--over-odds -110] [--under-odds -110]
"""

import argparse
import logging
import multiprocessing
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBA Prop Prediction Pipeline")
    parser.add_argument("--season",     default="2024-25", help="NBA season (e.g. 2024-25)")
    parser.add_argument("--bankroll",   type=float, default=1_000.0, help="Bankroll in dollars")
    parser.add_argument("--min-edge",   type=float, default=0.02,    help="Minimum edge threshold")
    parser.add_argument("--prop-line",  type=float, default=19.5,    help="Points prop line")
    parser.add_argument("--over-odds",  type=int,   default=-110,    help="American odds for over")
    parser.add_argument("--under-odds", type=int,   default=-110,    help="American odds for under")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Ingest ──────────────────────────────────────────────────────────────
    logger.info("=== STEP 1: Data Ingestion  (season=%s) ===", args.season)
    from src.ingestion import fetch_all
    raw = fetch_all(season=args.season)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    logger.info("=== STEP 2: Feature Engineering ===")
    from src.features import build_player_prop_features
    prop_df = build_player_prop_features(raw["player_logs"], raw["team_logs"])
    logger.info("Prop feature matrix: %s rows × %s cols", *prop_df.shape)

    # ── 3. Train prop ensemble ─────────────────────────────────────────────────
    logger.info("=== STEP 3: Prop Model Training ===")
    from src.train import train_prop
    result = train_prop(prop_df, prop_line=args.prop_line)
    logger.info("Prop model metrics: %s", result["metrics"])

    # ── 4. Daily predictions ───────────────────────────────────────────────────
    logger.info("=== STEP 4: Daily Predictions ===")
    from src.predict_daily import run_daily
    preds = run_daily(prop_line=args.prop_line)
    logger.info("Predictions ready: %d rows", len(preds))

    # ── 5. Evaluate props with EV + Kelly ──────────────────────────────────────
    logger.info("=== STEP 5: Evaluate Props (bankroll=$%.0f) ===", args.bankroll)
    from src.optimizer import evaluate_props

    recommendations = evaluate_props(
        preds,
        prop_line=args.prop_line,
        over_odds=args.over_odds,
        under_odds=args.under_odds,
        bankroll=args.bankroll,
        min_edge=args.min_edge,
    )
    bets = recommendations[recommendations["recommendation"] == "BET"]
    print(f"\n=== BET RECOMMENDATIONS ({len(bets)} of {len(recommendations)} props) ===")
    if bets.empty:
        print("No edges found above min_edge threshold.")
    else:
        cols = ["player_name", "side", "edge", "ev", "kelly_qtr", "stake"]
        print(bets[cols].to_string(index=False))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
