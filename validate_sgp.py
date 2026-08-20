"""Smoke-test for sgp_engine.py."""
import logging, sys
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

import pandas as pd
from src.sgp_engine import (
    build_correlation_matrix, lookup_correlation,
    joint_prob_two_legs, PropLeg, evaluate_sgp, evaluate_sgp_slate,
)

raw = pd.read_parquet("data/raw/player_game_logs_2024_25.parquet")
corr_df = build_correlation_matrix(raw, same_team_only=True)
print(f"Pairs computed: {len(corr_df)}")

pairs = [
    ("LeBron James",  "PTS", "Anthony Davis",  "PTS"),
    ("LeBron James",  "AST", "Anthony Davis",  "PTS"),
    ("Nikola Jokic",  "AST", "Jamal Murray",   "PTS"),
    ("Stephen Curry", "PTS", "Klay Thompson",  "PTS"),
    ("Luka Doncic",   "AST", "Kyrie Irving",   "PTS"),
]
print("\nPair                                                     rho      n")
print("-" * 70)
for pa, sa, pb, sb in pairs:
    r = lookup_correlation(corr_df, pa, sa, pb, sb)
    mask = (
        ((corr_df.player_a==pa)&(corr_df.stat_a==sa)&(corr_df.player_b==pb)&(corr_df.stat_b==sb)) |
        ((corr_df.player_a==pb)&(corr_df.stat_a==sb)&(corr_df.player_b==pa)&(corr_df.stat_b==sa))
    )
    n = int(corr_df[mask]["n_games"].sum())
    label = pa + " " + sa + " / " + pb + " " + sb
    print(f"{label:<53}  {r:>+.4f}  {n}")

print()
rho = lookup_correlation(corr_df, "LeBron James", "AST", "Anthony Davis", "PTS")
p_a, p_b = 0.55, 0.48
p_ind = p_a * p_b
p_adj = joint_prob_two_legs(p_a, p_b, rho)
print(f"LeBron AST>7.5 + AD PTS>24.5  (rho={rho:+.4f})")
print(f"  P_independent={p_ind:.4f}  P_adjusted={p_adj:.4f}"
      f"  diff={p_adj-p_ind:+.4f} ({(p_adj/p_ind-1)*100:+.1f}%)")

parlays = [
    ([PropLeg("LeBron James","AST",7.5,0.55), PropLeg("Anthony Davis","PTS",24.5,0.48)], +250),
    ([PropLeg("Nikola Jokic","AST",9.5,0.52), PropLeg("Jamal Murray","PTS",19.5,0.58)], +220),
    ([PropLeg("Stephen Curry","PTS",27.5,0.44), PropLeg("Klay Thompson","PTS",17.5,0.51)], +195),
]
results = evaluate_sgp_slate(parlays, corr_df, bankroll=1000, min_edge=0.02)
print()
print(results[[
    "LEGS","CORRELATION","P_INDEPENDENT","P_ADJUSTED",
    "BOOK_IMPLIED","EDGE","EV_PER_UNIT","KELLY_QTR","STAKE_DOLLARS","RECOMMENDATION"
]].to_string(index=False))
