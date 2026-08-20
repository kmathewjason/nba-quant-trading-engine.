"""Validation script for train_prop pipeline."""
import logging
import sys
import os
import json
import multiprocessing


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    import pandas as pd
    from src.train import train_prop

    prop_df = pd.read_parquet("data/processed/processed_features.parquet")
    result = train_prop(prop_df, test_fraction=0.2, xgb_weight=0.5, mlp_weight=0.5)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║              FINAL MODEL COMPARISON                 ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"{'Model':<28} {'Accuracy':>9} {'ROC-AUC':>9} {'Brier':>8} {'LogLoss':>9}")
    print("─" * 66)
    for name, m in result["metrics"].items():
        print(f"{name:<28} {m['accuracy']:>9.4f} {m['roc_auc']:>9.4f} {m['brier']:>8.4f} {m['log_loss']:>9.4f}")

    print("\nBest XGBoost params:", result["best_xgb_params"])

    print("\nSaved artefacts:")
    for f in sorted(os.listdir("models")):
        size = os.path.getsize(f"models/{f}")
        print(f"  models/{f}  ({size/1024:.1f} KB)")

    meta = json.load(open("models/prop_meta.json"))
    print("\nprop_meta.json ensemble_weights:", meta["ensemble_weights"])
    print("prop_meta.json input_dim:", meta["input_dim"])


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
