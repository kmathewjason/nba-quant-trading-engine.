"""
train.py — Model training & ensembling
=======================================
Two entry points:

  train(matchup_df)   — team win-probability pipeline (XGBoost + LogReg ensemble)
  train_prop(prop_df) — player points-prop pipeline:
                          1. Logistic Regression baseline
                          2. XGBoost with TimeSeriesSplit hyper-param tuning
                          3. Scikit-learn MLPClassifier (Feedforward Neural Network)
                          4. Weighted soft-vote ensemble of XGBoost + MLP
                          5. Accuracy, ROC-AUC, Brier score evaluation

All artefacts saved to models/.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Column identifiers ────────────────────────────────────────────────────────

_TEAM_FEATURE_PREFIX = ("home_", "away_")
_TEAM_TARGET_COL     = "target"

_PROP_FEATURE_COLS = [
    "DAYS_REST", "IS_HOME",
    "OPP_DEFRTG_ROLL5", "OPP_DEFRTG_ROLL10",
    "TEAM_PACE_ROLL5",  "TEAM_PACE_ROLL10",
    "PTS_ROLL3",  "PTS_ROLL5",  "PTS_ROLL10",
    "MIN_ROLL3",  "MIN_ROLL5",  "MIN_ROLL10",
    "USG_ROLL3",  "USG_ROLL5",  "USG_ROLL10",
    "TS_ROLL3",   "TS_ROLL5",   "TS_ROLL10",
]
_PROP_TARGET_COL = "PTS_OVER_LINE"
_DATE_COL        = "GAME_DATE"

# ── Hyper-parameters ──────────────────────────────────────────────────────────

_TEAM_XGB_PARAMS = {
    "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
}
_TEAM_LR_PARAMS = {
    "max_iter": 1000, "random_state": 42, "solver": "lbfgs", "C": 0.1,
}

# XGBoost grid — n_jobs=1 keeps OpenMP single-threaded (macOS ARM safe)
_XGB_PARAM_GRID = {
    "max_depth":        [3, 5],
    "learning_rate":    [0.03, 0.07],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}
_XGB_FIXED = {
    "n_estimators": 500, "eval_metric": "logloss",
    "random_state": 42,  "n_jobs": 1,
}

# MLP — three hidden layers mirrors the PyTorch FNN architecture
_MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation":         "relu",
    "solver":             "adam",
    "alpha":              1e-4,      # L2 regularisation (≈ weight_decay)
    "batch_size":         512,
    "learning_rate_init": 1e-3,
    "max_iter":           60,
    "early_stopping":     True,
    "validation_fraction": 0.15,
    "n_iter_no_change":   8,         # matches FNN patience
    "random_state":       42,
    "verbose":            False,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _chronological_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict chronological split — no shuffling."""
    df = df.sort_values(_DATE_COL).reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "roc_auc":  round(roc_auc_score(y_true, proba), 4),
        "log_loss": round(log_loss(y_true, proba), 4),
        "brier":    round(brier_score_loss(y_true, proba), 4),
    }


def _print_metrics(label: str, m: dict) -> None:
    logger.info(
        "%s — accuracy=%.4f  roc_auc=%.4f  brier=%.4f  log_loss=%.4f",
        label, m["accuracy"], m["roc_auc"], m["brier"], m["log_loss"],
    )


def _save_model(obj, filename: str) -> None:
    path = MODELS_DIR / filename
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    logger.info("Artefact saved → %s", path)


def _save_json(obj: dict, filename: str) -> None:
    path = MODELS_DIR / filename
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    logger.info("JSON saved → %s", path)


def load_model(filename: str):
    """Load a pickled model from models/."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline 1 — Team win-probability  (original, preserved)
# ═══════════════════════════════════════════════════════════════════════════════

def _team_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith(_TEAM_FEATURE_PREFIX) and "_ROLL" in c]


def train(
    matchup_df: pd.DataFrame,
    test_fraction: float = 0.2,
    ensemble_weights: tuple[float, float] = (0.6, 0.4),
) -> dict:
    """
    Train XGBoost + Logistic Regression ensemble on team matchup features.

    Returns
    -------
    dict — xgb, lr, imputer, metrics, feature_cols, ensemble_weights
    """
    feat_cols = _team_feature_cols(matchup_df)
    if not feat_cols:
        raise ValueError("No rolling feature columns found in matchup_df.")

    train_df, test_df = _chronological_split(matchup_df, test_fraction)

    X_train = train_df[feat_cols].values
    y_train = train_df[_TEAM_TARGET_COL].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df[_TEAM_TARGET_COL].values

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    logger.info("Training team XGBoost on %d samples …", len(X_train))
    xgb = XGBClassifier(**_TEAM_XGB_PARAMS)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    logger.info("Training team Logistic Regression on %d samples …", len(X_train))
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**_TEAM_LR_PARAMS)),
    ])
    lr_pipe.fit(X_train, y_train)

    w_xgb, w_lr = ensemble_weights
    ens_proba = (w_xgb * xgb.predict_proba(X_test)[:, 1]
                 + w_lr * lr_pipe.predict_proba(X_test)[:, 1])
    metrics = _evaluate(y_test, ens_proba)
    _print_metrics("Team ensemble", metrics)

    _save_model(xgb,     "xgb_model.pkl")
    _save_model(lr_pipe, "lr_model.pkl")
    _save_model(imputer, "imputer.pkl")

    return {
        "xgb": xgb, "lr": lr_pipe, "imputer": imputer,
        "metrics": metrics, "feature_cols": feat_cols,
        "ensemble_weights": ensemble_weights,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline 2 — Player points-prop
# ═══════════════════════════════════════════════════════════════════════════════

def _tune_xgb(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> tuple[XGBClassifier, dict]:
    """
    Grid-search XGBoost hyper-parameters with TimeSeriesSplit.
    Returns (model refitted on full X, best_params dict).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params: dict = {}
    best_score: float = float("inf")

    candidates = list(ParameterGrid(_XGB_PARAM_GRID))
    logger.info("XGBoost grid search: %d candidates × %d CV folds …",
                len(candidates), n_splits)

    for params in candidates:
        fold_scores: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            clf = XGBClassifier(**{**_XGB_FIXED, **params})
            clf.fit(X[train_idx], y[train_idx],
                    eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            val_proba = clf.predict_proba(X[val_idx])[:, 1]
            fold_scores.append(log_loss(y[val_idx], val_proba))
        mean_ll = float(np.mean(fold_scores))
        if mean_ll < best_score:
            best_score  = mean_ll
            best_params = params

    logger.info("Best XGBoost params (val log-loss=%.4f): %s", best_score, best_params)
    best_model = XGBClassifier(**{**_XGB_FIXED, **best_params})
    best_model.fit(X, y, verbose=False)
    return best_model, best_params


def train_prop(
    prop_df: pd.DataFrame,
    test_fraction: float = 0.2,
    xgb_weight: float = 0.5,
    mlp_weight: float = 0.5,
    xgb_cv_splits: int = 5,
) -> dict:
    """
    Full player points-prop training pipeline.

    Steps
    -----
    1. Chronological train/test split
    2. Impute NaNs (mean) → StandardScaler
    3. Logistic Regression baseline
    4. XGBoost with TimeSeriesSplit hyper-param grid search
    5. MLPClassifier (256→128→64, ReLU, Adam, early stopping)
    6. Ensemble: xgb_weight × XGB + mlp_weight × MLP
    7. Evaluate all models; save artefacts to models/

    Parameters
    ----------
    prop_df : pd.DataFrame
        Output of features.build_player_prop_features().
    test_fraction : float
        Chronologically latest fraction held out for evaluation.
    xgb_weight, mlp_weight : float
        Must sum to 1.0.
    xgb_cv_splits : int
        TimeSeriesSplit folds for XGBoost tuning.

    Returns
    -------
    dict — lr, xgb, mlp, imputer, scaler,
           best_xgb_params, feature_cols, metrics, ensemble_weights
    """
    if abs(xgb_weight + mlp_weight - 1.0) > 1e-6:
        raise ValueError("xgb_weight + mlp_weight must equal 1.0")

    feat_cols = [c for c in _PROP_FEATURE_COLS if c in prop_df.columns]
    if not feat_cols:
        raise ValueError("No expected feature columns found in prop_df.")

    # ── 1. Split ──────────────────────────────────────────────────────────────
    train_df, test_df = _chronological_split(prop_df, test_fraction)
    logger.info("Prop split → train: %d  test: %d", len(train_df), len(test_df))

    X_raw      = train_df[feat_cols].values
    y_train    = train_df[_PROP_TARGET_COL].values.astype(int)
    X_test_raw = test_df[feat_cols].values
    y_test     = test_df[_PROP_TARGET_COL].values.astype(int)

    # ── 2. Impute + Scale ─────────────────────────────────────────────────────
    imputer     = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_test  = scaler.transform(X_test_imp)

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    logger.info("Class imbalance pos_weight=%.2f", pos_weight)

    all_metrics: dict[str, dict] = {}

    # ── 3. Logistic Regression baseline ──────────────────────────────────────
    logger.info("=== Logistic Regression baseline ===")
    lr = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs",
                            class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    all_metrics["logistic_regression"] = _evaluate(y_test, lr_proba)
    _print_metrics("LR baseline", all_metrics["logistic_regression"])

    # ── 4. XGBoost + TimeSeriesSplit ──────────────────────────────────────────
    logger.info("=== XGBoost + TimeSeriesSplit grid search ===")
    xgb_model, best_xgb_params = _tune_xgb(X_train, y_train, n_splits=xgb_cv_splits)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    all_metrics["xgboost"] = _evaluate(y_test, xgb_proba)
    _print_metrics("XGBoost", all_metrics["xgboost"])

    # ── 5. MLP (Feedforward Neural Network) ──────────────────────────────────
    logger.info("=== MLP Neural Network (256→128→64, ReLU, Adam) ===")
    mlp = MLPClassifier(**_MLP_PARAMS)
    mlp.fit(X_train, y_train)
    logger.info("MLP converged in %d iterations", mlp.n_iter_)
    mlp_proba = mlp.predict_proba(X_test)[:, 1]
    all_metrics["mlp"] = _evaluate(y_test, mlp_proba)
    _print_metrics("MLP", all_metrics["mlp"])

    # ── 6. Ensemble (XGBoost + MLP) ───────────────────────────────────────────
    ens_proba = xgb_weight * xgb_proba + mlp_weight * mlp_proba
    all_metrics["ensemble"] = _evaluate(y_test, ens_proba)
    _print_metrics("Ensemble (XGB+MLP)", all_metrics["ensemble"])

    # ── 7. Persist artefacts ──────────────────────────────────────────────────
    _save_model(lr,        "prop_lr.pkl")
    _save_model(xgb_model, "prop_xgb.pkl")
    _save_model(mlp,       "prop_mlp.pkl")
    _save_model(imputer,   "prop_imputer.pkl")
    _save_model(scaler,    "prop_scaler.pkl")
    _save_json(
        {
            "best_xgb_params":  best_xgb_params,
            "ensemble_weights": {"xgb": xgb_weight, "mlp": mlp_weight},
            "feature_cols":     feat_cols,
            "input_dim":        len(feat_cols),
            "mlp_architecture": list(_MLP_PARAMS["hidden_layer_sizes"]),
            "metrics":          all_metrics,
        },
        "prop_meta.json",
    )

    return {
        "lr": lr, "xgb": xgb_model, "mlp": mlp,
        "imputer": imputer, "scaler": scaler,
        "best_xgb_params": best_xgb_params,
        "feature_cols":    feat_cols,
        "metrics":         all_metrics,
        "ensemble_weights": {"xgb": xgb_weight, "mlp": mlp_weight},
    }


def predict_prop(
    df: pd.DataFrame,
    xgb_model: XGBClassifier,
    mlp_model: MLPClassifier,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_cols: list[str],
    xgb_weight: float = 0.5,
    mlp_weight: float = 0.5,
) -> np.ndarray:
    """
    Generate ensemble P(PTS > line) for new/unseen rows.

    Returns
    -------
    np.ndarray of shape (n,).
    """
    X = scaler.transform(imputer.transform(df[feature_cols].values))
    return (xgb_weight * xgb_model.predict_proba(X)[:, 1]
            + mlp_weight * mlp_model.predict_proba(X)[:, 1])
