from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)


def compute_top_decile_lift(y_true: np.ndarray, y_proba: np.ndarray, decile: float = 0.10) -> float:
    """
    Compute Lift at top decile (e.g., top 10% highest predicted propensity leads).
    Lift = (Conversion rate in top decile) / (Overall base conversion rate).
    """
    total_samples = len(y_true)
    if total_samples == 0:
        return 0.0

    k = max(1, int(total_samples * decile))
    sorted_indices = np.argsort(y_proba)[::-1]
    top_k_indices = sorted_indices[:k]

    base_rate = float(np.mean(y_true))
    if base_rate == 0:
        return 0.0

    top_k_rate = float(np.mean(y_true[top_k_indices]))
    return float(top_k_rate / base_rate)


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive metric suite for imbalanced lead scoring.
    Includes PR-AUC, ROC-AUC, F1, Precision, Recall, Brier Score, and Decile Lifts.
    """
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    pr_auc = float(average_precision_score(y_true, y_proba))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_proba))

    decile_1_lift = compute_top_decile_lift(y_true, y_proba, decile=0.10)
    decile_2_lift = compute_top_decile_lift(y_true, y_proba, decile=0.20)

    # Conversion captured in top 10% and top 20%
    sorted_idx = np.argsort(y_proba)[::-1]
    top_10_pct_count = max(1, int(len(y_true) * 0.10))
    top_20_pct_count = max(1, int(len(y_true) * 0.20))
    total_converters = np.sum(y_true)

    captured_top_10 = (
        float(np.sum(y_true[sorted_idx[:top_10_pct_count]]) / total_converters)
        if total_converters > 0
        else 0.0
    )
    captured_top_20 = (
        float(np.sum(y_true[sorted_idx[:top_20_pct_count]]) / total_converters)
        if total_converters > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "brier_score": round(brier, 4),
        "top_10_pct_lift": round(decile_1_lift, 2),
        "top_20_pct_lift": round(decile_2_lift, 2),
        "top_10_pct_captured_conversions": round(captured_top_10, 4),
        "top_20_pct_captured_conversions": round(captured_top_20, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find the classification threshold tau* in (0.05, 0.95) that maximizes the target metric."""
    best_threshold = 0.5
    best_score = -1.0

    # Test thresholds from 0.05 to 0.95 in 0.01 increments
    for th in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_proba >= th).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = float(th)

    return round(best_threshold, 2)


def bootstrap_metric_confidence_intervals(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate empirical 95% bootstrap confidence intervals for key metrics on the test set.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    boot_pr_auc: List[float] = []
    boot_roc_auc: List[float] = []
    boot_f1: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Avoid degenerate resamples with single class
        if len(np.unique(y_true[idx])) < 2:
            continue

        y_t = y_true[idx]
        y_p = y_proba[idx]
        y_pred = (y_p >= threshold).astype(int)

        boot_pr_auc.append(average_precision_score(y_t, y_p))
        boot_roc_auc.append(roc_auc_score(y_t, y_p))
        boot_f1.append(f1_score(y_t, y_pred, zero_division=0))

    alpha = (1.0 - confidence_level) / 2.0
    results = {}
    for name, vals in [("pr_auc", boot_pr_auc), ("roc_auc", boot_roc_auc), ("f1", boot_f1)]:
        lower = float(np.percentile(vals, alpha * 100))
        upper = float(np.percentile(vals, (1.0 - alpha) * 100))
        results[name] = {
            "lower_95": round(lower, 4),
            "upper_95": round(upper, 4),
            "mean": round(float(np.mean(vals)), 4),
        }
    return results
