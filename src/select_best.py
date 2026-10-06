import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np

try:
    from src.config import (
        MODEL_PATH,
        METADATA_PATH,
        METRICS_PATH,
        FEATURE_MANIFEST_PATH,
        RANDOM_STATE,
    )
    from src.preprocessing import prepare_data
    from src.train_model import train_all_models
    from src.evaluate import (
        evaluate_predictions,
        find_optimal_threshold,
        bootstrap_metric_confidence_intervals,
    )
except ModuleNotFoundError:
    from config import (
        MODEL_PATH,
        METADATA_PATH,
        METRICS_PATH,
        FEATURE_MANIFEST_PATH,
        RANDOM_STATE,
    )
    from preprocessing import prepare_data
    from train_model import train_all_models
    from evaluate import (
        evaluate_predictions,
        find_optimal_threshold,
        bootstrap_metric_confidence_intervals,
    )


def run_model_selection_pipeline():
    """
    Orchestrate training, evaluation, champion selection, threshold tuning,
    bootstrap confidence interval estimation, and artifact serialization.
    """
    print("=" * 70)
    print("LEAD CONVERSION MODEL TRAINING & SELECTION PIPELINE")
    print("=" * 70)

    # 1. Prepare data and preprocessor
    preprocessor, X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_data()

    # 2. Train and cross-validate candidate models
    fitted_models, cv_results = train_all_models(X_train, y_train)

    # 3. Evaluate each model on the Validation Set
    val_comparisons = {}
    best_val_pr_auc = -1.0
    champion_name = None
    champion_model = None

    print("\n" + "=" * 70)
    print("VALIDATION SET PERFORMANCE COMPARISON")
    print("=" * 70)

    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_val_proba = model.predict(X_val).astype(float)

        # Baseline default evaluation (threshold = 0.5)
        default_eval = evaluate_predictions(y_val, y_val_proba, threshold=0.5)

        # Find optimal threshold tau* on validation set
        opt_thresh = find_optimal_threshold(y_val, y_val_proba, metric="f1")
        tuned_eval = evaluate_predictions(y_val, y_val_proba, threshold=opt_thresh)

        val_comparisons[name] = {
            "cv_pr_auc_mean": cv_results[name]["cv_pr_auc_mean"],
            "cv_pr_auc_std": cv_results[name]["cv_pr_auc_std"],
            "val_pr_auc": tuned_eval["pr_auc"],
            "val_roc_auc": tuned_eval["roc_auc"],
            "val_f1_default": default_eval["f1"],
            "optimal_threshold": opt_thresh,
            "val_f1_tuned": tuned_eval["f1"],
            "val_precision_tuned": tuned_eval["precision"],
            "val_recall_tuned": tuned_eval["recall"],
            "val_top_10_lift": tuned_eval["top_10_pct_lift"],
            "val_top_20_lift": tuned_eval["top_20_pct_lift"],
        }

        print(
            f"{name:30s} | Val PR-AUC: {tuned_eval['pr_auc']:.4f} | "
            f"ROC-AUC: {tuned_eval['roc_auc']:.4f} | "
            f"F1 (th={opt_thresh:.2f}): {tuned_eval['f1']:.4f} | "
            f"Top 10% Lift: {tuned_eval['top_10_pct_lift']:.2f}x"
        )

        # Select champion based on Validation PR-AUC
        if tuned_eval["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = tuned_eval["pr_auc"]
            champion_name = name
            champion_model = model

    print("\n" + "=" * 70)
    print(f"CHAMPION MODEL SELECTED: {champion_name} (Val PR-AUC = {best_val_pr_auc:.4f})")
    print("=" * 70)

    # 4. Final Evaluation on Unseen Test Set
    y_test_proba = champion_model.predict_proba(X_test)[:, 1]
    champion_threshold = val_comparisons[champion_name]["optimal_threshold"]
    test_eval = evaluate_predictions(y_test, y_test_proba, threshold=champion_threshold)

    # 5. Bootstrap Confidence Intervals on Test Set
    bootstrap_cis = bootstrap_metric_confidence_intervals(
        y_test,
        y_test_proba,
        threshold=champion_threshold,
        n_bootstrap=500,
        random_state=RANDOM_STATE,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION (CHAMPION MODEL)")
    print("=" * 70)
    print(f"Decision Threshold:       {champion_threshold:.2f}")
    print(f"Test PR-AUC:              {test_eval['pr_auc']:.4f}  [95% CI: {bootstrap_cis['pr_auc']['lower_95']:.4f} - {bootstrap_cis['pr_auc']['upper_95']:.4f}]")
    print(f"Test ROC-AUC:             {test_eval['roc_auc']:.4f}  [95% CI: {bootstrap_cis['roc_auc']['lower_95']:.4f} - {bootstrap_cis['roc_auc']['upper_95']:.4f}]")
    print(f"Test F1-Score:            {test_eval['f1']:.4f}  [95% CI: {bootstrap_cis['f1']['lower_95']:.4f} - {bootstrap_cis['f1']['upper_95']:.4f}]")
    print(f"Test Precision:           {test_eval['precision']:.4f}")
    print(f"Test Recall:              {test_eval['recall']:.4f}")
    print(f"Test Brier Score:         {test_eval['brier_score']:.4f}")
    print(f"Top 10% Decile Lift:      {test_eval['top_10_pct_lift']:.2f}x (captures {test_eval['top_10_pct_captured_conversions']*100:.1f}% of all converters)")
    print(f"Top 20% Decile Lift:      {test_eval['top_20_pct_lift']:.2f}x (captures {test_eval['top_20_pct_captured_conversions']*100:.1f}% of all converters)")
    print(f"Confusion Matrix:         TN={test_eval['confusion_matrix']['tn']}, FP={test_eval['confusion_matrix']['fp']}, FN={test_eval['confusion_matrix']['fn']}, TP={test_eval['confusion_matrix']['tp']}")

    # 6. Save Model Artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, MODEL_PATH)

    metrics_payload = {
        "champion_model": champion_name,
        "optimal_threshold": champion_threshold,
        "validation_comparisons": val_comparisons,
        "test_evaluation": test_eval,
        "bootstrap_95_confidence_intervals": bootstrap_cis,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    metadata_payload = {
        "model_name": champion_name,
        "model_class": champion_model.__class__.__name__,
        "model_parameters": champion_model.get_params(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_features_count": len(feature_names),
        "optimal_threshold": champion_threshold,
        "operational_mode": "pre_call_lead_scoring (leakage-free, excludes duration)",
        "test_pr_auc": test_eval["pr_auc"],
        "test_roc_auc": test_eval["roc_auc"],
        "test_f1": test_eval["f1"],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    print(f"\nChampion model saved to: {MODEL_PATH}")
    print(f"Metrics saved to:        {METRICS_PATH}")
    print(f"Metadata saved to:       {METADATA_PATH}")
    print("=" * 70)

    return champion_model, metrics_payload, metadata_payload


if __name__ == "__main__":
    run_model_selection_pipeline()
