from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    from src.config import RANDOM_STATE, CV_FOLDS
    from src.preprocessing import prepare_data
except ModuleNotFoundError:
    from config import RANDOM_STATE, CV_FOLDS
    from preprocessing import prepare_data


def get_candidate_models(pos_weight: float, random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Define baselines and tree-based ensemble candidate models."""
    return {
        "Dummy (Majority Class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM (Weighted)": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            scale_pos_weight=pos_weight,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        ),
        "XGBoost (Weighted)": XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def cross_validate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, float]]:
    """Perform Stratified K-Fold Cross Validation on the training split."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = {}

    print(f"Starting {cv_folds}-Fold Stratified Cross Validation on Training Set ({len(y_train)} samples)...")
    for name, model in models.items():
        pr_aucs = []
        roc_aucs = []
        f1s = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            if hasattr(model, "predict_proba"):
                y_va_proba = model.predict_proba(X_va)[:, 1]
            else:
                y_va_proba = model.predict(X_va).astype(float)

            y_va_pred = (y_va_proba >= 0.5).astype(int)

            pr_aucs.append(average_precision_score(y_va, y_va_proba))
            roc_aucs.append(roc_auc_score(y_va, y_va_proba))
            f1s.append(f1_score(y_va, y_va_pred, zero_division=0))

        cv_results[name] = {
            "cv_pr_auc_mean": round(float(np.mean(pr_aucs)), 4),
            "cv_pr_auc_std": round(float(np.std(pr_aucs)), 4),
            "cv_roc_auc_mean": round(float(np.mean(roc_aucs)), 4),
            "cv_roc_auc_std": round(float(np.std(roc_aucs)), 4),
            "cv_f1_mean": round(float(np.mean(f1s)), 4),
            "cv_f1_std": round(float(np.std(f1s)), 4),
        }
        print(
            f"  {name:30s} | PR-AUC: {cv_results[name]['cv_pr_auc_mean']:.4f} "
            f"+/- {cv_results[name]['cv_pr_auc_std']:.4f} | "
            f"ROC-AUC: {cv_results[name]['cv_roc_auc_mean']:.4f}"
        )

    return cv_results


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train all models on the complete training set after cross-validation."""
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    pos_weight = float(neg_count / max(1, pos_count))

    if models is None:
        models = get_candidate_models(pos_weight=pos_weight)

    cv_results = cross_validate_models(models, X_train, y_train)

    # Final fit on full training set
    print("\nFitting models on full training set...")
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

    return fitted_models, cv_results


if __name__ == "__main__":
    print("Loading data and running training pipeline...")
    preprocessor, X_train, y_train, X_val, y_val, X_test, y_test, feat_names = prepare_data()
    fitted_models, cv_results = train_all_models(X_train, y_train)
    print("\nTraining completed successfully.")
