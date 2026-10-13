import os
import json
import time
import warnings

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier    
    XGBOOST_AVAILABLE = True
    XGBOOST_ERROR = ""
except Exception as e:
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(e)

DATA_DIR = "data/processed"
MODEL_DIR = "artifacts/models"
REPORT_DIR = "artifacts/reports"

RANDOM_STATE = 42

CLV = 1560
RETENTION_COST = 50

def load_processed_data():
    """
    Load train/test splits
    """

    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv", index_col=0)
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv", index_col=0)
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv", index_col=0)["churn"]
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv", index_col=0)["churn"]

    return X_train, X_test, y_train, y_test 

def get_models(y_train: pd.Series) -> dict:

    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter = 1000,
            class_weight = "balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight = "balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return models


def cross_validate_model(model, X_train,y_train, n_splits: int = 5) -> dict:
    """
    5-fold stratified cross-validation on TRAIN ONLY.
    """

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    metrics = {
        "cv_roc_auc": results["test_roc_auc"].mean(),
        "cv_roc_auc_std": results["test_roc_auc"].std(),
        "cv_recall": results["test_recall"].mean(),
        "cv_precision": results["test_precision"].mean(),
        "cv_f1": results["test_f1"].mean(),
    }

    return metrics


def evaluate_model(model, X_test, y_test) -> dict:

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

def business_cost(metrics: dict) -> dict:

    fn_cost = metrics["false_negatives"] * CLV
    fp_cost = metrics["false_positives"] * RETENTION_COST

    return {
        "cost_missed_churners": fn_cost,
        "cost_wasted_offers": fp_cost,
        "total_cost": fn_cost + fp_cost,
    }


def train_all_models(save: bool = True) -> pd.DataFrame:

    X_train, X_test, y_train, y_test = load_processed_data()

    print(f" Train: {X_train.shape[0]} rows x {X_train.shape[1]} features"
          f"({y_train.mean():.1%} churn)")
    print(f" Test: {X_test.shape[0]} rows x {X_test.shape[1]} features"
          f"({y_test.mean():.1%} churn)")

    # Baseline sanity check
    
    majority_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"\n  Majority-class baseline accuracy: {majority_acc:.1%}")
    print("  (any model must beat this to be worth anything)")

    # Models
    models = get_models(y_train)
    print(f"\n Training {len(models)} models...")
    if not XGBOOST_AVAILABLE:
        print(f"  (XGBoost not available: {XGBOOST_ERROR})")

    rows = []
    fitted = {}

    for name, model in models.items():
        print(f"\n -- {name} --")
        t0 = time.time()

        cv_scores = cross_validate_model(model, X_train, y_train)
        print(f"     CV ROC-AUC : {cv_scores['cv_roc_auc']:.4f} "
              f"(+/- {cv_scores['cv_roc_auc_std']:.4f})")
        print(f"     CV Recall  : {cv_scores['cv_recall']:.4f}")

        # Fit model on full training set
        model.fit(X_train, y_train)
        fitted[name] = model

        # Evaluate on test set
        test_scores = evaluate_model(model, X_test, y_test)
        costs = business_cost(test_scores)

        elapsed = time.time() - t0
        print(f"     Test ROC-AUC: {test_scores['roc_auc']:.4f}")
        print(f"     Test Recall : {test_scores['recall']:.4f}  "
              f"-> catches {test_scores['recall']:.0%} of churners")
        print(f"     Missed churners (FN): {test_scores['false_negatives']}")
        print(f"     Business cost: ${costs['total_cost']:,}")
        print(f"     Trained in {elapsed:.1f}s")

        rows.append({
            "Model": name,
            "CV ROC-AUC": cv_scores["cv_roc_auc"],
            "Test ROC-AUC": test_scores["roc_auc"],
            "Test PR-AUC": test_scores["pr_auc"],
            "Accuracy": test_scores["accuracy"],
            "Precision": test_scores["precision"],
            "Recall": test_scores["recall"],
            "F1": test_scores["f1"],
            "FN (missed churners)": test_scores["false_negatives"],
            "FP (wasted offers)": test_scores["false_positives"],
            "Business Cost ($)": costs["total_cost"],
        })

    comparison = pd.DataFrame(rows).sort_values("Test ROC-AUC", ascending=False)

    print("\n" + "=" * 60)
    display_cols = ["Model", "CV ROC-AUC", "Test ROC-AUC", "Recall",
                    "Precision", "F1", "Business Cost ($)"]
    print(comparison[display_cols].to_string(index=False, float_format="%.4f"))

    best_by_auc = comparison.iloc[0]["Model"]
    best_by_cost = comparison.sort_values("Business Cost ($)").iloc[0]["Model"]

    print(f"\n  Best by ROC-AUC      : {best_by_auc}")
    print(f"  Best by business cost: {best_by_cost}")
    if best_by_auc != best_by_cost:
        print("  NOTE: These disagree - worth discussing in your README.")
        print("        The cost-optimal model is the one the business should ship.")

    # ── Save ──
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(REPORT_DIR, exist_ok=True)

        for name, model in fitted.items():
            slug = name.lower().replace(" ", "_")
            joblib.dump(model, f"{MODEL_DIR}/{slug}.joblib")
            print(f"  Saved {MODEL_DIR}/{slug}.joblib")

        comparison.to_csv(f"{REPORT_DIR}/model_comparison.csv", index=False)
        print(f"  Saved {REPORT_DIR}/model_comparison.csv")

        # Record which model wins, for Phase 6/7 to pick up
        best_slug = best_by_auc.lower().replace(" ", "_")
        with open(f"{REPORT_DIR}/best_model.json", "w") as f:
            json.dump({
                "best_by_roc_auc": best_by_auc,
                "best_by_business_cost": best_by_cost,
                "model_file": f"{best_slug}.joblib",
            }, f, indent=2)
        print(f"  Saved {REPORT_DIR}/best_model.json")

 
    return comparison
 
 
if __name__ == "__main__":
    train_all_models()