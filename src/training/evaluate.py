import os
import json

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

DATA_DIR = "data/processed"
REPORT_DIR = "artifacts/reports"
MODEL_DIR = "artifacts/models"
FIGURE_DIR = "artifacts/figures"

CLV = 1560
OFFER_COST = 50

SUCCESS_RATE = 0.30

def _read_target(path: str) -> pd.Series:

    Y = pd.read_csv(path, index_col=0)
    if isinstance(Y, pd.DataFrame):
        Y = Y.iloc[:, 0]
    return Y

def load_model_and_data(prefer : str = "business_cost"):

    with open(f"{REPORT_DIR}/best_model.json", "r") as f:
        best = json.load(f)

    key = "best_by_business_cost" if prefer == "business_cost" else "best_by_roc_auc"
    model_name = best[key]
    slug = model_name.lower().replace(" ", "_")

    model = joblib.load(f"{MODEL_DIR}/{slug}.joblib")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv", index_col=0)
    y_test = _read_target(f"{DATA_DIR}/y_test.csv")

    return model, X_test, y_test, model_name


def naive_cost(tp: int, fp: int, fn: int) -> float:

    return (fp * OFFER_COST) + (fn * CLV)


def net_benefit(tp: int, fp: int, success_rate: float = SUCCESS_RATE) -> float:

    saved = tp * success_rate * CLV
    spent = (tp + fp) * OFFER_COST
    return saved - spent



# The Threshold sweep

def sweep_thresholds(y_true, y_proba, success_rate: float = SUCCESS_RATE) -> pd.DataFrame:

    thresholds = np.arange(0.01, 1.00, 0.01)
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        flagged = tp + fp
        results.append({
            "threshold": round(float(threshold), 2),
            "flagged": int(flagged),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "precision": tp / flagged if flagged else 0.0,
            "naive_cost": naive_cost(tp, fp, fn),
            "net_benefit": net_benefit(tp, fp, success_rate),
        })

    return pd.DataFrame(results)

def analytic_threshold(success_rate: float = SUCCESS_RATE) -> float:
    return OFFER_COST / (success_rate * CLV)
 
 

 
def evaluate_top_k(y_true, y_proba, k_values=(50, 100, 200, 300, 500)) -> pd.DataFrame:
    order = np.argsort(y_proba)[::-1]          # highest probability first
    y_sorted = np.asarray(y_true)[order]
 
    total_churners = int(np.sum(y_true))
    base_rate = total_churners / len(y_true)
 
    rows = []
    for k in k_values:
        if k > len(y_true):
            continue
 
        caught = int(y_sorted[:k].sum())
        precision_k = caught / k
        rows.append({
            "k": k,
            "churners_found": caught,
            "precision_at_k": precision_k,
            "recall_at_k": caught / total_churners,
            "lift": precision_k / base_rate,
            "net_benefit": net_benefit(tp=caught, fp=k - caught),
        })
 
    return pd.DataFrame(rows)
 
 


 
def sensitivity_table(y_true, y_proba, rates=(0.10, 0.20, 0.30, 0.50, 1.00)) -> pd.DataFrame:

    rows = []
    for rate in rates:
        sweep = sweep_thresholds(y_true, y_proba, success_rate=rate)
        best = sweep.loc[sweep["net_benefit"].idxmax()]
 
        rows.append({
            "success_rate": rate,
            "analytic_threshold": round(analytic_threshold(rate), 3),
            "swept_threshold": best["threshold"],
            "flagged": int(best["flagged"]),
            "recall": round(best["recall"], 3),
            "precision": round(best["precision"], 3),
            "net_benefit": round(best["net_benefit"]),
        })
 
    return pd.DataFrame(rows)
 

 
def save_figures(y_true, y_proba, sweep: pd.DataFrame, tuned_threshold: float) -> None:

    os.makedirs(FIGURE_DIR, exist_ok=True)
 
    # ── ROC ──
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
 
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guessing")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall)")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/10_roc_curve.png", dpi=100)
    plt.close()
 
    # ── Precision-Recall ──
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    base_rate = float(np.mean(y_true))
 
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, linewidth=2, label=f"Model (PR-AUC = {ap:.3f})")
    plt.axhline(base_rate, linestyle="--", color="gray",
                label=f"Base rate ({base_rate:.1%})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/11_precision_recall_curve.png", dpi=100)
    plt.close()
 
    # ── Cost / benefit vs threshold ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    axes[0].plot(sweep["threshold"], sweep["naive_cost"], linewidth=2, color="#c0392b")
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Total cost ($)")
    axes[0].set_title("Original cost model — no usable minimum")
    axes[0].grid(alpha=0.3)
 
    axes[1].plot(sweep["threshold"], sweep["net_benefit"], linewidth=2, color="#27ae60")
    axes[1].axvline(tuned_threshold, linestyle="--", color="black",
                    label=f"Optimum = {tuned_threshold:.2f}")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Net benefit ($)")
    axes[1].set_title(f"With {SUCCESS_RATE:.0%} offer success rate")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/12_threshold_tuning.png", dpi=100)
    plt.close()
 
    print(f"  Saved 3 figures to {FIGURE_DIR}/")
 
 

 
def run_evaluation(save: bool = True) -> dict:
 
    # ── Load ──
    print("\nLoading Phase 5 model and test set...")
    model, X_test, y_test, model_name = load_model_and_data()
    y_proba = model.predict_proba(X_test)[:, 1]
 
    n = len(y_test)
    churners = int(y_test.sum())
    print(f"  Model: {model_name}")
    print(f"  Test set: {n} customers, {churners} churners "
          f"({churners / n:.1%})")
 
    # ── The default ──
    print("\n What the default 0.50 cutoff gives you")
    tn, fp, fn, tp = confusion_matrix(
        y_test, (y_proba >= 0.50).astype(int), labels=[0, 1]
    ).ravel()
    print(f"  Flagged {tp + fp} customers | caught {tp} | missed {fn}")
    print(f"  Recall {tp / (tp + fn):.3f} | precision {tp / (tp + fp):.3f}")
    print(f"  Naive cost  : ${naive_cost(tp, fp, fn):,.0f}")
    print(f"  Net benefit : ${net_benefit(tp, fp):,.0f}")
 
    # ── The degenerate sweep ──
    print("\n Sweeping thresholds under the ORIGINAL cost model")
    sweep = sweep_thresholds(y_test, y_proba)
    cheapest = sweep.loc[sweep["naive_cost"].idxmin()]
 
    print(f"  Cheapest cutoff: {cheapest['threshold']:.2f} "
          f"-> flags {int(cheapest['flagged'])} of {n} customers "
          f"({cheapest['flagged'] / n:.0%})")
    print(f"  Cost there: ${cheapest['naive_cost']:,.0f}")
    print(f"  Compare — flag EVERYONE, no model: "
          f"${(n - churners) * OFFER_COST:,.0f}")
    print("  The optimiser says 'contact nearly every customer'.")
    print("  That answer needs no model. The cost assumption is the problem:")
    print("  it treats a $50 offer as a guaranteed save.")
 
    # ── The fix ──
    print(f"\n Re-sweeping with a {SUCCESS_RATE:.0%} offer success rate")
    best = sweep.loc[sweep["net_benefit"].idxmax()]
    tuned = float(best["threshold"])
    derived = analytic_threshold()
 
    print(f"  Swept optimum    : {tuned:.2f}   <- use this one")
    print(f"  Analytic formula : {derived:.3f}   "
          f"(= {OFFER_COST} / ({SUCCESS_RATE} x {CLV}))")
    gap = tuned - derived
    if abs(gap) <= 0.02:
        print(f"  Gap              : {gap:+.3f} -- the model is well calibrated")
    elif gap > 0:
        print(f"  Gap              : {gap:+.3f} -- expected: class_weight="
              f"'balanced' inflates")
        print("                     predicted probabilities, so the model is "
              "not calibrated.")
        print("                     Trust the swept value; it uses observed "
              "outcomes.")
    else:
        print(f"  Gap              : {gap:+.3f} -- unusual direction; "
              "re-check the cost function")
    print(f"\n  At threshold {tuned:.2f}:")
    print(f"    flagged     {int(best['flagged'])} customers")
    print(f"    recall      {best['recall']:.3f}  "
          f"(was {tp / (tp + fn):.3f} at 0.50)")
    print(f"    precision   {best['precision']:.3f}")
    print(f"    net benefit ${best['net_benefit']:,.0f}  "
          f"(was ${net_benefit(tp, fp):,.0f} at 0.50)")
 
    # ── Sensitivity ──
    print("\n How much does the success-rate assumption matter?")
    sens = sensitivity_table(y_test, y_proba)
    print(sens.to_string(index=False))
    print("  Note the last row: at 100% success the threshold collapses to")
    print("  ~0.03, which is exactly the broken behaviour from step 3.")
 
    # ── Ranking ──
    print("\n The model as a ranked call list")
    topk = evaluate_top_k(y_test, y_proba)
    print(topk.to_string(index=False, float_format="%.3f"))
    if len(topk):
        r = topk.iloc[1] if len(topk) > 1 else topk.iloc[0]
        print(f"\n  Reading row k={int(r['k'])}: calling the top {int(r['k'])} "
              f"customers finds {int(r['churners_found'])} churners,")
        print(f"  which is {r['lift']:.1f}x better than calling "
              f"{int(r['k'])} customers at random.")
 
    # ── Save ──
    if save:
        print("\nSaving artifacts...")
        os.makedirs(REPORT_DIR, exist_ok=True)
 
        sweep.to_csv(f"{REPORT_DIR}/threshold_sweep.csv", index=False)
        sens.to_csv(f"{REPORT_DIR}/success_rate_sensitivity.csv", index=False)
        topk.to_csv(f"{REPORT_DIR}/top_k_evaluation.csv", index=False)
 
        with open(f"{REPORT_DIR}/tuned_threshold.json", "w") as f:
            json.dump({
                "model": model_name,
                "tuned_threshold": tuned,
                "analytic_threshold": round(derived, 4),
                "assumed_success_rate": SUCCESS_RATE,
                "clv": CLV,
                "offer_cost": OFFER_COST,
                "recall_at_tuned": round(float(best["recall"]), 4),
                "precision_at_tuned": round(float(best["precision"]), 4),
                "net_benefit_at_tuned": round(float(best["net_benefit"])),
            }, f, indent=2)
 
        print(f"  Saved threshold_sweep.csv, success_rate_sensitivity.csv,")
        print(f"        top_k_evaluation.csv, tuned_threshold.json")
        save_figures(y_test, y_proba, sweep, tuned)
 
    print("\n" + "=" * 70)
    print("  PHASE 6 COMPLETE — Ready for Phase 7 (SHAP explainability)")
    print("=" * 70)
 
    return {"sweep": sweep, "sensitivity": sens, "top_k": topk,
            "tuned_threshold": tuned}
 
 
if __name__ == "__main__":
    run_evaluation()