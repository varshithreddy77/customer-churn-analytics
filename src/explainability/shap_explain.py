import os
import json
 
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

DATA_DIR = "data/processed"
MODEL_DIR = "artifacts/models"
REPORT_DIR = "artifacts/reports"
FIGURE_DIR = "artifacts/figures"

FLAG_PHRASES = {
    "senior_citizen":       ("senior citizen", "not a senior citizen"),
    "partner":              ("has a partner", "no partner"),
    "dependents":           ("has dependents", "no dependents"),
    "phone_service":        ("has phone service", "no phone service"),
    "paperless_billing":    ("on paperless billing", "on paper billing"),
    "online_security":      ("has online security", "no online security"),
    "online_backup":        ("has online backup", "no online backup"),
    "device_protection":    ("has device protection", "no device protection"),
    "tech_support":         ("has tech support", "no tech support"),
    "streaming_tv":         ("streams TV", "no TV streaming"),
    "streaming_movies":     ("streams movies", "no movie streaming"),
    "multiple_lines":       ("has multiple lines", "single line"),
    "is_new_customer":      ("new customer (under 12 months)", "established customer"),
    "has_internet":         ("has internet service", "no internet service"),
    "gender":               ("male", "female"),
 
    "internet_service_Fiber optic":           ("fiber optic internet", "not on fiber optic"),
    "internet_service_No":                    ("no internet service", "has internet service"),
    "payment_method_Electronic check":        ("pays by electronic check", "not on electronic check"),
    "payment_method_Mailed check":            ("pays by mailed check", "not on mailed check"),
    "payment_method_Credit card (automatic)": ("pays by credit card (auto)", "not on credit card auto-pay"),
 
    "tenure_band_7-12_mo":  ("tenure 7-12 months", "outside 7-12 month band"),
    "tenure_band_13-24_mo": ("tenure 13-24 months", "outside 13-24 month band"),
    "tenure_band_25-48_mo": ("tenure 25-48 months", "outside 25-48 month band"),
    "tenure_band_49+_mo":   ("tenure over 48 months", "under 49 months tenure"),
}


NUMERIC_PHRASES = {
    "tenure":            ("long tenure ({v})", "short tenure ({v})",
                          "long tenure", "short tenure"),
    "monthly_charges":   ("high monthly charges (${v:.0f})", "low monthly charges (${v:.0f})",
                          "high monthly charges", "low monthly charges"),
    "total_charges":     ("high lifetime spend (${v:,.0f})", "low lifetime spend (${v:,.0f})",
                          "high lifetime spend", "low lifetime spend"),
    "addon_count":       ("{v:.0f} add-on services", "only {v:.0f} add-on services",
                          "several add-on services", "few add-on services"),
    "charge_per_tenure": ("high charges relative to tenure", "low charges relative to tenure",
                          "high charges relative to tenure", "low charges relative to tenure"),
}


def _months(value: float) -> str:
    n = int(round(value))
    return f"{n} month" if n == 1 else f"{n} months"
 
CONTRACT_NAMES = {0: "two-year contract", 1: "one-year contract", 2: "month-to-month contract"}


def unscale_numerics(X: pd.DataFrame) -> pd.DataFrame:

    try:
        pre = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")
        scaler = pre.named_transformers_["num"]
        cols = list(pre.transformers_[0][2])          # the numeric column list
 
        raw = X[cols].values * scaler.scale_ + scaler.mean_
        return pd.DataFrame(raw, columns=cols, index=X.index)
    except Exception as e:                            # noqa: BLE001
        print(f"  (could not unscale numerics: {e})")
        return pd.DataFrame(index=X.index)
 
 
def describe(feature: str, scaled_value: float, raw_value=None) -> str:
    """Turn one feature + its value into a phrase a human can act on."""
    if feature == "contract_risk":
        if raw_value is not None and not np.isnan(raw_value):
            return CONTRACT_NAMES.get(int(round(raw_value)), "contract type")
        return "month-to-month contract" if scaled_value > 0 else "longer-term contract"
 
    if feature in NUMERIC_PHRASES:
        high_t, low_t, high_plain, low_plain = NUMERIC_PHRASES[feature]
        above = scaled_value >= 0
 
        if raw_value is None or np.isnan(raw_value):
            return high_plain if above else low_plain
 
        template = high_t if above else low_t
        value = _months(raw_value) if feature == "tenure" else raw_value
        return template.format(v=value)
 
    if feature in FLAG_PHRASES:
        present, absent = FLAG_PHRASES[feature]
        return present if scaled_value >= 0.5 else absent
 
    return feature.replace("_", " ")
 
 

 
def _read_target(path: str) -> pd.Series:
    y = pd.read_csv(path, index_col=0)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return y
 
 
def load_everything():

    with open(f"{REPORT_DIR}/best_model.json") as f:
        best = json.load(f)
    model_name = best["best_by_business_cost"]
    slug = model_name.lower().replace(" ", "_")
    model = joblib.load(f"{MODEL_DIR}/{slug}.joblib")
 
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv", index_col=0)
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv", index_col=0)
    y_test = _read_target(f"{DATA_DIR}/y_test.csv")
 
    threshold = 0.50
    path = f"{REPORT_DIR}/tuned_threshold.json"
    if os.path.exists(path):
        with open(path) as f:
            threshold = json.load(f)["tuned_threshold"]
 
    return model, model_name, X_train, X_test, y_test, threshold
 
 

 
def compute_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame):

    if hasattr(model, "coef_"):
        print("  Using LinearExplainer (exact for linear models)")
        # SHAP needs a "background" sample to define what an average customer
        # looks like. 200 rows is plenty and keeps the run fast; passing the
        # full training set only makes shap subsample it anyway.
        background = X_train.sample(n=min(200, len(X_train)), random_state=42)
        explainer = shap.LinearExplainer(model, background)
        values = explainer.shap_values(X_test)
        base = float(np.ravel(explainer.expected_value)[0])
    else:
        print("  Using TreeExplainer")
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_test)
        base = explainer.expected_value
 
        if isinstance(values, list):          # [class_0, class_1]
            values = values[1]
            base = np.ravel(base)[-1]
        elif np.ndim(values) == 3:            # (n, p, n_classes)
            values = values[:, :, 1]
            base = np.ravel(base)[-1]
        base = float(np.ravel(base)[0])
 
    return np.asarray(values), base
 
 

 
def global_importance(shap_values: np.ndarray, X_test: pd.DataFrame) -> pd.DataFrame:

    return (
        pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_signed_shap": shap_values.mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
 
 

 
def reasons_for_customer(row_idx: int, shap_values, X_test, raw_numerics,
                         top_n: int = 3) -> str:

    contributions = shap_values[row_idx]
    features = list(X_test.columns)
 
    order = np.argsort(contributions)[::-1]           # biggest push first
    picked = []
 
    for j in order:
        if contributions[j] <= 0 or len(picked) >= top_n:
            break
        name = features[j]
        raw = raw_numerics.iloc[row_idx].get(name, np.nan) \
            if name in raw_numerics.columns else np.nan
        picked.append(describe(name, X_test.iloc[row_idx, j], raw))
 
    return ", ".join(picked) if picked else "no strong risk drivers"
 
 
def build_call_list(model, X_test, y_test, shap_values, raw_numerics,
                    threshold: float, top_k: int = 100) -> pd.DataFrame:

    proba = model.predict_proba(X_test)[:, 1]
    order = np.argsort(proba)[::-1][:top_k]
 
    rows = []
    for rank, i in enumerate(order, start=1):
        rows.append({
            "rank": rank,
            "customer_id": X_test.index[i],
            "churn_probability": round(float(proba[i]), 4),
            "flagged": bool(proba[i] >= threshold),
            "actually_churned": int(y_test.iloc[i]),
            "why": reasons_for_customer(i, shap_values, X_test, raw_numerics),
        })
 
    return pd.DataFrame(rows)
 
 

 
def plot_global_bar(importance: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart of the strongest drivers, with readable labels."""
    top = importance.head(top_n).iloc[::-1]
    labels = [describe(f, 1.0).capitalize() for f in top["feature"]]
    colors = ["#c0392b" if s > 0 else "#27ae60" for s in top["mean_signed_shap"]]
 
    plt.figure(figsize=(9, 7))
    plt.barh(labels, top["mean_abs_shap"], color=colors, edgecolor="black", linewidth=0.5)
    plt.xlabel("Mean |SHAP value| — average impact on prediction")
    plt.title("Which features drive churn predictions")
    plt.figtext(0.99, 0.01, "red = pushes toward churn   green = pushes away",
                ha="right", fontsize=9, style="italic")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/13_shap_global_importance.png", dpi=100)
    plt.close()
 
 
def plot_beeswarm(shap_values, X_test) -> None:
    """
    One dot per customer per feature. Position shows the contribution, colour
    shows whether that customer's feature value was high or low. Denser than
    the bar chart — it shows HOW a feature acts, not just how much.
    """
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title("SHAP summary — impact and direction per customer")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/14_shap_beeswarm.png", dpi=100, bbox_inches="tight")
    plt.close()
 
 
def plot_waterfall(row_idx, shap_values, X_test, raw_numerics, base_value,
                   proba, filename, title) -> None:

    contributions = shap_values[row_idx]
    features = list(X_test.columns)
 
    order = np.argsort(np.abs(contributions))[::-1][:8]
    labels, values = [], []
    for j in order:
        name = features[j]
        raw = raw_numerics.iloc[row_idx].get(name, np.nan) \
            if name in raw_numerics.columns else np.nan
        labels.append(describe(name, X_test.iloc[row_idx, j], raw))
        values.append(contributions[j])
 
    labels, values = labels[::-1], values[::-1]
    colors = ["#c0392b" if v > 0 else "#27ae60" for v in values]
 
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Contribution to the prediction (log-odds)")
    plt.title(f"{title}\nchurn probability {proba:.2f}   "
              f"(average customer {1 / (1 + np.exp(-base_value)):.2f})")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/{filename}", dpi=100)
    plt.close()
 

 
def run(save: bool = True) -> dict:
    print("=" * 70)
    print("  PHASE 7 — EXPLAINABILITY (SHAP)")
    print("=" * 70)
 
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
 
    print("\n[1/5] Loading model and data...")
    model, model_name, X_train, X_test, y_test, threshold = load_everything()
    print(f"  Model: {model_name}   tuned threshold: {threshold}")
 
    print("\n[2/5] Computing SHAP values...")
    shap_values, base_value = compute_shap(model, X_train, X_test)
    print(f"  Computed {shap_values.shape[0]} x {shap_values.shape[1]} values")
 
    raw_numerics = unscale_numerics(X_test)
 
   
    log_odds = model.decision_function(X_test) if hasattr(model, "decision_function") else None
    if log_odds is not None:
        reconstructed = base_value + shap_values.sum(axis=1)
        max_err = float(np.max(np.abs(reconstructed - log_odds)))
        print(f"  Additivity check: max error {max_err:.2e} "
              f"{'-- values sum to the prediction' if max_err < 1e-6 else '-- CHECK'}")
 
    print("\n[3/5] Global feature importance")
    importance = global_importance(shap_values, X_test)
    top = importance.head(10).copy()
    top["reads as"] = [describe(f, 1.0) for f in top["feature"]]
    print(top[["feature", "mean_abs_shap", "mean_signed_shap", "reads as"]]
          .to_string(index=False, float_format="%.4f"))
    print("\n  Compare this ranking against the Phase 3 EDA findings.")
    print("  Contract type, tenure and add-ons should be near the top.")
 
    print("\n[4/5] Building the call list with reasons")
    call_list = build_call_list(model, X_test, y_test, shap_values,
                                raw_numerics, threshold)
    print(call_list.head(8).to_string(index=False))
 
    hit_rate = call_list["actually_churned"].mean()
    print(f"\n  Of the top {len(call_list)}, {call_list['actually_churned'].sum()} "
          f"really churned ({hit_rate:.1%})")
 
    if save:
        print("\n[5/5] Saving figures and reports...")
        proba = model.predict_proba(X_test)[:, 1]
        highest = int(np.argmax(proba))
        lowest = int(np.argmin(proba))
 
        plot_global_bar(importance)
        plot_beeswarm(shap_values, X_test)
        plot_waterfall(highest, shap_values, X_test, raw_numerics, base_value,
                       proba[highest], "15_shap_waterfall_high_risk.png",
                       f"Highest-risk customer ({X_test.index[highest]})")
        plot_waterfall(lowest, shap_values, X_test, raw_numerics, base_value,
                       proba[lowest], "16_shap_waterfall_low_risk.png",
                       f"Lowest-risk customer ({X_test.index[lowest]})")
 
        importance.to_csv(f"{REPORT_DIR}/shap_global_importance.csv", index=False)
        call_list.to_csv(f"{REPORT_DIR}/call_list_with_reasons.csv", index=False)
 
        print(f"  Saved 4 figures to {FIGURE_DIR}/")
        print(f"  Saved shap_global_importance.csv, call_list_with_reasons.csv")
 
    print("\n" + "=" * 70)
    print("  PHASE 7 COMPLETE — Ready for Phase 8 (MLflow tracking)")
    print("=" * 70)
 
    return {"importance": importance, "call_list": call_list,
            "shap_values": shap_values}
 
 
if __name__ == "__main__":
    run()