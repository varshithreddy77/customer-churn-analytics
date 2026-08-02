import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
from src.ingestion.load_data import load_raw_data

# Clean Data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Handle missing values, correct data types and remove duplicates.
    """
    df = df.copy()

    df['total_charges'] = df['total_charges'].fillna(0)
    n_before = len(df)
    df = df[~df.duplicated(keep="first")]
    n_dropped = n_before - len(df)
    print(f"  Dropped {n_dropped} duplicate rows")
 
    return df
    
# Feature Engineering

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Feature 1: Tenure_band
    def to_band(t):
        if t <= 6:   return "0-6_mo"
        if t <= 12:  return "7-12_mo"
        if t <= 24:  return "13-24_mo"
        if t <= 48:  return "25-48_mo"
        return "49+_mo"

    df['tenure_band'] = df['tenure'].apply(to_band)
    
    addon_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
    ]
    df["addon_count"] = (df[addon_cols] == "Yes").sum(axis=1)
 
    # ── Feature 3: is_new_customer (H2 — danger zone flag) ──
    df["is_new_customer"] = (df["tenure"] < 12).astype(int)
 
    # ── Feature 4: charge_per_tenure (value-for-money proxy) ──
    # Add 1 to tenure to avoid div-by-zero for new customers
    df["charge_per_tenure"] = df["total_charges"] / (df["tenure"] + 1)
 
    # ── Feature 5: has_internet (add-ons only apply to internet customers) ──
    df["has_internet"] = (df["internet_service"] != "No").astype(int)
 
    # ── Feature 6: contract_risk (ordinal — H1) ──
    # Higher number = higher risk
    contract_map = {"Two year": 0, "One year": 1, "Month-to-month": 2}
    df["contract_risk"] = df["contract"].map(contract_map)
 
    return df
 
 
# ═══════════════════════════════════════════════════════════
#  STEP 3 — Encode Yes/No Binary Columns
# ═══════════════════════════════════════════════════════════
 
def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes/No columns to 1/0.
 
    Note: "No internet service" and "No phone service" are treated as "No"
    since they represent the same thing (service not used).
    """
    df = df.copy()
 
    binary_cols = [
        "partner", "dependents", "phone_service", "paperless_billing",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
        "multiple_lines",
    ]
 
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)
 
    # Encode target
    df["churn"] = (df["churn"] == "Yes").astype(int)
 
    # Encode gender
    df["gender"] = (df["gender"] == "Male").astype(int)
 
    return df
 
 
# ═══════════════════════════════════════════════════════════
#  STEP 4 — Build Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════
 
def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
        - Scales numeric features (StandardScaler)
        - One-hot encodes nominal categorical features
        - Leaves binary 0/1 columns untouched (passthrough)
 
    Why ColumnTransformer:
        - Single object handles all preprocessing
        - Fit ONCE on train, transform on test (no leakage)
        - Can be saved with joblib and loaded in the API later
    """
    numeric_features = [
        "tenure", "monthly_charges", "total_charges",
        "addon_count", "charge_per_tenure", "contract_risk",
    ]
 
    nominal_features = [
        "internet_service", "payment_method", "tenure_band",
    ]
 
    # Binary columns pass through unchanged
    binary_features = [
        "gender", "senior_citizen", "partner", "dependents",
        "phone_service", "paperless_billing",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
        "multiple_lines", "is_new_customer", "has_internet",
    ]
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), nominal_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder="drop",  # Drop any columns not listed (like 'contract' since we have contract_risk)
    )
 
    return preprocessor
 
 
# ═══════════════════════════════════════════════════════════
#  STEP 5 — End-to-End Pipeline
# ═══════════════════════════════════════════════════════════
 
def build_features_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> tuple:
    """
    Complete feature engineering pipeline from raw data to model-ready arrays.
 
    Steps:
        1. Load raw data
        2. Clean (handle missing/duplicates)
        3. Engineer derived features
        4. Encode binary columns
        5. Stratified train/test split (preserves churn rate in both sets)
        6. Fit preprocessor on TRAIN ONLY → transform both
        7. Save artifacts for downstream phases
 
    Parameters
    ----------
    test_size : float
        Proportion held out for test (default 0.2 = 80/20 split)
    random_state : int
        For reproducibility
    save_artifacts : bool
        If True, save train/test CSVs and fitted preprocessor
 
    Returns
    -------
    X_train, X_test : pd.DataFrame (transformed)
    y_train, y_test : pd.Series
    preprocessor : ColumnTransformer (fitted)
    """
    print("=" * 60)
    print("  PHASE 4 — FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
 
    # ── 1. Load ──
    print("\n[1/6] Loading raw data...")
    df = load_raw_data()
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
 
    # ── 2. Clean ──
    print("\n[2/6] Cleaning data...")
    df = clean_data(df)
    print(f"  After cleaning: {df.shape[0]} rows")
 
    # ── 3. Engineer ──
    print("\n[3/6] Engineering features...")
    df = engineer_features(df)
    new_features = ["tenure_band", "addon_count", "is_new_customer",
                    "charge_per_tenure", "has_internet", "contract_risk"]
    print(f"  Added 6 features: {new_features}")
 
    # ── 4. Encode binaries ──
    print("\n[4/6] Encoding binary columns...")
    df = encode_binary_columns(df)
    print(f"  All Yes/No columns → 0/1")
 
    # ── 5. Split ──
    print("\n[5/6] Stratified train/test split...")
    y = df["churn"]
    X = df.drop(columns=["churn", "contract"])  # drop 'contract' (replaced by contract_risk)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Critical: preserves ~26.5% churn rate in both sets
    )
    print(f"  Train: {X_train.shape[0]} rows ({y_train.mean():.1%} churn)")
    print(f"  Test:  {X_test.shape[0]} rows ({y_test.mean():.1%} churn)")
 
    # ── 6. Transform ──
    print("\n[6/6] Fitting preprocessor on train, transforming both...")
    preprocessor = build_preprocessor()
 
    # FIT only on train — prevents leakage
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
 
    # Get readable column names for the transformed arrays
    feature_names = _get_feature_names(preprocessor)
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
 
    print(f"  Final feature count: {len(feature_names)}")
 
    # ── Save artifacts ──
    if save_artifacts:
        print("\nSaving artifacts...")
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("artifacts/models", exist_ok=True)
 
        X_train_df.to_csv("data/processed/X_train.csv")
        X_test_df.to_csv("data/processed/X_test.csv")
        y_train.to_csv("data/processed/y_train.csv")
        y_test.to_csv("data/processed/y_test.csv")
 
        joblib.dump(preprocessor, "artifacts/models/preprocessor.joblib")
        print("  Saved X_train, X_test, y_train, y_test to data/processed/")
        print("  Saved preprocessor to artifacts/models/preprocessor.joblib")
 
    print("\n" + "=" * 60)
    print("  PHASE 4 COMPLETE — Ready for Phase 5 (Modeling)")
    print("=" * 60)
 
    return X_train_df, X_test_df, y_train, y_test, preprocessor
 
 
# ═══════════════════════════════════════════════════════════
#  Helper — Extract clean feature names from ColumnTransformer
# ═══════════════════════════════════════════════════════════
 
def _get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.
 
    Without this, transformed columns are just 'x0', 'x1', etc.
    With this, we get names like 'tenure', 'payment_method_Mailed check', etc.
    """
    names = []
    for transformer_name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if transformer == "passthrough":
            names.extend(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(cols))
        else:
            names.extend(cols)
    return names
 
 
# ── Run end-to-end when executed directly ──
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = build_features_pipeline()
 
    print("\nFinal feature preview:")
    print(X_train.head(3).T)
    print(f"\nFeature names ({len(X_train.columns)} total):")
    for col in X_train.columns:
        print(f"  - {col}")
            
