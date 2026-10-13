import pandas as pd
from typing import Dict, Any

EXPECTED_SCHEMA = {
    "gender": "object",
    "senior_citizen": "int64",
    "partner": "object",
    "dependents": "object",
    "tenure": "int64",
    "phone_service": "object",
    "multiple_lines": "object",
    "internet_service": "object",
    "online_security": "object",
    "online_backup": "object",
    "device_protection": "object",
    "tech_support": "object",
    "streaming_tv": "object",
    "streaming_movies": "object",
    "contract": "object",
    "paperless_billing": "object",
    "payment_method": "object",
    "monthly_charges": "float64",
    "total_charges": "float64",
    "churn": "object",
}


EXPECTED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "partner": {"Yes", "No"},
    "dependents": {"Yes", "No"},
    "churn": {"Yes", "No"},
    "contract": {"Month-to-month", "One year", "Two year"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "payment_method": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

def validate_schema(df: pd.DataFrame) ->  Dict[str, Any]:
    
    result = {"check": "schema", "passed": True, "issues": []}
    
    # Check missing columns
    missing = [col for col in EXPECTED_SCHEMA if col not in df.columns]

    if missing:
        result['passed'] = False
        result['issues'].append(f"Missing columns: {missing}")

    # Check data types (newer pandas reports text columns as "str" instead of "object")
    EQUIVALENT = {"object": {"object", "str"}, "str": {"object", "str"}}

    type_mistakes = {}
    for col, expected_type in EXPECTED_SCHEMA.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            if actual not in EQUIVALENT.get(expected_type, {expected_type}):
                type_mistakes[col] = {"expected": expected_type, "actual": actual}

    if type_mistakes:
        result['passed'] = False
        result['issues'].append(f"Type mismatches: {type_mistakes}")


    # Check for unexpected extra columns
    extra = [col for col in df.columns if col not in EXPECTED_SCHEMA]
    if extra:
        result['passed'] = False
        result['issues'].append(f"Extra columns: {extra}")

    return result


def validate_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    result = {"check": "missing_values", "passed": True, "issues": []}

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        result['passed'] = False
        total = len(df)
        details = {
            col: {"count": int(count), "pct": round(count/total * 100,2)}
            for col, count in missing.items()
        }
        result['issues'].append(f"Missing values found: {details}")
        result['missing_details'] = details
    
    return result


def validate_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    result = {"check": "duplicates", "passed": True, "issues": []}

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        result['passed'] = False
        result['issues'].append(f"Found {dup_rows} duplicate rows")
        
    if df.index.name is not None:
        dup_index = df.index.duplicated().sum()
        if dup_index > 0:
            result['passed'] = False
            result['issues'].append(f"Found {dup_index} duplicate index values")

    return result

def validate_value_ranges(df: pd.DataFrame) -> Dict[str, Any]:

    result = {"check": "value_ranges", "passed": True, "issues": []}
 
    range_checks = {
        "tenure": (0, 72),
        "monthly_charges": (0, 200),
        "total_charges": (0, 10000),
    }
 
    for col, (low, high) in range_checks.items():
        if col in df.columns:
            below = (df[col].dropna() < low).sum()
            above = (df[col].dropna() > high).sum()
            if below > 0:
                result["passed"] = False
                result["issues"].append(f"{col}: {below} values below {low}")
            if above > 0:
                result["passed"] = False
                result["issues"].append(f"{col}: {above} values above {high}")
 
    if "senior_citizen" in df.columns:
        invalid = ~df["senior_citizen"].isin([0, 1])
        if invalid.sum() > 0:
            result["passed"] = False
            result["issues"].append(
                f"senior_citizen: {invalid.sum()} values not in [0, 1]"
            )
 
    return result
 
 
def validate_categories(df: pd.DataFrame) -> Dict[str, Any]:

    result = {"check": "categories", "passed": True, "issues": []}
 
    for col, expected_values in EXPECTED_CATEGORIES.items():
        if col in df.columns:
            actual_values = set(df[col].dropna().unique())
            unexpected = actual_values - expected_values
            if unexpected:
                result["passed"] = False
                result["issues"].append(
                    f"{col}: unexpected values {unexpected}"
                )
 
    return result
 
 
def validate_target(df: pd.DataFrame) -> Dict[str, Any]:

    result = {"check": "target_integrity", "passed": True, "issues": []}
 
    if "churn" not in df.columns:
        result["passed"] = False
        result["issues"].append("Target column 'churn' not found")
        return result
 
    # Check for nulls in target
    n_null = df["churn"].isnull().sum()
    if n_null > 0:
        result["passed"] = False
        result["issues"].append(f"Target has {n_null} null values")
 
    # Check class distribution
    churn_rate = (df["churn"] == "Yes").mean()
    result["churn_rate"] = round(churn_rate * 100, 2)
    result["class_counts"] = df["churn"].value_counts().to_dict()
 
    if churn_rate < 0.10 or churn_rate > 0.90:
        result["issues"].append(
            f"Extreme class imbalance: churn rate = {churn_rate:.1%}"
        )
 
    return result
 
 
def validate(df: pd.DataFrame) -> Dict[str, Any]:

    checks = [
        validate_schema,
        validate_missing_values,
        validate_duplicates,
        validate_value_ranges,
        validate_categories,
        validate_target,
    ]
 
    report = {
        "dataset_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "checks": [],
        "overall_passed": True,
    }
 
    for check_fn in checks:
        result = check_fn(df)
        report["checks"].append(result)
        if not result["passed"]:
            report["overall_passed"] = False
 
    return report
 
 
def print_report(report: Dict[str, Any]) -> None:

    print("=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Dataset: {report['dataset_shape']['rows']} rows × "
          f"{report['dataset_shape']['columns']} columns")
    print()
 
    for check in report["checks"]:
        status = " PASS" if check["passed"] else " FAIL"
        print(f"  {status}  {check['check']}")
        for issue in check.get("issues", []):
            print(f"           → {issue}")
 
        # Print extra info for target check
        if check["check"] == "target_integrity" and "churn_rate" in check:
            print(f"           Churn rate: {check['churn_rate']}%")
            print(f"           Class counts: {check['class_counts']}")
 
    print()
    overall = "ALL CHECKS PASSED" if report["overall_passed"] else " SOME CHECKS FAILED"
    print(f"  Overall: {overall}")
    print("=" * 60)
 
 
# ── Run directly to see the report ──
if __name__ == "__main__":
    from src.ingestion.load_data import load_raw_data
    from src.features.build_features import clean_data

    df = clean_data(load_raw_data())
    report = validate(df)
    print_report(report)


    