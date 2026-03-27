"""
Run this after setup to verify everything is in place.
Usage: python verify_setup.py
"""
import sys
import os
 
 
def check(label, condition):
    status = "✅" if condition else "❌"
    print(f"  {status}  {label}")
    return condition
 
 
def main():
    print("=" * 50)
    print("  Setup Verification")
    print("=" * 50)
    print()
 
    all_good = True
 
    # 1. Check Python version
    v = sys.version_info
    all_good &= check(
        f"Python {v.major}.{v.minor}.{v.micro}",
        v.major == 3 and v.minor >= 9,
    )
 
    # 2. Check virtual environment is active
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    all_good &= check("Virtual environment active", in_venv)
 
    # 3. Check key packages
    packages = [
        "pandas", "numpy", "matplotlib", "seaborn",
        "sklearn", "xgboost", "shap", "mlflow",
        "fastapi", "streamlit", "duckdb", "pytest", "yaml",
    ]
    for pkg in packages:
        try:
            __import__(pkg)
            all_good &= check(f"Package: {pkg}", True)
        except ImportError:
            all_good &= check(f"Package: {pkg} — run: pip install -r requirements.txt", False)
 
    # 4. Check folder structure
    required_dirs = [
        "data/raw", "data/processed",
        "src/ingestion", "src/features", "src/training",
        "src/explainability", "src/serving", "src/registry",
        "src/sql", "src/utils",
        "api", "dashboard", "notebooks", "docs",
        "artifacts/models", "artifacts/reports", "artifacts/figures",
        "tests", "configs",
    ]
    for d in required_dirs:
        all_good &= check(f"Directory: {d}/", os.path.isdir(d))
 
    # 5. Check dataset
    dataset_path = "data/raw/Telco-Customer-Churn.csv"
    all_good &= check(
        f"Dataset: {dataset_path}",
        os.path.isfile(dataset_path),
    )
 
    # 6. Check config
    all_good &= check(
        "Config: configs/paths.yaml",
        os.path.isfile("configs/paths.yaml"),
    )
 
    # 7. Check business framing
    all_good &= check(
        "Docs: docs/business_framing.md",
        os.path.isfile("docs/business_framing.md"),
    )
 
    print()
    if all_good:
        print("  🎉  All checks passed! Ready for Phase 1.")
    else:
        print("  ⚠️   Some checks failed. Fix the ❌ items above.")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()