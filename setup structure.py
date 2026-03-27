import os
 
 
def create_structure():
    folders = [
        # Phase 0 — Business Framing
        "docs",
        # Phase 1 — Data
        "data/raw",
        "data/processed",
        # Phase 2 — SQL
        "src/sql",
        # Phase 3 — EDA
        "notebooks",
        # Phase 4-7 — Modeling, Evaluation, Explainability
        "src/ingestion",
        "src/features",
        "src/training",
        "src/explainability",
        # Phase 8 — Experiment Tracking
        "src/registry",
        # Phase 9 — Dashboard
        "dashboard",
        # Phase 10 — API + Docker + CI
        "src/serving",
        "src/utils",
        "api",
        # Outputs
        "artifacts/models",
        "artifacts/reports",
        "artifacts/figures",
        # Testing
        "tests",
        # Config
        "configs",
        ".github/workflows",
    ]
 
    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
 
    # Create __init__.py for all src packages
    src_packages = [
        "src",
        "src/ingestion",
        "src/features",
        "src/training",
        "src/explainability",
        "src/serving",
        "src/registry",
        "src/sql",
        "src/utils",
    ]
    for pkg in src_packages:
        init_path = os.path.join(pkg, "__init__.py")
        if not os.path.exists(init_path):
            open(init_path, "w").close()
 
    # Create placeholder files with phase labels
    placeholders = {
        "src/ingestion/load_data.py": '"""Phase 1 — Data loading from raw CSV."""\n',
        "src/ingestion/validate_data.py": '"""Phase 1 — Data quality checks and validation."""\n',
        "src/features/build_features.py": '"""Phase 4 — Feature engineering pipeline."""\n',
        "src/training/train.py": '"""Phase 5 — Model training (LR, RF, XGBoost)."""\n',
        "src/training/evaluate.py": '"""Phase 6 — Evaluation with business cost framing."""\n',
        "src/training/tune.py": '"""Phase 5 — Hyperparameter tuning."""\n',
        "src/explainability/shap_explain.py": '"""Phase 7 — SHAP global and local explanations."""\n',
        "src/serving/predict.py": '"""Phase 10 — Prediction logic."""\n',
        "src/registry/mlflow_registry.py": '"""Phase 8 — MLflow experiment tracking."""\n',
        "src/sql/queries.py": '"""Phase 2 — DuckDB SQL exploration and cohort analysis."""\n',
        "src/utils/helpers.py": '"""Shared utility functions."""\n',
        "api/main.py": '"""Phase 10 — FastAPI inference app."""\n',
        "api/schemas.py": '"""Phase 10 — Pydantic request/response schemas."""\n',
        "dashboard/app.py": '"""Phase 9 — Streamlit dashboard."""\n',
        "tests/test_ingestion.py": '"""Phase 1 — Tests for data ingestion."""\n',
    }
 
    created_count = 0
    for filepath, content in placeholders.items():
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write(content)
            created_count += 1
 
    print("=" * 50)
    print("  Project structure created successfully!")
    print("=" * 50)
    print(f"  Directories : {len(folders)}")
    print(f"  Source files : {created_count}")
    print(f"  __init__.py  : {len(src_packages)}")
    print()
    print("  Next steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. Download dataset to data/raw/")
    print("  3. git init && git add . && git commit -m 'Phase 0: scaffold'")
    print("=" * 50)
 
 
if __name__ == "__main__":
    create_structure()