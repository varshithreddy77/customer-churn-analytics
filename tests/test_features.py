import pytest
import pandas as pd
import numpy as np
from src.ingestion.load_data import load_raw_data
from src.features.build_features import (
    clean_data,
    engineer_features,
    encode_binary_columns,
    build_preprocessor,
    build_features_pipeline,
)
 
 
 
@pytest.fixture(scope="module")
def raw_df():
    """Load raw data once for all tests."""
    return load_raw_data()
 
 
@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    """Cleaned data (no missing, no duplicates)."""
    return clean_data(raw_df)
 
 
@pytest.fixture(scope="module")
def engineered_df(cleaned_df):
    """Data with engineered features."""
    return engineer_features(cleaned_df)
 
 
@pytest.fixture(scope="module")
def pipeline_output():
    """Run the full pipeline once and share across tests (it's slow)."""
    return build_features_pipeline(save_artifacts=False)
 
 
# ═══════════════════════════════════════════════════════════
#  Tests — Cleaning
# ═══════════════════════════════════════════════════════════
 
class TestCleanData:
    """Tests for the data cleaning step."""
 
    def test_no_missing_after_cleaning(self, cleaned_df):
        """All missing values should be handled."""
        assert cleaned_df.isnull().sum().sum() == 0
 
    def test_no_duplicates_after_cleaning(self, cleaned_df):
        """All duplicates should be removed."""
        assert cleaned_df.duplicated().sum() == 0
 
    def test_22_duplicates_removed(self, raw_df, cleaned_df):
        """Exactly 22 duplicates were identified in Phase 1 — verify they're gone."""
        assert len(raw_df) - len(cleaned_df) == 22
 
#  Tests — Feature Engineering

class TestEngineeredFeatures:
    """Tests for the 6 engineered features."""
 
    def test_tenure_band_categories(self, engineered_df):
        """tenure_band should have exactly 5 categories."""
        bands = engineered_df["tenure_band"].unique()
        expected = {"0-6_mo", "7-12_mo", "13-24_mo", "25-48_mo", "49+_mo"}
        assert set(bands) == expected
 
    def test_addon_count_range(self, engineered_df):
        """addon_count should be 0-6 (6 add-on services)."""
        assert engineered_df["addon_count"].min() >= 0
        assert engineered_df["addon_count"].max() <= 6
 
    def test_is_new_customer_binary(self, engineered_df):
        """is_new_customer should only be 0 or 1."""
        assert set(engineered_df["is_new_customer"].unique()).issubset({0, 1})
 
    def test_charge_per_tenure_no_inf(self, engineered_df):
        """charge_per_tenure should never be infinite (handled div-by-zero)."""
        assert not np.isinf(engineered_df["charge_per_tenure"]).any()
 
    def test_contract_risk_ordinal(self, engineered_df):
        """contract_risk: Two year=0, One year=1, Month-to-month=2."""
        risk_by_contract = engineered_df.groupby("contract")["contract_risk"].first()
        assert risk_by_contract["Two year"] == 0
        assert risk_by_contract["One year"] == 1
        assert risk_by_contract["Month-to-month"] == 2
 
#  Tests — Encoding
 
class TestEncoding:
    """Tests for binary encoding."""
 
    def test_churn_is_binary(self, engineered_df):
        """After encoding, churn should only be 0 or 1."""
        df = encode_binary_columns(engineered_df)
        assert set(df["churn"].unique()) == {0, 1}
 
    def test_churn_rate_preserved(self, engineered_df):
        """Encoding shouldn't change the churn rate."""
        df = encode_binary_columns(engineered_df)
        assert 0.25 < df["churn"].mean() < 0.28
 
 
#  Tests — Full Pipeline
 
class TestFullPipeline:
    """Integration tests for the complete pipeline."""
 
    def test_pipeline_runs_end_to_end(self, pipeline_output):
        """Full pipeline should execute without errors."""
        X_train, X_test, y_train, y_test, preprocessor = pipeline_output
        assert X_train is not None
        assert X_test is not None
        assert preprocessor is not None
 
    def test_train_test_split_ratio(self, pipeline_output):
        """80/20 split should produce roughly that ratio."""
        X_train, X_test, _, _, _ = pipeline_output
        total = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total
        assert 0.79 < train_ratio < 0.81
 
    def test_stratification_preserves_churn_rate(self, pipeline_output):
        """Stratified split should preserve churn rate in both sets."""
        _, _, y_train, y_test, _ = pipeline_output
        assert abs(y_train.mean() - y_test.mean()) < 0.01
 
    def test_no_target_leakage(self, pipeline_output):
        """X_train should not contain the target column."""
        X_train, _, _, _, _ = pipeline_output
        assert "churn" not in X_train.columns
 
    def test_no_duplicate_indices(self, pipeline_output):
        """Train and test should have non-overlapping customer IDs."""
        X_train, X_test, _, _, _ = pipeline_output
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0
 
    def test_no_nulls_in_output(self, pipeline_output):
        """Transformed features should have no missing values."""
        X_train, X_test, _, _, _ = pipeline_output
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0
 
    def test_same_columns_train_and_test(self, pipeline_output):
        """Train and test must have identical feature columns."""
        X_train, X_test, _, _, _ = pipeline_output
        assert list(X_train.columns) == list(X_test.columns)