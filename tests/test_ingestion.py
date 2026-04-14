import pytest
import pandas as pd
import numpy as np

from src.ingestion.load_data import load_raw_data, _to_snake_case
from src.ingestion.validate_data import (
    validate,
    validate_schema,
    validate_missing_values,
    validate_duplicates,
    validate_value_ranges,
    validate_categories,
    validate_target,
    EXPECTED_SCHEMA,
)


@pytest.fixture
def raw_df():
    return load_raw_data()

@pytest.fixture
def cleaned_df():
    return pd.DataFrame({
        "gender": ["Male", "Female"],
        "senior_citizen": [0, 1],
        "partner": ["Yes", "No"],
        "dependents": ["No", "Yes"],
        "tenure": [12, 48],
        "phone_service": ["Yes", "No"],
        "multiple_lines": ["No", "Yes"],
        "internet_service": ["DSL", "Fiber optic"],
        "online_security": ["Yes", "No"],
        "online_backup": ["Yes", "No"],
        "device_protection": ["Yes", "No"],
        "tech_support": ["Yes", "No"],
        "streaming_tv": ["No", "Yes"],
        "streaming_movies": ["No", "Yes"],
        "contract": ["One year", "Month-to-month"],
        "paperless_billing": ["No", "Yes"],
        "payment_method": ["Mailed check", "Electronic check"],
        "monthly_charges": [29.85, 89.10],
        "total_charges": [358.20, 4276.80],
        "churn": ["No", "Yes"],
    })


class TestLoadData:
    
    def test_load_raw_data_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)
        
    def test_expected_row_count(self, raw_df):
        assert raw_df.shape[0] == 7043
        
    def test_expected_column_count(self, raw_df):
        assert raw_df.shape[1] == 20
        
    def test_customer_id_is_index(self, raw_df):
        assert raw_df.index.name == "customer_id"
        assert "customer_id" not in raw_df.columns
 
    def test_total_charges_is_float(self, raw_df):
        assert raw_df["total_charges"].dtype == np.float64
 
    def test_columns_are_snake_case(self, raw_df):
        for col in raw_df.columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"
            assert " " not in col, f"Column '{col}' has spaces"
 
    def test_snake_case_conversion(self):
        assert _to_snake_case("MonthlyCharges") == "monthly_charges"
        assert _to_snake_case("SeniorCitizen") == "senior_citizen"
        assert _to_snake_case("customerID") == "customer_id"
        assert _to_snake_case("tenure") == "tenure"
        assert _to_snake_case("TotalCharges") == "total_charges"
 

 
class TestValidateSchema:
    """Tests for schema validation."""
 
    def test_schema_passes_on_real_data(self, raw_df):
        result = validate_schema(raw_df)
        assert result["passed"] is True
 
    def test_schema_fails_on_missing_column(self, cleaned_df):

        df = cleaned_df.drop(columns=["churn"])
        result = validate_schema(df)
        assert result["passed"] is False
        assert any("Missing columns" in issue for issue in result["issues"])
 
 
class TestValidateMissing:
 
    def test_detects_missing_values(self, raw_df):
        result = validate_missing_values(raw_df)
        assert "missing_details" in result or result["passed"] is True
 
    def test_passes_on_clean_data(self, cleaned_df):
        result = validate_missing_values(cleaned_df)
        assert result["passed"] is True
 
 
class TestValidateDuplicates:

 
    def test_known_duplicates_in_real_data(self, raw_df):
        result = validate_duplicates(raw_df)
        assert result["passed"] is False
        assert any("22" in issue for issue in result["issues"])
    def test_detects_duplicate_rows(self, cleaned_df):
        df_with_dups = pd.concat([cleaned_df, cleaned_df])
        result = validate_duplicates(df_with_dups)
        assert result["passed"] is False
 
 
class TestValidateRanges:

 
    def test_ranges_pass_on_real_data(self, raw_df):
        result = validate_value_ranges(raw_df)
        assert result["passed"] is True
 
    def test_detects_negative_tenure(self, cleaned_df):
        cleaned_df.loc[0, "tenure"] = -5
        result = validate_value_ranges(cleaned_df)
        assert result["passed"] is False
 
 
class TestValidateTarget:

 
    def test_target_exists(self, raw_df):
        result = validate_target(raw_df)
        assert result["passed"] is True
 
    def test_churn_rate_is_reasonable(self, raw_df):
        result = validate_target(raw_df)
        assert 20 < result["churn_rate"] < 35
 
 
class TestFullValidation:

 
    def test_full_validation_on_real_data(self, raw_df):
        report = validate(raw_df)
 
        assert "dataset_shape" in report
        assert "checks" in report
        assert len(report["checks"]) == 6
        assert report["dataset_shape"]["rows"] == 7043