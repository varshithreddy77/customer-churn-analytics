import duckdb
import pandas as pd
from src.ingestion.load_data import load_raw_data
 
 
def get_connection(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    df_reset = df.reset_index()
    con.register("customers", df_reset)
 
    return con
 
#  Query 1 — Overall Churn Summary

QUERY_CHURN_SUMMARY = """
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    SUM(CASE WHEN churn = 'No'  THEN 1 ELSE 0 END) AS retained,
    ROUND(AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_at_risk,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN total_charges ELSE 0 END), 2) AS total_revenue_lost
FROM customers
"""
 
 
#  Query 2 — Churn by Contract Type
 
QUERY_CHURN_BY_CONTRACT = """
SELECT
    contract,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure), 1) AS avg_tenure_months
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC
"""
 
 
#  Query 3 — Churn by Tenure Bands
 
QUERY_CHURN_BY_TENURE = """
SELECT
    CASE
        WHEN tenure <= 6  THEN '01. 0-6 months'
        WHEN tenure <= 12 THEN '02. 7-12 months'
        WHEN tenure <= 24 THEN '03. 13-24 months'
        WHEN tenure <= 48 THEN '04. 25-48 months'
        ELSE                   '05. 49+ months'
    END AS tenure_band,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customers
GROUP BY tenure_band
ORDER BY tenure_band
"""
 
 
#  Query 4 — Churn by Payment Method


QUERY_CHURN_BY_PAYMENT = """
SELECT
    payment_method,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(tenure), 1) AS avg_tenure
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC
"""
 
 
 
#  Query 5 — Churn by Internet Service Type
 


QUERY_CHURN_BY_INTERNET = """
SELECT
    internet_service,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure), 1) AS avg_tenure
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC
"""
 
 

#  Query 6 — High-Risk Customer Profile
 

QUERY_HIGH_RISK_PROFILE = """
SELECT
    contract,
    internet_service,
    payment_method,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM customers
GROUP BY contract, internet_service, payment_method
HAVING COUNT(*) >= 50
ORDER BY churn_rate_pct DESC
LIMIT 10
"""
 
 
#  Query 7 — Senior Citizen Analysis
 
 
QUERY_SENIOR_CITIZEN = """
SELECT
    CASE WHEN senior_citizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS customer_type,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure), 1) AS avg_tenure
FROM customers
GROUP BY customer_type
ORDER BY churn_rate_pct DESC
"""
 
 
#  Query 8 — Service Add-on Stickiness
 

QUERY_SERVICE_STICKINESS = """
SELECT
    addon_count,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM (
    SELECT
        *,
        (CASE WHEN online_security   = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN online_backup     = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN tech_support      = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN streaming_tv      = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN streaming_movies  = 'Yes' THEN 1 ELSE 0 END) AS addon_count
    FROM customers
    WHERE internet_service != 'No'
) sub
GROUP BY addon_count
ORDER BY addon_count
"""
 
 
 
#  Query 9 — Monthly Charges Segments
 
QUERY_CHARGES_SEGMENTS = """
SELECT
    CASE
        WHEN monthly_charges < 30  THEN '01. Under $30'
        WHEN monthly_charges < 60  THEN '02. $30–$59'
        WHEN monthly_charges < 90  THEN '03. $60–$89'
        ELSE '04. $90+'
    END AS charge_segment,
    COUNT(*) AS customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(AVG(CASE WHEN churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY charge_segment
ORDER BY charge_segment
"""
 
 
#  Query 10 — Revenue at Risk by Segment (Business KPI)
 

QUERY_REVENUE_AT_RISK = """
SELECT
    contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges * 12 ELSE 0 END), 2) AS annual_revenue_at_risk,
    ROUND(SUM(monthly_charges * 12), 2) AS total_annual_revenue,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges * 12 ELSE 0 END) / SUM(monthly_charges * 12) * 100, 2) AS pct_revenue_at_risk
FROM customers
GROUP BY contract
ORDER BY annual_revenue_at_risk DESC
"""
 
 
 
ALL_QUERIES = [
    ("1. CHURN SUMMARY", QUERY_CHURN_SUMMARY),
    ("2. CHURN BY CONTRACT TYPE", QUERY_CHURN_BY_CONTRACT),
    ("3. CHURN BY TENURE BAND", QUERY_CHURN_BY_TENURE),
    ("4. CHURN BY PAYMENT METHOD", QUERY_CHURN_BY_PAYMENT),
    ("5. CHURN BY INTERNET SERVICE", QUERY_CHURN_BY_INTERNET),
    ("6. HIGH-RISK CUSTOMER PROFILES", QUERY_HIGH_RISK_PROFILE),
    ("7. SENIOR CITIZEN ANALYSIS", QUERY_SENIOR_CITIZEN),
    ("8. SERVICE ADD-ON STICKINESS", QUERY_SERVICE_STICKINESS),
    ("9. MONTHLY CHARGES SEGMENTS", QUERY_CHARGES_SEGMENTS),
    ("10. REVENUE AT RISK BY CONTRACT", QUERY_REVENUE_AT_RISK),
]
 
 
def run_all_queries(df: pd.DataFrame, display: bool = True) -> dict:
    con = get_connection(df)
    results = {}
 
    for title, query in ALL_QUERIES:
        result_df = con.execute(query).fetchdf()
        results[title] = result_df
 
        if display:
            print()
            print("=" * 70)
            print(f"  {title}")
            print("=" * 70)
            print(result_df.to_string(index=False))
 
    con.close()
    return results
 
 
def run_single_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    con = get_connection(df)
    result = con.execute(query).fetchdf()
    con.close()
    return result
 
 
# ── Run all queries when executed directly ──
if __name__ == "__main__":
    print("Loading dataset...")
    df = load_raw_data()
    print(f"Loaded {df.shape[0]} customers with {df.shape[1]} features\n")
 
    results = run_all_queries(df)