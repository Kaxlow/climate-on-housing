"""
Test script to validate housing cost as % of income pipeline changes.

Run this after rebuilding the database with --marts-only to verify:
1. housing_cost_pct_income column exists in affordability mart
2. Values are reasonable (between 0-200%)
3. median_household_income is available from economic mart
4. Sample comparison of old vs new calculation
"""

import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "quoll.duckdb"

def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("=" * 80)
    print("Housing Cost Pipeline Changes Validation")
    print("=" * 80)

    # Test 1: Check if housing_cost_pct_income exists
    print("\n1. Checking if housing_cost_pct_income column exists...")
    result = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'mart'
          AND table_name = 'acs_county_affordability_annual'
          AND column_name = 'housing_cost_pct_income'
    """).fetchdf()

    if len(result) > 0:
        print("   [OK] housing_cost_pct_income column exists")
    else:
        print("   [FAIL] housing_cost_pct_income column NOT found")
        return

    # Test 2: Check data availability
    print("\n2. Checking data availability...")
    stats = con.execute("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(housing_cost_pct_income) as non_null_count,
            ROUND(COUNT(housing_cost_pct_income) * 100.0 / COUNT(*), 2) as pct_populated
        FROM mart.acs_county_affordability_annual
    """).fetchdf()
    print(f"   Total rows: {stats['total_rows'][0]:,}")
    print(f"   Non-null housing_cost_pct_income: {stats['non_null_count'][0]:,} ({stats['pct_populated'][0]}%)")

    # Test 3: Check value distribution
    print("\n3. Checking value distribution...")
    distribution = con.execute("""
        SELECT
            MIN(housing_cost_pct_income) as min_val,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY housing_cost_pct_income) as p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY housing_cost_pct_income) as median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY housing_cost_pct_income) as p75,
            MAX(housing_cost_pct_income) as max_val,
            AVG(housing_cost_pct_income) as mean_val
        FROM mart.acs_county_affordability_annual
        WHERE housing_cost_pct_income IS NOT NULL
    """).fetchdf()
    print("   Distribution:")
    for col in ['min_val', 'p25', 'median', 'p75', 'max_val', 'mean_val']:
        print(f"   {col:12s}: {distribution[col][0]:.2f}%")

    # Test 4: Compare with median_household_income from economic mart
    print("\n4. Verifying median_household_income from economic mart...")
    income_check = con.execute("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(dp03_income_and_benefits_total_households_median_household_income_est) as non_null_count
        FROM mart.acs_county_economic_annual
    """).fetchdf()
    print(f"   Total rows in economic mart: {income_check['total_rows'][0]:,}")
    print(f"   Non-null median household income: {income_check['non_null_count'][0]:,}")

    # Test 5: Sample comparison
    print("\n5. Sample comparison (2023 data)...")
    sample = con.execute("""
        WITH latest_year AS (
            SELECT MAX(year) as max_year FROM mart.acs_county_affordability_annual
        )
        SELECT
            a.fips,
            c.county_name,
            c.state,
            a.housing_cost_pct_income as new_housing_burden_pct,
            e.dp03_income_and_benefits_total_households_median_household_income_est as income_from_dp03,
            a.s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est as monthly_cost_s2503,
            a.median_owner_costs_mortgage as monthly_cost_b25088
        FROM mart.acs_county_affordability_annual a
        INNER JOIN latest_year ly ON a.year = ly.max_year
        LEFT JOIN ref.counties c ON a.fips = c.fips
        LEFT JOIN mart.acs_county_economic_annual e ON a.fips = e.fips AND a.year = e.year
        WHERE a.housing_cost_pct_income IS NOT NULL
          AND c.state IN ('CA', 'TX', 'NY', 'FL')
        ORDER BY RANDOM()
        LIMIT 10
    """).fetchdf()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(sample.to_string(index=False))

    # Test 6: Check for outliers
    print("\n6. Checking for outliers (>100% or <0%)...")
    outliers = con.execute("""
        SELECT
            COUNT(*) as outlier_count,
            COUNT(*) FILTER (WHERE housing_cost_pct_income > 100) as over_100_pct,
            COUNT(*) FILTER (WHERE housing_cost_pct_income < 0) as negative_pct
        FROM mart.acs_county_affordability_annual
        WHERE housing_cost_pct_income IS NOT NULL
    """).fetchdf()
    print(f"   Over 100%: {outliers['over_100_pct'][0]:,}")
    print(f"   Negative: {outliers['negative_pct'][0]:,}")

    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
