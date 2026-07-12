# Fixes to county_nri_feature_correlations.ipynb

## Date: 2026-07-09

## Issues Found and Fixed

### 1. Migration Rate Variable ✅ FIXED

**Issue**: Used `domestic_in_migration_rate` which only captures domestic in-migration (excludes international migration and out-migration)

**Fix**: Changed to TRUE net migration rate from StatsAmerica Components of Population Change data

**Changes**:
- Cell `nri-feature-corr-md-01`: Updated header to document StatsAmerica data source
- Cell `nri-feature-corr-code-03`: Updated `FEATURE_META` dictionary
  - Old: `"domestic_in_migration_rate": ("Demographic", "Net Migration")`
  - New: `"net_migration_rate": ("Demographic", "Net Migration Rate")`

- Cell `nri-feature-corr-code-09`: Complete rewrite to use StatsAmerica data
  - **Data source**: `mart.statsamerica_population_components_annual`
  - **Column used**: `total_net_migration` (domestic + international, in minus out)
  - **Calculation**: `avg(total_net_migration) / avg(population)` over 10 years
  - **Result**: True net migration rate as proportion of population (can be negative!)

**Impact**: 
- Now using TRUE net migration (in-migration minus out-migration)
- Can detect counties losing population (negative net migration)
- Much more accurate for climate impact analysis
- Aligns with StatsAmerica/Census Bureau population accounting methodology
- Example: A county might show +2% in-migration (ACS) but -1% net migration (StatsAmerica) if more people left than arrived

### 2. Disability Share Variable ⚠️ ISSUE IDENTIFIED (Not Fixed Yet)

**Issue**: Values are extremely large (890, 7370, 9373, etc.) suggesting raw counts instead of percentages

**Current Query**:
```python
f"avg({num('dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct').rsplit(' AS ', 1)[0]}) AS disability_share"
```

**Expected Behavior**: Should return percentage values (e.g., 10-25%)

**Actual Values Observed**:
- Mean: 9,373
- Range: 9 to 912,315
- These look like raw population counts, not percentages

**Investigation Needed**:
1. Check if the ACS column name is correct
2. Verify if the column actually contains percentages or counts
3. May need to calculate percentage manually if only counts are available

**Potential Fix** (requires verification):
```python
# If the column is actually counts, need to compute percentage:
# disability_count / total_population * 100
```

### 3. Income Variable ⚠️ NOT ADDRESSED YET

**Issue**: Uses county-wide median household income instead of homeowner-specific income

**Current**: `median_household_income` from `mart.acs_county_economic_annual` (DP03)

**Should Use**: Homeowner income from `mart.acs_county_affordability_annual` table:
- Column: `s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est`
- Source: Census Table S2503 (owner-occupied units only)

**Why It Matters**:
- Homeowner income is typically 20-40% higher than county-wide median
- Affects `homeownership_cost_pct_income` calculation accuracy
- Insurance burden calculations become more realistic

**Recommended Fix**:
```python
afford_selects = [
    f"avg({num('s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est').rsplit(' AS ', 1)[0]}) AS median_household_income",
    # ... rest of selects
]
```

### 4. Homeownership Cost Calculation ⚠️ INCOMPLETE

**Issue**: Missing mortgage/loan costs from homeownership burden

**Current Calculation**:
```python
afford["homeownership_cost_annual_usd"] = (
    afford["insurance_homeowners_annual_usd"]
    + afford["property_taxes_median_annual_usd"]
    + afford["utilities_monthly_usd"] * 12
)
```

**Missing Component**: Mortgage principal + interest payments

**Why It Matters**: 
- Mortgage is typically the largest component of homeownership costs
- Current calculation severely underestimates total cost burden
- For a $300k home with 6% interest, mortgage = ~$18k/year vs. insurance ($1.2k) + taxes ($2k) + utilities ($4k) = $7.2k

**Challenge**: 
- ACS doesn't provide average mortgage payment directly
- Available: Total monthly owner costs (includes mortgage + PITI)
- Could use: `s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est` × 12

**Potential Fix**:
```python
# Use total owner costs from S2503 which includes mortgage
afford["homeownership_cost_annual_usd"] = (
    afford["s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est"] * 12
)

# Or calculate mortgage separately if needed:
afford["mortgage_annual_est"] = (
    afford["total_owner_costs_monthly"] * 12 
    - afford["insurance_homeowners_annual_usd"]
    - afford["property_taxes_median_annual_usd"]
    - afford["utilities_monthly_usd"] * 12
).clip(lower=0)
```

## Summary of Fixes Applied

✅ **Fixed**:
1. Migration rate: Changed from `domestic_in_migration_rate` to `total_in_migration_rate`

⚠️ **Requires Further Investigation/Fixing**:
2. Disability share: Values suspiciously large (likely raw counts not percentages)
3. Income: Should use homeowner income from S2503, not county-wide median
4. Homeownership cost: Missing mortgage component in burden calculation

## Next Steps

1. **Immediate**: Investigate disability_share data source
   - Run: `SELECT fips, dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct FROM mart.acs_county_demographic_annual WHERE fips = '06037' LIMIT 1`
   - Check if values are percentages or counts

2. **High Priority**: Update income to use homeowner-specific income from S2503
   - Affects insurance burden calculations
   - More accurate for homeownership cost analysis

3. **High Priority**: Fix homeownership cost calculation to include mortgage
   - Current calculation underestimates by 2-3x
   - Use S2503 median monthly owner costs × 12

4. **After Fixes**: Re-run notebook and regenerate output files
   - `county_nri_feature_correlations.csv`
   - `county_nri_feature_coverage.csv`
   - `county_nri_feature_matrix.parquet`

5. **Validation**: Compare before/after correlation values
   - Document changes in correlation strength
   - Verify results make intuitive sense

## Related Documentation

- `docs/variable_fixes_summary.md` - Changes to migration rate and homeowner income in models
- `docs/county_climate_risk_prediction.md` - Model feature documentation
- `CLAUDE.md` - Mart table documentation
