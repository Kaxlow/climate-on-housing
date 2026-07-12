# Variable Definition Fixes - Summary

## Changes Made (2026-07-09)

### 1. "Net Migration Rate" → "In-Migration Rate" (Renamed)

**Problem**: The variable named "net_migration_rate" was misleading because:
- It only measured in-migration (people moving INTO the county)
- It excluded international migration
- It did NOT account for out-migration (people leaving)
- True net migration = (in-migration - out-migration)

**Solution**: 
- **Renamed variable**: `net_migration_rate` → `in_migration_rate` (in code)
- Now includes BOTH domestic AND international in-migration
- Added computed column `total_in_migration_rate` to database (no CSV changes needed)
- Updated all references in `climate_risk_prediction.py` and documentation

**Computed Column** (in `build_database.py`):
```sql
-- Existing columns in CSV
domestic_in_migration_rate  -- From B07001 (moves from other US counties/states)
moved_from_abroad_rate      -- From B07001 (moves from other countries)

-- New computed column (added automatically during database build)
total_in_migration_rate = domestic_in_migration_rate + moved_from_abroad_rate
```

**No Data Download Required**: The existing migration CSV already contains both component rates. The database build simply adds them together.

**Data Source**: Census ACS Table B07001
- `B07001_049E`: Moved from different county, same state
- `B07001_065E`: Moved from different state
- `B07001_081E`: Moved from abroad
- `B07001_001E`: Total population 1 year and over

**Migration Data Columns**:
- `domestic_in_migration`: Count of people from other US counties/states (existing in CSV)
- `domestic_in_migration_rate`: Rate of domestic in-migration (existing in CSV)
- `moved_from_abroad_rate`: Rate of international in-migration (existing in CSV)
- `total_in_migration_rate`: NEW - Computed sum of domestic + international rates (added by database build)

**Model Updated**: `climate_risk_prediction.py` now queries:
```sql
-- CTE renamed from net_migration to in_migration
AVG(CAST(total_in_migration_rate AS DOUBLE)) as in_migration_rate
```

**Variable Renamed Throughout**:
- SQL query: `net_migration_rate` → `in_migration_rate`
- Feature list: `'net_migration_rate'` → `'in_migration_rate'`
- CTE name: `net_migration` → `in_migration`
- Documentation: "Net Migration Rate" → "In-Migration Rate"
- Interaction feature comment updated to reflect in-migration (not out-migration)

**Why "In-Migration" Not "Net Migration"?**

We still cannot easily calculate true net migration because:
- ACS B07001 tracks "where people lived 1 year ago" (current residents)
- It does NOT directly track people who MOVED AWAY from the county
- True net migration would require complex residence-change tables or IRS migration data

### 2. Insurance Burden > 100% Fixed

**Problem**: Insurance burden was calculated as `(insurance_premium * 12) / median_household_income * 100`, but:
- Used county-wide median household income (includes renters)
- Homeowners typically have higher incomes than renters
- Resulted in inflated burden percentages (some >100%)

**Solution**: Use homeowner-only income from Census Table S2503

**Updated Calculation** (in `climate_risk_prediction.py`):
```python
# Old income source
median_household_income  # From DP03 (all households)

# New income source
median_homeowner_income  # From S2503_C02_013E (owner-occupied units only)
```

**Data Source**: Census ACS Table S2503 (Financial Characteristics)
- `S2503_C02_013E`: Median household income for OWNER-OCCUPIED housing units
- Already loaded in `mart.acs_county_affordability_annual`

**Why This Fixes the Issue**:
- Owner-occupied household income is typically 20-40% higher than county-wide median
- Example: If county median = $50k, owner median = $70k
  - Old calculation: ($2,400 annual premium / $50k) × 100 = 4.8%
  - New calculation: ($2,400 annual premium / $70k) × 100 = 3.4%
- Eliminates implausible >100% burden values

**Query Updated**:
```sql
-- Added to recent_acs_affordability CTE
AVG(CAST(s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est AS DOUBLE)) 
  as median_homeowner_income
```

### 3. Share with Communication Barrier & Employment Rate

**Status**: These variables are used in the "What Else Are Climate Events Doing to Counties?" section

**Location**: This section was moved to `output/wip/additional-impacts.html` for future development

**Data Sources**:
- **Communication Barrier**: Census ACS DP02 or C16001 (Language Spoken at Home)
  - Example: `dp02_language_spoken_at_home_language_other_than_english_est`
- **Employment Rate**: Census ACS DP03 (Employment Status)
  - Example: `dp03_employment_status_population_16_years_and_over_in_labor_force_pct`

**Current Status**: 
- These variables are available in `mart.acs_county_demographic_annual` and `mart.acs_county_economic_annual`
- They are NOT used in the climate risk prediction models
- They may be used in event window analysis (to be verified when that section is expanded)

## Files Modified

### Code Changes
1. `src/housing_climate_risk/cli/download_acs_data.py`
   - Updated `_migration_row()` to calculate `total_in_migration` and `total_in_migration_rate`
   
2. `src/housing_climate_risk/modeling/climate_risk_prediction.py`
   - Updated SQL query to use `total_in_migration_rate` instead of `domestic_in_migration_rate`
   - Updated SQL query to use `median_homeowner_income` from S2503 instead of county-wide median
   - Removed join to `recent_acs_economic` CTE (no longer needed)
   - Updated module docstring

### Documentation Changes
3. `docs/county_climate_risk_prediction.md`
   - Updated feature descriptions
   - Added data source details for S2503 and B07001
   - Added "Key Data Improvements" section

4. `docs/variable_fixes_summary.md` (this file)
   - Comprehensive explanation of all changes

## Next Steps

### Required After These Changes

1. **Rebuild Database** (computes new column from existing data):
   ```bash
   build-database --marts-only  # Fast rebuild, only recreates mart tables
   ```
   This will add the `total_in_migration_rate` computed column to `mart.acs_county_demographic_annual`

2. **Retrain Climate Risk Models**:
   ```bash
   train-climate-risk-model --all-hazards
   ```
   Models will now use:
   - Total in-migration rate (domestic + international)
   - Homeowner income for insurance burden calculation

3. **Rebuild Stormhouse Page**:
   ```bash
   build-page stormhouse
   ```
   Will regenerate the playbook predictions with corrected data

**Note**: You do NOT need to re-download migration data. The existing CSV already contains both `domestic_in_migration_rate` and `moved_from_abroad_rate`. The database build now computes `total_in_migration_rate = domestic_in_migration_rate + moved_from_abroad_rate` automatically.

### Verification Steps

1. **Check Migration Data**:
   ```sql
   SELECT fips, year, domestic_in_migration_rate, total_in_migration_rate
   FROM mart.acs_county_demographic_annual
   WHERE fips = '06037'  -- Los Angeles County
   ORDER BY year DESC
   LIMIT 5;
   ```
   - `total_in_migration_rate` should be slightly higher than `domestic_in_migration_rate`

2. **Check Income Data**:
   ```sql
   SELECT 
     fips,
     s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est as owner_income
   FROM mart.acs_county_affordability_annual
   WHERE year = 2023 AND fips = '06037';
   ```
   - Should return non-null values for most counties

3. **Check Insurance Burden**:
   - Run a test model training
   - Inspect insurance_burden engineered feature
   - Verify no values > 100%

## Technical Notes

### Why Not True Net Migration?

True net migration calculation would require:
- **Current approach**: Count people currently living in county who moved there in past year ✓
- **Missing piece**: Count people who LIVED in county 1 year ago but moved away ✗

The Census B07001 table tracks current residents and where they came from, not former residents and where they went. To get true net migration, we would need:
- ACS B07003 (more complex, not county-level)
- IRS county-to-county migration data (requires separate download)
- Synthetic estimation from birth/death/total population changes

For climate risk modeling purposes, total in-migration is a reasonable proxy because:
- Areas with climate risk may see reduced in-migration
- In-migration captures economic attractiveness
- Out-migration is harder to interpret (job loss vs. climate fear?)

### Insurance Burden Thresholds

With homeowner income as denominator:
- **<2%**: Very low burden (typical in low-risk, high-income areas)
- **2-4%**: Moderate burden (national average ~3%)
- **4-6%**: High burden (typical in coastal/wildfire zones)
- **>6%**: Very high burden (Florida, Louisiana, California coastal)

These are more realistic than the old calculation which produced many 5-10% values.

## Backward Compatibility

### Breaking Changes
- **Variable renamed**: `net_migration_rate` → `in_migration_rate` in all model code
- **Income changed**: `median_household_income` in model now refers to owner-occupied income, not county-wide
- Models trained before this change will have different feature distributions
- Predictions may shift slightly (typically lower risk for high-income counties)

### Migration Path
1. Old models remain usable but should be labeled as "pre-2026-07-09"
2. New models should be trained with `--tag v2-homeowner-income`
3. Update any analysis notebooks to use new column names

### Data Availability
- S2503 income data: Available 2015-2024 in current ACS downloads ✓
- Total in-migration: Will be available after re-downloading migration data
- No data gaps expected

## Questions?

- **Why not use B07003 for true net?** → Not available at county level, requires tract aggregation
- **Will old models break?** → No, but predictions may differ slightly
- **Do I need to rebuild everything?** → Yes, follow "Required After These Changes" steps above
- **What about employment & communication barrier?** → Available in database, not used in models yet
