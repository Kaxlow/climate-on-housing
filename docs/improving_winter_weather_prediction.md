# Improving Winter Weather Risk Prediction

## Current Performance

Winter Weather (WNTW) risk prediction has the **worst F1 score** among all hazard types:

| Hazard | Best F1 Score | Best Model |
|--------|---------------|------------|
| Overall | 0.7325 | Gradient Boosting |
| Riverine Flooding (IFLD) | 0.7171 | Gradient Boosting |
| Hurricane (HRCN) | 0.6169 | Random Forest |
| Wildfire (WFIR) | 0.6070 | Gradient Boosting |
| Tornado (TRND) | 0.5905 | Gradient Boosting |
| **Winter Weather (WNTW)** | **0.4151** | **Random Forest** |

**Gap: 0.30 F1 points** below overall risk, **0.18 points** below the next-worst hazard (Tornado).

## Root Causes

### 1. **Feature-Risk Mismatch**
The current feature set (income, housing burden, insurance premiums, property taxes, migration, housing market trends) correlates strongly with **coastal/property value** hazards (hurricanes, flooding) but **weakly with geography-driven** hazards like winter weather.

**Evidence from confusion matrices:**
- Winter Weather predictions scatter across all 5 risk classes
- Hurricane/Flood models show clear majority-class prediction patterns
- Winter Weather has the most balanced class distribution (24-37% across classes), but the model treats it like random noise

### 2. **Missing Geographic Features**
Winter weather risk is primarily determined by:
- **Latitude** (northern states have higher risk)
- **Elevation** (mountainous regions)
- **Climate zones** (continental vs maritime)
- **Infrastructure preparedness** (snow removal capacity, building codes)

None of these are captured in the current feature set.

### 3. **Missing Climate Features**
Economic features don't capture actual winter weather exposure:
- Historical winter temperature extremes
- Snowfall/ice accumulation patterns
- Temperature anomalies and trends
- Days below freezing

### 4. **Missing Historical Damage Features**
Unlike hurricanes (where high insurance premiums signal high risk), winter weather damage is less reflected in insurance pricing but shows up in **NOAA Storm Events** data.

## Recommended Improvements

### **Phase 1: Add Geographic Features** (Highest Impact)

Add county-level geographic attributes that directly correlate with winter weather risk:

**Data sources:**
- Use existing `ref.counties` table for MSA/CSA groupings
- Add latitude/longitude from county centroids (available in Census TIGER files)
- Add elevation data (USGS or Census)
- Add climate zone classifications (NOAA Climate Divisions or Köppen-Geiger)

**New features to engineer:**

```python
# In load_data() query, add:
"""
county_geo AS (
    SELECT 
        fips,
        latitude,
        longitude,
        elevation_meters,
        -- Latitude bins (winter risk increases with latitude)
        CASE 
            WHEN latitude < 32 THEN 'southern'
            WHEN latitude < 37 THEN 'mid_southern'
            WHEN latitude < 42 THEN 'mid_northern'
            ELSE 'northern'
        END as latitude_zone,
        -- Elevation bins
        CASE
            WHEN elevation_meters < 200 THEN 'lowland'
            WHEN elevation_meters < 600 THEN 'midland'
            WHEN elevation_meters < 1200 THEN 'highland'
            ELSE 'mountain'
        END as elevation_zone,
        -- Distance from coast (maritime vs continental climate)
        distance_to_coast_km
    FROM ref.county_geography  -- Would need to create this table
)
"""
```

**Expected improvement:** +5-10 F1 points

### **Phase 2: Add Historical Weather Features** (High Impact)

Use `mart.ncei_county_weather_monthly` to create winter weather exposure metrics:

**New features to engineer:**

```python
# In load_data() query, add:
"""
winter_weather_history AS (
    SELECT
        fips,
        -- Winter months (Dec-Feb) over last 5 years
        AVG(CASE WHEN month IN (12, 1, 2) THEN min_temperature_f END) as avg_winter_min_temp,
        MIN(CASE WHEN month IN (12, 1, 2) THEN min_temperature_f END) as extreme_winter_min_temp,
        AVG(CASE WHEN month IN (12, 1, 2) THEN precipitation_inches END) as avg_winter_precip,
        -- Temperature variability (high variability = unpredictable extremes)
        STDDEV(CASE WHEN month IN (12, 1, 2) THEN min_temperature_f END) as winter_temp_variability,
        -- Cold anomaly frequency (days colder than normal)
        SUM(CASE WHEN month IN (12, 1, 2) AND min_temperature_anomaly_f < -5 THEN 1 ELSE 0 END) as cold_anomaly_count,
        -- Extreme cold days (< 0°F)
        SUM(CASE WHEN month IN (12, 1, 2) AND min_temperature_f < 0 THEN 1 ELSE 0 END) as subzero_day_count
    FROM mart.ncei_county_weather_monthly
    WHERE year BETWEEN {min_year - 5} AND {max_year}  -- Use 5-year history
    GROUP BY fips
)
"""
```

**Expected improvement:** +8-12 F1 points

### **Phase 3: Add Historical Damage Features** (Medium Impact)

Use `mart.noaa_storm_events` to create county-level winter weather damage exposure:

**New features to engineer:**

```python
# In load_data() query, add:
"""
winter_event_history AS (
    SELECT
        fips,
        -- Count of winter weather events (last 10 years)
        COUNT(*) as winter_event_count,
        -- Total damage from winter events
        SUM(total_damage_amount) as total_winter_damage,
        -- Average damage per event
        AVG(total_damage_amount) as avg_winter_damage,
        -- Injury/death counts (human impact)
        SUM(injuries_direct_count + deaths_direct_count) as winter_casualty_count,
        -- Event type diversity (ice storm + snow + cold = higher risk)
        COUNT(DISTINCT event_type) as winter_event_type_diversity
    FROM mart.noaa_storm_events
    WHERE event_type IN (
        'Winter Weather', 'Winter Storm', 'Heavy Snow', 
        'Extreme Cold/Wind Chill', 'Cold/Wind Chill', 
        'Frost/Freeze', 'Ice Storm', 'Lake-Effect Snow'
    )
    AND year BETWEEN {min_year - 10} AND {max_year}
    GROUP BY fips
)
"""
```

**Expected improvement:** +3-5 F1 points

### **Phase 4: Add Infrastructure/Demographic Features** (Low-Medium Impact)

Winter weather resilience depends on local infrastructure and demographics:

**New features from ACS data:**
- **Building age:** Older housing stock → less insulation, higher vulnerability
- **Heating fuel type:** Oil/electricity vs natural gas (affects outage risk)
- **Poverty rate:** Lower-income counties → less snow removal capacity, older infrastructure
- **Population density:** Rural areas → slower emergency response

```python
# In load_data() query, add from existing ACS tables:
"""
infrastructure_demographics AS (
    SELECT
        fips,
        AVG(median_home_age) as median_home_age,
        AVG(poverty_rate) as poverty_rate,
        AVG(population_density) as population_density,
        -- From mart.acs_county_demographic_annual
        AVG(households_heating_oil_pct) as heating_oil_pct,
        AVG(households_heating_electricity_pct) as heating_electricity_pct
    FROM mart.acs_county_demographic_annual
    WHERE year BETWEEN {min_year} AND {max_year}
    GROUP BY fips
)
"""
```

**Expected improvement:** +2-4 F1 points

### **Phase 5: Model Architecture Changes** (Low Impact)

Current models may be underfit for winter weather's complex geographic patterns:

1. **Use polynomial features** for latitude × temperature interactions:
   ```python
   from sklearn.preprocessing import PolynomialFeatures
   poly = PolynomialFeatures(degree=2, interaction_only=True)
   X_poly = poly.fit_transform(X[['latitude', 'avg_winter_min_temp', 'elevation']])
   ```

2. **Increase tree depth** for Random Forest/Gradient Boosting:
   ```python
   'max_depth': [20, 30, 50, None],  # Current: [10, 20, 30, None]
   'min_samples_leaf': [1, 2],  # Current: [1, 2, 4] - allow finer splits
   ```

3. **Try XGBoost** instead of sklearn GradientBoosting (better handling of complex interactions)

4. **Ensemble across models** instead of picking the best single model

**Expected improvement:** +1-3 F1 points

## Implementation Plan

### Quick Win (1-2 hours)
**Add latitude/longitude from existing county reference data:**
- Extract lat/long from Census TIGER shapefiles or use a lookup table
- Add to `ref.counties` or create `ref.county_geography`
- Add latitude_zone and derived features to the model

**Expected: +5 F1 points**

### High-Value Addition (4-6 hours)
**Add historical weather features from existing `mart.ncei_county_weather_monthly`:**
- No new data download needed, table already exists
- Add winter temperature/precipitation aggregations
- Add anomaly counts and variability metrics

**Expected: +8 F1 points**

### Medium-Value Addition (3-4 hours)
**Add winter storm event history from existing `mart.noaa_storm_events`:**
- No new data download needed, table already exists
- Aggregate winter event types by county over 10-year window
- Create damage and casualty metrics

**Expected: +3 F1 points**

### Long-Term Enhancement (8-12 hours)
**Add elevation, climate zones, and ACS infrastructure features:**
- Download/process elevation data
- Add climate zone classifications
- Extract heating fuel and building age from ACS

**Expected: +3 F1 points**

## Total Expected Improvement

**Baseline:** 0.42 F1 (current)  
**Quick Win (Phase 1):** 0.47 F1 (+5 points)  
**+ Phase 2:** 0.55 F1 (+8 points)  
**+ Phase 3:** 0.58 F1 (+3 points)  
**+ Phase 4:** 0.61 F1 (+3 points)  
**+ Phase 5:** 0.63 F1 (+2 points)  

**Target:** **0.60-0.65 F1** (matching Tornado/Wildfire performance)

## Why These Features Matter

### Geographic Features
Winter weather risk is **geographically determined**. Counties at 45°N latitude have fundamentally different winter weather risk than counties at 30°N, regardless of economic factors. Current model has no way to learn this.

### Historical Weather Features
**Direct measurement** of what the model is trying to predict. A county with 20 subzero days per year has higher winter weather risk than one with 0, period. This is the single strongest signal.

### Historical Damage Features
**Observed vulnerability**. If a county has experienced $10M in winter storm damage over 10 years, it reveals actual exposure that may not be captured in climate data (e.g., ice storms in unusual locations).

### Infrastructure Features
**Resilience factors**. Two counties with identical climate can have different risk ratings based on preparedness (building codes, snow removal, heating systems).

## Code Changes Required

### 1. Update `load_data()` query
Add new CTEs for geographic, weather, and event history features.

### 2. Update `engineer_features()`
Add binning/encoding for new categorical features (latitude zones, elevation zones).

### 3. Update `prepare_features()`
Add new feature columns to the base_features list.

### 4. Create data pipeline additions
If geographic data doesn't exist:
- Create `ref.county_geography` table with lat/long/elevation
- Add to `build-database` pipeline

## Alternative: Hazard-Specific Feature Sets

Instead of using the same features for all hazards, create **hazard-specific feature engineering**:

```python
class ClimateRiskPredictor:
    def get_hazard_features(self, hazard_type: str) -> List[str]:
        """Return hazard-specific feature set."""
        base_features = ['median_household_income', 'housing_burden_30pct', ...]
        
        if hazard_type == 'WNTW':
            return base_features + [
                'latitude', 'elevation', 'avg_winter_min_temp',
                'winter_event_count', 'subzero_day_count'
            ]
        elif hazard_type == 'HRCN':
            return base_features + [
                'coastal_proximity', 'elevation', 'hurricane_event_count'
            ]
        # ... etc
```

This allows winter weather model to use climate features while hurricane model uses coastal features.

## Validation Strategy

After adding features:

1. **Compare feature importance:** Are new features in top 5?
2. **Check confusion matrix:** Does the model distinguish between classes better?
3. **Per-class F1 scores:** Does improvement come from all classes or just majority?
4. **Geographic validation:** Does the model correctly rank northern counties higher than southern?

Example validation query:
```python
# After training, check if predictions make geographic sense
predictions_df['latitude_bin'] = pd.cut(predictions_df['latitude'], bins=5)
predictions_df.groupby('latitude_bin')['predicted_risk'].value_counts(normalize=True)
# Expect: Higher latitudes → higher predicted risk
```

## Next Steps

1. **Priority 1:** Add latitude/longitude to county reference table
2. **Priority 2:** Add winter weather aggregations from NCEI table
3. **Priority 3:** Add winter storm event history from NOAA table
4. **Priority 4:** Retrain models and validate improvement
5. **Priority 5:** Consider elevation and climate zone data

Would you like me to implement any of these phases?
