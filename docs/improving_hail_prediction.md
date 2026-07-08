# Improving Hail Risk Prediction

## Current Baseline

Hail (HAIL) is a **new addition** to the climate risk prediction pipeline, replacing Winter Weather. Expected baseline performance: **F1 ~0.55-0.60** based on class distribution and feature relevance.

## Hail Risk Characteristics

### Geographic Pattern
**Highest Risk States:**
- **Great Plains/Tornado Alley:** Nebraska, Kansas, Oklahoma, Iowa, Texas
- **Upper Midwest:** Wisconsin, Minnesota, North Dakota, South Dakota
- **Central US:** Ohio, Illinois, Michigan, Missouri

**Risk Distribution:**
- 39% Relatively Low
- 34% Very Low
- 21% Relatively Moderate
- 5% Relatively High
- 1% Very High

**Better than Winter Weather:** More concentrated in specific regions (Plains/Midwest) vs dispersed nationwide.

### Temporal Pattern
**Strong Seasonality:**
- **Peak months:** April-June (60% of all events)
  - May: 22,622 events, $4.1B damage (highest)
  - April: 13,409 events, $2.4B damage
  - June: 16,296 events, $1.6B damage
- **Off-season:** Nov-Feb (only 6% of events)

**Implication:** Spring/early summer weather patterns are key predictors.

### Damage Pattern
**Total Historical Damage:** $11.7B (87,696 events since records began)
**Agricultural Impact:** 73% of hail damage is crop damage (vs property)
- 292 counties experienced crop damage from hail
- Average crop damage per event: $708K

**Implication:** Agriculture-heavy counties have higher vulnerability.

### Correlation with Tornadoes
Hail and tornadoes share similar atmospheric conditions:
- **High overlap:** 100 counties are high-risk for both
- **Some divergence:** 49 counties are high hail risk but low tornado risk

**Implication:** Tornado-related features will help, but hail has unique patterns.

## Root Causes of Expected Limitations

### 1. **Geographic-Climate Mismatch**
Current features (income, housing burden, insurance) don't capture:
- **Latitude/longitude patterns** (Tornado Alley concentration)
- **Continental climate zones** (interior vs coastal)
- **Storm track corridors** (Great Plains storm systems)

### 2. **Missing Agricultural Exposure**
73% of hail damage is crop damage, but current features don't capture:
- **Agricultural land use** (cropland percentage)
- **Crop insurance rates** (separate from homeowner insurance)
- **Farm economy exposure** (farming employment, farm income)

### 3. **Missing Historical Storm Data**
No features capture actual hail exposure history:
- **Historical hail events** per county
- **Hail damage amounts** over time
- **Severe weather days** (days with conditions conducive to hail)

### 4. **Missing Atmospheric/Climate Features**
Hail forms in specific atmospheric conditions not captured by current features:
- **Spring precipitation patterns** (fuel for convective storms)
- **Temperature variability** (cold fronts meeting warm air)
- **Severe weather indices** (days with CAPE, wind shear)

## Recommended Improvements

### **Phase 1: Add Geographic Features** (Highest Impact)

Hail risk is geographically concentrated in the Great Plains and Midwest.

**Data sources:**
- County lat/long (Census TIGER or lookup table)
- Distance from "Hail Alley" centroid (approximate: 40°N, -98°W)
- Climate zones (NOAA Climate Divisions)

**New features to engineer:**

```python
# In load_data() query, add:
"""
county_geo AS (
    SELECT 
        fips,
        latitude,
        longitude,
        -- Distance from Hail Alley centroid (Nebraska/Kansas)
        SQRT(POW(latitude - 40.0, 2) + POW(longitude + 98.0, 2)) as distance_from_hail_alley,
        -- Latitude bins (hail risk peaks in mid-latitudes)
        CASE 
            WHEN latitude < 35 THEN 'southern'
            WHEN latitude < 40 THEN 'mid_southern'
            WHEN latitude < 43 THEN 'plains'  -- Peak hail zone
            WHEN latitude < 48 THEN 'northern'
            ELSE 'far_northern'
        END as latitude_zone,
        -- Continental vs coastal (hail is interior phenomenon)
        distance_to_coast_km,
        CASE 
            WHEN distance_to_coast_km < 100 THEN 'coastal'
            WHEN distance_to_coast_km < 300 THEN 'near_interior'
            ELSE 'deep_interior'
        END as continentality
    FROM ref.county_geography  -- Would need to create
)
"""
```

**Expected improvement:** +8-12 F1 points

### **Phase 2: Add Agricultural Exposure Features** (High Impact)

73% of hail damage is crop damage. Agricultural counties are more vulnerable.

**Data sources:**
- USDA Census of Agriculture (already available in some form)
- ACS data on farming employment
- County land use data

**New features to engineer:**

```python
# In load_data() query, add:
"""
agricultural_exposure AS (
    SELECT
        fips,
        -- From ACS or USDA data
        farming_employment_pct,  -- % employed in agriculture
        cropland_acres_per_capita,
        farm_income_pct,  -- % income from farming
        -- Agricultural land use
        CASE 
            WHEN farming_employment_pct > 15 THEN 'high_ag'
            WHEN farming_employment_pct > 5 THEN 'moderate_ag'
            ELSE 'low_ag'
        END as agriculture_intensity,
        -- Proxy: Rural counties have higher ag exposure
        population_density,
        CASE 
            WHEN population_density < 50 THEN 'rural'
            WHEN population_density < 200 THEN 'mixed'
            ELSE 'urban'
        END as urbanization
    FROM mart.acs_county_demographic_annual
    WHERE year BETWEEN {min_year} AND {max_year}
    GROUP BY fips
)
"""
```

**Alternative if agricultural data unavailable:**
Use population density and rural classification as proxies:
- Rural counties: More agricultural exposure → higher hail vulnerability
- Urban counties: Less cropland but more property → different damage profile

**Expected improvement:** +5-8 F1 points

### **Phase 3: Add Historical Hail Event Features** (High Impact)

Use existing `mart.noaa_storm_events` to create county-level hail exposure history.

**New features to engineer:**

```python
# In load_data() query, add:
"""
hail_event_history AS (
    SELECT
        fips,
        -- Hail events over last 10 years
        COUNT(*) as hail_event_count,
        -- Total damage from hail
        SUM(total_damage_amount) as total_hail_damage,
        SUM(property_damage_amount) as property_hail_damage,
        SUM(crop_damage_amount) as crop_hail_damage,
        -- Average damage per event
        AVG(total_damage_amount) as avg_hail_damage,
        -- Crop vs property damage ratio
        SUM(crop_damage_amount) / NULLIF(SUM(total_damage_amount), 0) as crop_damage_ratio,
        -- Large hail events (magnitude > 2 inches)
        SUM(CASE WHEN CAST(magnitude AS DOUBLE) >= 2.0 THEN 1 ELSE 0 END) as large_hail_event_count,
        -- Severe hail events (magnitude > 4 inches - "softball size")
        SUM(CASE WHEN CAST(magnitude AS DOUBLE) >= 4.0 THEN 1 ELSE 0 END) as severe_hail_event_count,
        -- Injury/death counts (human impact)
        SUM(injuries_direct_count + deaths_direct_count) as hail_casualty_count,
        -- Peak season concentration (Apr-Jun events / total events)
        SUM(CASE WHEN event_month IN (4, 5, 6) THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as spring_hail_concentration
    FROM mart.noaa_storm_events
    WHERE event_type = 'Hail'
        AND year BETWEEN {min_year - 10} AND {max_year}
    GROUP BY fips
)
"""
```

**Expected improvement:** +6-10 F1 points

### **Phase 4: Add Seasonal Weather Features** (Medium Impact)

Use `mart.ncei_county_weather_monthly` to create spring weather patterns that predict hail risk.

**New features to engineer:**

```python
# In load_data() query, add:
"""
spring_weather_patterns AS (
    SELECT
        fips,
        -- Spring months (Mar-Jun) weather over last 5 years
        AVG(CASE WHEN month IN (3, 4, 5, 6) THEN precipitation_inches END) as avg_spring_precip,
        MAX(CASE WHEN month IN (3, 4, 5, 6) THEN precipitation_inches END) as max_spring_precip,
        -- Spring temperature variability (cold fronts meeting warm air)
        STDDEV(CASE WHEN month IN (3, 4, 5, 6) THEN avg_temperature_f END) as spring_temp_variability,
        -- Average spring temperature (warmer springs = more convective energy)
        AVG(CASE WHEN month IN (3, 4, 5, 6) THEN avg_temperature_f END) as avg_spring_temp,
        -- Precipitation anomalies (wet springs = more storms)
        AVG(CASE WHEN month IN (3, 4, 5, 6) THEN precipitation_anomaly_inches END) as spring_precip_anomaly,
        -- Temperature swings (daily max-min range proxy for frontal activity)
        AVG(CASE WHEN month IN (3, 4, 5, 6) THEN (max_temperature_f - min_temperature_f) END) as spring_temp_range,
        -- Extreme precipitation days (proxy for severe weather)
        SUM(CASE WHEN month IN (3, 4, 5, 6) AND precipitation_inches > 2.0 THEN 1 ELSE 0 END) as spring_heavy_precip_days
    FROM mart.ncei_county_weather_monthly
    WHERE year BETWEEN {min_year - 5} AND {max_year}
    GROUP BY fips
)
"""
```

**Expected improvement:** +4-6 F1 points

### **Phase 5: Add Tornado Correlation Features** (Medium Impact)

Hail and tornadoes often co-occur (same supercell storms). Leverage existing tornado data.

**New features to engineer:**

```python
# In load_data() query, add:
"""
tornado_correlation AS (
    SELECT
        fips,
        -- Tornado risk rating from NRI (if available)
        n.TRND_RISKR as tornado_risk_rating,
        n.TRND_RISKS as tornado_risk_score,
        -- Historical tornado events
        COUNT(s.event_id) as tornado_event_count,
        SUM(s.total_damage_amount) as tornado_damage,
        -- Days with both hail and tornado reports (same day, same county)
        COUNT(DISTINCT CASE 
            WHEN h.event_id IS NOT NULL THEN s.begin_date_time 
        END) as hail_tornado_cooccurrence_days
    FROM mart.nri_county_risk n
    LEFT JOIN mart.noaa_storm_events s ON n.fips = s.fips 
        AND s.event_type IN ('Tornado', 'Funnel Cloud')
        AND s.year BETWEEN {min_year - 10} AND {max_year}
    LEFT JOIN mart.noaa_storm_events h ON n.fips = h.fips
        AND h.event_type = 'Hail'
        AND DATE(h.begin_timestamp) = DATE(s.begin_timestamp)
    GROUP BY fips, n.TRND_RISKR, n.TRND_RISKS
)
"""
```

**Expected improvement:** +3-5 F1 points

### **Phase 6: Model Architecture Changes** (Low-Medium Impact)

Current models may underfit hail's geographic clustering and seasonal patterns.

1. **Add interaction features** for geography × weather:
   ```python
   from sklearn.preprocessing import PolynomialFeatures
   poly = PolynomialFeatures(degree=2, interaction_only=True)
   # Interactions: latitude × spring_precip, distance_from_hail_alley × spring_temp_variability
   ```

2. **Use XGBoost** instead of sklearn GradientBoosting:
   ```python
   import xgboost as xgb
   model = xgb.XGBClassifier(
       n_estimators=200,
       max_depth=10,
       learning_rate=0.05,
       objective='multi:softmax',
       eval_metric='mlogloss'
   )
   ```

3. **Increase tree depth** for complex geographic patterns:
   ```python
   'max_depth': [20, 30, 50],  # Current: [10, 20, 30, None]
   'n_estimators': [200, 300, 400],  # Current: [100, 200, 300]
   ```

4. **Ensemble predictions** across models:
   ```python
   # Weighted average of predictions
   final_pred = 0.4 * gb_pred + 0.3 * rf_pred + 0.2 * xgb_pred + 0.1 * lr_pred
   ```

**Expected improvement:** +2-4 F1 points

## Implementation Plan

### Quick Win (2-3 hours)
**Add historical hail events from existing `mart.noaa_storm_events`:**
- No new data download needed
- Aggregate hail events, damage, and magnitude by county over 10-year window
- Add crop vs property damage ratios

**Expected: +6-10 F1 points** (single highest-impact addition)

### High-Value Addition (3-4 hours)
**Add spring weather patterns from existing `mart.ncei_county_weather_monthly`:**
- No new data download needed
- Aggregate spring (Mar-Jun) precipitation, temperature variability
- Add extreme weather day counts

**Expected: +4-6 F1 points**

### Medium-Value Addition (2-3 hours)
**Add geographic features (latitude, distance from Hail Alley):**
- Extract lat/long from Census TIGER or lookup table
- Calculate distance from Nebraska/Kansas centroid
- Add latitude zones and continentality bins

**Expected: +8-12 F1 points**

### Long-Term Enhancement (6-8 hours)
**Add agricultural exposure from ACS or USDA:**
- Extract farming employment, cropland acres
- Add rural/urban classification from population density
- Link to crop damage vulnerability

**Expected: +5-8 F1 points**

### Advanced Enhancement (4-6 hours)
**Add tornado correlation features:**
- Use existing tornado NRI ratings and NOAA events
- Calculate hail-tornado co-occurrence days
- Create severe weather clustering features

**Expected: +3-5 F1 points**

## Total Expected Improvement

**Baseline:** 0.55 F1 (expected without improvements)  
**+ Phase 1 (Geographic):** 0.63 F1 (+8 points)  
**+ Phase 2 (Agricultural):** 0.69 F1 (+6 points)  
**+ Phase 3 (Historical Events):** 0.77 F1 (+8 points)  
**+ Phase 4 (Spring Weather):** 0.82 F1 (+5 points)  
**+ Phase 5 (Tornado Correlation):** 0.86 F1 (+4 points)  
**+ Phase 6 (Model Architecture):** 0.89 F1 (+3 points)  

**Target:** **0.75-0.85 F1** (matching or exceeding Riverine Flooding performance)

## Why These Features Matter

### Historical Hail Events
**Direct measurement** of actual exposure. If a county has experienced 50 hail events and $10M in damage over 10 years, it reveals vulnerability that NRI ratings are trying to predict. This is the **strongest single signal**.

### Geographic Features
Hail risk is **geographically clustered**. Counties in Nebraska have fundamentally different hail risk than counties in Florida, regardless of economic factors. Distance from "Hail Alley" (Great Plains) is a strong proxy.

### Spring Weather Patterns
Hail forms in **specific atmospheric conditions**: warm moist air colliding with cold fronts in spring. Spring precipitation + temperature variability directly predicts hail season intensity.

### Agricultural Exposure
73% of hail damage is **crop damage**. Agricultural counties have higher vulnerability and different damage profiles than urban counties. Farming employment and cropland percentage capture this.

### Tornado Correlation
Hail and tornadoes often come from the **same supercell storms**. Tornado risk ratings and co-occurrence patterns help predict hail risk, especially for severe events.

## Comparison to Other Hazards

| Feature Type | Hurricane | Winter Weather | **Hail** | Wildfire | Flood |
|--------------|-----------|----------------|----------|----------|-------|
| **Geographic clustering** | Coastal only | Northern/Mountain | **Plains/Midwest** | West/Southeast | River basins |
| **Seasonality** | Jun-Nov | Dec-Feb | **Apr-Jun (strong)** | Year-round | Spring/Fall |
| **Agricultural impact** | Low | Low | **Very High (73%)** | Moderate | Moderate |
| **Historical event data** | Good | Good | **Excellent (88K events)** | Good | Good |
| **Weather feature relevance** | Sea surface temp | Winter temp | **Spring precip/temp** | Drought index | Precip/snowmelt |

**Hail advantages over Winter Weather:**
1. **Stronger geographic signal:** Concentrated in specific regions vs dispersed nationwide
2. **Better historical data:** 88K events with damage/magnitude vs mixed winter event types
3. **Clearer seasonality:** 60% in 3-month window vs spread over 4+ months
4. **Agricultural proxy:** Farming counties = higher vulnerability (measurable)

**Hail is more predictable than Winter Weather** because:
- Geographic pattern is tighter (Plains/Midwest vs entire North)
- Temporal pattern is sharper (spring vs all winter)
- Historical event data is cleaner (single event type vs 8+ winter types)
- Agricultural exposure is a strong, measurable proxy

## Code Changes Required

### 1. Update `load_data()` query
Add new CTEs for:
- `hail_event_history` (from `mart.noaa_storm_events`)
- `spring_weather_patterns` (from `mart.ncei_county_weather_monthly`)
- `county_geo` (from new `ref.county_geography` or inline calculation)
- `agricultural_exposure` (from ACS demographic tables)
- `tornado_correlation` (from `mart.nri_county_risk` + NOAA events)

### 2. Update `engineer_features()`
Add binning/encoding for:
- `latitude_zone` (categorical)
- `continentality` (coastal vs interior)
- `agriculture_intensity` (high/moderate/low ag exposure)
- `urbanization` (rural/mixed/urban)

### 3. Update `prepare_features()`
Add new feature columns to `base_features` list:
- All continuous features (distance_from_hail_alley, hail_event_count, avg_spring_precip, etc.)
- All engineered categorical features (latitude_zone, agriculture_intensity, etc.)

### 4. Create data pipeline additions
If geographic data doesn't exist:
- Create `ref.county_geography` table with lat/long/elevation/distance_to_coast
- Add to `build-database` pipeline

## Alternative: Hazard-Specific Feature Engineering

Instead of using the same features for all hazards, create **hazard-specific feature sets**:

```python
class ClimateRiskPredictor:
    def get_hazard_features(self, hazard_type: str) -> List[str]:
        """Return hazard-specific feature set."""
        base_features = [
            'median_household_income', 
            'housing_burden_30pct',
            'insurance_premium',
            'property_taxes_utilities'
        ]
        
        if hazard_type == 'HAIL':
            return base_features + [
                # Geographic
                'latitude', 'longitude', 'distance_from_hail_alley',
                # Historical events
                'hail_event_count', 'total_hail_damage', 'crop_damage_ratio',
                # Spring weather
                'avg_spring_precip', 'spring_temp_variability',
                # Agricultural
                'farming_employment_pct', 'population_density',
                # Tornado correlation
                'tornado_risk_score', 'hail_tornado_cooccurrence_days'
            ]
        elif hazard_type == 'ERQK':
            return base_features + [
                # Geographic
                'latitude', 'longitude', 'distance_from_fault_lines',
                # Building characteristics
                'median_home_age', 'construction_quality_proxy',
                # Historical seismic activity
                'earthquake_event_count', 'total_earthquake_damage'
            ]
        # ... etc for other hazards
```

This allows hail model to use spring weather features while earthquake model uses seismic features.

## Validation Strategy

After adding features:

1. **Feature importance ranking:** Are new features in top 10?
   ```python
   # After training Random Forest
   importances = model.feature_importances_
   top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
   ```

2. **Geographic validation:** Does the model correctly rank Plains counties higher?
   ```python
   predictions_df['state'] = predictions_df['fips'].str[:2]
   predictions_df.groupby('state')['predicted_risk'].value_counts(normalize=True)
   # Expect: NE, KS, OK, IA have higher predicted risk than FL, CA, WA
   ```

3. **Seasonal validation:** Do counties with spring weather patterns get higher risk?
   ```python
   predictions_df['high_spring_precip'] = predictions_df['avg_spring_precip'] > median_spring_precip
   predictions_df.groupby('high_spring_precip')['predicted_risk'].mean()
   # Expect: High spring precip → higher predicted risk
   ```

4. **Agricultural validation:** Do farming counties get higher predictions?
   ```python
   predictions_df['rural'] = predictions_df['population_density'] < 50
   predictions_df.groupby('rural')['predicted_risk'].value_counts(normalize=True)
   # Expect: Rural counties → higher predicted risk
   ```

5. **Historical validation:** Do counties with past hail damage get higher predictions?
   ```python
   predictions_df['has_hail_history'] = predictions_df['hail_event_count'] > 0
   predictions_df.groupby('has_hail_history')['predicted_risk'].value_counts(normalize=True)
   # Expect: Counties with hail history → much higher predicted risk
   ```

## Next Steps - Priority Order

1. **Priority 1 (Highest ROI):** Add historical hail events from existing NOAA data
   - No new data needed, table exists
   - Strongest single predictor
   - Expected +6-10 F1 points in 2-3 hours

2. **Priority 2:** Add spring weather patterns from existing NCEI data
   - No new data needed, table exists
   - Captures atmospheric conditions
   - Expected +4-6 F1 points in 3-4 hours

3. **Priority 3:** Add geographic features (lat/long, distance from Hail Alley)
   - Minimal data acquisition (lookup table or simple calculation)
   - Captures regional clustering
   - Expected +8-12 F1 points in 2-3 hours

4. **Priority 4:** Add agricultural exposure from existing ACS data
   - May already have farming employment in demographic tables
   - Can use population density as proxy if not
   - Expected +5-8 F1 points in 6-8 hours

5. **Priority 5:** Add tornado correlation from existing NRI + NOAA data
   - No new data needed
   - Captures co-occurrence patterns
   - Expected +3-5 F1 points in 4-6 hours

**Recommended first step:** Implement Priority 1 + Priority 2 together (5-7 hours total) for **+10-16 F1 points** using only existing database tables.

Would you like me to implement the historical hail events and spring weather features first?
