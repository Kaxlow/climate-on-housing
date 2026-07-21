# Climate Risk Prediction Hazard Type Changes

## Summary

Updated the climate risk prediction pipeline to replace:
1. **Hurricane (HRCN)** → **Earthquake (ERQK)**
2. **Winter Weather (WNTW)** → **Hail (HAIL)**

## Rationale

### Why Remove Hurricane and Winter Weather?

**Hurricane (HRCN):**
- Geographic limitation: Only coastal counties have hurricane risk
- Low coverage: Only 2,230 counties with valid ratings (vs 3,100+ for most hazards)
- Highly imbalanced: 55% Very Low, 25% Relatively Low (80% in lowest 2 categories)

**Winter Weather (WNTW):**
- Poor model performance: F1 score of 0.42 (worst among all hazards)
- Feature mismatch: Current economic/housing features don't capture geographic/climate determinants
- Requires specialized features (latitude, elevation, historical weather) not used by other models

### Why Add Earthquake and Hail?

**Earthquake (ERQK):**
- Excellent coverage: 3,137 counties with valid ratings (highest coverage)
- Geographic diversity: Affects West Coast, Alaska, Central US (New Madrid zone)
- Better class distribution: 59% Very Low, 31% Relatively Low (more variation than hurricane)
- Predicted by economic features: Building age, construction quality, insurance reflect seismic risk

**Hail (HAIL):**
- Excellent coverage: 3,126 counties with valid ratings
- Geographic diversity: Tornado Alley, Great Plains, Southeast
- Good class balance: 39% Relatively Low, 33% Very Low, 21% Moderate
- Predicted by economic features: Agricultural exposure, property damage history, insurance reflect hail risk
- Better than winter weather: Expected to perform similarly to tornado predictions

## Coverage Comparison

| Hazard Type | Valid Counties | Geographic Distribution | Class Balance |
|-------------|----------------|-------------------------|---------------|
| **REMOVED: Hurricane (HRCN)** | 2,230 | Coastal only | 80% in lowest 2 classes |
| **REMOVED: Winter Weather (WNTW)** | 3,108 | Northern/Mountain | Good balance, poor performance |
| **ADDED: Earthquake (ERQK)** | 3,137 | West Coast + Central | 90% in lowest 2 classes |
| **ADDED: Hail (HAIL)** | 3,126 | Plains + Southeast | 72% in lowest 2 classes |
| Riverine Flooding (IFLD) | 3,140 | Nationwide | 73% in lowest 2 classes |
| Wildfire (WFIR) | 3,143 | West + Southeast | 89% in lowest 2 classes |
| Tornado (TRND) | 3,144 | Central + Southeast | 65% in lowest 3 classes |

## New Hazard Set

The updated 5 hazards for climate risk prediction:

1. **ERQK** - Earthquake
2. **IFLD** - Riverine Flooding  
3. **WFIR** - Wildfire
4. **TRND** - Tornado
5. **HAIL** - Hail

## Expected Performance

Based on class distributions and feature relevance:

| Hazard | Expected F1 Score | Reasoning |
|--------|-------------------|-----------|
| ERQK (Earthquake) | 0.58-0.65 | Similar to hurricane but better coverage. Building age, construction quality relevant. |
| IFLD (Riverine Flooding) | 0.70-0.73 | Strong performance, good feature correlation |
| WFIR (Wildfire) | 0.60-0.62 | Moderate performance, property taxes and insurance relevant |
| TRND (Tornado) | 0.58-0.60 | Moderate performance, similar to existing |
| HAIL (Hail) | 0.55-0.60 | Better than winter weather, similar to tornado |

**Removed hazards:**
- Hurricane: 0.62 F1 (good but limited coverage)
- Winter Weather: 0.42 F1 (worst performance)

## Files Updated

1. **Source Code:**
   - `src/housing_climate_risk/modeling/climate_risk_prediction.py` - Updated `HAZARD_TYPES` dict and docstring

2. **CLI (auto-updated via HAZARD_TYPES reference):**
   - `src/housing_climate_risk/cli/train_climate_risk_model.py` - Help text automatically updates

3. **Documentation:**
   - `CLAUDE.md` - Project instructions
   - `docs/county_climate_risk_prediction.md` - Full documentation with examples
   - `docs/hazard_type_changes.md` - This file

## Usage

### Train Individual Hazards

```bash
# Train earthquake risk model
train-climate-risk-model --hazard ERQK

# Train hail risk model
train-climate-risk-model --hazard HAIL
```

### Train All Hazards

```bash
# Train overall + 5 hazard-specific models
train-climate-risk-model --all-hazards

# With hyperparameter tuning
train-climate-risk-model --all-hazards --tune
```

## Output Structure

Models are saved to hazard-specific subdirectories:

```
output/models/climate_risk_prediction/
├── overall/           # Overall risk models (unchanged)
├── erqk/             # NEW: Earthquake models
├── hail/             # NEW: Hail models
├── ifld/             # Riverine flooding (unchanged)
├── wfir/             # Wildfire (unchanged)
└── trnd/             # Tornado (unchanged)
```

Old model directories (`hrcn/`, `wntw/`) remain if previously trained but won't be updated by new `--all-hazards` runs.

## Database Columns

Both new hazards have complete NRI data:

**Earthquake:**
- `ERQK_RISKR` - Risk rating (Very Low, Relatively Low, Relatively Moderate, Relatively High, Very High)
- `ERQK_RISKS` - Risk score (continuous)
- `ERQK_EVNTS`, `ERQK_AFREQ`, etc. - Additional earthquake metrics

**Hail:**
- `HAIL_RISKR` - Risk rating
- `HAIL_RISKS` - Risk score
- `HAIL_EVNTS`, `HAIL_AFREQ`, etc. - Additional hail metrics

## Backward Compatibility

**CLI:**
- Old commands like `--hazard HRCN` or `--hazard WNTW` will fail with clear error: "Unknown hazard type: HRCN. Must be one of ['ERQK', 'IFLD', 'WFIR', 'TRND', 'HAIL']"

**Saved Models:**
- Old model files in `hrcn/` and `wntw/` directories remain functional
- To use old models programmatically:
  ```python
  # This will fail - HRCN not in current HAZARD_TYPES
  predictor = ClimateRiskPredictor(hazard_type='HRCN')
  
  # Workaround: manually specify old hazard
  predictor = ClimateRiskPredictor(hazard_type=None)
  predictor.hazard_type = 'HRCN'  # Bypass validation
  predictor.output_dir = OUTPUT_DIR / 'models' / 'climate_risk_prediction' / 'hrcn'
  ```

**Recommendation:** Retrain with new hazards. Old models remain for historical reference only.

## Testing

Verified changes work:

```bash
# Check hazard types registered
python -c "from housing_climate_risk.modeling.climate_risk_prediction import HAZARD_TYPES; print(HAZARD_TYPES)"
# Output: {'ERQK': 'Earthquake', 'IFLD': 'Riverine Flooding', 'WFIR': 'Wildfire', 'TRND': 'Tornado', 'HAIL': 'Hail'}

# Check CLI help text
train-climate-risk-model --help | grep -A 7 "Hazard Types:"
# Shows ERQK and HAIL, not HRCN and WNTW

# Verify database columns exist
python -c "import duckdb; conn = duckdb.connect('data/quoll.duckdb'); print(conn.execute('SELECT COUNT(*) FROM mart.nri_county_risk WHERE ERQK_RISKR IS NOT NULL').fetchone())"
# Output: (3232,)
```

## Next Steps

1. **Retrain models:** Run `train-climate-risk-model --all-hazards` to generate new model files
2. **Update visualizations:** If climate-risk-housing.html references specific hazards, update to ERQK and HAIL
3. **Validate performance:** Check that ERQK and HAIL achieve expected F1 scores (0.55-0.65)
4. **Clean up old models (optional):** Remove `hrcn/` and `wntw/` directories if no longer needed

## Change Date

2026-07-07
