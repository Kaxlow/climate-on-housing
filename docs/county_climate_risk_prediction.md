# Climate Risk Prediction Models

Machine learning pipeline to predict FEMA NRI climate risk ratings based on county-level economic, housing, and demographic features.

## Overview

This module trains classification models to predict:
- **Overall climate risk** (FEMA NRI composite risk rating)
- **Hazard-specific risk** for 5 common hazard types (aligned with stormhouse-2.html):
  * **ERQK** - Earthquake
  * **IFLD** - Riverine Flooding
  * **WFIR** - Wildfire
  * **TRND** - Tornado
  * **HAIL** - Hail

Models predict risk ratings on a 5-level scale: Very Low, Relatively Low, Relatively Moderate, Relatively High, Very High.

## Features

All models use the same feature set:

### Base Features (7)
1. **Income** - Median household income for owner-occupied units only (Census S2503)
2. **Housing Burden** - Share of households with housing costs ≥ 30% of income
3. **Insurance Premium** - Mean homeowner insurance premium
4. **Property Taxes & Utilities** - Median owner costs with mortgage
5. **In-Migration Rate** - County total in-migration rate (domestic + international)
6. **Homes Sold YOY** - Year-over-year change in homes sold
7. **Median Days on Market YOY** - Year-over-year change in days on market

### Engineered Features (6)
8. **Insurance Burden** - Premium as % of income
9. **Housing Cost Burden** - Mortgage costs as % of income
10. **Market Cooling** - DOM YoY minus sales YoY (captures market velocity slowdown)
11. **Market Stress** - Weighted combination of housing burden and market cooling
12. **Burden-Migration Interaction** - Housing burden adjusted by in-migration tendency
13. **Insurance-Market Interaction** - Insurance premium adjusted by market velocity

## Models

Four model types are trained for each risk target:

1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Boosted tree ensemble
4. **Neural Network** - Multi-layer perceptron

All models use:
- Stratified train/test split (80/20 default)
- Balanced class weights
- 5-fold cross-validation
- Standard feature scaling
- Optional grid search hyperparameter tuning

## Usage

### Train Overall Risk Model

```bash
train-climate-risk-model
```

### Train Specific Hazard Model

```bash
# Earthquake risk
train-climate-risk-model --hazard ERQK

# Tornado risk
train-climate-risk-model --hazard TRND
```

### Train All Models (Overall + 5 Hazards)

```bash
train-climate-risk-model --all-hazards
```

### With Hyperparameter Tuning

```bash
train-climate-risk-model --all-hazards --tune
```

This runs grid search over:
- Regularization strength (Logistic Regression)
- Tree depth, sample splits, estimator count (Random Forest, Gradient Boosting)
- Network architecture, learning rate, activation (Neural Network)

Tuning significantly increases training time (5-10x) but may improve accuracy by 1-3 percentage points.

### Train Single Model Type

```bash
# Train only Random Forest for earthquake risk
train-climate-risk-model --hazard ERQK --model random_forest
```

### Custom Date Range

```bash
# Use 2020-2024 data instead of default 2021-2023
train-climate-risk-model --all-hazards --min-year 2020 --max-year 2024
```

## Output Structure

Models and artifacts are saved to `output/models/climate_risk_prediction/`:

```
output/models/climate_risk_prediction/
├── overall/                           # Overall risk models
│   ├── overall_random_forest_20260707_142315.joblib
│   ├── overall_scaler_20260707_142315.joblib
│   ├── overall_label_encoder_20260707_142315.joblib
│   ├── overall_results_20260707_142315.json
│   └── overall_feature_names_20260707_142315.json
├── erqk/                              # Earthquake-specific models
│   ├── erqk_random_forest_20260707_142330.joblib
│   └── ...
├── ifld/                              # Riverine flooding-specific models
├── wfir/                              # Wildfire-specific models
├── trnd/                              # Tornado-specific models
└── hail/                              # Hail-specific models
```

### File Types

- **`*_<model>_<timestamp>.joblib`**: Trained scikit-learn model
- **`*_scaler_<timestamp>.joblib`**: StandardScaler fitted to training data
- **`*_label_encoder_<timestamp>.joblib`**: LabelEncoder for risk rating classes
- **`*_results_<timestamp>.json`**: Performance metrics, confusion matrices, feature importance
- **`*_feature_names_<timestamp>.json`**: Feature list and label classes

## Evaluation Metrics

Each model reports:
- **Accuracy**: Fraction of correct predictions
- **F1 Score (weighted)**: Harmonic mean of precision and recall, weighted by class support
- **Cross-validation scores**: 5-fold CV F1 scores on training set
- **Classification report**: Per-class precision, recall, F1
- **Confusion matrix**: Predicted vs actual risk ratings
- **Feature importance**: Top 10 features (tree-based models only)

## County Coverage

Models are trained only on counties with valid risk ratings in the NRI dataset:

| Hazard Type | Counties | Notes |
|-------------|----------|-------|
| Overall Risk | ~2,800 | Excludes "Insufficient Data" counties |
| Earthquake (ERQK) | 3,145 | 59% Very Low, 31% Relatively Low |
| Riverine Flooding (IFLD) | 3,140 | 35% Very Low, 38% Relatively Low |
| Wildfire (WFIR) | 3,143 | 63% Very Low, 26% Relatively Low |
| Tornado (TRND) | 3,144 | 26% Very Low, 39% Relatively Low, 25% Moderate |
| Hail (HAIL) | 3,144 | 39% Relatively Low, 33% Very Low, 21% Moderate |

Counties with "Not Applicable", "No Rating", or "Insufficient Data" are excluded. This means predictions are only made for counties where the hazard is actually relevant and has been assessed.

## Key Implementation Details

### Hazard Selection Rationale

The 5 hazards were selected to align with the "Are climate risks priced into housing markets?" section of stormhouse-2.html. These hazards have:
1. **High county coverage** (>2,200 counties with valid ratings)
2. **Meaningful risk variation** (excluding "Not Applicable" ratings)
3. **Reasonable class distributions** (avoiding extreme imbalance)
4. **Relevance to housing markets** (direct property damage potential)

These hazards match the visualization used in the main StormHouse infographic, ensuring model predictions align with the public-facing analysis.

### Missing Value Handling

- Numeric features with missing values are filled with the median
- Categorical features are one-hot encoded with `drop_first=True`
- Counties missing the target risk rating are excluded from training

### Class Imbalance

All models use `class_weight='balanced'` to handle imbalanced risk rating distributions. This prevents the model from simply predicting the majority class (e.g., "Very Low").

## Example Results

Typical F1 scores (without hyperparameter tuning):

| Model Type | Overall Risk | Earthquake | Tornado | Hail |
|------------|--------------|------------|---------|------|
| Logistic Regression | 0.42 | 0.44 | 0.48 | 0.46 |
| Random Forest | 0.48 | 0.51 | 0.55 | 0.53 |
| Gradient Boosting | 0.50 | 0.53 | 0.57 | 0.55 |
| Neural Network | 0.46 | 0.49 | 0.53 | 0.51 |

*Note: These are illustrative estimates. Actual performance varies by data year range and tuning.*

## Configuration

Model hyperparameters are defined in `ClimateRiskPredictor.MODEL_CONFIGS`. To customize:

1. Edit the `default_params` dict for default hyperparameters
2. Edit the `param_grid` dict for grid search ranges

Example (in `climate_risk_prediction.py`):

```python
MODEL_CONFIGS = {
    'random_forest': {
        'default_params': {
            'n_estimators': 200,  # Increase from 100
            'max_depth': 20,      # Limit depth
            ...
        },
        'param_grid': {
            'n_estimators': [200, 300, 400],  # New search range
            ...
        }
    }
}
```

## Programmatic Usage

```python
from housing_climate_risk.modeling.climate_risk_prediction import (
    ClimateRiskPredictor,
    HAZARD_TYPES,
    train_all_hazards
)

# Train single hazard
predictor = ClimateRiskPredictor(hazard_type='ERQK')
df = predictor.load_data(min_year=2021, max_year=2023)
results = predictor.train_all_models(df, tune_hyperparams=False)
predictor.save_models()

# Train all hazards
all_results = train_all_hazards(tune_hyperparams=True, min_year=2020, max_year=2024)

# Access best model
best_name, best_model, best_f1 = predictor.get_best_model()
print(f"Best: {best_name} (F1: {best_f1:.4f})")
```

## Data Sources

- **FEMA NRI** (`mart.nri_county_risk`): Risk ratings and scores by hazard type
- **ACS Affordability** (`mart.acs_county_affordability_annual`): 
  - Homeowner income (S2503_C02_013E - median household income for owner-occupied units)
  - Housing burden, property taxes & utilities
- **ACS Demographics** (`mart.acs_county_demographic_annual`): 
  - Total in-migration rate (domestic + international, from B07001)
- **Insurance** (`mart.insurance_premiums_annual`): Mean homeowner premiums
- **Redfin** (`mart.redfin_county_monthly`): Housing market trends (sales, days on market)

All features are averaged over the specified year range (default 2021-2023).

**Key Data Improvements**:
- **Income**: Now uses owner-occupied household income instead of county-wide median, providing more accurate insurance burden calculations
- **In-Migration**: Now includes international migration in addition to domestic migration for a more complete picture of population inflows. Note: This is in-migration only, not net migration (we don't track out-migration)

## Troubleshooting

**Q: Model accuracy is low (~40% F1 score)**  
A: This is expected. Climate risk ratings reflect many factors beyond economic/housing data (geology, weather patterns, infrastructure). 40-55% F1 is a reasonable benchmark for this feature set. To improve:
- Add hazard-specific features (elevation, precipitation, wildfire fuel)
- Use larger date ranges to capture trends
- Enable hyperparameter tuning with `--tune`

**Q: Training fails with "only X counties with data"**  
A: Check that the year range overlaps with available ACS, insurance, and housing data. Some counties lack recent housing data. Try expanding `--min-year` or `--max-year`.

**Q: Grid search takes too long**  
A: Disable tuning or train a single model type:
```bash
train-climate-risk-model --model random_forest  # Skip tuning, single model
```

**Q: How do I use a trained model to predict new counties?**  
A: Load the saved artifacts and predict:
```python
import joblib
import pandas as pd
import numpy as np

# Load artifacts (replace timestamp with actual filename)
model = joblib.load('output/models/climate_risk_prediction/erqk/erqk_random_forest_20260707_142330.joblib')
scaler = joblib.load('output/models/climate_risk_prediction/erqk/erqk_scaler_20260707_142330.joblib')
encoder = joblib.load('output/models/climate_risk_prediction/erqk/erqk_label_encoder_20260707_142330.joblib')

# Prepare new data (must match training feature order)
new_data = pd.DataFrame({...})  # Same features as training
X_scaled = scaler.transform(new_data.values)

# Predict
y_pred_encoded = model.predict(X_scaled)
y_pred = encoder.inverse_transform(y_pred_encoded)
print(y_pred)  # ['Relatively High', 'Very Low', ...]
```

## Future Enhancements

Potential improvements:
- Add hazard-specific features (elevation for landslides, forest cover for wildfires)
- Train hierarchical models (predict overall risk from hazard-specific risks)
- Use regression to predict continuous risk scores instead of categorical ratings
- Add time-series features (climate trends over 5-10 years)
- Ensemble predictions across multiple models
- Spatial features (neighbor risk spillover, regional clustering)

## References

- FEMA National Risk Index: https://hazards.fema.gov/nri/
- NRI Technical Documentation: https://www.fema.gov/sites/default/files/documents/fema_national-risk-index_technical-documentation.pdf
- Census ACS: https://www.census.gov/programs-surveys/acs
