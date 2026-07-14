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

Models predict risk ratings on a 5-level ordinal scale: Very Low → Relatively Low → Relatively Moderate → Relatively High → Very High.

## Features

All models use the same 12 features, averaged over the specified year range (default 2021–2023):

| # | Feature | Source | Notes |
|---|---------|--------|-------|
| 1 | **Median Household Income of Homeowners** | ACS S2503 | Owner-occupied units only |
| 2 | **Net Resident Earnings Per Capita** | BEA via StatsAmerica | Net earnings by place of residence ÷ population |
| 3 | **Dividends, Interest, and Rent Per Capita** | BEA via StatsAmerica | Unearned income component ÷ population |
| 4 | **Transfer Receipts Per Capita** | BEA via StatsAmerica | Government transfers ÷ population |
| 5 | **Utilities as % of Income** | ACS DP04 + insurance data | Residual of monthly no-mortgage owner costs after subtracting taxes and insurance, annualised ÷ homeowner income |
| 6 | **Insurance as % of Income** | Insurance premiums + ACS | Mean annual premium ÷ homeowner income |
| 7 | **Property Taxes as % of Income** | ACS S2507 | Median annual real estate taxes (non-mortgaged owners) ÷ homeowner income |
| 8 | **Net Migration Rate** | StatsAmerica population components | Total net migration (domestic + international) per 1,000 residents |
| 9 | **Unemployment Rate** | ACS DP03 | Civilian labour force unemployment rate (%) |
| 10 | **New Listings YOY** | Redfin county monthly | Year-over-year ratio |
| 11 | **Homes Sold YOY** | Redfin county monthly | Year-over-year ratio |
| 12 | **Median Days on Market YOY** | Redfin county monthly | Absolute day delta year-over-year |

Missing feature values are imputed with the column median before training.

## Models

Four model types are trained for each risk target:

1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Boosted tree ensemble
4. **Neural Network** - Multi-layer perceptron

All models use:
- Ordinal target encoding (Very Low = 0, …, Very High = 4)
- Spatial train/test split via `GroupKFold` grouped by state FIPS
- Balanced class weights
- Spatial 5-fold cross-validation on the training set
- Standard feature scaling
- Optional grid search hyperparameter tuning

### Ordinal Encoding

Risk classes are mapped to integers that preserve their order:

| Label | Ordinal Value |
|-------|--------------|
| Very Low | 0 |
| Relatively Low | 1 |
| Relatively Moderate | 2 |
| Relatively High | 3 |
| Very High | 4 |

This ensures that predicting "Very High" when the truth is "Very Low" is penalised more than a one-step error, which nominal classification ignores.

### Spatial Cross-Validation

Training and evaluation use `GroupKFold(n_splits=5)` with state FIPS as the group key. The last fold's states form the holdout set; all counties from those states are withheld from training entirely. This prevents geographic leakage — adjacent counties that share climate exposure and demographics can no longer appear in both train and test.

Spatial CV on the training set gives an honest estimate of out-of-region generalisation, rather than the optimistic scores produced by random splits.

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

Tuning significantly increases training time (5–10×) but may improve accuracy by 1–3 percentage points.

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
- **`*_results_<timestamp>.json`**: Performance metrics, confusion matrices, feature importance
- **`*_feature_names_<timestamp>.json`**: Feature list and ordinal risk mapping

## Evaluation Metrics

Each model reports:
- **Ordinal MAE**: Mean absolute error on the 0–4 scale — the primary ranking metric. A score of 0.8 means predictions are off by less than one risk tier on average.
- **Adjacent Accuracy**: Fraction of predictions within ±1 ordinal step of the true label.
- **Accuracy**: Fraction of exact correct predictions.
- **F1 Score (weighted)**: Harmonic mean of precision and recall, weighted by class support.
- **Spatial CV F1**: 5-fold spatial cross-validation F1 on the training set.
- **Classification report**: Per-class precision, recall, F1.
- **Confusion matrix**: Ordered by risk level (Very Low → Very High).
- **Feature importance**: Top 10 features (tree-based models only).

The best model is selected by lowest ordinal MAE.

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

Counties with "Not Applicable", "No Rating", or "Insufficient Data" are excluded.

## Key Implementation Details

### Hazard Selection Rationale

The 5 hazards were selected to align with the "Are climate risks priced into housing markets?" section of stormhouse-2.html. These hazards have:
1. **High county coverage** (>2,200 counties with valid ratings)
2. **Meaningful risk variation** (excluding "Not Applicable" ratings)
3. **Reasonable class distributions** (avoiding extreme imbalance)
4. **Relevance to housing markets** (direct property damage potential)

### Missing Value Handling

- All numeric features with missing values are imputed with the column median after feature engineering.
- Counties missing the target risk rating are excluded from training.

### Class Imbalance

All models use `class_weight='balanced'` to handle imbalanced risk rating distributions.

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
    RISK_ORDER,
    RISK_ORDER_INVERSE,
    train_all_hazards
)

# Train single hazard
predictor = ClimateRiskPredictor(hazard_type='ERQK')
df = predictor.load_data(min_year=2021, max_year=2023)
results = predictor.train_all_models(df, tune_hyperparams=False)
predictor.save_models()

# Train all hazards
all_results = train_all_hazards(tune_hyperparams=True, min_year=2020, max_year=2024)

# Access best model (selected by lowest ordinal MAE)
best_name, best_model, best_mae = predictor.get_best_model()
print(f"Best: {best_name} (Ordinal MAE: {best_mae:.4f})")
```

## Data Sources

- **FEMA NRI** (`mart.nri_county_risk`): Risk ratings and scores by hazard type
- **ACS Affordability** (`mart.acs_county_affordability_annual`):
  - Homeowner income (S2503 median household income, owner-occupied units)
  - Median annual property taxes (S2507, non-mortgaged owners)
  - Median monthly owner costs with no mortgage (DP04)
- **BEA via StatsAmerica** (`mart.statsamerica_bea_personal_income_annual`):
  - Net earnings by place of residence, dividends/interest/rent, transfer receipts (all in thousands), population
- **StatsAmerica population components** (`mart.statsamerica_population_components_annual`):
  - Total net migration (domestic + international)
- **ACS Economic** (`mart.acs_county_economic_annual`):
  - Civilian labour force unemployment rate (DP03)
- **Insurance** (`mart.insurance_premiums_annual`): Mean annual homeowner premiums
- **Redfin** (`mart.redfin_county_monthly`): New listings YOY, homes sold YOY, median days on market YOY

## Troubleshooting

**Q: Model accuracy is low**  
A: Climate risk ratings reflect many factors beyond economic/housing data (geology, weather patterns, infrastructure). The ordinal MAE is the more informative metric — a MAE below 1.0 means predictions are within one risk tier on average. To improve:
- Add hazard-specific features (elevation, precipitation, wildfire fuel)
- Use larger date ranges to capture trends
- Enable hyperparameter tuning with `--tune`

**Q: Training fails with "only X counties with data"**  
A: Check that the year range overlaps with available ACS, insurance, BEA, and housing data. Try expanding `--min-year` or `--max-year`.

**Q: Grid search takes too long**  
A: Disable tuning or train a single model type:
```bash
train-climate-risk-model --model random_forest
```

**Q: How do I use a trained model to predict new counties?**  
A: Load the saved artifacts and predict:
```python
import joblib
import numpy as np
from housing_climate_risk.modeling.climate_risk_prediction import RISK_ORDER_INVERSE

# Load artifacts (replace timestamp with actual filename)
model = joblib.load('output/models/climate_risk_prediction/erqk/erqk_random_forest_20260707_142330.joblib')
scaler = joblib.load('output/models/climate_risk_prediction/erqk/erqk_scaler_20260707_142330.joblib')

# Prepare new data (must match training feature order)
new_data = pd.DataFrame({...})  # Same 12 features as training
X_scaled = scaler.transform(new_data.values)

# Predict — returns ordinal integers 0-4
y_pred_ordinal = model.predict(X_scaled)
y_pred_labels = [RISK_ORDER_INVERSE[i] for i in y_pred_ordinal]
print(y_pred_labels)  # ['Relatively High', 'Very Low', ...]
```

## Future Enhancements

- Add hazard-specific physical features (elevation, forest cover, flood zone share, seismic zone)
- Train hierarchical models (predict overall risk from hazard-specific risks)
- Add time-series features (climate trends over 5–10 years)
- Ensemble predictions across multiple models
- Swap `GradientBoostingClassifier` for LightGBM for faster training and native ordinal support

## References

- FEMA National Risk Index: https://hazards.fema.gov/nri/
- NRI Technical Documentation: https://www.fema.gov/sites/default/files/documents/fema_national-risk-index_technical-documentation.pdf
- Census ACS: https://www.census.gov/programs-surveys/acs
