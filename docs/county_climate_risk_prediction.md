# Climate Risk Prediction Models

This module provides machine learning models to predict county-level climate risk ratings based on economic, housing, and demographic features.

## Overview

**Target Variable**: FEMA National Risk Index (NRI) Risk Rating
- Very Low
- Relatively Low  
- Relatively Moderate
- Relatively High
- Very High

**Input Features**:
1. **Income** - Median household income
2. **Housing Burden** - Share of households with housing costs ≥ 30% of income
3. **Insurance** - Mean homeowner insurance premium
4. **Property Taxes & Utilities** - Median owner costs with mortgage
5. **Net Migration** - County net migration rate
6. **Homes Sold YOY** - Year-over-year change in homes sold
7. **Median Days on Market YOY** - Year-over-year change in days on market

**Derived Features**:
- Insurance burden (premium as % of income)
- Housing cost burden (costs as % of income)
- Market cooling indicator (DOM increase - sales decrease)
- Market stress (burden × cooling)
- Burden-migration interaction (high burden × low migration)
- Insurance-market interaction (premium × market timing)

## Models Supported

The pipeline supports four model types with configurable hyperparameters:

1. **Logistic Regression** - Baseline linear classifier
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Sequential boosted trees
4. **Neural Network** - Multi-layer perceptron

All models use:
- Balanced class weights to handle imbalanced risk ratings
- 5-fold cross-validation for training evaluation
- Train/test split for final evaluation
- StandardScaler for feature normalization
- Optional GridSearchCV for hyperparameter tuning

## Usage

### Command Line Interface

Train all models with default parameters:

```bash
train-climate-risk-model
```

Train with hyperparameter tuning:

```bash
train-climate-risk-model --tune
```

Train a specific model:

```bash
train-climate-risk-model --model random_forest
```

Customize data years:

```bash
train-climate-risk-model --min-year 2020 --max-year 2023
```

Full options:

```bash
train-climate-risk-model --help
```

### Python API

```python
from housing_climate_risk.modeling.climate_risk_prediction import ClimateRiskPredictor

# Initialize predictor
predictor = ClimateRiskPredictor()

# Load data
df = predictor.load_data(min_year=2021, max_year=2023)

# Train all models
results = predictor.train_all_models(df, tune_hyperparams=False)

# Save models
predictor.save_models()

# Get best model
best_name, best_model, best_f1 = predictor.get_best_model()
print(f"Best: {best_name} with F1={best_f1:.4f}")
```

Train a single model:

```python
from housing_climate_risk.modeling.climate_risk_prediction import ClimateRiskPredictor

predictor = ClimateRiskPredictor()
df = predictor.load_data()

# Engineer features
df_engineered = predictor.engineer_features(df)
X, y, feature_names = predictor.prepare_features(df_engineered)

# Prepare data
from sklearn.model_selection import train_test_split
y_encoded = predictor.label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_train_scaled = predictor.scaler.fit_transform(X_train)
X_test_scaled = predictor.scaler.transform(X_test)

# Train model
model, train_metrics = predictor.train_model(
    'random_forest',
    X_train_scaled,
    y_train,
    tune_hyperparams=True
)

# Evaluate
eval_metrics = predictor.evaluate_model(
    model, X_test_scaled, y_test, 'random_forest'
)
```

### Feature Engineering

Use the standalone feature engineering module:

```python
from housing_climate_risk.modeling.preprocessing import FeatureEngineer

engineer = FeatureEngineer()

# Fit and transform training data
df_train_features = engineer.fit_transform(df_train)

# Transform test data with same parameters
df_test_features = engineer.transform(df_test)
```

## Configuration

Model configurations are stored in `configs/model_config.yaml`. This file controls:

- Data year ranges
- Feature engineering parameters (income brackets, burden levels)
- Model hyperparameters (default and grid search ranges)
- Evaluation metrics
- Output settings

To add a new model:

1. Add the model configuration to `model_config.yaml`
2. Add the model class to `ClimateRiskPredictor.MODEL_CONFIGS` in `climate_risk_prediction.py`
3. Define default parameters and param_grid for tuning

To adjust existing model parameters:

1. Edit `default_params` in `model_config.yaml` for the target model
2. No code changes needed

## Output

Training produces several artifacts in `output/models/climate_risk_prediction/`:

- `{model_name}_{timestamp}.joblib` - Trained model
- `scaler_{timestamp}.joblib` - Fitted StandardScaler
- `label_encoder_{timestamp}.joblib` - Fitted LabelEncoder
- `results_{timestamp}.json` - Full evaluation results
- `feature_names_{timestamp}.json` - Feature names and label classes

Results JSON contains:
- Feature names
- Label classes (risk ratings)
- Per-model metrics:
  - Cross-validation scores (mean, std)
  - Test accuracy and F1 scores
  - Full classification report
  - Confusion matrix
  - Top 10 feature importances (for tree models)

## Model Evaluation Metrics

Models are evaluated using:

1. **Cross-Validation F1 Score** (training set, 5-fold)
   - Weighted average across classes
   - Accounts for class imbalance
   - Mean and standard deviation reported

2. **Test Set Accuracy**
   - Overall classification accuracy
   - Percentage of correct predictions

3. **Test Set F1 Score** (weighted)
   - Harmonic mean of precision and recall
   - Weighted by class support
   - Primary metric for model selection

4. **Classification Report**
   - Per-class precision, recall, F1
   - Macro and weighted averages

5. **Confusion Matrix**
   - True vs predicted risk ratings
   - Shows misclassification patterns

6. **Feature Importance** (tree models only)
   - Top 10 most predictive features
   - Gini importance or permutation importance

## Data Requirements

The pipeline requires the following database tables:

- `mart.nri_county_risk` - FEMA NRI risk ratings (target)
- `mart.acs_county_affordability_annual` - Income, burden, costs
- `mart.insurance_premiums_annual` - Insurance premiums
- `mart.acs_county_demographic_annual` - Net migration
- `mart.redfin_county_monthly` - Housing market metrics

Run `build-database` to ensure all required tables exist.

## Testing

Run the test suite to verify the pipeline:

```bash
python tests/test_climate_risk_prediction.py
```

Tests cover:
- Data loading from database
- Feature engineering
- Model training (single model, no tuning)
- Model persistence (save/load)

## Advanced Usage

### Custom Hyperparameters

Pass custom parameters to override defaults:

```python
custom_params = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 5
}

model, metrics = predictor.train_model(
    'random_forest',
    X_train_scaled,
    y_train,
    custom_params=custom_params
)
```

### Loading Saved Models

```python
import joblib

# Load model
model = joblib.load('output/models/climate_risk_prediction/random_forest_20260706_123456.joblib')

# Load preprocessing artifacts
scaler = joblib.load('output/models/climate_risk_prediction/scaler_20260706_123456.joblib')
label_encoder = joblib.load('output/models/climate_risk_prediction/label_encoder_20260706_123456.joblib')

# Make predictions
X_new_scaled = scaler.transform(X_new)
y_pred_encoded = model.predict(X_new_scaled)
y_pred = label_encoder.inverse_transform(y_pred_encoded)
```

### Feature Metadata

Get interpretable feature descriptions:

```python
from housing_climate_risk.modeling.preprocessing import get_feature_metadata

metadata = get_feature_metadata(feature_names)

for feature, info in metadata.items():
    print(f"{feature}: {info['description']} ({info.get('unit', 'N/A')})")
```

## Architecture

```
modeling/
├── climate_risk_prediction.py   # Main modeling pipeline
├── preprocessing.py              # Feature engineering utilities
└── README_CLIMATE_RISK_PREDICTION.md

cli/
└── train_climate_risk_model.py  # CLI entry point

configs/
└── model_config.yaml             # Model configurations

tests/
└── test_climate_risk_prediction.py  # Unit tests

output/models/climate_risk_prediction/  # Trained models and results
```

## Future Enhancements

Potential improvements:

1. **Additional Models**
   - XGBoost
   - LightGBM
   - Support Vector Machines
   - Ensemble voting classifier

2. **Feature Engineering**
   - Temporal features (lag values, trends)
   - Spatial features (neighboring county risk)
   - Climate event frequency features
   - Insurance availability indicators

3. **Model Interpretation**
   - SHAP values for feature importance
   - Partial dependence plots
   - Individual prediction explanations

4. **Production Deployment**
   - Model serving API
   - Batch prediction pipeline
   - Model monitoring and retraining

5. **Hyperparameter Tuning**
   - Bayesian optimization
   - Random search
   - Early stopping for neural networks
