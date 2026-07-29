# County Relative Median PPSF YoY Modeling

## Purpose

This component predicts a county's relative position within its FEMA National
Risk Index (NRI) risk group in terms of Median Price-Per-Square-Foot (PPSF)
year-over-year growth.

The component compares Elastic Net with gradient-boosted regression trees and
selects the model with the lowest repeated nested cross-validation mean absolute
error (MAE) for four model populations:

1. Very Low
2. Low
3. Medium
4. High + Very High

High and Very High counties are pooled because the Very High group has only 17
eligible counties, which is insufficient for a stable standalone tree model.
Each county retains its original NRI group, and the response remains centered
on that original group's median before pooling. The pooled model therefore
learns a shared feature-to-relative-PPSF relationship without redefining a
Very High county as High.

These models estimate predictive associations. They do not establish that a
county feature causes housing-price growth to change.

### High + Very High Interaction Design

The pooled model receives the catalog-selected base county predictors plus
`is_very_high`, which equals one for Very High counties and zero for High
counties. After fold-local median imputation, the pipeline appends one
predictor-by-`is_very_high` interaction for every selected base predictor.
The exact input count is recorded in the model manifest because catalog
retention and correlation pruning can change it.

1. Catalog-selected shared main effects.
2. One Very High sample indicator.
3. One interaction per base predictor, permitting each predictor's relationship with the
   target to differ for Very High counties.

The transformation occurs inside the fitted pipeline after imputation, so
validation-fold values do not affect the imputed inputs or interactions.

## Source Code

- Dataset construction:
  `src/housing_climate_risk/modeling/county_relative_ppsf/data.py`
- Training and evaluation:
  `src/housing_climate_risk/modeling/county_relative_ppsf/train.py`
- Command-line entry point:
  `src/housing_climate_risk/modeling/county_relative_ppsf/cli.py`
- Tests:
  `tests/test_county_relative_ppsf.py`

## Unit of Observation

The modeling table contains exactly one row per county FIPS code. Source tables
are aggregated at their native frequency before being joined, preventing annual
features from being duplicated across monthly Redfin rows.

Counties must have:

- An assigned NRI risk group.
- A non-null Median PPSF YoY target.
- At least 60 valid monthly Median PPSF YoY observations in the latest 10
  calendar years.

The current dataset covers January 2016 through December 2025 and contains
2,689 counties:

| Risk group | Counties |
|---|---:|
| Very Low | 987 |
| Low | 1,123 |
| Medium | 404 |
| High | 156 |
| Very High | 17 |

## Target

For county \(i\) in NRI risk group \(g\):

\[
y_i =
\operatorname{median}_{t}(\text{Median PPSF YoY}_{i,t})
-
\operatorname{median}_{j \in g}
\left[
\operatorname{median}_{t}(\text{Median PPSF YoY}_{j,t})
\right]
\]

The target is stored as `relative_median_ppsf_yoy`. A value of `0.02` means the
county's 10-year median monthly PPSF YoY is two percentage points above the
median eligible county in its risk group.

The group median is calculated only after applying the housing-history
eligibility rule. The target itself is never included as a predictor.

## Predictors

`feature.catalog` is the authoritative candidate-feature list. Dataset
construction reads rows where `retained = true` and aggregates their corresponding
numeric `feature.*` columns over the latest ten calendar years.

Three retained fields have technical model roles and are not predictors:

- `median_ppsf_yoy` is the prediction target.
- `housing_market_index` contains the PPSF target and would leak it.
- `risk_rating` is the categorical grouping field.

Within each model population, candidates must have at least three valid values,
the configured minimum non-null coverage, and more than one unique value.
Candidates are then considered in deterministic catalog order; a feature is
excluded when the absolute value of its Spearman correlation with an already
selected feature is at least 0.85. The saved `feature_coverage.csv` records the
retained partner and signed observed correlation for every such exclusion.
Missing selected values are median-imputed inside each training fold, preventing
validation-fold values from affecting preprocessing.

Requested replacement features—combined event count, the three ownership-cost
shares, and the three BEA income components—are considered first during
correlation pruning. This ensures an older context feature is removed when it
is redundant with the replacement it was meant to supersede.

## Candidate Models

### Elastic Net

Pipeline:

1. Median imputation.
2. Standardization of predictors.
3. Standardization of the target inside `TransformedTargetRegressor`.
4. Elastic Net regression.

Inner-fold tuning searches:

- `alpha`: \(10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\)
- `l1_ratio`: 0.1, 0.5, 0.9, 1.0

### Gradient-Boosted Trees

Pipeline:

1. Median imputation.
2. `GradientBoostingRegressor` with Huber loss.

Each inner search evaluates 12 deterministic random parameter combinations
from:

- 75, 125, 200, or 300 trees.
- Learning rate 0.02, 0.05, or 0.10.
- Maximum depth 1, 2, or 3.
- Minimum leaf size 3, 5, 10, or 20.
- Subsample 0.7, 0.9, or 1.0.

## Evaluation Design

Model comparison uses repeated nested K-fold cross-validation independently
within each model population. High and Very High counties are evaluated
together as one 173-county population.

- Outer evaluation: three shuffled repeats.
- Outer folds: five for every current model population.
- Inner tuning: up to three shuffled folds within every outer training fold.
- Selection metric: mean outer-fold MAE.
- Tie breaker: mean outer-fold RMSE.
- Baseline: median target value from the outer training fold.
- Additional metrics: RMSE, \(R^2\), and Spearman rank correlation.

Every county receives one held-out prediction in each outer repeat. The saved
out-of-fold prediction is the mean of those three predictions. The final model
is tuned and refit on all counties in its model population only after
model-family selection.

Nested evaluation is necessary because reporting the same inner-fold scores
used for hyperparameter selection would produce optimistic results.

## Production Run Results

The run used random seed `20260723`, three outer repeats, five outer folds,
three inner folds, and the pooled High + Very High population.

| Model population | Selected model | MAE (pp) | RMSE (pp) | R² | Spearman | Baseline MAE (pp) | MAE improvement |
|---|---|---:|---:|---:|---:|---:|---:|
| Very Low | Elastic Net | 2.764 | 3.908 | 0.048 | 0.310 | 2.906 | 4.9% |
| Low | Gradient-boosted trees | 1.690 | 2.500 | 0.180 | 0.511 | 2.013 | 16.1% |
| Medium | Gradient-boosted trees | 1.257 | 1.643 | 0.210 | 0.486 | 1.442 | 12.8% |
| High + Very High | Gradient-boosted trees | 1.181 | 1.496 | 0.188 | 0.446 | 1.324 | 10.8% |

`pp` means percentage points of Median PPSF YoY.

The selected pooled model for the High and Very High risk groups produces distinct fitted and out-of-fold predictions. Its sample indicator and predictor interactions allow the fitted
relationship to differ between High and Very High counties. All four selected
models outperform their held-out median baseline, but their \(R^2\) values
remain modest. Their predictions are appropriate for exploratory ranking and
explanation, not high-confidence forecasts.

## Saved Artifacts

The production run writes to `output/models/county_relative_ppsf/`:

- `manifest.json`: data definition, evaluation configuration, selected models,
  features, hyperparameters, and model paths.
- `county_modeling_dataset.parquet`: one-row-per-county modeling snapshot.
- `evaluation_summary.csv`: repeated outer-fold model comparison.
- `fold_metrics.csv`: every repeat/fold score and inner-fold best parameters.
- `county_predictions.csv`: observed target, repeated out-of-fold prediction,
  residual, and final fitted prediction.
- `feature_coverage.csv`: included and excluded features by risk group.
- `feature_importance.csv`: selected-model coefficients or tree importance.
- `model_comparison.png`: MAE comparison by risk group.
- `observed_vs_oof.png`: held-out prediction diagnostic.
- `models/*.joblib`: four fitted model artifacts, including one shared
  `high_very_high.joblib`.

Feature importance is model-specific. Elastic Net coefficients are in
standardized target and predictor units. Tree impurity importance is unsigned
and can favor continuous predictors. Neither should be interpreted as a causal
effect.

For the pooled model, High and Very High importance is measured separately.
Within each original group, each base feature is permuted 20 times while its
sample indicator remains fixed. Importance is the resulting mean increase in
in-sample MAE. Negative mean increases are set to zero for ranking. This
subgroup permutation method captures the combined use of a feature's main and
interaction effects and allows the two groups to have different top-ten lists.
The 17-county Very High ranking is correspondingly less stable than the
156-county High ranking.

## Reproduction

After installing the package:

```powershell
pip install -e .
train-county-relative-ppsf --n-jobs 1
```

Without an editable install:

```powershell
python -m housing_climate_risk.modeling.county_relative_ppsf.cli --n-jobs 1
```

The default output directory is
`output/models/county_relative_ppsf`. The component does not alter or connect
to `climate-risk-housing.html`.

## Downstream Contract

Each `joblib` file contains:

- `pipeline`: fitted scikit-learn estimator.
- `model_group`: model's training population.
- `risk_groups`: original NRI groups represented by the model.
- `sample_indicator`: pooled-sample indicator input, when applicable.
- `interaction_features`: engineered predictor interactions.
- `model_name`: selected family.
- `features`: ordered required feature names.
- `target` and `target_definition`.
- `best_parameters`.

Example:

```python
import joblib

artifact = joblib.load(
    "output/models/county_relative_ppsf/models/low.joblib"
)
features = artifact["features"]
prediction = artifact["pipeline"].predict(county_frame[features])
```

The prediction is relative Median PPSF YoY in proportion units; multiply by 100
for percentage points. The infographic builder consumes the saved feature
importance and county modeling dataset but does not retrain the models.
