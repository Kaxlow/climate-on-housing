# County Clustering Processes

This document describes the county clustering workflows used in the project,
with emphasis on the profile outputs currently used by `stormhouse.html`.

The clustering workflows are exploratory profile-building tools. They are meant
to group counties into useful peer groups based on similar attributes, then
summarize how those groups line up with FEMA National Risk Index (NRI) risk
ratings.

## Current Stormhouse Workflows

`stormhouse.html` currently uses two county profile workflows:

1. Economic profiles
2. Insurance profiles

Both workflows follow the same broad modeling pattern:

1. Build a county-level feature matrix.
2. Clean, winsorize where appropriate, and impute missing values.
3. Normalize feature distributions with `QuantileTransformer`.
4. Reduce dimensionality with PCA.
5. Fit Gaussian mixture candidates.
6. Select the best 4-5 cluster solution using balance, assignment confidence,
   and separation metrics.
7. Assign every county to its most likely profile.
8. Cross-tab profile assignments against NRI risk rating.
9. Compute direct feature lifts for each NRI risk group.
10. Save CSV/model outputs and inject a summarized payload into
    `stormhouse_data.js`.

## Economic Profile Clustering

Implementation:

- `src/housing_climate_risk/modeling/economic_risk_profiles.py`
- wired into `src/housing_climate_risk/page_data/stormhouse.py`
- output directory: `output/visualizations/stormhouse_economic_profiles/`

### Feature Matrix

The economic profile matrix is built from the county economic and population
panel used by `county_profiles.py`.

Feature families used for clustering include:

- population size
- income per person
- employment per resident
- weekly wages
- transfer receipts share
- investment and rent income share
- business-owner income share
- natural increase
- domestic and international migration
- selected 10-year trends

Demographic features are not used to assign or label economic profiles. Age,
sex, race, and ethnicity shares are summarized only after counties have already
been assigned to an economic profile.

For most base features, the matrix includes:

- 10-year average
- latest value
- selected trend/slope features

### Model

The current Stormhouse economic run uses:

- imputation: median
- scaling: quantile normalization to a normal distribution
- dimensionality reduction: PCA, up to 8 components
- clustering: GaussianMixture with diagonal covariance
- candidate `k`: 4 and 5

The current selected solution is `k=5`.

Final economic profiles:

- High-wage investment-income counties
- Smaller mixed-economy counties
- Small transfer-reliant counties
- Large high-wage metro counties
- Larger average-economy counties

### Outputs

The workflow writes:

- `stormhouse_economic_profile_assignments.csv`
- `stormhouse_economic_profile_summary.csv`
- `stormhouse_nri_rating_economic_lifts.csv`
- `stormhouse_economic_profile_labels.csv`
- `stormhouse_economic_profile_scores.csv`
- `stormhouse_economic_profile_model.joblib`

Key fields:

- `economic_profile`: numeric cluster id
- `economic_profile_label`: plain label for the cluster
- `assignment_confidence`: GMM posterior probability for the assigned profile
- `assignment_margin`: difference between best and second-best profile
- `second_best_profile`: next most likely profile
- `second_best_profile_label`: plain label for the next most likely profile
- `demographic_description`: post-assignment demographic description; these
  demographics are not clustering inputs

## Housing Insurance Profile Clustering

Implementation:

- `src/housing_climate_risk/modeling/insurance_risk_profiles.py`
- wired into `src/housing_climate_risk/page_data/stormhouse.py`
- output directory: `output/visualizations/stormhouse_insurance_profiles/`

### Feature Matrix

The housing insurance profile matrix is built from
`data/county_processed_data.feather`.

Feature families include:

- home-insurance premium level
- home-insurance premium level percentiles
- recent/average home-insurance premium level
- home-insurance premium growth
- home-insurance premium growth percentiles
- home-insurance premium trend/slope
- home-insurance premium volatility
- home-insurance nonrenewal rate
- home-insurance nonrenewal-rate percentiles
- home-insurance nonrenewal-rate growth
- home-insurance nonrenewal-rate growth percentiles
- home-insurance nonrenewal-rate trend/slope
- home-insurance nonrenewal-rate volatility

The feature filter intentionally excludes:

- policy counts
- coverage totals
- property values
- property-tax fields
- NFIP flood-insurance claims/payments

This keeps the model focused on housing insurance premium and nonrenewal
pressure rather than broader housing cost burden, flood-claim history, or market
size.

### Model

The current Stormhouse insurance run uses:

- winsorization: 1st to 99th percentile clipping
- imputation: median
- scaling: quantile normalization to a normal distribution
- dimensionality reduction: PCA, up to 8 components
- clustering: GaussianMixture with diagonal covariance
- candidate `k`: 4 and 5

The current selected solution is `k=5`.

Final insurance profiles:

- Fast-rising lower-premium counties
- Stable low-volatility insurance counties
- Typical lower-premium counties
- High-premium slower-growth counties
- High-premium high-nonrenewal counties

### Current Cluster Quality

For the selected `k=5` insurance model:

- mean assignment confidence: about `0.963`
- median assignment confidence: about `0.998`
- low-confidence rate under 0.60: about `1.8%`
- silhouette score: about `0.117`
- Davies-Bouldin index: about `2.898`
- cluster sizes: `319`, `468`, `1302`, `725`, `320`

Interpretation:

- Assignment confidence is strong, so most counties have a clear best profile
  under the GMM.
- The silhouette score is modest, which is expected for county socioeconomic
  and insurance data. These profiles are overlapping gradients, not perfectly
  separated natural classes.
- The selected `k=5` solution has stronger separation and confidence than the
  `k=4` candidate while still passing the broad balance check.

### Outputs

The workflow writes:

- `stormhouse_insurance_profile_assignments.csv`
- `stormhouse_insurance_profile_summary.csv`
- `stormhouse_nri_rating_insurance_lifts.csv`
- `stormhouse_insurance_profile_labels.csv`
- `stormhouse_insurance_profile_scores.csv`
- `stormhouse_insurance_profile_model.joblib`

Key fields:

- `insurance_profile`: numeric cluster id
- `insurance_profile_label`: plain label for the cluster
- `assignment_confidence`: GMM posterior probability for the assigned profile
- `assignment_margin`: difference between best and second-best profile
- `second_best_profile`: next most likely profile
- `second_best_profile_label`: plain label for the next most likely profile

## NRI Risk Group Summaries

After county profiles are assigned, each workflow compares the profiles against
FEMA NRI risk rating groups:

- Very Low
- Low
- Moderate
- High
- Very High

The Stormhouse page uses three related views:

1. Summary cards
   - show the most common profiles in each NRI risk group
   - show the plainest high/low traits for that risk group

2. Stacked bar chart
   - shows the full profile mix within each NRI risk group
   - hover text includes profile share and average assignment confidence

3. Feature contrast table
   - compares each NRI risk group median with the national county median
   - reports traits that are unusually high or low after standardization

## Feature Lifts

Feature lift is computed as:

```text
(risk_group_median - national_county_median) / national_county_standard_deviation
```

So:

- `+1.0` means the risk group median is one national county standard deviation
  above the national county median.
- `-1.0` means it is one national county standard deviation below.
- Values near zero mean the risk group is close to the national county median
  for that feature.

Stormhouse usually displays these in plain terms:

- slightly higher/lower than typical
- higher/lower than typical
- much higher/lower than typical

## Assignment Confidence

The Gaussian mixture model returns a probability for each possible cluster.
`assignment_confidence` is the probability assigned to the winning profile.

Example:

```text
assigned profile: High-premium high-nonrenewal counties
assignment_confidence: 0.91
```

This means the model assigns the county to that profile with 91% posterior
probability under the fitted mixture model.

`assignment_margin` is also saved. It is the gap between the best and second-best
profile probability. A high margin means the county is a cleaner example of its
assigned profile. A low margin means the county sits closer to a boundary between
two profiles.

Stormhouse shows average assignment confidence in chart and legend hover text.

## Running The Stormhouse Profile Workflows

The profile workflows are run as part of the Stormhouse page build:

```powershell
python -m housing_climate_risk.cli.build_page stormhouse
```

This regenerates:

- `output/visualizations/stormhouse.html`
- `output/visualizations/stormhouse_data.js`
- economic profile outputs
- insurance profile outputs

The build may emit a nonfatal joblib warning if Windows `wmic` is unavailable.
The warning affects CPU-core detection only; the outputs are still generated.

## Earlier Broad County Clustering

The repo also contains broader county clustering machinery under:

- `src/housing_climate_risk/modeling/county_clustering/`

Those workflows support broader experiments such as:

- all features without size variables
- climate and insurance features
- housing time-series features
- HDBSCAN plus KNN reassignment of noise counties
- broad PCA + GMM county profiles

Those outputs live under:

- `output/county_clustering/`

The Stormhouse sections described above are narrower, page-specific passes that
focus on interpretable economic and insurance contrasts by NRI risk rating.

## Interpretation Guidance

These clusters should be interpreted as profile groupings, not formal county
types.

Useful readings:

- Which profile is most common in each NRI risk group?
- Which profile becomes more common as NRI risk rises?
- Which features directly differ most from the national county median?
- Are assignments high-confidence or borderline?

Less useful readings:

- Treating cluster labels as precise definitions
- Assuming a profile causes NRI risk
- Comparing raw cluster ids across different runs

The plain labels are generated from each cluster's strongest feature contrasts.
If the feature matrix or `k` changes, profile ids and labels may change.
