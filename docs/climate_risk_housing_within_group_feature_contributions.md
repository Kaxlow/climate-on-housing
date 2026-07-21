# Within-Group Feature Contributions in Stormhouse 2

This document describes the methodology behind **County features most correlated with relative position within its risk group** in `climate-risk-housing.html`.

## Analysis population

The page uses affected county-event records from the DuckDB marts for events from 2016 through 2025. Median PPSF YoY is measured over the complete event window from 12 months before event start through 36 months after event end. Records without a complete window are excluded.

Each county-event receives an average Median PPSF YoY across that window. Within each FEMA NRI risk-rating group, its relative position is:

```text
relative_position = county_event_average_ppsf_yoy - risk_group_median_average_ppsf_yoy
```

A positive value means the county-event is above its risk group's median; a negative value means it is below.

## Feature correlations

For every NRI risk group and candidate county feature, the builder computes Spearman's rank correlation between the feature and `relative_position`. The calculation follows the **Correlations By NRI Risk Group** section of `scripts/eda/county_nri_risk_group_position_correlations.ipynb`:

- Correlations are calculated separately for each risk group.
- The analysis grain is county-event, matching the notebook.
- At least 10 valid paired observations are required. A paired observation is one county-event record in which both the county feature value and the event window's `relative_position` are present and numeric.
- Features are sorted by the absolute value of Spearman's correlation coefficient.
- The 10 strongest features are displayed for each risk group.

The candidate features match the notebook's county feature set, with three exclusions: Median PPSF YoY because it is the outcome, Homeownership Cost Share by the page's existing specification, and Accom. & Food Wages % Total Wages because the page specification excludes it from this correlation analysis.

## Example county eligibility

The page displays one example county above and one below the Median PPSF YoY median for each NRI risk group. Before those examples are selected, a county must have a non-null value for every feature in its risk group's top-10 correlation list. This completeness rule is applied in addition to the event-window coverage, IQR-band distance, and PPSF YoY outlier criteria. It prevents the selected county profile from showing missing values for any of the features used to explain its relative position.

## County contribution marker

The marker combines the selected county's feature magnitude with the feature's within-group correlation.

First, the county's feature value is percentile-ranked among unique counties in the selected NRI risk group. Average ranks are used for ties. The percentile is centered and scaled to the range `[-1, 1]`:

```text
centered_feature_rank = 2 * feature_percentile - 1
```

where `feature_percentile` is expressed from 0 to 1. The contribution score is:

```text
contribution = spearman_r * centered_feature_rank
```

The score is displayed on a scale from `-1` to `1`:

- Negative contribution: associated with lower relative Median PPSF YoY.
- Zero contribution: neutral on this rank-correlation scale.
- Positive contribution: associated with higher relative Median PPSF YoY.

This construction accounts for both requested dimensions. A stronger absolute correlation moves the marker farther from the center, and a feature value farther from the within-group median also moves it farther from the center. The sign correctly reverses when a high feature value is associated with lower PPSF YoY, or a low feature value is associated with higher PPSF YoY.

## Interpretation limits

The marker is an association score, not a causal estimate, model coefficient, or predicted percentage-point change in Median PPSF YoY. Correlated features may overlap in what they measure, repeated county-event observations can give counties affected by more events more weight in the Spearman calculation, and missing feature values reduce the observations available for a correlation.

## Data sources

The builder reads from DuckDB marts including FEMA disaster declarations, NOAA storm events, Redfin county monthly housing data, FEMA NRI county risk, ACS county economic and demographic data, StatsAmerica county data, and NCEI county weather data. The exact mart-to-feature mapping is defined in `src/housing_climate_risk/page_data/climate_risk_housing.py`.
