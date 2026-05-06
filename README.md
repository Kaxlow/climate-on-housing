# Analyzing the Impact of Climate on Housing

This is a work-in-progress page that visualizes the impact of climate on the housing market across counties. There are multiple views that examine how different housing market metrics are affected by different types of extreme climate events across the months before and after the event’s occurrence, and how housing market movements vary by county attributes. The underlying pipeline for querying, processing, and transforming raw data runs on Python. The page is rendered with HTML, CSS, and JavaScript code.

## Visualizations
Visualizations of the impact of climate on the housing market: https://kaxlow.github.io/climate-on-housing/output/visualizations/index.html

Viewing the impact of climate on the housing market in the context of county profiles: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-on-housing.html

County-level housing market responses to extreme climate events by NRI risk ratings: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-1.html

County-level housing market responses by year: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-2.html

Clustering counties by housing market response: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-3.html 

## Clustering Scripts

The clustering logic used by `src/climate-housing-analysis.ipynb` lives in two importable Python modules:

- `src/cluster_county_econ_demographics.py` clusters counties by economic and demographic characteristics. It selects the best model across KMeans and Ward agglomerative candidates, writes `output/visualizations/county_profiles.csv`, `output/visualizations/county_profile_assignments.csv`, `output/visualizations/county_profile_model.joblib`, and returns county profile labels for the notebook to merge into `housing_df`.
- `src/cluster_housing_yoy_responses.py` clusters counties by housing-market YOY response for each incident type. It evaluates Ward agglomerative and KMeans candidates, selects the best model by silhouette score per incident type, writes response cluster assignments, summaries, and comparison tables, and returns labels/interpreted cluster descriptions for the notebook visualization export step.

Run the notebook from `src/` or the repository root. The notebook adds `src/` to `sys.path`, calls both scripts after the source data is cleaned, and applies the returned cluster labels back to county housing rows before writing visualization artifacts.
