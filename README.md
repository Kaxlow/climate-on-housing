# Analyzing the Impact of Climate on Housing

This is a work-in-progress page that visualizes the impact of climate on the housing market across counties. There are multiple views that examine how different housing market metrics are affected by different types of extreme climate events across the months before and after the event’s occurrence, and how housing market movements vary by county attributes. The underlying pipeline for querying, processing, and transforming raw data runs on Python. The page is rendered with HTML, CSS, and JavaScript code.

## Visualizations
Visualizations of the impact of climate on the housing market: https://kaxlow.github.io/climate-on-housing/output/visualizations/index.html

Viewing the impact of climate on the housing market in the context of county profiles: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-on-housing.html

County-level housing market responses to extreme climate events by NRI risk ratings: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-1.html

County-level housing market responses by year: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-2.html

Clustering counties by housing market response: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-3.html

Pre-incident housing market strength tiers: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-4.html

## Page Data Pipeline

The current HTML pages are backed by page-specific notebooks that call `src/climate_housing_page_utils.py`:

- `src/climate-housing-index.ipynb` builds the data assets used by `output/visualizations/index.html`.
- `src/climate-on-housing-page.ipynb` builds the data assets used by `output/visualizations/climate-on-housing.html`.
- `src/climate-housing-story-1.ipynb` builds the data assets used by `output/visualizations/climate-housing-story-1.html`.
- `src/climate-housing-story-2.ipynb` builds the data assets used by `output/visualizations/climate-housing-story-2.html`.
- `src/climate-housing-story-3.ipynb` builds the county-summary and response-cluster assets used by `output/visualizations/climate-housing-story-3.html`.
- `src/climate-housing-story-4.ipynb` builds the Story 2 income-window data plus pre-incident market-strength tier assets used by `output/visualizations/climate-housing-story-4.html`.

Each notebook calls `build_and_serve(page, html_file)`, which exports the relevant CSV and JSON files under `output/visualizations/`, updates `incident_housing_manifest.json`, and serves the matching HTML page locally. Run the page-specific notebook from `src/` or the repository root.

`src/climate-housing-analysis.ipynb` is a legacy monolithic analysis notebook. It is not the current build entrypoint for the HTML pages.

## Clustering Scripts

The reusable clustering logic lives in these importable Python modules:

- `src/cluster_county_econ_demographics.py` clusters counties by economic and demographic characteristics. It selects the best model across KMeans and Ward agglomerative candidates and writes `output/visualizations/county_profiles.csv`, `output/visualizations/county_profile_assignments.csv`, and `output/visualizations/county_profile_model.joblib`.
- `src/cluster_housing_yoy_responses.py` clusters counties by housing-market YOY response for each incident type. It evaluates Ward agglomerative and KMeans candidates, selects the best model by silhouette score per incident type, and writes response cluster assignments, summaries, and comparison tables.
- `src/cluster_pre_incident_market_strength.py` clusters incident counties into pre-incident market-strength tiers and writes the story 4 tier assets.
