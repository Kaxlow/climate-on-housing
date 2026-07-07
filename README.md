# Analyzing the Impact of Climate on Housing

This project aims to show the impact of climate on the housing market across counties. There are multiple views that examine how different housing market metrics are affected by different types of extreme climate events across the months before and after the event’s occurrence, and how housing market movements vary by county attributes. The underlying pipeline for querying, processing, and transforming raw data runs on Python. The pages are rendered with HTML, CSS, and JavaScript code.

## Which Way the Wind Blows: How Extreme Climate Events Matter To Housing Markets
https://kaxlow.github.io/climate-on-housing/output/visualizations/stormhouse-2.html

## Additional Visualizations
Visualizations of the impact of climate on the housing market: https://kaxlow.github.io/climate-on-housing/output/visualizations/index.html

County-level housing market responses to extreme climate events by NRI risk ratings: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-1.html

County-level housing market responses by year: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-2.html

Clustering counties by housing market response: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-3.html

Pre-incident housing market strength tiers: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-4.html

County-level housing market responses by income group: https://kaxlow.github.io/climate-on-housing/output/visualizations/climate-housing-story-5.html

## Page Data Pipeline

Shared source loading lives under `src/housing_climate_risk/data_sources/`. Raw readers and cached file access are in `raw.py`; prepared reusable datasets are exposed from `processed.py`. New analyses should import those shared loaders instead of reading source CSVs directly.

The database build pipeline is the first step for pages that read from the shared DuckDB store. `build-database` loads raw source files into `data/quoll.duckdb`, creates `raw`, `ref`, `meta`, and `mart` schemas, and standardizes county, housing, disaster, weather, and ACS tables used by the visualizations. During raw loading, fields that should not contain negative values are converted to `NULL`; signed fields such as YoY changes, anomalies, temperatures, and coordinates are preserved.

```bash
pip install -e .
build-database
```

To rebuild only the reference and mart tables from already loaded raw tables:

```bash
build-database --marts-only
```

Page-specific data builders live under `src/housing_climate_risk/page_data/`. The registry in `registry.py` maps page names such as `story-5` and `stormhouse` to the correct builder. Registered page builders write static artifacts under `output/visualizations/`.

After the database and shared source files are current, build a page with:

```bash
build-page story-5
```

To rebuild and serve one page locally, use:

```bash
build-page stormhouse --serve
```

To rebuild every registered page bundle:

```bash
build-page all
```

Use `pip install -e .` once from the repository root to make `housing_climate_risk` importable and install the CLI commands, including `build-database`, `build-page`, the data download commands, and the clustering runners.

## Clustering Scripts

The reusable clustering logic lives in these importable Python modules:

- `src/housing_climate_risk/modeling/county_profiles.py` clusters counties by economic and demographic characteristics. It selects the best model across KMeans and Ward agglomerative candidates and writes `output/visualizations/county_profiles.csv`, `output/visualizations/county_profile_assignments.csv`, and `output/visualizations/county_profile_model.joblib`.
- `src/housing_climate_risk/modeling/housing_response_clusters.py` clusters counties by housing-market YOY response for each incident type. It evaluates Ward agglomerative and KMeans candidates, selects the best model by silhouette score per incident type, and writes response cluster assignments, summaries, and comparison tables.
- `src/housing_climate_risk/modeling/pre_incident_market_strength.py` clusters incident counties into pre-incident market-strength tiers and writes the story 4 tier assets.
