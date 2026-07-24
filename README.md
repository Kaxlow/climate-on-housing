# Climate Risk to Housing

This repository builds the infographic: [Are Climate Risks Priced Into Housing Markets?](https://kaxlow.github.io/climate-on-housing/output/climate-risk-housing.html)

The page is developed from climate and county data across the United States. Data is loaded into and read from a database in the form of `data/quoll.duckdb`. County boundaries come from `data/fipsgeo/us_counties_boundaries_shapefile.json`. After cleaning, preparing, and transforming the data, the analysis results are published in the form of a story describing how climate risk is relevant to homeowners across the country.

## Methodology

See [Climate Risk and Housing Methodology](docs/climate-risk-housing-methodology.md)
for the data sources, risk and housing definitions, disaster event selection,
event-window construction, within-risk-group feature analysis, assumptions, and
limitations behind the infographic.

See [County Relative Median PPSF YoY Modeling](docs/county-relative-ppsf-modeling.md)
for the separate county-level Elastic Net and gradient-boosted tree comparison,
cross-validation design, results, limitations, and downstream model contract.

## Build

Install the package once:

```bash
pip install -e .
```

Rebuild the infographic:

```powershell
build-climate-risk-housing
```

Train the separate county relative Median PPSF YoY models with:

```powershell
train-county-relative-ppsf --n-jobs 1
```

This modeling command writes four model artifacts and evaluation results to
`output/models/county_relative_ppsf/`. High and Very High counties share one
pooled model. The command does not modify the infographic; the page builder
consumes the resulting feature-importance and county-modeling artifacts.

This page-only command uses the existing `data/quoll.duckdb`. To rebuild both the
database and infographic, run:

```powershell
build-database
build-climate-risk-housing
```

## Infographic Pipeline

1. **Retain source data.** Provider extracts under `data/` supply county identifiers,
   Redfin housing history, FEMA National Risk Index ratings, FEMA and NOAA events,
   NCEI weather, ACS characteristics, and StatsAmerica economic and migration data.
   Notebooks and utilities under `scripts/` support cleaning, validation, and EDA;
   they are not invoked by the page builder.

2. **Build DuckDB.** `build-database` runs
   `src/housing_climate_risk/cli/build_database.py`. It loads the retained files into
   the `raw` schema, records file and ACS metadata, creates county reference tables,
   and materializes the normalized `mart` tables in `data/quoll.duckdb`. The FEMA
   declaration mart represents unique county-incidents while preserving the
   declaration-level source rows in `raw`. Use
   `build-database --marts-only` when the raw tables are already current and only the
   reference and mart layers need rebuilding.

3. **Query page datasets.** `build-climate-risk-housing` runs
   `src/housing_climate_risk/page_data/climate_risk_housing.py` and opens DuckDB in
   read-only mode. The builder creates the county price/risk histories, risk-group
   feature comparisons, and Climate Playbook data directly from the marts.

4. **Construct event windows.** During the page build, helpers in
   `src/housing_climate_risk/page_data/event_windows.py` combine FEMA declarations and
   qualifying NOAA events with monthly Redfin observations. They align each county's
   housing history from 12 months before event start through the configured post-event
   horizons, then retain complete observations for the grouped plots.

5. **Attach geography.** The builder reads
   `data/fipsgeo/us_counties_boundaries_shapefile.json`, keeps the counties represented
   in the page data, and adds their GeoJSON boundaries to the page payload.

6. **Generate the deliverable.** The builder writes the infographic to
   `output/climate-risk-housing.html` and writes its deferred county-history and
   Climate Playbook payloads beside it as
   `output/climate-risk-housing-county-history.js` and
   `output/climate-risk-housing-playbook.js`. Keep all three files together when
   publishing or opening the page. The page queries no database at runtime; D3 and
   Google Fonts remain external browser resources.
