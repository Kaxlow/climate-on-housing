# Climate Risk to Housing

This repository builds the infographic: [Are Climate Risks Priced Into Housing Markets?](https://kaxlow.github.io/climate-on-housing/output/climate-risk-housing.html)

The page is developed from climate and county data across the United States. Data is loaded into and read from a database in the form of `data/quoll.duckdb`. County boundaries come from `data/fipsgeo/us_counties_boundaries_shapefile.json`. After cleaning, preparing, and transforming the data, the analysis results are published in the form of a story describing how climate risk is relevant to homeowners across the country.

## Methodology

See [Climate Risk and Housing Methodology](docs/climate-risk-housing-methodology.md)
for the data sources, risk and housing definitions, disaster event selection,
event-window construction, within-risk-group feature analysis, assumptions, and
limitations behind the infographic.

## Build

Install the package once:

```bash
pip install -e .
```

Populate the local data workspace from the latest available provider releases:

```powershell
download-data all
```

This command creates the required directories; downloads Redfin monthly county
housing data, Census ACS and boundaries, FEMA NRI and disaster declarations,
NOAA, and StatsAmerica inputs;
builds derived NOAA county mappings; validates filenames and schemas; and writes
the ignored local receipt `data/download_receipt.yaml`. Provider metadata and
expected schemas are defined in
[`config/data_sources.yaml`](config/data_sources.yaml).
Third-party attribution and licensing scope are summarized in
[`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md).

The Redfin inputs come from the public
[Redfin Data Center Download Hub](https://www.redfin.com/news/data-center/downloads/),
with metric definitions and revision notes documented in Redfin's
[methodology](https://www.redfin.com/news/data-center/methodology/). The raw
provider files remain ignored locally.

The bootstrap does not download the following packaged or optional inputs:

- `data/fipsgeo/fips_master_v2.csv` (comes packaged with repository)
- `data/20260401_county_processed_data/county_processed_data.feather` (to be supplied by user due to private nature. optional; adds private insurance features)

Set `CENSUS_API_KEY` before bootstrapping. The pipeline intentionally selects
the latest available data. Annual versioned provider URLs are recorded where
available; mutable APIs and unversioned downloads can change future results.
Redfin's mutable files are downloaded at their current revision, while the
housing mart retains January 2012 through December 2025 to preserve the existing
analysis period.

Rebuild the infographic:

```powershell
build-climate-risk-housing
```

This page-only command uses the existing `data/quoll.duckdb`. To rebuild both the
database and infographic, run:

```powershell
build-database
build-climate-risk-housing
```

## Infographic Pipeline

1. **Acquire source data.** Run `download-data all` to populate the ignored local
   `data/` workspace. Provider extracts supply county identifiers,
   Redfin Data Center housing history, FEMA National Risk Index ratings, FEMA and NOAA events,
   NCEI weather, ACS characteristics, and StatsAmerica economic and migration data.
   Notebooks and utilities under `scripts/` support cleaning, validation, and EDA;
   they are not invoked by the page builder.

2. **Build DuckDB.** `build-database` runs
   `src/housing_climate_risk/cli/build_database.py`. It loads the retained files into
   the `raw` schema, records file and ACS metadata, creates county reference tables,
   materializes normalized provider tables in `mart`, blends domain variables into
   the `feature` schema, and persists event-window inputs and summaries in the
   `analysis` schema in `data/quoll.duckdb`. The FEMA
   declaration mart represents unique county-incidents while preserving the
   declaration-level source rows in `raw`. Use
   `build-database --marts-only` when the raw tables are already current and only the
   reference and mart layers need rebuilding.

3. **Query page datasets.** `build-climate-risk-housing` runs
   `src/housing_climate_risk/page_data/climate_risk_housing.py` and opens DuckDB in
   read-only mode. The builder creates the county price/risk histories, risk-group
   feature comparisons, and Climate Playbook data directly from the marts.

4. **Construct event windows.** Shared production helpers in
   `src/housing_climate_risk/page_data/event_windows.py` combine FEMA declarations and
   qualifying NOAA events with monthly Redfin observations. They align each county's
   housing history from 12 months before event start through the configured post-event
   horizons, then retain complete observations for grouped plots. The database build
   persists the publication-notebook cohort and aggregate diagnostics under `analysis`.

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

## Notebooks

Publication notebooks are documented in
[notebooks/README.md](notebooks/README.md). They follow the same pipeline layers:

- `01_data_quality` audits provider data in `raw`.
- `02_feature_exploration` examines blended domain marts in `feature`.
- `03_event_window` examines persisted event-window marts in `analysis`.

Regenerate the executed notebooks with:

```powershell
pip install -e ".[notebooks]"
python -m ipykernel install --prefix .jupyter-kernels --name quoll-intelligence --display-name "Quoll Intelligence"
build-database --marts-only
python notebooks/build_notebooks.py --execute
```
