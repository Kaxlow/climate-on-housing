# Climate Risk to Housing

This repository builds an infographic:

`output/climate-risk-housing.html`

The page reads its source-of-truth data from `data/quoll.duckdb`, uses county
boundaries from `data/fipsgeo/us_counties_boundaries_shapefile.json`, and embeds
the resulting data directly into the generated HTML.

## Build

Install the package once:

```bash
pip install -e .
```

Rebuild the infographic:

```powershell
build-climate-risk-housing
```

The workflow is:

1. Data preparation and analysis under `scripts/` populate the DuckDB marts.
2. `build-database` can rebuild `data/quoll.duckdb` from the retained source data.
3. `src/housing_climate_risk/page_data/event_windows.py` constructs reusable
   climate-event housing windows.
4. `src/housing_climate_risk/page_data/climate_risk_housing.py` queries the marts and
   writes the self-contained HTML deliverable.