# Analysis notebooks

The publication notebooks mirror the production data pipeline. They are
explanatory and diagnostic consumers; they do not own database transformations.

| Directory | Data layer | Purpose |
| --- | --- | --- |
| `01_data_quality` | `raw.*` | One audit per provider covering supplied files and tables, grain, coverage, schema, missingness, suppression, keys, numeric ranges, limitations, and downstream readiness. |
| `02_feature_exploration` | `feature.*` | Domain-level review of blended economic, demographic, climate/hazard, and housing-market features. |
| `03_event_window` | `analysis.*` | Aggregate housing-market metrics around complete extreme-event windows, overall and by NRI risk rating. |

## Rebuild

Install the project and notebook dependencies:

```powershell
pip install -e ".[notebooks]"
python -m ipykernel install --prefix .jupyter-kernels --name quoll-intelligence --display-name "Quoll Intelligence"
```

Build the database layers and regenerate executed notebooks:

```powershell
build-database --marts-only
python notebooks/build_notebooks.py --execute
```

The local kernelspec directory is ignored by Git. To generate notebooks without
executing them, omit `--execute`.

## Ownership

- `src/housing_climate_risk/cli/build_database.py` orchestrates database builds.
- `src/housing_climate_risk/cli/feature_marts.py` defines the five domain feature
  marts and `feature.catalog`.
- `src/housing_climate_risk/cli/analysis_marts.py` defines the persisted event
  cohort, monthly county-event windows, and aggregate summaries.
- `src/housing_climate_risk/page_data/event_windows.py` owns the reusable event
  selection and temporal-alignment functions.
- `notebooks/build_notebooks.py` owns notebook presentation and diagnostics only.
