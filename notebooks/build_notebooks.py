"""Generate and optionally execute the publication notebooks.

Run from the repository root:

    python notebooks/build_notebooks.py
    python notebooks/build_notebooks.py --execute
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
LOCAL_JUPYTER_PREFIX = ROOT / ".jupyter-kernels"
os.environ["JUPYTER_PATH"] = str(LOCAL_JUPYTER_PREFIX / "share" / "jupyter")


def md(text: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def write_notebook(path: Path, cells: list, *, execute: bool) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Quoll Intelligence",
                "language": "python",
                "name": "quoll-intelligence",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    if execute:
        client = NotebookClient(
            notebook,
            timeout=900,
            kernel_name="quoll-intelligence",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    print(f"{'Executed' if execute else 'Generated'} {path.relative_to(ROOT)}")


PROVIDERS = {
    "census_acs": {
        "title": "Census ACS Raw Data Quality",
        "patterns": ["census_acs5_%"],
        "primary_patterns": [
            "census_acs5_county_affordability_%",
            "census_acs5_county_dp02_%",
            "census_acs5_county_dp03_%",
            "census_acs5_county_dp04_%",
            "census_acs5_county_dp05_%",
            "census_acs5_county_migration_%",
            "census_acs5_county_population_%",
        ],
        "key_candidates": [["year", "state", "county"]],
        "date_candidates": ["year"],
        "geo_candidates": ["state", "county"],
        "numeric_hints": ["year", "median_household_income", "total_population", "median_home_value"],
        "suppression_codes": ["-222222222", "-333333333", "-555555555", "-666666666", "-888888888", "-999999999", "(X)", "N", "-"],
        "grain": "One county-year per ACS product in data tables; dictionary, metadata, coverage, and failure tables have their own supporting grains.",
        "limitations": "ACS five-year estimates represent rolling periods, margins of error vary by county and measure, variable labels can change across releases, and suppressed or unavailable values must not be treated as zero.",
        "assessment": "PASS WITH LIMITATIONS when county-year keys are valid and suppression values are interpreted as missing. Downstream marts must preserve margins of error and release-year provenance.",
    },
    "fema": {
        "title": "FEMA Raw Data Quality",
        "patterns": ["fema_%", "nri_table_counties"],
        "primary_patterns": ["fema_disaster_declarations", "fema_web_disaster_summaries", "nri_table_counties"],
        "key_candidates": [["disasterNumber", "fipsStateCode", "fipsCountyCode", "declarationDate"], ["STCOFIPS"], ["disasterNumber"]],
        "date_candidates": ["declarationDate", "incidentBeginDate", "incidentEndDate", "lastRefresh"],
        "geo_candidates": ["state", "fipsStateCode", "fipsCountyCode", "STCOFIPS"],
        "numeric_hints": ["disasterNumber", "RISK_SCORE", "EAL_SCORE", "SOVI_SCORE", "RESL_SCORE", "totalAmountIhpApproved"],
        "suppression_codes": ["", "N/A", "null"],
        "grain": "Declarations are declaration-area records; web summaries are disaster-level financial summaries; NRI is one county snapshot per NRI release.",
        "limitations": "FEMA declarations are administrative records rather than direct measures of hazard intensity. A county incident can appear under more than one declaration, missing incident end dates require an explicit rule, and assistance totals reflect program administration and eligibility.",
        "assessment": "PASS WITH LIMITATIONS. Declaration records require county-event deduplication and incident-type exclusions before event-window use; NRI ratings are treated as a current county risk grouping.",
    },
    "fipsgeo": {
        "title": "FIPSGeo Reference Data Quality",
        "patterns": ["fips_master_v2"],
        "primary_patterns": ["fips_master_v2"],
        "key_candidates": [["fips"]],
        "date_candidates": [],
        "geo_candidates": ["fips", "state", "county_name"],
        "numeric_hints": ["latitude", "longitude"],
        "suppression_codes": ["", "null"],
        "grain": "One canonical county or county-equivalent geography per five-digit FIPS code.",
        "limitations": "County-equivalent names and boundaries can change over time. This reference is a project crosswalk, not a historical geography model.",
        "assessment": "PASS when five-digit FIPS values are unique and state/county identifiers are populated; unmatched provider geographies remain a downstream join limitation.",
    },
    "ncei": {
        "title": "NCEI Raw Data Quality",
        "patterns": ["ncei_%"],
        "primary_patterns": ["ncei_climate_at_a_glance_county_monthly"],
        "key_candidates": [["fips", "date", "parameter"]],
        "date_candidates": ["date", "year", "month", "fetched_at"],
        "geo_candidates": ["fips", "state", "county"],
        "numeric_hints": ["year", "month", "value", "anomaly", "rank"],
        "suppression_codes": ["", "null"],
        "grain": "One county-month-parameter observation from Climate at a Glance.",
        "limitations": "County values are gridded/aggregated climate summaries, parameter coverage may differ by month, and ranks/anomalies depend on the provider reference period.",
        "assessment": "PASS WITH LIMITATIONS when county-month-parameter keys are unique and the four required parameters (tavg, tmin, tmax, pcp) have adequate coverage.",
    },
    "noaa": {
        "title": "NOAA Storm Events Raw Data Quality",
        "patterns": ["noaa_%"],
        "primary_patterns": ["noaa_storm_events_county_damage", "noaa_storm_events_zone_county_mapping"],
        "key_candidates": [["event_id", "cz_type", "state_fips", "cz_fips"], ["state_fips", "cz_fips", "cz_name"]],
        "date_candidates": ["begin_date_time", "end_date_time", "begin_yearmonth", "year"],
        "geo_candidates": ["state_fips", "county_fips", "cz_fips", "cz_name", "mapped_fips"],
        "numeric_hints": ["year", "event_id", "property_damage", "crop_damage", "total_damage", "injuries_direct", "deaths_direct"],
        "suppression_codes": ["", "null"],
        "grain": "Storm Events detail records resolved to counties, plus a separate forecast-zone-to-county mapping.",
        "limitations": "Damage is reported rather than independently estimated, zero can mean no reported damage, zone events require geographic allocation, and reporting practices change over time.",
        "assessment": "PASS WITH LIMITATIONS. Only resolved county events with valid timestamps are eligible; the project extreme-event definition applies a $1 billion total-damage threshold.",
    },
    "redfin": {
        "title": "Redfin Raw Data Quality",
        "patterns": ["redfin_%"],
        "primary_patterns": ["redfin_housing_market_by_county"],
        "key_candidates": [["PERIOD_BEGIN", "REGION", "PROPERTY_TYPE"]],
        "date_candidates": ["PERIOD_BEGIN", "PERIOD_END", "LAST_UPDATED"],
        "geo_candidates": ["REGION", "STATE_CODE"],
        "numeric_hints": ["MEDIAN_SALE_PRICE", "MEDIAN_PPSF", "MEDIAN_PPSF_YOY", "HOMES_SOLD", "INVENTORY", "MEDIAN_DOM"],
        "suppression_codes": ["", "-888888888", "-999999999", "null"],
        "grain": "One county-period-property-type housing-market observation.",
        "limitations": "Redfin coverage reflects transactions and listings visible to Redfin, county histories are incomplete in some places, revision timing is provider-controlled, and YOY measures require prior-year observations.",
        "assessment": "PASS WITH LIMITATIONS for All Residential county-month analysis after FIPS resolution, numeric parsing, sentinel removal, and complete-window filtering.",
    },
    "statsamerica": {
        "title": "StatsAmerica BEA and CEW Raw Data Quality",
        "patterns": ["statsamerica_%"],
        "primary_patterns": ["statsamerica_bea_per_capita_income", "statsamerica_bea_personal_income", "statsamerica_cew_total_ownership", "statsamerica_population_components"],
        "key_candidates": [["Statefips", "Countyfips", "Year", "Linecode"], ["Statefips", "Countyfips", "Year", "NAICS Code", "Ownership Code"], ["Statefips", "Countyfips", "Year"]],
        "date_candidates": ["Year"],
        "geo_candidates": ["Statefips", "Countyfips", "Description"],
        "numeric_hints": ["Year", "Data", "BEA Per Capita Personal Income", "Employment", "Wages", "Average Wage", "Births", "Deaths"],
        "suppression_codes": ["", "(D)", "(L)", "(N)", "null"],
        "grain": "County-year-linecode for BEA personal income, county-year-industry-ownership for CEW, and county-year for per-capita income and population components.",
        "limitations": "BEA and CEW concepts have different universes and revision schedules, CEW values can be disclosure-suppressed, and exact-year joins reduce coverage when releases are not synchronized.",
        "assessment": "PASS WITH LIMITATIONS after filtering county rows, honoring disclosure flags, selecting documented BEA linecodes/CEW industries, and retaining exact source years.",
    },
}


def provider_notebook(slug: str, cfg: dict[str, object]) -> list:
    return [
        md(
            f"""
            # {cfg["title"]}

            **Purpose.** Audit the files ingested for this provider and the corresponding `raw.*`
            DuckDB tables before any normalization, blending, or analytical transformation.

            This notebook covers the supplied files/tables, observation grain, date and geography
            coverage, column types and meanings, missingness and suppression, duplicate/invalid
            keys, numeric ranges, suspicious values, source limitations, and downstream readiness.
            """
        ),
        md("## Setup and provider rules"),
        code(
            f"""
            from pathlib import Path
            import re
            import duckdb
            import numpy as np
            import pandas as pd
            from IPython.display import display

            pd.set_option("display.max_columns", 80)
            pd.set_option("display.max_rows", 120)

            ROOT = Path.cwd()
            while not (ROOT / "data" / "quoll.duckdb").exists() and ROOT != ROOT.parent:
                ROOT = ROOT.parent
            DB_PATH = ROOT / "data" / "quoll.duckdb"
            con = duckdb.connect(str(DB_PATH), read_only=True)

            PROVIDER = {slug!r}
            TABLE_PATTERNS = {cfg["patterns"]!r}
            PRIMARY_PATTERNS = {cfg["primary_patterns"]!r}
            KEY_CANDIDATES = {cfg["key_candidates"]!r}
            DATE_CANDIDATES = {cfg["date_candidates"]!r}
            GEO_CANDIDATES = {cfg["geo_candidates"]!r}
            NUMERIC_HINTS = {cfg["numeric_hints"]!r}
            SUPPRESSION_CODES = {cfg["suppression_codes"]!r}

            def matches(name, patterns):
                return any(re.fullmatch(pattern.replace("%", ".*"), name, flags=re.I) for pattern in patterns)

            def qi(value):
                return '"' + value.replace('"', '""') + '"'

            raw_tables = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'raw' ORDER BY table_name"
            ).df()["table_name"].tolist()
            provider_tables = [name for name in raw_tables if matches(name, TABLE_PATTERNS)]
            primary_tables = [name for name in provider_tables if matches(name, PRIMARY_PATTERNS)]
            provider_tables, primary_tables
            """
        ),
        md("## Files and tables supplied"),
        code(
            """
            file_inventory = con.execute(
                '''
                SELECT table_name, filename, source_folder, source_path,
                       loaded_at, row_count, detected_columns,
                       upstream_source_url, content_sha256
                FROM meta.files
                WHERE table_schema = 'raw'
                ORDER BY table_name
                '''
            ).df()
            file_inventory = file_inventory.loc[file_inventory["table_name"].isin(provider_tables)]

            table_rows = []
            for table in provider_tables:
                row_count = con.execute(f"SELECT count(*) FROM raw.{qi(table)}").fetchone()[0]
                column_count = con.execute(
                    "SELECT count(*) FROM information_schema.columns "
                    "WHERE table_schema='raw' AND table_name=?", [table]
                ).fetchone()[0]
                table_rows.append({"table_name": table, "rows": row_count, "columns": column_count,
                                   "primary_data_table": table in primary_tables})
            table_inventory = pd.DataFrame(table_rows)
            display(file_inventory)
            display(table_inventory)
            """
        ),
        md(
            f"""
            ## Observation grain

            {cfg["grain"]}

            The checks below infer candidate keys from the raw columns. A repeated candidate key is
            reported rather than silently removed because some provider tables legitimately contain
            additional dimensions.
            """
        ),
        md("## Column types and meanings"),
        code(
            """
            schema_frames = []
            for table in primary_tables:
                schema = con.execute(f"DESCRIBE raw.{qi(table)}").df()
                schema.insert(0, "table_name", table)
                schema["inferred_meaning"] = (
                    schema["column_name"].str.replace("_", " ", regex=False)
                    .str.replace(r"(?<=[a-z])(?=[A-Z])", " ", regex=True)
                    .str.strip()
                )
                schema_frames.append(schema)
            schema_inventory = pd.concat(schema_frames, ignore_index=True) if schema_frames else pd.DataFrame()
            display(schema_inventory)
            """
        ),
        md("## Date and geographic coverage"),
        code(
            """
            coverage_rows = []
            for table in primary_tables:
                columns = con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='raw' AND table_name=?", [table]
                ).df()["column_name"].tolist()
                row = {"table_name": table}
                for column in DATE_CANDIDATES:
                    if column in columns:
                        normalized_column = column.lower()
                        if normalized_column == "year" or normalized_column.endswith("_year"):
                            coverage_type = "INTEGER"
                        elif normalized_column == "month" or normalized_column.endswith("_month"):
                            coverage_type = "INTEGER"
                        else:
                            coverage_type = "TIMESTAMP"
                        values = con.execute(
                            f"SELECT min(try_cast({qi(column)} AS {coverage_type})), "
                            f"max(try_cast({qi(column)} AS {coverage_type})) "
                            f"FROM raw.{qi(table)}"
                        ).fetchone()
                        row[f"{column}_min"] = values[0]
                        row[f"{column}_max"] = values[1]
                for column in GEO_CANDIDATES:
                    if column in columns:
                        row[f"{column}_distinct"] = con.execute(
                            f"SELECT count(DISTINCT {qi(column)}) FROM raw.{qi(table)}"
                        ).fetchone()[0]
                coverage_rows.append(row)
            coverage = pd.DataFrame(coverage_rows)
            display(coverage)
            """
        ),
        md("## Missingness and suppression codes"),
        code(
            """
            missing_rows = []
            suppression_rows = []
            suppression_sql = ", ".join("?" for _ in SUPPRESSION_CODES)
            for table in primary_tables:
                columns = con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='raw' AND table_name=? ORDER BY ordinal_position", [table]
                ).df()["column_name"].tolist()
                row_count = con.execute(f"SELECT count(*) FROM raw.{qi(table)}").fetchone()[0]
                # Profile all columns for compact tables and the first 80 for unusually wide sources.
                for column in columns[:80]:
                    null_count, blank_count = con.execute(
                        f"SELECT count(*) FILTER (WHERE {qi(column)} IS NULL), "
                        f"count(*) FILTER (WHERE trim(cast({qi(column)} AS VARCHAR))='') "
                        f"FROM raw.{qi(table)}"
                    ).fetchone()
                    missing_rows.append({
                        "table_name": table, "column_name": column,
                        "missing_count": null_count + blank_count,
                        "missing_pct": (null_count + blank_count) / row_count * 100 if row_count else np.nan,
                    })
                    if SUPPRESSION_CODES:
                        suppressed = con.execute(
                            f"SELECT count(*) FROM raw.{qi(table)} "
                            f"WHERE trim(cast({qi(column)} AS VARCHAR)) IN ({suppression_sql})",
                            SUPPRESSION_CODES,
                        ).fetchone()[0]
                        if suppressed:
                            suppression_rows.append({
                                "table_name": table, "column_name": column,
                                "suppression_or_sentinel_count": suppressed,
                            })
            missingness = pd.DataFrame(missing_rows).sort_values(
                ["missing_pct", "table_name"], ascending=[False, True]
            )
            suppression = pd.DataFrame(suppression_rows)
            display(missingness)
            display(suppression if not suppression.empty else pd.DataFrame(
                {"result": ["No configured literal suppression codes were present in profiled columns; nulls remain material."]}
            ))
            """
        ),
        md("## Duplicate or invalid keys"),
        code(
            """
            key_rows = []
            for table in primary_tables:
                columns = set(con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='raw' AND table_name=?", [table]
                ).df()["column_name"])
                keys = next((candidate for candidate in KEY_CANDIDATES if set(candidate).issubset(columns)), [])
                if not keys:
                    key_rows.append({"table_name": table, "candidate_key": None,
                                     "duplicate_key_groups": np.nan, "invalid_key_rows": np.nan})
                    continue
                key_expr = ", ".join(qi(column) for column in keys)
                invalid = " OR ".join(
                    f"{qi(column)} IS NULL OR trim(cast({qi(column)} AS VARCHAR))=''" for column in keys
                )
                duplicate_groups = con.execute(
                    f"SELECT count(*) FROM (SELECT {key_expr}, count(*) n "
                    f"FROM raw.{qi(table)} GROUP BY {key_expr} HAVING count(*) > 1)"
                ).fetchone()[0]
                invalid_rows = con.execute(
                    f"SELECT count(*) FROM raw.{qi(table)} WHERE {invalid}"
                ).fetchone()[0]
                key_rows.append({"table_name": table, "candidate_key": " + ".join(keys),
                                 "duplicate_key_groups": duplicate_groups,
                                 "invalid_key_rows": invalid_rows})
            key_quality = pd.DataFrame(key_rows)
            display(key_quality)
            """
        ),
        md("## Numeric ranges and suspicious values"),
        code(
            """
            numeric_rows = []
            for table in primary_tables:
                columns = set(con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='raw' AND table_name=?", [table]
                ).df()["column_name"])
                for column in [name for name in NUMERIC_HINTS if name in columns]:
                    numeric = (
                        f"try_cast(replace(trim(cast({qi(column)} AS VARCHAR)), ',', '') AS DOUBLE)"
                    )
                    result = con.execute(
                        f"SELECT count(*) FILTER (WHERE {numeric} IS NOT NULL), "
                        f"min({numeric}), max({numeric}), "
                        f"count(*) FILTER (WHERE {numeric} < 0) "
                        f"FROM raw.{qi(table)}"
                    ).fetchone()
                    numeric_rows.append({
                        "table_name": table, "column_name": column,
                        "numeric_count": result[0], "minimum": result[1],
                        "maximum": result[2], "negative_count": result[3],
                        "review_flag": (
                            "review negative values/sentinels" if result[3] else
                            "review extreme min/max against provider definition"
                        ),
                    })
            numeric_ranges = pd.DataFrame(numeric_rows)
            display(numeric_ranges)
            """
        ),
        md(
            f"""
            ## Source-specific limitations

            {cfg["limitations"]}

            ## Downstream readiness

            **Assessment: {cfg["assessment"]}**

            This assessment is conditional on the displayed inventories and checks. The normalized
            `mart.*` builders—not this notebook—own parsing, suppression handling, geographic
            resolution, deduplication, and downstream transformations.
            """
        ),
        code("con.close()"),
    ]


FEATURE_NOTEBOOKS = {
    "economic_and_affordability_features": {
        "title": "Economic and Affordability Features",
        "tables": ["county_economic_annual"],
        "time_column": "year",
        "excluded_catalog_features": ["accommodation_food_wage_share"],
    },
    "demographic_features": {
        "title": "Demographic Features",
        "tables": ["county_demographic_annual"],
        "time_column": "year",
    },
    "climate_and_hazard_features": {
        "title": "Climate and Hazard Features",
        "tables": ["county_climate_monthly", "county_risk"],
        "time_column": "climate_month",
    },
    "housing_market_features": {
        "title": "Housing Market Features",
        "tables": ["county_housing_monthly"],
        "time_column": "housing_month",
    },
}


def feature_notebook(slug: str, cfg: dict[str, object]) -> list:
    return [
        md(
            f"""
            # {cfg["title"]}

            **Purpose.** Examine the definition, provenance, transformation, grain, joined
            completeness, distribution, outliers, redundancy, temporal alignment, retention
            decision, and infographic use of the domain features.

            This notebook is a diagnostic consumer of the `feature.*` semantic marts. Database
            construction owns every feature transformation.
            """
        ),
        md("## Setup and authoritative feature definitions"),
        code(
            f"""
            from pathlib import Path
            import duckdb
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import display

            pd.set_option("display.max_columns", 100)
            pd.set_option("display.max_rows", 120)
            sns.set_theme(style="whitegrid")

            ROOT = Path.cwd()
            while not (ROOT / "data" / "quoll.duckdb").exists() and ROOT != ROOT.parent:
                ROOT = ROOT.parent
            con = duckdb.connect(str(ROOT / "data" / "quoll.duckdb"), read_only=True)
            FEATURE_TABLES = {cfg["tables"]!r}
            EXCLUDED_CATALOG_FEATURES = {cfg.get("excluded_catalog_features", [])!r}
            TABLE_NAMES = ", ".join("?" for _ in FEATURE_TABLES)

            catalog = con.execute(
                f"SELECT * FROM feature.catalog "
                f"WHERE feature_table IN ({{TABLE_NAMES}}) "
                f"OR (feature_table IS NULL AND lower(category) = lower(?))"
                f"ORDER BY retained DESC, feature_table, feature_name",
                [*FEATURE_TABLES, {cfg["title"].replace(" Features", "")!r}],
            ).df()
            catalog = catalog.loc[
                ~catalog["feature_name"].isin(EXCLUDED_CATALOG_FEATURES)
            ].copy()
            display(catalog)
            """
        ),
        md(
            """
            The catalog above is the answer key for definition, unit, source, transformation,
            grain, retention/exclusion, and infographic use. The remaining sections test whether
            the built data match those definitions.
            """
        ),
        md("## Geographic and temporal grain"),
        code(
            """
            inventory_rows = []
            for table in FEATURE_TABLES:
                columns = con.execute(f"DESCRIBE feature.{table}").df()["column_name"].tolist()
                time_column = next((c for c in ["year", "climate_month", "housing_month"] if c in columns), None)
                time_sql = (
                    f"min({time_column}) AS min_time, max({time_column}) AS max_time, "
                    f"count(DISTINCT {time_column}) AS periods"
                    if time_column else
                    "NULL AS min_time, NULL AS max_time, NULL AS periods"
                )
                row = con.execute(
                    f"SELECT count(*) AS rows, count(DISTINCT fips) AS counties, {time_sql} "
                    f"FROM feature.{table}"
                ).df()
                row.insert(0, "feature_table", table)
                inventory_rows.append(row)
            inventory = pd.concat(inventory_rows, ignore_index=True)
            display(inventory)
            """
        ),
        md("## Completeness after joins"),
        code(
            """
            completeness_frames = []
            for table in FEATURE_TABLES:
                schema = con.execute(f"DESCRIBE feature.{table}").df()
                has_year = "year" in schema["column_name"].tolist()
                numeric_columns = schema.loc[
                    schema["column_type"].str.contains(
                        "DOUBLE|FLOAT|DECIMAL|INTEGER|BIGINT|HUGEINT", case=False, regex=True
                    ),
                    "column_name",
                ].tolist()
                feature_columns = [
                    column for column in numeric_columns
                    if column not in {"year", "month"} and not column.endswith("_count")
                ]
                row_count = con.execute(f"SELECT count(*) FROM feature.{table}").fetchone()[0]
                rows = []
                for column in feature_columns:
                    if has_year:
                        present, first_year, last_year = con.execute(
                            f'SELECT count("{column}"), '
                            f'min(year) FILTER (WHERE "{column}" IS NOT NULL), '
                            f'max(year) FILTER (WHERE "{column}" IS NOT NULL) '
                            f"FROM feature.{table}"
                        ).fetchone()
                        in_scope_rows = con.execute(
                            f"SELECT count(*) FROM feature.{table} "
                            f"WHERE year BETWEEN ? AND ?",
                            [first_year, last_year],
                        ).fetchone()[0] if first_year is not None and last_year is not None else 0
                    else:
                        present = con.execute(
                            f'SELECT count("{column}") FROM feature.{table}'
                        ).fetchone()[0]
                        first_year = last_year = None
                        in_scope_rows = row_count
                    rows.append({
                        "feature_table": table, "feature_name": column,
                        "non_null_rows": present, "total_rows": row_count,
                        "global_complete_pct": present / row_count * 100 if row_count else np.nan,
                        "first_observed_year": first_year,
                        "last_observed_year": last_year,
                        "in_scope_rows": in_scope_rows,
                        "in_scope_complete_pct": (
                            present / in_scope_rows * 100 if in_scope_rows else np.nan
                        ),
                    })
                completeness_frames.append(pd.DataFrame(rows))
            completeness = pd.concat(completeness_frames, ignore_index=True)
            display(completeness.sort_values(["feature_table", "global_complete_pct"]))
            """
        ),
        md("## Distributions, outliers, and implausible values"),
        code(
            """
            frames = []
            for table in FEATURE_TABLES:
                schema = con.execute(f"DESCRIBE feature.{table}").df()
                numeric_columns = schema.loc[
                    schema["column_type"].str.contains(
                        "DOUBLE|FLOAT|DECIMAL|INTEGER|BIGINT|HUGEINT", case=False, regex=True
                    ),
                    "column_name",
                ].tolist()
                numeric_columns = [
                    c for c in numeric_columns
                    if c not in {"year", "month"} and not c.endswith("_count")
                ]
                if not numeric_columns:
                    continue
                quoted = ", ".join(f'"{column}"' for column in numeric_columns)
                frame = con.execute(f"SELECT {quoted} FROM feature.{table}").df()
                description = frame.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.reset_index(
                    names="feature_name"
                )
                description.insert(0, "feature_table", table)
                q1 = frame.quantile(0.25)
                q3 = frame.quantile(0.75)
                iqr = q3 - q1
                outlier_counts = ((frame.lt(q1 - 1.5 * iqr)) | (frame.gt(q3 + 1.5 * iqr))).sum()
                description["iqr_outlier_count"] = description["feature_name"].map(outlier_counts)
                description["review_flag"] = np.where(
                    (description["min"] < 0)
                    & description["feature_name"].str.contains(
                        "income|wage|population|price|value|precipitation|score", case=False
                    ),
                    "negative value requires review",
                    np.where(description["iqr_outlier_count"] > 0, "distribution has IQR outliers", "none"),
                )
                frames.append(description)
            distributions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            display(distributions)
            """
        ),
        code(
            """
            plot_data = distributions.sort_values("iqr_outlier_count", ascending=False).head(12)
            if not plot_data.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=plot_data, y="feature_name", x="iqr_outlier_count", hue="feature_table", ax=ax)
                ax.set_title("Features with the most IQR-flagged observations")
                ax.set_xlabel("Observations outside 1.5 × IQR")
                ax.set_ylabel("")
                plt.tight_layout()
                plt.show()
            """
        ),
        md("## Redundancy among related features"),
        code(
            """
            correlation_frames = []
            for table in FEATURE_TABLES:
                schema = con.execute(f"DESCRIBE feature.{table}").df()
                numeric_columns = schema.loc[
                    schema["column_type"].str.contains("DOUBLE|FLOAT|DECIMAL", case=False, regex=True),
                    "column_name",
                ].tolist()
                if len(numeric_columns) < 2:
                    continue
                quoted = ", ".join(f'"{column}"' for column in numeric_columns)
                frame = con.execute(f"SELECT {quoted} FROM feature.{table}").df()
                corr = frame.corr(min_periods=100)
                pairs = (
                    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    .stack()
                    .rename("correlation")
                    .reset_index()
                    .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
                )
                pairs.insert(0, "feature_table", table)
                correlation_frames.append(pairs)
            correlations = pd.concat(correlation_frames, ignore_index=True) if correlation_frames else pd.DataFrame()
            if not correlations.empty:
                correlations["absolute_correlation"] = correlations["correlation"].abs()
                display(correlations.sort_values("absolute_correlation", ascending=False).head(30))
            else:
                display(pd.DataFrame({"result": ["Fewer than two continuous features were available."]}))
            """
        ),
        md("## Temporal alignment"),
        code(
            """
            alignment_frames = []
            for table in FEATURE_TABLES:
                columns = con.execute(f"DESCRIBE feature.{table}").df()["column_name"].tolist()
                time_column = next((c for c in ["year", "climate_month", "housing_month"] if c in columns), None)
                flags = [c for c in columns if c.startswith("has_")]
                if time_column and flags:
                    aggregates = ", ".join(
                        f'avg(cast("{flag}" AS INTEGER)) * 100 AS "{flag}_pct"' for flag in flags
                    )
                    alignment = con.execute(
                        f"SELECT {time_column}, count(DISTINCT fips) AS counties, {aggregates} "
                        f"FROM feature.{table} GROUP BY {time_column} ORDER BY {time_column}"
                    ).df()
                    alignment.insert(0, "feature_table", table)
                    alignment_frames.append(alignment)
                elif time_column:
                    alignment = con.execute(
                        f"SELECT {time_column}, count(DISTINCT fips) AS counties "
                        f"FROM feature.{table} GROUP BY {time_column} ORDER BY {time_column}"
                    ).df()
                    alignment.insert(0, "feature_table", table)
                    alignment_frames.append(alignment)
            alignment = pd.concat(alignment_frames, ignore_index=True) if alignment_frames else pd.DataFrame()
            display(alignment)
            """
        ),
        md(
            """
            ## Retention and infographic use

            Retention is decided in `feature.catalog`, not from visual inspection in this
            notebook. A feature is retained when its definition is interpretable, its unit and
            grain are stable, its completeness is adequate for its intended comparison, and it
            does not duplicate the outcome or a stronger related measure. The `infographic_use`
            and `exclusion_reason` catalog fields record the current decision.
            """
        ),
        code(
            """
            display(catalog[[
                "feature_name", "retained", "exclusion_reason", "infographic_use",
                "source_tables", "transformation"
            ]])
            con.close()
            """
        ),
    ]


def event_window_notebook() -> list:
    return [
        md(
            """
            # Housing Market Metrics Around Extreme Climate Events

            **Purpose.** Show only the project housing-market metrics around the configured
            event windows, first across all affected county-events and then grouped by FEMA
            National Risk Index rating.

            This notebook queries persisted `analysis.*` marts. Production database code owns
            event selection, deduplication, window alignment, completeness, and aggregation.
            """
        ),
        md("## Setup and canonical event-window definition"),
        code(
            """
            from pathlib import Path
            import duckdb
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from IPython.display import display

            pd.set_option("display.max_columns", 80)
            pd.set_option("display.max_rows", 200)
            sns.set_theme(style="whitegrid")

            ROOT = Path.cwd()
            while not (ROOT / "data" / "quoll.duckdb").exists() and ROOT != ROOT.parent:
                ROOT = ROOT.parent
            con = duckdb.connect(str(ROOT / "data" / "quoll.duckdb"), read_only=True)

            config = con.execute("SELECT * FROM analysis.event_window_config ORDER BY parameter").df()
            display(config)
            """
        ),
        md(
            """
            ### Interpretation of the configuration

            - **Extreme events:** qualifying FEMA declarations plus NOAA Storm Events with at
              least the configured total-damage threshold.
            - **FEMA inclusion:** county records with valid incident starts, excluding the listed
              non-climate or out-of-scope incident types.
            - **NOAA inclusion:** county-resolved events with valid starts and damage at or above
              the threshold.
            - **County-event deduplication:** exact production `event_key`; distinct events in the
              same county remain distinct.
            - **Start/end:** non-positive months align to event start; positive months align to
              event end. Missing ends use the start.
            - **Completeness:** a county-event-metric must contain every required month for the
              selected horizon.
            - **Overlap and repeated counties:** retained. One housing observation may contribute
              to multiple distinct event windows.
            - **Comparison:** descriptive medians of affected county-events. No external control
              baseline is used in this notebook.
            """
        ),
        md("## Event scope and cohort"),
        code(
            """
            event_scope = con.execute(
                '''
                SELECT
                    event_source,
                    count(*) AS county_events,
                    count(DISTINCT fips) AS counties,
                    count(DISTINCT source_event_id) AS source_events,
                    min(event_start_month) AS first_event_month,
                    max(event_start_month) AS last_event_month
                FROM analysis.extreme_events_county
                GROUP BY event_source
                ORDER BY event_source
                '''
            ).df()
            display(event_scope)

            overlap = con.execute(
                '''
                WITH ranges AS (
                    SELECT DISTINCT event_key, fips, event_start_month, event_end_month
                    FROM analysis.extreme_events_county
                )
                SELECT
                    count(*) AS overlapping_event_pairs,
                    count(DISTINCT a.fips) AS counties_with_overlap
                FROM ranges a
                JOIN ranges b
                  ON a.fips = b.fips
                 AND a.event_key < b.event_key
                 AND a.event_start_month <= b.event_end_month
                 AND b.event_start_month <= a.event_end_month
                '''
            ).df()
            display(overlap)
            """
        ),
        md("## Sample sizes at every horizon"),
        code(
            """
            sample_sizes = con.execute(
                '''
                SELECT
                    aggregation_level,
                    risk_rating,
                    horizon_months,
                    metric_name,
                    max(county_event_count) AS complete_county_events,
                    max(county_count) AS counties
                FROM analysis.housing_event_window_summary
                GROUP BY aggregation_level, risk_rating, horizon_months, metric_name
                ORDER BY aggregation_level, risk_rating, horizon_months, metric_name
                '''
            ).df()
            display(sample_sizes)
            """
        ),
        md("## Aggregate results across all counties"),
        code(
            """
            summary = con.execute(
                "SELECT * FROM analysis.housing_event_window_summary"
            ).df()
            metric_labels = {
                "median_ppsf_yoy": "Median PPSF YOY",
                "housing_market_index": "Housing Market Index",
                "avg_sale_to_list_yoy": "Average Sale-to-List YOY",
                "homes_sold_yoy": "Homes Sold YOY",
                "inventory_yoy": "Inventory YOY",
                "new_listings_yoy": "New Listings YOY",
                "median_dom_yoy": "Median Days on Market YOY",
                "price_drops_yoy": "Price Drops YOY",
            }
            horizons = sorted(summary["horizon_months"].dropna().unique())

            for horizon in horizons:
                data = summary.loc[
                    (summary["aggregation_level"] == "all_counties")
                    & (summary["horizon_months"] == horizon)
                ]
                fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True)
                for ax, (metric, label) in zip(axes.flat, metric_labels.items()):
                    metric_data = data.loc[data["metric_name"] == metric].sort_values("event_window_month")
                    ax.plot(metric_data["event_window_month"], metric_data["median_value"], color="#2a6f97")
                    ax.fill_between(
                        metric_data["event_window_month"],
                        metric_data["q25_value"],
                        metric_data["q75_value"],
                        alpha=0.18,
                        color="#2a6f97",
                    )
                    ax.axvline(0, color="black", linestyle="--", linewidth=1)
                    ax.set_title(label)
                    ax.set_ylabel("Median (IQR band)")
                fig.suptitle(f"All affected county-events: -12 to +{horizon} months", y=1.01)
                fig.supxlabel("Event-window month")
                plt.tight_layout()
                plt.show()
            """
        ),
        md("## Aggregate results grouped by NRI risk rating"),
        code(
            """
            risk_order = [
                "Very Low", "Relatively Low", "Relatively Moderate",
                "Relatively High", "Very High",
            ]
            palette = dict(zip(risk_order, sns.color_palette("viridis", len(risk_order))))

            for horizon in horizons:
                data = summary.loc[
                    (summary["aggregation_level"] == "nri_risk_rating")
                    & (summary["horizon_months"] == horizon)
                ].copy()
                fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True)
                for ax, (metric, label) in zip(axes.flat, metric_labels.items()):
                    metric_data = data.loc[data["metric_name"] == metric]
                    for rating in risk_order:
                        line = metric_data.loc[
                            metric_data["risk_rating"] == rating
                        ].sort_values("event_window_month")
                        if not line.empty:
                            ax.plot(
                                line["event_window_month"], line["median_value"],
                                label=rating, color=palette[rating],
                            )
                    ax.axvline(0, color="black", linestyle="--", linewidth=1)
                    ax.set_title(label)
                    ax.set_ylabel("Median")
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    title="NRI risk rating",
                    loc="center left",
                    bbox_to_anchor=(0.86, 0.5),
                    frameon=False,
                )
                fig.suptitle(f"Affected county-events by NRI risk rating: -12 to +{horizon} months", y=1.01)
                fig.supxlabel("Event-window month")
                plt.tight_layout(rect=(0, 0.04, 0.85, 1))
                plt.show()
            """
        ),
        md(
            """
            ## Limitations

            These are descriptive event-aligned medians, not causal estimates. Overlapping events
            and repeated counties are retained, county-events are not statistically independent,
            the complete-window rule changes the available sample by metric and horizon, and longer
            horizons exclude more recent events. FEMA declaration and NOAA damage-reporting
            processes also differ. Conclusions should therefore focus on visible patterns and
            uncertainty, not treatment effects.
            """
        ),
        code("con.close()"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute every notebook before writing it.")
    args = parser.parse_args()

    for slug, cfg in PROVIDERS.items():
        write_notebook(
            NOTEBOOKS_DIR / "01_data_quality" / f"{slug}_data_quality.ipynb",
            provider_notebook(slug, cfg),
            execute=args.execute,
        )

    for slug, cfg in FEATURE_NOTEBOOKS.items():
        write_notebook(
            NOTEBOOKS_DIR / "02_feature_exploration" / f"{slug}.ipynb",
            feature_notebook(slug, cfg),
            execute=args.execute,
        )

    write_notebook(
        NOTEBOOKS_DIR / "03_event_window" / "housing_market_event_windows.ipynb",
        event_window_notebook(),
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
