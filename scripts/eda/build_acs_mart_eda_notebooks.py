"""Build EDA notebooks for ACS DuckDB mart tables."""

from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap
import sys


ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = ROOT / "scripts" / "eda"

INVALID_PERCENT_COLUMNS_BY_TABLE = {
    "mart.acs_county_affordability_annual": {
        "dp04_total_housing_units_pct",
        "dp04_units_in_structure_total_housing_units_pct",
        "dp04_year_structure_built_total_housing_units_pct",
        "dp04_rooms_total_housing_units_pct",
        "dp04_bedrooms_total_housing_units_pct",
        "dp04_housing_tenure_occupied_housing_units_pct",
        "dp04_year_householder_moved_into_unit_occupied_housing_units_pct",
        "dp04_vehicles_available_occupied_housing_units_pct",
        "dp04_house_heating_fuel_occupied_housing_units_pct",
        "dp04_selected_characteristics_occupied_housing_units_pct",
        "dp04_occupants_per_room_occupied_housing_units_pct",
        "dp04_value_owner_occupied_units_pct",
        "dp04_mortgage_status_owner_occupied_units_pct",
        "dp04_selected_monthly_owner_costs_housing_units_mortgage_pct",
        "dp04_selected_monthly_owner_costs_housing_units_no_mortgage_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_pct",
        "dp04_gross_rent_occupied_units_paying_rent_pct",
        "dp04_gross_rent_as_a_pct_of_household_income_occupied_units_paying_rent_pct",
    },
    "mart.acs_county_demographic_annual": {
        "dp05_total_population_pct",
        "dp05_citizen_voting_age_population_citizen_18_and_over_population_pct",
    },
    "mart.acs_county_economic_annual": {
        "dp03_population_16_plus_pct",
        "dp03_civilian_labor_force_pct",
        "dp03_females_16_years_and_over_pct",
        "dp03_own_children_of_the_householder_under_6_years_pct",
        "dp03_own_children_of_the_householder_6_to_17_years_pct",
        "dp03_commuting_to_work_workers_16_years_and_over_pct",
        "dp03_occupation_civilian_employed_population_16_plus_pct",
        "dp03_industry_civilian_employed_population_16_plus_pct",
        "dp03_class_of_worker_civilian_employed_population_16_plus_pct",
        "dp03_income_and_benefits_total_households_pct",
        "dp03_income_and_benefits_families_pct",
        "dp03_income_and_benefits_nonfamily_households_pct",
        "dp03_health_insurance_coverage_civilian_noninstitutionalized_population_pct",
        "dp03_health_insurance_coverage_civilian_noninstitutionalized_population_under_19_years_pct",
        "dp03_health_insurance_coverage_civilian_noninstitutionalized_population_19_to_64_years_pct",
        "dp03_health_insurance_coverage_civilian_noninstitutionalized_population_19_to_64_years_in_labor_force_pct",
        "dp03_health_insurance_coverage_civilian_noninstitutionalized_population_19_to_64_years_in_labor_force_employed_pct",
        "dp03_health_insurance_coverage_civilian_noninstitutionalized_population_19_to_64_years_in_labor_force_unemployed_pct",
        "dp03_health_insurance_coverage_civilian_noninstitutionalized_population_19_to_64_years_not_in_labor_force_pct",
    },
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


SETUP = r"""
from pathlib import Path
import re
import textwrap

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", 160)
pd.set_option("display.max_rows", 120)
sns.set_theme(style="whitegrid")

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent.parent

DB_PATH = ROOT / "data" / "quoll.duckdb"

def q(sql: str) -> pd.DataFrame:
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return connection.execute(sql).df()
    finally:
        connection.close()

def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def numeric_expr(column: str) -> str:
    quoted = quote_ident(column)
    as_text = f"cast({quoted} AS VARCHAR)"
    return f"try_cast(nullif(nullif(nullif({as_text}, 'N'), '-'), '(X)') AS DOUBLE)"

def readable_column_name(column: str) -> str:
    # Compact display label; canonical mart names remain intact for queries.
    name = re.sub(r"^(?:dp\d+|s\d+|b\d+(?:_c\d+)?)_", "", column.lower())
    replacements = {
        "civilian_noninstitutionalized_population": "civilian_pop",
        "civilian_employed_population_16_plus": "employed_pop_16_plus",
        "population_16_plus": "pop_16_plus",
        "workers_16_years_and_over": "workers_16_plus",
        "income_and_benefits": "income",
        "employment_status": "employment",
        "health_insurance_coverage": "health_insurance",
        "poverty_status": "poverty",
        "occupied_housing_units": "occupied_units",
        "householder": "hhldr",
        "household": "hh",
        "population": "pop",
        "years_and_over": "plus",
        "years": "yrs",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    tokens = [token for token in name.split("_") if token]
    if len(tokens) > 10:
        tokens = tokens[:5] + ["..."] + tokens[-4:]
    return "_".join(tokens)

def readable_name_map(columns: list[str]) -> dict[str, str]:
    # Preserve canonical mart field names. Shortened labels can hide the ACS
    # universe or subgroup that distinguishes otherwise similar measures.
    return {column: column for column in columns}

def plot_label(column: str, width: int = 42) -> str:
    # Wrap the complete canonical name; never truncate semantic information.
    return textwrap.fill(column.replace("_", " "), width=width)

def completeness_for_columns(columns: list[str]) -> pd.DataFrame:
    parts = []
    for col in columns:
        quoted = quote_ident(col)
        as_text = f"cast({quoted} AS VARCHAR)"
        parts.append(
            f"SELECT '{col}' AS column_name, "
            f"count(*) FILTER (WHERE {quoted} IS NULL OR {as_text} IN ('', 'N', '-', '(X)')) AS missing_or_suppressed_count, "
            f"count(DISTINCT {quoted}) AS distinct_count "
            f"FROM {TABLE}"
        )
    out = q(" UNION ALL ".join(parts))
    out.insert(1, "readable_name", out["column_name"].map(readable_names))
    out["missing_or_suppressed_pct"] = out["missing_or_suppressed_count"] / total_rows * 100
    return out.sort_values(["missing_or_suppressed_pct", "missing_or_suppressed_count"], ascending=False)

def numeric_profile(columns: list[str], limit: int = 80) -> pd.DataFrame:
    rows = []
    for col in columns[:limit]:
        expr = numeric_expr(col)
        rows.append(
            f"SELECT '{col}' AS column_name, "
            f"count({expr}) AS numeric_count, "
            f"avg({expr}) AS mean, "
            f"min({expr}) AS min, "
            f"quantile_cont({expr}, 0.50) AS p50, "
            f"quantile_cont({expr}, 0.95) AS p95, "
            f"max({expr}) AS max "
            f"FROM {TABLE}"
        )
    out = q(" UNION ALL ".join(rows)).sort_values("numeric_count", ascending=False)
    out.insert(1, "readable_name", out["column_name"].map(readable_names))
    return out

def annual_median(column: str) -> pd.DataFrame:
    expr = numeric_expr(column)
    return q(f'''
        SELECT year, count({expr}) AS numeric_count, median({expr}) AS median_value
        FROM {TABLE}
        GROUP BY year
        ORDER BY year
    ''')

def choose_latest_focus_columns(
    candidate_columns: list[str],
    all_columns: list[str],
    measure_columns: list[str],
    latest_year: int,
    max_columns: int = 14,
) -> list[str]:
    # Select current, numeric ACS variables from this mart's latest year.
    selected = [c for c in candidate_columns if c in all_columns]
    selected = list(dict.fromkeys(selected))

    excluded_suffixes = ("_moe", "_pct_moe")
    pool = [
        c
        for c in measure_columns
        if c not in selected
        and c not in INVALID_PERCENT_COLUMNS
        and not c.endswith(excluded_suffixes)
        and c not in {"source_table", "county_fips", "NAME"}
    ]
    pool = sorted(
        pool,
        key=lambda c: (
            not c.startswith("median_"),
            not c.endswith("_pct"),
            not c.endswith("_est"),
            c,
        ),
    )

    if pool:
        rows = []
        for col in pool[:300]:
            rows.append(
                f"SELECT '{col}' AS column_name, count({numeric_expr(col)}) AS numeric_count "
                f"FROM {TABLE} WHERE year = {latest_year}"
            )
        numeric_counts = q(" UNION ALL ".join(rows))
        usable = numeric_counts[numeric_counts["numeric_count"].gt(0)]["column_name"].tolist()
        selected.extend(usable)

    return list(dict.fromkeys(selected))[:max_columns]
"""


def common_cells(table: str, title: str, focus_columns: list[str], prefix_notes: str) -> list[dict]:
    focus_literal = repr(focus_columns)
    invalid_percent_literal = repr(INVALID_PERCENT_COLUMNS_BY_TABLE.get(table, set()))
    return [
        md("## Setup"),
        code(
            SETUP
            + f'\n\nTABLE = "{table}"'
            + f"\nINVALID_PERCENT_COLUMNS = {invalid_percent_literal}"
            + f"\nFOCUS_CANDIDATES = {focus_literal}\nDB_PATH"
        ),
        md("""## Data Scope, Frequency, And Structure

Start here to understand the observation grain and organization of the wide table. Canonical mart column names are retained throughout tables and charts so ACS universes and subgroups remain explicit. Normalized suffixes are `_est`, `_moe`, `_pct`, and `_pct_moe`."""),
        code(
            r"""
row_count = q(f"SELECT count(*) AS rows FROM {TABLE}")
total_rows = int(row_count.loc[0, "rows"])
schema = q(f"DESCRIBE {TABLE}")
columns = schema["column_name"].tolist()
id_cols = ["fips", "state_fips", "year", "source_table", "county_fips", "NAME"]
measure_cols = [c for c in columns if c not in id_cols and c not in INVALID_PERCENT_COLUMNS]
estimate_cols = [c for c in measure_cols if c.endswith("_est")]
moe_cols = [c for c in measure_cols if c.endswith("_moe") and not c.endswith("_pct_moe")]
percent_cols = [c for c in measure_cols if c.endswith("_pct") and c not in INVALID_PERCENT_COLUMNS]
percent_moe_cols = [c for c in measure_cols if c.endswith("_pct_moe")]
latest_year = int(q(f"SELECT max(year) AS year FROM {TABLE}").loc[0, "year"])
focus_existing = choose_latest_focus_columns(FOCUS_CANDIDATES, columns, measure_cols, latest_year)
readable_names = readable_name_map(columns)

table_overview = pd.DataFrame({
    "property": ["observation_grain", "row_count", "column_count", "identifier_columns", "measure_columns", "year_range"],
    "value": [
        "one county × year × ACS source table", total_rows, len(columns), len(id_cols), len(measure_cols),
        f"{q(f'SELECT min(year) FROM {TABLE}').iloc[0, 0]}–{latest_year}",
    ],
})

schema_inventory = schema.rename(columns={"column_name": "column", "column_type": "duckdb_type"}).copy()
schema_inventory = schema_inventory.loc[~schema_inventory["column"].isin(INVALID_PERCENT_COLUMNS)].copy()
schema_inventory.insert(1, "readable_name", schema_inventory["column"].map(readable_names))
schema_inventory["role"] = schema_inventory["column"].map(
    lambda c: "identifier / dimension" if c in id_cols else "ACS measure"
)
schema_inventory["value_type"] = schema_inventory["column"].map(
    lambda c: (
        "percent margin of error" if c.endswith("_pct_moe") else
        "percent estimate" if c.endswith("_pct") else
        "estimate margin of error" if c.endswith("_moe") else
        "estimate" if c.endswith("_est") else
        "identifier / named field"
    )
)
structure_summary = (
    schema_inventory.groupby(["role", "value_type"], dropna=False).size()
    .rename("column_count").reset_index().sort_values(["role", "column_count"], ascending=[True, False])
)

display(table_overview)
display(pd.DataFrame({
    "group": ["all_columns", "id_columns", "measure_columns", "estimate_columns", "moe_columns", "percent_estimate_columns", "percent_moe_columns"],
    "count": [len(columns), len(id_cols), len(measure_cols), len(estimate_cols), len(moe_cols), len(percent_cols), len(percent_moe_cols)],
}))
display(structure_summary)
display(schema_inventory)
display(pd.DataFrame({
    "column_name": focus_existing,
    "readable_name": [readable_names[c] for c in focus_existing],
}))
"""
        ),
        code(
            r"""
sample_columns = [c for c in id_cols if c in columns] + focus_existing[:8]
sample = q(f"SELECT {', '.join(quote_ident(c) for c in sample_columns)} FROM {TABLE} USING SAMPLE 5 ROWS")
display(sample.rename(columns=readable_names))
"""
        ),
        md("""## Data Quality: Coverage And Keys

The mart is expected to be annual. This section verifies the observed year cadence, county and source-table coverage by year, key uniqueness, and geographic identifier consistency."""),
        code(
            r'''
coverage = q(f"""
    SELECT
        min(year) AS min_year,
        max(year) AS max_year,
        count(DISTINCT year) AS year_count,
        count(DISTINCT fips) AS county_count,
        count(DISTINCT state_fips) AS state_count,
        count(DISTINCT source_table) AS source_table_count
    FROM {TABLE}
""")
display(coverage)

annual_coverage = q(f"""
    SELECT year, count(*) AS rows, count(DISTINCT fips) AS counties, count(DISTINCT source_table) AS source_tables
    FROM {TABLE}
    GROUP BY year
    ORDER BY year
""")
display(annual_coverage)

ax = annual_coverage.plot(x="year", y="counties", marker="o", figsize=(10, 4), title="County coverage by year")
ax.set_ylabel("Distinct counties")
plt.show()
'''
        ),
        code(
            r'''
key_check = q(f"""
    SELECT
        count(*) AS rows,
        count(DISTINCT fips || '-' || year::VARCHAR || '-' || coalesce(source_table, '')) AS distinct_fips_year_source,
        count(*) - count(DISTINCT fips || '-' || year::VARCHAR || '-' || coalesce(source_table, '')) AS duplicate_fips_year_source_rows,
        count(*) FILTER (WHERE fips IS NULL) AS missing_fips,
        count(*) FILTER (WHERE year IS NULL) AS missing_year,
        count(*) FILTER (WHERE county_fips IS NOT NULL AND right(fips, 3) <> county_fips) AS county_fips_mismatch_rows
    FROM {TABLE}
""")
display(key_check)

dupes = q(f"""
    SELECT fips, year, source_table, count(*) AS rows
    FROM {TABLE}
    GROUP BY fips, year, source_table
    HAVING count(*) > 1
    ORDER BY rows DESC, fips, year
    LIMIT 50
""")
display(dupes)
'''
        ),
        md("## Column Families"),
        code(
            f'''
import re

prefix_counts = {{}}
for col in measure_cols:
    match = re.match(r"([a-z]+\\d+(?:_c\\d+)?)_", col)
    prefix = match.group(1) if match else "derived_or_named"
    prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

prefix_counts = pd.DataFrame(
    sorted(prefix_counts.items(), key=lambda item: item[1], reverse=True),
    columns=["prefix", "column_count"],
)
display(prefix_counts)
print({prefix_notes!r})
'''
        ),
        md("""## Data Completeness And Missingness

SQL nulls and ACS suppression markers (`N`, `-`, and `(X)`) count as unavailable. Both canonical and concise field names are shown."""),
        code(
            r"""
display(completeness_for_columns(id_cols))
completeness_cols = list(dict.fromkeys(focus_existing + measure_cols[:120]))
display(completeness_for_columns([c for c in completeness_cols if c in columns]).head(120))
"""
        ),
        code(
            r"""
display(completeness_for_columns(focus_existing))
"""
        ),
        md("""## Data Quality: Value Range And Numeric Distribution

Numeric-castable measures report usable count, minimum, mean, median, 95th percentile, and maximum to make observed ranges and possible outliers explicit."""),
        code(
            r"""
profile_cols = focus_existing + [c for c in estimate_cols[:30] if c not in focus_existing] + [c for c in percent_cols[:30] if c not in focus_existing]
numeric_summary = numeric_profile(profile_cols, limit=100)
display(numeric_summary)
"""
        ),
        code(
            r'''
latest_selects = ["fips", "NAME", "state_fips", "year"]
for col in focus_existing:
    latest_selects.append(f"{numeric_expr(col)} AS {quote_ident(col)}")

latest_focus = q(f"""
    SELECT {", ".join(latest_selects)}
    FROM {TABLE}
    WHERE year = {latest_year}
""")
display(latest_focus.head().rename(columns=readable_names))
display(latest_focus[focus_existing].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T if focus_existing else pd.DataFrame())
'''
        ),
        code(
            r"""
plot_cols = [c for c in focus_existing if c in latest_focus.columns][:12]
if plot_cols:
    plot_frame = latest_focus[plot_cols].copy()
    plot_frame.columns = [plot_label(column) for column in plot_cols]
    plot_frame.hist(figsize=(15, 11), bins=35, color="#4C72B0", edgecolor="white")
    plt.suptitle(f"Latest-year distributions ({latest_year})")
    plt.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.4, w_pad=1.5)
    plt.show()
"""
        ),
        md("## Annual Trends"),
        code(
            r"""
trend_frames = []
for col in focus_existing[:12]:
    tmp = annual_median(col)
    tmp["column_name"] = readable_names[col]
    trend_frames.append(tmp)

annual_trends = pd.concat(trend_frames, ignore_index=True) if trend_frames else pd.DataFrame()
display(annual_trends)
"""
        ),
        code(
            r"""
if not annual_trends.empty:
    grid = sns.relplot(
        data=annual_trends,
        x="year",
        y="median_value",
        col="column_name",
        col_wrap=3,
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=3.6,
        aspect=1.2,
    )
    trend_labels = annual_trends["column_name"].drop_duplicates().tolist()
    for axis, label in zip(grid.axes.flat, trend_labels):
        axis.set_title(textwrap.fill(label.replace("_", " "), width=34), fontsize=10, pad=10)
    grid.fig.subplots_adjust(top=0.92, hspace=0.42, wspace=0.25)
    plt.show()
"""
        ),
        md("## Latest-Year County Rankings"),
        code(
            r"""
for col in focus_existing[:8]:
    display(
        latest_focus[["fips", "NAME", "state_fips", "year", col]]
        .dropna(subset=[col])
        .sort_values(col, ascending=False)
        .head(20)
    )
"""
        ),
        md("## Cross-Section Correlations"),
        code(
            r"""
numeric_focus = latest_focus[focus_existing].select_dtypes(include="number") if focus_existing else pd.DataFrame()
if numeric_focus.shape[1] >= 2:
    corr = numeric_focus.corr()
    plt.figure(figsize=(min(14, 1 + numeric_focus.shape[1]), min(12, 1 + numeric_focus.shape[1])))
    sns.heatmap(corr, cmap="vlag", center=0, annot=numeric_focus.shape[1] <= 10)
    plt.title(f"Latest-year focus column correlations ({latest_year})")
    plt.show()
    display(corr)
"""
        ),
        md("## Data Quality: Suppressed And Non-Numeric Values"),
        code(
            r'''
suppression_checks = []
for col in focus_existing:
    quoted = quote_ident(col)
    as_text = f"cast({quoted} AS VARCHAR)"
    suppression_checks.append(
        q(f"""
            SELECT
                '{col}' AS column_name,
                count(*) FILTER (WHERE {as_text} = 'N') AS n_values,
                count(*) FILTER (WHERE {as_text} = '-') AS dash_values,
                count(*) FILTER (WHERE {as_text} = '(X)') AS x_values,
                count(*) FILTER (WHERE {quoted} IS NOT NULL AND {numeric_expr(col)} IS NULL AND {as_text} NOT IN ('N', '-', '(X)', '')) AS other_non_numeric_values
            FROM {TABLE}
        """)
    )

suppression_summary = pd.concat(suppression_checks, ignore_index=True) if suppression_checks else pd.DataFrame()
display(suppression_summary)
'''
        ),
    ]


def notebook(
    title: str,
    table: str,
    focus_columns: list[str],
    prefix_notes: str,
    feature_description: str,
    extra_cells: list[dict] | None = None,
) -> dict:
    return {
        "cells": [
            md(
                f"""# {title}

Dataset: `{table}` in `data/quoll.duckdb`

This notebook performs exploratory data analysis on a wide ACS county-year mart. It emphasizes coverage, completeness, key integrity, numeric castability, latest-year distributions, annual trends, rankings, and correlations for high-signal fields."""
            ),
            md(
                f"""## County Features Covered

{feature_description}"""
            ),
            *common_cells(table, title, focus_columns, prefix_notes),
            *(extra_cells or []),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


DEMOGRAPHIC_SUMMARY_CELLS_LEGACY = [
    md(
        """## Demographic Topic Summaries

The following summaries organize the wide demographic mart into ten interpretable subjects. Each topic shows its canonical-to-readable field mapping, latest-year completeness and range statistics, and distributions for representative county percentage measures. Estimates are retained in the catalog, while percentage fields are prioritized for comparisons across counties of different sizes."""
    ),
    code(
        r'''
demographic_topics = {
    "Household type": {"source": "%_dp02_%", "patterns": ["households_by_type"]},
    "Educational attainment": {"source": "%_dp02_%", "patterns": ["educational_attainment"]},
    "Disability status": {"source": "%_dp02_%", "patterns": ["disability_status"]},
    "Residence 1 year ago": {"source": "%_dp02_%", "patterns": ["residence_1_year_ago"]},
    "Citizenship status": {"source": "%_dp02_%", "patterns": ["citizenship_status"]},
    "Language spoken at home": {"source": "%_dp02_%", "patterns": ["language_spoken_at_home"]},
    "Ancestry": {"source": "%_dp02_%", "patterns": ["ancestry"]},
    "Computers and internet use": {"source": "%_dp02_%", "patterns": ["computers_and_internet_use"]},
    "Sex and age": {"source": "%_dp05_%", "patterns": ["dp05_total_population_"]},
    "Race": {"source": "%_dp05_%", "patterns": ["dp05_race_"]},
}

topic_fields: dict[str, list[str]] = {}
topic_focus_fields: dict[str, list[str]] = {}
for topic, specification in demographic_topics.items():
    fields = [
        c for c in measure_cols
        if any(pattern in c for pattern in specification["patterns"])
        and (c.endswith("_est") or c.endswith("_pct"))
        and c not in INVALID_PERCENT_COLUMNS
    ]
    # Percent estimates provide comparable county distributions. Keep up to 12
    # substantive fields per topic, excluding denominator-like total percentages
    # when more specific measures exist.
    percent_fields = [c for c in fields if c.endswith("_pct")]
    substantive_pct = [
        c for c in percent_fields
        if not c.endswith(("total_pct", "total_population_pct", "total_households_pct"))
    ]
    focus = (substantive_pct or percent_fields or [c for c in fields if c.endswith("_est")])[:12]
    topic_fields[topic] = fields
    topic_focus_fields[topic] = focus

    display(pd.DataFrame({
        "topic": topic,
        "column_name": fields,
        "readable_name": [readable_names[c] for c in fields],
        "value_type": ["percent estimate" if c.endswith("_pct") else "estimate" for c in fields],
    }))
'''
    ),
    code(
        r'''
topic_latest_frames: dict[str, pd.DataFrame] = {}
source_latest_frames: dict[str, pd.DataFrame] = {}
for source_pattern in sorted({specification["source"] for specification in demographic_topics.values()}):
    source_fields = list(dict.fromkeys(
        column
        for topic, specification in demographic_topics.items()
        if specification["source"] == source_pattern
        for column in topic_focus_fields[topic]
    ))
    selects = ", ".join(
        f"{numeric_expr(column)} AS {quote_ident(readable_names[column])}"
        for column in source_fields
    )
    source_latest_frames[source_pattern] = q(f"""
        SELECT fips, NAME AS county_name, state_fips, year, {selects}
        FROM {TABLE}
        WHERE year = {latest_year}
          AND source_table LIKE '{source_pattern}'
    """).drop_duplicates(["fips", "year"])

for topic, specification in demographic_topics.items():
    focus = topic_focus_fields[topic]
    if not focus:
        print(f"{topic}: no matching fields")
        continue
    measure_names = [readable_names[c] for c in focus]
    frame = source_latest_frames[specification["source"]][
        ["fips", "county_name", "state_fips", "year", *measure_names]
    ].copy()
    available_focus = [
        column for column in focus
        if frame[readable_names[column]].notna().any()
    ]
    if not available_focus:
        print(f"{topic}: matching fields are unavailable in {latest_year}")
        continue
    measure_names = [readable_names[column] for column in available_focus]
    frame = frame[["fips", "county_name", "state_fips", "year", *measure_names]]
    topic_latest_frames[topic] = frame

    print(f"\n{topic} - {len(topic_fields[topic])} fields; showing {len(available_focus)} county measures")
    summary = frame[measure_names].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).T.rename(columns={"50%": "median"})
    summary.insert(0, "complete_pct", frame[measure_names].notna().mean().mul(100))
    display(summary)

    plot_columns = available_focus[:6]
    if plot_columns:
        plot_frame = frame[[readable_names[column] for column in plot_columns]].copy()
        plot_frame.columns = [plot_label(column) for column in plot_columns]
        plot_frame.hist(figsize=(15, 8), bins=30, color="#4C72B0", edgecolor="white")
        plt.suptitle(f"{topic}: latest-year county distributions ({latest_year})")
        plt.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=1.5)
        plt.show()

    trend_parts = []
    for column in available_focus[:4]:
        trend = annual_median(column)
        trend["measure"] = readable_names[column]
        trend_parts.append(trend)
    annual = pd.concat(trend_parts, ignore_index=True)
    grid = sns.relplot(
        data=annual, x="year", y="median_value", col="measure",
        col_wrap=2, kind="line", marker="o",
        facet_kws={"sharey": False}, height=3.6, aspect=1.25,
    )
    for axis, column in zip(grid.axes.flat, available_focus[:4]):
        axis.set_title(plot_label(column), fontsize=10, pad=10)
    grid.fig.suptitle(f"{topic}: county median over time", y=1.02)
    grid.fig.subplots_adjust(top=0.88, hspace=0.42, wspace=0.25)
    plt.show()

    representative = measure_names[0]
    ranked = frame[["fips", "county_name", "state_fips", "year", representative]].dropna()
    display(pd.concat([
        ranked.nsmallest(10, representative).assign(rank_group="lowest"),
        ranked.nlargest(10, representative).assign(rank_group="highest"),
    ]))
'''
    ),
]


DEMOGRAPHIC_SUMMARY_CELLS = [
    md(
        """## Demographic Topic Summaries

This section is a guided overview rather than an exhaustive profile. It uses a small set of interpretable indicators to show what the mart contains and the broad patterns visible across years and counties. Denominator fields, margins of error, raw counts, and redundant subcategories are intentionally omitted; consult the schema and completeness sections above for the full inventory."""
    ),
    code(
        r'''
demographic_topics = {
    "Households": {
        "source": "%_dp02_%",
        "measures": {
            "Married-couple households (%)": "dp02_households_by_type_total_households_married_couple_household_pct",
            "Households with children under 18 (%)": "dp02_households_by_type_total_households_households_with_one_or_more_people_under_18_pct",
            "Households with someone age 65+ (%)": "dp02_households_by_type_total_households_households_with_one_or_more_people_65_plus_pct",
        },
    },
    "Education": {
        "source": "%_dp02_%",
        "measures": {
            "High school graduate or higher (%)": "dp02_educational_attainment_high_school_graduate_or_higher_pct",
            "Bachelor's degree or higher (%)": "dp02_educational_attainment_bachelors_degree_or_higher_pct",
        },
    },
    "Disability": {
        "source": "%_dp02_%",
        "measures": {
            "Population with a disability (%)": "dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct",
            "Under 18 with a disability (%)": "dp02_disability_status_of_the_civilian_noninstitutionalized_population_under_18_with_a_disability_pct",
            "Age 65+ with a disability (%)": "dp02_disability_status_of_the_civilian_noninstitutionalized_population_65_plus_with_a_disability_pct",
        },
    },
    "Residential mobility": {
        "source": "%_dp02_%",
        "measures": {
            "Same house one year ago (%)": "dp02_residence_1_year_ago_population_1_year_and_over_same_house_pct",
            "Moved within same county (%)": "dp02_residence_1_year_ago_population_1_year_and_over_different_house_same_county_pct",
            "Moved from another state (%)": "dp02_residence_1_year_ago_population_1_year_and_over_different_house_different_county_different_state_pct",
        },
    },
    "Language": {
        "source": "%_dp02_%",
        "measures": {
            "Language other than English at home (%)": "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_pct",
            "English less than very well (%)": "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct",
        },
    },
    "Computer and internet access": {
        "source": "%_dp02_%",
        "measures": {
            "Households with a computer (%)": "dp02_computers_and_internet_use_total_households_with_a_computer_pct",
            "Households with broadband (%)": "dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct",
        },
    },
    "Age": {
        "source": "%_dp05_%",
        "measures": {
            "Median age (years)": "dp05_total_population_median_age_est",
            "Population under 18 (%)": "dp05_total_population_under_18_pct",
            "Population age 65+ (%)": "dp05_total_population_65_plus_pct",
        },
    },
    "Race": {
        "source": "%_dp05_%",
        "measures": {
            "White alone (%)": "dp05_race_total_population_one_race_white_pct",
            "Black or African American alone (%)": "dp05_race_total_population_one_race_black_or_african_american_pct",
            "Asian alone (%)": "dp05_race_total_population_one_race_asian_pct",
            "Two or more races (%)": "dp05_race_total_population_two_or_more_races_pct",
        },
    },
}

topic_catalog = []
for topic, specification in demographic_topics.items():
    for label, column in specification["measures"].items():
        topic_catalog.append({
            "topic": topic,
            "indicator": label,
            "column_name": column,
            "available": column in columns,
        })
topic_catalog = pd.DataFrame(topic_catalog)
display(topic_catalog)
'''
    ),
    code(
        r'''
topic_latest_frames: dict[str, pd.DataFrame] = {}
topic_annual_frames: dict[str, pd.DataFrame] = {}

for topic, specification in demographic_topics.items():
    measures = {
        label: column
        for label, column in specification["measures"].items()
        if column in columns
    }
    if not measures:
        print(f"{topic}: no configured indicators are available")
        continue

    preferred_source = q(f"""
        SELECT max(source_table) AS source_table
        FROM {TABLE}
        WHERE source_table LIKE '{specification["source"]}'
    """).loc[0, "source_table"]
    select_parts = [
        f"{numeric_expr(column)} AS {quote_ident(label)}"
        for label, column in measures.items()
    ]
    panel = q(f"""
        SELECT fips, NAME AS county_name, state_fips, year,
               {", ".join(select_parts)}
        FROM {TABLE}
        WHERE source_table = '{preferred_source}'
    """).drop_duplicates(["fips", "year"])

    usable = [label for label in measures if panel[label].notna().any()]
    panel = panel[["fips", "county_name", "state_fips", "year", *usable]]
    topic_latest_frames[topic] = panel.loc[panel["year"].eq(latest_year)].copy()

    annual = (
        panel.groupby("year")[usable]
        .median()
        .reset_index()
        .melt(id_vars="year", var_name="indicator", value_name="county_median")
    )
    topic_annual_frames[topic] = annual

    print(f"\n{topic}")
    display(
        topic_latest_frames[topic][usable]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T[["count", "mean", "25%", "50%", "75%", "min", "max"]]
        .rename(columns={"50%": "median"})
        .round(2)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.lineplot(
        data=annual, x="year", y="county_median",
        hue="indicator", marker="o", ax=axes[0],
    )
    axes[0].set_title("Typical county over time")
    axes[0].set_ylabel("Median across counties")
    axes[0].legend(title="", fontsize=8)

    latest_long = topic_latest_frames[topic].melt(
        id_vars=["fips", "county_name", "state_fips", "year"],
        value_vars=usable, var_name="indicator", value_name="value",
    )
    sns.boxplot(
        data=latest_long, y="indicator", x="value",
        color="#4C72B0", showfliers=False, ax=axes[1],
    )
    axes[1].set_title(f"County variation in {latest_year}")
    axes[1].set_xlabel("County value (outliers hidden)")
    axes[1].set_ylabel("")
    fig.suptitle(topic, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()
'''
    ),
]


ECONOMIC_SUMMARY_CELLS = [
    md(
        """## Economic Topic Summaries Across Time And Counties

The following views organize DP03 measures into interpretable economic topics. For each topic they inventory the matching fields, summarize latest-year completeness and cross-county distributions, chart county medians through time, and list counties at both ends of a representative measure."""
    ),
    code(
        r'''
economic_topics = {
    "Labor force and employment": ["population_16_plus_in_labor_force", "unemployment_rate"],
    "Commuting": ["commuting_to_work"],
    "Occupation": ["occupation_civilian"],
    "Industry": ["industry_civilian"],
    "Class of worker": ["class_of_worker"],
    "Household and family income": ["income_and_benefits"],
    "Earnings": ["earnings_"],
    "Public assistance": ["public_assistance", "food_stamp", "snap"],
    "Poverty": ["poverty_status"],
    "Health insurance": ["health_insurance_coverage"],
}

economic_topic_fields: dict[str, list[str]] = {}
economic_topic_focus: dict[str, list[str]] = {}
for topic, patterns in economic_topics.items():
    fields = [
        column for column in measure_cols
        if any(pattern in column for pattern in patterns)
        and (column.endswith("_est") or column.endswith("_pct"))
        and column not in INVALID_PERCENT_COLUMNS
    ]
    percent_fields = [
        column for column in fields
        if column.endswith("_pct")
        and not column.endswith(("total_pct", "total_households_pct"))
    ]
    estimate_fields = [
        column for column in fields
        if column.endswith("_est")
        and any(token in column for token in ("median", "mean", "per_capita"))
    ]
    if topic == "Labor force and employment":
        focus = [
            column for column in [
                "dp03_population_16_plus_in_labor_force_pct",
                "dp03_population_16_plus_in_labor_force_civilian_labor_force_employed_pct",
                "dp03_civilian_labor_force_unemployment_rate_pct",
            ]
            if column in measure_cols
        ]
    else:
        candidates = (
            estimate_fields + percent_fields
            if topic in {"Household and family income", "Earnings"}
            else percent_fields + estimate_fields
        )
        focus = list(dict.fromkeys(candidates))[:10]
    economic_topic_fields[topic] = fields
    economic_topic_focus[topic] = focus
    display(pd.DataFrame({
        "topic": topic,
        "column_name": fields,
        "readable_name": [readable_names[column] for column in fields],
        "value_type": [
            "percent estimate" if column.endswith("_pct") else "estimate"
            for column in fields
        ],
    }))
'''
    ),
    code(
        r'''
economic_latest: dict[str, pd.DataFrame] = {}
economic_annual: dict[str, pd.DataFrame] = {}

for topic, focus in economic_topic_focus.items():
    if not focus:
        print(f"{topic}: no matching fields")
        continue

    selects = ", ".join(
        f"{numeric_expr(column)} AS {quote_ident(readable_names[column])}"
        for column in focus
    )
    latest = q(f"""
        SELECT fips, NAME AS county_name, state_fips, year, {selects}
        FROM {TABLE}
        WHERE year = {latest_year}
          AND source_table LIKE '%_dp03_%'
    """).drop_duplicates(["fips", "year"])
    available_focus = [
        column for column in focus
        if latest[readable_names[column]].notna().any()
    ]
    if not available_focus:
        print(f"{topic}: configured measures are unavailable in {latest_year}")
        continue
    measure_names = [readable_names[column] for column in available_focus]
    latest = latest[["fips", "county_name", "state_fips", "year", *measure_names]]
    economic_latest[topic] = latest

    print(f"\n{topic} - {len(economic_topic_fields[topic])} fields; showing {len(available_focus)} measures")
    summary = latest[measure_names].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).T.rename(columns={"50%": "median"})
    summary.insert(0, "complete_pct", latest[measure_names].notna().mean().mul(100))
    display(summary)

    plot_columns = available_focus[:6]
    plot_frame = latest[[readable_names[column] for column in plot_columns]].copy()
    plot_frame.columns = [plot_label(column) for column in plot_columns]
    plot_count = len(plot_columns)
    plot_layout = (1, plot_count) if plot_count <= 3 else (2, 3)
    plot_size = (5 * plot_count, 4.8) if plot_count <= 3 else (15, 8)
    plot_frame.hist(
        figsize=plot_size, layout=plot_layout, bins=30,
        color="#4C72B0", edgecolor="white",
    )
    plt.suptitle(f"{topic}: cross-county distributions ({latest_year})")
    plt.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=1.5)
    plt.show()

    trend_parts = []
    for column in available_focus[:4]:
        trend = annual_median(column)
        trend["measure"] = readable_names[column]
        trend_parts.append(trend)
    annual = pd.concat(trend_parts, ignore_index=True).dropna(subset=["median_value"])
    economic_annual[topic] = annual
    if annual.empty:
        print(f"{topic}: no annual numeric observations to plot")
        continue
    trend_count = min(4, len(available_focus))
    trend_wrap = trend_count if trend_count <= 3 else 2
    grid = sns.relplot(
        data=annual, x="year", y="median_value", col="measure",
        col_wrap=trend_wrap, kind="line", marker="o",
        facet_kws={"sharey": False}, height=3.6, aspect=1.25,
    )
    for axis, column in zip(grid.axes.flat, available_focus[:4]):
        axis.set_title(plot_label(column), fontsize=10, pad=10)
    grid.fig.suptitle(f"{topic}: county median over time", y=0.99)
    grid.fig.subplots_adjust(top=0.84, hspace=0.42, wspace=0.25)
    plt.show()

    representative = measure_names[0]
    ranked = latest[["fips", "county_name", "state_fips", "year", representative]].dropna()
    display(pd.concat([
        ranked.nsmallest(10, representative).assign(rank_group="lowest"),
        ranked.nlargest(10, representative).assign(rank_group="highest"),
    ]))
'''
    ),
]


AFFORDABILITY_BREAKDOWN_CELLS = [
    md(
        """## Homeownership Cost Breakdown

This county-year view uses the same ACS sources and definitions as the HTML page. It puts the major homeowner cost signals on annual-dollar terms and reports the share of owner households spending at least 30% of income on selected owner costs.

- **Homeowner insurance:** weighted midpoint of B25141 annual cost bands for owners with and without a mortgage.
- **Property taxes:** B25103 total county median real-estate taxes paid (`B25103_001E`).
- **Utilities:** electricity and gas monthly bands annualized, plus annual water/sewer and other-fuel bands.
- **Burdened owner households:** mortgaged and non-mortgaged owner households at 30% or more of income, divided by households with a computed burden value.
- **Total homeownership costs:** S2503 median monthly housing costs across all owner-occupied units, annualized. This ACS total already includes applicable mortgage payments, taxes, insurance, utilities, and related regular costs, so the components should not be added to it again.
- **Homeownership cost share:** annualized S2503 median monthly owner cost divided by S2503 median owner-household income.

Band-derived component values are approximations, and their ACS universes differ slightly; use them as a composition view rather than an accounting identity."""
    ),
    code(
        r'''
def dollars_from_slug(value: str) -> float:
    return float(value.replace("_", ""))

def band_midpoint(column: str) -> float | None:
    less_than = re.search(r"less_than_dollars_([0-9_]+)_est$", column)
    bounded = re.search(r"dollars_([0-9_]+)_to_dollars_([0-9_]+)_est$", column)
    open_ended = re.search(r"dollars_([0-9_]+)_or_more_est$", column)
    if less_than:
        return dollars_from_slug(less_than.group(1)) / 2
    if bounded:
        return (dollars_from_slug(bounded.group(1)) + dollars_from_slug(bounded.group(2))) / 2
    if open_ended:
        lower = dollars_from_slug(open_ended.group(1))
        return lower * 1.125
    return None

def band_columns(prefix: str, required_text: str = "") -> list[str]:
    return [
        c for c in columns
        if c.startswith(prefix)
        and required_text in c
        and band_midpoint(c) is not None
    ]

insurance_bins = [
    c for c in columns
    if c.startswith("b25141_")
    and (
        "_total_mortgage_" in c
        or "_total_not_mortgaged_" in c
    )
    and band_midpoint(c) is not None
]
electricity_bins = band_columns("b25132_", "_charged_for_electricity_")
gas_bins = band_columns("b25133_", "_charged_for_gas_")
water_bins = band_columns("b25134_", "_charged_for_water_and_sewer_")
other_fuel_bins = band_columns("b25135_", "_charged_for_other_fuels_")
electricity_zero_cols = [
    "b25132_monthly_electricity_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"
]
gas_zero_cols = [
    "b25133_monthly_gas_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"
]
water_zero_cols = [
    "b25134_annual_water_and_sewer_costs_total_not_charged_or_payment_included_in_other_fees_est"
]
other_fuel_zero_cols = [
    "b25135_annual_other_fuel_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"
]

burden_columns = {
    "mortgage_total": "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_est",
    "mortgage_30_34": "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_30_0_to_34_9_percent_est",
    "mortgage_35_plus": "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_35_0_percent_or_more_est",
    "mortgage_not_computed": "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_not_computed_est",
    "no_mortgage_total": "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_est",
    "no_mortgage_30_34": "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_30_0_to_34_9_percent_est",
    "no_mortgage_35_plus": "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_35_0_percent_or_more_est",
    "no_mortgage_not_computed": "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_not_computed_est",
}
total_cost_column = "s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est"
owner_income_column = "s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est"
property_tax_column = "median_property_taxes"

def source_component_frame(source_pattern: str, selected_columns: list[str], include_geography: bool = False) -> pd.DataFrame:
    selected_columns = [c for c in selected_columns if c in columns]
    geography = ", NAME AS county_name, state_fips" if include_geography else ""
    measures = "".join(f", {numeric_expr(c)} AS {quote_ident(c)}" for c in selected_columns)
    frame = q(f"""
        SELECT fips, year{geography}{measures}
        FROM {TABLE}
        WHERE year = {latest_year}
          AND source_table LIKE '{source_pattern}'
    """)
    return frame.drop_duplicates(["fips", "year"]).set_index(["fips", "year"])

# Reading each source-table slice separately avoids repeatedly aggregating the
# entire 1,700+ column affordability mart.
homeowner_costs = source_component_frame(
    "%_housing_financial_characteristics_%",
    [total_cost_column, owner_income_column],
    include_geography=True,
)
source_groups = [
    ("%_dp04_%", [*burden_columns.values()]),
    ("%_b25103_%", [property_tax_column]),
    ("%_b25141_%", insurance_bins),
    ("%_b25132_%", electricity_bins),
    ("%_b25133_%", gas_bins),
    ("%_b25134_%", water_bins),
    ("%_b25135_%", other_fuel_bins),
]
for source_pattern, selected_columns in source_groups:
    component_frame = source_component_frame(source_pattern, selected_columns)
    homeowner_costs = homeowner_costs.join(component_frame, how="outer")
homeowner_costs = homeowner_costs.reset_index()

def weighted_band_cost(
    frame: pd.DataFrame,
    band_cols: list[str],
    zero_cols: list[str] | None = None,
) -> pd.Series:
    if not band_cols:
        return pd.Series(float("nan"), index=frame.index)
    zero_cols = [c for c in (zero_cols or []) if c in frame]
    counts = frame[band_cols].apply(pd.to_numeric, errors="coerce")
    midpoints = pd.Series({c: band_midpoint(c) for c in band_cols})
    denominator = counts.sum(axis=1, min_count=1)
    if zero_cols:
        denominator = denominator.add(
            frame[zero_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1),
            fill_value=0,
        )
    return counts.mul(midpoints, axis=1).sum(axis=1, min_count=1).div(denominator.where(denominator.gt(0)))

homeowner_costs["homeowner_insurance_annual"] = weighted_band_cost(homeowner_costs, insurance_bins)
homeowner_costs["electricity_annual"] = weighted_band_cost(
    homeowner_costs, electricity_bins, electricity_zero_cols
) * 12
homeowner_costs["gas_annual"] = weighted_band_cost(
    homeowner_costs, gas_bins, gas_zero_cols
) * 12
homeowner_costs["water_sewer_annual"] = weighted_band_cost(
    homeowner_costs, water_bins, water_zero_cols
)
homeowner_costs["other_fuel_annual"] = weighted_band_cost(
    homeowner_costs, other_fuel_bins, other_fuel_zero_cols
)
homeowner_costs["utilities_annual"] = homeowner_costs[
    ["electricity_annual", "gas_annual", "water_sewer_annual", "other_fuel_annual"]
].sum(axis=1, min_count=1)
homeowner_costs["property_taxes_annual"] = (
    pd.to_numeric(homeowner_costs[property_tax_column], errors="coerce")
    if property_tax_column in homeowner_costs else float("nan")
)
homeowner_costs["total_homeownership_cost_annual"] = (
    pd.to_numeric(homeowner_costs[total_cost_column], errors="coerce") * 12
    if total_cost_column in homeowner_costs else float("nan")
)
homeowner_costs["homeownership_cost_pct_income"] = (
    homeowner_costs["total_homeownership_cost_annual"]
    / pd.to_numeric(homeowner_costs[owner_income_column], errors="coerce").replace(0, pd.NA)
    * 100
)

def numeric_or_zero(column: str) -> pd.Series:
    if column not in homeowner_costs:
        return pd.Series(0.0, index=homeowner_costs.index)
    return pd.to_numeric(homeowner_costs[column], errors="coerce").fillna(0)

burdened_count = sum(
    (numeric_or_zero(burden_columns[key]) for key in ["mortgage_30_34", "mortgage_35_plus", "no_mortgage_30_34", "no_mortgage_35_plus"]),
    start=pd.Series(0.0, index=homeowner_costs.index),
)
burden_denominator = (
    numeric_or_zero(burden_columns["mortgage_total"])
    + numeric_or_zero(burden_columns["no_mortgage_total"])
    - numeric_or_zero(burden_columns["mortgage_not_computed"])
    - numeric_or_zero(burden_columns["no_mortgage_not_computed"])
)
homeowner_costs["burdened_owner_households_pct"] = burdened_count.div(
    burden_denominator.where(burden_denominator.gt(0))
) * 100

breakdown_columns = [
    "homeowner_insurance_annual",
    "property_taxes_annual",
    "utilities_annual",
    "burdened_owner_households_pct",
    "total_homeownership_cost_annual",
    "homeownership_cost_pct_income",
]
display(homeowner_costs[["fips", "county_name", "state_fips", "year", *breakdown_columns]].head(20))
display(homeowner_costs[breakdown_columns].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T)
'''
    ),
    code(
        r'''
cost_components = ["homeowner_insurance_annual", "property_taxes_annual", "utilities_annual"]
plot_data = homeowner_costs[cost_components].melt(var_name="component", value_name="annual_usd").dropna()
if not plot_data.empty:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=plot_data, x="component", y="annual_usd", showfliers=False)
    plt.title(f"County homeowner cost components ({latest_year})")
    plt.xlabel("")
    plt.ylabel("Estimated annual dollars")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

display(
    homeowner_costs[["fips", "county_name", "state_fips", "year", *breakdown_columns]]
    .sort_values("total_homeownership_cost_annual", ascending=False)
    .head(25)
)
'''
    ),
    md(
        """## Affordability Across Time And Counties

The analysis below rebuilds the same measures for every available county-year. It separates:

- **All-owner total cost:** annualized S2503 median monthly housing costs across
  all owner-occupied units, matching the HTML page.
- **Cost components:** estimated homeowner insurance, property taxes, and utilities in annual dollars.
- **Outcome indicator:** the share of owner households with selected costs at or above 30% of income.
- **Affordability ratio:** annualized all-owner housing cost as a percentage of
  median owner-household income.

County medians and interquartile bands show the national time pattern without allowing a few extreme counties to dominate. County rankings, changes, and selected-county trajectories then show geographic variation. Dollar values are nominal ACS dollars; use inflation-adjusted values for real purchasing-power comparisons."""
    ),
    code(
        r'''
def source_component_panel(
    source_pattern: str,
    selected_columns: list[str],
    include_geography: bool = False,
) -> pd.DataFrame:
    selected_columns = [c for c in selected_columns if c in columns]
    geography = ", NAME AS county_name, state_fips" if include_geography else ""
    measures = "".join(f", {numeric_expr(c)} AS {quote_ident(c)}" for c in selected_columns)
    frame = q(f"""
        SELECT fips, year{geography}{measures}
        FROM {TABLE}
        WHERE source_table LIKE '{source_pattern}'
    """)
    return frame.drop_duplicates(["fips", "year"]).set_index(["fips", "year"])

affordability_panel = source_component_panel(
    "%_housing_financial_characteristics_%",
    [total_cost_column, owner_income_column],
    include_geography=True,
)
for source_pattern, selected_columns in source_groups:
    affordability_panel = affordability_panel.join(
        source_component_panel(source_pattern, selected_columns),
        how="outer",
    )
affordability_panel = affordability_panel.reset_index()

affordability_panel["homeowner_insurance_annual"] = weighted_band_cost(
    affordability_panel, insurance_bins
)
affordability_panel["electricity_annual"] = weighted_band_cost(
    affordability_panel, electricity_bins, electricity_zero_cols
) * 12
affordability_panel["gas_annual"] = weighted_band_cost(
    affordability_panel, gas_bins, gas_zero_cols
) * 12
affordability_panel["water_sewer_annual"] = weighted_band_cost(
    affordability_panel, water_bins, water_zero_cols
)
affordability_panel["other_fuel_annual"] = weighted_band_cost(
    affordability_panel, other_fuel_bins, other_fuel_zero_cols
)
affordability_panel["utilities_annual"] = affordability_panel[
    ["electricity_annual", "gas_annual", "water_sewer_annual", "other_fuel_annual"]
].sum(axis=1, min_count=1)
affordability_panel["property_taxes_annual"] = pd.to_numeric(
    affordability_panel.get(property_tax_column), errors="coerce"
)
affordability_panel["total_homeownership_cost_annual"] = pd.to_numeric(
    affordability_panel.get(total_cost_column), errors="coerce"
) * 12
affordability_panel["homeownership_cost_pct_income"] = (
    affordability_panel["total_homeownership_cost_annual"]
    / pd.to_numeric(
        affordability_panel.get(owner_income_column), errors="coerce"
    ).replace(0, pd.NA)
    * 100
)

def panel_numeric_or_zero(column: str) -> pd.Series:
    if column not in affordability_panel:
        return pd.Series(0.0, index=affordability_panel.index)
    return pd.to_numeric(affordability_panel[column], errors="coerce").fillna(0)

panel_burdened = sum(
    (
        panel_numeric_or_zero(burden_columns[key])
        for key in ["mortgage_30_34", "mortgage_35_plus", "no_mortgage_30_34", "no_mortgage_35_plus"]
    ),
    start=pd.Series(0.0, index=affordability_panel.index),
)
panel_denominator = (
    panel_numeric_or_zero(burden_columns["mortgage_total"])
    + panel_numeric_or_zero(burden_columns["no_mortgage_total"])
    - panel_numeric_or_zero(burden_columns["mortgage_not_computed"])
    - panel_numeric_or_zero(burden_columns["no_mortgage_not_computed"])
)
affordability_panel["burdened_owner_households_pct"] = (
    panel_burdened.div(panel_denominator.where(panel_denominator.gt(0))) * 100
)

panel_measures = [
    "total_homeownership_cost_annual",
    "homeowner_insurance_annual",
    "property_taxes_annual",
    "utilities_annual",
    "burdened_owner_households_pct",
    "homeownership_cost_pct_income",
]
display(
    affordability_panel[["fips", "county_name", "state_fips", "year", *panel_measures]]
    .sort_values(["year", "fips"])
    .head(20)
)
'''
    ),
    code(
        r'''
annual_distribution = (
    affordability_panel.groupby("year")[panel_measures]
    .agg(["count", "median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
)
annual_distribution.columns = [
    f"{measure}_{stat}"
    for measure, stat in annual_distribution.columns
]
annual_distribution = annual_distribution.rename(
    columns=lambda c: c.replace("<lambda_0>", "p25").replace("<lambda_1>", "p75")
)
display(annual_distribution)

trend_long = affordability_panel.melt(
    id_vars=["fips", "year"],
    value_vars=panel_measures,
    var_name="measure",
    value_name="value",
).dropna()
trend_summary = (
    trend_long.groupby(["year", "measure"])["value"]
    .agg(
        median="median",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
    )
    .reset_index()
)

fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=False)
for ax, measure in zip(axes.flat, panel_measures):
    data = trend_summary[trend_summary["measure"].eq(measure)]
    ax.plot(data["year"], data["median"], marker="o")
    ax.fill_between(data["year"], data["p25"], data["p75"], alpha=0.2)
    ax.set_title(measure.replace("_", " ").title())
    ax.set_ylabel("Percent" if measure.endswith("_pct") else "Nominal annual dollars")
    ax.set_xlabel("Year")
    ax.set_xticks(sorted(data["year"].unique()))
    ax.tick_params(axis="x", labelbottom=True, rotation=45)
for ax in axes.flat[len(panel_measures):]:
    ax.axis("off")
fig.suptitle("County median affordability and interquartile range over time", y=1.01)
plt.tight_layout()
plt.show()
'''
    ),
    code(
        r'''
latest_counties = affordability_panel.loc[
    affordability_panel["year"].eq(latest_year),
    ["fips", "county_name", "state_fips", "year", *panel_measures],
].copy()

latest_long = latest_counties.melt(
    id_vars=["fips", "county_name", "state_fips", "year"],
    value_vars=panel_measures,
    var_name="measure",
    value_name="value",
).dropna()
grid = sns.catplot(
    data=latest_long,
    x="value",
    col="measure",
    col_wrap=2,
    kind="box",
    sharex=False,
    height=3.5,
    aspect=1.35,
    showfliers=False,
)
grid.set_titles("{col_name}")
grid.set_axis_labels("County value", "")
grid.fig.suptitle(f"Cross-county affordability variation ({latest_year})", y=1.02)
plt.show()

for measure in panel_measures:
    print(f"\nHighest counties: {measure}")
    display(
        latest_counties[["fips", "county_name", "state_fips", measure]]
        .dropna(subset=[measure])
        .nlargest(15, measure)
    )
    print(f"Lowest counties: {measure}")
    display(
        latest_counties[["fips", "county_name", "state_fips", measure]]
        .dropna(subset=[measure])
        .nsmallest(15, measure)
    )
'''
    ),
    code(
        r'''
first_year = int(affordability_panel["year"].min())
endpoints = affordability_panel[
    affordability_panel["year"].isin([first_year, latest_year])
].pivot_table(index=["fips", "county_name", "state_fips"], columns="year", values=panel_measures)

change_frames = []
for measure in panel_measures:
    if (measure, first_year) not in endpoints or (measure, latest_year) not in endpoints:
        continue
    change = (
        endpoints[(measure, latest_year)] - endpoints[(measure, first_year)]
    ).rename("absolute_change").to_frame()
    change["measure"] = measure
    change["first_year_value"] = endpoints[(measure, first_year)]
    change["latest_year_value"] = endpoints[(measure, latest_year)]
    change_frames.append(change.reset_index())
county_changes = pd.concat(change_frames, ignore_index=True)

display(
    county_changes.sort_values(["measure", "absolute_change"], ascending=[True, False])
    .groupby("measure")
    .head(15)
)

trajectory_fips = (
    latest_counties.nlargest(3, "total_homeownership_cost_annual")["fips"].tolist()
    + latest_counties.nsmallest(3, "total_homeownership_cost_annual")["fips"].tolist()
)
trajectory_data = affordability_panel[
    affordability_panel["fips"].isin(trajectory_fips)
].copy()
trajectory_data["county"] = (
    trajectory_data["county_name"].fillna(trajectory_data["fips"])
    + " ("
    + trajectory_data["state_fips"].fillna("")
    + ")"
)
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=trajectory_data,
    x="year",
    y="total_homeownership_cost_annual",
    hue="county",
    marker="o",
)
plt.title("Total homeownership cost trajectories for selected high- and low-cost counties")
plt.ylabel("Nominal annual dollars")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
'''
    ),
]


NOTEBOOKS = {
    "acs_county_economic_annual_eda.ipynb": notebook(
        "ACS County Economic Annual EDA",
        "mart.acs_county_economic_annual",
        [
            "dp03_population_16_plus_est",
            "dp03_population_16_plus_in_labor_force_pct",
            "dp03_civilian_labor_force_unemployment_rate_pct",
            "dp03_income_and_benefits_median_household_income_est",
            "dp03_income_and_benefits_mean_household_income_est",
            "dp03_poverty_status_all_people_below_poverty_level_pct",
        ],
        "DP03 columns generally cover ACS economic characteristics such as employment, income, earnings, poverty, and insurance coverage.",
        "This mart pertains to county economic conditions. It includes labor-force participation, employment and unemployment, commuting and work status, household and family income, earnings, public assistance, poverty, and health-insurance coverage measures. Most variables are ACS DP03 estimate, margin-of-error, percent, and percent-margin fields by county and year.",
        extra_cells=ECONOMIC_SUMMARY_CELLS,
    ),
    "acs_county_demographic_annual_eda.ipynb": notebook(
        "ACS County Demographic Annual EDA",
        "mart.acs_county_demographic_annual",
        [
            "dp05_total_population_est",
            "dp05_total_population_median_age_est",
            "dp02_educational_attainment_population_25_plus_bachelors_degree_pct",
            "migration_population_1yr_plus",
            "same_house_1_year_ago",
            "moved_from_different_state",
        ],
        "DP02 columns generally cover selected social characteristics; DP05 columns cover demographic and housing estimates. Named migration columns are derived convenience fields.",
        "This mart pertains to county demographic and social characteristics. It includes household composition, family structure, relationship and marital status, school enrollment, educational attainment, veteran status, disability, residence one year ago, place of birth, citizenship, language, ancestry, age, sex, race, ethnicity, and population totals. It combines ACS DP02 and DP05 profile fields with named migration convenience variables by county and year.",
        extra_cells=DEMOGRAPHIC_SUMMARY_CELLS,
    ),
    "acs_county_affordability_annual_eda.ipynb": notebook(
        "ACS County Affordability Annual EDA",
        "mart.acs_county_affordability_annual",
        [
            "median_household_income",
            "median_gross_rent",
            "median_home_value",
            "median_property_taxes",
            "median_owner_costs_mortgage",
            "median_owner_costs_no_mortgage",
            "renter_30_to_34pct_income",
            "renter_35_to_39pct_income",
            "renter_40_to_49pct_income",
            "renter_50pct_plus_income",
            "owner_mortgage_30_to_34pct_income",
            "owner_mortgage_35_to_39pct_income",
            "owner_mortgage_40_to_49pct_income",
            "owner_mortgage_50pct_plus_income",
        ],
        "This mart combines derived affordability fields with ACS housing table families such as DP04, S2503, S2506, S2507, and B251xx.",
        "This mart pertains to county housing affordability and housing-cost burden. It includes derived fields such as median household income, median gross rent, median home value, owner costs with and without a mortgage, owner costs as a percent of income, renter cost-burden bands, and owner cost-burden bands. It also includes related ACS housing table families such as DP04, S2503, S2506, S2507, and B251xx by county and year.",
        extra_cells=AFFORDABILITY_BREAKDOWN_CELLS,
    ),
}


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    requested = set(sys.argv[1:])
    notebooks = {
        name: nb
        for name, nb in NOTEBOOKS.items()
        if not requested or name in requested
    }
    unknown = requested.difference(NOTEBOOKS)
    if unknown:
        raise SystemExit(f"Unknown notebook(s): {', '.join(sorted(unknown))}")
    for name, nb in notebooks.items():
        path = EDA_DIR / name
        path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
