"""Build EDA notebook for the NCEI county monthly weather mart."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = ROOT / "scripts" / "eda"
NOTEBOOK_PATH = EDA_DIR / "ncei_county_weather_monthly_eda.ipynb"
CELL_IDS = count(1)


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"ncei-weather-md-{next(CELL_IDS):02d}",
        "metadata": {},
        "source": source.strip().splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"ncei-weather-code-{next(CELL_IDS):02d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


NOTEBOOK = {
    "cells": [
        md(
            """
# NCEI County Monthly Weather Mart EDA

Dataset: `mart.ncei_county_weather_monthly` in `data/quoll.duckdb`.

This notebook explores county-level monthly NOAA NCEI Climate at a Glance temperature and precipitation data for the latest downloaded ten-calendar-year window.
"""
        ),
        md("## Setup"),
        code(
            r"""
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)
sns.set_theme(style="whitegrid", context="notebook")

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent.parent

DB_PATH = ROOT / "data" / "quoll.duckdb"
TABLE = "mart.ncei_county_weather_monthly"
con = duckdb.connect(str(DB_PATH), read_only=True)


def q(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def pct(value: float) -> str:
    return f"{value:.1%}"


DB_PATH
"""
        ),
        md("## Shape And Schema"),
        code(
            r"""
row_count = q(f"SELECT count(*) AS rows FROM {TABLE}")
schema = q(f"DESCRIBE {TABLE}")
display(row_count)
display(schema)

sample = q(f'''
    SELECT *
    FROM {TABLE}
    ORDER BY weather_month DESC, fips
    LIMIT 10
''')
display(sample)
"""
        ),
        md("## Coverage And Key Integrity"),
        code(
            r"""
coverage = q(f'''
    SELECT
        min(weather_month) AS min_weather_month,
        max(weather_month) AS max_weather_month,
        count(DISTINCT weather_month) AS month_count,
        count(DISTINCT year) AS year_count,
        count(DISTINCT fips) AS county_count,
        count(DISTINCT state_fips) AS state_count,
        count(*) AS rows,
        count(*) FILTER (WHERE observed_parameter_count = 4) AS complete_parameter_rows
    FROM {TABLE}
''')
coverage["complete_parameter_row_pct"] = coverage["complete_parameter_rows"] / coverage["rows"]
display(coverage)

key_check = q(f'''
    SELECT
        count(*) AS rows,
        count(DISTINCT fips || '-' || cast(weather_month AS VARCHAR)) AS distinct_county_months,
        count(*) - count(DISTINCT fips || '-' || cast(weather_month AS VARCHAR)) AS duplicate_county_month_rows,
        count(*) FILTER (WHERE fips IS NULL) AS missing_fips,
        count(*) FILTER (WHERE weather_month IS NULL) AS missing_weather_month
    FROM {TABLE}
''')
display(key_check)

dupes = q(f'''
    SELECT fips, weather_month, count(*) AS rows
    FROM {TABLE}
    GROUP BY fips, weather_month
    HAVING count(*) > 1
    ORDER BY rows DESC, fips, weather_month
    LIMIT 25
''')
display(dupes)
"""
        ),
        code(
            r"""
monthly_coverage = q(f'''
    SELECT
        weather_month,
        count(*) AS rows,
        count(DISTINCT fips) AS counties,
        count(*) FILTER (WHERE observed_parameter_count = 4) AS complete_parameter_counties
    FROM {TABLE}
    GROUP BY weather_month
    ORDER BY weather_month
''')
display(monthly_coverage)

fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(data=monthly_coverage, x="weather_month", y="counties", marker="o", ax=ax)
ax.set_title("County coverage by weather month")
ax.set_xlabel("Weather month")
ax.set_ylabel("Counties")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
        ),
        md("## Missingness And Distributions"),
        code(
            r"""
measures = [
    "avg_temperature_f",
    "min_temperature_f",
    "max_temperature_f",
    "precipitation_inches",
    "avg_temperature_anomaly_f",
    "min_temperature_anomaly_f",
    "max_temperature_anomaly_f",
    "precipitation_anomaly_inches",
]

missing_parts = []
for column in measures:
    missing_parts.append(
        f"SELECT '{column}' AS column_name, "
        f"count(*) AS rows, "
        f"count({column}) AS non_null_rows, "
        f"count(*) - count({column}) AS missing_rows, "
        f"avg({column}) AS mean, "
        f"median({column}) AS median, "
        f"min({column}) AS min, "
        f"quantile_cont({column}, 0.95) AS p95, "
        f"max({column}) AS max "
        f"FROM {TABLE}"
    )
missing = q(" UNION ALL ".join(missing_parts))
missing["missing_pct"] = missing["missing_rows"] / missing["rows"]
display(missing.sort_values("missing_pct", ascending=False))
"""
        ),
        code(
            r"""
dist = q(f'''
    SELECT
        avg_temperature_f,
        min_temperature_f,
        max_temperature_f,
        precipitation_inches,
        avg_temperature_anomaly_f,
        precipitation_anomaly_inches
    FROM {TABLE}
''')

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, column in zip(axes.flat, dist.columns):
    data = dist[column].dropna()
    if data.empty:
        ax.set_visible(False)
        continue
    upper = data.quantile(0.995)
    lower = data.quantile(0.005)
    sns.histplot(data.clip(lower, upper), bins=40, ax=ax, color="#4C78A8")
    ax.set_title(column)
plt.tight_layout()
plt.show()
"""
        ),
        md("## Seasonal Patterns"),
        code(
            r"""
monthly_summary = q(f'''
    SELECT
        month,
        count(*) AS rows,
        avg(avg_temperature_f) AS mean_avg_temperature_f,
        avg(precipitation_inches) AS mean_precipitation_inches,
        median(avg_temperature_anomaly_f) AS median_avg_temperature_anomaly_f,
        median(precipitation_anomaly_inches) AS median_precipitation_anomaly_inches
    FROM {TABLE}
    GROUP BY month
    ORDER BY month
''')
display(monthly_summary)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.lineplot(data=monthly_summary, x="month", y="mean_avg_temperature_f", marker="o", ax=axes[0], color="#E45756")
axes[0].set_title("Mean average temperature by calendar month")
axes[0].set_ylabel("Degrees F")
sns.lineplot(data=monthly_summary, x="month", y="mean_precipitation_inches", marker="o", ax=axes[1], color="#4C78A8")
axes[1].set_title("Mean precipitation by calendar month")
axes[1].set_ylabel("Inches")
for ax in axes:
    ax.set_xticks(range(1, 13))
plt.tight_layout()
plt.show()
"""
        ),
        md("## State-Year Summaries"),
        code(
            r"""
state_year = q(f'''
    SELECT
        state,
        state_long,
        year,
        count(DISTINCT fips) AS counties,
        avg(avg_temperature_f) AS mean_avg_temperature_f,
        sum(precipitation_inches) / nullif(count(DISTINCT fips), 0) AS mean_county_annual_precipitation_inches,
        avg(avg_temperature_anomaly_f) AS mean_avg_temperature_anomaly_f,
        avg(precipitation_anomaly_inches) AS mean_precipitation_anomaly_inches
    FROM {TABLE}
    GROUP BY state, state_long, year
    ORDER BY state, year
''')
display(state_year.head(20))

latest_year = int(state_year["year"].max())
latest = state_year[state_year["year"].eq(latest_year)].copy()
display(latest.sort_values("mean_avg_temperature_anomaly_f", ascending=False).head(15))
display(latest.sort_values("mean_precipitation_anomaly_inches", ascending=False).head(15))
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
temp_plot = latest.dropna(subset=["mean_avg_temperature_anomaly_f"]).nlargest(20, "mean_avg_temperature_anomaly_f")
precip_plot = latest.dropna(subset=["mean_precipitation_anomaly_inches"]).nlargest(20, "mean_precipitation_anomaly_inches")

sns.barplot(data=temp_plot, y="state", x="mean_avg_temperature_anomaly_f", ax=axes[0], color="#E45756")
axes[0].set_title(f"Warmest average temperature anomalies by state, {latest_year}")
axes[0].set_xlabel("Mean county monthly anomaly, deg F")
axes[0].set_ylabel("")

sns.barplot(data=precip_plot, y="state", x="mean_precipitation_anomaly_inches", ax=axes[1], color="#4C78A8")
axes[1].set_title(f"Wettest precipitation anomalies by state, {latest_year}")
axes[1].set_xlabel("Mean county monthly anomaly, inches")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
"""
        ),
        md("## County Extremes"),
        code(
            r"""
county_extremes = q(f'''
    SELECT
        fips,
        county_name,
        state,
        min(avg_temperature_f) AS min_avg_temperature_f,
        max(avg_temperature_f) AS max_avg_temperature_f,
        max(precipitation_inches) AS max_monthly_precipitation_inches,
        avg(avg_temperature_anomaly_f) AS mean_avg_temperature_anomaly_f,
        avg(precipitation_anomaly_inches) AS mean_precipitation_anomaly_inches
    FROM {TABLE}
    GROUP BY fips, county_name, state
''')

display(county_extremes.sort_values("max_avg_temperature_f", ascending=False).head(20))
display(county_extremes.sort_values("min_avg_temperature_f").head(20))
display(county_extremes.sort_values("max_monthly_precipitation_inches", ascending=False).head(20))
"""
        ),
        md("## Join Readiness"),
        code(
            r"""
join_check = q(f'''
    SELECT
        count(*) AS weather_rows,
        count(*) FILTER (WHERE redfin.fips IS NOT NULL) AS rows_with_redfin_same_month,
        count(*) FILTER (WHERE storm.fips IS NOT NULL) AS rows_with_noaa_storm_same_month
    FROM {TABLE} AS weather
    LEFT JOIN mart.redfin_county_monthly AS redfin
        ON weather.fips = redfin.fips
       AND weather.weather_month = date_trunc('month', redfin.period_begin)
       AND redfin.property_type = 'All Residential'
    LEFT JOIN mart.noaa_storm_events AS storm
        ON weather.fips = storm.fips
       AND weather.year = storm.event_year
       AND weather.month = storm.event_month
''')
for column in ["rows_with_redfin_same_month", "rows_with_noaa_storm_same_month"]:
    join_check[f"{column}_pct"] = join_check[column] / join_check["weather_rows"]
display(join_check)
"""
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(NOTEBOOK, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
