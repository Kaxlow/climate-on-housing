"""Build EDA notebooks for source data files.

This script keeps the generated notebooks consistent and self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = ROOT / "scripts" / "eda"


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


def notebook(title: str, dataset: str, cells: list[dict]) -> dict:
    return {
        "cells": [
            md(
                f"""# {title}

Dataset: `{dataset}`

This notebook performs exploratory data analysis for structure, completeness, date coverage, geographic coverage, and high-signal distributions. Run cells from top to bottom after installing the project dependencies."""
            ),
            *cells,
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = r"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 80)
sns.set_theme(style="whitegrid")

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent.parent
"""

MISSING_HELPER = r"""
def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum()
    unique_values = []
    for col in frame.columns:
        try:
            unique_values.append(frame[col].nunique(dropna=True))
        except TypeError:
            unique_values.append(np.nan)
    return (
        pd.DataFrame({
            "missing_count": missing,
            "missing_pct": missing / len(frame) * 100,
            "dtype": frame.dtypes.astype(str),
            "unique_values": unique_values,
        })
        .sort_values("missing_pct", ascending=False)
    )
"""


disaster_cells = [
    md("## Setup"),
    code(SETUP + '\n\nDATA_PATH = ROOT / "data" / "fema" / "FEMA_Disaster_Declarations.csv"\nDATA_PATH'),
    md("## Load And Inspect"),
    code(
        r"""
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
date_cols = ["declarationDate", "incidentBeginDate", "incidentEndDate", "disasterCloseoutDate", "lastIAFilingDate", "lastRefresh"]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

display(df.head())
display(df.shape)
display(df.dtypes.to_frame("dtype"))
"""
    ),
    md("## Completeness And Keys"),
    code(MISSING_HELPER + "\n\nmissing_summary(df).head(30)"),
    code(
        r"""
key_cols = [c for c in ["id", "disasterNumber", "femaDeclarationString", "incidentId"] if c in df.columns]
pd.DataFrame({
    "column": key_cols,
    "unique_values": [df[c].nunique(dropna=True) for c in key_cols],
    "duplicate_rows_by_column": [df[c].duplicated().sum() for c in key_cols],
})
"""
    ),
    md("## Time Coverage"),
    code(
        r"""
date_overview = []
for col in date_cols:
    if col in df.columns:
        date_overview.append({
            "column": col,
            "min": df[col].min(),
            "max": df[col].max(),
            "missing_pct": df[col].isna().mean() * 100,
        })
pd.DataFrame(date_overview)
"""
    ),
    code(
        r"""
annual = df.assign(year=df["declarationDate"].dt.year).groupby("year", dropna=False).size()
ax = annual.plot(kind="line", marker="o", figsize=(12, 4), title="FEMA disaster declarations by declaration year")
ax.set_xlabel("Year")
ax.set_ylabel("Declarations")
plt.show()
"""
    ),
    md("## Geography And Incident Types"),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
df["state"].value_counts().head(20).plot(kind="bar", ax=axes[0], title="Top states by declaration count")
df["incidentType"].value_counts().head(20).plot(kind="bar", ax=axes[1], title="Top incident types")
for ax in axes:
    ax.set_ylabel("Count")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
pd.crosstab(df["state"], df["incidentType"]).assign(total=lambda x: x.sum(axis=1)).sort_values("total", ascending=False).head(20)
"""
    ),
    md("## Program Flags"),
    code(
        r"""
program_cols = [c for c in ["ihProgramDeclared", "iaProgramDeclared", "paProgramDeclared", "hmProgramDeclared"] if c in df.columns]
program_rates = df[program_cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=False)
ax = program_rates.plot(kind="bar", figsize=(7, 4), title="Share of declarations with each program flag")
ax.set_ylabel("Share")
plt.show()
program_rates.to_frame("share")
"""
    ),
]


nfip_cells = [
    md("## Setup"),
    code(
        SETUP
        + r'''
import duckdb

DATA_PATH = ROOT / "data" / "fema" / "FEMA_National_Flood_Insurance_Claims.csv"
csv_path = DATA_PATH.as_posix()
con = duckdb.connect()
con.execute("PRAGMA threads=4")
csv_rel = f"read_csv_auto('{csv_path}', sample_size=200000, ignore_errors=true)"
DATA_PATH
'''
    ),
    md("## Schema And Scale"),
    code(
        r"""
schema = con.execute(f"DESCRIBE SELECT * FROM {csv_rel}").df()
display(schema)
row_count = con.execute(f"SELECT count(*) AS rows FROM {csv_rel}").df()
row_count
"""
    ),
    code(
        r"""
sample = con.execute(f"SELECT * FROM {csv_rel} USING SAMPLE 100000 ROWS").df()
display(sample.head())
display(sample.dtypes.to_frame("dtype"))
"""
    ),
    md("## Completeness In Sample"),
    code(MISSING_HELPER + "\n\nmissing_summary(sample).head(35)"),
    md("## Claim Amounts"),
    code(
        r"""
amount_cols = [
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim",
    "netBuildingPaymentAmount",
    "netContentsPaymentAmount",
    "netIccPaymentAmount",
    "buildingDamageAmount",
    "contentsDamageAmount",
]
available_amount_cols = [c for c in amount_cols if c in sample.columns]
sample[available_amount_cols].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).T
"""
    ),
    code(
        r'''
claim_expr = "coalesce(amountPaidOnBuildingClaim, 0) + coalesce(amountPaidOnContentsClaim, 0) + coalesce(amountPaidOnIncreasedCostOfComplianceClaim, 0)"
annual_claims = con.execute(f"""
    SELECT
        yearOfLoss,
        count(*) AS claims,
        sum({claim_expr}) AS total_paid,
        avg({claim_expr}) AS avg_paid
    FROM {csv_rel}
    WHERE yearOfLoss IS NOT NULL
    GROUP BY yearOfLoss
    ORDER BY yearOfLoss
""").df()
display(annual_claims.tail(20))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
annual_claims.plot(x="yearOfLoss", y="claims", ax=axes[0], legend=False, title="NFIP claims by year")
annual_claims.plot(x="yearOfLoss", y="total_paid", ax=axes[1], legend=False, title="Total paid by year")
axes[0].set_ylabel("Claims")
axes[1].set_ylabel("Dollars")
plt.tight_layout()
plt.show()
'''
    ),
    md("## Geography"),
    code(
        r'''
state_summary = con.execute(f"""
    SELECT
        state,
        count(*) AS claims,
        sum({claim_expr}) AS total_paid,
        avg({claim_expr}) AS avg_paid
    FROM {csv_rel}
    WHERE state IS NOT NULL
    GROUP BY state
    ORDER BY total_paid DESC
    LIMIT 25
""").df()
display(state_summary)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
state_summary.sort_values("claims").plot(x="state", y="claims", kind="barh", ax=axes[0], legend=False, title="Top states by claims")
state_summary.sort_values("total_paid").plot(x="state", y="total_paid", kind="barh", ax=axes[1], legend=False, title="Top states by total paid")
plt.tight_layout()
plt.show()
'''
    ),
    code(
        r'''
con.execute(f"""
    SELECT
        state,
        reportedCity,
        count(*) AS claims,
        sum({claim_expr}) AS total_paid
    FROM {csv_rel}
    WHERE state IS NOT NULL AND reportedCity IS NOT NULL
    GROUP BY state, reportedCity
    ORDER BY total_paid DESC
    LIMIT 30
""").df()
'''
    ),
    md("## Flood Zone And Building Characteristics"),
    code(
        r"""
for col in ["floodZoneCurrent", "ratedFloodZone", "occupancyType", "causeOfDamage", "primaryResidenceIndicator"]:
    if col in sample.columns:
        display(sample[col].value_counts(dropna=False).head(20).to_frame("sample_count"))
"""
    ),
]


nri_cells = [
    md("## Setup"),
    code(SETUP + '\n\nDATA_PATH = ROOT / "data" / "fema" / "NRI_Table_Counties.csv"\nDATA_PATH'),
    md("## Load And Inspect"),
    code(
        r"""
df = pd.read_csv(DATA_PATH, low_memory=False)
display(df.head())
display(df.shape)
display(df.dtypes.value_counts().to_frame("column_count"))
"""
    ),
    md("## Completeness"),
    code(MISSING_HELPER + "\n\nmissing_summary(df).head(40)"),
    md("## Core Risk Measures"),
    code(
        r"""
core_cols = [
    "STATE", "STATEABBRV", "COUNTY", "STCOFIPS", "POPULATION", "BUILDVALUE", "AGRIVALUE", "AREA",
    "RISK_VALUE", "RISK_SCORE", "RISK_RATNG", "EAL_SCORE", "EAL_RATNG", "EAL_VALT",
    "SOVI_SCORE", "SOVI_RATNG", "RESL_SCORE", "RESL_RATNG",
]
core = df[[c for c in core_cols if c in df.columns]].copy()
display(core.describe(include="all").T)
"""
    ),
    code(
        r"""
rank_cols = [c for c in ["STATEABBRV", "COUNTY", "STCOFIPS", "POPULATION", "RISK_VALUE", "RISK_SCORE", "RISK_RATNG", "EAL_VALT", "SOVI_SCORE", "RESL_SCORE"] if c in df.columns]
df[rank_cols].sort_values("RISK_SCORE", ascending=False).head(25)
"""
    ),
    md("## Distributions And Ratings"),
    code(
        r"""
numeric_focus = [c for c in ["RISK_SCORE", "EAL_SCORE", "SOVI_SCORE", "RESL_SCORE", "POPULATION", "BUILDVALUE", "EAL_VALT"] if c in df.columns]
df[numeric_focus].hist(figsize=(14, 10), bins=35)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
rating_cols = [c for c in ["RISK_RATNG", "EAL_RATNG", "SOVI_RATNG", "RESL_RATNG"] if c in df.columns]
for col in rating_cols:
    display(df[col].value_counts(dropna=False).to_frame("count"))
"""
    ),
    md("## Hazard-Level Signals"),
    code(
        r"""
hazard_risk_cols = [c for c in df.columns if c.endswith("_RISKS")]
hazard_scores = df[hazard_risk_cols].mean(numeric_only=True).sort_values(ascending=False)
display(hazard_scores.to_frame("mean_risk_score").head(25))

ax = hazard_scores.head(20).sort_values().plot(kind="barh", figsize=(10, 7), title="Highest average hazard risk scores")
ax.set_xlabel("Mean NRI hazard risk score")
plt.show()
"""
    ),
    code(
        r"""
hazard_event_cols = [c for c in df.columns if c.endswith("_EVNTS")]
event_totals = df[hazard_event_cols].sum(numeric_only=True).sort_values(ascending=False)
display(event_totals.head(25).to_frame("total_events"))
"""
    ),
    md("## Relationships"),
    code(
        r"""
corr_cols = [c for c in ["RISK_SCORE", "EAL_SCORE", "SOVI_SCORE", "RESL_SCORE", "POPULATION", "BUILDVALUE", "AGRIVALUE", "AREA"] if c in df.columns]
corr = df[corr_cols].corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="vlag", center=0)
plt.title("Correlation among core county measures")
plt.show()
corr
"""
    ),
]


county_cells = [
    md("## Setup"),
    code(
        SETUP
        + '\n\nDATA_PATH = ROOT / "data" / "20260401_county_processed_data" '
        '/ "county_processed_data.feather"\nDATA_PATH'
    ),
    md("## Load And Inspect"),
    code(
        r"""
df = pd.read_feather(DATA_PATH)
display(df.head())
display(df.shape)
display(df.dtypes.to_frame("dtype"))
"""
    ),
    md("## Completeness And Identifiers"),
    code(MISSING_HELPER + "\n\nmissing_summary(df).head(30)"),
    code(
        r"""
id_cols = [c for c in ["fips", "county_name", "state", "state_long", "msa_code", "msa_name"] if c in df.columns]
pd.DataFrame({
    "column": id_cols,
    "unique_values": [df[c].nunique(dropna=True) for c in id_cols],
    "missing_pct": [df[c].isna().mean() * 100 for c in id_cols],
})
"""
    ),
    md("## Geographic Coverage"),
    code(
        r"""
state_counts = df["state"].value_counts().sort_values(ascending=False)
display(state_counts.to_frame("counties"))
ax = state_counts.plot(kind="bar", figsize=(14, 4), title="County records by state")
ax.set_ylabel("Counties")
plt.show()
"""
    ),
    md("## Nested Data Availability"),
    code(
        r"""
nested_cols = [
    "metrics", "nfip_claims", "property_tax", "insurance_premiums_14_to_24",
    "insurance_non_renewal_rates", "nri_climate", "temp_max_min",
    "fema_disaster_declarations", "storm_events",
]
nested_cols = [c for c in nested_cols if c in df.columns]

def is_non_empty_nested(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    if hasattr(value, "shape") and hasattr(value, "size"):
        return value.size > 0
    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return True

availability = []
for col in nested_cols:
    non_null = df[col].notna()
    non_empty = df[col].map(is_non_empty_nested)
    availability.append({
        "column": col,
        "non_null_count": int(non_null.sum()),
        "non_null_pct": non_null.mean() * 100,
        "non_empty_count": int(non_empty.sum()),
        "example_type": type(df.loc[non_null, col].iloc[0]).__name__ if non_null.any() else None,
    })
pd.DataFrame(availability).sort_values("non_null_pct", ascending=False)
"""
    ),
    code(
        r"""
for col in nested_cols:
    first = df.loc[df[col].notna(), col].head(1)
    if not first.empty:
        print(f"\n{col}: {type(first.iloc[0]).__name__}")
        display(first.iloc[0])
"""
    ),
    md("## Expand Dictionary-Like Columns"),
    code(
        r"""
def expand_dict_column(frame: pd.DataFrame, column: str, prefix: str | None = None) -> pd.DataFrame:
    prefix = prefix or column
    values = frame[column].map(lambda x: x if isinstance(x, dict) else {})
    expanded = pd.json_normalize(values).add_prefix(f"{prefix}.")
    expanded.index = frame.index
    return expanded

expanded_parts = []
for col in nested_cols:
    sample_values = df[col].dropna().head(20)
    if any(isinstance(x, dict) for x in sample_values):
        expanded_parts.append(expand_dict_column(df, col))

expanded = pd.concat(expanded_parts, axis=1) if expanded_parts else pd.DataFrame(index=df.index)
display(expanded.shape)
display(expanded.head())
"""
    ),
    md("## Numeric Feature Overview"),
    code(
        r"""
numeric_base = df.select_dtypes(include="number")
numeric_expanded = expanded.select_dtypes(include="number") if not expanded.empty else pd.DataFrame(index=df.index)
numeric = pd.concat([numeric_base, numeric_expanded], axis=1)

summary = numeric.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
display(summary.sort_values("std", ascending=False).head(40))
"""
    ),
    code(
        r"""
candidate_cols = [c for c in numeric.columns if any(token in c.lower() for token in ["risk", "claim", "premium", "tax", "population", "income", "value"])]
plot_cols = candidate_cols[:12]
if plot_cols:
    numeric[plot_cols].hist(figsize=(14, 10), bins=30)
    plt.tight_layout()
    plt.show()
else:
    print("No matching numeric columns found for quick histograms.")
"""
    ),
    md("## MSA And Redfin Coverage"),
    code(
        r"""
if "has_redfin_data" in df.columns:
    display(df["has_redfin_data"].value_counts(dropna=False).to_frame("count"))
if "msa_type" in df.columns:
    display(df["msa_type"].value_counts(dropna=False).to_frame("count"))
if "msa_name" in df.columns:
    display(df["msa_name"].value_counts(dropna=True).head(25).to_frame("counties"))
"""
    ),
    md("## Nested Exploration Helpers"),
    code(
        r"""
def first_non_null(column: str):
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else None

def dict_scalar_frame(column: str) -> pd.DataFrame:
    rows = []
    for _, row in df[["fips", "county_name", "state", column]].dropna(subset=[column]).iterrows():
        value = row[column]
        if not isinstance(value, dict):
            continue
        flat = {"fips": row["fips"], "county_name": row["county_name"], "state": row["state"]}
        for key, item in value.items():
            if isinstance(item, dict):
                if "value" in item:
                    flat[key] = item.get("value")
                for pct_key, pct_value in item.get("percentiles", {}).items():
                    flat[f"{key}.{pct_key}"] = pct_value
            elif np.isscalar(item) or item is None:
                flat[key] = item
        rows.append(flat)
    return pd.DataFrame(rows)

def records_from_array_column(column: str) -> pd.DataFrame:
    rows = []
    for _, row in df[["fips", "county_name", "state", column]].dropna(subset=[column]).iterrows():
        values = row[column]
        if isinstance(values, dict):
            values = [values]
        for item in values:
            if isinstance(item, dict):
                rows.append({"fips": row["fips"], "county_name": row["county_name"], "state": row["state"], **item})
    return pd.DataFrame(rows)

def nested_numeric_summary(frame: pd.DataFrame, id_cols=("fips", "county_name", "state")) -> pd.DataFrame:
    numeric = frame.drop(columns=[c for c in id_cols if c in frame.columns], errors="ignore").select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.sort_values("std", ascending=False)

def completeness_report(frame: pd.DataFrame, id_cols=("fips", "county_name", "state")) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.drop(columns=[c for c in id_cols if c in frame.columns], errors="ignore")
    missing = data.isna().sum()
    return (
        pd.DataFrame({
            "non_null_count": data.notna().sum(),
            "missing_count": missing,
            "missing_pct": missing / len(data) * 100,
            "dtype": data.dtypes.astype(str),
        })
        .sort_values(["missing_pct", "missing_count"], ascending=False)
    )
"""
    ),
    md("## Nested Column: metrics"),
    code(
        r"""
metrics_example = first_non_null("metrics")
print(f"Metric keys: {len(metrics_example)}")
display(pd.Series({k: type(v).__name__ for k, v in metrics_example.items()}).to_frame("type").head(60))

metric_lengths = []
for value in df["metrics"].dropna():
    if isinstance(value, dict):
        metric_lengths.append({k: len(v) for k, v in value.items() if hasattr(v, "__len__") and not isinstance(v, str)})
metric_lengths = pd.DataFrame(metric_lengths)
display(completeness_report(metric_lengths).head(60))
display(metric_lengths.describe().T.head(60))
"""
    ),
    code(
        r"""
metrics_rows = []
selected_metrics = [
    "MEDIAN_SALE_PRICE", "MEDIAN_LIST_PRICE", "HOMES_SOLD", "INVENTORY",
    "MONTHS_OF_SUPPLY", "MEDIAN_DOM", "SOLD_ABOVE_LIST", "PRICE_DROPS",
]
for _, row in df[["fips", "county_name", "state", "metrics"]].dropna(subset=["metrics"]).iterrows():
    value = row["metrics"]
    if not isinstance(value, dict) or "dates" not in value:
        continue
    dates = pd.to_datetime(value["dates"], errors="coerce")
    for metric in selected_metrics:
        if metric in value:
            metric_values = np.asarray(value[metric], dtype="float64")
            n = min(len(dates), len(metric_values))
            metrics_rows.extend(
                {
                    "fips": row["fips"],
                    "county_name": row["county_name"],
                    "state": row["state"],
                    "date": dates[i],
                    "metric": metric,
                    "value": metric_values[i],
                }
                for i in range(n)
            )

metrics_long = pd.DataFrame(metrics_rows)
display(metrics_long.head())
display(completeness_report(metrics_long))
display(metrics_long.groupby("metric")["value"].describe().sort_values("count", ascending=False))
"""
    ),
    code(
        r"""
if not metrics_long.empty:
    latest_metrics = (
        metrics_long.dropna(subset=["date"])
        .sort_values("date")
        .groupby(["fips", "county_name", "state", "metric"], as_index=False)
        .tail(1)
    )
    latest_wide = (
        latest_metrics.pivot(index=["fips", "county_name", "state"], columns="metric", values="value")
        .reset_index()
        .rename_axis(columns=None)
    )
    display(latest_wide.head())
    display(latest_wide.describe().T)

    if "MEDIAN_SALE_PRICE" in latest_wide.columns:
        top_price_cols = [c for c in ["fips", "county_name", "state", "MEDIAN_SALE_PRICE", "HOMES_SOLD", "INVENTORY"] if c in latest_wide.columns]
        top_price = latest_wide.sort_values("MEDIAN_SALE_PRICE", ascending=False).head(20)
        display(top_price[top_price_cols])
"""
    ),
    md("## Nested Column: nfip_claims"),
    code(
        r"""
nfip_example = first_non_null("nfip_claims")
display(pd.Series({k: type(v).__name__ for k, v in nfip_example.items()}).to_frame("type"))

nfip_rows = []
for _, row in df[["fips", "county_name", "state", "nfip_claims"]].dropna(subset=["nfip_claims"]).iterrows():
    value = row["nfip_claims"]
    if not isinstance(value, dict):
        continue
    year_values = np.asarray(value.get("yearOfLoss", []))
    n = len(year_values)
    for i in range(n):
        record = {"fips": row["fips"], "county_name": row["county_name"], "state": row["state"], "yearOfLoss": year_values[i]}
        for key, item in value.items():
            if key == "yearOfLoss" or not hasattr(item, "__len__"):
                continue
            if i < len(item):
                record[key] = item[i]
        nfip_rows.append(record)
nfip_long = pd.DataFrame(nfip_rows)
display(nfip_long.head())
display(nfip_long.shape)
display(completeness_report(nfip_long).head(40))
"""
    ),
    code(
        r"""
if not nfip_long.empty:
    claim_amount_cols = [
        "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
        "amountPaidOnIncreasedCostOfComplianceClaim", "netBuildingPaymentAmount", "netContentsPaymentAmount",
    ]
    claim_amount_cols = [c for c in claim_amount_cols if c in nfip_long.columns]
    nfip_long["total_paid"] = nfip_long[claim_amount_cols].fillna(0).sum(axis=1)
    display(nfip_long[["yearOfLoss", "total_paid", *claim_amount_cols]].describe().T)
    display(
        nfip_long.groupby(["fips", "county_name", "state"], as_index=False)
        .agg(claim_records=("yearOfLoss", "size"), total_paid=("total_paid", "sum"), avg_paid=("total_paid", "mean"))
        .sort_values("total_paid", ascending=False)
        .head(25)
    )
    annual_nfip = nfip_long.groupby("yearOfLoss", as_index=False).agg(claim_records=("total_paid", "size"), total_paid=("total_paid", "sum"))
    annual_nfip.plot(x="yearOfLoss", y="total_paid", figsize=(12, 4), title="Nested NFIP total paid by year")
    plt.show()
"""
    ),
    md("## Nested Column: property_tax"),
    code(
        r"""
property_tax_flat = dict_scalar_frame("property_tax")
display(property_tax_flat.head())
display(completeness_report(property_tax_flat))
display(nested_numeric_summary(property_tax_flat))

value_cols = [c for c in property_tax_flat.columns if c.endswith("_2024") or c.endswith("_2023_2024")]
if value_cols:
    display(property_tax_flat[["fips", "county_name", "state", *value_cols]].sort_values(value_cols[0], ascending=False).head(25))
    property_tax_flat[value_cols].hist(figsize=(12, 4), bins=30)
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## Nested Column: insurance_premiums_14_to_24"),
    code(
        r"""
premium_rows = []
premium_history_rows = []
for _, row in df[["fips", "county_name", "state", "insurance_premiums_14_to_24"]].dropna(subset=["insurance_premiums_14_to_24"]).iterrows():
    value = row["insurance_premiums_14_to_24"]
    if not isinstance(value, dict):
        continue
    flat = {"fips": row["fips"], "county_name": row["county_name"], "state": row["state"]}
    for section in ["averages", "growth_rates", "latest"]:
        for key, item in value.get(section, {}).items():
            if isinstance(item, dict):
                for pct_key, pct_value in item.items():
                    flat[f"{section}.{key}.{pct_key}"] = pct_value
            elif np.isscalar(item) or item is None:
                flat[f"{section}.{key}"] = item
    premium_rows.append(flat)

    historical = value.get("historical", {})
    years = np.arange(value.get("averages", {}).get("start_year", 2014), value.get("averages", {}).get("end_year", 2024) + 1)
    for metric in ["mean", "median"]:
        values = historical.get(metric)
        if values is None:
            continue
        for year, premium in zip(years, values):
            premium_history_rows.append({"fips": row["fips"], "county_name": row["county_name"], "state": row["state"], "year": year, "metric": metric, "premium": premium})

premium_flat = pd.DataFrame(premium_rows)
premium_history = pd.DataFrame(premium_history_rows)
display(premium_flat.head())
display(completeness_report(premium_flat).head(40))
display(completeness_report(premium_history).head(20))
display(nested_numeric_summary(premium_flat).head(30))
"""
    ),
    code(
        r"""
if not premium_history.empty:
    display(premium_history.groupby(["year", "metric"])["premium"].describe().tail(12))
    premium_trend = premium_history.groupby(["year", "metric"], as_index=False)["premium"].median()
    sns.lineplot(data=premium_trend, x="year", y="premium", hue="metric")
    plt.title("Median county insurance premium history")
    plt.show()

latest_mean_col = "latest.mean"
if latest_mean_col in premium_flat.columns:
    display(premium_flat.sort_values(latest_mean_col, ascending=False).head(25)[["fips", "county_name", "state", latest_mean_col]])
"""
    ),
    md("## Nested Column: insurance_non_renewal_rates"),
    code(
        r"""
nonrenewal_rows = []
nonrenewal_history_rows = []
for _, row in df[["fips", "county_name", "state", "insurance_non_renewal_rates"]].dropna(subset=["insurance_non_renewal_rates"]).iterrows():
    value = row["insurance_non_renewal_rates"]
    if not isinstance(value, dict):
        continue
    flat = {"fips": row["fips"], "county_name": row["county_name"], "state": row["state"], "years_of_data": value.get("years_of_data")}
    for section in ["averages", "growth_rates", "latest"]:
        for key, item in value.get(section, {}).items():
            if isinstance(item, dict):
                for pct_key, pct_value in item.items():
                    flat[f"{section}.{key}.{pct_key}"] = pct_value
            elif np.isscalar(item) or item is None:
                flat[f"{section}.{key}"] = item
    nonrenewal_rows.append(flat)

    historical = value.get("historical", {})
    end_year = value.get("averages", {}).get("end_year", 2023)
    n_years = len(next(iter(historical.values()))) if historical else 0
    years = np.arange(end_year - n_years + 1, end_year + 1)
    for metric, values in historical.items():
        for year, metric_value in zip(years, values):
            nonrenewal_history_rows.append({"fips": row["fips"], "county_name": row["county_name"], "state": row["state"], "year": year, "metric": metric, "value": metric_value})

nonrenewal_flat = pd.DataFrame(nonrenewal_rows)
nonrenewal_history = pd.DataFrame(nonrenewal_history_rows)
display(nonrenewal_flat.head())
display(completeness_report(nonrenewal_flat).head(40))
display(completeness_report(nonrenewal_history).head(20))
display(nested_numeric_summary(nonrenewal_flat).head(30))
"""
    ),
    code(
        r"""
if not nonrenewal_history.empty:
    rate_history = nonrenewal_history[nonrenewal_history["metric"].eq("non_renewal_rate")]
    display(rate_history.groupby("year")["value"].describe())
    rate_history.groupby("year")["value"].median().plot(figsize=(10, 4), marker="o", title="Median county non-renewal rate")
    plt.show()

latest_rate_col = "latest.non_renewal_rate"
if latest_rate_col in nonrenewal_flat.columns:
    display(nonrenewal_flat.sort_values(latest_rate_col, ascending=False).head(25)[["fips", "county_name", "state", latest_rate_col, "latest.num_policies_total"]])
"""
    ),
    md("## Nested Column: nri_climate"),
    code(
        r"""
nri_flat = dict_scalar_frame("nri_climate")
display(nri_flat.head())
display(completeness_report(nri_flat).head(40))
display(nested_numeric_summary(nri_flat).head(30))

nri_risk_rows = []
for _, row in df[["fips", "county_name", "state", "nri_climate"]].dropna(subset=["nri_climate"]).iterrows():
    value = row["nri_climate"]
    if not isinstance(value, dict):
        continue
    for item in value.get("RISKS_BREAKDOWN", []):
        if isinstance(item, dict):
            flat = {"fips": row["fips"], "county_name": row["county_name"], "state": row["state"]}
            for key, val in item.items():
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        flat[f"{key}.{sub_key}"] = sub_val
                else:
                    flat[key] = val
            nri_risk_rows.append(flat)
nri_risks = pd.DataFrame(nri_risk_rows)
display(nri_risks.head())
display(nri_risks.shape)
display(completeness_report(nri_risks).head(40))
"""
    ),
    code(
        r"""
if not nri_risks.empty:
    score_col = "risk_score" if "risk_score" in nri_risks.columns else None
    if score_col:
        display(nri_risks.groupby("risk")[score_col].describe().sort_values("mean", ascending=False).head(25))
        top_risks = nri_risks.groupby("risk")[score_col].mean().sort_values(ascending=False).head(20)
        top_risks.sort_values().plot(kind="barh", figsize=(10, 6), title="Average NRI risk score by hazard")
        plt.show()
    display(nri_flat.sort_values("RISK_SCORE", ascending=False).head(25)[["fips", "county_name", "state", "RISK_SCORE", "RISK_RATNG", "RISK_VALUE"]])
"""
    ),
    md("## Nested Column: temp_max_min"),
    code(
        r"""
temp_records = records_from_array_column("temp_max_min")
display(temp_records.head())
display(temp_records.shape)
display(completeness_report(temp_records).head(60))
display(nested_numeric_summary(temp_records).head(40))
"""
    ),
    code(
        r"""
if not temp_records.empty:
    trend_cols = [c for c in temp_records.columns if "trend_slope" in c and c.endswith("_f")]
    display(temp_records.groupby("month")[trend_cols].median().head(12) if trend_cols else "No Fahrenheit trend columns found")
    avg_cols = [c for c in ["tmax_average_temp_f", "tmin_average_temp_f"] if c in temp_records.columns]
    if avg_cols:
        monthly = temp_records.groupby("month")[avg_cols].median()
        monthly.plot(figsize=(10, 4), marker="o", title="Median county monthly temperatures")
        plt.ylabel("Degrees F")
        plt.show()
"""
    ),
    md("## Nested Column: fema_disaster_declarations"),
    code(
        r"""
fema_rows = []
fema_breakdown_rows = []
for _, row in df[["fips", "county_name", "state", "fema_disaster_declarations"]].dropna(subset=["fema_disaster_declarations"]).iterrows():
    value = row["fema_disaster_declarations"]
    if not isinstance(value, dict):
        continue
    years = value.get("fyDeclared", [])
    summary = value.get("summary", [])
    for year, count in zip(years, summary):
        fema_rows.append({"fips": row["fips"], "county_name": row["county_name"], "state": row["state"], "fyDeclared": year, "declarations": count})
    for incident_type, counts in value.get("breakdown", {}).items():
        for year, count in zip(years, counts):
            fema_breakdown_rows.append({"fips": row["fips"], "county_name": row["county_name"], "state": row["state"], "fyDeclared": year, "incident_type": incident_type, "declarations": count})

fema_summary = pd.DataFrame(fema_rows)
fema_breakdown = pd.DataFrame(fema_breakdown_rows)
display(fema_summary.head())
display(fema_breakdown.head())
display(completeness_report(fema_summary))
display(completeness_report(fema_breakdown))
"""
    ),
    code(
        r"""
if not fema_summary.empty:
    display(fema_summary.groupby("fyDeclared")["declarations"].sum().tail(20).to_frame("declarations"))
    county_disasters = fema_summary.groupby(["fips", "county_name", "state"], as_index=False)["declarations"].sum().sort_values("declarations", ascending=False)
    display(county_disasters.head(25))
    fema_summary.groupby("fyDeclared")["declarations"].sum().plot(figsize=(12, 4), marker="o", title="Nested FEMA declarations by fiscal year")
    plt.show()

if not fema_breakdown.empty:
    display(fema_breakdown.groupby("incident_type")["declarations"].sum().sort_values(ascending=False).head(25).to_frame("declarations"))
"""
    ),
    md("## Nested Column: storm_events"),
    code(
        r"""
storm_records = records_from_array_column("storm_events")
display(storm_records.head())
display(storm_records.shape)
display(completeness_report(storm_records).head(60))
display(nested_numeric_summary(storm_records).head(40))
"""
    ),
    code(
        r"""
if not storm_records.empty:
    damage_col = "total_damage_total"
    event_col = "total_events" if "total_events" in storm_records.columns else None
    group_cols = ["event_type"] if "event_type" in storm_records.columns else []
    if group_cols and damage_col in storm_records.columns:
        display(storm_records.groupby(group_cols)[damage_col].sum().sort_values(ascending=False).head(25).to_frame("total_damage"))
    if damage_col in storm_records.columns:
        display(
            storm_records.groupby(["fips", "county_name", "state"], as_index=False)[damage_col]
            .sum()
            .sort_values(damage_col, ascending=False)
            .head(25)
        )
    if event_col:
        display(storm_records.groupby(group_cols)[event_col].sum().sort_values(ascending=False).head(25).to_frame("events") if group_cols else storm_records[event_col].describe())
"""
    ),
]


NOTEBOOKS = {
    "fema_disaster_declarations_eda.ipynb": notebook(
        "FEMA Disaster Declarations EDA",
        "data/fema/FEMA_Disaster_Declarations.csv",
        disaster_cells,
    ),
    "fema_national_flood_insurance_claims_eda.ipynb": notebook(
        "FEMA National Flood Insurance Claims EDA",
        "data/fema/FEMA_National_Flood_Insurance_Claims.csv",
        nfip_cells,
    ),
    "nri_table_counties_eda.ipynb": notebook(
        "NRI Table Counties EDA",
        "data/fema/NRI_Table_Counties.csv",
        nri_cells,
    ),
    "county_processed_data_eda.ipynb": notebook(
        "County Processed Data EDA",
        "data/20260401_county_processed_data/county_processed_data.feather",
        county_cells,
    ),
}


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        path = EDA_DIR / name
        path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
