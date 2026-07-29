"""Build NRI grouped ACS county feature distribution notebook."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = ROOT / "scripts" / "eda"
NOTEBOOK_PATH = EDA_DIR / "nri_acs_county_feature_distributions.ipynb"
CELL_IDS = count(1)


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"nri-acs-md-{next(CELL_IDS):02d}",
        "metadata": {},
        "source": source.strip().splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"nri-acs-code-{next(CELL_IDS):02d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


NOTEBOOK = {
    "cells": [
        md(
            """
# NRI Risk Rating And ACS County Feature Distributions

Dataset: `mart.nri_county_risk`, `mart.acs_county_affordability_annual`, `mart.acs_county_demographic_annual`, and `mart.acs_county_economic_annual` in `data/quoll.duckdb`.

This notebook plots the latest available county-level economic and demographic feature values grouped by each county's FEMA NRI risk rating.
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

pd.set_option("display.max_columns", 160)
pd.set_option("display.max_rows", 120)
sns.set_theme(style="whitegrid", context="notebook")

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent.parent

DB_PATH = ROOT / "data" / "quoll.duckdb"
con = duckdb.connect(str(DB_PATH), read_only=True)

RISK_ORDER = [
    "Very Low",
    "Relatively Low",
    "Relatively Moderate",
    "Relatively High",
    "Very High",
]

PALETTE = {
    "Very Low": "#4C78A8",
    "Relatively Low": "#72B7B2",
    "Relatively Moderate": "#F2CF5B",
    "Relatively High": "#F58518",
    "Very High": "#E45756",
}

MISSING_SENTINELS = [
    "",
    "N",
    "-",
    "(X)",
    "-666666666",
    "-666666666.0",
    "-888888888",
    "-888888888.0",
    "-999999999",
    "-999999999.0",
]


def q(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(MISSING_SENTINELS, np.nan), errors="coerce")


def safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    return numerator.where(denominator.gt(0)).div(denominator.where(denominator.gt(0))) * scale


def weighted_midpoint(row: pd.Series, bins: list[tuple[str, float]]) -> float:
    weights = np.array([row.get(column, np.nan) for column, _ in bins], dtype="float64")
    midpoints = np.array([midpoint for _, midpoint in bins], dtype="float64")
    valid = np.isfinite(weights) & np.isfinite(midpoints) & (weights >= 0)
    if not valid.any() or weights[valid].sum() <= 0:
        return np.nan
    return float(np.average(midpoints[valid], weights=weights[valid]))


def plot_distribution(
    frame: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    *,
    percent: bool = False,
    dollars: bool = False,
) -> None:
    plot_data = frame.dropna(subset=["risk_rating", column]).copy()
    plot_data = plot_data[plot_data["risk_rating"].isin(RISK_ORDER)]
    plot_data = plot_data[np.isfinite(plot_data[column])]
    if plot_data.empty:
        print(f"No valid observations to plot for {title}.")
        return

    lower = plot_data[column].quantile(0.01)
    upper = plot_data[column].quantile(0.99)
    if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
        plot_data["_plot_value"] = plot_data[column].clip(lower=lower, upper=upper)
        range_note = f"Display range clipped to 1st-99th percentile: {lower:,.2f} to {upper:,.2f}."
    else:
        plot_data["_plot_value"] = plot_data[column]
        range_note = "Display range uses full valid range."

    plot_data["risk_rating"] = pd.Categorical(plot_data["risk_rating"], categories=RISK_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxenplot(
        data=plot_data,
        x="risk_rating",
        y="_plot_value",
        hue="risk_rating",
        order=RISK_ORDER,
        hue_order=RISK_ORDER,
        palette=PALETTE,
        linewidth=0.8,
        showfliers=False,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_data.sample(min(len(plot_data), 900), random_state=7),
        x="risk_rating",
        y="_plot_value",
        order=RISK_ORDER,
        color="#222222",
        alpha=0.22,
        jitter=0.22,
        size=2,
        ax=ax,
    )
    ax.set_title(title)
    ax.text(0, 1.02, range_note, transform=ax.transAxes, fontsize=9, color="#555555")
    ax.set_xlabel("NRI risk rating")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=22)
    if percent:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    if dollars:
        ax.yaxis.set_major_formatter(lambda value, _: f"${value:,.0f}")
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    plt.show()
"""
        ),
        md("## Load Latest County Features"),
        code(
            r'''
latest_years = q("""
    SELECT 'affordability' AS mart, max(year) AS latest_year FROM mart.acs_county_affordability_annual
    UNION ALL
    SELECT 'demographic' AS mart, max(year) AS latest_year FROM mart.acs_county_demographic_annual
    UNION ALL
    SELECT 'economic' AS mart, max(year) AS latest_year FROM mart.acs_county_economic_annual
""")
display(latest_years)
'''
        ),
        code(
            r'''
features = q("""
WITH
afford AS (
    SELECT *
    FROM mart.acs_county_affordability_annual
    WHERE year = (SELECT max(year) FROM mart.acs_county_affordability_annual)
),
demo AS (
    SELECT *
    FROM mart.acs_county_demographic_annual
    WHERE year = (SELECT max(year) FROM mart.acs_county_demographic_annual)
),
econ AS (
    SELECT *
    FROM mart.acs_county_economic_annual
    WHERE year = (SELECT max(year) FROM mart.acs_county_economic_annual)
)
SELECT
    n.fips,
    n.STATE AS state_name,
    n.STATEABBRV AS state_abbrev,
    n.COUNTY AS county_name,
    n.risk_score,
    n.risk_rating,
    afford.year AS affordability_year,
    demo.year AS demographic_year,
    econ.year AS economic_year,

    afford.median_household_income,
    afford.median_owner_costs_mortgage,
    afford.dp04_selected_monthly_owner_costs_housing_units_mortgage_median_est,
    afford.median_property_taxes,

    afford.b25132_monthly_electricity_costs_total_charged_for_electricity_less_than_dollars_50_est,
    afford.b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_50_to_dollars_99_est,
    afford.b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_100_to_dollars_149_est,
    afford.b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_150_to_dollars_199_est,
    afford.b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_200_to_dollars_249_est,
    afford.b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_250_or_more_est,

    afford.b25133_monthly_gas_costs_total_charged_for_gas_less_than_dollars_25_est,
    afford.b25133_monthly_gas_costs_total_charged_for_gas_dollars_25_to_dollars_49_est,
    afford.b25133_monthly_gas_costs_total_charged_for_gas_dollars_50_to_dollars_74_est,
    afford.b25133_monthly_gas_costs_total_charged_for_gas_dollars_75_to_dollars_99_est,
    afford.b25133_monthly_gas_costs_total_charged_for_gas_dollars_100_to_dollars_149_est,
    afford.b25133_monthly_gas_costs_total_charged_for_gas_dollars_150_or_more_est,

    afford.b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_less_than_dollars_125_est,
    afford.b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_125_to_dollars_249_est,
    afford.b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_250_to_dollars_499_est,
    afford.b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_500_to_dollars_749_est,
    afford.b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_750_to_dollars_999_est,
    afford.b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_1_000_or_more_est,

    afford.b25135_annual_other_fuel_costs_total_charged_for_other_fuels_less_than_dollars_250_est,
    afford.b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_250_to_dollars_749_est,
    afford.b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_750_or_more_est,

    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_less_than_dollars_100_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_100_to_dollars_299_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_300_to_dollars_499_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_500_to_dollars_799_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_800_to_dollars_999_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_1000_to_dollars_1499_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_1500_to_dollars_1999_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_2000_to_dollars_2499_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_2500_to_dollars_2999_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_3000_to_dollars_3499_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_3500_to_dollars_3999_est,
    afford.b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_4000_or_more_est,

    econ.dp03_population_16_plus_in_labor_force_civilian_labor_force_unemployed_pct,

    demo.dp05_total_population_65_plus_pct,
    demo.dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct,
    demo.dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct,
    demo.dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct
FROM mart.nri_county_risk AS n
LEFT JOIN afford ON n.fips = afford.fips
LEFT JOIN demo ON n.fips = demo.fips
LEFT JOIN econ ON n.fips = econ.fips
""")

id_columns = [
    "fips",
    "state_name",
    "state_abbrev",
    "county_name",
    "risk_rating",
]
for column in features.columns.difference(id_columns):
    features[column] = to_number(features[column])

features["risk_rating"] = pd.Categorical(features["risk_rating"], categories=RISK_ORDER, ordered=True)
features = features[features["risk_rating"].isin(RISK_ORDER)].copy()

numeric_columns = features.select_dtypes(include="number").columns.tolist()
id_frame = (
    features[id_columns]
    .sort_values(["risk_rating", "fips"])
    .drop_duplicates("fips")
    .set_index("fips")
)
numeric_frame = features.groupby("fips", observed=True)[numeric_columns].max()
features = id_frame.join(numeric_frame, how="left").reset_index()
features["risk_rating"] = pd.Categorical(features["risk_rating"], categories=RISK_ORDER, ordered=True)
display(features.head())
display(features["risk_rating"].value_counts(dropna=False).sort_index())
'''
        ),
        md("## Derive Plot Features"),
        code(
            r"""
electricity_bins_monthly = [
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_less_than_dollars_50_est", 25),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_50_to_dollars_99_est", 74.5),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_100_to_dollars_149_est", 124.5),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_150_to_dollars_199_est", 174.5),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_200_to_dollars_249_est", 224.5),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_250_or_more_est", 275),
]

gas_bins_monthly = [
    ("b25133_monthly_gas_costs_total_charged_for_gas_less_than_dollars_25_est", 12.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_25_to_dollars_49_est", 37),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_50_to_dollars_74_est", 62),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_75_to_dollars_99_est", 87),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_100_to_dollars_149_est", 124.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_150_or_more_est", 175),
]

water_bins_annual = [
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_less_than_dollars_125_est", 62.5),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_125_to_dollars_249_est", 187),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_250_to_dollars_499_est", 374.5),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_500_to_dollars_749_est", 624.5),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_750_to_dollars_999_est", 874.5),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_1_000_or_more_est", 1125),
]

other_fuel_bins_annual = [
    ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_less_than_dollars_250_est", 125),
    ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_250_to_dollars_749_est", 499.5),
    ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_750_or_more_est", 875),
]

insurance_bins_annual = [
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_less_than_dollars_100_est", 50),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_100_to_dollars_299_est", 199.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_300_to_dollars_499_est", 399.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_500_to_dollars_799_est", 649.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_800_to_dollars_999_est", 899.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_1000_to_dollars_1499_est", 1249.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_1500_to_dollars_1999_est", 1749.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_2000_to_dollars_2499_est", 2249.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_2500_to_dollars_2999_est", 2749.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_3000_to_dollars_3499_est", 3249.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_3500_to_dollars_3999_est", 3749.5),
    ("b25141_homeowners_insurance_costs_by_mortgage_status_total_mortgage_dollars_4000_or_more_est", 4250),
]

features["income"] = features["median_household_income"].where(features["median_household_income"].gt(0))
features["selected_owner_cost_annual"] = (
    features["dp04_selected_monthly_owner_costs_housing_units_mortgage_median_est"].where(
        features["dp04_selected_monthly_owner_costs_housing_units_mortgage_median_est"].gt(0)
    )
    * 12
)
features["property_taxes_annual"] = features["median_property_taxes"].where(
    features["median_property_taxes"].ge(0)
)
features["home_insurance_annual"] = features.apply(weighted_midpoint, axis=1, bins=insurance_bins_annual)
features["electricity_annual"] = features.apply(weighted_midpoint, axis=1, bins=electricity_bins_monthly) * 12
features["gas_annual"] = features.apply(weighted_midpoint, axis=1, bins=gas_bins_monthly) * 12
features["water_sewer_annual"] = features.apply(weighted_midpoint, axis=1, bins=water_bins_annual)
features["other_fuel_annual"] = features.apply(weighted_midpoint, axis=1, bins=other_fuel_bins_annual)
features["utilities_annual"] = features[["electricity_annual", "gas_annual", "water_sewer_annual", "other_fuel_annual"]].sum(axis=1, min_count=1)

# ACS selected monthly owner costs include mortgage payments, taxes, insurance, utilities,
# fuels, and related regular owner costs. Use the residual as a mortgage-cost proxy so
# the requested components sum back to the selected owner-cost total without double-counting.
known_non_mortgage_cost = features[["utilities_annual", "home_insurance_annual", "property_taxes_annual"]].sum(axis=1, min_count=1)
features["mortgage_annual"] = (features["selected_owner_cost_annual"] - known_non_mortgage_cost).clip(lower=0)
features["home_ownership_cost"] = features[["utilities_annual", "home_insurance_annual", "property_taxes_annual", "mortgage_annual"]].sum(axis=1, min_count=1)
features["home_ownership_cost_pct_income"] = safe_ratio(features["home_ownership_cost"], features["income"], scale=100)

features["unemployment_rate"] = features["dp03_population_16_plus_in_labor_force_civilian_labor_force_unemployed_pct"]
features["share_age_65_plus"] = features["dp05_total_population_65_plus_pct"]
features["share_with_disability"] = features["dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct"]
features["share_communication_barrier"] = features["dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct"]
features["share_without_internet_access"] = 100 - features["dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct"]

non_negative_columns = [
    "income",
    "selected_owner_cost_annual",
    "property_taxes_annual",
    "home_insurance_annual",
    "electricity_annual",
    "gas_annual",
    "water_sewer_annual",
    "other_fuel_annual",
    "utilities_annual",
    "mortgage_annual",
    "home_ownership_cost",
    "home_ownership_cost_pct_income",
    "unemployment_rate",
    "share_age_65_plus",
    "share_with_disability",
    "share_communication_barrier",
    "share_without_internet_access",
]
for column in non_negative_columns:
    features[column] = features[column].where(features[column].ge(0))

percent_columns = [
    "home_ownership_cost_pct_income",
    "unemployment_rate",
    "share_age_65_plus",
    "share_with_disability",
    "share_communication_barrier",
    "share_without_internet_access",
]
for column in percent_columns:
    features[column] = features[column].where(features[column].le(100))

plot_features = [
    ("income", "Income", "Median household income", True, False),
    ("home_ownership_cost", "Home Ownership Cost", "Estimated annual cost", True, False),
    ("home_ownership_cost_pct_income", "Home Ownership Cost As Share Of Income", "Share of median household income", False, True),
    ("unemployment_rate", "Unemployment Rate", "Percent", False, True),
    ("share_age_65_plus", "Share Of Population Age 65+", "Percent", False, True),
    ("share_with_disability", "Share Of Population With Disability Status", "Percent", False, True),
    ("share_communication_barrier", "Share With Communication Barrier / Limited English Fluency", "Percent", False, True),
    ("share_without_internet_access", "Share Of Households Without Internet Access", "Percent", False, True),
]

summary_columns = [item[0] for item in plot_features]
summary = (
    features.groupby("risk_rating", observed=True)[summary_columns]
    .agg(["count", "median", "mean", "min", "max"])
    .round(2)
)
display(summary)
"""
        ),
        md("## Feature Distributions By NRI Risk Rating"),
        code(
            r"""
column, title, ylabel, dollars, percent = plot_features[0]
plot_distribution(
    features,
    column,
    f"{title} by NRI risk rating",
    ylabel,
    dollars=dollars,
    percent=percent,
)
"""
        ),
        md("## Income Grouping Compared With NRI Risk Rating"),
        code(
            r"""
income_comparison = features.dropna(subset=["income", "risk_rating"]).copy()
income_comparison = income_comparison[income_comparison["risk_rating"].isin(RISK_ORDER)]
income_comparison["risk_rating"] = pd.Categorical(
    income_comparison["risk_rating"],
    categories=RISK_ORDER,
    ordered=True,
)

income_labels = [
    "Lowest income quintile",
    "Second income quintile",
    "Middle income quintile",
    "Fourth income quintile",
    "Highest income quintile",
]
income_comparison["income_group"] = pd.qcut(
    income_comparison["income"],
    q=5,
    labels=income_labels,
    duplicates="drop",
)
income_comparison["income_group"] = pd.Categorical(
    income_comparison["income_group"],
    categories=income_labels,
    ordered=True,
)

income_group_summary = (
    income_comparison.groupby("income_group", observed=True)
    .agg(
        counties=("fips", "count"),
        min_income=("income", "min"),
        median_income=("income", "median"),
        max_income=("income", "max"),
    )
    .round(0)
)
display(income_group_summary)

risk_by_income_count = pd.crosstab(
    income_comparison["income_group"],
    income_comparison["risk_rating"],
).reindex(index=income_labels, columns=RISK_ORDER)
risk_by_income_share = risk_by_income_count.div(risk_by_income_count.sum(axis=1), axis=0) * 100

income_by_risk_count = pd.crosstab(
    income_comparison["risk_rating"],
    income_comparison["income_group"],
).reindex(index=RISK_ORDER, columns=income_labels)
income_by_risk_share = income_by_risk_count.div(income_by_risk_count.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

risk_by_income_share.plot(
    kind="bar",
    stacked=True,
    color=[PALETTE[rating] for rating in RISK_ORDER],
    ax=axes[0],
)
axes[0].set_title("NRI risk rating mix within income quintiles")
axes[0].set_xlabel("Income group")
axes[0].set_ylabel("Share of counties")
axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
axes[0].tick_params(axis="x", rotation=25)
axes[0].legend(title=None, loc="upper left", bbox_to_anchor=(1.02, 1))

income_by_risk_share.plot(
    kind="bar",
    stacked=True,
    colormap="viridis",
    ax=axes[1],
)
axes[1].set_title("Income quintile mix within NRI risk ratings")
axes[1].set_xlabel("NRI risk rating")
axes[1].set_ylabel("Share of counties")
axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
axes[1].tick_params(axis="x", rotation=25)
axes[1].legend(title=None, loc="upper left", bbox_to_anchor=(1.02, 1))

for ax in axes:
    ax.grid(axis="x", visible=False)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(
    risk_by_income_count,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    cbar_kws={"label": "County count"},
)
plt.title("County count by income quintile and NRI risk rating")
plt.xlabel("NRI risk rating")
plt.ylabel("Income group")
plt.tight_layout()
plt.show()
"""
        ),
        md("## Remaining Feature Distributions By NRI Risk Rating"),
        code(
            r"""
for column, title, ylabel, dollars, percent in plot_features[1:]:
    plot_distribution(
        features,
        column,
        f"{title} by NRI risk rating",
        ylabel,
        dollars=dollars,
        percent=percent,
    )
"""
        ),
        md("## Home Ownership Cost Component Breakdown"),
        code(
            r"""
component_columns = {
    "Utilities": "utilities_annual",
    "Home insurance": "home_insurance_annual",
    "Property taxes": "property_taxes_annual",
}

component_summary = (
    features.groupby("risk_rating", observed=True)[list(component_columns.values()) + ["income", "home_ownership_cost"]]
    .median()
    .reset_index()
)

component_long = component_summary.melt(
    id_vars=["risk_rating", "income", "home_ownership_cost"],
    value_vars=list(component_columns.values()),
    var_name="component_column",
    value_name="median_annual_cost",
)
component_long["component"] = component_long["component_column"].map({v: k for k, v in component_columns.items()})
component_long["share_of_income"] = component_long["median_annual_cost"] / component_long["income"] * 100
component_long["risk_rating"] = pd.Categorical(component_long["risk_rating"], categories=RISK_ORDER, ordered=True)

display(component_long.sort_values(["risk_rating", "component"]))

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

sns.barplot(
    data=component_long,
    x="risk_rating",
    y="median_annual_cost",
    hue="component",
    order=RISK_ORDER,
    ax=axes[0],
)
axes[0].set_title("Median annual home ownership cost components")
axes[0].set_xlabel("NRI risk rating")
axes[0].set_ylabel("Median annual cost")
axes[0].yaxis.set_major_formatter(lambda value, _: f"${value:,.0f}")
axes[0].tick_params(axis="x", rotation=22)
axes[0].legend(title=None, loc="upper left", bbox_to_anchor=(1.02, 1))

sns.barplot(
    data=component_long,
    x="risk_rating",
    y="share_of_income",
    hue="component",
    order=RISK_ORDER,
    ax=axes[1],
)
axes[1].set_title("Median component cost as share of income")
axes[1].set_xlabel("NRI risk rating")
axes[1].set_ylabel("Share of median household income")
axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
axes[1].tick_params(axis="x", rotation=22)
axes[1].legend(title=None, loc="upper left", bbox_to_anchor=(1.02, 1))

for ax in axes:
    ax.grid(axis="x", visible=False)

plt.tight_layout()
plt.show()
"""
        ),
        md("## Notes"),
        md(
            """
- Latest available annual values are selected independently from the three ACS mart tables.
- Income uses `median_household_income`.
- Home ownership cost uses the ACS selected monthly owner-cost median for mortgaged units, annualized. The component view focuses on utilities, insurance, and property taxes. Utilities and insurance are estimated from ACS binned distributions using bin midpoints; property taxes use the county median from ACS B25103 (`B25103_001E`).
- Households without internet access are computed as `100 - share with broadband internet subscription`.
"""
        ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(NOTEBOOK, indent=1) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
