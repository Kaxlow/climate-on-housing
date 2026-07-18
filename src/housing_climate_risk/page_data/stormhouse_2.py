from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from housing_climate_risk.page_data.annual_event_metrics import build_additional_annual_metrics
from housing_climate_risk.page_data.event_windows import (
    build_affected_event_windows,
    event_window_months,
    filter_complete_event_window_lines,
    load_disaster_events,
    load_redfin_county_monthly,
)


ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "quoll.duckdb"
COUNTIES_PATH = ROOT / "data" / "fipsgeo" / "us_counties_boundaries_shapefile.json"
OUT_PATH = ROOT / "output" / "visualizations" / "stormhouse-2.html"

RISK_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]
RISK_MAP = {
    "Very Low": "Very Low",
    "Relatively Low": "Low",
    "Relatively Moderate": "Medium",
    "Moderate": "Medium",
    "Relatively High": "High",
    "Very High": "Very High",
}
RISK_NUMERIC = {rating: index + 1 for index, rating in enumerate(RISK_ORDER)}
HAZARDS = [
    {"key": "overall", "label": "Overall NRI", "score": "risk_score", "rating": "risk_rating"},
    {"key": "river_flood", "label": "River Flood", "score": "IFLD_RISKS", "rating": "IFLD_RISKR"},
    {"key": "tornado", "label": "Tornado", "score": "TRND_RISKS", "rating": "TRND_RISKR"},
    {"key": "wildfire", "label": "Wildfire", "score": "WFIR_RISKS", "rating": "WFIR_RISKR"},
    {"key": "hail", "label": "Hail", "score": "HAIL_RISKS", "rating": "HAIL_RISKR"},
    {"key": "earthquake", "label": "Earthquake", "score": "ERQK_RISKS", "rating": "ERQK_RISKR"},
]

# Mapping from model feature key to display config for the playbook section.
# featureLabel links model inputs back to the binned feature-association payload.
PLAYBOOK_FEATURE_MAP = [
    {"modelKey": "homes_sold_yoy", "label": "Homes Sold YoY", "featureLabel": "Homes Sold YOY", "format": "pct"},
    {"modelKey": "new_listings_yoy", "label": "New Listings YoY", "featureLabel": "New Listings YOY", "format": "pct"},
    {"modelKey": "unemployment_rate", "label": "Unemployment Rate", "featureLabel": "Unemployment", "format": "percent"},
    {"modelKey": "insurance_pct_income", "label": "Insurance as % Income", "featureLabel": "Insurance % of Income", "format": "percent"},
    {"modelKey": "median_dom_yoy", "label": "Median DOM YoY", "featureLabel": "Median Days on Market YOY", "format": "number"},
    {"modelKey": "property_taxes_pct_income", "label": "Property Tax as % Income", "featureLabel": "Property Tax % of Income", "format": "percent"},
    {"modelKey": "utilities_pct_income", "label": "Utilities as % Income", "featureLabel": "Utilities % of Income", "format": "percent"},
    {"modelKey": "median_homeowner_income", "label": "Median Household Income", "featureLabel": "Income", "format": "currency"},
    {"modelKey": "net_migration_rate", "label": "Net Migration Rate", "featureLabel": "Net Migration Rate", "format": "number"},
    {"modelKey": "net_earnings_per_capita", "label": "Net Earnings Per Capita", "featureLabel": "Net Earnings per Capita", "format": "currency"},
    {"modelKey": "dividends_interest_rent_per_capita", "label": "Dividends / Interest / Rent Per Capita", "featureLabel": "Dividends/Interest/Rent per Capita", "format": "currency"},
    {"modelKey": "transfer_receipts_per_capita", "label": "Transfer Receipts Per Capita", "featureLabel": "Transfer Receipts per Capita", "format": "currency"},
]


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def serialize_number(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def rating_clean(value: object) -> str | None:
    if pd.isna(value):
        return None
    return RISK_MAP.get(str(value), str(value))


def feature_bucket_labels(values: pd.Series, fmt: str, count: int = 5) -> tuple[pd.Series, list[str]]:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=values.index, dtype="object"), []
    edges = valid.quantile(np.linspace(0, 1, count + 1)).to_numpy()
    edges = np.unique(edges)
    if len(edges) < 3:
        label = f"All values ({format_bucket_value(valid.median(), fmt)})"
        return pd.Series(label, index=values.index, dtype="object").where(values.notna()), [label]
    labels = [f"B{i + 1}: {format_bucket_value(edges[i], fmt)} to {format_bucket_value(edges[i + 1], fmt)}" for i in range(len(edges) - 1)]
    buckets = pd.cut(values, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    return buckets.astype("object"), labels


def format_bucket_value(value: float, fmt: str) -> str:
    if pd.isna(value):
        return "n/a"
    if fmt == "currency":
        return f"${value:,.0f}"
    if fmt in {"percent", "pct"}:
        return f"{value:,.1f}%"
    return f"{value:,.1f}"


def classify_difference(value: object, baseline: object, neutral_band: float) -> str:
    if pd.isna(value) or pd.isna(baseline):
        return "neutral"
    diff = float(value) - float(baseline)
    if diff > neutral_band:
        return "higher"
    if diff < -neutral_band:
        return "lower"
    return "neutral"


def classify_bucket_position(bucket_order: int | None, bucket_count: int, corr: object) -> str:
    if bucket_order is None or not bucket_count or pd.isna(corr) or float(corr) == 0:
        return "neutral"
    midpoint = (bucket_count - 1) / 2
    if bucket_order == midpoint:
        return "neutral"
    higher_bin = bucket_order > midpoint
    positive_corr = float(corr) > 0
    if higher_bin == positive_corr:
        return "higher"
    return "lower"


def weighted_bucket_average(frame: pd.DataFrame, buckets: list[tuple[str, float]], *, zero_cols: list[str] | None = None) -> pd.Series:
    total = pd.Series(0.0, index=frame.index)
    weighted = pd.Series(0.0, index=frame.index)
    for column in zero_cols or []:
        if column in frame:
            total = total.add(pd.to_numeric(frame[column], errors="coerce").fillna(0), fill_value=0)
    for column, midpoint in buckets:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
            total = total.add(values, fill_value=0)
            weighted = weighted.add(values * midpoint, fill_value=0)
    return weighted.where(total > 0) / total.where(total > 0)


def mean_available(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index)
    return frame[available].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def build_price_risk(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    hazard_cols: list[str] = []
    for hazard in HAZARDS:
        if hazard["key"] == "overall":
            hazard_cols.extend(["risk_score", "risk_rating"])
        else:
            hazard_cols.extend([hazard["score"], hazard["rating"]])
    nri = con.execute(
        f"""
        SELECT fips, COUNTY, STATEABBRV, {", ".join(hazard_cols)}
        FROM mart.nri_county_risk
        WHERE fips IS NOT NULL
        """
    ).df()
    ppsf = con.execute(
        """
        SELECT
            fips,
            any_value(REGION) AS county_label,
            any_value(STATE_CODE) AS state_code,
            avg(CASE
                WHEN try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL
                ELSE try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE)
            END) AS avg_median_ppsf_yoy,
            count(*) FILTER (
                WHERE try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) IS NOT NULL
                  AND try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) > -888888000
            ) AS observed_months
        FROM mart.redfin_county_monthly
        WHERE property_type = 'All Residential'
          AND period_begin >= DATE '2025-01-01'
          AND period_begin < DATE '2026-01-01'
          AND fips IS NOT NULL
        GROUP BY fips
        """
    ).df()
    df = ppsf.merge(nri, on="fips", how="inner")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["avg_median_ppsf_yoy"] = pd.to_numeric(df["avg_median_ppsf_yoy"], errors="coerce")
    for hazard in HAZARDS:
        df[hazard["score"]] = clean_numeric(df[hazard["score"]])
        df[hazard["rating"]] = df[hazard["rating"]].map(rating_clean)
    df = df.dropna(subset=["avg_median_ppsf_yoy"]).copy()
    df["risk_rating_clean"] = df["risk_rating"].map(rating_clean)
    cap_lower = df["avg_median_ppsf_yoy"].quantile(0.01)
    cap_upper = df["avg_median_ppsf_yoy"].quantile(0.99)
    score_corr = df.dropna(subset=["risk_score"])[["risk_score", "avg_median_ppsf_yoy"]].corr(method="spearman").iloc[0, 1]
    rating_numeric = df["risk_rating_clean"].map(RISK_NUMERIC)
    rating_corr = pd.DataFrame({"rating": rating_numeric, "ppsf": df["avg_median_ppsf_yoy"]}).dropna().corr(method="spearman").iloc[0, 1]
    counties = []
    for row in df.itertuples(index=False):
        hazards = {}
        for hazard in HAZARDS:
            rating = getattr(row, hazard["rating"])
            hazards[hazard["key"]] = {
                "score": serialize_number(getattr(row, hazard["score"]), 3),
                "rating": rating,
                "ratingValue": RISK_NUMERIC.get(rating),
            }
        counties.append(
            {
                "fips": row.fips,
                "county": row.county_label if pd.notna(row.county_label) else f"{row.COUNTY}, {row.STATEABBRV}",
                "state": row.state_code if pd.notna(row.state_code) else row.STATEABBRV,
                "avgPpsfYoy": serialize_number(row.avg_median_ppsf_yoy, 5),
                "avgPpsfYoyCapped": serialize_number(min(max(row.avg_median_ppsf_yoy, cap_lower), cap_upper), 5),
                "ppsfYoyWasCapped": bool(row.avg_median_ppsf_yoy < cap_lower or row.avg_median_ppsf_yoy > cap_upper),
                "observedMonths": int(row.observed_months),
                "hazards": hazards,
            }
        )
    history = con.execute(
        """
        SELECT
            r.fips,
            date_part('year', r.period_begin)::INTEGER AS year,
            any_value(r.REGION) AS county_label,
            any_value(r.STATE_CODE) AS state_code,
            avg(CASE
                WHEN try_cast(replace(r.MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL
                ELSE try_cast(replace(r.MEDIAN_PPSF_YOY, ',', '') AS DOUBLE)
            END) AS median_ppsf_yoy
        FROM mart.redfin_county_monthly AS r
        WHERE r.property_type = 'All Residential'
          AND r.period_begin >= DATE '2016-01-01'
          AND r.period_begin < DATE '2026-01-01'
          AND r.fips IS NOT NULL
        GROUP BY r.fips, date_part('year', r.period_begin)
        """
    ).df()
    history["fips"] = history["fips"].astype(str).str.zfill(5)
    history["median_ppsf_yoy"] = pd.to_numeric(history["median_ppsf_yoy"], errors="coerce")

    # Merge all hazard columns to enable hazard-specific filtering
    history = history.merge(nri[["fips"] + hazard_cols], on="fips", how="left")
    history["riskRating"] = history["risk_rating"].map(rating_clean)

    # Clean hazard columns
    for hazard in HAZARDS:
        if hazard["key"] != "overall":
            history[hazard["rating"]] = history[hazard["rating"]].map(rating_clean)

    history = history.dropna(subset=["median_ppsf_yoy", "riskRating"]).copy()

    # Create hazard-specific histories for each hazard type
    rating_histories = {}
    for hazard in HAZARDS:
        hazard_key = hazard["key"]
        if hazard_key == "overall":
            hazard_history = history.copy()
            group_col = "riskRating"
        else:
            # Group by the hazard-specific rating, not overall
            rating_col = hazard["rating"]
            hazard_history = history[history[rating_col].notna()].copy()
            hazard_history["_hazard_rating"] = hazard_history[rating_col]
            group_col = "_hazard_rating"

        grouped = (
            hazard_history.groupby([group_col, "year"], observed=False)["median_ppsf_yoy"]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={group_col: "riskRating", 0.25: "q1", 0.5: "median", 0.75: "q3"})
        )
        rating_histories[hazard_key] = [
            {
                "riskRating": row.riskRating,
                "year": int(row.year),
                "q1": serialize_number(row.q1, 5),
                "median": serialize_number(row.median, 5),
                "q3": serialize_number(row.q3, 5),
            }
            for row in grouped.itertuples(index=False)
            if row.riskRating in RISK_ORDER
        ]
    # Create county history records with hazard information
    county_history_records = []
    for row in history.itertuples(index=False):
        record = {
            "fips": row.fips,
            "county": row.county_label,
            "state": row.state_code,
            "year": int(row.year),
            "ppsfYoy": serialize_number(row.median_ppsf_yoy, 5),
            "riskRating": row.riskRating,
        }
        # Add hazard-specific ratings
        for hazard in HAZARDS:
            if hazard["key"] != "overall":
                hazard_rating = getattr(row, hazard["rating"], None)
                record[f"{hazard['key']}_rating"] = hazard_rating
        county_history_records.append(record)

    return {
        "hazards": [{"key": h["key"], "label": h["label"]} for h in HAZARDS],
        "counties": counties,
        "countyHistory": county_history_records,
        "ratingHistoriesByHazard": rating_histories,
        "summary": {
            "countyCount": int(df["fips"].nunique()),
            "medianAvgPpsfYoy": serialize_number(df["avg_median_ppsf_yoy"].median(), 5),
            "scoreSpearman": serialize_number(score_corr, 3),
            "ratingSpearman": serialize_number(rating_corr, 3),
            "ppsfCapLower": serialize_number(cap_lower, 5),
            "ppsfCapUpper": serialize_number(cap_upper, 5),
        },
    }


def build_geojson(fips_set: set[str]) -> dict[str, object]:
    from shapely.geometry import mapping, shape
    from shapely.geometry.multipolygon import MultiPolygon
    from shapely.geometry.polygon import Polygon, orient

    raw = json.loads(COUNTIES_PATH.read_text(encoding="utf-8"))
    features = []
    for feature in raw["features"]:
        props = feature.get("properties", {})
        fips = str(props.get("GEOID") or props.get("GEOID10") or "").zfill(5)
        if fips in fips_set:
            tolerance = 0.08 if fips.startswith("02") else 0.025
            geometry = shape(feature["geometry"]).simplify(tolerance, preserve_topology=True)
            if not geometry.is_empty:
                if isinstance(geometry, Polygon):
                    geometry = orient(geometry, sign=-1.0)
                elif isinstance(geometry, MultiPolygon):
                    geometry = MultiPolygon([orient(part, sign=-1.0) for part in geometry.geoms])
                features.append({"type": "Feature", "properties": {"fips": fips}, "geometry": mapping(geometry)})
    return {"type": "FeatureCollection", "features": features}


def aggregate_lines(frame: pd.DataFrame, group_cols: list[str], metric: str, annual: bool = False) -> list[dict[str, object]]:
    if frame.empty:
        return []

    if annual:
        # For annual data, convert event_window_month to event_window_year
        frame_copy = frame.copy()
        frame_copy["event_window_year"] = (frame_copy["event_window_month"] / 12).round().astype(int)
        q = (
            frame_copy.dropna(subset=[metric, "event_window_year"])
            .groupby(group_cols + ["event_window_year"], observed=False)[metric]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
        )
        return [
            {
                **{col: getattr(row, col) for col in group_cols},
                "month": int(row.event_window_year * 12),  # Convert back to months for consistency
                "q1": serialize_number(row.q1, 5),
                "median": serialize_number(row.median, 5),
                "q3": serialize_number(row.q3, 5),
            }
            for row in q.itertuples(index=False)
        ]
    else:
        q = (
            frame.dropna(subset=[metric, "event_window_month"])
            .groupby(group_cols + ["event_window_month"], observed=False)[metric]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
        )
        return [
            {
                **{col: getattr(row, col) for col in group_cols},
                "month": int(row.event_window_month),
                "q1": serialize_number(row.q1, 5),
                "median": serialize_number(row.median, 5),
                "q3": serialize_number(row.q3, 5),
            }
            for row in q.itertuples(index=False)
        ]


# OLD IMPLEMENTATION - REPLACED BY annual_event_metrics.py
# def build_additional_event_metrics(con: duckdb.DuckDBPyConnection, affected: pd.DataFrame, complete: pd.DataFrame, nri: pd.DataFrame) -> list[dict[str, object]]:
    """
    Build additional time-series metrics around event windows for the section:
    "What Else Are Climate Events Doing to Counties?"

    Metrics:
    - Net migration rate
    - Home insurance cost as share of income
    - Employment rate / unemployment rate
    - Market cooling (Median DOM YOY - Homes Sold YOY)
    - Market stress (housing burden * 0.5 + market cooling * 0.5)
    - Housing burden-migration interaction
    - Home insurance-market interaction
    """
    # Since we don't have all metrics in the redfin monthly table, we'll use what's available
    # and compute derived metrics. For demonstration, we'll focus on market-based metrics
    # that can be computed from housing data

    metrics = []
    demo = latest_by_fips(
        con,
        "mart.acs_county_demographic_annual",
        [
            "fips",
            "year",
            "domestic_in_migration_rate",
            "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct",
        ],
    )
    econ = latest_by_fips(
        con,
        "mart.acs_county_economic_annual",
        ["fips", "year", "dp03_income_and_benefits_total_households_median_household_income_est", "dp03_civilian_labor_force_unemployment_rate_pct"],
    )
    afford = latest_by_fips(
        con,
        "mart.acs_county_affordability_annual",
        [
            "fips",
            "year",
            "owner_mortgage_cost_burden_30pct_plus",
            *[
                f"b25141_homeowners_insurance_costs_by_mortgage_status_total_{status}_{suffix}_est"
                for status in ["mortgage", "not_mortgaged"]
                for suffix in [
                    "less_than_dollars_100",
                    "dollars_100_to_dollars_299",
                    "dollars_300_to_dollars_499",
                    "dollars_500_to_dollars_799",
                    "dollars_800_to_dollars_999",
                    "dollars_1000_to_dollars_1499",
                    "dollars_1500_to_dollars_1999",
                    "dollars_2000_to_dollars_2499",
                    "dollars_2500_to_dollars_2999",
                    "dollars_3000_to_dollars_3499",
                    "dollars_3500_to_dollars_3999",
                    "dollars_4000_or_more",
                ]
            ],
        ],
    )
    if not afford.empty:
        afford["insurance_premium"] = weighted_bucket_average(
            afford,
            [
                (f"b25141_homeowners_insurance_costs_by_mortgage_status_total_{status}_{suffix}_est", midpoint)
                for status in ["mortgage", "not_mortgaged"]
                for suffix, midpoint in [
                    ("less_than_dollars_100", 50),
                    ("dollars_100_to_dollars_299", 200),
                    ("dollars_300_to_dollars_499", 400),
                    ("dollars_500_to_dollars_799", 650),
                    ("dollars_800_to_dollars_999", 900),
                    ("dollars_1000_to_dollars_1499", 1250),
                    ("dollars_1500_to_dollars_1999", 1750),
                    ("dollars_2000_to_dollars_2499", 2250),
                    ("dollars_2500_to_dollars_2999", 2750),
                    ("dollars_3000_to_dollars_3499", 3250),
                    ("dollars_3500_to_dollars_3999", 3750),
                    ("dollars_4000_or_more", 4250),
                ]
            ],
        )
        # Merge with econ to get median_household_income for insurance_income_share calculation
        afford = afford.merge(econ[["fips", "dp03_income_and_benefits_total_households_median_household_income_est"]], on="fips", how="left")
        afford["insurance_income_share"] = afford["insurance_premium"] / afford["dp03_income_and_benefits_total_households_median_household_income_est"].where(afford["dp03_income_and_benefits_total_households_median_household_income_est"] > 0) * 100
    static = (
        demo[["fips", "domestic_in_migration_rate", "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct"]]
        .merge(econ[["fips", "dp03_civilian_labor_force_unemployment_rate_pct"]], on="fips", how="outer")
        .merge(afford[["fips", "owner_mortgage_cost_burden_30pct_plus", "insurance_premium", "insurance_income_share"]], on="fips", how="outer")
    )
    static = static.rename(
        columns={
            "domestic_in_migration_rate": "net_migration_rate",
            "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct": "communication_barrier",
            "dp03_civilian_labor_force_unemployment_rate_pct": "unemployment_rate",
            "owner_mortgage_cost_burden_30pct_plus": "housing_burden_30pct",
        }
    )
    static["employment_rate"] = 100 - static["unemployment_rate"]
    complete = complete.merge(static, on="fips", how="left")

    # Market cooling: median_dom_yoy - homes_sold_yoy
    if "median_dom_yoy" in complete.columns and "homes_sold_yoy" in complete.columns:
        complete_with_cooling = complete.copy()
        complete_with_cooling["market_cooling"] = pd.to_numeric(complete_with_cooling["median_dom_yoy"], errors="coerce") - pd.to_numeric(complete_with_cooling["homes_sold_yoy"], errors="coerce")
        cooling_agg = aggregate_lines(complete_with_cooling.assign(series="All affected counties"), ["series"], "market_cooling")
        cooling_by_risk = aggregate_lines(complete_with_cooling.dropna(subset=["riskRating"]), ["riskRating"], "market_cooling")
        metrics.append({
            "key": "market_cooling",
            "label": "Market Cooling (Median DOM YOY - Homes Sold YOY)",
            "description": "Is there market cooling after an event?",
            "conclusion": "Market cooling rises when days on market increase or homes sold falls; that can pressure house price growth lower.",
            "aggregate": cooling_agg,
            "byRating": cooling_by_risk,
        })
        complete = complete_with_cooling

    derived_specs = [
        ("communication_barrier", "Share with communication barrier", "If this remains high and unchanged for higher-risk groups, it indicates vulnerable populations that may not relocate easily.", "annual"),
        ("net_migration_rate", "Net migration", "Migration changes reveal whether people move to or from affected higher-risk counties after events.", "annual"),
        ("insurance_income_share", "Home insurance cost as share of income", "A rising insurance burden can contribute to unaffordability and may indicate insurers pricing in climate risk.", "annual"),
        ("employment_rate", "Employment rate", "Employment declines after events can weaken household demand and reduce house price growth.", "annual"),
    ]
    if "market_cooling" in complete.columns:
        complete["market_stress"] = complete["housing_burden_30pct"] * 0.5 + complete["market_cooling"] * 0.5
        complete["housing_burden_migration_interaction"] = complete["housing_burden_30pct"] * (1 - complete["net_migration_rate"] / 100)
        complete["insurance_market_interaction"] = complete["insurance_premium"] * (1 + complete["median_dom_yoy"] / 100)
        derived_specs.extend(
            [
                ("market_stress", "Market Stress", "A composite of affordability pressure and market slowdown shows how vulnerable the market becomes after events.", "monthly/annual mix"),
                ("housing_burden_migration_interaction", "Housing burden-migration interaction", "Higher values indicate unaffordable markets where populations may also be less able to relocate.", "annual"),
                ("insurance_market_interaction", "Home insurance-market interaction", "This combines insurance premiums with buyer hesitation measured through days on market.", "monthly/annual mix"),
            ]
        )
    for key, label, description, frequency in derived_specs:
        if key not in complete.columns:
            continue
        metric_frame = complete.dropna(subset=[key, "riskRating"]).copy()
        if metric_frame.empty:
            continue

        # Standardize metrics with very high magnitudes (> 100)
        metric_std = metric_frame[key].std()
        metric_mean = metric_frame[key].mean()
        if metric_mean > 100 or metric_std > 100:
            # Standardize to mean 0, std 1
            metric_frame[f"{key}_standardized"] = (metric_frame[key] - metric_mean) / metric_std if metric_std > 0 else 0
            standardized_key = f"{key}_standardized"
            label_suffix = " (standardized)"
        else:
            standardized_key = key
            label_suffix = ""

        # Use annual aggregation for annual-frequency data
        is_annual = frequency == "annual"

        metrics.append(
            {
                "key": key,
                "label": label + label_suffix,
                "description": description,
                "frequency": frequency,
                "conclusion": f"{label} changes around climate events in a way that can alter local demand, affordability, or buyer confidence, and ultimately push house price growth up or down.",
                "aggregate": aggregate_lines(metric_frame.assign(series="All affected counties"), ["series"], standardized_key, annual=is_annual),
                "byRating": aggregate_lines(metric_frame, ["riskRating"], standardized_key, annual=is_annual),
            }
        )

    # Homes sold YOY (as indicator of buyer activity)
    if "homes_sold_yoy" in complete.columns:
        homes_agg = aggregate_lines(complete.assign(series="All affected counties"), ["series"], "homes_sold_yoy")
        homes_by_risk = aggregate_lines(complete.dropna(subset=["riskRating"]), ["riskRating"], "homes_sold_yoy")
        metrics.append({
            "key": "homes_sold_yoy",
            "label": "Homes Sold YOY",
            "description": "Change in buyer activity after climate events",
            "aggregate": homes_agg,
            "byRating": homes_by_risk,
        })

    # Median Days on Market YOY (indicator of market liquidity)
    if "median_dom_yoy" in complete.columns:
        dom_agg = aggregate_lines(complete.assign(series="All affected counties"), ["series"], "median_dom_yoy")
        dom_by_risk = aggregate_lines(complete.dropna(subset=["riskRating"]), ["riskRating"], "median_dom_yoy")
        metrics.append({
            "key": "median_dom_yoy",
            "label": "Median Days on Market YOY",
            "description": "Are homes taking longer to sell after events?",
            "aggregate": dom_agg,
            "byRating": dom_by_risk,
        })

    return metrics


def _build_window_data(
    affected: pd.DataFrame,
    nri: pd.DataFrame,
    metric: str,
    *,
    pre_months: int,
    post_months: int,
    anchor_col: str = "event_window_month",
    sample_per_group: int = 2,
    eligible_feature_fips_by_risk: dict[str, set[str]] | None = None,
) -> dict[str, object]:
    """Build by-rating aggregates + example lines for one event-window definition.

    Both time-window frames use the same raw ``affected`` dataframe but filter
    to counties that have complete monthly data across the frame's required months.
    """
    required = event_window_months(pre_months, post_months)
    complete = filter_complete_event_window_lines(
        affected,
        x_col=anchor_col,
        line_col="line_id",
        metric_col=metric,
        required_x_values=required,
    ).copy()
    complete = complete.loc[complete[anchor_col].isin(required)].copy()
    complete = complete.merge(nri[["fips", "riskRating"]], on="fips", how="left")
    complete_for_agg = complete.copy()
    if anchor_col != "event_window_month":
        complete_for_agg["event_window_month"] = complete_for_agg[anchor_col]
    by_rating = aggregate_lines(complete_for_agg.dropna(subset=["riskRating"]), ["riskRating"], metric)
    affected_counties = (
        complete.dropna(subset=["riskRating"])[["fips", "riskRating"]]
        .drop_duplicates()
        .groupby(["fips", "riskRating"], as_index=False)
        .size()
    )
    risk_counts = complete.drop_duplicates(["line_id", "riskRating"]).groupby("riskRating", dropna=True)["line_id"].nunique()

    # Compute per-county-event average metric over the window, then percentile within risk group.
    line_avg = (
        complete.dropna(subset=[metric, "riskRating"])
        .groupby(["line_id", "fips", "county_label", "state_code", "riskRating"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "avg_metric"})
    )
    line_avg["pct_rank"] = (
        line_avg.groupby("riskRating")["avg_metric"]
        .rank(method="average", pct=True, na_option="keep")
        .mul(100)
        .round(1)
    )

    bands = pd.DataFrame(by_rating)
    eligible_line_ids: set[str] = set()
    if not bands.empty:
        bands["iqr_width"] = bands["q3"] - bands["q1"]
        max_width_by_risk = bands.groupby("riskRating")["iqr_width"].max().to_dict()
        bands["lower_allowed"] = bands.apply(
            lambda row: row["q1"] - 0.5 * max_width_by_risk.get(row["riskRating"], np.nan),
            axis=1,
        )
        bands["upper_allowed"] = bands.apply(
            lambda row: row["q3"] + 0.5 * max_width_by_risk.get(row["riskRating"], np.nan),
            axis=1,
        )
        band_join = complete.dropna(subset=[metric, "riskRating"]).merge(
            bands[["riskRating", "month", "lower_allowed", "upper_allowed"]],
            left_on=["riskRating", anchor_col],
            right_on=["riskRating", "month"],
            how="inner",
        )
        band_join["inside_sample_band"] = band_join[metric].between(
            band_join["lower_allowed"],
            band_join["upper_allowed"],
            inclusive="both",
        )
        line_band_fit = (
            band_join.groupby("line_id", as_index=False)
            .agg(months=(anchor_col, "nunique"), all_inside=("inside_sample_band", "all"))
        )
        eligible_line_ids = set(
            line_band_fit.loc[
                line_band_fit["months"].eq(len(required)) & line_band_fit["all_inside"],
                "line_id",
            ].astype(str)
        )

    # Compute per-line extrema to keep payload metadata and retain the older +/-100 guard.
    line_extremes = (
        complete.dropna(subset=[metric, "riskRating"])
        .groupby("line_id", as_index=False)[metric]
        .agg(
            min_metric="min",
            max_metric="max",
            max_abs=lambda s: s.abs().max(),
        )
    )
    line_avg = line_avg.merge(line_extremes, on="line_id", how="left")

    example_lines = []
    for risk in RISK_ORDER:
        group = line_avg.loc[line_avg["riskRating"].eq(risk)].dropna(subset=["pct_rank"]).copy()
        if eligible_feature_fips_by_risk is not None:
            eligible_fips = eligible_feature_fips_by_risk.get(risk, set())
            group = group.loc[group["fips"].astype(str).isin(eligible_fips)].copy()
        if group.empty:
            continue
        if eligible_line_ids:
            group = group.loc[group["line_id"].astype(str).isin(eligible_line_ids)].copy()
        if group.empty:
            continue
        # Sample counties shown in the "What Sets Apart..." plot should remain
        # visually interpretable: require all median PPSF YoY values to stay within +/-100%.
        group = group.loc[group["max_metric"].le(100) & group["min_metric"].ge(-100)].copy()
        if group.empty:
            continue
        # Within the bounded set, still avoid the most extreme eligible lines.
        p95 = group["max_abs"].quantile(0.95)
        mild = group.loc[group["max_abs"].le(p95)]
        if len(mild) >= 2:
            group = mild
        group_median = group["avg_metric"].median()
        below = group.loc[group["avg_metric"].lt(group_median)].sort_values("avg_metric").head(1)
        above = group.loc[group["avg_metric"].gt(group_median)].sort_values("avg_metric", ascending=False).head(1)
        candidates = pd.concat([below, above], ignore_index=True)
        if candidates.empty:
            candidates = group.sort_values("pct_rank").head(sample_per_group)

        for candidate in candidates.itertuples(index=False):
            rows = complete.loc[complete["line_id"].eq(candidate.line_id)].sort_values(anchor_col)
            sample_position = "Above group median" if candidate.avg_metric > group_median else "Below group median"
            example_lines.append(
                {
                    "riskRating": risk,
                    "lineId": candidate.line_id,
                    "fips": candidate.fips,
                    "county": candidate.county_label,
                    "state": candidate.state_code,
                    "pctRank": float(candidate.pct_rank),
                    "avgPpsfYoy": serialize_number(candidate.avg_metric, 5),
                    "groupMedianPpsfYoy": serialize_number(group_median, 5),
                    "samplePosition": sample_position,
                    "minPpsfYoy": serialize_number(candidate.min_metric, 3),
                    "maxPpsfYoy": serialize_number(candidate.max_metric, 3),
                    "values": [
                        {"month": int(getattr(row, anchor_col)), "value": serialize_number(getattr(row, metric), 5)}
                        for row in rows.itertuples(index=False)
                        if pd.notna(getattr(row, metric))
                    ],
                }
            )

    # Per-county percentile rank of average PPSF YoY within its risk group over this window.
    # Keyed by fips → percentile (0–100). Counties appearing in multiple events get their
    # best (highest avg_metric) line's rank.
    county_pct = (
        line_avg.sort_values("avg_metric", ascending=False)
        .drop_duplicates(subset=["fips"])
        [["fips", "pct_rank"]]
        .set_index("fips")["pct_rank"]
        .to_dict()
    )

    return {
        "byRating": by_rating,
        "affectedCounties": [
            {"fips": row.fips, "riskRating": row.riskRating}
            for row in affected_counties.itertuples(index=False)
        ],
        "riskCounts": {str(k): int(v) for k, v in risk_counts.items()},
        "exampleCountyLines": example_lines,
        "countyEventWindowPctRank": county_pct,
    }


def build_event_windows(
    con: duckdb.DuckDBPyConnection,
    eligible_feature_fips_by_risk: dict[str, set[str]] | None = None,
) -> dict[str, object]:
    events = load_disaster_events(con)
    events = events.loc[events["event_start_month"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-01"))].copy()
    housing = load_redfin_county_monthly(con)
    for column in ["median_ppsf_yoy", "avg_sale_to_list_yoy", "homes_sold_yoy", "inventory_yoy", "housing_market_index"]:
        if column in housing:
            housing.loc[pd.to_numeric(housing[column], errors="coerce").le(-888888000), column] = np.nan
    metric = "median_ppsf_yoy"

    # Window A: 1 year before event start → 3 years after event end (pre=12, post=36)
    # Window B: 1 year before event end → 5 years after event end (pre=12, post=60, anchored at event end)
    # We use pre_event_months=24 for the raw build to cover both windows.
    affected = build_affected_event_windows(events, housing, pre_event_months=24, post_event_months=60)
    if affected.empty:
        empty = {"byRating": [], "affectedCounties": [], "riskCounts": {}, "exampleCountyLines": []}
        return {
            "windowA": empty,
            "windowB": empty,
            "summary": {"events": 0},
            "additionalMetrics": [],
        }

    nri = con.execute("SELECT fips, risk_rating FROM mart.nri_county_risk WHERE fips IS NOT NULL").df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)

    # Window A uses event_window_month (relative to event start), pre=12, post=36
    window_a = _build_window_data(
        affected,
        nri,
        metric,
        pre_months=12,
        post_months=36,
        eligible_feature_fips_by_risk=eligible_feature_fips_by_risk,
    )

    # Window B is anchored at event END: use months_after_event_end for post, months_from_event_start for pre.
    # We derive a combined "end-anchored" month column: negative = months before event end, positive = after.
    affected_b = affected.copy()
    affected_b["ewm_end"] = np.where(
        affected_b["months_from_event_start"].le(0),
        # pre-event period: distance from event start, negative
        affected_b["months_from_event_start"],
        # post-event period: months_after_event_end
        affected_b["months_after_event_end"],
    )
    affected_b["line_id"] = affected_b["event_key"]
    window_b = _build_window_data(
        affected_b,
        nri,
        metric,
        pre_months=12,
        post_months=60,
        anchor_col="ewm_end",
        eligible_feature_fips_by_risk=eligible_feature_fips_by_risk,
    )

    total_complete_lines = len(
        set(window_a["exampleCountyLines"][0]["lineId"] for x in window_a["exampleCountyLines"])
        if window_a["exampleCountyLines"] else []
    )

    # Build additional metrics for "What Else Are Climate Events Doing" section
    events_for_annual = events[['fips', 'event_key', 'event_source', 'source_event_id', 'event_type', 'event_name', 'event_start_month', 'event_end_month']].drop_duplicates(subset=['event_key'])
    annual_metrics = build_additional_annual_metrics(con, events_for_annual, nri[['fips', 'riskRating']], pre_years=2, post_years=3)

    # Add monthly metrics from window A complete set
    required_a = event_window_months(12, 36)
    complete_a = filter_complete_event_window_lines(
        affected, x_col="event_window_month", line_col="line_id", metric_col=metric, required_x_values=required_a,
    ).copy()
    complete_a = complete_a.merge(nri[["fips", "riskRating"]], on="fips", how="left")

    monthly_metrics = []
    for col, label, desc, conclusion in [
        ("homes_sold_yoy", "Homes Sold YOY", "Change in buyer activity after climate events", "Homes sold YOY changes reveal shifts in buyer demand following climate events."),
        ("median_dom_yoy", "Median Days on Market YOY", "Are homes taking longer to sell after events?", "Days on market YOY changes indicate shifts in market liquidity after climate events."),
    ]:
        if col in complete_a.columns:
            monthly_metrics.append({
                "key": col,
                "label": label,
                "description": desc,
                "frequency": "monthly",
                "isAnnual": False,
                "conclusion": conclusion,
                "aggregate": aggregate_lines(complete_a.assign(series="All affected counties"), ["series"], col),
                "byRating": aggregate_lines(complete_a.dropna(subset=["riskRating"]), ["riskRating"], col),
            })

    additional_metrics = annual_metrics + monthly_metrics

    return {
        "windowA": window_a,
        "windowB": window_b,
        "summary": {
            "events": int(events["event_key"].nunique()),
        },
        "additionalMetrics": additional_metrics,
    }


def latest_by_fips(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> pd.DataFrame:
    quoted = ", ".join(f'"{column}"' for column in columns)
    df = con.execute(
        f"""
        SELECT {quoted}
        FROM {table}
        WHERE fips IS NOT NULL AND year IS NOT NULL
        """
    ).df()
    if df.empty:
        return df
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    value_columns = [column for column in columns if column not in {"fips", "year"}]
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column].le(-888888000), column] = np.nan
    collapsed = df.groupby(["fips", "year"], as_index=False)[value_columns].max()
    return collapsed.sort_values(["fips", "year"]).groupby("fips", as_index=False).tail(1).reset_index(drop=True)


def ten_year_avg_by_fips(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> pd.DataFrame:
    quoted = ", ".join(f'"{column}"' for column in columns)
    df = con.execute(
        f"""
        SELECT {quoted}
        FROM {table}
        WHERE fips IS NOT NULL
          AND year IS NOT NULL
          AND year >= (SELECT max(year) FROM {table}) - 9
        """
    ).df()
    if df.empty:
        return df
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    value_columns = [column for column in columns if column not in {"fips", "year"}]
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column].le(-888888000), column] = np.nan
    return df.groupby("fips", as_index=False)[value_columns].mean()


def build_playbook_data(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """Build data for the County Climate Risk Playbook section."""
    models_dir = ROOT / "output" / "models" / "climate_risk_prediction" / "overall"
    if not models_dir.exists():
        return {"available": False, "message": "Climate risk prediction models not found"}

    results_files = sorted(models_dir.glob("overall_results_*.json"), reverse=True)
    if not results_files:
        return {"available": False, "message": "No model results found"}

    with open(results_files[0], "r") as f:
        model_results = json.load(f)

    best_model_name = max(
        model_results["models"].items(),
        key=lambda x: x[1].get("accuracy", 0),
    )[0]
    best_model = model_results["models"][best_model_name]
    top_features = best_model.get("top_features", [])

    nri = con.execute("SELECT fips, risk_rating, risk_score FROM mart.nri_county_risk WHERE fips IS NOT NULL").df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)

    counties = con.execute("""
        SELECT DISTINCT fips, any_value(REGION) as county, any_value(STATE_CODE) as state
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
        GROUP BY fips
    """).df()
    counties["fips"] = counties["fips"].astype(str).str.zfill(5)
    playbook_counties = nri.merge(counties, on="fips", how="inner")

    # Load per-county values for each PLAYBOOK_FEATURE_MAP entry that maps to a profileKey
    econ = latest_by_fips(con, "mart.acs_county_economic_annual", [
        "fips", "year",
        "dp03_income_and_benefits_total_households_median_household_income_est",
        "dp03_civilian_labor_force_unemployment_rate_pct",
    ])
    demo = latest_by_fips(con, "mart.acs_county_demographic_annual", [
        "fips", "year", "domestic_in_migration_rate",
    ])
    redfin_pf = con.execute("""
        SELECT
            lpad(fips, 5, '0') AS fips,
            avg(CASE WHEN try_cast(replace(HOMES_SOLD_YOY,',','') AS DOUBLE) <= -888888000 THEN NULL
                     ELSE try_cast(replace(HOMES_SOLD_YOY,',','') AS DOUBLE) END) AS homes_sold_yoy,
            avg(CASE WHEN try_cast(replace(NEW_LISTINGS_YOY,',','') AS DOUBLE) <= -888888000 THEN NULL
                     ELSE try_cast(replace(NEW_LISTINGS_YOY,',','') AS DOUBLE) END) AS new_listings_yoy,
            avg(CASE WHEN try_cast(replace(MEDIAN_DOM_YOY,',','') AS DOUBLE) <= -888888000 THEN NULL
                     ELSE try_cast(replace(MEDIAN_DOM_YOY,',','') AS DOUBLE) END) AS median_dom_yoy
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
          AND period_begin IS NOT NULL
          AND extract(year FROM period_begin) >= (
              SELECT max(extract(year FROM period_begin)) - 9
              FROM mart.redfin_county_monthly
              WHERE period_begin IS NOT NULL
          )
          AND coalesce(property_type, PROPERTY_TYPE_1) = 'All Residential'
        GROUP BY fips
    """).df()
    redfin_pf["fips"] = redfin_pf["fips"].astype(str).str.zfill(5)

    # BEA income components
    bea = latest_by_fips(con, "mart.statsamerica_bea_personal_income_annual", [
        "fips", "year",
        "net_earnings_by_place_of_residence",
        "dividends_interest_rent",
        "transfer_receipts",
        "population",
    ]) if "mart.statsamerica_bea_personal_income_annual" in [r[0] for r in con.execute("SHOW TABLES").fetchall()] else pd.DataFrame(columns=["fips"])

    afford_pf = latest_by_fips(con, "mart.acs_county_affordability_annual", [
        "fips", "year",
        "housing_cost_pct_income",
        "owner_mortgage_cost_burden_30pct_plus",
    ])

    # Estimated insurance, tax, utilities need the full computation — reuse from feature payload context
    # For playbook we compute simplified per-county income-share ratios using ACS estimates
    county_features = (
        playbook_counties[["fips", "riskRating", "riskScore" if "riskScore" in playbook_counties.columns else "risk_score"]]
        .rename(columns={"risk_score": "riskScore"})
        .merge(econ[["fips", "dp03_income_and_benefits_total_households_median_household_income_est", "dp03_civilian_labor_force_unemployment_rate_pct"]], on="fips", how="left")
        .merge(demo[["fips", "domestic_in_migration_rate"]], on="fips", how="left")
        .merge(redfin_pf, on="fips", how="left")
        .merge(afford_pf[["fips", "housing_cost_pct_income"]], on="fips", how="left")
    )
    if not bea.empty and "population" in bea.columns:
        bea["net_earnings_per_capita"] = pd.to_numeric(bea["net_earnings_by_place_of_residence"], errors="coerce") * 1000 / pd.to_numeric(bea["population"], errors="coerce").replace(0, np.nan)
        bea["dividends_interest_rent_per_capita"] = pd.to_numeric(bea["dividends_interest_rent"], errors="coerce") * 1000 / pd.to_numeric(bea["population"], errors="coerce").replace(0, np.nan)
        bea["transfer_receipts_per_capita"] = pd.to_numeric(bea["transfer_receipts"], errors="coerce") * 1000 / pd.to_numeric(bea["population"], errors="coerce").replace(0, np.nan)
        county_features = county_features.merge(bea[["fips", "net_earnings_per_capita", "dividends_interest_rent_per_capita", "transfer_receipts_per_capita"]], on="fips", how="left")

    def _pct_income(series: pd.Series, income: pd.Series) -> pd.Series:
        return (series / income.replace(0, np.nan) * 100).where(income.notna() & series.notna())

    # Build county feature rows dict keyed by fips
    county_feature_map: dict[str, dict[str, object]] = {}
    for row in county_features.itertuples(index=False):
        income = getattr(row, "dp03_income_and_benefits_total_households_median_household_income_est", None)
        county_feature_map[row.fips] = {
            "income": serialize_number(income, 0),
            "unemploymentRate": serialize_number(getattr(row, "dp03_civilian_labor_force_unemployment_rate_pct", None), 2),
            "netMigration": serialize_number(getattr(row, "domestic_in_migration_rate", None), 2),
            "homesSoldYoy": serialize_number(getattr(row, "homes_sold_yoy", None), 5),
            "newListingsYoy": serialize_number(getattr(row, "new_listings_yoy", None), 5),
            "medianDomYoy": serialize_number(getattr(row, "median_dom_yoy", None), 5),
            "housingCostPctIncome": serialize_number(getattr(row, "housing_cost_pct_income", None), 2),
            "netEarningsPerCapita": serialize_number(getattr(row, "net_earnings_per_capita", None), 0) if hasattr(row, "net_earnings_per_capita") else None,
            "dividendsInterestRentPerCapita": serialize_number(getattr(row, "dividends_interest_rent_per_capita", None), 0) if hasattr(row, "dividends_interest_rent_per_capita") else None,
            "transferReceiptsPerCapita": serialize_number(getattr(row, "transfer_receipts_per_capita", None), 0) if hasattr(row, "transfer_receipts_per_capita") else None,
        }

    return {
        "available": True,
        "model": {
            "name": best_model_name,
            "accuracy": serialize_number(best_model.get("accuracy"), 4),
            "f1Weighted": serialize_number(best_model.get("f1_weighted"), 4),
            "featureNames": model_results["feature_names"],
            "topFeatures": top_features,
        },
        "featureMap": PLAYBOOK_FEATURE_MAP,
        "counties": [
            {
                "fips": row.fips,
                "county": row.county,
                "state": row.state,
                "riskRating": row.riskRating,
                "riskScore": serialize_number(row.risk_score, 3),
            }
            for row in playbook_counties.itertuples(index=False)
        ],
        "countyFeatures": county_feature_map,
    }


def build_county_playbook_data(
    con: duckdb.DuckDBPyConnection,
) -> dict[str, object]:
    """Build county hazard ratings, monthly PPSF history, and event periods."""
    hazard_cols: list[str] = []
    for hazard in HAZARDS:
        if hazard["key"] == "overall":
            hazard_cols.extend(["risk_score", "risk_rating"])
        else:
            hazard_cols.extend([hazard["score"], hazard["rating"]])
    nri = con.execute(
        f"""
        SELECT fips, COUNTY, STATEABBRV, {", ".join(hazard_cols)}
        FROM mart.nri_county_risk
        WHERE fips IS NOT NULL
        """
    ).df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    labels = con.execute(
        """
        SELECT lpad(fips, 5, '0') AS fips,
               any_value(REGION) AS county_label,
               any_value(STATE_CODE) AS state_code
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
        GROUP BY fips
        """
    ).df()
    labels["fips"] = labels["fips"].astype(str).str.zfill(5)
    nri = nri.merge(labels, on="fips", how="left")
    for hazard in HAZARDS:
        nri[hazard["score"]] = clean_numeric(nri[hazard["score"]])
        nri[hazard["rating"]] = nri[hazard["rating"]].map(rating_clean)
    counties = []
    for row in nri.itertuples(index=False):
        state = (
            str(row.state_code)
            if pd.notna(row.state_code)
            else str(row.STATEABBRV)
            if pd.notna(row.STATEABBRV)
            else ""
        )
        county = (
            str(row.county_label)
            if pd.notna(row.county_label)
            else f"{row.COUNTY}, {state}"
            if pd.notna(row.COUNTY)
            else f"County FIPS {row.fips}"
        )
        hazards = {}
        for hazard in HAZARDS:
            rating = getattr(row, hazard["rating"])
            hazards[hazard["key"]] = {
                "score": serialize_number(getattr(row, hazard["score"]), 3),
                "rating": rating,
                "ratingValue": RISK_NUMERIC.get(rating),
            }
        counties.append(
            {
                "fips": row.fips,
                "county": county,
                "state": state,
                "riskRating": hazards["overall"]["rating"],
                "riskScore": hazards["overall"]["score"],
                "hazards": hazards,
            }
        )
    if not counties:
        return {"available": False, "message": "County housing and NRI data are unavailable"}

    history = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            date_trunc('month', period_begin)::DATE AS month,
            avg(CASE
                WHEN try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL
                ELSE try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE)
            END) AS median_ppsf_yoy
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
          AND period_begin IS NOT NULL
          AND coalesce(property_type, PROPERTY_TYPE_1) = 'All Residential'
          AND extract(year FROM period_begin) >= (
              SELECT max(extract(year FROM period_begin)) - 9
              FROM mart.redfin_county_monthly
              WHERE period_begin IS NOT NULL
          )
        GROUP BY fips, date_trunc('month', period_begin)
        ORDER BY fips, month
        """
    ).df()
    history["fips"] = history["fips"].astype(str).str.zfill(5)
    history["median_ppsf_yoy"] = pd.to_numeric(history["median_ppsf_yoy"], errors="coerce")
    history = history.dropna(subset=["median_ppsf_yoy"])

    events = load_disaster_events(con)
    events = events.loc[
        events["event_start_month"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-01"))
    ].drop_duplicates("event_key")
    county_fips = {str(county["fips"]).zfill(5) for county in counties}
    events = events.loc[events["fips"].isin(county_fips)].sort_values(["fips", "event_start_month"])

    monthly_history_by_fips: dict[str, list[dict[str, object]]] = {}
    for row in history.itertuples(index=False):
        monthly_history_by_fips.setdefault(row.fips, []).append(
            {
                "month": row.month.strftime("%Y-%m"),
                "value": serialize_number(row.median_ppsf_yoy, 5),
            }
        )
    events_by_fips: dict[str, list[dict[str, object]]] = {}
    for row in events.itertuples(index=False):
        events_by_fips.setdefault(row.fips, []).append(
            {
                "eventKey": row.event_key,
                "source": row.event_source,
                "type": row.event_type,
                "name": row.event_name,
                "start": row.event_start_month.strftime("%Y-%m"),
                "end": row.event_end_month.strftime("%Y-%m"),
            }
        )

    return {
        "available": True,
        "hazards": [{"key": hazard["key"], "label": hazard["label"]} for hazard in HAZARDS],
        "counties": counties,
        "monthlyHistoryByFips": monthly_history_by_fips,
        "eventsByFips": events_by_fips,
        "historyStart": history["month"].min().strftime("%Y-%m"),
        "historyEnd": history["month"].max().strftime("%Y-%m"),
    }


def build_feature_payload(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    nri = con.execute("SELECT fips, risk_rating, risk_score FROM mart.nri_county_risk WHERE fips IS NOT NULL").df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)
    nri["riskValue"] = nri["riskRating"].map(RISK_NUMERIC)

    econ_cols = [
        "fips",
        "year",
        "dp03_income_and_benefits_total_households_median_household_income_est",
        "dp03_civilian_labor_force_unemployment_rate_pct",
    ]
    demo_cols = [
        "fips",
        "year",
        "domestic_in_migration_rate",
        "dp05_total_population_65_plus_pct",
        "dp02_households_by_type_total_households_households_with_one_or_more_people_65_plus_pct",
        "dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct",
        "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct",
        "dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct",
    ]
    affordability_cols = [
        "fips",
        "year",
        "s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est",
        "s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_median_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_30_0_to_34_9_percent_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_35_0_percent_or_more_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_30_0_to_34_9_percent_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_35_0_percent_or_more_pct",
        "median_owner_costs_mortgage",
        "housing_cost_pct_income",
        "owner_mortgage_cost_burden_30pct_plus",
        "b25132_monthly_electricity_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_less_than_dollars_50_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_50_to_dollars_99_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_100_to_dollars_149_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_150_to_dollars_199_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_200_to_dollars_249_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_250_or_more_est",
        "b25133_monthly_gas_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_less_than_dollars_25_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_25_to_dollars_49_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_50_to_dollars_74_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_75_to_dollars_99_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_100_to_dollars_149_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_150_or_more_est",
        "b25134_annual_water_and_sewer_costs_total_not_charged_or_payment_included_in_other_fees_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_less_than_dollars_125_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_125_to_dollars_249_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_250_to_dollars_499_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_500_to_dollars_749_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_750_to_dollars_999_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_1_000_or_more_est",
        "b25135_annual_other_fuel_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est",
        "b25135_annual_other_fuel_costs_total_charged_for_other_fuels_less_than_dollars_250_est",
        "b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_250_to_dollars_749_est",
        "b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_750_or_more_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_no_real_estate_taxes_paid_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_less_than_dollars_800_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_dollars_800_to_dollars_1_499_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_dollars_1_500_or_more_est",
        "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_no_real_estate_taxes_paid_est",
        "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_less_than_dollars_800_est",
        "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_dollars_800_to_dollars_1_499_est",
        "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_dollars_1_500_or_more_est",
    ]
    insurance_cols = [
        f"b25141_homeowners_insurance_costs_by_mortgage_status_total_{status}_{suffix}_est"
        for status in ["mortgage", "not_mortgaged"]
        for suffix in [
            "less_than_dollars_100",
            "dollars_100_to_dollars_299",
            "dollars_300_to_dollars_499",
            "dollars_500_to_dollars_799",
            "dollars_800_to_dollars_999",
            "dollars_1000_to_dollars_1499",
            "dollars_1500_to_dollars_1999",
            "dollars_2000_to_dollars_2499",
            "dollars_2500_to_dollars_2999",
            "dollars_3000_to_dollars_3499",
            "dollars_3500_to_dollars_3999",
            "dollars_4000_or_more",
        ]
    ]
    affordability_cols.extend(insurance_cols)

    econ = ten_year_avg_by_fips(con, "mart.acs_county_economic_annual", econ_cols)
    demo = ten_year_avg_by_fips(con, "mart.acs_county_demographic_annual", demo_cols)
    afford = ten_year_avg_by_fips(con, "mart.acs_county_affordability_annual", affordability_cols)
    weather = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            avg(avg_temperature_f) AS avg_temperature_f,
            avg(precipitation_inches) AS precipitation_inches
        FROM mart.ncei_county_weather_monthly
        WHERE fips IS NOT NULL
          AND weather_month IS NOT NULL
          AND extract(year FROM weather_month) >= (
              SELECT max(extract(year FROM weather_month)) - 9
              FROM mart.ncei_county_weather_monthly
              WHERE weather_month IS NOT NULL
          )
        GROUP BY fips
        """
    ).df()
    weather["fips"] = weather["fips"].astype(str).str.zfill(5)
    migration = con.execute(
        """
        WITH net AS (
          SELECT lpad(fips, 5, '0') AS fips, avg(CAST(total_net_migration AS DOUBLE)) AS avg_total_net_migration
          FROM mart.statsamerica_population_components_annual
          WHERE fips IS NOT NULL
            AND year >= (SELECT max(year) FROM mart.statsamerica_population_components_annual) - 9
          GROUP BY fips
        ),
        pop AS (
          SELECT lpad(fips, 5, '0') AS fips,
                 avg(try_cast(replace(nullif(trim(cast(total_population AS VARCHAR)), ''), ',', '') AS DOUBLE)) AS avg_population
          FROM mart.acs_county_demographic_annual
          WHERE fips IS NOT NULL
            AND year >= (SELECT max(year) FROM mart.acs_county_demographic_annual) - 9
          GROUP BY fips
        )
        SELECT net.fips, avg_total_net_migration / nullif(avg_population, 0) AS net_migration_rate
        FROM net
        LEFT JOIN pop ON net.fips = pop.fips
        """
    ).df()
    migration["fips"] = migration["fips"].astype(str).str.zfill(5)

    for frame in [econ, demo, afford, weather]:
        for column in frame.columns:
            if column != "fips":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    afford["estimated_annual_home_insurance"] = weighted_bucket_average(
        afford,
        [
            (f"b25141_homeowners_insurance_costs_by_mortgage_status_total_{status}_{suffix}_est", midpoint)
            for status in ["mortgage", "not_mortgaged"]
            for suffix, midpoint in [
                ("less_than_dollars_100", 50),
                ("dollars_100_to_dollars_299", 200),
                ("dollars_300_to_dollars_499", 400),
                ("dollars_500_to_dollars_799", 650),
                ("dollars_800_to_dollars_999", 900),
                ("dollars_1000_to_dollars_1499", 1250),
                ("dollars_1500_to_dollars_1999", 1750),
                ("dollars_2000_to_dollars_2499", 2250),
                ("dollars_2500_to_dollars_2999", 2750),
                ("dollars_3000_to_dollars_3499", 3250),
                ("dollars_3500_to_dollars_3999", 3750),
                ("dollars_4000_or_more", 4250),
            ]
        ],
    )
    afford["estimated_annual_property_tax"] = weighted_bucket_average(
        afford,
        [
            (f"{prefix}_{suffix}_est", midpoint)
            for prefix in ["s2506_owner_occupied_units_mortgage_real_estate_taxes", "s2507_owner_occupied_units_no_mortgage_real_estate_taxes"]
            for suffix, midpoint in [
                ("less_than_dollars_800", 400),
                ("dollars_800_to_dollars_1_499", 1150),
                ("dollars_1_500_or_more", 2000),
            ]
        ],
        zero_cols=[
            "s2506_owner_occupied_units_mortgage_real_estate_taxes_no_real_estate_taxes_paid_est",
            "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_no_real_estate_taxes_paid_est",
        ],
    )
    electricity = weighted_bucket_average(
        afford,
        [
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_less_than_dollars_50_est", 25),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_50_to_dollars_99_est", 75),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_100_to_dollars_149_est", 125),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_150_to_dollars_199_est", 175),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_200_to_dollars_249_est", 225),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_250_or_more_est", 275),
        ],
        zero_cols=["b25132_monthly_electricity_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"],
    ) * 12
    gas = weighted_bucket_average(
        afford,
        [
            ("b25133_monthly_gas_costs_total_charged_for_gas_less_than_dollars_25_est", 12.5),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_25_to_dollars_49_est", 37.5),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_50_to_dollars_74_est", 62.5),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_75_to_dollars_99_est", 87.5),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_100_to_dollars_149_est", 125),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_150_or_more_est", 175),
        ],
        zero_cols=["b25133_monthly_gas_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"],
    ) * 12
    water = weighted_bucket_average(
        afford,
        [
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_less_than_dollars_125_est", 62.5),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_125_to_dollars_249_est", 187.5),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_250_to_dollars_499_est", 375),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_500_to_dollars_749_est", 625),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_750_to_dollars_999_est", 875),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_1_000_or_more_est", 1125),
        ],
        zero_cols=["b25134_annual_water_and_sewer_costs_total_not_charged_or_payment_included_in_other_fees_est"],
    )
    other_fuel = weighted_bucket_average(
        afford,
        [
            ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_less_than_dollars_250_est", 125),
            ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_250_to_dollars_749_est", 500),
            ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_750_or_more_est", 875),
        ],
        zero_cols=["b25135_annual_other_fuel_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"],
    )
    afford["estimated_annual_utilities"] = electricity + gas + water + other_fuel
    afford["income_median_household_usd"] = afford["s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est"]
    afford["insurance_homeowners_pct_income"] = afford["estimated_annual_home_insurance"] / afford["income_median_household_usd"].replace(0, np.nan) * 100
    afford["property_taxes_pct_income"] = afford["s2506_owner_occupied_units_mortgage_real_estate_taxes_median_est"] / afford["income_median_household_usd"].replace(0, np.nan) * 100
    afford["utilities_pct_income"] = afford["estimated_annual_utilities"] / afford["income_median_household_usd"].replace(0, np.nan) * 100
    afford["housing_burden_30pct_plus_share"] = mean_available(
        afford,
        [
            "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_30_0_to_34_9_percent_pct",
            "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_35_0_percent_or_more_pct",
            "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_30_0_to_34_9_percent_pct",
            "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_35_0_percent_or_more_pct",
        ],
    )
    afford["homeownership_cost_pct_income"] = (
        afford["s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est"] * 12
        / afford["income_median_household_usd"].replace(0, np.nan)
        * 100
    )

    features = (
        nri[["fips", "riskRating", "riskValue", "risk_score"]]
        .merge(econ[["fips", *econ_cols[2:]]], on="fips", how="left")
        .merge(demo[["fips", *demo_cols[2:]]], on="fips", how="left")
        .merge(migration, on="fips", how="left")
        .merge(afford[[
            "fips",
            "median_owner_costs_mortgage",
            "housing_cost_pct_income",
            "owner_mortgage_cost_burden_30pct_plus",
            "estimated_annual_home_insurance",
            "estimated_annual_property_tax",
            "estimated_annual_utilities",
            "income_median_household_usd",
            "insurance_homeowners_pct_income",
            "property_taxes_pct_income",
            "utilities_pct_income",
            "housing_burden_30pct_plus_share",
            "homeownership_cost_pct_income",
        ]], on="fips", how="left")
        .merge(weather, on="fips", how="left")
    )
    bea_features = con.execute(
        """
        WITH bea AS (
          SELECT
            lpad(fips, 5, '0') AS fips,
            avg(net_earnings_by_place_of_residence_thousands * 1000.0 / nullif(population, 0)) AS net_earnings_per_capita,
            avg(dividends_interest_rent_thousands * 1000.0 / nullif(population, 0)) AS dividends_interest_rent_per_capita,
            avg(transfer_receipts_thousands * 1000.0 / nullif(population, 0)) AS transfer_receipts_per_capita
          FROM mart.statsamerica_bea_personal_income_annual
          WHERE fips IS NOT NULL
            AND year >= (SELECT max(year) FROM mart.statsamerica_bea_personal_income_annual) - 9
            AND population > 0
          GROUP BY fips
        ),
        cew AS (
          SELECT
            lpad(s.fips, 5, '0') AS fips,
            avg(s.total_wages_dollars / nullif(t.total_wages_dollars, 0) * 100) AS accom_food_wages_pct_total_wages
          FROM mart.statsamerica_cew_county_sector_annual s
          JOIN mart.statsamerica_cew_county_annual t
            ON lpad(s.fips, 5, '0') = lpad(t.fips, 5, '0')
           AND s.year = t.year
          WHERE s.naics_code = '72'
            AND s.year >= (SELECT max(year) FROM mart.statsamerica_cew_county_sector_annual) - 9
            AND t.total_wages_dollars > 0
          GROUP BY s.fips
        )
        SELECT
          coalesce(bea.fips, cew.fips) AS fips,
          net_earnings_per_capita,
          dividends_interest_rent_per_capita,
          transfer_receipts_per_capita,
          accom_food_wages_pct_total_wages
        FROM bea
        FULL OUTER JOIN cew ON bea.fips = cew.fips
        """
    ).df()
    if not bea_features.empty:
        bea_features["fips"] = bea_features["fips"].astype(str).str.zfill(5)
        features = features.merge(bea_features, on="fips", how="left")
    redfin_features = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            avg(CASE WHEN try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) END) AS median_ppsf_yoy,
            avg(CASE WHEN try_cast(replace(AVG_SALE_TO_LIST_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(AVG_SALE_TO_LIST_YOY, ',', '') AS DOUBLE) END) AS avg_sale_to_list_yoy,
            avg(CASE WHEN try_cast(replace(HOMES_SOLD_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(HOMES_SOLD_YOY, ',', '') AS DOUBLE) END) AS homes_sold_yoy,
            avg(CASE WHEN try_cast(replace(INVENTORY_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(INVENTORY_YOY, ',', '') AS DOUBLE) END) AS inventory_yoy,
            avg(CASE WHEN try_cast(replace(NEW_LISTINGS_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(NEW_LISTINGS_YOY, ',', '') AS DOUBLE) END) AS new_listings_yoy,
            avg(CASE WHEN try_cast(replace(MEDIAN_DOM_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(MEDIAN_DOM_YOY, ',', '') AS DOUBLE) END) AS median_dom_yoy,
            avg(CASE WHEN try_cast(replace(PRICE_DROPS_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(PRICE_DROPS_YOY, ',', '') AS DOUBLE) END) AS price_drops_yoy
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
          AND period_begin IS NOT NULL
          AND extract(year FROM period_begin) >= (
              SELECT max(extract(year FROM period_begin)) - 9
              FROM mart.redfin_county_monthly
              WHERE period_begin IS NOT NULL
          )
          AND coalesce(property_type, PROPERTY_TYPE_1) = 'All Residential'
        GROUP BY fips
        """
    ).df()
    redfin_features["fips"] = redfin_features["fips"].astype(str).str.zfill(5)
    features = features.merge(redfin_features, on="fips", how="left")
    features["no_broadband_pct"] = 100 - features["dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct"]
    feature_defs = [
        ("Economic", "Income", "income_median_household_usd", "currency", "mart.acs_county_affordability_annual"),
        ("Economic", "Insurance % of Income", "insurance_homeowners_pct_income", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Property Tax % of Income", "property_taxes_pct_income", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Utilities % of Income", "utilities_pct_income", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Housing Burden", "housing_burden_30pct_plus_share", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Homeownership Cost Share", "homeownership_cost_pct_income", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Unemployment", "dp03_civilian_labor_force_unemployment_rate_pct", "percent", "mart.acs_county_economic_annual"),
        ("Economic", "Net Earnings per Capita", "net_earnings_per_capita", "currency", "mart.statsamerica_bea_personal_income_annual"),
        ("Economic", "Dividends/Interest/Rent per Capita", "dividends_interest_rent_per_capita", "currency", "mart.statsamerica_bea_personal_income_annual"),
        ("Economic", "Transfer Receipts per Capita", "transfer_receipts_per_capita", "currency", "mart.statsamerica_bea_personal_income_annual"),
        ("Economic", "Accom. & Food Wages % Total Wages", "accom_food_wages_pct_total_wages", "percent", "mart.statsamerica_cew_county_sector_annual"),
        ("Demographic", "Net Migration Rate", "net_migration_rate", "number", "mart.statsamerica_population_components_annual"),
        ("Demographic", "Age >= 65 Years", "dp05_total_population_65_plus_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Demographic", "Disability Status", "dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Demographic", "Communication Barrier", "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Demographic", "No Internet Access", "no_broadband_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Housing Market", "Median PPSF YOY", "median_ppsf_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "Average Sale-to-List YOY", "avg_sale_to_list_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "Homes Sold YOY", "homes_sold_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "Inventory YOY", "inventory_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "New Listings YOY", "new_listings_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "Median Days on Market YOY", "median_dom_yoy", "number", "mart.redfin_county_monthly"),
        ("Housing Market", "Price Drops YOY", "price_drops_yoy", "pct", "mart.redfin_county_monthly"),
        ("Climate", "Temperature", "avg_temperature_f", "number", "mart.ncei_county_weather_monthly"),
        ("Climate", "Precipitation", "precipitation_inches", "number", "mart.ncei_county_weather_monthly"),
    ]
    excluded_feature_labels = {
        "Median PPSF YOY",
        "Homeownership Cost Share",
        "Accom. & Food Wages % Total Wages",
    }
    for _, _, column, _, _ in feature_defs:
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    rows = []
    correlations = []
    feature_bin_impacts: dict[str, list[dict[str, object]]] = {}
    county_feature_bins: dict[str, dict[str, dict[str, object]]] = {
        str(fips): {} for fips in features["fips"].astype(str)
    }
    global_risk_value_baseline = features["riskValue"].median()
    global_risk_score_baseline = features["risk_score"].median()
    for category, label, column, fmt, source in feature_defs:
        valid = features.dropna(subset=[column, "riskRating", "riskValue"]).copy()
        if valid.empty:
            continue
        risk_corr = valid[["riskValue", column]].corr(method="spearman").iloc[0, 1]
        correlations.append({"feature": label, "category": category, "corr": serialize_number(risk_corr, 3)})
        valid["bucket"], bucket_order = feature_bucket_labels(valid[column], fmt)
        valid = valid.dropna(subset=["bucket"]).copy()
        totals = valid.groupby("riskRating", observed=False)["fips"].nunique().reindex(RISK_ORDER).fillna(0)
        counts = valid.groupby(["riskRating", "bucket"], observed=False)["fips"].nunique()
        bucket_summaries = (
            valid.groupby("bucket", observed=False)
            .agg(
                median_risk_value=("riskValue", "median"),
                median_risk_score=("risk_score", "median"),
                count=("fips", "nunique"),
            )
            .reset_index()
        )
        bucket_summary_by_name = {}
        for summary in bucket_summaries.itertuples(index=False):
            bucket_order_index = bucket_order.index(str(summary.bucket)) if str(summary.bucket) in bucket_order else None
            association = classify_bucket_position(bucket_order_index, len(bucket_order), risk_corr)
            bucket_summary_by_name[str(summary.bucket)] = {
                "bucket": str(summary.bucket),
                "bucketOrder": bucket_order_index,
                "bucketCount": len(bucket_order),
                "medianRiskValue": serialize_number(summary.median_risk_value, 3),
                "medianRiskScore": serialize_number(summary.median_risk_score, 3),
                "riskAssociation": association,
                "riskCorrelation": serialize_number(risk_corr, 3),
                "count": int(summary.count),
                "baselineRiskValue": serialize_number(global_risk_value_baseline, 3),
                "baselineRiskScore": serialize_number(global_risk_score_baseline, 3),
                "format": fmt,
            }
        feature_bin_impacts[label] = [bucket_summary_by_name[bucket] for bucket in bucket_order if bucket in bucket_summary_by_name]
        for row in valid[["fips", column, "bucket"]].itertuples(index=False):
            summary = bucket_summary_by_name.get(str(row.bucket))
            if not summary:
                continue
            county_feature_bins.setdefault(str(row.fips), {})[label] = {
                "value": serialize_number(row[1], 4),
                "bucket": str(row.bucket),
                "bucketOrder": summary["bucketOrder"],
                "bucketCount": summary["bucketCount"],
                "riskAssociation": summary["riskAssociation"],
                "medianRiskValue": summary["medianRiskValue"],
                "medianRiskScore": summary["medianRiskScore"],
                "riskCorrelation": summary["riskCorrelation"],
                "baselineRiskValue": summary["baselineRiskValue"],
                "baselineRiskScore": summary["baselineRiskScore"],
            }
        for rating in RISK_ORDER:
            for bucket in bucket_order:
                count = int(counts.get((rating, bucket), 0))
                total = int(totals.get(rating, 0))
                rows.append(
                    {
                        "category": category,
                        "feature": label,
                        "riskRating": rating,
                        "bucket": bucket,
                        "bucketOrder": bucket_order.index(bucket),
                        "share": serialize_number(count / total if total else None, 4),
                        "count": count,
                        "total": total,
                        "format": fmt,
                        "source": source,
                    }
                )
    candidate_feature_defs = [definition for definition in feature_defs if definition[1] not in excluded_feature_labels]
    selected_feature_labels = [
        item["feature"]
        for item in sorted(
            [item for item in correlations if item["feature"] not in excluded_feature_labels],
            key=lambda item: abs(item["corr"] or 0),
            reverse=True,
        )[:8]
    ]
    top = [item for item in correlations if item["feature"] in selected_feature_labels]
    top = sorted(top, key=lambda item: abs(item["corr"] or 0), reverse=True)
    selected_feature_defs = [definition for definition in feature_defs if definition[1] in selected_feature_labels]
    feature_display_meta = {
        label: {"category": category, "column": column, "format": fmt, "source": source}
        for category, label, column, fmt, source in feature_defs
    }
    county_profiles = []
    for row in features.itertuples(index=False):
        feature_values = {
            label: serialize_number(getattr(row, meta["column"]), 5)
            for label, meta in feature_display_meta.items()
            if hasattr(row, meta["column"])
        }
        county_profiles.append(
            {
                "fips": row.fips,
                "riskRating": row.riskRating,
                "featureValues": feature_values,
                "income": serialize_number(getattr(row, "income_median_household_usd"), 2),
                "housingBurden": serialize_number(getattr(row, "housing_burden_30pct_plus_share"), 2),
                "insurance": serialize_number(getattr(row, "estimated_annual_home_insurance"), 2),
                "propertyTaxes": serialize_number(getattr(row, "estimated_annual_property_tax"), 2),
                "utilities": serialize_number(getattr(row, "estimated_annual_utilities"), 2),
                "netMigration": serialize_number(getattr(row, "net_migration_rate"), 5),
                "homesSoldYoy": serialize_number(getattr(row, "homes_sold_yoy"), 5),
                "medianDomYoy": serialize_number(getattr(row, "median_dom_yoy"), 5),
                "newListingsYoy": serialize_number(getattr(row, "new_listings_yoy"), 5),
                "unemploymentRate": serialize_number(getattr(row, "dp03_civilian_labor_force_unemployment_rate_pct"), 2),
            }
        )

    # --- Option B: within-group feature correlations ---
    # For each risk rating, Spearman correlation of each feature with the continuous
    # NRI risk_score (0–100) restricted to counties in that tier. Uses the raw score
    # rather than the integer riskValue so there is meaningful variance within each tier.
    events_for_position = load_disaster_events(con)
    events_for_position = events_for_position.loc[
        events_for_position["event_start_month"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-01"))
    ].copy()
    housing_for_position = load_redfin_county_monthly(con)
    housing_for_position.loc[pd.to_numeric(housing_for_position["median_ppsf_yoy"], errors="coerce").le(-888888000), "median_ppsf_yoy"] = np.nan
    affected_for_position = build_affected_event_windows(
        events_for_position,
        housing_for_position,
        pre_event_months=24,
        post_event_months=60,
    )
    required_position_months = event_window_months(12, 36)
    complete_position = filter_complete_event_window_lines(
        affected_for_position,
        x_col="event_window_month",
        line_col="line_id",
        metric_col="median_ppsf_yoy",
        required_x_values=required_position_months,
    ).copy()
    complete_position = complete_position.loc[complete_position["event_window_month"].isin(required_position_months)].copy()
    complete_position = complete_position.merge(nri[["fips", "riskRating"]], on="fips", how="left")
    line_position = (
        complete_position.dropna(subset=["riskRating", "median_ppsf_yoy"])
        .groupby(["line_id", "fips", "riskRating"], as_index=False)["median_ppsf_yoy"]
        .mean()
        .rename(columns={"median_ppsf_yoy": "avg_ppsf_yoy"})
    )
    line_position["group_median_ppsf_yoy"] = line_position.groupby("riskRating", observed=False)["avg_ppsf_yoy"].transform("median")
    line_position["relative_position"] = line_position["avg_ppsf_yoy"] - line_position["group_median_ppsf_yoy"]
    position_analysis = line_position.merge(features, on="fips", how="inner", suffixes=("", "_feature"))

    within_group_correlations: dict[str, list[dict]] = {}
    within_group_top_features: dict[str, list[dict]] = {}
    within_group_feature_bins: dict[str, dict[str, dict[str, object]]] = {}
    feature_lookup = {
        label: {"category": category, "column": column, "format": fmt, "source": source}
        for category, label, column, fmt, source in feature_defs
    }
    for rating in RISK_ORDER:
        group = position_analysis[position_analysis["riskRating"] == rating].copy()
        rating_corrs = []
        for _, label, column, _, _ in candidate_feature_defs:
            valid_group = group.dropna(subset=[column, "relative_position"])
            if len(valid_group) < 10:
                corr = None
            else:
                corr = serialize_number(
                    valid_group[["relative_position", column]].corr(method="spearman").iloc[0, 1], 3
                )
            rating_corrs.append({"feature": label, "corr": corr})
        within_group_correlations[rating] = rating_corrs
        selected_corrs = [item for item in rating_corrs if item["corr"] is not None]
        selected_corrs = sorted(selected_corrs, key=lambda item: abs(item["corr"] or 0), reverse=True)[:10]
        within_group_top_features[rating] = selected_corrs
        within_group_feature_bins[rating] = {}
        group_position_baseline = group["relative_position"].median()
        for selected in selected_corrs:
            label = selected["feature"]
            meta = feature_lookup[label]
            column = meta["column"]
            valid_group = group.dropna(subset=[column, "relative_position"]).copy()
            if valid_group.empty:
                continue
            valid_group["bucket"], bucket_order = feature_bucket_labels(valid_group[column], meta["format"])
            valid_group = valid_group.dropna(subset=["bucket"]).copy()
            county_feature_values = valid_group[["fips", column]].drop_duplicates("fips").copy()
            county_feature_values["feature_percentile"] = (
                county_feature_values[column]
                .rank(method="average", pct=True, na_option="keep")
                .mul(100)
            )
            valid_group = valid_group.merge(
                county_feature_values[["fips", "feature_percentile"]],
                on="fips",
                how="left",
            )
            valid_group["feature_contribution"] = (
                (valid_group["feature_percentile"] / 100 * 2 - 1) * float(selected["corr"])
            )
            bucket_summaries = (
                valid_group.groupby("bucket", observed=False)
                .agg(
                    median_relative_position=("relative_position", "median"),
                    median_avg_ppsf_yoy=("avg_ppsf_yoy", "median"),
                    count=("line_id", "nunique"),
                )
                .reset_index()
            )
            bucket_summary_by_name = {}
            for summary in bucket_summaries.itertuples(index=False):
                bucket_order_index = bucket_order.index(str(summary.bucket)) if str(summary.bucket) in bucket_order else None
                ppsf_association = classify_bucket_position(bucket_order_index, len(bucket_order), selected["corr"])
                bucket_summary_by_name[str(summary.bucket)] = {
                    "bucket": str(summary.bucket),
                    "bucketOrder": bucket_order_index,
                    "bucketCount": len(bucket_order),
                    "medianRelativePosition": serialize_number(summary.median_relative_position, 5),
                    "medianAvgPpsfYoy": serialize_number(summary.median_avg_ppsf_yoy, 5),
                    "relativePpsfAssociation": ppsf_association,
                    "ppsfAssociation": ppsf_association,
                    "relativePpsfCorrelation": selected["corr"],
                    "ppsfCorrelation": selected["corr"],
                    "count": int(summary.count),
                    "baselineRelativePosition": serialize_number(group_position_baseline, 5),
                }
            for row in valid_group[
                ["fips", column, "bucket", "feature_percentile", "feature_contribution"]
            ].drop_duplicates("fips").itertuples(index=False):
                summary = bucket_summary_by_name.get(str(row.bucket))
                if not summary:
                    continue
                within_group_feature_bins[rating].setdefault(str(row.fips), {})[label] = {
                    "value": serialize_number(row[1], 4),
                    "valuePercentile": serialize_number(row.feature_percentile, 2),
                    "contribution": serialize_number(row.feature_contribution, 4),
                    "bucket": str(row.bucket),
                    "bucketOrder": summary["bucketOrder"],
                    "bucketCount": summary["bucketCount"],
                    "corr": selected["corr"],
                    "relativePpsfAssociation": summary["relativePpsfAssociation"],
                    "ppsfAssociation": summary["ppsfAssociation"],
                    "relativePpsfCorrelation": summary["relativePpsfCorrelation"],
                    "ppsfCorrelation": summary["ppsfCorrelation"],
                    "medianRelativePosition": summary["medianRelativePosition"],
                    "medianAvgPpsfYoy": summary["medianAvgPpsfYoy"],
                    "baselineRelativePosition": summary["baselineRelativePosition"],
                }

    # --- Within-group percentile ranks (event-window PPSF YoY) ---
    # Populated by main() after build_event_windows completes, since this function
    # doesn't have access to event-window data. Placeholder here; merged in main().
    return {
        "riskOrder": RISK_ORDER,
        "rows": rows,
        "correlations": correlations,
        "topFeatures": top,
        "selectedFeatures": selected_feature_labels,
        "featureMeta": feature_display_meta,
        "countyProfiles": county_profiles,
        "featureBinImpacts": feature_bin_impacts,
        "countyFeatureBins": county_feature_bins,
        "withinGroupCorrelations": within_group_correlations,
        "withinGroupTopFeatures": within_group_top_features,
        "withinGroupFeatureBins": within_group_feature_bins,
        "withinGroupPercentiles": {},
    }


def make_html(data: dict[str, object]) -> str:
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    return HTML_TEMPLATE.replace("__PAYLOAD__", payload)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Which Way the Wind Blows: Climate Risk and U.S. Housing Markets</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    :root {
      --paper: #f6f2ea;
      --ink: #172026;
      --muted: #66717b;
      --panel: #fffdf8;
      --line: #d8d0c4;
      --teal: #0f766e;
      --shadow: 0 18px 45px rgba(23, 32, 38, 0.10);
    }
    /* ---- sidebar card layout ---- */
    .card-with-sidebar { display: flex; gap: 0; }
    .card-sidebar { width: 140px; min-width: 120px; flex-shrink: 0; border-right: 1px solid var(--line); padding: 12px 10px; display: flex; flex-direction: column; gap: 6px; }
    .card-sidebar .sidebar-label { font-size: 11px; font-weight: 800; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 4px; }
    .card-sidebar button { border-radius: 6px; padding: 7px 10px; font-size: 12px; text-align: left; white-space: nowrap; border: 1px solid var(--line); background: #fff; cursor: pointer; font-weight: 700; }
    .card-sidebar button.active { border-color: transparent; color: white; }
    .card-main { flex: 1; min-width: 0; padding: 16px; }
    /* Right-aligned sidebar (hazard, risk toggles) */
    .card-with-sidebar-right { display: flex; gap: 0; }
    .card-sidebar-right { width: 140px; min-width: 120px; flex-shrink: 0; border-left: 1px solid var(--line); padding: 12px 10px; display: flex; flex-direction: column; gap: 6px; }
    .card-sidebar-right .sidebar-label { font-size: 11px; font-weight: 800; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 4px; }
    .card-sidebar-right button { border-radius: 6px; padding: 9px 10px; font-size: 12px; text-align: left; border: 1px solid var(--line); background: #fff; cursor: pointer; font-weight: 700; width: 100%; display: flex; align-items: center; gap: 6px; }
    .card-sidebar-right button.active { border-color: transparent; color: white; }
    .card-main-left { flex: 1; min-width: 0; padding: 16px; }
    .control-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 10px 0 12px; }
    .toggle-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .control-bar .sidebar-label { font-size: 11px; font-weight: 800; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-right: 4px; }
    .control-bar button { border-radius: 999px; padding: 8px 11px; font-size: 12px; text-align: left; border: 1px solid var(--line); background: #fff; cursor: pointer; font-weight: 800; display: inline-flex; align-items: center; gap: 6px; }
    .control-bar button.active { border-color: transparent; color: white; }
    .pricing-viz-grid { align-items: center; margin-bottom: 32px; }
    .map-with-legend { position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 0; }
    .map-with-legend .legend { position: absolute; top: 100%; left: 0; justify-content: center; width: 100%; }
    .feature-detail-list { display: grid; grid-template-columns: minmax(170px, 1fr) minmax(90px, auto) minmax(280px, 1.15fr) minmax(280px, 1.15fr); column-gap: 12px; align-items: center; }
    .feature-detail-header, .feature-detail-row { display: contents; }
    .feature-detail-header > div { padding: 0 0 7px; color: var(--muted); font-size: 10px; font-weight: 800; text-transform: uppercase; }
    .feature-detail-row > div { min-width: 0; padding: 9px 0; border-top: 1px solid rgba(216,208,196,.7); font-size: 13px; }
    .feature-scale { display: grid; grid-template-columns: 68px minmax(120px, 1fr) 68px; gap: 6px; align-items: center; color: var(--muted); font-size: 10px; }
    .feature-scale-bar { position: relative; height: 10px; border-radius: 999px; border: 1px solid rgba(23,32,38,.14); background: linear-gradient(90deg, #16803c 0%, #e0b33b 50%, #b42318 100%); }
    .feature-scale-bar::after { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: rgba(23,32,38,.42); }
    .feature-scale-bar.percentile { background: linear-gradient(90deg, #e8f1f5 0%, #6fa3b8 50%, #22566b 100%); }
    .feature-scale-bar.percentile::after { display: none; }
    .feature-scale-arrow { position: absolute; top: -8px; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 8px solid #172026; transform: translateX(-5px); }
    .county-count { margin-top: 8px; text-align: center; color: var(--ink); font-size: 20px; font-weight: 850; }
    #risk-play-button { margin-left: 4px; visibility: hidden; opacity: 0; pointer-events: none; transition: opacity 180ms ease; }
    #risk-play-button.visible { visibility: visible; opacity: 1; pointer-events: auto; }
    .playbook-map-wrap { position: relative; height: 380px; overflow: hidden; }
    .playbook-map-wrap .chart { height: 380px; }
    .playbook-map-controls { position: absolute; top: 10px; right: 10px; z-index: 3; display: flex; gap: 6px; }
    .playbook-map-controls button { min-width: 34px; justify-content: center; box-shadow: 0 2px 8px rgba(23,32,38,.16); }
    .hazard-icon { display: inline-flex; width: 22px; justify-content: center; margin-right: 6px; }
    .hazard-rating-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px 16px; margin: 10px 0 18px; }
    .hazard-rating-item { display: flex; justify-content: space-between; gap: 10px; align-items: center; padding: 9px 0; border-bottom: 1px solid var(--line); font-size: 13px; }
    .playbook-commentary { margin-top: 12px; padding: 13px 15px; border-left: 4px solid var(--teal); background: #edf7f4; font-size: 14px; line-height: 1.55; }
    /* ---- window frame transition ---- */
    .window-frame { transition: opacity 380ms ease, transform 380ms ease; }
    .window-frame.sliding-left { opacity: 0; transform: translateX(-32px); pointer-events: none; }
    .window-frame.sliding-right { opacity: 0; transform: translateX(32px); pointer-events: none; }
    .window-frame.sliding-in-left { animation: slideInLeft 380ms ease both; }
    .window-frame.sliding-in-right { animation: slideInRight 380ms ease both; }
    @keyframes slideInLeft { from { opacity: 0; transform: translateX(-32px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes slideInRight { from { opacity: 0; transform: translateX(32px); } to { opacity: 1; transform: translateX(0); } }
    /* Arrow nav for event window */
    .window-arrow { position: absolute; top: 50%; transform: translateY(-50%); width: 36px; height: 36px; border-radius: 50%; background: rgba(23,32,38,.75); color: white; display: flex; align-items: center; justify-content: center; font-size: 18px; cursor: pointer; opacity: 0; transition: opacity 200ms; z-index: 5; border: none; padding: 0; }
    .window-arrow:hover { background: rgba(23,32,38,.92); }
    .window-arrow-left { left: 8px; }
    .window-arrow-right { right: 8px; }
    .event-chart-wrap:hover .window-arrow { opacity: 1; }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main { width: min(1180px, calc(100vw - 28px)); margin: 0 auto; padding: 18px 0 56px; }
    .hero { min-height: 72vh; display: flex; align-items: center; border-bottom: 1px solid var(--line); padding: 30px 0 26px; }
    .eyebrow { text-transform: uppercase; font-size: 12px; font-weight: 800; color: var(--teal); letter-spacing: .08em; margin-bottom: 16px; }
    h1 { margin: 0; max-width: 1080px; font-size: clamp(44px, 7.8vw, 98px); line-height: .95; letter-spacing: 0; }
    .dek { margin: 24px 0 0; max-width: 1020px; color: #41505a; font-size: clamp(20px, 2.4vw, 30px); line-height: 1.32; }
    .slide { min-height: 96vh; padding: 56px 0; border-bottom: 1px solid var(--line); opacity: 0; transform: translateY(18px); transition: opacity 520ms ease, transform 520ms ease; }
    .slide.visible { opacity: 1; transform: translateY(0); }
    .slide.transition-out { opacity: 0; transform: translateX(-40px); }
    .slide.transition-in { opacity: 1; transform: translateX(0); animation: slideFade 420ms ease both; }
    @keyframes slideFade { from { opacity: 0; transform: translateX(48px); } to { opacity: 1; transform: translateX(0); } }
    h2 { margin: 0; font-size: clamp(30px, 4vw, 52px); line-height: 1.02; letter-spacing: 0; }
    .section-copy { color: #46545f; font-size: 17px; line-height: 1.55; margin: 14px 0 0; max-width: 900px; }
    .toolbar, .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
    button { font: inherit; font-weight: 800; font-size: 12px; border: 1px solid var(--line); background: #fff; color: #41505a; border-radius: 999px; padding: 8px 11px; cursor: pointer; }
    button.active { background: var(--ink); border-color: var(--ink); color: white; }
    button:disabled { cursor: default; opacity: .7; }
    .viz-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-top: 12px; }
    .timeseries-grid { grid-template-columns: minmax(0, 1.15fr) minmax(0, .85fr); }
    .feature-grid { grid-template-columns: 1fr; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: var(--shadow); min-width: 0; }
    .panel h3 { margin: 0 0 4px; font-size: 17px; }
    .sub, .note { color: var(--muted); font-size: 13px; line-height: 1.4; margin: 0 0 8px; }
    .footnote { color: var(--muted); font-size: 11px; line-height: 1.4; margin: 6px 0 0; font-style: italic; }
    .chart { width: 100%; height: 430px; display: block; }
    .chart.tall { height: 520px; }
    .chart.xtall { height: 620px; }
    .chart.compressing { transform-origin: left center; animation: compressLeft 360ms ease both; }
    @keyframes compressLeft { 0% { transform: scaleX(1); } 45% { transform: scaleX(.94); } 100% { transform: scaleX(1); } }
    .axis text { fill: var(--muted); font-size: 11px; }
    .axis path, .axis line, .grid line { stroke: #d7d0c5; }
    .grid path { display: none; }
    .event-line { stroke: #172026; stroke-width: 1.5; stroke-dasharray: 5 5; }
    .event-line { pointer-events: none; }
    .line { fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
    .example-county { pointer-events: stroke; }
    .line.background { opacity: .18; }
    .band { opacity: .18; }
    .band.background { opacity: .05; }
    .band { pointer-events: none; }
    .county { stroke: white; stroke-width: .22; vector-effect: non-scaling-stroke; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 12px; color: var(--muted); font-size: 12px; align-items: center; margin-top: 8px; }
    .scale-legend { width: min(100%, 320px); display: grid; grid-template-columns: 70px 1fr 70px; gap: 8px; align-items: center; }
    .scale-bar { height: 10px; border-radius: 999px; border: 1px solid rgba(23,32,38,.12); }
    .scale-legend span:last-child { text-align: right; }
    .swatch { width: 16px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 5px; vertical-align: -1px; }
    .takeaway { margin-top: 16px; border-left: 5px solid var(--teal); background: #edf7f4; padding: 14px 16px; font-size: 16px; line-height: 1.45; font-weight: 750; }
    .sources { font-size: 12px; color: var(--muted); line-height: 1.5; margin-top: 14px; }
    .sources a { color: #205f90; }
    .tooltip { position: fixed; display: none; max-width: 300px; background: #172026; color: white; padding: 9px 10px; border-radius: 7px; font-size: 12px; line-height: 1.35; pointer-events: none; z-index: 10; }
    .county-line-label { font-size: 10px; fill: var(--muted); pointer-events: none; }
    @media (max-width: 900px) {
      .viz-grid { grid-template-columns: 1fr; }
      .timeseries-grid { grid-template-columns: 1fr; }
      .card-with-sidebar-right { flex-direction: column-reverse; }
      .card-sidebar-right { width: 100%; border-left: none; border-top: 1px solid var(--line); flex-direction: row; flex-wrap: wrap; }
      .hero { min-height: auto; }
      h1 { font-size: clamp(42px, 12vw, 66px); }
      .dek { font-size: 19px; line-height: 1.42; }
      .chart, .chart.tall, .chart.xtall { height: 360px; }
      .feature-detail-header { display: none; }
      .feature-detail-list { grid-template-columns: minmax(140px, 1fr) auto; }
      .feature-detail-row > :nth-child(3), .feature-detail-row > :nth-child(4) { grid-column: 1 / -1; }
      .hazard-rating-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <section class="hero" id="top">
    <div>
      <div class="eyebrow" id="t-hero-eyebrow"></div>
      <h1 id="t-hero-h1"></h1>
      <p class="dek" id="t-hero-dek"></p>
    </div>
  </section>

  <section class="slide" id="pricing">
    <h2 id="t-pricing-h2"></h2>

    <!-- County PPSF YoY histories — full width, footnotes below chart -->
    <div class="panel" style="margin-top:12px;">
      <h3 id="t-scatter-title"></h3>
      <p class="sub" id="t-scatter-sub"></p>
      <svg id="score-scatter" class="chart" style="height:340px;"></svg>
      <p class="footnote" id="t-scatter-fn1"></p>
      <p class="footnote" id="t-scatter-fn2"></p>
    </div>
    <div class="takeaway" id="score-scatter-takeaway"></div>
  </section>

  <section class="slide" id="pricing-grouping">
    <h2 id="t-pricing-grouping-subtitle"></h2>
    <p class="section-copy" id="t-pricing-nri-placeholder" style="margin-top:6px;"></p>

    <!-- PPSF YoY grouped by risk rating + map — side by side, hazard toggle on right -->
    <div class="panel" style="margin-top:16px;">
      <h3 id="t-pricing-card-title"></h3>
      <p class="sub" id="t-pricing-card-text"></p>
      <div class="control-bar" id="rating-hazard-sidebar-right">
        <div class="sidebar-label" id="t-hazard-sidebar-label"></div>
      </div>
      <div class="viz-grid pricing-viz-grid">
        <div>
          <svg id="rating-scatter" class="chart xtall"></svg>
        </div>
        <div class="map-with-legend">
          <svg id="rating-map" class="chart"></svg>
          <div class="legend" id="rating-map-legend"></div>
        </div>
      </div>
    </div>
    <div class="takeaway" id="pricing-takeaway"></div>
    <div class="sources" id="t-pricing-sources"></div>
  </section>

  <section class="slide" id="events">
    <h2 id="t-events-h2"></h2>
    <p class="section-copy" id="t-events-copy"></p>

    <!-- Event window + affected map in one card, risk toggle on right, arrow nav for windows -->
    <div class="panel" style="margin-top:12px;">
      <h3 id="t-events-card-title"></h3>
      <div class="control-bar" id="risk-frame-sidebar-right">
        <div class="sidebar-label" id="t-risk-sidebar-label"></div>
        <div class="toggle-row" id="risk-rating-toggles"></div>
        <button id="risk-play-button" type="button">&#9654; Play</button>
      </div>
      <div class="viz-grid timeseries-grid">
        <div class="event-chart-wrap" style="position:relative;">
          <div id="event-window-frame" class="window-frame">
            <svg id="event-window" class="chart tall"></svg>
          </div>
          <button class="window-arrow window-arrow-left" id="event-arrow-left" type="button">&#8592;</button>
          <button class="window-arrow window-arrow-right" id="event-arrow-right" type="button">&#8594;</button>
        </div>
        <div>
          <svg id="affected-map" class="chart"></svg>
          <div class="county-count" id="affected-county-count"></div>
        </div>
      </div>
      <div class="takeaway" id="event-window-takeaway" style="margin-top:10px; font-size:14px;"></div>
    </div>
    <div class="takeaway" id="event-takeaway"></div>
    <div class="sources" id="t-events-sources"></div>
  </section>

  <section class="slide" id="features">
    <h2 id="t-features-h2"></h2>
    <p class="section-copy" id="t-features-copy"></p>

    <!-- Chart + map side by side, risk group toggle on right -->
    <div class="panel" style="margin-top:12px;">
      <div class="control-bar" id="feature-risk-sidebar">
        <div class="sidebar-label" id="t-feature-sidebar-label"></div>
      </div>
      <div class="viz-grid timeseries-grid">
        <div>
          <svg id="feature-event-window" class="chart tall"></svg>
        </div>
        <div>
          <svg id="feature-county-map" class="chart"></svg>
        </div>
      </div>
      <div id="county-features-card" style="margin-top:16px; display:none; border-top:1px solid var(--line); padding-top:14px;">
        <h3 id="county-features-title"></h3>
        <p class="sub" id="t-feature-corr-label" style="font-weight:700; margin-bottom:6px;"></p>
        <div id="within-group-correlations"></div>
      </div>
    </div>

    <div class="takeaway" id="feature-takeaway"></div>
    <div class="sources" id="t-features-sources"></div>
  </section>

  <section class="slide" id="playbook">
    <h2 id="t-playbook-h2"></h2>

    <!-- County search, fixed map viewport, hazard profile, and housing history -->
    <div class="panel" style="margin-top:12px;">
      <div style="margin-bottom:10px;">
        <input type="text" id="county-search" style="padding:9px 14px; border:1px solid var(--line); border-radius:999px; font-size:13px; width:min(400px,100%); background:#fff;">
      </div>
      <div id="county-results" style="max-height:200px; overflow-y:auto; margin-bottom:10px; border:1px solid var(--line); border-radius:6px; display:none;"></div>
      <div class="playbook-map-wrap">
        <svg id="county-selection-map" class="chart"></svg>
        <div class="playbook-map-controls">
          <button id="playbook-map-zoom-in" type="button" title="Zoom in" aria-label="Zoom in">+</button>
          <button id="playbook-map-zoom-minus" type="button" title="Zoom out" aria-label="Zoom out">&#8722;</button>
          <button id="playbook-map-zoom-toggle" type="button" style="display:none;"></button>
        </div>
      </div>

      <div id="playbook-display" style="display:none; margin-top:20px; border-top:1px solid var(--line); padding-top:16px;">
        <div class="hazard-rating-grid" id="playbook-hazard-ratings"></div>
        <h3 id="t-playbook-history-title" style="margin-top:18px;"></h3>
        <svg id="playbook-ppsf-history" class="chart tall"></svg>
        <div class="playbook-commentary" id="playbook-event-commentary"></div>
      </div>
    </div>

    <div class="sources" id="t-playbook-sources"></div>
  </section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
/* ============================================================
   TEXT — Every visible string on the page lives here.
   Edit any value below and rebuild to update the page.
   ============================================================ */
const TEXT = {
  // ---- Hero ----
  heroEyebrow: "Which Way the Wind Blows",
  heroH1: "Are Climate Risks Priced Into Housing Markets?",
  heroDek: "Climate change results in more severe weather events and natural disasters that cause substantial damage to properties and in extreme cases, devastate local communities. What does all this mean to you as a homeowner?",

  // ---- Pricing section ----
  pricingH2: "To Begin: What Does Growth in Housing Markets Look Like Across the United States?",
  scatterTitle: "Median Price-Per-Square-Foot (PPSF) Year-Over-Year (YoY) by County",
  scatterSub: "Each line represents a county's Median PPSF YoY over the last 10 years.",
  scatterFootnote1: "* Outliers beyond the 1st–99th percentile are excluded.",
  scatterFootnote2: "* County-year observations with fewer than 3 reported months are excluded.",
  pricingScoreScatterTakeaway: "From county-level median house price growth over the last 10 years, there is significant variation and there doesn't seem to be a clear pattern.<br>However, the impact of climate change is uneven across the country, so looking from a climate angle might reveal a more meaningful pattern.",
  pricingGroupingSubtitle: "A Climate Perspective: What Does House Price Growth Look Like When Grouping Counties by Climate Risk?",
  pricingNriPlaceholder: "The FEMA National Risk Index (NRI) serves as a measure of climate-related risk exposure. It summarizes a county's expected annual loss, social vulnerability, and community resilience across natural hazards. Counties are assigned a risk rating along a scale from \"Very Low\" to \"Very High\".",
  pricingCardTitle: "Median PPSF YoY by Climate Risk",
  pricingCardText: "House price growth of counties grouped by their FEMA National Risk Index (NRI) risk rating. Risk ratings are available for specific hazards and for overall climate risk.",
  hazardSidebarLabel: "Hazard type",
  pricingTakeaway: "A pattern now emerges: Counties with higher risk tend to show lower levels of house price growth.<br>Housing markets are influenced by events across time. With respect to climate events, is there a shift in house price growth?",
  pricingSources: 'Sources: <a href="https://hazards.fema.gov/nri/" target="_blank" rel="noopener">FEMA National Risk Index</a>, local mart <code>data/quoll.duckdb: mart.nri_county_risk</code>; <a href="https://www.redfin.com/news/data-center/" target="_blank" rel="noopener">Redfin Data Center</a>, local mart <code>mart.redfin_county_monthly</code>. House prices use the average monthly <code>MEDIAN_PPSF_YOY</code> value in 2025.',

  // ---- Events section ----
  eventsH2: "Observing Climate Events: What Happens to House Price Growth?",
  eventsCopy: "Severe weather incidents and natural disasters can change the housing market outlook in its area as buyers and sellers re-evaluate their positions based on the perceived increased risk.",
  eventsCardTitle: "Median PPSF YoY Around Extreme Climate Events",
  riskSidebarLabel: "Risk rating",
  eventWindowATakeaway: "Past the 2-year mark post-event, house price growth momentum diverges across different risk groups. Growth weakening is more pronounced in higher risk groups.",
  eventWindowBTakeaway: "Around the 4-year mark post-event, house price growth across different risk groups begin to converge to the same level. It appears that the event’s impact fades out from view eventually.",
  eventsTakeaway: "In higher-risk counties, there appears to be some time lag after an event before house price growth declines significantly. Homeowners who made it through the period of weakness then experienced some subsequent recovery.<br>The wide band within each risk group shows that counties are hardly uniform, even within the same risk category. Why does this variation exist?",
  eventsSources: "Sources: local marts <code>mart.fema_disaster_declarations</code>, <code>mart.noaa_storm_events</code>, <code>mart.redfin_county_monthly</code>, and <code>mart.nri_county_risk</code>.",

  // ---- Features section ----
  featuresH2: "What Sets Apart Counties Within the Same Risk Group?",
  featuresCopy: "A county's features can make its housing market more vulnerable or resilient to destructive weather events, and also influence its housing market.",
  featureSidebarLabel: "Risk Rating",
  featureCorrLabel: "County features most correlated with relative position within its risk group",
  featuresTakeaway: "Since counties at different price growth levels within the same risk group have distinctly different features, a county's features are correlated with its housing market performance.<br>Given the significance of climate risk to housing markets, we can paint a picture of a county's climate risk that will be invaluable to homeowners.",
  featuresSources: "Sources: local marts <code>mart.acs_county_economic_annual</code>, <code>mart.acs_county_demographic_annual</code>, <code>mart.acs_county_affordability_annual</code>, <code>mart.ncei_county_weather_monthly</code>, and <code>mart.nri_county_risk</code>. Cost components are midpoint estimates from ACS cost buckets.",

  // ---- Playbook section ----
  playbookH2: "Climate Playbook: What to Know About Your County's Climate Exposure",
  playbookSearchPlaceholder: "Search for a county by name, state, or FIPS…",
  playbookHistoryTitle: "Monthly Median PPSF YoY Over the Past 10 Years",
  playbookZoomOut: "Zoom out",
  playbookZoomIn: "Zoom to county",
  playbookEventLegend: "Extreme event period",
  playbookMissingDataLegend: "Missing county data",
  playbookTakeaways: {
    noEvents: "<strong>No qualifying extreme climate events were recorded for this county during the displayed period.</strong><br>{expectation}",
    eventSummary: "<strong>{eventCount} qualifying extreme {eventNoun} shown.</strong> {observation}{alignment}<ul style=\"margin:8px 0 0;padding-left:20px;max-height:170px;overflow:auto;\">{details}</ul>",
    insufficientHistory: "There is not enough pre- and post-event housing history to measure a change.",
    observedChange: "Across {measuredCount} measurable {eventNoun}, the median 12-month PPSF YoY change {direction} by {magnitude} percentage points.",
    expectationComparison: " This is {alignment} with the group-level expectation for {risk} NRI risk. {expectation}",
    eventDetail: "<li>{name} ({start} to {end}): {result}</li>",
    unavailableExpectation: "A group-level post-event expectation is unavailable for this NRI rating.",
  },
  playbookTakeawayTerms: {
    event: "event",
    events: "events",
    declined: "declined",
    increased: "increased",
    unchanged: "was broadly unchanged",
    aligned: "broadly in line",
    notAligned: "not clearly in line",
    down: "down",
    up: "up",
    littleChanged: "little changed",
    insufficient: "insufficient pre/post data",
  },
  playbookSources: "Sources: FEMA National Risk Index and local mart <code>mart.nri_county_risk</code>; Redfin Data Center and local mart <code>mart.redfin_county_monthly</code>; local marts <code>mart.fema_disaster_declarations</code> and <code>mart.noaa_storm_events</code>.",
  riskImpacts: {
    "Very Low": "Counties with Very Low climate risk tend to maintain steady house price growth around climate events, with minimal disruption to market momentum.",
    "Low": "Counties with Low climate risk typically see modest softening of house price growth about two years after the event, but generally recover within three years.",
    "Medium": "Counties with Medium climate risk experience noticeable softening of house price growth around the two-year mark after the event.",
    "High": "Counties with High climate risk see significant deceleration in house price growth following an event, with the decline primarily occurring 18-24 months after event end.",
    "Very High": "Counties with Very High climate risk face substantial impacts on house price growth, with softening trends that can persist for several years after events.",
  },
};

/* Populate all text elements from the TEXT object above.
   Each entry maps a TEXT key to an element id. Uses innerHTML
   so sources with <a>/<code> tags render correctly. */
function hydrateText() {
  const map = {
    heroEyebrow: "t-hero-eyebrow",
    heroH1: "t-hero-h1",
    heroDek: "t-hero-dek",
    pricingH2: "t-pricing-h2",
    scatterTitle: "t-scatter-title",
    scatterSub: "t-scatter-sub",
    scatterFootnote1: "t-scatter-fn1",
    scatterFootnote2: "t-scatter-fn2",
    pricingScoreScatterTakeaway: "score-scatter-takeaway",
    pricingTakeaway: "pricing-takeaway",
    pricingGroupingSubtitle: "t-pricing-grouping-subtitle",
    pricingNriPlaceholder: "t-pricing-nri-placeholder",
    pricingCardTitle: "t-pricing-card-title",
    pricingCardText: "t-pricing-card-text",
    hazardSidebarLabel: "t-hazard-sidebar-label",
    pricingSources: "t-pricing-sources",
    eventsH2: "t-events-h2",
    eventsCopy: "t-events-copy",
    eventsCardTitle: "t-events-card-title",
    riskSidebarLabel: "t-risk-sidebar-label",
    eventsTakeaway: "event-takeaway",
    eventsSources: "t-events-sources",
    featuresH2: "t-features-h2",
    featuresCopy: "t-features-copy",
    featureSidebarLabel: "t-feature-sidebar-label",
    featureCorrLabel: "t-feature-corr-label",
    featuresTakeaway: "feature-takeaway",
    featuresSources: "t-features-sources",
    playbookH2: "t-playbook-h2",
    playbookHistoryTitle: "t-playbook-history-title",
    playbookSources: "t-playbook-sources",
  };
  for (const [key, id] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (el && TEXT[key] != null) el.innerHTML = TEXT[key];
  }
  document.getElementById("county-search").placeholder = TEXT.playbookSearchPlaceholder;
}

const HAZARD_ICONS = {
  overall: "\u{1F30E}",
  river_flood: "\u{1F30A}",
  tornado: "\u{1F32A}",
  wildfire: "\u{1F525}",
  hail: "\u{1F9CA}",
  earthquake: "\u{1F4A5}",
};

const DATA = __PAYLOAD__;
const RISK_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"];
const RISK_COLORS = {"Very Low":"#16803c","Low":"#79b851","Medium":"#e0b33b","High":"#df7d2f","Very High":"#b42318"};
const COUNTY_LINE_COLOR = "#5b7a8a";
const fmtPct = d3.format("+.1%");
const fmtShare = d3.format(".0%");
const fmtAxisPct = value => Math.abs(value) >= 10 ? `${value > 0 ? "+" : ""}${d3.format(".2s")(value * 100)}%` : fmtPct(value);
const fmtNum = d3.format(",.1f");
const fmtMoney = d3.format("$,.0f");
const tooltip = d3.select("#tooltip");
const countyByFips = new Map(DATA.priceRisk.counties.map(d => [d.fips, d]));
const playbookCountyByFips = new Map((DATA.playbook?.counties || []).map(d => [d.fips, d]));
let ratingHazard = "overall";
// Event section state
let selectedRisk = "Very Low";
let riskTimer = null;
let riskAutoPaused = false;
let activeEventWindow = "A"; // "A" or "B"
// Features section state
let selectedFeatureRisk = "Medium";
let selectedFeatureCounty = null;
// Playbook state
let selectedCountyFips = null;
let playbookMapZoomed = false;
let playbookZoomBehavior = null;
let playbookMapTransform = d3.zoomIdentity;
let playbookSelectedTransform = d3.zoomIdentity;

function hazardLabel(key) { return DATA.priceRisk.hazards.find(h => h.key === key)?.label || key; }
function hazardCounty(county, key) { return county?.hazards?.[key] || {}; }
function capValue(value, domain) {
  if (value == null || !Number.isFinite(value)) return null;
  return Math.max(domain[0], Math.min(domain[1], value));
}
function robustDomain(values) {
  const valid = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  if (!valid.length) return [0, 1];
  const lo = d3.quantileSorted(valid, .01);
  const hi = d3.quantileSorted(valid, .99);
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}
function outlierFreeValues(values) {
  const sorted = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  if (!sorted.length) return sorted;
  const lo = d3.quantileSorted(sorted, .01);
  const hi = d3.quantileSorted(sorted, .99);
  return sorted.filter(v => v >= lo && v <= hi);
}
function colorScale(values, color, robust = false) {
  const valid = values.filter(v => v != null).sort(d3.ascending);
  const domain = robust ? robustDomain(valid) : d3.extent(valid);
  return d3.scaleSequentialSymlog(domain[0] === domain[1] ? [domain[0] - 1, domain[1] + 1] : domain, color).constant(.05);
}
function scaleLegendHtml(domain, colorA, colorB, label, formatter = fmtNum) {
  return `<div class="scale-legend"><span>${formatter(domain[0])}</span><div class="scale-bar" style="background:linear-gradient(90deg,${colorA},${colorB})"></div><span>${formatter(domain[1])}</span></div><span>${label}</span>`;
}
function pctText(value) { return value == null ? "n/a" : fmtPct(value); }
function ratingValue(rating) { return RISK_ORDER.indexOf(rating) + 1; }
function sparsePctTicks(domain) {
  const [lo, hi] = domain;
  return [lo, 0, 0.1, 1, hi]
    .filter(v => v != null && Number.isFinite(v) && v >= lo && v <= hi)
    .filter((v, i, arr) => arr.findIndex(x => Math.abs(x - v) < 1e-9) === i);
}

function drawMap(svgId, fillFn, tooltipFn, legendId, legendHtml, clickFn) {
  const svg = d3.select(svgId);
  const width = svg.node().clientWidth || 520;
  const height = svg.node().clientHeight || 430;
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);
  svg.on("mouseleave", () => tooltip.style("display","none"));
  svg.append("g").selectAll("path")
    .data(DATA.geojson.features)
    .join("path")
    .attr("class","county")
    .attr("d", path)
    .attr("fill", d => fillFn(countyByFips.get(d.properties.fips), d.properties.fips))
    .style("cursor", clickFn ? "pointer" : null)
    .on("mousemove", (event, d) => {
      const html = tooltipFn(countyByFips.get(d.properties.fips), d.properties.fips);
      if (!html) return;
      tooltip.style("display","block").style("left", `${event.clientX+12}px`).style("top", `${event.clientY+12}px`).html(html);
    })
    .on("click", clickFn ? (event, d) => clickFn(d.properties.fips) : null);
  if (legendId) d3.select(legendId).html(legendHtml || "");
}

function drawScoreScatter() {
  let data = DATA.priceRisk.countyHistory.filter(d => d.ppsfYoy != null && d.riskRating != null);
  const allVals = data.map(d => d.ppsfYoy);
  const sorted = [...allVals].filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  const p1 = d3.quantileSorted(sorted, .01);
  const p99 = d3.quantileSorted(sorted, .99);
  data = data.filter(d => d.ppsfYoy >= p1 && d.ppsfYoy <= p99);
  const svg = d3.select("#score-scatter");
  const width = svg.node().clientWidth || 960, height = svg.node().clientHeight || 340;
  const margin = {top: 18, right: 18, bottom: 46, left: 68};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const yDomain = [p1, p99];
  const x = d3.scaleLinear().domain(d3.extent(data, d => d.year)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
  const yTicks = sparsePctTicks(y.domain());
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).tickFormat(d3.format("d")).ticks(6));
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickFormat(fmtAxisPct));
  svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Year");
  svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",20).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Annual average Median PPSF YoY");
  const line = d3.line().x(d => x(d.year)).y(d => y(Math.max(p1, Math.min(p99, d.ppsfYoy))));
  svg.append("g").selectAll("path").data(d3.groups(data, d => d.fips)).join("path")
    .attr("class","line")
    .attr("stroke", "#5b7a8a")
    .attr("stroke-width", 1.1)
    .attr("opacity", .12)
    .attr("d", d => line(d[1].sort((a,b)=>d3.ascending(a.year,b.year))))
    .on("mousemove", (event, d) => tooltip.style("display","block").style("left",`${event.clientX+12}px`).style("top",`${event.clientY+12}px`).html(`<strong>${d[1][0].county}</strong>${d[1][0].riskRating} risk<br>Annual PPSF YoY history`))
    .on("mouseleave", () => tooltip.style("display","none"));
}

function drawRatingScatter() {
  const ratingHistory = DATA.priceRisk.ratingHistoriesByHazard[ratingHazard] || [];
  const data = ratingHistory.filter(d => d.median != null);
  const svg = d3.select("#rating-scatter");
  const width = svg.node().clientWidth || 520, height = svg.node().clientHeight || 620;
  const margin = {top: 18, right: 100, bottom: 54, left: 68};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const yDomain = robustDomain(data.flatMap(d => [d.q1, d.median, d.q3]));
  const x = d3.scaleLinear().domain(d3.extent(data, d => d.year)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
  const yTicks = sparsePctTicks(y.domain());
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).tickFormat(d3.format("d")).ticks(6));
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickFormat(fmtAxisPct));
  svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",20).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Annual average Median PPSF YoY");
  const area = d3.area().x(d => x(d.year)).y0(d => y(capValue(d.q1, yDomain))).y1(d => y(capValue(d.q3, yDomain)));
  const line = d3.line().x(d => x(d.year)).y(d => y(capValue(d.median, yDomain)));
  for (const risk of RISK_ORDER) {
    const rows = data.filter(d => d.riskRating === risk).sort((a,b)=>d3.ascending(a.year,b.year));
    if (!rows.length) continue;
    svg.append("path").datum(rows).attr("class","band").attr("fill",RISK_COLORS[risk]).attr("d",area);
    svg.append("path").datum(rows).attr("class","line").attr("stroke",RISK_COLORS[risk]).attr("d",line);
    const last = rows.at(-1);
    const labelX = x(last.year) + 8;
    const labelY = y(capValue(last.median, yDomain)) + 4;
    svg.append("rect").attr("x", labelX - 3).attr("y", labelY - 11).attr("width", risk.length * 6 + 6).attr("height", 16).attr("fill", "white").attr("fill-opacity", 0.85).attr("rx", 3);
    svg.append("text").attr("x", labelX).attr("y", labelY).attr("fill", RISK_COLORS[risk]).attr("font-size", 13).attr("font-weight", 800).text(risk);
  }
}

function drawRatingMap() {
  drawMap("#rating-map",
    (county, fips) => {
      if (!county) return "#ece7df";
      return RISK_COLORS[hazardCounty(county, ratingHazard).rating] || "#ece7df";
    },
    (county, fips) => county ? `<strong>${county.county}</strong>${hazardLabel(ratingHazard)} rating: ${hazardCounty(county, ratingHazard).rating ?? "n/a"}` : "",
    "#rating-map-legend",
    RISK_ORDER.map(r => `<span><span class="swatch" style="background:${RISK_COLORS[r]}"></span>${r}</span>`).join("")
  );
}

function drawLineChart(svgId, source, groupKey, horizonLimit, activeRisk = null, minMonth = -12, opts = {}) {
  const data = source.filter(d => d.month >= minMonth && d.month <= horizonLimit);
  const svg = d3.select(svgId);
  const width = svg.node().clientWidth || 700, height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: 96, bottom: 42, left: 58};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const x = d3.scaleLinear().domain([minMonth, horizonLimit]).range([margin.left, width - margin.right]);
  const hideOthers = opts.hideOtherGroups;
  const domainData = hideOthers && activeRisk ? data.filter(d => d[groupKey] === activeRisk) : data;
  const values = domainData.flatMap(d => [d.q1, d.median, d.q3]).filter(v => v != null);
  if (opts.extraDomainValues) values.push(...opts.extraDomainValues);
  const y = d3.scaleLinear().domain(d3.extent(values)).nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  const useYears = horizonLimit > 12;
  const yearTicks = [minMonth, 0, ...d3.range(12, horizonLimit + 1, 12)];
  const axis = useYears
    ? d3.axisBottom(x).tickValues(yearTicks).tickFormat(d => d < 0 ? `${Math.abs(d / 12)}y pre` : d === 0 ? "event" : `${d / 12}y`)
    : d3.axisBottom(x).ticks(8);
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(axis);
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  svg.append("line").attr("class","event-line").attr("x1",x(0)).attr("x2",x(0)).attr("y1",margin.top).attr("y2",height-margin.bottom);
  const grouped = d3.group(data, d => groupKey ? d[groupKey] : "All affected counties");
  for (const [key, rows] of grouped) {
    rows.sort((a,b)=>d3.ascending(a.month,b.month));
    if (hideOthers && key !== activeRisk) continue;
    const isBackground = !hideOthers && activeRisk && RISK_ORDER.indexOf(key) < RISK_ORDER.indexOf(activeRisk);
    const isHidden = !hideOthers && activeRisk && RISK_ORDER.indexOf(key) > RISK_ORDER.indexOf(activeRisk);
    if (isHidden) continue;
    const color = groupKey === "riskRating" ? RISK_COLORS[key] : "#0f766e";
    const areaFn = d3.area().x(d=>x(d.month)).y0(d=>y(d.q1)).y1(d=>y(d.q3));
    const lineFn = d3.line().x(d=>x(d.month)).y(d=>y(d.median));
    svg.append("path").datum(rows).attr("class",`band ${isBackground ? "background" : ""}`).attr("fill",color).attr("d",areaFn);
    svg.append("path").datum(rows).attr("class",`line ${isBackground ? "background" : ""}`).attr("stroke",color).attr("d",lineFn);
    const last = rows.at(-1);
    if (last && !isBackground) svg.append("text").attr("x",x(last.month)+5).attr("y",y(last.median)+4).attr("fill",color).attr("font-size",12).attr("font-weight",800).text(key);
  }
  svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text(useYears ? "Years from event start / after event end" : "Months from event start / after event end");
  return {x, y, margin, width, height};
}

function initButtons() {
  // Hazard sidebar (right) for pricing section — vertical, uniform size, with icons
  const hazardContainer = d3.select("#rating-hazard-sidebar-right");
  hazardContainer.selectAll("button")
    .data(DATA.priceRisk.hazards).join("button")
    .html(d => `<span style="font-size:14px;">${HAZARD_ICONS[d.key]||""}</span> ${d.label}`)
    .classed("active", d => d.key === ratingHazard)
    .style("background", d => d.key === ratingHazard ? "var(--ink)" : null)
    .on("click", (event, d) => {
      ratingHazard = d.key;
      hazardContainer.selectAll("button").classed("active", x => x.key === ratingHazard)
        .style("background", x => x.key === ratingHazard ? "var(--ink)" : null);
      drawRatingScatter(); drawRatingMap();
    });

  // Event section: arrow nav for windows
  d3.select("#event-arrow-left").on("click", () => switchEventWindow("left"));
  d3.select("#event-arrow-right").on("click", () => switchEventWindow("right"));

  // Event section: risk rating sidebar (right, color-coded)
  const riskSidebar = d3.select("#risk-rating-toggles");
  riskSidebar.selectAll("button.risk-toggle")
    .data(RISK_ORDER).join("button")
    .attr("class", "risk-toggle")
    .text(d=>d)
    .style("border-left", d=>`4px solid ${RISK_COLORS[d]}`)
    .classed("active", d=>d===selectedRisk)
    .style("background", d=>d===selectedRisk ? RISK_COLORS[d] : null)
    .on("click",(event,d)=>{
      selectedRisk=d; clearInterval(riskTimer); riskAutoPaused=true;
      riskSidebar.selectAll("button.risk-toggle").classed("active",x=>x===selectedRisk)
        .style("background", x=>x===selectedRisk ? RISK_COLORS[x] : null);
      d3.select("#risk-play-button").classed("visible", true);
      renderEventSection();
    });
  d3.select("#risk-play-button").on("click", startRiskTimer);

  // Features risk group sidebar (right, color-coded)
  const featureSidebar = d3.select("#feature-risk-sidebar");
  featureSidebar.selectAll("button")
    .data(RISK_ORDER).join("button")
    .text(d=>d)
    .style("border-left", d=>`4px solid ${RISK_COLORS[d]}`)
    .classed("active", d=>d===selectedFeatureRisk)
    .style("background", d=>d===selectedFeatureRisk ? RISK_COLORS[d] : null)
    .on("click",(event,d)=>{
      selectedFeatureRisk=d; selectedFeatureCounty=null;
      featureSidebar.selectAll("button").classed("active",x=>x===selectedFeatureRisk)
        .style("background", x=>x===selectedFeatureRisk ? RISK_COLORS[x] : null);
      drawFeatureHeatmaps();
    });
}

function switchEventWindow(direction) {
  const newWindow = direction === "right" ? "B" : "A";
  if (newWindow === activeEventWindow) return;
  const frame = d3.select("#event-window-frame");
  const slideOut = direction === "right" ? "sliding-left" : "sliding-right";
  const slideIn = direction === "right" ? "sliding-in-right" : "sliding-in-left";
  frame.classed(slideOut, true);
  setTimeout(() => {
    activeEventWindow = newWindow;
    renderEventSection();
    frame.classed(slideOut, false).classed(slideIn, true);
    setTimeout(() => frame.classed(slideIn, false), 400);
  }, 200);
}

function median(values) {
  const valid = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  return valid.length ? d3.median(valid) : null;
}

function relativeBinLabel(bin) {
  if (!bin || bin.bucketOrder == null || !bin.bucketCount) return "No bin";
  const labels = ["Very low", "Low", "Medium", "High", "Very high"];
  if (bin.bucketCount === 1) return "Medium";
  const index = Math.round((bin.bucketOrder / (bin.bucketCount - 1)) * (labels.length - 1));
  return labels[Math.max(0, Math.min(labels.length - 1, index))];
}

function countyDisplayName(county) {
  if (!county) return "";
  const name = county.county || "";
  const suffix = county.state ? `, ${county.state}` : "";
  return suffix && name.endsWith(suffix) ? name : `${name}${suffix}`;
}

function riskAssociationText(association) {
  if (association === "higher") return `<span style="color:#b42318;font-size:11px;">Associated with higher NRI risk</span>`;
  if (association === "lower") return `<span style="color:#16803c;font-size:11px;">Associated with lower NRI risk</span>`;
  return `<span style="color:#66717b;font-size:11px;">No clear NRI risk association</span>`;
}

function featureScaleHtml(contribution) {
  const score = contribution == null || !Number.isFinite(+contribution)
    ? null
    : Math.max(-1, Math.min(1, +contribution));
  const p = score == null ? null : 50 + score * 50;
  const marker = p == null ? "" : `<span class="feature-scale-arrow" style="left:${p}%;"></span>`;
  return `<div class="feature-scale"><span>Lower PPSF YoY</span><span class="feature-scale-bar">${marker}</span><span style="text-align:right;">Higher PPSF YoY</span></div>`;
}

function featurePercentileScaleHtml(percentile) {
  const p = percentile == null || !Number.isFinite(+percentile)
    ? null
    : Math.max(0, Math.min(100, +percentile));
  const marker = p == null ? "" : `<span class="feature-scale-arrow" style="left:${p}%;"></span>`;
  return `<div class="feature-scale"><span>Low value</span><span class="feature-scale-bar percentile">${marker}</span><span style="text-align:right;">High value</span></div>`;
}

// ---- render event section with two windows ----
function activeWindowData() {
  return activeEventWindow === "A" ? DATA.eventWindows.windowA : DATA.eventWindows.windowB;
}

function renderEventSection() {
  const wd = activeWindowData();
  d3.select("#risk-rating-toggles").selectAll("button.risk-toggle").classed("active", d => d === selectedRisk)
    .style("background", d => d === selectedRisk ? RISK_COLORS[d] : null);
  d3.select("#risk-play-button").classed("visible", riskAutoPaused);
  const ta = activeEventWindow === "A" ? TEXT.eventWindowATakeaway : TEXT.eventWindowBTakeaway;
  d3.select("#event-window-takeaway").text(ta);
  drawLineChart("#event-window", wd.byRating, "riskRating",
    activeEventWindow === "A" ? 36 : 60,
    selectedRisk,
    -12
  );
  drawAffectedMap(wd);
}

function drawAffectedMap(wd) {
  const affected = new Map((wd.affectedCounties || []).map(d => [d.fips, d.riskRating]));
  const selectedCount = new Set((wd.affectedCounties || []).filter(d => d.riskRating === selectedRisk).map(d => d.fips)).size;
  d3.select("#affected-county-count").text(`${d3.format(",d")(selectedCount)} counties`);
  drawMap("#affected-map",
    (county, fips) => affected.get(fips) === selectedRisk ? RISK_COLORS[selectedRisk] : "#e6dfd5",
    (county, fips) => affected.get(fips) ? `<strong>${county?.county || fips}</strong>Affected county<br>NRI rating: ${affected.get(fips)}` : "",
    null, null
  );
}

function startRiskTimer() {
  clearInterval(riskTimer);
  riskAutoPaused = false;
  d3.select("#risk-play-button").classed("visible", false);
  riskTimer = setInterval(() => {
    selectedRisk = RISK_ORDER[(RISK_ORDER.indexOf(selectedRisk) + 1) % RISK_ORDER.length];
    d3.select("#risk-rating-toggles").selectAll("button.risk-toggle").classed("active", d => d === selectedRisk)
      .style("background", d => d === selectedRisk ? RISK_COLORS[d] : null);
    renderEventSection();
  }, 3200);
}

// ---- features section ----
function drawFeatureHeatmaps() {
  d3.select("#feature-risk-sidebar").selectAll("button").classed("active", d => d === selectedFeatureRisk)
    .style("background", d => d === selectedFeatureRisk ? RISK_COLORS[d] : null);
  const wd = DATA.eventWindows.windowA;
  const examples = (wd.exampleCountyLines || []).filter(d => d.riskRating === selectedFeatureRisk);
  // Collect county line values so the y-domain encompasses them
  const countyVals = examples.flatMap(d => d.values.filter(v => v.month >= -12 && v.month <= 36 && v.value != null).map(v => v.value));
  // Only show the selected risk group (hideOtherGroups=true)
  const {x, y, margin, width, height} = drawLineChart("#feature-event-window", wd.byRating, "riskRating", 36, selectedFeatureRisk, -12, {hideOtherGroups: true, extraDomainValues: countyVals});

  // Draw stratified sample county lines (different color from median, with labels)
  // Reuse the same y scale from drawLineChart so county lines align with the IQR band
  const svg = d3.select("#feature-event-window");
  const lineFn = d3.line().defined(d => d.value != null).x(d => x(d.month)).y(d => y(d.value));

  const g = svg.append("g");
  g.selectAll("path.example-county").data(examples).join("path")
    .attr("class","line example-county")
    .attr("stroke", COUNTY_LINE_COLOR)
    .attr("stroke-width", d => (selectedFeatureCounty && d.lineId === selectedFeatureCounty.lineId) ? 2.5 : 1)
    .attr("opacity", d => (selectedFeatureCounty && d.lineId === selectedFeatureCounty.lineId) ? 0.85 : 0.25)
    .attr("d", d => lineFn(d.values.filter(v => v.month >= -12 && v.month <= 36)))
    .style("cursor","pointer")
    .on("click",(event,d)=>{ selectedFeatureCounty=d; drawFeatureHeatmaps(); })
    .on("mousemove",(event,d)=>tooltip.style("display","block").style("left",`${event.clientX+12}px`).style("top",`${event.clientY+12}px`).html(`<strong>${d.county}, ${d.state}</strong>${d.riskRating} risk · ${Math.round(d.pctRank)}th percentile<br>Click to inspect features`))
    .on("mouseleave",()=>tooltip.style("display","none"));

  // Labels to the right of each county line
  examples.forEach(d => {
    const lastPt = d.values.filter(v => v.month >= -12 && v.month <= 36 && v.value != null).at(-1);
    if (!lastPt) return;
    const isSelected = selectedFeatureCounty && d.lineId === selectedFeatureCounty.lineId;
    svg.append("text")
      .attr("class", "county-line-label")
      .attr("x", x(lastPt.month) + 4)
      .attr("y", y(lastPt.value) + 3)
      .attr("opacity", isSelected ? 1 : 0.55)
      .attr("font-weight", isSelected ? 700 : 400)
      .text(`${d.county} (${d.samplePosition === "Above group median" ? "above" : "below"})`);
  });

  // Highlight selected county on map
  drawMap("#feature-county-map",
    (county, fips) => selectedFeatureCounty && fips === selectedFeatureCounty.fips ? (RISK_COLORS[selectedFeatureRisk] || "#0f766e") : "#d8d0c4",
    (county, fips) => county ? `<strong>${county.county}</strong>${selectedFeatureCounty && fips === selectedFeatureCounty.fips ? "<br>Selected county" : ""}` : "",
    null, null
  );
  drawCountyFeaturePanelV2();
}

function drawCountyFeaturePanelV2() {
  if (!selectedFeatureCounty) {
    d3.select("#county-features-card").style("display","none");
    return;
  }
  d3.select("#county-features-card").style("display","block");
  d3.select("#county-features-title").text(`${countyDisplayName(selectedFeatureCounty)} - ${selectedFeatureCounty.riskRating} Risk`);

  const featureMeta = DATA.features.featureMeta || {};
  const wgBins = ((DATA.features.withinGroupFeatureBins || {})[selectedFeatureCounty.riskRating] || {})[selectedFeatureCounty.fips] || {};
  const wgTop = ((DATA.features.withinGroupTopFeatures || {})[selectedFeatureCounty.riskRating] || [])
    .slice(0, 10);
  if (wgTop.length) {
    const html2 = `<div class="feature-detail-list">
      <div class="feature-detail-header">
        <div>Feature</div><div>County value</div><div>Percentile within risk group</div><div>Contribution to Median PPSF YoY</div>
      </div>
      ${wgTop.map(c => {
        const bin = wgBins[c.feature];
        const fmt = featureMeta[c.feature]?.format || "number";
        return `<div class="feature-detail-row">
          <div style="font-weight:600;">${c.feature}</div>
          <div style="text-align:right;">${formatFeatureVal(bin?.value, fmt)}</div>
          <div>${featurePercentileScaleHtml(bin?.valuePercentile)}</div>
          <div>${featureScaleHtml(bin?.contribution)}</div>
        </div>`;
      }).join("")}
    </div>`;
    d3.select("#within-group-correlations").html(html2);
  } else {
    d3.select("#within-group-correlations").text("Within-group bin data not available.");
  }
}

function drawCountyFeaturePanel() {
  if (!selectedFeatureCounty) {
    d3.select("#county-features-card").style("display","none");
    return;
  }
  d3.select("#county-features-card").style("display","block");
  d3.select("#county-features-title").text(`${selectedFeatureCounty.county}, ${selectedFeatureCounty.state} — ${selectedFeatureCounty.riskRating} Risk`);

  const profile = DATA.features.countyProfiles.find(d => d.fips === selectedFeatureCounty.fips);
  const correlations = DATA.features.correlations || [];
  const corrByFeature = Object.fromEntries(correlations.map(c => [c.feature, c.corr]));

  if (profile) {
    const rows = [
      ["Income", profile.income, "currency"],
      ["Home ownership burden", profile.housingBurden, "percent"],
      ["Home insurance", profile.insurance, "currency"],
      ["Property tax", profile.propertyTaxes, "currency"],
      ["Utilities", profile.utilities, "currency"],
      ["Net migration", profile.netMigration, "number"],
      ["Homes Sold YoY", profile.homesSoldYoy, "pct"],
      ["Median DOM YoY", profile.medianDomYoy, "number"],
      ["Unemployment rate", profile.unemploymentRate, "percent"],
      ["New Listings YoY", profile.newListingsYoy, "pct"],
    ];
    const html = `<div style="display:grid;grid-template-columns:1fr auto auto;gap:6px;font-size:13px;line-height:1.7;">
      ${rows.map(([k,v,fmt]) => {
        const corr = corrByFeature[k] || 0;
        const arrow = corr > 0.05 ? `<span style="color:#b42318;">↑</span>` : corr < -0.05 ? `<span style="color:#16803c;">↓</span>` : `<span style="color:#66717b;">→</span>`;
        const label = corr > 0.05 ? `<span style="color:#b42318;font-size:11px;">Associated with higher NRI risk</span>` : corr < -0.05 ? `<span style="color:#16803c;font-size:11px;">Associated with lower NRI risk</span>` : `<span style="color:#66717b;font-size:11px;">No clear NRI risk association</span>`;
        const disp = formatFeatureVal(v, fmt);
        return `<div style="font-weight:600;">${k}</div><div style="text-align:right;">${disp}</div><div style="text-align:center;">${arrow} ${label}</div>`;
      }).join("")}
    </div>`;
    d3.select("#selected-county-features").html(html);
  } else {
    d3.select("#selected-county-features").text("Feature profile not available for this county.");
  }

  // Within-group correlations
  const wgCorrs = (DATA.features.withinGroupCorrelations || {})[selectedFeatureCounty.riskRating] || [];
  const sortedWg = [...wgCorrs].filter(c => c.corr != null).sort((a,b)=>Math.abs(b.corr)-Math.abs(a.corr)).slice(0,6);
  if (sortedWg.length) {
    const html2 = `<div style="display:grid;grid-template-columns:1fr auto;gap:5px;font-size:13px;line-height:1.7;">
      ${sortedWg.map(c=>{
        const arrow = c.corr > 0 ? `<span style="color:#b42318;">↑ within group</span>` : `<span style="color:#16803c;">↓ within group</span>`;
        return `<div style="font-weight:600;">${c.feature}</div><div>${arrow}</div>`;
      }).join("")}
    </div>`;
    d3.select("#within-group-correlations").html(html2);
  } else {
    d3.select("#within-group-correlations").text("Within-group correlation data not available.");
  }
}

function formatFeatureVal(v, fmt) {
  if (v == null) return "n/a";
  if (fmt === "currency") return fmtMoney(v);
  if (fmt === "percent") return `${fmtNum(v)}%`;
  if (fmt === "pct") return fmtPct(v);
  return fmtNum(v);
}

// ---- playbook section ----
function initPlaybookLegacy() {
  if (!DATA.playbook || !DATA.playbook.available) {
    d3.select("#model-unavailable").style("display","block");
    d3.select("#county-search").style("display","none");
    d3.select("#county-selection-map").style("display","none");
    return;
  }

  // Draw county selection map — all grey initially
  drawMap("#county-selection-map",
    (county, fips) => fips === selectedCountyFips ? "#172026" : "#d8d0c4",
    (county, fips) => {
      const c = DATA.playbook.counties.find(x => x.fips === fips);
      return c ? `<strong>${countyDisplayName(c)}</strong><br>Risk: ${c.riskRating||"Unknown"}` : "";
    },
    null, null,
    (fips) => {
      const c = DATA.playbook.counties.find(x => x.fips === fips);
      if (c) { selectedCountyFips = c.fips; showCountyPlaybook(c); }
    }
  );

  // Search
  const input = d3.select("#county-search");
  const results = d3.select("#county-results");
  input.on("input", function() {
    const q = this.value.toLowerCase().trim();
    if (q.length < 2) { results.style("display","none").html(""); return; }
    const matches = DATA.playbook.counties.filter(c =>
      c.county.toLowerCase().includes(q) || c.state.toLowerCase().includes(q) || c.fips.includes(q)
    ).slice(0,20);
    results.style("display", matches.length ? "block" : "none").html("")
      .selectAll("div").data(matches).join("div")
      .style("padding","8px 12px").style("cursor","pointer")
      .style("border-bottom","1px solid var(--line)").style("font-size","13px")
      .html(d=>`<strong>${countyDisplayName(d)}</strong> <span style="color:var(--muted);">(${d.riskRating||"Unknown risk"})</span>`)
      .on("click",(event,d)=>{ selectedCountyFips=d.fips; showCountyPlaybook(d); input.property("value",""); results.style("display","none").html(""); });
  });
}

function showCountyPlaybookLegacy(county) {
  d3.select("#playbook-display").style("display","block");
  const risk = county.riskRating || "Unknown";
  d3.select("#projected-risk-rating").html(`<span style="color:${RISK_COLORS[risk]||'#666'};font-size:18px;font-weight:800;">${risk}</span>`);

  // Re-draw map to highlight selected county
  drawMap("#county-selection-map",
    (c, fips) => fips === county.fips ? "#172026" : "#d8d0c4",
    (c, fips) => {
      const pc = DATA.playbook.counties.find(x => x.fips === fips);
      return pc ? `<strong>${countyDisplayName(pc)}</strong><br>Risk: ${pc.riskRating||"Unknown"}` : "";
    },
    null, null,
    (fips) => {
      const pc = DATA.playbook.counties.find(x => x.fips === fips);
      if (pc) { selectedCountyFips = pc.fips; showCountyPlaybook(pc); }
    }
  );

  // Model features with county values and the same binned NRI associations used above.
  const featureMap = Object.fromEntries((DATA.playbook.featureMap || []).map(f => [f.modelKey, f]));
  const rankedKeys = (DATA.playbook.model.topFeatures || []).map(d => d.feature);
  const allKeys = DATA.playbook.model.featureNames || [];
  const modelKeys = [...rankedKeys, ...allKeys.filter(key => !rankedKeys.includes(key))];
  const profile = DATA.features.countyProfiles.find(d => d.fips === county.fips);
  const countyBins = (DATA.features.countyFeatureBins || {})[county.fips] || {};

  const html = modelKeys.map(key => {
    const cfg = featureMap[key] || {label: key, featureLabel: key, format: "number"};
    const featureLabel = cfg.featureLabel || cfg.label || key;
    const value = profile?.featureValues?.[featureLabel];
    const bin = countyBins[featureLabel];
    const fmt = DATA.features.featureMeta?.[featureLabel]?.format || cfg.format || "number";
    return `<div style="display:grid;grid-template-columns:minmax(170px,1.2fr) auto minmax(180px,1fr);gap:10px;align-items:center;margin-bottom:7px;font-size:13px;">
      <div style="font-weight:700;">${cfg.label || featureLabel}</div>
      <div style="text-align:right;">${formatFeatureVal(value, fmt)}</div>
      <div>${riskAssociationText(bin?.riskAssociation)}</div>
    </div>`;
  }).join("");
  d3.select("#playbook-county-features").html(html ||
    `<p style="color:var(--muted);font-size:14px;">Model feature data not available.</p>`);

  // Expected impact
  d3.select("#expected-impact").html(`<p style="line-height:1.55;font-size:15px;">${TEXT.riskImpacts[risk] || "Impact data not available for this risk level."}</p>`);
}

function drawPlaybookMapLegacy(county = null, zoomed = false) {
  const svg = d3.select("#county-selection-map");
  const width = svg.node().clientWidth || 1100;
  const height = svg.node().clientHeight || 380;
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);
  const group = svg.append("g");

  group.selectAll("path")
    .data(DATA.geojson.features)
    .join("path")
    .attr("class", "county")
    .attr("d", path)
    .attr("fill", d => d.properties.fips === county?.fips ? "#172026" : "#d8d0c4")
    .style("cursor", d => DATA.playbook.counties.some(c => c.fips === d.properties.fips) ? "pointer" : null)
    .on("mousemove", (event, d) => {
      const profile = DATA.playbook.counties.find(c => c.fips === d.properties.fips);
      if (!profile) return;
      tooltip.style("display", "block")
        .style("left", `${event.clientX + 12}px`)
        .style("top", `${event.clientY + 12}px`)
        .html(`<strong>${countyDisplayName(profile)}</strong><br>Overall NRI rating: ${profile.hazards?.overall?.rating || "n/a"}`);
    })
    .on("mouseleave", () => tooltip.style("display", "none"))
    .on("click", (event, d) => {
      const profile = DATA.playbook.counties.find(c => c.fips === d.properties.fips);
      if (profile) selectPlaybookCounty(profile);
    });

  if (county && zoomed) {
    const selectedFeature = DATA.geojson.features.find(d => d.properties.fips === county.fips);
    if (selectedFeature) {
      const [cx, cy] = path.centroid(selectedFeature);
      const scale = county.state === "AK" ? 3.2 : county.state === "HI" ? 4.2 : 5.2;
      group.transition().duration(520)
        .attr("transform", `translate(${width / 2},${height / 2}) scale(${scale}) translate(${-cx},${-cy})`);
    }
  }

  d3.select("#playbook-map-zoom-toggle")
    .style("display", county ? "inline-flex" : "none")
    .text(zoomed ? TEXT.playbookZoomOut : TEXT.playbookZoomIn);
}

function updatePlaybookZoomControl() {
  const county = playbookCountyByFips.get(selectedCountyFips);
  const zoomed = playbookMapTransform.k > 1.01;
  d3.select("#playbook-map-zoom-toggle")
    .style("display", county || zoomed ? "inline-flex" : "none")
    .text(zoomed ? TEXT.playbookZoomOut : TEXT.playbookZoomIn);
}

function drawPlaybookMap(county = null, autoZoom = false) {
  const svg = d3.select("#county-selection-map");
  const width = svg.node().clientWidth || 1100;
  const height = svg.node().clientHeight || 380;
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);
  const group = svg.append("g");

  group.selectAll("path")
    .data(DATA.geojson.features)
    .join("path")
    .attr("class", "county")
    .attr("d", path)
    .attr("fill", d => d.properties.fips === county?.fips ? "#172026" : "#d8d0c4")
    .style("cursor", "pointer")
    .on("mousemove", (event, d) => {
      const profile = playbookCountyByFips.get(d.properties.fips);
      if (!profile) return;
      tooltip.style("display", "block")
        .style("left", `${event.clientX + 12}px`)
        .style("top", `${event.clientY + 12}px`)
        .html(`<strong>${countyDisplayName(profile)}</strong><br>Overall NRI rating: ${profile.hazards?.overall?.rating || "n/a"}`);
    })
    .on("mouseleave", () => tooltip.style("display", "none"))
    .on("click", (event, d) => {
      const profile = playbookCountyByFips.get(d.properties.fips);
      if (profile) selectPlaybookCounty(profile);
    });

  playbookZoomBehavior = d3.zoom()
    .scaleExtent([1, 12])
    .translateExtent([[-width * .35, -height * .35], [width * 1.35, height * 1.35]])
    .on("zoom", event => {
      playbookMapTransform = event.transform;
      playbookMapZoomed = event.transform.k > 1.01;
      group.attr("transform", event.transform);
      updatePlaybookZoomControl();
    });
  svg.call(playbookZoomBehavior).on("dblclick.zoom", null);
  svg.call(playbookZoomBehavior.transform, d3.zoomIdentity);

  if (county) {
    const selectedFeature = DATA.geojson.features.find(d => d.properties.fips === county.fips);
    if (selectedFeature) {
      const [cx, cy] = path.centroid(selectedFeature);
      const scale = county.state === "AK" ? 3.2 : county.state === "HI" ? 4.2 : 5.2;
      playbookSelectedTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(scale).translate(-cx, -cy);
      if (autoZoom) svg.transition().duration(520).call(playbookZoomBehavior.transform, playbookSelectedTransform);
    }
  } else {
    playbookSelectedTransform = d3.zoomIdentity;
    playbookMapTransform = d3.zoomIdentity;
    playbookMapZoomed = false;
    updatePlaybookZoomControl();
  }
}

function drawPlaybookHistory(county) {
  const parseMonth = d3.utcParse("%Y-%m");
  const observedHistory = ((DATA.playbook.monthlyHistoryByFips || {})[county.fips] || [])
    .filter(d => d.value != null)
    .map(d => ({...d, date: parseMonth(d.month)}))
    .sort((a, b) => d3.ascending(a.date, b.date));
  const historyStart = parseMonth(DATA.playbook.historyStart);
  const historyEnd = parseMonth(DATA.playbook.historyEnd);
  const historyDomainEnd = d3.utcDay.offset(d3.utcMonth.offset(historyEnd, 1), -1);
  const observedByMonth = new Map(observedHistory.map(d => [d.month, d.value]));
  const history = d3.utcMonth.range(historyStart, d3.utcMonth.offset(historyEnd, 1)).map(date => {
    const month = d3.utcFormat("%Y-%m")(date);
    return {date, month, value: observedByMonth.has(month) ? observedByMonth.get(month) : null};
  });
  const observed = history.filter(d => d.value != null);
  const events = ((DATA.playbook.eventsByFips || {})[county.fips] || [])
    .map(d => ({...d, startDate: parseMonth(d.start), endDate: parseMonth(d.end)}));
  const svg = d3.select("#playbook-ppsf-history");
  const width = svg.node().clientWidth || 1050;
  const height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: 24, bottom: 42, left: 62};
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();

  const x = d3.scaleUtc().domain([historyStart, historyDomainEnd]).range([margin.left, width - margin.right]);
  const valueExtent = observed.length ? d3.extent(observed, d => d.value) : [-0.1, 0.1];
  const padding = Math.max((valueExtent[1] - valueExtent[0]) * 0.12, 0.01);
  const y = d3.scaleLinear().domain([valueExtent[0] - padding, valueExtent[1] + padding]).nice()
    .range([height - margin.bottom, margin.top]);

  svg.append("g").attr("class", "grid").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(""));

  const chartStart = x.domain()[0], chartEnd = x.domain()[1];
  svg.append("g").selectAll("rect.missing-period")
    .data(history.filter(d => d.value == null))
    .join("rect")
    .attr("class", "missing-period")
    .attr("x", d => x(d.date))
    .attr("y", margin.top)
    .attr("width", d => Math.max(2, x(d3.utcMonth.offset(d.date, 1)) - x(d.date)))
    .attr("height", height - margin.top - margin.bottom)
    .attr("fill", "#b8c0c5").attr("opacity", .3);
  svg.append("g").selectAll("rect.event-period")
    .data(events.filter(d => d.endDate >= chartStart && d.startDate <= chartEnd))
    .join("rect")
    .attr("class", "event-period")
    .attr("x", d => x(d3.max([d.startDate, chartStart])))
    .attr("y", margin.top)
    .attr("width", d => Math.max(3, x(d3.min([d.endDate, chartEnd])) - x(d3.max([d.startDate, chartStart]))))
    .attr("height", height - margin.top - margin.bottom)
    .attr("fill", "#df7d2f").attr("opacity", .16)
    .on("mousemove", (event, d) => tooltip.style("display", "block")
      .style("left", `${event.clientX + 12}px`).style("top", `${event.clientY + 12}px`)
      .html(`<strong>${d.name || d.type}</strong><br>${d3.utcFormat("%b %Y")(d.startDate)} to ${d3.utcFormat("%b %Y")(d.endDate)}<br>${String(d.source).toUpperCase()}`))
    .on("mouseleave", () => tooltip.style("display", "none"));

  svg.append("path").datum(history).attr("class", "line")
    .attr("stroke", "#0f766e").attr("stroke-width", 2.4)
    .attr("d", d3.line().defined(d => d.value != null).x(d => x(d.date)).y(d => y(d.value)));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(d3.utcYear.every(1)).tickFormat(d3.utcFormat("%Y")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  svg.append("text").attr("x", margin.left).attr("y", 13).attr("fill", "#66717b").attr("font-size", 11)
    .text("Median PPSF YoY");
  const legendItems = [];
  if (events.length) legendItems.push({label: TEXT.playbookEventLegend, color: "#df7d2f", opacity: .22});
  if (history.some(d => d.value == null)) legendItems.push({label: TEXT.playbookMissingDataLegend, color: "#b8c0c5", opacity: .45});
  const legend = svg.append("g").attr("transform", `translate(${Math.max(margin.left, width - 310)},10)`);
  legendItems.forEach((item, index) => {
    const offset = index * 150;
    legend.append("rect").attr("x", offset).attr("width", 14).attr("height", 9).attr("fill", item.color).attr("opacity", item.opacity);
    legend.append("text").attr("x", offset + 20).attr("y", 9).attr("font-size", 10).attr("fill", "#66717b").text(item.label);
  });
  if (!observed.length) {
    svg.append("text").attr("x", width / 2).attr("y", height / 2)
      .attr("text-anchor", "middle").attr("fill", "#66717b")
      .text(TEXT.playbookMissingDataLegend);
  }
  renderPlaybookCommentary(county, history, events);
}

function fillTextTemplate(template, values) {
  return Object.entries(values).reduce(
    (text, [key, value]) => text.replaceAll(`{${key}}`, value == null ? "" : String(value)),
    template || ""
  );
}

function renderPlaybookCommentary(county, history, events) {
  const container = d3.select("#playbook-event-commentary");
  const risk = county.hazards?.overall?.rating || county.riskRating || "Unknown";
  const copy = TEXT.playbookTakeaways;
  const terms = TEXT.playbookTakeawayTerms;
  const expectation = TEXT.riskImpacts[risk] || copy.unavailableExpectation;
  if (!events.length) {
    container.html(fillTextTemplate(copy.noEvents, {expectation}));
    return;
  }

  const assessments = events.map(event => {
    const preStart = d3.utcMonth.offset(event.startDate, -12);
    const postEnd = d3.utcMonth.offset(event.endDate, 12);
    const before = history.filter(d => d.value != null && d.date >= preStart && d.date < event.startDate).map(d => d.value);
    const after = history.filter(d => d.value != null && d.date > event.endDate && d.date <= postEnd).map(d => d.value);
    const beforeMedian = before.length >= 3 ? d3.median(before) : null;
    const afterMedian = after.length >= 3 ? d3.median(after) : null;
    const delta = beforeMedian == null || afterMedian == null ? null : afterMedian - beforeMedian;
    return {...event, delta};
  });
  const measured = assessments.filter(d => d.delta != null);
  const medianDelta = measured.length ? d3.median(measured, d => d.delta) : null;
  let observation = copy.insufficientHistory;
  let alignment = "";
  if (medianDelta != null) {
    const direction = medianDelta < -0.01 ? terms.declined : medianDelta > 0.01 ? terms.increased : terms.unchanged;
    observation = fillTextTemplate(copy.observedChange, {
      measuredCount: measured.length,
      eventNoun: measured.length === 1 ? terms.event : terms.events,
      direction,
      magnitude: Math.abs(medianDelta * 100).toFixed(1),
    });
    const expectedDecline = ["Low", "Medium", "High", "Very High"].includes(risk);
    const aligned = risk === "Very Low" ? Math.abs(medianDelta) <= 0.01 : expectedDecline && medianDelta < 0;
    alignment = fillTextTemplate(copy.expectationComparison, {
      alignment: aligned ? terms.aligned : terms.notAligned,
      risk,
      expectation,
    });
  }
  const details = assessments.map(d => {
    const result = d.delta == null
      ? terms.insufficient
      : `${d.delta < -0.01 ? terms.down : d.delta > 0.01 ? terms.up : terms.littleChanged} ${Math.abs(d.delta * 100).toFixed(1)} pp`;
    return fillTextTemplate(copy.eventDetail, {
      name: d.name || d.type,
      start: d3.utcFormat("%b %Y")(d.startDate),
      end: d3.utcFormat("%b %Y")(d.endDate),
      result,
    });
  }).join("");
  container.html(fillTextTemplate(copy.eventSummary, {
    eventCount: events.length,
    eventNoun: events.length === 1 ? terms.event : terms.events,
    observation,
    alignment,
    details,
  }));
}

function renderPlaybookHazards(county) {
  const hazards = DATA.playbook.hazards || DATA.priceRisk.hazards || [];
  d3.select("#playbook-hazard-ratings").html(hazards.map(hazard => {
    const rating = county.hazards?.[hazard.key]?.rating || "No rating";
    const color = RISK_COLORS[rating] || "#66717b";
    return `<div class="hazard-rating-item"><span><span class="hazard-icon">${HAZARD_ICONS[hazard.key] || ""}</span>${hazard.label}</span><strong style="color:${color};">${rating}</strong></div>`;
  }).join(""));
}

function selectPlaybookCounty(county) {
  selectedCountyFips = county.fips;
  playbookMapZoomed = true;
  d3.select("#playbook-display").style("display", "block");
  drawPlaybookMap(county, true);
  renderPlaybookHazards(county);
  drawPlaybookHistory(county);
}

function initPlaybook() {
  if (!DATA.playbook?.available) {
    d3.select("#county-search").property("disabled", true).property("placeholder", DATA.playbook?.message || "County data unavailable");
    return;
  }
  drawPlaybookMap();
  const input = d3.select("#county-search");
  const results = d3.select("#county-results");
  input.on("input", function() {
    const query = this.value.toLowerCase().trim();
    if (query.length < 2) { results.style("display", "none").html(""); return; }
    const matches = DATA.playbook.counties.filter(c =>
      c.county.toLowerCase().includes(query) || c.state.toLowerCase().includes(query) || c.fips.includes(query)
    ).slice(0, 20);
    results.style("display", matches.length ? "block" : "none").html("")
      .selectAll("div").data(matches).join("div")
      .style("padding", "8px 12px").style("cursor", "pointer")
      .style("border-bottom", "1px solid var(--line)").style("font-size", "13px")
      .html(d => `<strong>${countyDisplayName(d)}</strong> <span style="color:var(--muted);">(${d.hazards?.overall?.rating || "Unknown risk"})</span>`)
      .on("click", (event, d) => { selectPlaybookCounty(d); input.property("value", ""); results.style("display", "none").html(""); });
  });
  d3.select("#playbook-map-zoom-in").on("click", () => {
    if (playbookZoomBehavior) d3.select("#county-selection-map").transition().duration(220).call(playbookZoomBehavior.scaleBy, 1.5);
  });
  d3.select("#playbook-map-zoom-minus").on("click", () => {
    if (playbookZoomBehavior) d3.select("#county-selection-map").transition().duration(220).call(playbookZoomBehavior.scaleBy, 1 / 1.5);
  });
  d3.select("#playbook-map-zoom-toggle").on("click", () => {
    if (!playbookZoomBehavior) return;
    const target = playbookMapTransform.k > 1.01 ? d3.zoomIdentity : playbookSelectedTransform;
    d3.select("#county-selection-map").transition().duration(420).call(playbookZoomBehavior.transform, target);
  });
}

function showCountyPlaybook(county) {
  selectPlaybookCounty(county);
}

// ---- bootstrap ----
hydrateText();
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) entry.target.classList.add("visible","transition-in");
    else if (entry.boundingClientRect.top < 0) entry.target.classList.add("transition-out");
  });
}, { threshold: .18 });
document.querySelectorAll(".slide").forEach(el => observer.observe(el));
initButtons();
drawScoreScatter();
drawRatingScatter();
drawRatingMap();
renderEventSection();
drawFeatureHeatmaps();
initPlaybook();
startRiskTimer();
window.addEventListener("resize", () => {
  drawScoreScatter();
  drawRatingScatter();
  drawRatingMap();
  renderEventSection();
  drawFeatureHeatmaps();
  const playbookCounty = DATA.playbook?.counties?.find(c => c.fips === selectedCountyFips);
  drawPlaybookMap(playbookCounty || null, playbookCounty ? playbookMapZoomed : false);
  if (playbookCounty) drawPlaybookHistory(playbookCounty);
});
</script>
</body>
</html>"""


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        price_risk = build_price_risk(con)
        features = build_feature_payload(con)
        eligible_feature_fips_by_risk: dict[str, set[str]] = {}
        for risk in RISK_ORDER:
            top_labels = {
                item["feature"]
                for item in features.get("withinGroupTopFeatures", {}).get(risk, [])
            }
            county_bins = features.get("withinGroupFeatureBins", {}).get(risk, {})
            eligible_feature_fips_by_risk[risk] = {
                str(fips)
                for fips, values in county_bins.items()
                if top_labels
                and all(
                    label in values and values[label].get("value") is not None
                    for label in top_labels
                )
            }
        event_windows = build_event_windows(con, eligible_feature_fips_by_risk)
        playbook = build_county_playbook_data(con)

    # Inject event-window percentile ranks from Window A into the features payload.
    # This replaces the old static-feature percentile computation.
    features["withinGroupPercentiles"] = event_windows["windowA"].get("countyEventWindowPctRank", {})

    geojson = build_geojson({county["fips"] for county in playbook["counties"]})
    data = {"priceRisk": price_risk, "eventWindows": event_windows, "features": features, "playbook": playbook, "geojson": geojson}
    OUT_PATH.write_text(make_html(data), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
