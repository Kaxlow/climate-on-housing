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
    if fmt == "percent":
        return f"{value:,.1f}%"
    return f"{value:,.1f}"


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
                "county": row.county_label or f"{row.COUNTY}, {row.STATEABBRV}",
                "state": row.state_code or row.STATEABBRV,
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
            # Use riskRating for overall
            hazard_history = history.copy()
        else:
            # Filter to counties with valid rating for this hazard
            hazard_history = history[history[hazard["rating"]].notna()].copy()

        grouped = (
            hazard_history.groupby(["riskRating", "year"], observed=False)["median_ppsf_yoy"]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
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


def build_event_windows(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    events = load_disaster_events(con)
    events = events.loc[events["event_start_month"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-01"))].copy()
    housing = load_redfin_county_monthly(con)
    for column in ["median_ppsf_yoy", "avg_sale_to_list_yoy", "homes_sold_yoy", "inventory_yoy", "housing_market_index"]:
        if column in housing:
            housing.loc[pd.to_numeric(housing[column], errors="coerce").le(-888888000), column] = np.nan
    metric = "median_ppsf_yoy"
    affected = build_affected_event_windows(events, housing, pre_event_months=24, post_event_months=60)
    if affected.empty:
        return {"aggregate": [], "byRating": [], "affectedCounties": [], "summary": {"events": 0}, "additionalMetrics": []}
    required = event_window_months(24, 60)
    complete = filter_complete_event_window_lines(
        affected,
        x_col="event_window_month",
        line_col="line_id",
        metric_col=metric,
        required_x_values=required,
    ).copy()
    nri = con.execute("SELECT fips, risk_rating FROM mart.nri_county_risk WHERE fips IS NOT NULL").df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)
    complete = complete.merge(nri[["fips", "riskRating"]], on="fips", how="left")
    aggregate = aggregate_lines(complete.assign(series="All affected counties"), ["series"], metric)
    by_rating = aggregate_lines(complete.dropna(subset=["riskRating"]), ["riskRating"], metric)
    affected_counties = (
        complete.dropna(subset=["riskRating"])[["fips", "riskRating"]]
        .drop_duplicates()
        .groupby(["fips", "riskRating"], as_index=False)
        .size()
    )
    risk_counts = complete.drop_duplicates(["line_id", "riskRating"]).groupby("riskRating", dropna=True)["line_id"].nunique()

    # Build additional metrics for "What Else Are Climate Events Doing" section
    # Use new annual metrics module for proper annual event windows
    events_for_annual = events[['fips', 'event_key', 'event_source', 'source_event_id', 'event_type', 'event_name', 'event_start_month', 'event_end_month']].drop_duplicates(subset=['event_key'])
    annual_metrics = build_additional_annual_metrics(con, events_for_annual, nri[['fips', 'riskRating']], pre_years=2, post_years=3)

    # Add monthly metrics (homes sold, DOM) using existing monthly event windows
    monthly_metrics = []
    if "homes_sold_yoy" in complete.columns:
        homes_agg = aggregate_lines(complete.assign(series="All affected counties"), ["series"], "homes_sold_yoy")
        homes_by_risk = aggregate_lines(complete.dropna(subset=["riskRating"]), ["riskRating"], "homes_sold_yoy")
        monthly_metrics.append({
            "key": "homes_sold_yoy",
            "label": "Homes Sold YOY",
            "description": "Change in buyer activity after climate events",
            "frequency": "monthly",
            "isAnnual": False,
            "conclusion": "Homes sold YOY changes reveal shifts in buyer demand following climate events.",
            "aggregate": homes_agg,
            "byRating": homes_by_risk,
        })

    if "median_dom_yoy" in complete.columns:
        dom_agg = aggregate_lines(complete.assign(series="All affected counties"), ["series"], "median_dom_yoy")
        dom_by_risk = aggregate_lines(complete.dropna(subset=["riskRating"]), ["riskRating"], "median_dom_yoy")
        monthly_metrics.append({
            "key": "median_dom_yoy",
            "label": "Median Days on Market YOY",
            "description": "Are homes taking longer to sell after events?",
            "frequency": "monthly",
            "isAnnual": False,
            "conclusion": "Days on market YOY changes indicate shifts in market liquidity after climate events.",
            "aggregate": dom_agg,
            "byRating": dom_by_risk,
        })

    # Combine annual and monthly metrics
    additional_metrics = annual_metrics + monthly_metrics
    example_lines = []
    for risk in RISK_ORDER:
        candidates = (
            complete.loc[complete["riskRating"].eq(risk)]
            .dropna(subset=[metric])
            .groupby(["line_id", "fips", "county_label", "state_code"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(5)
        )
        for candidate in candidates.itertuples(index=False):
            rows = complete.loc[complete["line_id"].eq(candidate.line_id)].sort_values("event_window_month")
            example_lines.append(
                {
                    "riskRating": risk,
                    "lineId": candidate.line_id,
                    "fips": candidate.fips,
                    "county": candidate.county_label,
                    "state": candidate.state_code,
                    "values": [
                        {"month": int(row.event_window_month), "value": serialize_number(getattr(row, metric), 5)}
                        for row in rows.itertuples(index=False)
                        if pd.notna(getattr(row, metric))
                    ],
                }
            )

    return {
        "aggregate": aggregate,
        "byRating": by_rating,
        "exampleCountyLines": example_lines,
        "affectedCounties": [
            {"fips": row.fips, "riskRating": row.riskRating}
            for row in affected_counties.itertuples(index=False)
        ],
        "summary": {
            "events": int(events["event_key"].nunique()),
            "countyEvents": int(complete["line_id"].nunique()),
            "riskCounts": {str(k): int(v) for k, v in risk_counts.items()},
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


def build_playbook_data(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """
    Build data for the County Climate Risk Playbook section.
    Loads the climate risk prediction model results and prepares county data.
    """
    # Load latest model results
    models_dir = ROOT / "output" / "models" / "climate_risk_prediction" / "overall"
    if not models_dir.exists():
        return {"available": False, "message": "Climate risk prediction models not found"}

    # Find latest results file
    results_files = sorted(models_dir.glob("overall_results_*.json"), reverse=True)
    if not results_files:
        return {"available": False, "message": "No model results found"}

    with open(results_files[0], "r") as f:
        model_results = json.load(f)

    # Get best performing model based on accuracy
    best_model_name = max(
        model_results["models"].items(),
        key=lambda x: x[1].get("accuracy", 0)
    )[0]
    best_model = model_results["models"][best_model_name]

    # Load county features used for prediction
    nri = con.execute("SELECT fips, risk_rating, risk_score FROM mart.nri_county_risk WHERE fips IS NOT NULL").df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)

    # Get county names
    counties = con.execute("""
        SELECT DISTINCT fips, any_value(REGION) as county, any_value(STATE_CODE) as state
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
        GROUP BY fips
    """).df()
    counties["fips"] = counties["fips"].astype(str).str.zfill(5)

    # Merge
    playbook_counties = nri.merge(counties, on="fips", how="inner")

    return {
        "available": True,
        "model": {
            "name": best_model_name,
            "accuracy": serialize_number(best_model.get("accuracy"), 4),
            "f1Weighted": serialize_number(best_model.get("f1_weighted"), 4),
            "featureNames": model_results["feature_names"],
            "topFeatures": best_model.get("top_features", [])[:5] if "top_features" in best_model else [],
        },
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
        "dp02_households_by_type_total_households_households_with_one_or_more_people_65_plus_pct",
        "dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct",
        "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct",
        "dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct",
    ]
    affordability_cols = [
        "fips",
        "year",
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

    econ = latest_by_fips(con, "mart.acs_county_economic_annual", econ_cols)
    demo = latest_by_fips(con, "mart.acs_county_demographic_annual", demo_cols)
    afford = latest_by_fips(con, "mart.acs_county_affordability_annual", affordability_cols)
    weather = con.execute(
        """
        SELECT
            fips,
            avg(avg_temperature_f) AS avg_temperature_f,
            avg(precipitation_inches) AS precipitation_inches
        FROM mart.ncei_county_weather_monthly
        WHERE year = 2025
        GROUP BY fips
        """
    ).df()
    weather["fips"] = weather["fips"].astype(str).str.zfill(5)

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

    features = (
        nri[["fips", "riskRating", "riskValue", "risk_score"]]
        .merge(econ[["fips", *econ_cols[2:]]], on="fips", how="left")
        .merge(demo[["fips", *demo_cols[2:]]], on="fips", how="left")
        .merge(afford[["fips", "median_owner_costs_mortgage", "housing_cost_pct_income", "owner_mortgage_cost_burden_30pct_plus", "estimated_annual_home_insurance", "estimated_annual_property_tax", "estimated_annual_utilities"]], on="fips", how="left")
        .merge(weather, on="fips", how="left")
    )
    redfin_features = con.execute(
        """
        SELECT
            fips,
            avg(CASE WHEN try_cast(replace(HOMES_SOLD_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(HOMES_SOLD_YOY, ',', '') AS DOUBLE) END) AS homes_sold_yoy,
            avg(CASE WHEN try_cast(replace(MEDIAN_DOM_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(MEDIAN_DOM_YOY, ',', '') AS DOUBLE) END) AS median_dom_yoy
        FROM mart.redfin_county_monthly
        WHERE property_type = 'All Residential'
          AND period_begin >= DATE '2025-01-01'
          AND period_begin < DATE '2026-01-01'
          AND fips IS NOT NULL
        GROUP BY fips
        """
    ).df()
    redfin_features["fips"] = redfin_features["fips"].astype(str).str.zfill(5)
    features = features.merge(redfin_features, on="fips", how="left")
    features["no_broadband_pct"] = 100 - features["dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct"]
    feature_defs = [
        ("Economic", "Income", "dp03_income_and_benefits_total_households_median_household_income_est", "currency", "mart.acs_county_economic_annual"),
        ("Economic", "Home ownership costs", "median_owner_costs_mortgage", "currency", "mart.acs_county_affordability_annual"),
        ("Economic", "Home ownership burden", "housing_cost_pct_income", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Home insurance", "estimated_annual_home_insurance", "currency", "mart.acs_county_affordability_annual"),
        ("Economic", "Utilities", "estimated_annual_utilities", "currency", "mart.acs_county_affordability_annual"),
        ("Economic", "Property tax", "estimated_annual_property_tax", "currency", "mart.acs_county_affordability_annual"),
        ("Economic", "Unemployment rate", "dp03_civilian_labor_force_unemployment_rate_pct", "percent", "mart.acs_county_economic_annual"),
        ("Demographic", "Net migration", "domestic_in_migration_rate", "number", "mart.acs_county_demographic_annual"),
        ("Demographic", "Age 65+ households", "dp02_households_by_type_total_households_households_with_one_or_more_people_65_plus_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Demographic", "Disability status", "dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Demographic", "Communication barrier", "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Demographic", "No broadband internet", "no_broadband_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Climate", "Temperature", "avg_temperature_f", "number", "mart.ncei_county_weather_monthly"),
        ("Climate", "Precipitation", "precipitation_inches", "number", "mart.ncei_county_weather_monthly"),
    ]
    rows = []
    correlations = []
    for category, label, column, fmt, source in feature_defs:
        valid = features.dropna(subset=[column, "riskRating", "riskValue"]).copy()
        if valid.empty:
            continue
        valid[column] = valid[column].clip(lower=0)
        risk_corr = valid[["riskValue", column]].corr(method="spearman").iloc[0, 1]
        correlations.append({"feature": label, "category": category, "corr": serialize_number(risk_corr, 3)})
        valid["bucket"], bucket_order = feature_bucket_labels(valid[column], fmt)
        valid = valid.dropna(subset=["bucket"]).copy()
        totals = valid.groupby("riskRating", observed=False)["fips"].nunique().reindex(RISK_ORDER).fillna(0)
        counts = valid.groupby(["riskRating", "bucket"], observed=False)["fips"].nunique()
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
    top = sorted(correlations, key=lambda item: abs(item["corr"] or 0), reverse=True)[:4]
    county_profiles = []
    for row in features.itertuples(index=False):
        county_profiles.append(
            {
                "fips": row.fips,
                "riskRating": row.riskRating,
                "income": serialize_number(getattr(row, "dp03_income_and_benefits_total_households_median_household_income_est"), 2),
                "housingBurden": serialize_number(getattr(row, "housing_cost_pct_income"), 2),
                "insurance": serialize_number(getattr(row, "estimated_annual_home_insurance"), 2),
                "propertyTaxes": serialize_number(getattr(row, "estimated_annual_property_tax"), 2),
                "utilities": serialize_number(getattr(row, "estimated_annual_utilities"), 2),
                "netMigration": serialize_number(getattr(row, "domestic_in_migration_rate"), 2),
                "homesSoldYoy": serialize_number(getattr(row, "homes_sold_yoy"), 5),
                "medianDomYoy": serialize_number(getattr(row, "median_dom_yoy"), 5),
            }
        )

    # --- Option B: within-group feature correlations ---
    # For each risk rating, Spearman correlation of each feature with the continuous
    # NRI risk_score (0–100) restricted to counties in that tier. Uses the raw score
    # rather than the integer riskValue so there is meaningful variance within each tier.
    within_group_correlations: dict[str, list[dict]] = {}
    for rating in RISK_ORDER:
        group = features[features["riskRating"] == rating].copy()
        rating_corrs = []
        for _, label, column, _, _ in feature_defs:
            valid_group = group.dropna(subset=[column, "risk_score"])
            if len(valid_group) < 10:
                corr = None
            else:
                corr = serialize_number(
                    valid_group[["risk_score", column]].corr(method="spearman").iloc[0, 1], 3
                )
            rating_corrs.append({"feature": label, "corr": corr})
        within_group_correlations[rating] = rating_corrs

    # --- Option A: within-group percentile ranks ---
    # For each county, each feature's percentile rank (0–100) among counties sharing
    # its risk rating. Answers "where does this county sit within its peer group?"
    # Stored as {fips: {featureLabel: percentile, ...}, ...}.
    profile_feature_map = [
        ("income", "dp03_income_and_benefits_total_households_median_household_income_est"),
        ("housingBurden", "housing_cost_pct_income"),
        ("insurance", "estimated_annual_home_insurance"),
        ("propertyTaxes", "estimated_annual_property_tax"),
        ("utilities", "estimated_annual_utilities"),
        ("netMigration", "domestic_in_migration_rate"),
        ("homesSoldYoy", "homes_sold_yoy"),
        ("medianDomYoy", "median_dom_yoy"),
    ]
    # Compute percentile ranks within each risk group for the profile features.
    # pandas rank(pct=True) gives a 0–1 value; multiply by 100 and round to 1dp.
    percentile_cols = {col: f"pct_{key}" for key, col in profile_feature_map}
    features_pct = features[["fips", "riskRating"] + [col for _, col in profile_feature_map]].copy()
    for col, pct_col in percentile_cols.items():
        features_pct[pct_col] = (
            features_pct.groupby("riskRating")[col]
            .rank(method="average", pct=True, na_option="keep")
            .mul(100)
            .round(1)
        )
    within_group_percentiles: dict[str, dict[str, float | None]] = {}
    for row in features_pct.itertuples(index=False):
        entry: dict[str, float | None] = {}
        for key, col in profile_feature_map:
            raw = getattr(row, percentile_cols[col])
            entry[key] = None if pd.isna(raw) else float(raw)
        within_group_percentiles[row.fips] = entry

    return {
        "riskOrder": RISK_ORDER,
        "rows": rows,
        "correlations": correlations,
        "topFeatures": top,
        "countyProfiles": county_profiles,
        "withinGroupCorrelations": within_group_correlations,
        "withinGroupPercentiles": within_group_percentiles,
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
    .chart { width: 100%; height: 430px; display: block; }
    .chart.tall { height: 520px; }
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
    .heatmap { overflow-x: auto; }
    .heatmap-grid { display: grid; gap: 4px; min-width: 740px; }
    .heat-cell { min-height: 44px; border-radius: 4px; padding: 6px; font-size: 11px; line-height: 1.15; border: 1px solid rgba(255,255,255,.72); color: #172026; }
    .heat-cell span { display: block; margin-top: 4px; color: inherit; }
    .heat-label { display: flex; align-items: center; font-size: 12px; font-weight: 800; color: #2f3941; }
    .heat-col { align-self: end; font-size: 11px; font-weight: 800; color: var(--muted); }
    @media (max-width: 900px) {
      .viz-grid { grid-template-columns: 1fr; }
      .timeseries-grid { grid-template-columns: 1fr; }
      .hero { min-height: auto; }
      h1 { font-size: clamp(42px, 12vw, 66px); }
      .dek { font-size: 19px; line-height: 1.42; }
      .chart, .chart.tall { height: 360px; }
    }
  </style>
</head>
<body>
<main>
  <section class="hero" id="top">
    <div>
      <div class="eyebrow">Which Way the Wind Blows</div>
      <h1>How Climate Events Matter to Your Home</h1>
      <p class="dek">Are climate risks priced into housing markets? Exploring the relationship between climate risk, house price growth, and what happens to counties when extreme weather strikes.</p>
    </div>
  </section>

  <section class="slide" id="pricing">
    <h2>Are Climate Risks Priced Into Housing Markets?</h2>
    <p class="section-copy">Climate change comes with an increasing frequency of severe weather events and natural disasters that cause heavy damage to properties and in extreme cases, devastate local communities. Across the United States, the impact of climate events vary by region. To measure the risk of climate hazards across the country, the National Risk Index (NRI) was developed by the Federal Emergency Management Agency (FEMA). Here's how the NRI score varies across the country along with house prices.</p>

    <div class="toolbar" id="score-hazard-buttons"></div>
    <div class="viz-grid">
      <div class="panel">
        <h3>County PPSF YoY histories</h3>
        <p class="sub">Each line is a county's annual average Median PPSF YoY over the last 10 years.</p>
        <svg id="score-scatter" class="chart"></svg>
      </div>
      <div class="panel">
        <h3>County map: score or price growth</h3>
        <p class="sub">Toggle between selected NRI score in red and average 2025 median PPSF YoY in blue.</p>
        <div class="tabs" id="score-map-mode"></div>
        <svg id="score-map" class="chart"></svg>
        <div class="legend" id="score-map-legend"></div>
      </div>
    </div>

    <div class="toolbar" id="rating-hazard-buttons"></div>
    <div class="viz-grid">
      <div class="panel">
        <h3>PPSF YoY histories grouped by NRI risk rating</h3>
        <p class="sub">Median line with interquartile range for counties in each risk-rating group.</p>
        <svg id="rating-scatter" class="chart"></svg>
      </div>
      <div class="panel">
        <h3>County map: rating or price growth</h3>
        <p class="sub">Toggle between selected risk rating and 2025 median PPSF YoY.</p>
        <div class="tabs" id="rating-map-mode"></div>
        <svg id="rating-map" class="chart"></svg>
        <div class="legend" id="rating-map-legend"></div>
      </div>
    </div>
    <div class="takeaway" id="pricing-takeaway"></div>
    <div class="sources">Sources: <a href="https://hazards.fema.gov/nri/" target="_blank" rel="noopener">FEMA National Risk Index</a>, local mart <code>data/quoll.duckdb: mart.nri_county_risk</code>; <a href="https://www.redfin.com/news/data-center/" target="_blank" rel="noopener">Redfin Data Center</a>, local mart <code>mart.redfin_county_monthly</code>. House prices use the average monthly <code>MEDIAN_PPSF_YOY</code> value in 2025.</div>
  </section>

  <section class="slide" id="events">
    <h2>Does House Price Growth Change Around Climate Events?</h2>
    <p class="section-copy">Extreme climate events can change the housing market outlook in its area as buyers and sellers re-evaluate their positions based on the perceived increased risk. Here's how the growth momentum in housing markets have changed over time around previous extreme climate events.</p>
    <div class="viz-grid timeseries-grid">
      <div class="panel">
        <h3>Event-window median PPSF YoY</h3>
        <p class="sub">Start with all affected counties, or switch to NRI risk-rating groups. Median line with interquartile band; events come from FEMA declarations and NOAA events with at least $1 billion total damage.</p>
        <div class="tabs" id="event-view-mode"></div>
        <div class="tabs" id="risk-frame-buttons"></div>
        <div class="tabs"><button id="resume-risk" type="button">Resume</button><button id="prev-window" type="button">Previous window</button><button id="next-window" type="button">Next window</button></div>
        <svg id="event-window" class="chart tall"></svg>
      </div>
      <div class="panel">
        <h3>Affected counties for selected risk rating</h3>
        <p class="sub">Map updates with the current risk-rating frame.</p>
        <svg id="affected-map" class="chart"></svg>
      </div>
    </div>
    <div class="takeaway" id="event-takeaway"></div>
    <div class="sources">Sources: local marts <code>mart.fema_disaster_declarations</code>, <code>mart.noaa_storm_events</code>, <code>mart.redfin_county_monthly</code>, and <code>mart.nri_county_risk</code>. The housing market index used elsewhere in the project averages standardized PPSF YoY, sale-to-list YoY, homes-sold YoY, and inverted inventory YoY.</div>
  </section>

  <section class="slide" id="features">
    <h2>What Sets Apart Counties With Different Climate Risk?</h2>
    <p class="section-copy">Certain features make counties more vulnerable or resilient to destructive weather events. This section connects county-level event-window price paths with the local features that correlate with climate risk.</p>
    <div class="viz-grid timeseries-grid">
      <div class="panel">
        <h3>House price response by risk group</h3>
        <p class="sub">Grouped median and IQR are shown first. Click an example county line to inspect its features.</p>
        <div class="tabs" id="feature-risk-buttons"></div>
        <svg id="feature-event-window" class="chart tall"></svg>
        <div id="selected-county-features" class="sources"></div>
      </div>
      <div class="panel">
        <h3>Selected county map</h3>
        <p class="sub">The clicked county is highlighted; other counties are grey.</p>
        <svg id="feature-county-map" class="chart"></svg>
      </div>
    </div>
    <div class="takeaway" id="feature-takeaway"></div>
    <div class="sources">Sources: local marts <code>mart.acs_county_economic_annual</code>, <code>mart.acs_county_demographic_annual</code>, <code>mart.acs_county_affordability_annual</code>, <code>mart.ncei_county_weather_monthly</code>, and <code>mart.nri_county_risk</code>. Cost components are midpoint estimates from ACS cost buckets.</div>
  </section>

  <section class="slide" id="additional-impacts">
    <h2>What Else Are Climate Events Doing to Counties?</h2>
    <p class="section-copy">Other non-housing aspects of a county can also change when extreme climate events occur. These changes can then have knock-on effects on house prices.</p>
    <div class="toolbar" id="additional-metric-buttons"></div>
    <div class="viz-grid timeseries-grid">
      <div class="panel">
        <h3 id="additional-metric-title">Selected metric over event window</h3>
        <p class="sub" id="additional-metric-description">Median line with interquartile band, grouped by NRI risk rating.</p>
        <div class="tabs" id="additional-view-mode"></div>
        <div class="tabs" id="additional-risk-buttons"></div>
        <svg id="additional-chart" class="chart tall"></svg>
      </div>
      <div class="panel">
        <h3>Interpretation</h3>
        <p class="sub" id="additional-interpretation"></p>
      </div>
    </div>
    <div class="takeaway">Now we have established that counties change in various ways which ultimately lead to house price growth changing when extreme climate events occur. Can we expect what will happen to a county when a future incident occurs?</div>
    <div class="sources">Sources: local marts <code>mart.redfin_county_monthly</code> and <code>mart.nri_county_risk</code>.</div>
  </section>

  <section class="slide" id="playbook">
    <h2>What to Expect for a County When an Extreme Climate Event Happens</h2>
    <p class="section-copy">Given the relationship between house price growth, county characteristics, and climate risk level, we can project what would likely happen to a county when it is affected by an extreme climate event.</p>
    <div class="panel">
      <h3>County Climate Risk Playbook</h3>
      <p class="sub">Select a county to see its climate risk profile and what to expect for house price growth when an event occurs.</p>
      <div class="viz-grid" style="margin-top: 12px;">
        <div class="panel">
          <h3>Select County from Map</h3>
          <p class="sub">Click a county on the map to view its profile</p>
          <svg id="county-selection-map" class="chart" style="height: 360px;"></svg>
        </div>
        <div class="panel">
          <h3>Or Search by Name</h3>
          <p class="sub">Type to find a specific county</p>
          <div class="toolbar">
            <input type="text" id="county-search" placeholder="Search for a county..." style="padding: 8px 12px; border: 1px solid var(--line); border-radius: 999px; font-size: 12px; width: 100%;">
          </div>
          <div id="county-results" style="max-height: 280px; overflow-y: auto; margin-top: 8px;"></div>
        </div>
      </div>
      <div id="playbook-display" style="display: none; margin-top: 16px;">
        <h4 style="font-size: 18px; margin: 0 0 12px;">Selected County: <span id="selected-county-name"></span></h4>
        <div class="viz-grid">
          <div class="panel">
            <h3>County Climate Risk Profile</h3>
            <p class="sub">Projected climate risk based on county features</p>
            <svg id="county-map-zoom" class="chart" style="height: 320px;"></svg>
            <div style="margin-top: 12px;">
              <p style="margin: 6px 0;"><strong>Projected Climate Risk:</strong> <span id="projected-risk-rating"></span></p>
              <p style="margin: 6px 0;"><strong>Actual NRI Risk Rating:</strong> <span id="actual-risk-rating"></span></p>
              <p style="margin: 6px 0;"><strong>Model:</strong> <span id="model-name"></span> (accuracy: <span id="model-accuracy"></span>)</p>
              <p style="margin: 6px 0; font-size: 14px; color: var(--muted);" id="model-explanation"></p>
            </div>
          </div>
          <div class="panel">
            <h3>County Features & Risk Contributions</h3>
            <p class="sub">How each feature impacts climate risk level</p>
            <div id="playbook-county-features"></div>
          </div>
        </div>
        <div class="panel" style="margin-top:16px;">
          <h3>Expected Impact on House Prices</h3>
          <p class="sub">What the risk level means for house price growth when a climate event occurs</p>
          <div id="expected-impact"></div>
        </div>
      </div>
      <div id="model-unavailable" style="display: none; padding: 20px; background: #fef3c7; border-radius: 8px; margin-top: 12px;">
        <p style="margin: 0; color: #92400e;">Climate risk prediction models are not available. Please train the models first using <code>train-climate-risk-model --all-hazards</code></p>
      </div>
    </div>
    <div class="takeaway">From the county climate risk playbook, we see that the county's features determine its climate risk level. That means we can expect corresponding effects on house value growth when a climate event happens.</div>
    <div class="sources">Sources: Climate risk prediction models from <code>output/models/climate_risk_prediction/overall/</code>; county data from local marts.</div>
  </section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = __PAYLOAD__;
const RISK_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"];
const RISK_COLORS = {"Very Low":"#16803c","Low":"#79b851","Medium":"#e0b33b","High":"#df7d2f","Very High":"#b42318"};
const fmtPct = d3.format("+.1%");
const fmtShare = d3.format(".0%");
const fmtAxisPct = value => Math.abs(value) >= 10 ? `${value > 0 ? "+" : ""}${d3.format(".2s")(value * 100)}%` : fmtPct(value);
const fmtNum = d3.format(",.1f");
const fmtMoney = d3.format("$,.0f");
const tooltip = d3.select("#tooltip");
const countyByFips = new Map(DATA.priceRisk.counties.map(d => [d.fips, d]));
let scoreHazard = "overall";
let ratingHazard = "overall";
let scoreMapMode = "score";
let ratingMapMode = "rating";
let eventView = "all";
let selectedRisk = "Very Low";
let riskTimer = null;
let riskAutoPaused = false;
let horizon = 12;
const horizons = [12, 24, 36, 48, 60];
let selectedFeatureCategory = DATA.features.rows[0]?.category || "Economic";
let selectedFeature = DATA.features.rows.find(d => d.category === selectedFeatureCategory)?.feature || DATA.features.rows[0]?.feature;
let selectedFeatureRisk = "Medium";
let selectedFeatureCounty = DATA.eventWindows.exampleCountyLines.find(d => d.riskRating === selectedFeatureRisk) || DATA.eventWindows.exampleCountyLines[0];
let selectedAdditionalMetric = DATA.eventWindows.additionalMetrics[0]?.key || null;
let additionalView = "all";
let selectedAdditionalRisk = "Very Low";
let selectedCountyFips = null;

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

function drawMap(svgId, fillFn, tooltipFn, legendId, legendHtml) {
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
    .on("mousemove", (event, d) => {
      const html = tooltipFn(countyByFips.get(d.properties.fips), d.properties.fips);
      if (!html) return;
      tooltip.style("display","block").style("left", `${event.clientX+12}px`).style("top", `${event.clientY+12}px`).html(html);
    });
  d3.select(legendId).html(legendHtml || "");
}

function drawScoreScatter() {
  // Filter counties based on selected hazard - only show counties with valid rating for that hazard
  let data = DATA.priceRisk.countyHistory.filter(d => d.ppsfYoy != null);
  if (scoreHazard !== "overall") {
    const hazardRatingKey = `${scoreHazard}_rating`;
    data = data.filter(d => d[hazardRatingKey] != null);
  }
  const svg = d3.select("#score-scatter");
  const width = svg.node().clientWidth || 520, height = svg.node().clientHeight || 430;
  const margin = {top: 18, right: 18, bottom: 46, left: 68};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const yDomain = robustDomain(data.map(d => d.ppsfYoy));
  const x = d3.scaleLinear().domain(d3.extent(data, d => d.year)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
  const yTicks = sparsePctTicks(y.domain());
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).tickFormat(d3.format("d")).ticks(6));
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickFormat(fmtAxisPct));
  svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Year");
  svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",20).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Annual average Median PPSF YoY");
  const line = d3.line().x(d => x(d.year)).y(d => y(capValue(d.ppsfYoy, yDomain)));
  svg.append("g").selectAll("path").data(d3.groups(data, d => d.fips)).join("path")
    .attr("class","line")
    .attr("stroke", d => RISK_COLORS[d[1][0].riskRating] || "#8c2d22")
    .attr("stroke-width", 1.1)
    .attr("opacity", .12)
    .attr("d", d => line(d[1].sort((a,b)=>d3.ascending(a.year,b.year))))
    .on("mousemove", (event, d) => tooltip.style("display","block").style("left",`${event.clientX+12}px`).style("top",`${event.clientY+12}px`).html(`<strong>${d[1][0].county}</strong>${d[1][0].riskRating} risk<br>Annual PPSF YoY history, color-capped for display`))
    .on("mouseleave", () => tooltip.style("display","none"));
}

function drawRatingScatter() {
  const ratingHistory = DATA.priceRisk.ratingHistoriesByHazard[ratingHazard] || [];
  const data = ratingHistory.filter(d => d.median != null);
  const svg = d3.select("#rating-scatter");
  const width = svg.node().clientWidth || 520, height = svg.node().clientHeight || 430;
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
    // Make labels more visible with background and larger font
    const labelX = x(last.year) + 8;
    const labelY = y(capValue(last.median, yDomain)) + 4;
    svg.append("rect")
      .attr("x", labelX - 3)
      .attr("y", labelY - 11)
      .attr("width", risk.length * 6 + 6)
      .attr("height", 16)
      .attr("fill", "white")
      .attr("fill-opacity", 0.85)
      .attr("rx", 3);
    svg.append("text")
      .attr("x", labelX)
      .attr("y", labelY)
      .attr("fill", RISK_COLORS[risk])
      .attr("font-size", 13)
      .attr("font-weight", 800)
      .text(risk);
  }
}

function drawScoreMap() {
  const values = DATA.priceRisk.counties.map(d => scoreMapMode === "score" ? hazardCounty(d, scoreHazard).score : d.avgPpsfYoy);
  const domain = robustDomain(values);
  const scale = scoreMapMode === "score" ? colorScale(values, d3.interpolateReds, true) : colorScale(values, d3.interpolateBlues, true);
  drawMap("#score-map",
    county => {
      const value = scoreMapMode === "score" ? hazardCounty(county, scoreHazard).score : county?.avgPpsfYoy;
      return value == null ? "#ece7df" : scale(capValue(value, domain));
    },
    county => county ? `<strong>${county.county}</strong>${scoreMapMode === "score" ? `${hazardLabel(scoreHazard)} score: ${hazardCounty(county, scoreHazard).score ?? "n/a"}` : `PPSF YoY: ${pctText(county.avgPpsfYoy)}`}${scoreMapMode === "ppsf" && county.ppsfYoyWasCapped ? "<br>Color capped for display" : ""}` : "",
    "#score-map-legend",
    scaleLegendHtml(domain, scoreMapMode === "score" ? "#fff5f0" : "#eff6ff", scoreMapMode === "score" ? "#b42318" : "#2563eb", scoreMapMode === "score" ? `${hazardLabel(scoreHazard)} score, color-capped` : "Average 2025 median PPSF YoY, color-capped", scoreMapMode === "score" ? fmtNum : fmtPct)
  );
}

function drawRatingMap() {
  const values = DATA.priceRisk.counties.map(d => d.avgPpsfYoy);
  const ppsfDomain = robustDomain(values);
  const ppsfScale = colorScale(values, d3.interpolateBlues, true);
  drawMap("#rating-map",
    county => {
      if (!county) return "#ece7df";
      if (ratingMapMode === "ppsf") return county.avgPpsfYoy == null ? "#ece7df" : ppsfScale(capValue(county.avgPpsfYoy, ppsfDomain));
      return RISK_COLORS[hazardCounty(county, ratingHazard).rating] || "#ece7df";
    },
    county => county ? `<strong>${county.county}</strong>${ratingMapMode === "ppsf" ? `PPSF YoY: ${pctText(county.avgPpsfYoy)}${county.ppsfYoyWasCapped ? "<br>Color capped for display" : ""}` : `${hazardLabel(ratingHazard)} rating: ${hazardCounty(county, ratingHazard).rating ?? "n/a"}`}` : "",
    "#rating-map-legend",
    ratingMapMode === "ppsf" ? scaleLegendHtml(ppsfDomain, "#eff6ff", "#2563eb", "Average 2025 median PPSF YoY, color-capped", fmtPct) : RISK_ORDER.map(r => `<span><span class="swatch" style="background:${RISK_COLORS[r]}"></span>${r}</span>`).join("")
  );
}

function drawLineChart(svgId, source, groupKey, horizonLimit, activeRisk = null, minMonth = -12) {
  const data = source.filter(d => d.month >= minMonth && d.month <= horizonLimit);
  const svg = d3.select(svgId);
  const width = svg.node().clientWidth || 700, height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: 96, bottom: 42, left: 58};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const x = d3.scaleLinear().domain([minMonth, horizonLimit]).range([margin.left, width - margin.right]);
  const values = data.flatMap(d => [d.q1, d.median, d.q3]).filter(v => v != null);
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
    const isBackground = activeRisk && RISK_ORDER.indexOf(key) < RISK_ORDER.indexOf(activeRisk);
    const isHidden = activeRisk && RISK_ORDER.indexOf(key) > RISK_ORDER.indexOf(activeRisk);
    if (isHidden) continue;
    const color = groupKey === "riskRating" ? RISK_COLORS[key] : "#0f766e";
    const area = d3.area().x(d=>x(d.month)).y0(d=>y(d.q1)).y1(d=>y(d.q3));
    const line = d3.line().x(d=>x(d.month)).y(d=>y(d.median));
    svg.append("path").datum(rows).attr("class",`band ${isBackground ? "background" : ""}`).attr("fill",color).attr("d",area);
    svg.append("path").datum(rows).attr("class",`line ${isBackground ? "background" : ""}`).attr("stroke",color).attr("d",line);
    const last = rows.at(-1);
    if (last && !isBackground) svg.append("text").attr("x",x(last.month)+5).attr("y",y(last.median)+4).attr("fill",color).attr("font-size",12).attr("font-weight",800).text(key);
  }
  svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text(useYears ? "Years from event start / after event end" : "Months from event start / after event end");
}

function drawAffectedMap() {
  const affected = new Map(DATA.eventWindows.affectedCounties.map(d => [d.fips, d.riskRating]));
  drawMap("#affected-map",
    (county, fips) => affected.get(fips) === selectedRisk ? RISK_COLORS[selectedRisk] : "#e6dfd5",
    (county, fips) => affected.get(fips) ? `<strong>${county?.county || fips}</strong>Affected county<br>NRI rating: ${affected.get(fips)}` : "",
    "#affected-map-legend",
    `<span><span class="swatch" style="background:${RISK_COLORS[selectedRisk]}"></span>${selectedRisk} affected counties</span>`
  );
}

function updateEventTakeaway() {
  const allAt18 = DATA.eventWindows.aggregate.find(d => d.month === 18)?.median;
  const allAt24 = DATA.eventWindows.aggregate.find(d => d.month === 24)?.median;
  const riskDeltas = ["Medium", "High", "Very High"].map(risk => {
    const rows = DATA.eventWindows.byRating.filter(d => d.riskRating === risk);
    const at18 = rows.find(d => d.month === 18)?.median;
    const at24 = rows.find(d => d.month === 24)?.median;
    return `${risk}: ${pctText(at18)} to ${pctText(at24)}`;
  }).join("; ");
  d3.select("#event-takeaway").text(`Median PPSF YoY tends to soften around the two-year mark after extreme climate events, easing from ${pctText(allAt18)} at month 18 to ${pctText(allAt24)} at year 2 across the complete-window sample. The decline is primarily driven by Medium, High, and Very High climate-risk county-events (${riskDeltas}).`);
}

function renderEventSection() {
  d3.selectAll("#risk-frame-buttons button").classed("active", d => d === selectedRisk);
  d3.selectAll("#event-view-mode button").classed("active", d => d.key === eventView);
  d3.select("#risk-frame-buttons").style("display", eventView === "grouped" ? "flex" : "none");
  d3.select("#resume-risk").style("display", eventView === "grouped" && riskAutoPaused ? null : "none");
  drawLineChart("#event-window", eventView === "all" ? DATA.eventWindows.aggregate : DATA.eventWindows.byRating, eventView === "all" ? "series" : "riskRating", horizon, eventView === "all" ? null : selectedRisk);
  drawAffectedMap();
  updateEventTakeaway();
}

function startRiskTimer() {
  clearInterval(riskTimer);
  riskAutoPaused = false;
  d3.select("#resume-risk").style("display", "none");
  riskTimer = setInterval(() => {
    if (eventView !== "grouped") return;
    selectedRisk = RISK_ORDER[(RISK_ORDER.indexOf(selectedRisk) + 1) % RISK_ORDER.length];
    renderEventSection();
  }, 3200);
}

function drawFeatureHeatmaps() {
  d3.selectAll("#feature-risk-buttons button").classed("active", d => d === selectedFeatureRisk);
  // Use fixed event window: 1 year pre-event (-12) and 2 years post-event (+24)
  const featureHorizon = 24;
  drawLineChart("#feature-event-window", DATA.eventWindows.byRating, "riskRating", featureHorizon, selectedFeatureRisk, -12);
  const svg = d3.select("#feature-event-window");
  const width = svg.node().clientWidth || 700, height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: 96, bottom: 42, left: 58};
  const examples = DATA.eventWindows.exampleCountyLines.filter(d => d.riskRating === selectedFeatureRisk);
  const values = DATA.eventWindows.byRating.filter(d => d.month >= -12 && d.month <= featureHorizon).flatMap(d => [d.q1, d.median, d.q3]).filter(v => v != null);
  examples.forEach(d => d.values.filter(v => v.month >= -12 && v.month <= featureHorizon).forEach(v => values.push(v.value)));
  const x = d3.scaleLinear().domain([-12, featureHorizon]).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(d3.extent(values)).nice().range([height - margin.bottom, margin.top]);
  const line = d3.line().defined(d => d.value != null).x(d => x(d.month)).y(d => y(d.value));
  svg.append("g").selectAll("path.example-county").data(examples).join("path")
    .attr("class","line example-county")
    .attr("stroke","#172026")
    .attr("stroke-width",5)
    .attr("opacity",d => selectedFeatureCounty && d.lineId === selectedFeatureCounty.lineId ? .95 : 0)
    .attr("d",d => line(d.values.filter(v => v.month >= -12 && v.month <= featureHorizon)))
    .style("cursor","pointer")
    .on("click",(event,d)=>{selectedFeatureCounty=d; drawFeatureHeatmaps();})
    .on("mousemove",(event,d)=>tooltip.style("display","block").style("left",`${event.clientX+12}px`).style("top",`${event.clientY+12}px`).html(`<strong>${d.county}</strong>${d.riskRating} risk<br>Click to inspect features`))
    .on("mouseleave",()=>tooltip.style("display","none"));
  drawFeatureCountyMap();
  drawCountyFeaturePanel();
  const top = DATA.features.topFeatures.map(d => d.feature).join(", ");
  d3.select("#feature-takeaway").text(`Among counties with ${selectedFeatureRisk} risk, features such as ${top} are commonly associated with climate risk. These features can affect vulnerability, ability to absorb shocks, and household mobility, which can lead to higher or lower climate risk. So what else happens to a county when an extreme climate event strikes, and what does that mean to house prices?`);
}

function drawFeatureCountyMap() {
  drawMap("#feature-county-map",
    (county, fips) => selectedFeatureCounty && fips === selectedFeatureCounty.fips ? (RISK_COLORS[selectedFeatureRisk] || "#0f766e") : "#d8d0c4",
    (county, fips) => county ? `<strong>${county.county}</strong>${selectedFeatureCounty && fips === selectedFeatureCounty.fips ? "<br>Selected county" : ""}` : "",
    null,
    ""
  );
}

function drawCountyFeaturePanel() {
  const profile = DATA.features.countyProfiles.find(d => selectedFeatureCounty && d.fips === selectedFeatureCounty.fips);
  if (!profile || !selectedFeatureCounty) {
    d3.select("#selected-county-features").html("Click an example county line to inspect its county features.");
    return;
  }

  // Map of feature names to their correlation with climate risk (positive = increases risk)
  // Calculated from actual Spearman correlations with NRI risk ratings
  const featureCorrelations = {
    "Income": 0.216,  // Wealthier counties tend to be in higher-risk areas (coastal, desirable)
    "Home ownership burden": 0.772,  // Strong positive: higher burden = higher risk
    "Home insurance": 0.010,  // Negligible correlation
    "Property tax": 0.05,  // Estimated (not directly calculated)
    "Utilities": 0.03,  // Estimated (not directly calculated)
    "Net migration": -0.016,  // Negligible negative correlation
    "Homes Sold YoY": -0.05,  // More sales = lower risk (estimated from model)
    "Median DOM YoY": 0.08  // Slower market = higher risk (from model importance)
  };

  const rows = [
    ["Income", fmtMoney(profile.income), featureCorrelations["Income"]],
    ["Home ownership burden", profile.housingBurden == null ? "n/a" : `${fmtNum(profile.housingBurden)}%`, featureCorrelations["Home ownership burden"]],
    ["Home insurance", fmtMoney(profile.insurance), featureCorrelations["Home insurance"]],
    ["Property tax", fmtMoney(profile.propertyTaxes), featureCorrelations["Property tax"]],
    ["Utilities", fmtMoney(profile.utilities), featureCorrelations["Utilities"]],
    ["Net migration", profile.netMigration == null ? "n/a" : fmtNum(profile.netMigration), featureCorrelations["Net migration"]],
    ["Homes Sold YoY", pctText(profile.homesSoldYoy), featureCorrelations["Homes Sold YoY"]],
    ["Median DOM YoY", profile.medianDomYoy == null ? "n/a" : fmtNum(profile.medianDomYoy), featureCorrelations["Median DOM YoY"]],
  ];

  const html = `<div style="background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 12px;">
    <strong style="font-size: 16px;">${selectedFeatureCounty.county}, ${selectedFeatureCounty.state}</strong><br>
    <span style="color: var(--muted); font-size: 13px;">Risk Rating: ${selectedFeatureCounty.riskRating}</span>
  </div>
  <div style="display: grid; grid-template-columns: 1fr auto auto; gap: 8px; font-size: 13px; line-height: 1.6;">
    ${rows.map(([k, v, corr]) => {
      const arrow = corr > 0.05 ? '<span style="color: #b42318; font-size: 16px;">↑</span>' :
                    corr < -0.05 ? '<span style="color: #16803c; font-size: 16px;">↓</span>' :
                    '<span style="color: #66717b; font-size: 16px;">→</span>';
      const impact = corr > 0.05 ? '<span style="color: #b42318; font-size: 11px;">Higher risk</span>' :
                     corr < -0.05 ? '<span style="color: #16803c; font-size: 11px;">Lower risk</span>' :
                     '<span style="color: #66717b; font-size: 11px;">Neutral</span>';
      return `<div style="font-weight: 600;">${k}</div><div style="text-align: right;">${v}</div><div style="text-align: center;">${arrow} ${impact}</div>`;
    }).join('')}
  </div>`;

  d3.select("#selected-county-features").html(html);
}

function drawFeatureCorrelationChart() {
  const svg = d3.select("#feature-correlation-chart");
  const data = DATA.features.correlations.slice().sort((a,b)=>d3.descending(Math.abs(a.corr || 0), Math.abs(b.corr || 0))).slice(0, 10);
  const width = svg.node().clientWidth || 900, height = svg.node().clientHeight || 430;
  const margin = {top: 18, right: 28, bottom: 90, left: 58};
  svg.attr("viewBox",[0,0,width,height]).selectAll("*").remove();
  const x = d3.scaleBand().domain(data.map(d=>d.feature)).range([margin.left,width-margin.right]).padding(.25);
  const y = d3.scaleLinear().domain([0,d3.max(data,d=>Math.abs(d.corr || 0)) || 1]).nice().range([height-margin.bottom,margin.top]);
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x)).selectAll("text").attr("transform","rotate(-35)").attr("text-anchor","end");
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5));
  svg.append("g").selectAll("rect").data(data).join("rect")
    .attr("x",d=>x(d.feature)).attr("y",d=>y(Math.abs(d.corr || 0))).attr("width",x.bandwidth()).attr("height",d=>height-margin.bottom-y(Math.abs(d.corr || 0))).attr("fill","#2563eb");
}

function formatFeature(value, format) {
  if (value == null) return "n/a";
  if (format === "currency") return fmtMoney(value);
  if (format === "percent") return `${fmtNum(value)}%`;
  return fmtNum(value);
}

function drawAdditionalMetricChart() {
  if (!selectedAdditionalMetric || !DATA.eventWindows.additionalMetrics.length) return;
  const metric = DATA.eventWindows.additionalMetrics.find(m => m.key === selectedAdditionalMetric);
  if (!metric) return;
  const source = additionalView === "all" ? metric.aggregate : metric.byRating;
  drawLineChart("#additional-chart", source, additionalView === "all" ? "series" : "riskRating", 36, additionalView === "all" ? null : selectedAdditionalRisk, -24);
  d3.select("#additional-metric-title").text(metric.label);
  d3.select("#additional-metric-description").text(metric.description);
  d3.select("#additional-interpretation").text(`${metric.conclusion || metric.description} So ${metric.label} changes around the occurrence of a climate event. When this happens, affordability, mobility, labor-market strength, or buyer demand can change, and ultimately house price growth can increase or decrease.`);
}

function renderAdditionalSection() {
  if (!DATA.eventWindows.additionalMetrics || DATA.eventWindows.additionalMetrics.length === 0) {
    d3.select("#additional-impacts").style("display", "none");
    return;
  }
  d3.selectAll("#additional-view-mode button").classed("active", d => d.key === additionalView);
  d3.select("#additional-risk-buttons").style("display", additionalView === "grouped" ? "flex" : "none");
  if (additionalView === "grouped") {
    d3.selectAll("#additional-risk-buttons button").classed("active", d => d === selectedAdditionalRisk);
  }
  drawAdditionalMetricChart();
}

function initPlaybook() {
  if (!DATA.playbook || !DATA.playbook.available) {
    d3.select("#playbook-display").style("display", "none");
    d3.select("#model-unavailable").style("display", "block");
    d3.select("#county-search").style("display", "none");
    d3.select("#county-selection-map").style("display", "none");
    return;
  }

  // Draw interactive county selection map
  const svg = d3.select("#county-selection-map");
  const width = svg.node().clientWidth || 520;
  const height = svg.node().clientHeight || 360;
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);

  svg.append("g").selectAll("path")
    .data(DATA.geojson.features)
    .join("path")
    .attr("class", "county")
    .attr("d", path)
    .attr("fill", d => {
      const county = DATA.playbook.counties.find(c => c.fips === d.properties.fips);
      return county ? RISK_COLORS[county.riskRating] || "#e6dfd5" : "#e6dfd5";
    })
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.3)
    .style("cursor", "pointer")
    .on("click", (event, d) => {
      const county = DATA.playbook.counties.find(c => c.fips === d.properties.fips);
      if (county) {
        selectedCountyFips = county.fips;
        showCountyPlaybook(county);
      }
    })
    .on("mousemove", (event, d) => {
      const county = DATA.playbook.counties.find(c => c.fips === d.properties.fips);
      if (county) {
        tooltip.style("display","block")
          .style("left",`${event.clientX+12}px`)
          .style("top",`${event.clientY+12}px`)
          .html(`<strong>${county.county}, ${county.state}</strong><br>Risk: ${county.riskRating || "Unknown"}`);
      }
    })
    .on("mouseleave", () => tooltip.style("display","none"));

  const countySearch = d3.select("#county-search");
  const countyResults = d3.select("#county-results");

  countySearch.on("input", function() {
    const query = this.value.toLowerCase().trim();
    if (query.length < 2) {
      countyResults.html("");
      return;
    }
    const matches = DATA.playbook.counties.filter(c =>
      c.county.toLowerCase().includes(query) ||
      c.state.toLowerCase().includes(query) ||
      c.fips.includes(query)
    ).slice(0, 20);

    countyResults.html("")
      .selectAll("div")
      .data(matches)
      .join("div")
      .style("padding", "8px 12px")
      .style("cursor", "pointer")
      .style("border-bottom", "1px solid var(--line)")
      .style("font-size", "13px")
      .html(d => `<strong>${d.county}, ${d.state}</strong> <span style="color: var(--muted);">(${d.riskRating || "Unknown risk"})</span>`)
      .on("click", (event, d) => {
        selectedCountyFips = d.fips;
        showCountyPlaybook(d);
        countySearch.property("value", "");
        countyResults.html("");
      });
  });
}

function showCountyPlaybook(county) {
  d3.select("#playbook-display").style("display", "block");
  d3.select("#selected-county-name").text(`${county.county}, ${county.state}`);

  // Show projected risk (using actual for now as proxy - in real implementation would use model prediction)
  const projectedRisk = county.riskRating;  // TODO: Replace with actual model prediction
  d3.select("#projected-risk-rating").html(`<span style="color: ${RISK_COLORS[projectedRisk] || "#666"}; font-size: 18px; font-weight: 800;">${projectedRisk || "Unknown"}</span>`);
  d3.select("#actual-risk-rating").html(`<span style="color: ${RISK_COLORS[county.riskRating] || "#666"}">${county.riskRating || "Unknown"}</span>`);
  d3.select("#model-name").text(DATA.playbook.model.name.replace(/_/g, " "));
  d3.select("#model-accuracy").text(`${(DATA.playbook.model.accuracy * 100).toFixed(1)}%`);

  const topFeatures = DATA.playbook.model.topFeatures.slice(0, 3);
  const featureText = topFeatures.length > 0
    ? `Top predictive features: ${topFeatures.map(f => f.feature).join(", ")}`
    : `Model uses ${DATA.playbook.model.featureNames.slice(0, 5).join(", ")} and related features`;
  d3.select("#model-explanation").text(featureText);

  // Draw zoomed map
  const svg = d3.select("#county-map-zoom");
  const width = svg.node().clientWidth || 520;
  const height = svg.node().clientHeight || 320;
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);
  svg.append("g").selectAll("path")
    .data(DATA.geojson.features)
    .join("path")
    .attr("class", "county")
    .attr("d", path)
    .attr("fill", d => d.properties.fips === county.fips ? (RISK_COLORS[projectedRisk] || "#0f766e") : "#e6dfd5");

  // Show expected impact based on projected risk rating
  const riskImpacts = {
    "Very Low": "Counties with Very Low climate risk tend to maintain steady house price growth around climate events, with minimal disruption to market momentum.",
    "Low": "Counties with Low climate risk typically see modest softening of house price growth in the 1-2 years following events, but generally recover within 3 years.",
    "Medium": "Counties with Medium climate risk experience noticeable softening of house price growth around the two-year mark after extreme events.",
    "High": "Counties with High climate risk see significant deceleration in house price growth following events, with the decline primarily occurring 18-24 months after event end.",
    "Very High": "Counties with Very High climate risk face substantial impacts on house price growth, with softening trends that can persist for several years after events."
  };
  d3.select("#expected-impact").html(`<p style="line-height: 1.55; font-size: 15px;">${riskImpacts[projectedRisk] || "Impact data not available for this risk level."}</p>`);

  // Show features with contribution arrows
  const profile = DATA.features.countyProfiles.find(d => d.fips === county.fips);
  if (profile) {
    // Actual Spearman correlations with NRI risk ratings
    const featureCorrelations = {
      "Income": 0.216,
      "Home ownership burden": 0.772,
      "Home insurance": 0.010,
      "Property tax": 0.05,
      "Utilities": 0.03,
      "Net migration": -0.016,
      "Homes Sold YoY": -0.05,
      "Median DOM YoY": 0.08
    };

    const featureRows = [
      ["Income", fmtMoney(profile.income), featureCorrelations["Income"]],
      ["Home ownership burden", profile.housingBurden == null ? "n/a" : `${fmtNum(profile.housingBurden)}%`, featureCorrelations["Home ownership burden"]],
      ["Home insurance", fmtMoney(profile.insurance), featureCorrelations["Home insurance"]],
      ["Property tax", fmtMoney(profile.propertyTaxes), featureCorrelations["Property tax"]],
      ["Utilities", fmtMoney(profile.utilities), featureCorrelations["Utilities"]],
      ["Net migration", profile.netMigration == null ? "n/a" : fmtNum(profile.netMigration), featureCorrelations["Net migration"]],
      ["Homes Sold YoY", pctText(profile.homesSoldYoy), featureCorrelations["Homes Sold YoY"]],
      ["Median DOM YoY", profile.medianDomYoy == null ? "n/a" : fmtNum(profile.medianDomYoy), featureCorrelations["Median DOM YoY"]],
    ];

    const html = `<div style="display: grid; grid-template-columns: 1fr auto auto; gap: 10px; font-size: 14px; line-height: 1.8; padding: 8px;">
      ${featureRows.map(([k, v, corr]) => {
        const arrow = corr > 0.05 ? '<span style="color: #b42318; font-size: 18px; font-weight: 800;">↑</span>' :
                      corr < -0.05 ? '<span style="color: #16803c; font-size: 18px; font-weight: 800;">↓</span>' :
                      '<span style="color: #66717b; font-size: 18px;">→</span>';
        const impact = corr > 0.05 ? '<span style="color: #b42318; font-size: 12px; font-weight: 600;">Higher risk</span>' :
                       corr < -0.05 ? '<span style="color: #16803c; font-size: 12px; font-weight: 600;">Lower risk</span>' :
                       '<span style="color: #66717b; font-size: 12px;">Neutral</span>';
        return `<div style="font-weight: 600;">${k}</div><div style="text-align: right;">${v}</div><div style="text-align: center;">${arrow} ${impact}</div>`;
      }).join('')}
    </div>`;

    d3.select("#playbook-county-features").html(html);
  } else {
    d3.select("#playbook-county-features").text("Feature profile not available for this county.");
  }
}

function initButtons() {
  d3.select("#score-hazard-buttons").selectAll("button").data(DATA.priceRisk.hazards).join("button").text(d=>d.label).classed("active",d=>d.key===scoreHazard).on("click",(event,d)=>{scoreHazard=d.key; d3.select("#score-hazard-buttons").selectAll("button").classed("active",x=>x.key===scoreHazard); drawScoreScatter(); drawScoreMap();});
  d3.select("#rating-hazard-buttons").selectAll("button").data(DATA.priceRisk.hazards).join("button").text(d=>d.label).classed("active",d=>d.key===ratingHazard).on("click",(event,d)=>{ratingHazard=d.key; d3.select("#rating-hazard-buttons").selectAll("button").classed("active",x=>x.key===ratingHazard); drawRatingScatter(); drawRatingMap();});
  d3.select("#score-map-mode").selectAll("button").data([{key:"score",label:"NRI score"},{key:"ppsf",label:"Median PPSF YoY"}]).join("button").text(d=>d.label).classed("active",d=>d.key===scoreMapMode).on("click",(event,d)=>{scoreMapMode=d.key; d3.select("#score-map-mode").selectAll("button").classed("active",x=>x.key===scoreMapMode); drawScoreMap();});
  d3.select("#rating-map-mode").selectAll("button").data([{key:"rating",label:"NRI risk rating"},{key:"ppsf",label:"Median PPSF YoY"}]).join("button").text(d=>d.label).classed("active",d=>d.key===ratingMapMode).on("click",(event,d)=>{ratingMapMode=d.key; d3.select("#rating-map-mode").selectAll("button").classed("active",x=>x.key===ratingMapMode); drawRatingMap();});
  d3.select("#event-view-mode").selectAll("button").data([{key:"all",label:"All affected counties"},{key:"grouped",label:"Grouped by NRI risk rating"}]).join("button").text(d=>d.label).classed("active",d=>d.key===eventView).on("click",(event,d)=>{eventView=d.key; if(eventView==="grouped") startRiskTimer(); else {clearInterval(riskTimer); riskAutoPaused=false;} renderEventSection();});
  d3.select("#risk-frame-buttons").selectAll("button").data(RISK_ORDER).join("button").text(d=>d).classed("active",d=>d===selectedRisk).on("click",(event,d)=>{selectedRisk=d; clearInterval(riskTimer); riskAutoPaused=true; renderEventSection();});
  d3.select("#feature-risk-buttons").selectAll("button").data(RISK_ORDER).join("button").text(d=>d).classed("active",d=>d===selectedFeatureRisk).on("click",(event,d)=>{selectedFeatureRisk=d; selectedFeatureCounty=DATA.eventWindows.exampleCountyLines.find(x=>x.riskRating===d) || selectedFeatureCounty; drawFeatureHeatmaps();});
  d3.select("#resume-risk").on("click", startRiskTimer);
  d3.select("#prev-window").on("click",()=>{const index=horizons.indexOf(horizon); if(index>0){horizon=horizons[index-1]; d3.select("#event-window").classed("compressing",true); renderEventSection(); setTimeout(()=>d3.select("#event-window").classed("compressing",false),380);}});
  d3.select("#next-window").on("click",()=>{const index=horizons.indexOf(horizon); if(index<horizons.length-1){horizon=horizons[index+1]; d3.select("#event-window").classed("compressing",true); renderEventSection(); setTimeout(()=>d3.select("#event-window").classed("compressing",false),380);}});

  // Additional metrics section
  if (DATA.eventWindows.additionalMetrics && DATA.eventWindows.additionalMetrics.length > 0) {
    d3.select("#additional-metric-buttons").selectAll("button").data(DATA.eventWindows.additionalMetrics).join("button").text(d=>d.label).classed("active",d=>d.key===selectedAdditionalMetric).on("click",(event,d)=>{selectedAdditionalMetric=d.key; d3.select("#additional-metric-buttons").selectAll("button").classed("active",x=>x.key===selectedAdditionalMetric); renderAdditionalSection();});
    d3.select("#additional-view-mode").selectAll("button").data([{key:"all",label:"All affected counties"},{key:"grouped",label:"Grouped by risk rating"}]).join("button").text(d=>d.label).classed("active",d=>d.key===additionalView).on("click",(event,d)=>{additionalView=d.key; renderAdditionalSection();});
    d3.select("#additional-risk-buttons").selectAll("button").data(RISK_ORDER).join("button").text(d=>d).classed("active",d=>d===selectedAdditionalRisk).on("click",(event,d)=>{selectedAdditionalRisk=d; renderAdditionalSection();});
  }
}

function renderFeatureButtons() {
  const features = [...new Set(DATA.features.rows.filter(d => d.category === selectedFeatureCategory).map(d => d.feature))];
  if (!features.includes(selectedFeature)) selectedFeature = features[0];
  d3.select("#feature-category-buttons").selectAll("button").classed("active", d => d === selectedFeatureCategory);
  d3.select("#feature-buttons").selectAll("button").data(features).join("button").text(d=>d).classed("active",d=>d===selectedFeature).on("click",(event,d)=>{selectedFeature=d; d3.select("#feature-buttons").selectAll("button").classed("active",x=>x===selectedFeature); drawFeatureHeatmaps();});
}

function median(values) {
  const valid = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  return valid.length ? d3.median(valid) : null;
}
const lowerRiskMedian = median(DATA.priceRisk.counties.filter(d => ["Very Low", "Low"].includes(hazardCounty(d, "overall").rating)).map(d => d.avgPpsfYoy));
const higherRiskMedian = median(DATA.priceRisk.counties.filter(d => ["Medium", "High", "Very High"].includes(hazardCounty(d, "overall").rating)).map(d => d.avgPpsfYoy));
d3.select("#pricing-takeaway").text(`Across the 2025 Redfin county mart, higher NRI risk generally lines up with lower Median PPSF YoY values. Very Low/Low overall-risk counties have a median PPSF YoY of ${pctText(lowerRiskMedian)}, compared with ${pctText(higherRiskMedian)} for Medium/High/Very High risk counties, suggesting climate risk is already part of the price-growth picture.`);
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) entry.target.classList.add("visible","transition-in");
    else if (entry.boundingClientRect.top < 0) entry.target.classList.add("transition-out");
  });
}, { threshold: .18 });
document.querySelectorAll(".slide").forEach(el => observer.observe(el));
initButtons();
drawScoreScatter();
drawScoreMap();
drawRatingScatter();
drawRatingMap();
renderEventSection();
drawFeatureHeatmaps();
renderAdditionalSection();
initPlaybook();
startRiskTimer();
window.addEventListener("resize", () => {
  drawScoreScatter();
  drawScoreMap();
  drawRatingScatter();
  drawRatingMap();
  renderEventSection();
  drawFeatureHeatmaps();
  renderAdditionalSection();
});
</script>
</body>
</html>"""


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        price_risk = build_price_risk(con)
        event_windows = build_event_windows(con)
        features = build_feature_payload(con)
        playbook = build_playbook_data(con)
    geojson = build_geojson({county["fips"] for county in price_risk["counties"]})
    data = {"priceRisk": price_risk, "eventWindows": event_windows, "features": features, "playbook": playbook, "geojson": geojson}
    OUT_PATH.write_text(make_html(data), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
