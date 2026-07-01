from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

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
    {"key": "hurricane", "label": "Hurricane", "score": "HRCN_RISKS", "rating": "HRCN_RISKR"},
    {"key": "river_flood", "label": "River flood", "score": "IFLD_RISKS", "rating": "IFLD_RISKR"},
    {"key": "wildfire", "label": "Wildfire", "score": "WFIR_RISKS", "rating": "WFIR_RISKR"},
    {"key": "tornado", "label": "Tornado", "score": "TRND_RISKS", "rating": "TRND_RISKR"},
    {"key": "winter_weather", "label": "Winter weather", "score": "WNTW_RISKS", "rating": "WNTW_RISKR"},
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
    return {
        "hazards": [{"key": h["key"], "label": h["label"]} for h in HAZARDS],
        "counties": counties,
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


def aggregate_lines(frame: pd.DataFrame, group_cols: list[str], metric: str) -> list[dict[str, object]]:
    if frame.empty:
        return []
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


def build_event_windows(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    events = load_disaster_events(con)
    events = events.loc[events["event_start_month"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-01"))].copy()
    housing = load_redfin_county_monthly(con)
    for column in ["median_ppsf_yoy", "avg_sale_to_list_yoy", "homes_sold_yoy", "inventory_yoy", "housing_market_index"]:
        if column in housing:
            housing.loc[pd.to_numeric(housing[column], errors="coerce").le(-888888000), column] = np.nan
    metric = "median_ppsf_yoy"
    affected = build_affected_event_windows(events, housing, pre_event_months=12, post_event_months=60)
    if affected.empty:
        return {"aggregate": [], "byRating": [], "affectedCounties": [], "summary": {"events": 0}}
    required = event_window_months(12, 60)
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
    return {
        "aggregate": aggregate,
        "byRating": by_rating,
        "affectedCounties": [
            {"fips": row.fips, "riskRating": row.riskRating}
            for row in affected_counties.itertuples(index=False)
        ],
        "summary": {
            "events": int(events["event_key"].nunique()),
            "countyEvents": int(complete["line_id"].nunique()),
            "riskCounts": {str(k): int(v) for k, v in risk_counts.items()},
        },
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


def build_feature_payload(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    nri = con.execute("SELECT fips, risk_rating FROM mart.nri_county_risk WHERE fips IS NOT NULL").df()
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
        "median_household_income",
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
        nri[["fips", "riskRating", "riskValue"]]
        .merge(econ[["fips", *econ_cols[2:]]], on="fips", how="left")
        .merge(demo[["fips", *demo_cols[2:]]], on="fips", how="left")
        .merge(afford[["fips", "median_owner_costs_mortgage", "median_household_income", "estimated_annual_home_insurance", "estimated_annual_property_tax", "estimated_annual_utilities"]], on="fips", how="left")
        .merge(weather, on="fips", how="left")
    )
    features["no_broadband_pct"] = 100 - features["dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct"]
    feature_defs = [
        ("Economic", "Income", "dp03_income_and_benefits_total_households_median_household_income_est", "currency", "mart.acs_county_economic_annual"),
        ("Economic", "Home ownership costs", "median_owner_costs_mortgage", "currency", "mart.acs_county_affordability_annual"),
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
    return {"riskOrder": RISK_ORDER, "rows": rows, "correlations": correlations, "topFeatures": top}


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
    .line { fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
    .line.background { opacity: .18; }
    .band { opacity: .18; }
    .band.background { opacity: .05; }
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
      <h1>How climate events matter to your home</h1>
      <p class="dek">Climate risk, local house price growth, and the market response around extreme events.</p>
    </div>
  </section>

  <section class="slide" id="pricing">
    <h2>Are climate risks priced into housing markets?</h2>
    <p class="section-copy">Climate change comes with an increasing frequency of severe weather events and natural disasters that can damage properties and disrupt local communities. Across the United States, climate event impacts vary by region. FEMA's National Risk Index measures county-level hazard risk; Redfin county data shows local house price growth in 2025.</p>

    <div class="toolbar" id="score-hazard-buttons"></div>
    <div class="viz-grid">
      <div class="panel">
        <h3>NRI score against 2025 median PPSF YoY</h3>
        <p class="sub">Each point is a county. PPSF YoY is capped at the 1st and 99th percentiles for display.</p>
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
        <h3>NRI risk rating against 2025 median PPSF YoY</h3>
        <p class="sub">Each point is a county, jittered within its rating band.</p>
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
    <h2>Does house price growth change around climate events?</h2>
    <p class="section-copy">An extreme climate event can alter local housing-market outlooks as buyers and sellers re-evaluate risk. The event windows below align county price growth from 12 months before event start through the months after event end.</p>
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
    <h2>What sets apart counties with different climate risk?</h2>
    <p class="section-copy">Certain features make counties more vulnerable or resilient to destructive weather events. This section summarizes how county features differ across FEMA NRI risk-rating groups.</p>
    <div class="viz-grid feature-grid">
      <div class="panel">
        <h3>Risk rating x feature heatmaps</h3>
        <p class="sub">Select a feature to compare feature-value buckets against NRI risk ratings. Cells show the share of counties in each risk-rating group that fall into each feature bucket.</p>
        <div class="toolbar" id="feature-category-buttons"></div>
        <div class="toolbar" id="feature-buttons"></div>
        <div id="feature-heatmaps"></div>
        <div class="legend" id="feature-legend"></div>
        <div class="sources" id="feature-source"></div>
      </div>
    </div>
    <div class="takeaway" id="feature-takeaway"></div>
    <div class="sources">Sources: local marts <code>mart.acs_county_economic_annual</code>, <code>mart.acs_county_demographic_annual</code>, <code>mart.acs_county_affordability_annual</code>, <code>mart.ncei_county_weather_monthly</code>, and <code>mart.nri_county_risk</code>. Cost components are midpoint estimates from ACS cost buckets.</div>
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
let horizon = 12;
const horizons = [12, 24, 36, 48, 60];
let selectedFeatureCategory = DATA.features.rows[0]?.category || "Economic";
let selectedFeature = DATA.features.rows.find(d => d.category === selectedFeatureCategory)?.feature || DATA.features.rows[0]?.feature;

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
  const data = DATA.priceRisk.counties.filter(d => d.avgPpsfYoyCapped != null && hazardCounty(d, scoreHazard).score != null);
  const svg = d3.select("#score-scatter");
  const width = svg.node().clientWidth || 520, height = svg.node().clientHeight || 430;
  const margin = {top: 18, right: 18, bottom: 46, left: 68};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const x = d3.scaleLinear().domain(d3.extent(data, d => hazardCounty(d, scoreHazard).score)).nice().range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(d3.extent(data, d => d.avgPpsfYoyCapped)).nice().range([height - margin.bottom, margin.top]);
  const yTicks = sparsePctTicks(y.domain());
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(6));
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickFormat(fmtAxisPct));
  svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text(`${hazardLabel(scoreHazard)} score`);
  svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",20).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Average 2025 median PPSF YoY, capped");
  svg.append("g").selectAll("circle").data(data).join("circle")
    .attr("cx", d => x(hazardCounty(d, scoreHazard).score)).attr("cy", d => y(d.avgPpsfYoyCapped)).attr("r", 3.1).attr("fill", "#8c2d22").attr("opacity", .42)
    .on("mousemove", (event, d) => tooltip.style("display","block").style("left",`${event.clientX+12}px`).style("top",`${event.clientY+12}px`).html(`<strong>${d.county}</strong>${hazardLabel(scoreHazard)} score: ${hazardCounty(d, scoreHazard).score}<br>PPSF YoY: ${pctText(d.avgPpsfYoy)}${d.ppsfYoyWasCapped ? "<br>Point capped for display" : ""}`))
    .on("mouseleave", () => tooltip.style("display","none"));
}

function drawRatingScatter() {
  const data = DATA.priceRisk.counties.filter(d => d.avgPpsfYoyCapped != null && hazardCounty(d, ratingHazard).ratingValue != null);
  const svg = d3.select("#rating-scatter");
  const width = svg.node().clientWidth || 520, height = svg.node().clientHeight || 430;
  const margin = {top: 18, right: 18, bottom: 54, left: 68};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const x = d3.scalePoint().domain(RISK_ORDER).range([margin.left, width - margin.right]).padding(.55);
  const y = d3.scaleLinear().domain(d3.extent(data, d => d.avgPpsfYoyCapped)).nice().range([height - margin.bottom, margin.top]);
  const yTicks = sparsePctTicks(y.domain());
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x));
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickFormat(fmtAxisPct));
  svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",20).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Average 2025 median PPSF YoY, capped");
  svg.append("g").selectAll("circle").data(data).join("circle")
    .attr("cx", d => x(hazardCounty(d, ratingHazard).rating) + ((d.fips.charCodeAt(4) % 9) - 4) * 3.4)
    .attr("cy", d => y(d.avgPpsfYoyCapped))
    .attr("r", 3.1)
    .attr("fill", d => RISK_COLORS[hazardCounty(d, ratingHazard).rating] || "#b8b1a8")
    .attr("opacity", .48)
    .on("mousemove", (event, d) => tooltip.style("display","block").style("left",`${event.clientX+12}px`).style("top",`${event.clientY+12}px`).html(`<strong>${d.county}</strong>${hazardLabel(ratingHazard)} rating: ${hazardCounty(d, ratingHazard).rating}<br>PPSF YoY: ${pctText(d.avgPpsfYoy)}${d.ppsfYoyWasCapped ? "<br>Point capped for display" : ""}`))
    .on("mouseleave", () => tooltip.style("display","none"));
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

function drawLineChart(svgId, source, groupKey, horizonLimit, activeRisk = null) {
  const data = source.filter(d => d.month <= horizonLimit);
  const svg = d3.select(svgId);
  const width = svg.node().clientWidth || 700, height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: 96, bottom: 42, left: 58};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const x = d3.scaleLinear().domain([-12, horizonLimit]).range([margin.left, width - margin.right]);
  const values = data.flatMap(d => [d.q1, d.median, d.q3]).filter(v => v != null);
  const y = d3.scaleLinear().domain(d3.extent(values)).nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  const useYears = horizonLimit > 12;
  const yearTicks = [-12, 0, ...d3.range(12, horizonLimit + 1, 12)];
  const axis = useYears
    ? d3.axisBottom(x).tickValues(yearTicks).tickFormat(d => d < 0 ? "1y pre" : d === 0 ? "event" : `${d / 12}y`)
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
  d3.select("#resume-risk").style("display", eventView === "grouped" ? null : "none");
  drawLineChart("#event-window", eventView === "all" ? DATA.eventWindows.aggregate : DATA.eventWindows.byRating, eventView === "all" ? "series" : "riskRating", horizon, eventView === "all" ? null : selectedRisk);
  drawAffectedMap();
  updateEventTakeaway();
}

function startRiskTimer() {
  clearInterval(riskTimer);
  riskTimer = setInterval(() => {
    if (eventView !== "grouped") return;
    selectedRisk = RISK_ORDER[(RISK_ORDER.indexOf(selectedRisk) + 1) % RISK_ORDER.length];
    renderEventSection();
  }, 3200);
}

function drawFeatureHeatmaps() {
  const host = d3.select("#feature-heatmaps").html("");
  const rows = DATA.features.rows.filter(d => d.category === selectedFeatureCategory && d.feature === selectedFeature);
  const buckets = [...new Map(rows.sort((a,b)=>d3.ascending(a.bucketOrder,b.bucketOrder)).map(d => [d.bucket, d.bucket])).values()];
  const maxShare = d3.max(rows, d => d.share) || 1;
  const color = d3.scaleSequential([0, maxShare], d3.interpolateBlues);
  const grid = host.append("div").attr("class","heatmap").append("div").attr("class","heatmap-grid")
    .style("grid-template-columns", `140px repeat(${buckets.length}, minmax(115px, 1fr))`);
  grid.append("div");
  buckets.forEach(bucket => grid.append("div").attr("class","heat-col").text(bucket));
  RISK_ORDER.forEach(rating => {
    grid.append("div").attr("class","heat-label").text(rating);
    buckets.forEach(bucket => {
      const cell = rows.find(d => d.riskRating === rating && d.bucket === bucket);
      const dark = cell && cell.share != null && cell.share / maxShare > .62;
      grid.append("div").attr("class","heat-cell")
        .style("background", cell && cell.share != null ? color(cell.share) : "#ece7df")
        .style("color", dark ? "#ffffff" : "#172026")
        .html(cell ? `<strong>${fmtShare(cell.share)}</strong><span>${cell.count} of ${cell.total} counties</span>` : "n/a")
        .on("mousemove", (event) => {
          if (!cell) return;
          tooltip.style("display","block").style("left",`${event.clientX+12}px`).style("top",`${event.clientY+12}px`).html(`<strong>${selectedFeature}</strong>${rating}<br>${bucket}<br>${fmtShare(cell.share)} of counties (${cell.count} of ${cell.total})`);
        })
        .on("mouseleave", () => tooltip.style("display","none"));
    });
  });
  d3.select("#feature-legend").html(scaleLegendHtml([0, maxShare], "#eff6ff", "#08519c", "Share of counties within risk-rating group", fmtShare));
  const source = rows.find(d => d.source)?.source || "DuckDB marts";
  d3.select("#feature-source").html(`Selected feature source: local DuckDB mart <code>${source}</code>; risk ratings from <code>mart.nri_county_risk</code>.`);
  const top = DATA.features.topFeatures.map(d => `${d.feature} (${d.corr >= 0 ? "higher" : "lower"} with risk)`).join(", ");
  d3.select("#feature-takeaway").text(`Counties with the strongest feature relationships to climate risk in this mart extract include: ${top}.`);
}

function formatFeature(value, format) {
  if (value == null) return "n/a";
  if (format === "currency") return fmtMoney(value);
  if (format === "percent") return `${fmtNum(value)}%`;
  return fmtNum(value);
}

function initButtons() {
  d3.select("#score-hazard-buttons").selectAll("button").data(DATA.priceRisk.hazards).join("button").text(d=>d.label).classed("active",d=>d.key===scoreHazard).on("click",(event,d)=>{scoreHazard=d.key; d3.select("#score-hazard-buttons").selectAll("button").classed("active",x=>x.key===scoreHazard); drawScoreScatter(); drawScoreMap();});
  d3.select("#rating-hazard-buttons").selectAll("button").data(DATA.priceRisk.hazards).join("button").text(d=>d.label).classed("active",d=>d.key===ratingHazard).on("click",(event,d)=>{ratingHazard=d.key; d3.select("#rating-hazard-buttons").selectAll("button").classed("active",x=>x.key===ratingHazard); drawRatingScatter(); drawRatingMap();});
  d3.select("#score-map-mode").selectAll("button").data([{key:"score",label:"NRI score"},{key:"ppsf",label:"Median PPSF YoY"}]).join("button").text(d=>d.label).classed("active",d=>d.key===scoreMapMode).on("click",(event,d)=>{scoreMapMode=d.key; d3.select("#score-map-mode").selectAll("button").classed("active",x=>x.key===scoreMapMode); drawScoreMap();});
  d3.select("#rating-map-mode").selectAll("button").data([{key:"rating",label:"NRI risk rating"},{key:"ppsf",label:"Median PPSF YoY"}]).join("button").text(d=>d.label).classed("active",d=>d.key===ratingMapMode).on("click",(event,d)=>{ratingMapMode=d.key; d3.select("#rating-map-mode").selectAll("button").classed("active",x=>x.key===ratingMapMode); drawRatingMap();});
  d3.select("#event-view-mode").selectAll("button").data([{key:"all",label:"All affected counties"},{key:"grouped",label:"Grouped by NRI risk rating"}]).join("button").text(d=>d.label).classed("active",d=>d.key===eventView).on("click",(event,d)=>{eventView=d.key; if(eventView==="grouped") startRiskTimer(); else clearInterval(riskTimer); renderEventSection();});
  d3.select("#risk-frame-buttons").selectAll("button").data(RISK_ORDER).join("button").text(d=>d).classed("active",d=>d===selectedRisk).on("click",(event,d)=>{selectedRisk=d; clearInterval(riskTimer); renderEventSection();});
  const categories = [...new Set(DATA.features.rows.map(d => d.category))];
  d3.select("#feature-category-buttons").selectAll("button").data(categories).join("button").text(d=>d).classed("active",d=>d===selectedFeatureCategory).on("click",(event,d)=>{selectedFeatureCategory=d; selectedFeature=DATA.features.rows.find(row=>row.category===d)?.feature; renderFeatureButtons(); drawFeatureHeatmaps();});
  renderFeatureButtons();
  d3.select("#resume-risk").on("click", startRiskTimer);
  d3.select("#prev-window").on("click",()=>{const index=horizons.indexOf(horizon); if(index>0){horizon=horizons[index-1]; d3.select("#event-window").classed("compressing",true); renderEventSection(); setTimeout(()=>d3.select("#event-window").classed("compressing",false),380);}});
  d3.select("#next-window").on("click",()=>{const index=horizons.indexOf(horizon); if(index<horizons.length-1){horizon=horizons[index+1]; d3.select("#event-window").classed("compressing",true); renderEventSection(); setTimeout(()=>d3.select("#event-window").classed("compressing",false),380);}});
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
startRiskTimer();
window.addEventListener("resize", () => {
  drawScoreScatter();
  drawScoreMap();
  drawRatingScatter();
  drawRatingMap();
  renderEventSection();
  drawFeatureHeatmaps();
});
</script>
</body>
</html>"""


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        price_risk = build_price_risk(con)
        event_windows = build_event_windows(con)
        features = build_feature_payload(con)
    geojson = build_geojson({county["fips"] for county in price_risk["counties"]})
    data = {"priceRisk": price_risk, "eventWindows": event_windows, "features": features, "geojson": geojson}
    OUT_PATH.write_text(make_html(data), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
