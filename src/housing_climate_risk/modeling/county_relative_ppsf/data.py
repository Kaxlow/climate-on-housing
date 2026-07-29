from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd


RISK_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]
RISK_MAP = {
    "Very Low": "Very Low",
    "Relatively Low": "Low",
    "Relatively Moderate": "Medium",
    "Moderate": "Medium",
    "Relatively High": "High",
    "Very High": "Very High",
}

MODEL_ROLE_EXCLUSIONS = {
    "median_ppsf_yoy": "prediction_target",
    "housing_market_index": "contains_prediction_target",
    "risk_rating": "categorical_grouping_field",
}
MODEL_FEATURE_PRIORITY = (
    "extreme_event_count",
    "homeowners_insurance_pct_income",
    "property_taxes_pct_income",
    "utilities_pct_income",
    "earnings_by_place_of_work_per_capita_usd",
    "dividends_interest_rent_per_capita_usd",
    "transfer_receipts_per_capita_usd",
)

ELECTRICITY_BINS = [
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_less_than_dollars_50_est", 25),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_50_to_dollars_99_est", 75),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_100_to_dollars_149_est", 125),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_150_to_dollars_199_est", 175),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_200_to_dollars_249_est", 225),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_250_or_more_est", 275),
]
GAS_BINS = [
    ("b25133_monthly_gas_costs_total_charged_for_gas_less_than_dollars_25_est", 12.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_25_to_dollars_49_est", 37.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_50_to_dollars_74_est", 62.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_75_to_dollars_99_est", 87.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_100_to_dollars_149_est", 125),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_150_or_more_est", 175),
]
WATER_BINS = [
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_less_than_dollars_125_est", 62.5),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_125_to_dollars_249_est", 187.5),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_250_to_dollars_499_est", 375),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_500_to_dollars_749_est", 625),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_750_to_dollars_999_est", 875),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_1_000_or_more_est", 1125),
]
FUEL_BINS = [
    ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_less_than_dollars_250_est", 125),
    ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_250_to_dollars_749_est", 500),
    ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_750_or_more_est", 875),
]
INSURANCE_BINS = [
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
]


@dataclass(frozen=True)
class DatasetBuildResult:
    frame: pd.DataFrame
    metadata: dict[str, object]


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _numeric_expression(column: str) -> str:
    quoted = _quote_ident(column)
    return f"try_cast(replace(nullif(trim(cast({quoted} AS VARCHAR)), ''), ',', '') AS DOUBLE)"


def _annual_average_query(table: str, columns: list[str]) -> str:
    select_parts = [f"avg({_numeric_expression(column)}) AS {_quote_ident(column)}" for column in columns]
    return f"""
        SELECT
            lpad(fips, 5, '0') AS fips,
            {", ".join(select_parts)}
        FROM mart.{_quote_ident(table)}
        WHERE fips IS NOT NULL
          AND year >= (SELECT max(year) FROM mart.{_quote_ident(table)}) - 9
        GROUP BY fips
    """


def _weighted_midpoint(frame: pd.DataFrame, bins: list[tuple[str, float]]) -> pd.Series:
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for column, midpoint in bins:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & values.ge(0)
        numerator = numerator.add(values.where(valid, 0) * midpoint, fill_value=0)
        denominator = denominator.add(values.where(valid, 0), fill_value=0)
    return numerator.where(denominator.gt(0)).div(denominator.where(denominator.gt(0)))


def _build_affordability_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    base_columns = [
        "s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est",
        "s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_median_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_30_0_to_34_9_percent_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_35_0_percent_or_more_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_30_0_to_34_9_percent_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_35_0_percent_or_more_pct",
    ]
    all_bins = ELECTRICITY_BINS + GAS_BINS + WATER_BINS + FUEL_BINS + INSURANCE_BINS
    frame = con.execute(
        _annual_average_query(
            "acs_county_affordability_annual",
            base_columns + [column for column, _ in all_bins],
        )
    ).df()
    for column in frame.columns:
        if column != "fips":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    income = frame[base_columns[0]]
    monthly_housing_costs = frame[base_columns[1]]
    property_taxes = frame[base_columns[2]]
    insurance = _weighted_midpoint(frame, INSURANCE_BINS)
    utilities_monthly = (
        _weighted_midpoint(frame, ELECTRICITY_BINS)
        + _weighted_midpoint(frame, GAS_BINS)
        + _weighted_midpoint(frame, WATER_BINS) / 12
        + _weighted_midpoint(frame, FUEL_BINS) / 12
    )
    burden_columns = base_columns[3:]
    result = pd.DataFrame(
        {
            "fips": frame["fips"],
            "income_median_household_usd": income,
            "insurance_homeowners_pct_income": insurance / income * 100,
            "property_taxes_pct_income": property_taxes / income * 100,
            "utilities_pct_income": utilities_monthly * 12 / income * 100,
            "housing_burden_30pct_plus_share": frame[burden_columns].mean(axis=1, skipna=True),
            "homeownership_cost_pct_income": monthly_housing_costs * 12 / income * 100,
        }
    )
    return result.replace([np.inf, -np.inf], np.nan)


def _build_economic_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    unemployment_column = "dp03_civilian_labor_force_unemployment_rate_pct"
    unemployment = con.execute(
        _annual_average_query("acs_county_economic_annual", [unemployment_column])
    ).df().rename(columns={unemployment_column: "unemployment_rate_pct"})
    bea = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            avg(net_earnings_by_place_of_residence_thousands * 1000.0
                / nullif(population, 0)) AS net_earnings_per_capita,
            avg(dividends_interest_rent_thousands * 1000.0
                / nullif(population, 0)) AS dividends_interest_rent_per_capita,
            avg(transfer_receipts_thousands * 1000.0
                / nullif(population, 0)) AS transfer_receipts_per_capita
        FROM mart.statsamerica_bea_personal_income_annual
        WHERE fips IS NOT NULL
          AND year >= (SELECT max(year) FROM mart.statsamerica_bea_personal_income_annual) - 9
          AND population > 0
        GROUP BY fips
        """
    ).df()
    cew = con.execute(
        """
        SELECT
            lpad(s.fips, 5, '0') AS fips,
            avg(s.total_wages_dollars / nullif(t.total_wages_dollars, 0) * 100)
                AS accom_food_wages_pct_total_wages
        FROM mart.statsamerica_cew_county_sector_annual AS s
        INNER JOIN mart.statsamerica_cew_county_annual AS t
          ON lpad(s.fips, 5, '0') = lpad(t.fips, 5, '0')
         AND s.year = t.year
        WHERE s.naics_code = '72'
          AND s.year >= (SELECT max(year) FROM mart.statsamerica_cew_county_sector_annual) - 9
          AND t.total_wages_dollars > 0
        GROUP BY s.fips
        """
    ).df()
    return unemployment.merge(bea, on="fips", how="outer").merge(cew, on="fips", how="outer")


def _build_demographic_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    source_columns = {
        "dp05_total_population_65_plus_pct": "age_65_plus_share",
        "dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct": "disability_share",
        "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct": "english_less_than_very_well_share",
        "dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct": "broadband_subscription_share",
        "total_population": "avg_population",
    }
    demographics = con.execute(
        _annual_average_query("acs_county_demographic_annual", list(source_columns))
    ).df().rename(columns=source_columns)
    migration = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            avg(CAST(total_net_migration AS DOUBLE)) AS avg_total_net_migration
        FROM mart.statsamerica_population_components_annual
        WHERE fips IS NOT NULL
          AND year >= (SELECT max(year) FROM mart.statsamerica_population_components_annual) - 9
        GROUP BY fips
        """
    ).df()
    frame = demographics.merge(migration, on="fips", how="outer")
    frame["net_migration_rate"] = frame["avg_total_net_migration"] / frame["avg_population"]
    frame["no_broadband_internet_share"] = 100 - frame["broadband_subscription_share"]
    return frame[
        [
            "fips",
            "net_migration_rate",
            "age_65_plus_share",
            "disability_share",
            "english_less_than_very_well_share",
            "no_broadband_internet_share",
        ]
    ].replace([np.inf, -np.inf], np.nan)


def _clean_redfin_expression(column: str) -> str:
    numeric = _numeric_expression(column)
    return f"CASE WHEN {numeric} <= -888888000 THEN NULL ELSE {numeric} END"


def _catalog_model_features(
    con: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, list[str], dict[str, tuple[str, str]]]:
    """Return numeric model candidates from the authoritative retained catalog."""
    catalog = con.execute(
        """
        SELECT feature_table, feature_name, category, definition, unit,
               temporal_grain, infographic_use
        FROM feature.catalog
        WHERE retained
          AND feature_table IS NOT NULL
        ORDER BY feature_table, feature_name
        """
    ).df()
    catalog["model_exclusion_reason"] = catalog["feature_name"].map(
        MODEL_ROLE_EXCLUSIONS
    )
    candidates = catalog.loc[catalog["model_exclusion_reason"].isna()].copy()
    priority = {
        feature: index for index, feature in enumerate(MODEL_FEATURE_PRIORITY)
    }
    candidates["_model_priority"] = candidates["feature_name"].map(priority).fillna(
        len(priority)
    )
    candidates = candidates.sort_values(
        ["_model_priority", "feature_table", "feature_name"]
    )
    feature_columns: list[str] = []
    feature_meta: dict[str, tuple[str, str]] = {}
    for row in candidates.itertuples(index=False):
        schema = con.execute(
            f"DESCRIBE feature.{_quote_ident(row.feature_table)}"
        ).df().set_index("column_name")
        column_type = str(schema.loc[row.feature_name, "column_type"]).upper()
        if not any(
            token in column_type
            for token in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL")
        ):
            catalog.loc[
                catalog["feature_name"].eq(row.feature_name),
                "model_exclusion_reason",
            ] = "non_numeric"
            continue
        label = (
            str(row.infographic_use).split(":", 1)[1].strip()
            if pd.notna(row.infographic_use) and ":" in str(row.infographic_use)
            else str(row.feature_name).replace("_", " ").title()
        )
        feature_columns.append(row.feature_name)
        feature_meta[row.feature_name] = (str(row.category), label)
    return catalog, feature_columns, feature_meta


def _aggregate_catalog_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
) -> pd.DataFrame:
    select_parts = [
        f"avg(try_cast({_quote_ident(column)} AS DOUBLE)) AS {_quote_ident(column)}"
        for column in columns
    ]
    schema_columns = set(
        con.execute(f"DESCRIBE feature.{_quote_ident(table)}")
        .df()["column_name"]
        .tolist()
    )
    if "year" in schema_columns:
        time_filter = (
            f"AND year >= (SELECT max(year) - 9 FROM feature.{_quote_ident(table)})"
        )
    elif "climate_month" in schema_columns:
        time_filter = (
            "AND extract(year FROM climate_month) >= "
            f"(SELECT max(extract(year FROM climate_month)) - 9 FROM feature.{_quote_ident(table)})"
        )
    elif "housing_month" in schema_columns:
        time_filter = (
            "AND extract(year FROM housing_month) >= "
            f"(SELECT max(extract(year FROM housing_month)) - 9 FROM feature.{_quote_ident(table)})"
        )
    else:
        time_filter = ""
    return con.execute(
        f"""
        SELECT lpad(fips, 5, '0') AS fips, {", ".join(select_parts)}
        FROM feature.{_quote_ident(table)}
        WHERE fips IS NOT NULL
          {time_filter}
        GROUP BY fips
        """
    ).df()


def _build_housing_features_and_target(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    source_columns = {
        "AVG_SALE_TO_LIST_YOY": "avg_sale_to_list_yoy",
        "HOMES_SOLD_YOY": "homes_sold_yoy",
        "INVENTORY_YOY": "inventory_yoy",
        "NEW_LISTINGS_YOY": "new_listings_yoy",
        "MEDIAN_DOM_YOY": "median_dom_yoy",
        "PRICE_DROPS_YOY": "price_drops_yoy",
    }
    feature_selects = [
        f"avg({_clean_redfin_expression(source)}) AS {_quote_ident(alias)}"
        for source, alias in source_columns.items()
    ]
    target_expression = _clean_redfin_expression("MEDIAN_PPSF_YOY")
    return con.execute(
        f"""
        SELECT
            lpad(fips, 5, '0') AS fips,
            any_value(REGION) AS county,
            any_value(STATE_CODE) AS state,
            median({target_expression}) AS county_median_ppsf_yoy,
            count({target_expression}) AS observed_housing_months,
            {", ".join(feature_selects)}
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


def build_county_modeling_dataset(
    con: duckdb.DuckDBPyConnection,
    *,
    minimum_housing_months: int = 60,
) -> DatasetBuildResult:
    """Build one modeling observation per county from the DuckDB marts."""
    catalog, feature_columns, feature_meta = _catalog_model_features(con)
    housing = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            any_value(county_name) AS county,
            any_value(state_code) AS state,
            median(median_ppsf_yoy) AS county_median_ppsf_yoy,
            count(median_ppsf_yoy) AS observed_housing_months
        FROM feature.county_housing_monthly
        WHERE fips IS NOT NULL
          AND housing_month IS NOT NULL
          AND extract(year FROM housing_month) >= (
              SELECT max(extract(year FROM housing_month)) - 9
              FROM feature.county_housing_monthly
          )
        GROUP BY fips
        """
    ).df()
    nri = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            COUNTY AS nri_county,
            STATEABBRV AS nri_state,
            risk_score,
            risk_rating
        FROM mart.nri_county_risk
        WHERE fips IS NOT NULL
        """
    ).df()

    frame = housing.merge(nri, on="fips", how="inner")
    catalog_candidates = catalog.loc[
        catalog["feature_name"].isin(feature_columns)
    ]
    for table, rows in catalog_candidates.groupby("feature_table", sort=True):
        columns = [
            column
            for column in rows["feature_name"].tolist()
            if not (table == "county_risk" and column == "risk_score")
        ]
        if not columns:
            continue
        features = _aggregate_catalog_table(con, str(table), columns)
        frame = frame.merge(features, on="fips", how="left")
    frame["risk_group"] = frame["risk_rating"].map(RISK_MAP)
    frame["county"] = frame["county"].fillna(frame["nri_county"])
    frame["state"] = frame["state"].fillna(frame["nri_state"])
    frame = frame.loc[
        frame["risk_group"].isin(RISK_ORDER)
        & frame["county_median_ppsf_yoy"].notna()
        & frame["observed_housing_months"].ge(minimum_housing_months)
    ].copy()
    frame[feature_columns] = frame[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    frame[feature_columns] = frame[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    group_stats = (
        frame.groupby("risk_group", observed=False)["county_median_ppsf_yoy"]
        .agg(
            group_median_ppsf_yoy="median",
            group_q1_ppsf_yoy=lambda values: values.quantile(0.25),
            group_q3_ppsf_yoy=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    frame = frame.merge(group_stats, on="risk_group", how="left")
    frame["relative_median_ppsf_yoy"] = (
        frame["county_median_ppsf_yoy"] - frame["group_median_ppsf_yoy"]
    )
    group_iqr = frame["group_q3_ppsf_yoy"] - frame["group_q1_ppsf_yoy"]
    frame["relative_median_ppsf_yoy_iqr"] = (
        frame["relative_median_ppsf_yoy"] / group_iqr.replace(0, np.nan)
    )
    frame["feature_non_null_count"] = frame[feature_columns].notna().sum(axis=1)
    frame = frame.sort_values(["risk_group", "fips"]).drop_duplicates("fips", keep="first")

    housing_range = con.execute(
        """
        SELECT min(period_begin), max(period_begin)
        FROM mart.redfin_county_monthly
        WHERE period_begin IS NOT NULL
          AND extract(year FROM period_begin) >= (
              SELECT max(extract(year FROM period_begin)) - 9
              FROM mart.redfin_county_monthly
              WHERE period_begin IS NOT NULL
          )
        """
    ).fetchone()
    metadata = {
        "target": "relative_median_ppsf_yoy",
        "target_definition": (
            "County median monthly MEDIAN_PPSF_YOY over the latest 10 calendar years "
            "minus the median among eligible counties in the same NRI risk group."
        ),
        "minimum_housing_months": minimum_housing_months,
        "housing_period_start": str(housing_range[0]),
        "housing_period_end": str(housing_range[1]),
        "county_count": int(frame["fips"].nunique()),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_meta": feature_meta,
        "catalog_model_exclusions": {
            row.feature_name: row.model_exclusion_reason
            for row in catalog.loc[
                catalog["model_exclusion_reason"].notna()
            ].itertuples(index=False)
        },
        "risk_group_counts": {
            group: int((frame["risk_group"] == group).sum()) for group in RISK_ORDER
        },
    }
    return DatasetBuildResult(frame=frame.reset_index(drop=True), metadata=metadata)
