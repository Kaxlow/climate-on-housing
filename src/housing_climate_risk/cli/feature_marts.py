"""Build domain-oriented feature marts and their metadata catalog."""

from __future__ import annotations

import pandas as pd


NOAA_DAMAGE_THRESHOLD = 1_000_000_000
EXCLUDED_FEMA_INCIDENT_TYPES = (
    "Biological",
    "Dam/Levee Break",
    "Chemical",
    "Terrorist",
    "Other",
    "Toxic Substances",
)
ELECTRICITY_COST_BINS = (
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_less_than_dollars_50_est", 25),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_50_to_dollars_99_est", 75),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_100_to_dollars_149_est", 125),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_150_to_dollars_199_est", 175),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_200_to_dollars_249_est", 225),
    ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_250_or_more_est", 275),
)
GAS_COST_BINS = (
    ("b25133_monthly_gas_costs_total_charged_for_gas_less_than_dollars_25_est", 12.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_25_to_dollars_49_est", 37.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_50_to_dollars_74_est", 62.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_75_to_dollars_99_est", 87.5),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_100_to_dollars_149_est", 125),
    ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_150_or_more_est", 175),
)
WATER_COST_BINS = (
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_less_than_dollars_125_est", 62.5),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_125_to_dollars_249_est", 187.5),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_250_to_dollars_499_est", 375),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_500_to_dollars_749_est", 625),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_750_to_dollars_999_est", 875),
    ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_1_000_or_more_est", 1125),
)
FUEL_COST_BINS = (
    ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_less_than_dollars_250_est", 125),
    ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_250_to_dollars_749_est", 500),
    ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_750_or_more_est", 875),
)
INSURANCE_COST_BINS = tuple(
    (
        f"b25141_homeowners_insurance_costs_by_mortgage_status_total_{status}_{suffix}_est",
        midpoint,
    )
    for status in ("mortgage", "not_mortgaged")
    for suffix, midpoint in (
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
    )
)


def _weighted_midpoint_sql(bins: tuple[tuple[str, float], ...]) -> str:
    values = [
        f"greatest(coalesce(max(try_cast(\"{column}\" AS DOUBLE)), 0), 0)"
        for column, _ in bins
    ]
    numerator = " + ".join(
        f"({value}) * {midpoint}" for value, (_, midpoint) in zip(values, bins, strict=True)
    )
    denominator = " + ".join(f"({value})" for value in values)
    return f"({numerator}) / nullif(({denominator}), 0)"


def _assert_unique_key(con, table_name: str, columns: tuple[str, ...]) -> None:
    key_sql = ", ".join(columns)
    duplicate_count = con.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT {key_sql}
            FROM {table_name}
            GROUP BY {key_sql}
            HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicate_count:
        raise RuntimeError(
            f"{table_name} violates its declared key ({key_sql}): "
            f"{duplicate_count:,} duplicate groups"
        )


def _catalog_rows() -> list[dict[str, object]]:
    common = {
        "geographic_grain": "county",
        "retained": True,
        "exclusion_reason": None,
    }
    rows = [
        ("county_economic_annual", "median_household_income_usd", "Economic and affordability", "Median annual household income.", "USD/year", "mart.acs_county_affordability_annual; mart.acs_county_economic_annual", "Coalesced ACS affordability and DP03 estimates at matching county-year.", "annual", "Feature comparison: Income"),
        ("county_economic_annual", "unemployment_rate_pct", "Economic and affordability", "Civilian labor-force unemployment rate.", "percent", "mart.acs_county_economic_annual", "Numeric ACS DP03 percentage estimate.", "annual", "Feature comparison: Unemployment"),
        ("county_economic_annual", "median_home_value_usd", "Economic and affordability", "Median owner-occupied home value.", "USD", "mart.acs_county_affordability_annual", "Numeric ACS affordability estimate.", "annual", "Affordability context"),
        ("county_economic_annual", "median_gross_rent_usd_month", "Economic and affordability", "Median monthly gross rent.", "USD/month", "mart.acs_county_affordability_annual", "Numeric ACS affordability estimate.", "annual", "Affordability context"),
        ("county_economic_annual", "median_owner_costs_mortgage_usd_month", "Economic and affordability", "Median monthly owner costs for mortgaged homes.", "USD/month", "mart.acs_county_affordability_annual", "Numeric ACS affordability estimate.", "annual", "Affordability context"),
        ("county_economic_annual", "housing_cost_pct_income", "Economic and affordability", "Annualized median owner costs divided by owner-household median income.", "percent", "mart.acs_county_affordability_annual", "Project-derived ACS ratio; denominator must be positive.", "annual", "Feature comparison: Homeownership Cost Share"),
        ("county_economic_annual", "owner_cost_burden_30pct_plus_pct", "Economic and affordability", "Share of mortgaged owners spending at least 30% of income on housing.", "percent", "mart.acs_county_affordability_annual", "Burdened mortgaged-owner households divided by total mortgaged owner-occupied housing units; official ACS percentage buckets are used when the denominator field is unavailable.", "annual", "Feature comparison: Cost-Burdened Households"),
        ("county_economic_annual", "median_property_taxes_usd_year", "Economic and affordability", "Median annual real-estate taxes paid.", "USD/year", "mart.acs_county_affordability_annual", "Numeric ACS B25103 estimate.", "annual", "Feature comparison: Property Tax"),
        ("county_economic_annual", "per_capita_personal_income_usd", "Economic and affordability", "BEA per-capita personal income.", "USD/person/year", "mart.statsamerica_bea_per_capita_income_annual", "County-level BEA value joined on exact year.", "annual", "Economic context"),
        ("county_economic_annual", "net_earnings_per_capita_usd", "Economic and affordability", "Net earnings by place of residence per resident.", "USD/person/year", "mart.statsamerica_bea_personal_income_annual", "Thousands of dollars multiplied by 1,000 and divided by positive population.", "annual", "Feature comparison: Net Earnings per Capita"),
        ("county_economic_annual", "transfer_receipts_per_capita_usd", "Economic and affordability", "Transfer receipts per resident.", "USD/person/year", "mart.statsamerica_bea_personal_income_annual", "Thousands of dollars multiplied by 1,000 and divided by positive population.", "annual", "Feature comparison: Transfer Receipts per Capita"),
        ("county_economic_annual", "average_annual_wage_usd", "Economic and affordability", "Average annual wage across covered employment.", "USD/worker/year", "mart.statsamerica_cew_county_annual", "County total-ownership CEW value joined on exact year.", "annual", "Economic context"),
        ("county_economic_annual", "homeowners_insurance_pct_income", "Economic and affordability", "Estimated annual homeowners-insurance cost as a share of county median household income.", "percent", "mart.acs_county_affordability_annual", "Weighted midpoint across ACS B25141 insurance-cost buckets for mortgaged and non-mortgaged owners, divided by same-year positive county median household income.", "annual", "Feature comparison: Homeowners Insurance Share of Income"),
        ("county_economic_annual", "property_taxes_pct_income", "Economic and affordability", "Median annual real-estate taxes as a share of county median household income.", "percent", "mart.acs_county_affordability_annual", "ACS B25103 median real-estate taxes divided by same-year positive county median household income.", "annual", "Feature comparison: Property Taxes Share of Income"),
        ("county_economic_annual", "utilities_pct_income", "Economic and affordability", "Estimated annual electricity, gas, water/sewer, and other-fuel costs as a share of county median household income.", "percent", "mart.acs_county_affordability_annual", "Weighted midpoint estimates from ACS B25132-B25135 cost buckets, annualized where monthly, divided by same-year positive county median household income.", "annual", "Feature comparison: Utilities Share of Income"),
        ("county_economic_annual", "earnings_by_place_of_work_per_capita_usd", "Economic and affordability", "BEA earnings by place of work per resident.", "USD/person/year", "mart.statsamerica_bea_personal_income_annual", "BEA thousands of dollars multiplied by 1,000 and divided by same-year positive population.", "annual", "Feature comparison: Earnings by Place of Work per Capita"),
        ("county_economic_annual", "dividends_interest_rent_per_capita_usd", "Economic and affordability", "BEA dividends, interest, and rent per resident.", "USD/person/year", "mart.statsamerica_bea_personal_income_annual", "BEA thousands of dollars multiplied by 1,000 and divided by same-year positive population.", "annual", "Feature comparison: Dividends, Interest, and Rent per Capita"),
        ("county_demographic_annual", "total_population", "Demographic", "County resident population.", "people", "mart.acs_county_demographic_annual", "Numeric ACS population estimate.", "annual", "Population denominator and context"),
        ("county_demographic_annual", "age_65_plus_pct", "Demographic", "Share of residents age 65 or older.", "percent", "mart.acs_county_demographic_annual", "Numeric ACS DP05 percentage estimate.", "annual", "Feature comparison: Age >= 65 Years"),
        ("county_demographic_annual", "disability_pct", "Demographic", "Share of the civilian noninstitutionalized population with a disability.", "percent", "mart.acs_county_demographic_annual", "Numeric ACS DP02 percentage estimate.", "annual", "Feature comparison: Disability Status"),
        ("county_demographic_annual", "communication_barrier_pct", "Demographic", "Share age 5+ speaking English less than very well.", "percent", "mart.acs_county_demographic_annual", "Numeric ACS DP02 percentage estimate.", "annual", "Feature comparison: Communication Barrier"),
        ("county_demographic_annual", "no_broadband_pct", "Demographic", "Households without a broadband subscription.", "percent", "mart.acs_county_demographic_annual", "100 minus the ACS broadband-subscription percentage.", "annual", "Feature comparison: No Internet Access"),
        ("county_demographic_annual", "net_migration_rate_pct", "Demographic", "Net domestic plus international migration as a share of population.", "percent", "mart.statsamerica_population_components_annual; mart.acs_county_demographic_annual", "Total net migration divided by same-year positive ACS population and multiplied by 100.", "annual", "Feature comparison: Net Migration Rate"),
        ("county_climate_monthly", "avg_temperature_f", "Climate and hazard", "Monthly county average temperature.", "degrees Fahrenheit", "mart.ncei_county_weather_monthly", "NCEI parameter pivot.", "monthly", "Feature comparison: Temperature"),
        ("county_climate_monthly", "precipitation_inches", "Climate and hazard", "Monthly county precipitation.", "inches", "mart.ncei_county_weather_monthly", "NCEI parameter pivot.", "monthly", "Feature comparison: Precipitation"),
        ("county_climate_monthly", "fema_event_count", "Climate and hazard", "Canonical qualifying county-events with FEMA provenance beginning in the month.", "county-events", "mart.climate_events", "Semantic incident count after FEMA EM/DR and cross-source reconciliation.", "monthly", "Event definition and Climate Playbook"),
        ("county_climate_monthly", "noaa_extreme_event_count", "Climate and hazard", "Canonical billion-dollar county-events with NOAA provenance beginning in the month.", "county-events", "mart.climate_events", "Semantic incident count after within-source and cross-source reconciliation.", "monthly", "Event definition and Climate Playbook"),
        ("county_climate_monthly", "extreme_event_count", "Climate and hazard", "Unique qualifying climate incidents beginning in the month.", "county-events", "mart.climate_events", "Each canonical incident is counted once even when represented by FEMA and NOAA records.", "monthly", "Feature comparison: Combined Extreme Event Count"),
        ("county_housing_monthly", "median_ppsf_yoy", "Housing market", "Year-over-year change in median sale price per square foot.", "proportion", "mart.redfin_county_monthly", "Redfin Data Center percentage divided by 100 for All Residential rows.", "monthly", "Primary housing outcome"),
        ("county_housing_monthly", "housing_market_index", "Housing market", "Composite standardized housing-market movement index.", "z-score", "mart.redfin_county_monthly", "Mean of global z-scores for PPSF YOY, sale-to-list YOY, homes sold YOY, and inverted inventory YOY.", "monthly", "Event-window supporting outcome"),
        ("county_housing_monthly", "avg_sale_to_list_yoy", "Housing market", "Year-over-year percentage-point change in average sale-to-list ratio.", "proportion", "mart.redfin_county_monthly", "Redfin Data Center percentage points divided by 100.", "monthly", "Event-window metric"),
        ("county_housing_monthly", "homes_sold_yoy", "Housing market", "Year-over-year change in homes sold.", "proportion", "mart.redfin_county_monthly", "Redfin Data Center percentage divided by 100.", "monthly", "Event-window metric"),
        ("county_housing_monthly", "inventory_yoy", "Housing market", "Year-over-year change in inventory.", "proportion", "mart.redfin_county_monthly", "Redfin Data Center percentage divided by 100.", "monthly", "Event-window metric"),
        ("county_housing_monthly", "new_listings_yoy", "Housing market", "Year-over-year change in new listings.", "proportion", "mart.redfin_county_monthly", "Redfin Data Center percentage divided by 100.", "monthly", "Event-window metric"),
        ("county_housing_monthly", "median_dom_yoy", "Housing market", "Year-over-year change in median days on market.", "proportion", "mart.redfin_county_monthly", "Redfin Data Center percentage divided by 100.", "monthly", "Event-window metric"),
        ("county_housing_monthly", "price_drops_yoy", "Housing market", "Year-over-year percentage-point change in the share of active listings with price drops.", "proportion", "mart.redfin_county_monthly", "Redfin Data Center percentage points divided by 100.", "monthly", "Event-window metric"),
        ("county_risk", "risk_score", "Climate and hazard", "FEMA National Risk Index composite risk score.", "score", "mart.nri_county_risk", "Numeric NRI score.", "snapshot", "Risk map and grouping"),
        ("county_risk", "risk_rating", "Climate and hazard", "FEMA National Risk Index categorical risk rating.", "category", "mart.nri_county_risk", "Canonical NRI rating label.", "snapshot", "Risk-group comparisons"),
        ("county_risk", "expected_annual_loss_score", "Climate and hazard", "NRI expected annual loss score.", "score", "mart.nri_county_risk", "Numeric NRI EAL score.", "snapshot", "Risk context"),
        ("county_risk", "social_vulnerability_score", "Climate and hazard", "NRI social vulnerability score.", "score", "mart.nri_county_risk", "Numeric NRI SOVI score.", "snapshot", "Risk context"),
        ("county_risk", "community_resilience_score", "Climate and hazard", "NRI community resilience score.", "score", "mart.nri_county_risk", "Numeric NRI RESL score.", "snapshot", "Risk context"),
    ]
    columns = [
        "feature_table",
        "feature_name",
        "category",
        "definition",
        "unit",
        "source_tables",
        "transformation",
        "temporal_grain",
        "infographic_use",
    ]
    result = [dict(common, **dict(zip(columns, row, strict=True))) for row in rows]
    model_exclusions = {
        "fema_event_count": "Replaced by combined extreme_event_count.",
        "noaa_extreme_event_count": "Replaced by combined extreme_event_count.",
        "risk_rating": "Used to define the within-risk-group analysis population.",
        "risk_score": "Excluded from the within-risk-group feature display.",
        "community_resilience_score": "Excluded from the within-risk-group feature display.",
        "expected_annual_loss_score": "Excluded from the within-risk-group feature display.",
        "social_vulnerability_score": "Excluded from the within-risk-group feature display.",
        "median_home_value_usd": "Excluded from the within-risk-group feature display.",
        "median_gross_rent_usd_month": "Excluded from the within-risk-group feature display.",
        "median_owner_costs_mortgage_usd_month": "Excluded from the within-risk-group feature display.",
        "price_drops_yoy": "Excluded from the within-risk-group feature display.",
        "housing_cost_pct_income": "Replaced by ownership-cost component shares.",
        "median_property_taxes_usd_year": "Replaced by property_taxes_pct_income.",
        "per_capita_personal_income_usd": "Replaced by BEA per-capita income components.",
        "net_earnings_per_capita_usd": "Replaced by requested earnings-by-place-of-work component.",
    }
    for row in result:
        reason = model_exclusions.get(str(row["feature_name"]))
        if reason:
            row["retained"] = False
            row["exclusion_reason"] = reason
    result.extend(
        [
            dict(
                common,
                feature_table=None,
                feature_name="median_ppsf_yoy_ten_year_average",
                category="Housing market",
                definition="Ten-year average of the primary PPSF YOY outcome.",
                unit="proportion",
                source_tables="mart.redfin_county_monthly",
                transformation="Average across months and years.",
                temporal_grain="ten-year county summary",
                infographic_use=None,
                retained=False,
                exclusion_reason="Excluded from feature ranking because it duplicates the primary outcome.",
            ),
            dict(
                common,
                feature_table=None,
                feature_name="accommodation_food_wage_share",
                category="Economic and affordability",
                definition="Accommodation and food-service wages as a share of total wages.",
                unit="percent",
                source_tables="mart.statsamerica_cew_county_sector_annual",
                transformation="Sector wages divided by county total wages.",
                temporal_grain="annual",
                infographic_use=None,
                retained=False,
                exclusion_reason="Excluded from the infographic feature comparison after editorial review.",
            ),
        ]
    )
    return result


def create_feature_marts(con) -> None:
    """Create the five requested domain feature marts and their catalog."""

    con.execute("CREATE SCHEMA IF NOT EXISTS feature")
    insurance_annual_usd = _weighted_midpoint_sql(INSURANCE_COST_BINS)
    electricity_monthly_usd = _weighted_midpoint_sql(ELECTRICITY_COST_BINS)
    gas_monthly_usd = _weighted_midpoint_sql(GAS_COST_BINS)
    water_annual_usd = _weighted_midpoint_sql(WATER_COST_BINS)
    fuel_annual_usd = _weighted_midpoint_sql(FUEL_COST_BINS)

    con.execute("DROP TABLE IF EXISTS feature.county_economic_annual")
    con.execute(
        f"""
        CREATE TABLE feature.county_economic_annual AS
        WITH keys AS (
            SELECT fips, year FROM mart.acs_county_economic_annual
            UNION
            SELECT fips, year FROM mart.acs_county_affordability_annual
            UNION
            SELECT fips, year FROM mart.statsamerica_bea_per_capita_income_annual
            UNION
            SELECT fips, year FROM mart.statsamerica_bea_personal_income_annual
            UNION
            SELECT fips, year FROM mart.statsamerica_cew_county_annual
        ),
        econ AS (
            SELECT
                fips,
                year,
                max(try_cast(dp03_income_and_benefits_total_households_median_household_income_est AS DOUBLE))
                    AS dp03_median_household_income_usd,
                max(try_cast(dp03_civilian_labor_force_unemployment_rate_pct AS DOUBLE))
                    AS unemployment_rate_pct
            FROM mart.acs_county_economic_annual
            GROUP BY fips, year
        ),
        afford_components AS (
            SELECT
                fips,
                year,
                max(try_cast(median_household_income AS DOUBLE)) AS affordability_median_household_income_usd,
                max(try_cast(median_home_value AS DOUBLE)) AS median_home_value_usd,
                max(try_cast(median_gross_rent AS DOUBLE)) AS median_gross_rent_usd_month,
                max(try_cast(median_owner_costs_mortgage AS DOUBLE)) AS median_owner_costs_mortgage_usd_month,
                max(try_cast(housing_cost_pct_income AS DOUBLE)) AS housing_cost_pct_income,
                max(try_cast(owner_mortgage_cost_burden_30pct_plus AS DOUBLE))
                    AS owner_cost_burden_30pct_plus_households,
                max(try_cast(
                    dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_est
                    AS DOUBLE
                )) AS owner_cost_burden_denominator_households,
                max(try_cast(
                    dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_30_0_to_34_9_percent_pct
                    AS DOUBLE
                )) AS owner_cost_burden_30_to_34_pct,
                max(try_cast(
                    dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_35_0_percent_or_more_pct
                    AS DOUBLE
                )) AS owner_cost_burden_35_plus_pct,
                max(try_cast(median_property_taxes AS DOUBLE)) AS median_property_taxes_usd_year
                ,{insurance_annual_usd} AS homeowners_insurance_annual_usd
                ,(({electricity_monthly_usd}) + ({gas_monthly_usd})) * 12
                    + ({water_annual_usd}) + ({fuel_annual_usd})
                    AS utilities_annual_usd
            FROM mart.acs_county_affordability_annual
            GROUP BY fips, year
        ),
        afford AS (
            SELECT
                * EXCLUDE (
                    owner_cost_burden_30pct_plus_households,
                    owner_cost_burden_denominator_households,
                    owner_cost_burden_30_to_34_pct,
                    owner_cost_burden_35_plus_pct
                ),
                coalesce(
                    owner_cost_burden_30pct_plus_households * 100.0
                        / nullif(owner_cost_burden_denominator_households, 0),
                    owner_cost_burden_30_to_34_pct + owner_cost_burden_35_plus_pct
                ) AS owner_cost_burden_30pct_plus_pct
            FROM afford_components
        ),
        bea AS (
            SELECT
                fips,
                year,
                max(net_earnings_by_place_of_residence_thousands) * 1000.0
                    / nullif(max(population), 0) AS net_earnings_per_capita_usd,
                max(earnings_by_place_of_work_thousands) * 1000.0
                    / nullif(max(population), 0) AS earnings_by_place_of_work_per_capita_usd,
                max(dividends_interest_rent_thousands) * 1000.0
                    / nullif(max(population), 0) AS dividends_interest_rent_per_capita_usd,
                max(transfer_receipts_thousands) * 1000.0
                    / nullif(max(population), 0) AS transfer_receipts_per_capita_usd
            FROM mart.statsamerica_bea_personal_income_annual
            GROUP BY fips, year
        ),
        pci AS (
            SELECT
                fips,
                year,
                max(per_capita_personal_income_dollars)::DOUBLE AS per_capita_personal_income_usd
            FROM mart.statsamerica_bea_per_capita_income_annual
            GROUP BY fips, year
        ),
        cew AS (
            SELECT
                fips,
                year,
                max(avg_annual_wage_dollars) AS average_annual_wage_usd
            FROM mart.statsamerica_cew_county_annual
            GROUP BY fips, year
        )
        SELECT
            lpad(keys.fips, 5, '0') AS fips,
            keys.year,
            coalesce(afford.affordability_median_household_income_usd, econ.dp03_median_household_income_usd)
                AS median_household_income_usd,
            econ.unemployment_rate_pct,
            afford.median_home_value_usd,
            afford.median_gross_rent_usd_month,
            afford.median_owner_costs_mortgage_usd_month,
            afford.housing_cost_pct_income,
            afford.owner_cost_burden_30pct_plus_pct,
            afford.median_property_taxes_usd_year,
            afford.homeowners_insurance_annual_usd
                / nullif(coalesce(
                    afford.affordability_median_household_income_usd,
                    econ.dp03_median_household_income_usd
                ), 0) * 100.0 AS homeowners_insurance_pct_income,
            afford.median_property_taxes_usd_year
                / nullif(coalesce(
                    afford.affordability_median_household_income_usd,
                    econ.dp03_median_household_income_usd
                ), 0) * 100.0 AS property_taxes_pct_income,
            afford.utilities_annual_usd
                / nullif(coalesce(
                    afford.affordability_median_household_income_usd,
                    econ.dp03_median_household_income_usd
                ), 0) * 100.0 AS utilities_pct_income,
            pci.per_capita_personal_income_usd,
            bea.net_earnings_per_capita_usd,
            bea.earnings_by_place_of_work_per_capita_usd,
            bea.dividends_interest_rent_per_capita_usd,
            bea.transfer_receipts_per_capita_usd,
            cew.average_annual_wage_usd,
            afford.fips IS NOT NULL AS has_acs_affordability,
            econ.fips IS NOT NULL AS has_acs_economic,
            bea.fips IS NOT NULL OR pci.fips IS NOT NULL AS has_bea,
            cew.fips IS NOT NULL AS has_cew
        FROM keys
        LEFT JOIN econ USING (fips, year)
        LEFT JOIN afford USING (fips, year)
        LEFT JOIN pci USING (fips, year)
        LEFT JOIN bea USING (fips, year)
        LEFT JOIN cew USING (fips, year)
        WHERE keys.fips IS NOT NULL AND keys.year IS NOT NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS feature.county_demographic_annual")
    con.execute(
        """
        CREATE TABLE feature.county_demographic_annual AS
        WITH demo AS (
            SELECT
                fips,
                year,
                max(try_cast(total_population AS DOUBLE)) AS total_population,
                max(try_cast(dp05_total_population_65_plus_pct AS DOUBLE)) AS age_65_plus_pct,
                max(try_cast(
                    dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct
                    AS DOUBLE
                )) AS disability_pct,
                max(try_cast(
                    dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct
                    AS DOUBLE
                )) AS communication_barrier_pct,
                100.0 - max(try_cast(
                    dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct
                    AS DOUBLE
                )) AS no_broadband_pct
            FROM mart.acs_county_demographic_annual
            GROUP BY fips, year
        ),
        migration AS (
            SELECT
                fips,
                year,
                max(total_net_migration)::DOUBLE AS total_net_migration
            FROM mart.statsamerica_population_components_annual
            GROUP BY fips, year
        ),
        keys AS (
            SELECT fips, year FROM demo
            UNION
            SELECT fips, year FROM migration
        )
        SELECT
            lpad(keys.fips, 5, '0') AS fips,
            keys.year,
            demo.total_population,
            demo.age_65_plus_pct,
            demo.disability_pct,
            demo.communication_barrier_pct,
            demo.no_broadband_pct,
            migration.total_net_migration,
            migration.total_net_migration / nullif(demo.total_population, 0) * 100.0
                AS net_migration_rate_pct,
            demo.fips IS NOT NULL AS has_acs_demographic,
            migration.fips IS NOT NULL AS has_statsamerica_migration
        FROM keys
        LEFT JOIN demo USING (fips, year)
        LEFT JOIN migration USING (fips, year)
        WHERE keys.fips IS NOT NULL AND keys.year IS NOT NULL
        """
    )

    excluded_fema = ", ".join(f"'{value.replace(chr(39), chr(39) * 2)}'" for value in EXCLUDED_FEMA_INCIDENT_TYPES)
    con.execute("DROP TABLE IF EXISTS feature.county_climate_monthly")
    con.execute(
        f"""
        CREATE TABLE feature.county_climate_monthly AS
        WITH events AS (
            SELECT
                fips,
                event_start_month AS climate_month,
                count(*) FILTER (
                    WHERE has_fema AND event_type NOT IN ({excluded_fema})
                ) AS fema_event_count,
                count(*) FILTER (WHERE has_noaa) AS noaa_extreme_event_count,
                count(*) AS extreme_event_count,
                sum(CASE WHEN has_noaa THEN coalesce(total_damage_amount, 0) ELSE 0 END)
                    AS noaa_extreme_damage_usd
            FROM mart.climate_events
            WHERE fips IS NOT NULL
              AND event_start_month IS NOT NULL
              AND (
                    (has_fema AND event_type NOT IN ({excluded_fema}))
                    OR has_noaa
              )
            GROUP BY fips, climate_month
        ),
        keys AS (
            SELECT fips, weather_month AS climate_month FROM mart.ncei_county_weather_monthly
            UNION
            SELECT fips, climate_month FROM events
        )
        SELECT
            lpad(keys.fips, 5, '0') AS fips,
            keys.climate_month,
            extract(year FROM keys.climate_month)::INTEGER AS year,
            extract(month FROM keys.climate_month)::INTEGER AS month,
            weather.avg_temperature_f,
            weather.min_temperature_f,
            weather.max_temperature_f,
            weather.precipitation_inches,
            weather.avg_temperature_anomaly_f,
            weather.precipitation_anomaly_inches,
            coalesce(events.fema_event_count, 0)::INTEGER AS fema_event_count,
            coalesce(events.noaa_extreme_event_count, 0)::INTEGER AS noaa_extreme_event_count,
            coalesce(events.extreme_event_count, 0)::INTEGER AS extreme_event_count,
            coalesce(events.noaa_extreme_damage_usd, 0) AS noaa_extreme_damage_usd,
            weather.fips IS NOT NULL AS has_ncei_weather
        FROM keys
        LEFT JOIN mart.ncei_county_weather_monthly AS weather
            ON keys.fips = weather.fips AND keys.climate_month = weather.weather_month
        LEFT JOIN events
            ON keys.fips = events.fips AND keys.climate_month = events.climate_month
        WHERE keys.fips IS NOT NULL AND keys.climate_month IS NOT NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS feature.county_housing_monthly")
    con.execute(
        """
        CREATE TABLE feature.county_housing_monthly AS
        WITH raw_typed AS (
            SELECT
                lpad(fips, 5, '0') AS fips,
                REGION AS county_name,
                STATE_CODE AS state_code,
                period_begin AS housing_month,
                period_end,
                try_cast(MEDIAN_SALE_PRICE AS DOUBLE) AS median_sale_price_usd,
                try_cast(MEDIAN_PPSF AS DOUBLE) AS median_ppsf_usd,
                try_cast(MEDIAN_PPSF_YOY AS DOUBLE) AS median_ppsf_yoy,
                try_cast(AVG_SALE_TO_LIST_YOY AS DOUBLE) AS avg_sale_to_list_yoy,
                try_cast(HOMES_SOLD_YOY AS DOUBLE) AS homes_sold_yoy,
                try_cast(INVENTORY_YOY AS DOUBLE) AS inventory_yoy,
                try_cast(NEW_LISTINGS_YOY AS DOUBLE) AS new_listings_yoy,
                try_cast(MEDIAN_DOM_YOY AS DOUBLE) AS median_dom_yoy,
                try_cast(PRICE_DROPS_YOY AS DOUBLE) AS price_drops_yoy
            FROM mart.redfin_county_monthly
            WHERE coalesce(property_type, PROPERTY_TYPE_1) = 'All Residential'
              AND fips IS NOT NULL
              AND period_begin IS NOT NULL
        ),
        typed AS (
            SELECT
                fips,
                max(county_name) AS county_name,
                max(state_code) AS state_code,
                housing_month,
                max(period_end) AS period_end,
                max(median_sale_price_usd) AS median_sale_price_usd,
                max(median_ppsf_usd) AS median_ppsf_usd,
                max(median_ppsf_yoy) AS median_ppsf_yoy,
                max(avg_sale_to_list_yoy) AS avg_sale_to_list_yoy,
                max(homes_sold_yoy) AS homes_sold_yoy,
                max(inventory_yoy) AS inventory_yoy,
                max(new_listings_yoy) AS new_listings_yoy,
                max(median_dom_yoy) AS median_dom_yoy,
                max(price_drops_yoy) AS price_drops_yoy
            FROM raw_typed
            GROUP BY fips, housing_month
        ),
        cleaned AS (
            SELECT
                * REPLACE (
                    CASE WHEN median_ppsf_yoy <= -888888000 THEN NULL ELSE median_ppsf_yoy END AS median_ppsf_yoy,
                    CASE WHEN avg_sale_to_list_yoy <= -888888000 THEN NULL ELSE avg_sale_to_list_yoy END AS avg_sale_to_list_yoy,
                    CASE WHEN homes_sold_yoy <= -888888000 THEN NULL ELSE homes_sold_yoy END AS homes_sold_yoy,
                    CASE WHEN inventory_yoy <= -888888000 THEN NULL ELSE inventory_yoy END AS inventory_yoy,
                    CASE WHEN new_listings_yoy <= -888888000 THEN NULL ELSE new_listings_yoy END AS new_listings_yoy,
                    CASE WHEN median_dom_yoy <= -888888000 THEN NULL ELSE median_dom_yoy END AS median_dom_yoy,
                    CASE WHEN price_drops_yoy <= -888888000 THEN NULL ELSE price_drops_yoy END AS price_drops_yoy
                )
            FROM typed
        ),
        standardized AS (
            SELECT
                *,
                (median_ppsf_yoy - avg(median_ppsf_yoy) OVER ()) / nullif(stddev_samp(median_ppsf_yoy) OVER (), 0)
                    AS ppsf_z,
                (avg_sale_to_list_yoy - avg(avg_sale_to_list_yoy) OVER ()) / nullif(stddev_samp(avg_sale_to_list_yoy) OVER (), 0)
                    AS sale_to_list_z,
                (homes_sold_yoy - avg(homes_sold_yoy) OVER ()) / nullif(stddev_samp(homes_sold_yoy) OVER (), 0)
                    AS homes_sold_z,
                -1 * (inventory_yoy - avg(inventory_yoy) OVER ()) / nullif(stddev_samp(inventory_yoy) OVER (), 0)
                    AS inventory_z
            FROM cleaned
        )
        SELECT
            * EXCLUDE (ppsf_z, sale_to_list_z, homes_sold_z, inventory_z),
            (coalesce(ppsf_z, 0) + coalesce(sale_to_list_z, 0)
                + coalesce(homes_sold_z, 0) + coalesce(inventory_z, 0))
                / nullif(
                    (ppsf_z IS NOT NULL)::INTEGER + (sale_to_list_z IS NOT NULL)::INTEGER
                    + (homes_sold_z IS NOT NULL)::INTEGER + (inventory_z IS NOT NULL)::INTEGER,
                    0
                ) AS housing_market_index
        FROM standardized
        """
    )

    con.execute("DROP TABLE IF EXISTS feature.county_risk")
    con.execute(
        """
        CREATE TABLE feature.county_risk AS
        SELECT
            lpad(fips, 5, '0') AS fips,
            STATEABBRV AS state_abbrev,
            COUNTY AS county_name,
            nri_version,
            risk_score,
            risk_rating,
            try_cast(EAL_SCORE AS DOUBLE) AS expected_annual_loss_score,
            try_cast(SOVI_SCORE AS DOUBLE) AS social_vulnerability_score,
            try_cast(RESL_SCORE AS DOUBLE) AS community_resilience_score
        FROM mart.nri_county_risk
        WHERE fips IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY fips ORDER BY nri_version DESC NULLS LAST) = 1
        """
    )

    con.execute("DROP TABLE IF EXISTS feature.catalog")
    catalog = pd.DataFrame(_catalog_rows())
    con.register("_feature_catalog_df", catalog)
    try:
        con.execute("CREATE TABLE feature.catalog AS SELECT * FROM _feature_catalog_df")
    finally:
        con.unregister("_feature_catalog_df")

    con.execute("CREATE INDEX IF NOT EXISTS idx_feature_economic_fips_year ON feature.county_economic_annual (fips, year)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_feature_demographic_fips_year ON feature.county_demographic_annual (fips, year)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_feature_climate_fips_month ON feature.county_climate_monthly (fips, climate_month)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_feature_housing_fips_month ON feature.county_housing_monthly (fips, housing_month)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_feature_risk_fips ON feature.county_risk (fips)")

    _assert_unique_key(con, "feature.county_economic_annual", ("fips", "year"))
    _assert_unique_key(con, "feature.county_demographic_annual", ("fips", "year"))
    _assert_unique_key(con, "feature.county_climate_monthly", ("fips", "climate_month"))
    _assert_unique_key(con, "feature.county_housing_monthly", ("fips", "housing_month"))
    _assert_unique_key(con, "feature.county_risk", ("fips",))
