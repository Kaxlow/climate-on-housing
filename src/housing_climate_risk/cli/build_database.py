from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from housing_climate_risk.paths import ACS_DIR, CLIMATE_DAMAGE_DIR, CLIMATE_DIR, DATA_DIR, FEMA_DIR, FIPSGEO_DIR, HOUSING_DIR


DATABASE_PATH = DATA_DIR / "quoll.duckdb"


RAW_FILES = {
    "fips_master_v2": FIPSGEO_DIR / "fips_master_v2.csv",
    "redfin_housing_market_by_county": HOUSING_DIR / "Redfin-Housing-Market-By-County.csv",
    "nri_table_counties": FEMA_DIR / "NRI_Table_Counties.csv",
    "fema_disaster_declarations": FEMA_DIR / "FEMA_Disaster_Declarations.csv",
    "fema_web_disaster_summaries": CLIMATE_DAMAGE_DIR
    / "raw"
    / "fema_web_disaster_summaries"
    / "FemaWebDisasterSummaries.csv",
    "noaa_storm_events_county_damage": CLIMATE_DAMAGE_DIR / "noaa_storm_events_county_damage.csv",
    "noaa_storm_events_zone_county_mapping": CLIMATE_DAMAGE_DIR / "noaa_storm_events_zone_county_mapping.csv",
    "ncei_climate_at_a_glance_county_monthly": CLIMATE_DIR / "ncei_climate_at_a_glance_county_monthly.csv",
}


ACS_DATA_PATTERNS = {
    "economic": [r"_dp03_"],
    "demographic": [r"_dp02_", r"_dp05_", r"_migration_", r"_population_"],
    "affordability": [r"_affordability_", r"_dp04_", r"_s250", r"_b251", r"_housing_financial_"],
}


ACS_KEY_COLUMNS = {"year", "state", "county"}
ACS_VARIABLE_RE = re.compile(r"^[A-Z]+\d+(?:_C\d+)?_\d+(?:E|M|PE|PM)$")
ACS_SPECIAL_NUMERIC_VALUES = {-222222222, -333333333, -555555555, -666666666, -888888888, -999999999}


def _load_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is not installed. Install dependencies with `pip install -e .` "
            "or `pip install duckdb`, then rerun `build-database`."
        ) from exc
    return duckdb


def _configure_connection(con) -> None:
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET threads = 2")


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str | Path) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def _table(schema: str, name: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(name)}"


def _sanitize_table_name(path: Path) -> str:
    stem = path.name
    for suffix in [".csv", ".parquet", ".feather"]:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    name = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_").lower()
    return re.sub(r"_+", "_", name)


def _column_names(con, schema: str, table_name: str) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema, table_name],
        ).fetchall()
    ]


def _register_file_metadata(con, *, table_schema: str, table_name: str, source_path: Path) -> None:
    columns = _column_names(con, table_schema, table_name)
    row_count = con.execute(f"SELECT count(*) FROM {_table(table_schema, table_name)}").fetchone()[0]
    resolved_source_path = source_path.resolve()
    source_folder = (
        str(resolved_source_path.parent.relative_to(DATA_DIR))
        if resolved_source_path.is_relative_to(DATA_DIR)
        else str(resolved_source_path.parent)
    )
    con.execute(
        """
        INSERT INTO meta.files (
            table_schema,
            table_name,
            filename,
            source_folder,
            source_path,
            loaded_at,
            row_count,
            detected_columns
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            table_schema,
            table_name,
            source_path.name,
            source_folder,
            str(resolved_source_path),
            datetime.now(timezone.utc).isoformat(),
            row_count,
            json.dumps(columns),
        ],
    )


def _load_csv_raw(con, *, table_name: str, source_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    con.execute(f"DROP TABLE IF EXISTS {_table('raw', table_name)}")
    con.execute(
        f"""
        CREATE TABLE {_table('raw', table_name)} AS
        SELECT *
        FROM read_csv_auto(
            {_quote_literal(source_path)},
            header = true,
            all_varchar = true,
            ignore_errors = true,
            union_by_name = true
        )
        """
    )
    _register_file_metadata(con, table_schema="raw", table_name=table_name, source_path=source_path)


def _load_feather_raw(con, *, table_name: str, source_path: Path) -> None:
    if not source_path.exists():
        return
    df = pd.read_feather(source_path)
    for column in df.columns:
        df[column] = df[column].map(_json_or_scalar).astype("string")
    con.register("_county_processed_df", df)
    con.execute(f"DROP TABLE IF EXISTS {_table('raw', table_name)}")
    con.execute(f"CREATE TABLE {_table('raw', table_name)} AS SELECT * FROM _county_processed_df")
    con.unregister("_county_processed_df")
    _register_file_metadata(con, table_schema="raw", table_name=table_name, source_path=source_path)


def _json_or_scalar(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return value


def _load_raw(con) -> dict[str, list[str]]:
    loaded: dict[str, list[str]] = {"acs": []}
    for table_name, source_path in RAW_FILES.items():
        _load_csv_raw(con, table_name=table_name, source_path=source_path)
        loaded[table_name] = [str(source_path)]

    for source_path in sorted(ACS_DIR.glob("*.csv")):
        table_name = _sanitize_table_name(source_path)
        _load_csv_raw(con, table_name=table_name, source_path=source_path)
        loaded["acs"].append(table_name)

    _load_feather_raw(con, table_name="county_processed_data", source_path=DATA_DIR / "county_processed_data.feather")
    return loaded


def _create_meta(con) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS meta")
    con.execute("DROP TABLE IF EXISTS meta.files")
    con.execute(
        """
        CREATE TABLE meta.files (
            table_schema VARCHAR,
            table_name VARCHAR,
            filename VARCHAR,
            source_folder VARCHAR,
            source_path VARCHAR,
            loaded_at VARCHAR,
            row_count BIGINT,
            detected_columns JSON
        )
        """
    )


def _clean_acs_label(label: str) -> str:
    text = str(label)
    text = text.replace(":", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace("'", "")
    text = text.replace("$", " dollars ")
    text = text.replace("%", " percent ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    replacements = {
        "percentage_of_families_and_people_whose_income_in_the_past_12_months_is_below_the_poverty_level": "poverty_status",
        "income_in_the_past_12_months_below_poverty_level": "poverty_status",
        "in_the_past_12_months": "past_12_months",
        "with_related_children_of_the_householder": "with_related_children",
        "related_children_of_the_householder": "related_children",
        "population_16_years_and_over": "population_16_plus",
        "65_years_and_over": "65_plus",
        "25_years_and_over": "25_plus",
        "18_years_and_over": "18_plus",
        "under_18_years": "under_18",
        "under_5_years": "under_5",
        "owner_occupied_housing_units": "owner_occupied_units",
        "renter_occupied_housing_units": "renter_occupied_units",
        "with_a_mortgage": "mortgage",
        "without_a_mortgage": "no_mortgage",
        "margin_of_error": "moe",
        "percentage": "pct",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _acs_label_parts(label: str) -> list[str]:
    stat_words = {
        "estimate",
        "margin of error",
        "percent",
        "percent margin of error",
    }
    generic_headings = {
        "employment_status",
        "housing_occupancy",
        "sex_and_age",
    }

    parts: list[str] = []
    for raw_part in str(label).split("!!"):
        raw_part = raw_part.strip()
        if raw_part.lower() in stat_words:
            continue
        part = _clean_acs_label(raw_part)
        if not part or part in stat_words:
            continue
        if part in generic_headings:
            continue
        if part not in parts:
            parts.append(part)
    return parts


def _concept_is_specific(concept: str) -> bool:
    generic = {
        "acs demographic and housing estimates",
        "selected economic characteristics",
        "selected housing characteristics",
        "selected social characteristics in the united states",
    }
    return str(concept).strip().lower() not in generic


def _acs_stat_suffix(variable: str, label: str) -> str:
    label_lower = str(label).lower()
    if variable.endswith("PM") or label_lower.startswith("percent margin of error"):
        return "pct_moe"
    if variable.endswith("PE") or label_lower.startswith("percent"):
        return "pct"
    if variable.endswith("M") or label_lower.startswith("margin of error"):
        return "moe"
    return "est"


def _acs_feature_name(variable: str, label: str, group: str, concept: str = "") -> str:
    parts = _acs_label_parts(label)
    if _concept_is_specific(concept) and (not parts or parts == ["total"] or parts[0] == "total"):
        concept_part = _clean_acs_label(concept)
        parts = [concept_part, *[part for part in parts if part != concept_part]]
    if not parts:
        parts = [re.sub(r"[^0-9A-Za-z]+", "_", variable).strip("_").lower()]

    compact_parts: list[str] = []
    for part in parts:
        if compact_parts and part == compact_parts[-1]:
            continue
        if any(part == existing or part.startswith(f"{existing}_") for existing in compact_parts):
            continue
        compact_parts.append(part)

    label_slug = "_".join(compact_parts)
    group_slug = re.sub(r"[^0-9A-Za-z]+", "_", str(group or "")).strip("_").lower()
    prefix = f"{group_slug}_" if group_slug else ""
    suffix = _acs_stat_suffix(variable, label)
    base = f"{prefix}{label_slug}"
    max_chars = 140
    max_base_chars = max_chars - len(suffix) - 1
    if len(base) > max_base_chars:
        base = base[:max_base_chars].rsplit("_", 1)[0]
    feature = f"{base}_{suffix}"
    if feature[0].isdigit():
        feature = f"acs_{feature}"
    return feature


def _dedupe_feature_alias(alias: str, variable: str, used_aliases: set[str]) -> str:
    if alias not in used_aliases:
        return alias
    digest = hashlib.sha1(variable.encode("utf-8")).hexdigest()[:8]
    suffix = f"_v{digest}"
    return f"{alias[: 140 - len(suffix)]}{suffix}"


def _build_acs_variable_lookup() -> dict[str, dict[str, str]]:
    frames = []
    for source_path in sorted(ACS_DIR.glob("*variable_dictionary*.csv")):
        dictionary_df = pd.read_csv(source_path, dtype=str)
        dictionary_df["dictionary_file"] = source_path.name
        frames.append(dictionary_df)
    if not frames:
        return {}

    variables_df = pd.concat(frames, ignore_index=True)
    variables_df["year_sort"] = pd.to_numeric(variables_df["year"], errors="coerce").fillna(0)
    variables_df = variables_df.sort_values(["variable", "year_sort"], ascending=[True, False])
    variables_df = variables_df.drop_duplicates(subset=["variable"], keep="first")

    lookup: dict[str, dict[str, str]] = {}
    used_features: dict[str, str] = {}
    for row in variables_df.to_dict("records"):
        variable = str(row["variable"])
        feature = _acs_feature_name(
            variable,
            str(row.get("label", "")),
            str(row.get("group", "")),
            str(row.get("concept", "")),
        )
        if feature in used_features and used_features[feature] != variable:
            digest = hashlib.sha1(variable.encode("utf-8")).hexdigest()[:8]
            suffix = f"_v{digest}"
            feature = f"{feature[: 140 - len(suffix)]}{suffix}"
        used_features[feature] = variable
        lookup[variable] = {
            "feature_name": feature,
            "label": str(row.get("label", "")),
            "concept": str(row.get("concept", "")),
            "predicate_type": str(row.get("predicate_type", "")),
            "group": str(row.get("group", "")),
            "dictionary_file": str(row.get("dictionary_file", "")),
        }
    return lookup


def _create_acs_variable_features(con, acs_table_names: list[str], variable_lookup: dict[str, dict[str, str]]) -> None:
    rows = []
    for table_name in acs_table_names:
        for column in _column_names(con, "raw", table_name):
            if column not in variable_lookup:
                continue
            row = variable_lookup[column]
            rows.append(
                {
                    "source_table": table_name,
                    "variable": column,
                    "feature_name": row["feature_name"],
                    "label": row["label"],
                    "concept": row["concept"],
                    "predicate_type": row["predicate_type"],
                    "group": row["group"],
                    "dictionary_file": row["dictionary_file"],
                }
            )

    con.execute("DROP TABLE IF EXISTS meta.acs_variable_features")
    feature_df = pd.DataFrame(
        rows,
        columns=[
            "source_table",
            "variable",
            "feature_name",
            "label",
            "concept",
            "predicate_type",
            "group",
            "dictionary_file",
        ],
    )
    con.register("_acs_variable_features_df", feature_df)
    con.execute("CREATE TABLE meta.acs_variable_features AS SELECT * FROM _acs_variable_features_df")
    con.unregister("_acs_variable_features_df")


def _create_ref_tables(con) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS ref")
    con.execute("DROP TABLE IF EXISTS ref.counties")
    con.execute(
        """
        CREATE TABLE ref.counties AS
        SELECT DISTINCT
            lpad(fips, 5, '0') AS fips,
            substr(lpad(fips, 5, '0'), 1, 2) AS state_fips,
            county_name,
            state,
            state_long,
            msa_code,
            msa_name,
            msa_type,
            csa_code,
            csa_name
        FROM raw.fips_master_v2
        WHERE fips IS NOT NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS ref.states")
    con.execute(
        """
        CREATE TABLE ref.states AS
        SELECT DISTINCT
            state_fips,
            state,
            state_long
        FROM ref.counties
        WHERE state_fips IS NOT NULL
          AND state IS NOT NULL
        ORDER BY state_fips
        """
    )


def _redfin_fips_expr() -> str:
    return """
    (
        SELECT c.fips
        FROM ref.counties c
        WHERE lower(raw_redfin.STATE_CODE) = lower(c.state)
          AND (
            lower(raw_redfin.REGION) = lower(c.county_name || ', ' || c.state)
            OR _normalize_place_name(split_part(raw_redfin.REGION, ',', 1)) = _normalize_place_name(c.county_name)
            OR _normalize_place_name(regexp_replace(split_part(raw_redfin.REGION, ',', 1), '\\s+County$', '', 'i')) = _normalize_place_name(c.county_name)
            OR _normalize_place_name(split_part(raw_redfin.REGION, ',', 1)) = _normalize_place_name(regexp_replace(c.county_name, '\\s+(County|Parish|City and Borough|Borough|Census Area|Municipality|city)$', '', 'i'))
          )
        LIMIT 1
    )
    """


def _create_core_marts(con) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS mart")
    con.create_function(
        "_normalize_place_name",
        _normalize_place_name,
        ["VARCHAR"],
        "VARCHAR",
    )

    con.execute("DROP TABLE IF EXISTS mart.redfin_county_monthly")
    con.execute(
        f"""
        CREATE TABLE mart.redfin_county_monthly AS
        SELECT
            {_redfin_fips_expr()} AS fips,
            try_cast(PERIOD_BEGIN AS DATE) AS period_begin,
            try_cast(PERIOD_END AS DATE) AS period_end,
            PROPERTY_TYPE AS property_type,
            raw_redfin.*
        FROM raw.redfin_housing_market_by_county AS raw_redfin
        """
    )

    con.execute("DROP TABLE IF EXISTS mart.nri_county_risk")
    con.execute(
        """
        CREATE TABLE mart.nri_county_risk AS
        SELECT
            lpad(STCOFIPS, 5, '0') AS fips,
            NRI_VER AS nri_version,
            try_cast(RISK_SCORE AS DOUBLE) AS risk_score,
            RISK_RATNG AS risk_rating,
            raw_nri.*
        FROM raw.nri_table_counties AS raw_nri
        WHERE STCOFIPS IS NOT NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS mart.fema_disaster_declarations")
    con.execute(
        """
        CREATE TABLE mart.fema_disaster_declarations AS
        WITH parsed_fema AS (
            SELECT
                try_cast(incidentBeginDate AS TIMESTAMP) AS parsed_incident_begin_date,
                try_cast(incidentEndDate AS TIMESTAMP) AS parsed_incident_end_date,
                raw_fema.*
            FROM raw.fema_disaster_declarations AS raw_fema
        ),
        incident_type_duration AS (
            SELECT
                incidentType,
                avg(date_diff('day', parsed_incident_begin_date, parsed_incident_end_date)) AS average_duration_days
            FROM parsed_fema
            WHERE parsed_incident_begin_date IS NOT NULL
              AND parsed_incident_end_date IS NOT NULL
              AND parsed_incident_end_date >= parsed_incident_begin_date
              AND incidentType IS NOT NULL
            GROUP BY incidentType
        )
        SELECT
            lpad(fipsStateCode, 2, '0') || lpad(fipsCountyCode, 3, '0') AS fips,
            lpad(fipsStateCode, 2, '0') AS state_fips,
            try_cast(fyDeclared AS INTEGER) AS declared_year,
            try_cast(declarationDate AS TIMESTAMP) AS declaration_date,
            parsed_incident_begin_date AS incident_begin_date,
            coalesce(
                parsed_incident_end_date,
                parsed_incident_begin_date
                    + CAST(round(duration.average_duration_days) AS BIGINT) * INTERVAL 1 DAY
            ) AS incident_end_date,
            parsed_fema.* EXCLUDE (parsed_incident_begin_date, parsed_incident_end_date)
        FROM parsed_fema
        LEFT JOIN incident_type_duration AS duration
            ON parsed_fema.incidentType = duration.incidentType
        WHERE fipsStateCode IS NOT NULL
          AND fipsCountyCode IS NOT NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS mart.fema_disaster_financial_assistance")
    con.execute(
        """
        CREATE TABLE mart.fema_disaster_financial_assistance AS
        SELECT
            try_cast(summary.disasterNumber AS INTEGER) AS disaster_number,
            try_cast(summary.totalNumberIaApproved AS INTEGER) AS ia_approved_application_count,
            try_cast(summary.totalAmountIhpApproved AS DOUBLE) AS ihp_approved_amount,
            try_cast(summary.totalAmountHaApproved AS DOUBLE) AS housing_assistance_approved_amount,
            try_cast(summary.totalAmountOnaApproved AS DOUBLE) AS other_needs_assistance_approved_amount,
            try_cast(summary.totalObligatedAmountPa AS DOUBLE) AS public_assistance_obligated_amount,
            try_cast(summary.totalObligatedAmountCatAb AS DOUBLE) AS public_assistance_categories_ab_obligated_amount,
            try_cast(summary.totalObligatedAmountCatC2g AS DOUBLE) AS public_assistance_categories_c_to_g_obligated_amount,
            try_cast(summary.totalObligatedAmountHmgp AS DOUBLE) AS hazard_mitigation_grant_obligated_amount,
            coalesce(try_cast(summary.totalAmountIhpApproved AS DOUBLE), 0)
                + coalesce(try_cast(summary.totalObligatedAmountPa AS DOUBLE), 0)
                + coalesce(try_cast(summary.totalObligatedAmountHmgp AS DOUBLE), 0)
                AS total_fema_financial_assistance_amount,
            try_cast(summary.paLoadDate AS TIMESTAMP) AS public_assistance_load_timestamp,
            try_cast(summary.iaLoadDate AS TIMESTAMP) AS individual_assistance_load_timestamp,
            try_cast(summary.lastRefresh AS TIMESTAMP) AS source_last_refresh_timestamp,
            summary.hash AS source_hash,
            summary.id AS source_record_id,
            summary.*
        FROM raw.fema_web_disaster_summaries AS summary
        WHERE summary.disasterNumber IS NOT NULL
        """
    )

    con.execute("DROP TABLE IF EXISTS mart.noaa_storm_events")
    con.execute(
        """
        CREATE TABLE mart.noaa_storm_events AS
        WITH ref_county_names AS (
            SELECT
                state_fips,
                _normalize_place_name(county_name) AS county_name_norm,
                min(fips) AS fips,
                count(*) AS match_count
            FROM ref.counties
            GROUP BY state_fips, _normalize_place_name(county_name)
        ),
        zone_county_mapping AS (
            SELECT
                lpad(mapped_fips, 5, '0') AS mapped_fips,
                mapped_county_name,
                state_fips,
                cz_fips,
                cz_name,
                mapping_method,
                mapping_confidence,
                mapping_note
            FROM raw.noaa_storm_events_zone_county_mapping
            WHERE mapped_fips IS NOT NULL
              AND trim(mapped_fips) <> ''
        ),
        noaa_resolved AS (
            SELECT
                coalesce(valid_county.fips, name_county.fips, lpad(raw_noaa.county_fips, 5, '0')) AS resolved_fips,
                lpad(raw_noaa.county_fips, 5, '0') AS noaa_county_fips,
                NULL AS zone_mapped_fips,
                NULL AS zone_mapped_county_name,
                NULL AS zone_mapping_method,
                NULL AS zone_mapping_confidence,
                NULL AS zone_mapping_note,
                CASE
                    WHEN valid_county.fips IS NOT NULL THEN 'county_fips'
                    WHEN name_county.fips IS NOT NULL THEN 'county_name'
                    ELSE 'unmatched'
                END AS fips_resolution_method,
                raw_noaa.*
            FROM raw.noaa_storm_events_county_damage AS raw_noaa
            LEFT JOIN ref.counties AS valid_county
                ON lpad(raw_noaa.county_fips, 5, '0') = valid_county.fips
            LEFT JOIN ref_county_names AS name_county
                ON lpad(raw_noaa.state_fips, 2, '0') = name_county.state_fips
               AND _normalize_place_name(raw_noaa.cz_name) = name_county.county_name_norm
               AND name_county.match_count = 1
            WHERE raw_noaa.county_fips IS NOT NULL
              AND raw_noaa.cz_type = 'C'
            UNION ALL
            SELECT
                zone_mapping.mapped_fips AS resolved_fips,
                NULL AS noaa_county_fips,
                zone_mapping.mapped_fips AS zone_mapped_fips,
                zone_mapping.mapped_county_name AS zone_mapped_county_name,
                zone_mapping.mapping_method AS zone_mapping_method,
                zone_mapping.mapping_confidence AS zone_mapping_confidence,
                zone_mapping.mapping_note AS zone_mapping_note,
                'zone_' || zone_mapping.mapping_method AS fips_resolution_method,
                raw_noaa.*
            FROM raw.noaa_storm_events_county_damage AS raw_noaa
            INNER JOIN zone_county_mapping AS zone_mapping
                ON lpad(raw_noaa.state_fips, 2, '0') = zone_mapping.state_fips
               AND raw_noaa.cz_fips = zone_mapping.cz_fips
               AND raw_noaa.cz_name = zone_mapping.cz_name
            WHERE raw_noaa.cz_type = 'Z'
        )
        SELECT
            resolved_fips AS fips,
            lpad(state_fips, 2, '0') AS state_fips_padded,
            try_cast(year AS INTEGER) AS event_year,
            try_cast(substr(begin_yearmonth, 5, 2) AS INTEGER) AS event_month,
            try_strptime(begin_date_time, '%d-%b-%y %H:%M:%S') AS begin_timestamp,
            try_strptime(end_date_time, '%d-%b-%y %H:%M:%S') AS end_timestamp,
            try_cast(injuries_direct AS INTEGER) AS injuries_direct_count,
            try_cast(deaths_direct AS INTEGER) AS deaths_direct_count,
            try_cast(property_damage AS DOUBLE) AS property_damage_amount,
            try_cast(crop_damage AS DOUBLE) AS crop_damage_amount,
            try_cast(total_damage AS DOUBLE) AS total_damage_amount,
            noaa_county_fips,
            fips_resolution_method,
            zone_mapped_fips,
            zone_mapped_county_name,
            zone_mapping_method,
            zone_mapping_confidence,
            zone_mapping_note,
            noaa_resolved.* EXCLUDE (
                resolved_fips,
                noaa_county_fips,
                fips_resolution_method,
                zone_mapped_fips,
                zone_mapped_county_name,
                zone_mapping_method,
                zone_mapping_confidence,
                zone_mapping_note
            )
        FROM noaa_resolved
        """
    )

    con.execute("DROP TABLE IF EXISTS mart.ncei_county_weather_monthly")
    con.execute(
        """
        CREATE TABLE mart.ncei_county_weather_monthly AS
        WITH typed_weather AS (
            SELECT
                lpad(fips, 5, '0') AS fips,
                try_cast(date AS DATE) AS weather_month,
                try_cast(year AS INTEGER) AS year,
                try_cast(month AS INTEGER) AS month,
                parameter,
                try_cast(value AS DOUBLE) AS value,
                try_cast(anomaly AS DOUBLE) AS anomaly,
                try_cast(rank AS DOUBLE) AS rank,
                source_url,
                try_cast(fetched_at AS TIMESTAMP) AS fetched_at
            FROM raw.ncei_climate_at_a_glance_county_monthly
            WHERE fips IS NOT NULL
              AND parameter IN ('tavg', 'tmin', 'tmax', 'pcp')
        )
        SELECT
            weather.fips,
            counties.state_fips,
            counties.state,
            counties.state_long,
            counties.county_name,
            weather.weather_month,
            weather.year,
            weather.month,
            max(value) FILTER (WHERE parameter = 'tavg') AS avg_temperature_f,
            max(value) FILTER (WHERE parameter = 'tmin') AS min_temperature_f,
            max(value) FILTER (WHERE parameter = 'tmax') AS max_temperature_f,
            max(value) FILTER (WHERE parameter = 'pcp') AS precipitation_inches,
            max(anomaly) FILTER (WHERE parameter = 'tavg') AS avg_temperature_anomaly_f,
            max(anomaly) FILTER (WHERE parameter = 'tmin') AS min_temperature_anomaly_f,
            max(anomaly) FILTER (WHERE parameter = 'tmax') AS max_temperature_anomaly_f,
            max(anomaly) FILTER (WHERE parameter = 'pcp') AS precipitation_anomaly_inches,
            max(rank) FILTER (WHERE parameter = 'tavg') AS avg_temperature_rank,
            max(rank) FILTER (WHERE parameter = 'tmin') AS min_temperature_rank,
            max(rank) FILTER (WHERE parameter = 'tmax') AS max_temperature_rank,
            max(rank) FILTER (WHERE parameter = 'pcp') AS precipitation_rank,
            max(fetched_at) AS latest_fetched_at,
            count(DISTINCT parameter) AS observed_parameter_count,
            string_agg(DISTINCT source_url, '; ' ORDER BY source_url) AS source_urls
        FROM typed_weather AS weather
        LEFT JOIN ref.counties AS counties
            ON weather.fips = counties.fips
        WHERE weather.weather_month IS NOT NULL
        GROUP BY
            weather.fips,
            counties.state_fips,
            counties.state,
            counties.state_long,
            counties.county_name,
            weather.weather_month,
            weather.year,
            weather.month
        """
    )

    con.execute("DROP TABLE IF EXISTS mart.county_snapshot")
    con.execute(
        """
        CREATE TABLE mart.county_snapshot AS
        SELECT
            c.*,
            n.nri_version,
            n.risk_score AS nri_risk_score,
            n.risk_rating AS nri_risk_rating,
            processed.has_redfin_data
        FROM ref.counties c
        LEFT JOIN mart.nri_county_risk n
            ON c.fips = n.fips
        LEFT JOIN raw.county_processed_data processed
            ON c.fips = lpad(cast(processed.fips AS VARCHAR), 5, '0')
        """
    )


def _normalize_place_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.lower().strip()
    replacements = {
        "ñ": "n",
        "&": "and",
        ".": "",
        "'": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+(county|parish|city and borough|borough|census area|municipality|city)$", "", text)
    return text.strip()


def _acs_category(table_name: str) -> str | None:
    if any(token in table_name for token in ["metadata", "failures", "variable_dictionary", "coverage"]):
        return None
    for category, patterns in ACS_DATA_PATTERNS.items():
        if any(re.search(pattern, table_name) for pattern in patterns):
            return category
    return None


def _acs_feature_aliases(con, table_name: str, variable_lookup: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    raw_columns = [
        column
        for column in _column_names(con, "raw", table_name)
        if column.lower() not in ACS_KEY_COLUMNS
    ]
    aliases = []
    used_aliases: set[str] = set()
    used_alias_keys: set[str] = set()
    for column in raw_columns:
        alias = variable_lookup.get(column, {}).get("feature_name", column)
        alias = _dedupe_feature_alias(alias, column, used_aliases)
        while alias.lower() in used_alias_keys:
            digest = hashlib.sha1(f"{table_name}:{column}".encode("utf-8")).hexdigest()[:8]
            suffix = f"_v{digest}"
            alias = f"{alias[: 140 - len(suffix)]}{suffix}"
        used_aliases.add(alias)
        used_alias_keys.add(alias.lower())
        aliases.append((column, alias))
    return aliases


def _acs_select_sql(con, table_name: str, variable_lookup: dict[str, dict[str, str]]) -> str:
    select_columns = []
    for column, alias in _acs_feature_aliases(con, table_name, variable_lookup):
        select_columns.append(f"acs_source.{_quote_ident(column)} AS {_quote_ident(alias)}")
    columns = ",\n        ".join(select_columns)
    return f"""
    SELECT
        lpad(state, 2, '0') || lpad(county, 3, '0') AS fips,
        lpad(state, 2, '0') AS state_fips,
        try_cast(year AS INTEGER) AS year,
        {_quote_literal(table_name)} AS source_table,
        {columns}
    FROM {_table('raw', table_name)} AS acs_source
    WHERE year IS NOT NULL
      AND state IS NOT NULL
      AND county IS NOT NULL
    """


def _null_acs_special_values_in_mart(con, mart_name: str, columns: list[str]) -> None:
    special_values = ", ".join(_quote_literal(str(value)) for value in sorted(ACS_SPECIAL_NUMERIC_VALUES))
    for column in columns:
        con.execute(
            f"""
            UPDATE {_table('mart', mart_name)}
            SET {_quote_ident(column)} = NULL
            WHERE replace(CAST({_quote_ident(column)} AS VARCHAR), ',', '') IN ({special_values})
            """
        )


def _create_acs_mart(con, *, mart_name: str, table_names: list[str], variable_lookup: dict[str, dict[str, str]]) -> None:
    con.execute(f"DROP TABLE IF EXISTS {_table('mart', mart_name)}")
    columns: dict[str, str] = {
        "fips": "VARCHAR",
        "state_fips": "VARCHAR",
        "year": "INTEGER",
        "source_table": "VARCHAR",
    }
    column_keys = {column.lower() for column in columns}
    for table_name in table_names:
        for _, alias in _acs_feature_aliases(con, table_name, variable_lookup):
            alias_key = alias.lower()
            if alias_key not in column_keys:
                columns[alias] = "VARCHAR"
                column_keys.add(alias_key)

    column_defs = ",\n        ".join(f"{_quote_ident(column)} {column_type}" for column, column_type in columns.items())
    con.execute(f"CREATE TABLE {_table('mart', mart_name)} (\n        {column_defs}\n    )")
    for table_name in table_names:
        con.execute(f"INSERT INTO {_table('mart', mart_name)} BY NAME {_acs_select_sql(con, table_name, variable_lookup)}")
    _null_acs_special_values_in_mart(con, mart_name, [column for column in columns if column not in {"fips", "state_fips", "year", "source_table"}])


def _create_acs_marts(con, acs_table_names: list[str], variable_lookup: dict[str, dict[str, str]]) -> None:
    grouped = {"economic": [], "demographic": [], "affordability": []}
    for table_name in acs_table_names:
        category = _acs_category(table_name)
        if category:
            grouped[category].append(table_name)

    _create_acs_mart(con, mart_name="acs_county_economic_annual", table_names=grouped["economic"], variable_lookup=variable_lookup)
    _create_acs_mart(con, mart_name="acs_county_demographic_annual", table_names=grouped["demographic"], variable_lookup=variable_lookup)
    _create_acs_mart(con, mart_name="acs_county_affordability_annual", table_names=grouped["affordability"], variable_lookup=variable_lookup)


def _existing_acs_raw_tables(con) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'raw'
              AND table_name LIKE 'census_acs5_%'
            ORDER BY table_name
            """
        ).fetchall()
    ]


def _create_indexes(con) -> None:
    index_specs = [
        ("idx_ref_counties_fips", "ref.counties", "fips"),
        ("idx_redfin_fips_period", "mart.redfin_county_monthly", "fips, period_begin"),
        ("idx_nri_fips", "mart.nri_county_risk", "fips"),
        ("idx_fema_fips_year", "mart.fema_disaster_declarations", "fips, declared_year"),
        ("idx_fema_financial_assistance_disaster_number", "mart.fema_disaster_financial_assistance", "disaster_number"),
        ("idx_noaa_fips_year", "mart.noaa_storm_events", "fips, event_year"),
        ("idx_ncei_weather_fips_month", "mart.ncei_county_weather_monthly", "fips, weather_month"),
        ("idx_acs_econ_fips_year", "mart.acs_county_economic_annual", "fips, year"),
        ("idx_acs_demo_fips_year", "mart.acs_county_demographic_annual", "fips, year"),
        ("idx_acs_afford_fips_year", "mart.acs_county_affordability_annual", "fips, year"),
    ]
    for index_name, table_name, columns in index_specs:
        con.execute(f"CREATE INDEX IF NOT EXISTS {_quote_ident(index_name)} ON {table_name} ({columns})")


def build_database(database_path: Path = DATABASE_PATH, *, skip_indexes: bool = False, marts_only: bool = False) -> None:
    duckdb = _load_duckdb()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database_path))
    try:
        _configure_connection(con)
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        if marts_only:
            loaded = {"acs": _existing_acs_raw_tables(con)}
        else:
            _create_meta(con)
            loaded = _load_raw(con)
        variable_lookup = _build_acs_variable_lookup()
        con.execute("CREATE SCHEMA IF NOT EXISTS meta")
        _create_acs_variable_features(con, loaded["acs"], variable_lookup)
        _create_ref_tables(con)
        _create_core_marts(con)
        con.close()
        con = duckdb.connect(str(database_path))
        _configure_connection(con)
        _create_acs_marts(con, loaded["acs"], variable_lookup)
        if not skip_indexes:
            _create_indexes(con)
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DuckDB database for Quoll Intelligence data.")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH, help="Output DuckDB database path.")
    parser.add_argument("--skip-indexes", action="store_true", help="Skip secondary index creation for faster builds.")
    parser.add_argument("--marts-only", action="store_true", help="Rebuild ref and mart tables from existing raw tables.")
    args = parser.parse_args()
    build_database(args.database, skip_indexes=args.skip_indexes, marts_only=args.marts_only)
    print(f"Built database: {args.database}")


if __name__ == "__main__":
    main()
