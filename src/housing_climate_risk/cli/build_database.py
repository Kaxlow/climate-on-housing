from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from housing_climate_risk.cli.analysis_marts import create_analysis_marts
from housing_climate_risk.cli.feature_marts import create_feature_marts

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
ACS_DIR = DATA_DIR / "acs"
CLIMATE_DAMAGE_DIR = DATA_DIR / "climate_damage"
CLIMATE_DIR = DATA_DIR / "climate"
FEMA_DIR = DATA_DIR / "fema"
FIPSGEO_DIR = DATA_DIR / "fipsgeo"
HOUSING_DIR = DATA_DIR / "housing"
DATABASE_PATH = DATA_DIR / "quoll.duckdb"
COUNTY_PROCESSED_DATA_PATH = (
    DATA_DIR / "20260401_county_processed_data" / "county_processed_data.feather"
)
STATSAMERICA_DIR = DATA_DIR / "statsamerica"


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
    "statsamerica_population_components": STATSAMERICA_DIR / "Components of Population Change - U.S., States, and Counties.csv",
    "statsamerica_bea_per_capita_income": STATSAMERICA_DIR / "BEA - US, States, Counties - Per Capita Income.csv",
    "statsamerica_bea_personal_income": STATSAMERICA_DIR / "BEA - US, States, Counties - Personal Income.csv",
    "statsamerica_cew_total_ownership": STATSAMERICA_DIR / "CEW - US, States, Counties - Total Ownership.csv",
}

RAW_SOURCE_URLS = {
    "redfin_housing_market_by_county": "https://www.redfin.com/news/data-center/",
    "nri_table_counties": "https://hazards.fema.gov/nri/data-resources",
    "fema_disaster_declarations": "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries",
    "fema_web_disaster_summaries": "https://www.fema.gov/api/open/v1/FemaWebDisasterSummaries",
    "noaa_storm_events_county_damage": "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
    "noaa_storm_events_zone_county_mapping": "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
    "ncei_climate_at_a_glance_county_monthly": "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/",
}


ACS_DATA_PATTERNS = {
    "economic": [r"_dp03_"],
    "demographic": [r"_dp02_", r"_dp05_", r"_migration_", r"_population_"],
    "affordability": [r"_affordability_", r"_dp04_", r"_s250", r"_b251", r"_housing_financial_"],
}


ACS_KEY_COLUMNS = {"year", "state", "county"}
ACS_EXCLUDED_FEATURE_COLUMNS = {"median_owner_costs_pct_income"}
ACS_VARIABLE_RE = re.compile(r"^[A-Z]+\d+(?:_C\d+)?_\d+(?:E|M|PE|PM)$")
ACS_NORMALIZED_VARIABLE_RE = re.compile(r"^[a-z]+\d+(?:_c\d+)?_\d+_(?:est|moe|pct|pct_moe)$")
ACS_SPECIAL_NUMERIC_VALUES = {-222222222, -333333333, -555555555, -666666666, -888888888, -999999999}
NEGATIVE_ALLOWED_COLUMN_TOKENS = (
    "anomaly",
    "change",
    "delta",
    "diff",
    "growth",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "mom",
    "pct_change",
    "percent_change",
    "temp",
    "temperature",
    "yoy",
)
NONNEGATIVE_COLUMN_TOKENS = (
    "amount",
    "application",
    "area",
    "assistance",
    "claim",
    "count",
    "damage",
    "death",
    "dollar",
    "eal",
    "estimate",
    "expense",
    "grant",
    "home",
    "household",
    "housing",
    "income",
    "injur",
    "inventory",
    "loss",
    "market_value",
    "moe",
    "number",
    "obligated",
    "pct",
    "percent",
    "population",
    "price",
    "rank",
    "rate",
    "ratio",
    "risk_score",
    "sale",
    "score",
    "sqft",
    "total",
    "unit",
    "value",
)


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


def _read_csv_column_names(con, source_path: Path) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE SELECT *
            FROM read_csv_auto(
                {_quote_literal(source_path)},
                header = true,
                all_varchar = true,
                ignore_errors = true,
                union_by_name = true
            )
            """
        ).fetchall()
    ]


def _normalized_column_key(column: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(column).lower()).strip("_")


def _is_acs_measure_column(column: str) -> bool:
    return (
        ACS_VARIABLE_RE.match(str(column)) is not None
        or ACS_NORMALIZED_VARIABLE_RE.match(str(column)) is not None
    )


def _acs_raw_column_name(column: str) -> str:
    """Normalize a Census API variable ID without discarding its provenance."""
    match = re.fullmatch(r"([A-Z]+\d+(?:_C\d+)?)_(\d+)(E|M|PE|PM)", str(column))
    if not match:
        return str(column)
    group, sequence, statistic = match.groups()
    statistic_name = {"E": "est", "M": "moe", "PE": "pct", "PM": "pct_moe"}[statistic]
    return f"{group.lower()}_{sequence}_{statistic_name}"


def _acs_source_variable(column: str, variable_lookup: dict[str, dict[str, str]]) -> str | None:
    """Resolve either an original or normalized raw column to its Census variable ID."""
    if column in variable_lookup:
        return column
    match = re.fullmatch(r"([a-z]+\d+(?:_c\d+)?)_(\d+)_(est|moe|pct|pct_moe)", str(column))
    if not match:
        return None
    group, sequence, statistic_name = match.groups()
    statistic = {"est": "E", "moe": "M", "pct": "PE", "pct_moe": "PM"}[statistic_name]
    variable = f"{group.upper()}_{sequence}{statistic}"
    return variable if variable in variable_lookup else None


def _allows_negative_values(table_name: str, column: str) -> bool:
    column_key = _normalized_column_key(column)
    if table_name == "ncei_climate_at_a_glance_county_monthly" and column_key in {"value", "anomaly"}:
        return True
    column_parts = set(column_key.split("_"))
    short_token_match = any(token in column_parts for token in NEGATIVE_ALLOWED_COLUMN_TOKENS if len(token) <= 3)
    long_token_match = any(
        token in column_key
        for token in NEGATIVE_ALLOWED_COLUMN_TOKENS
        if len(token) > 3
    )
    return short_token_match or long_token_match


def _should_null_negative_values(table_name: str, column: str) -> bool:
    column_key = _normalized_column_key(column)
    if not column_key:
        return False
    if _allows_negative_values(table_name, column):
        return False
    if _is_acs_measure_column(column):
        return True
    if column_key in {
        "fips",
        "state",
        "county",
        "year",
        "month",
        "date",
        "period_begin",
        "period_end",
        "begin_yearmonth",
        "end_yearmonth",
    }:
        return False
    return any(token in column_key for token in NONNEGATIVE_COLUMN_TOKENS)


def _clean_negative_values_select_sql(
    *,
    table_name: str,
    columns: list[str],
    source_alias: str = "source",
    output_aliases: dict[str, str] | None = None,
) -> str:
    select_columns = []
    for column in columns:
        qualified_column = f"{_quote_ident(source_alias)}.{_quote_ident(column)}"
        output_column = (output_aliases or {}).get(column, column)
        if _should_null_negative_values(table_name, column):
            numeric_expr = f"try_cast(replace(trim({qualified_column}), ',', '') AS DOUBLE)"
            select_columns.append(
                f"CASE WHEN {numeric_expr} < 0 THEN NULL ELSE {qualified_column} END AS {_quote_ident(output_column)}"
            )
        else:
            select_columns.append(f"{qualified_column} AS {_quote_ident(output_column)}")
    return ",\n            ".join(select_columns)


def _null_negative_values_in_frame(df: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if not _should_null_negative_values(table_name, str(column)):
            continue
        numeric_values = pd.to_numeric(out[column], errors="coerce")
        out.loc[numeric_values < 0, column] = pd.NA
    return out


def _file_sha256(source_path: Path) -> str:
    digest = hashlib.sha256()
    with source_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_source_url(source_path: Path) -> str | None:
    try:
        relative_path = source_path.resolve().relative_to(CLIMATE_DAMAGE_DIR.resolve())
    except ValueError:
        return None
    manifest_path = CLIMATE_DAMAGE_DIR / "climate_damage_source_manifest.csv"
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        matches = [
            row.get("url", "").strip()
            for row in rows
            if Path(row.get("path", "")).as_posix() == relative_path.as_posix()
            and row.get("url", "").strip()
        ]
    return ";".join(dict.fromkeys(matches)) or None


def _register_file_metadata(con, *, table_schema: str, table_name: str, source_path: Path) -> None:
    columns = _column_names(con, table_schema, table_name)
    row_count = con.execute(f"SELECT count(*) FROM {_table(table_schema, table_name)}").fetchone()[0]
    resolved_source_path = source_path.resolve()
    source_folder = (
        str(resolved_source_path.parent.relative_to(DATA_DIR))
        if resolved_source_path.is_relative_to(DATA_DIR)
        else str(resolved_source_path.parent)
    )
    source_url = _manifest_source_url(source_path) or RAW_SOURCE_URLS.get(table_name)
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
            detected_columns,
            upstream_source_url,
            content_sha256
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            source_url,
            _file_sha256(source_path),
        ],
    )


def _load_csv_raw(con, *, table_name: str, source_path: Path, normalize_acs_columns: bool = False) -> None:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    columns = _read_csv_column_names(con, source_path)
    output_aliases = {column: _acs_raw_column_name(column) for column in columns} if normalize_acs_columns else None
    cleaned_select = _clean_negative_values_select_sql(
        table_name=table_name,
        columns=columns,
        output_aliases=output_aliases,
    )
    con.execute(f"DROP TABLE IF EXISTS {_table('raw', table_name)}")
    con.execute(
        f"""
        CREATE TABLE {_table('raw', table_name)} AS
        SELECT
            {cleaned_select}
        FROM read_csv_auto(
            {_quote_literal(source_path)},
            header = true,
            all_varchar = true,
            ignore_errors = true,
            union_by_name = true
        ) AS source
        """
    )
    _register_file_metadata(con, table_schema="raw", table_name=table_name, source_path=source_path)


def _load_feather_raw(con, *, table_name: str, source_path: Path) -> None:
    if not source_path.exists():
        return
    df = pd.read_feather(source_path)
    df = _null_negative_values_in_frame(df, table_name=table_name)
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
        _load_csv_raw(con, table_name=table_name, source_path=source_path, normalize_acs_columns=True)
        loaded[table_name] = [str(source_path)]

    for source_path in sorted(ACS_DIR.glob("*.csv")):
        table_name = _sanitize_table_name(source_path)
        _load_csv_raw(con, table_name=table_name, source_path=source_path)
        loaded["acs"].append(table_name)

    _load_feather_raw(
        con,
        table_name="county_processed_data",
        source_path=COUNTY_PROCESSED_DATA_PATH,
    )
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
            detected_columns JSON,
            upstream_source_url VARCHAR,
            content_sha256 VARCHAR
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
        raw_part = re.sub(
            r"^(?:percent margin of error|margin of error|percent estimate|estimate|percent)\s+",
            "",
            raw_part,
            flags=re.IGNORECASE,
        )
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
    if str(group).upper() == "DP02" and parts and parts[0] == "educational_attainment":
        parts = [part for part in parts if part != "population_25_plus"]
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
    variables_df["year_sort"] = pd.to_numeric(variables_df["year"], errors="coerce").fillna(0).astype(int)
    variables_df = variables_df.sort_values(
        ["variable", "year_sort", "dictionary_file"],
        ascending=[True, False, False],
    )
    variables_df = variables_df.drop_duplicates(subset=["variable", "year_sort"], keep="first")

    lookup: dict[str, dict[str, str]] = {}
    for row in variables_df.to_dict("records"):
        variable = str(row["variable"])
        feature = _acs_feature_name(
            variable,
            str(row.get("label", "")),
            str(row.get("group", "")),
            str(row.get("concept", "")),
        )
        year = int(row["year_sort"])
        year_entry = {
            "year": year,
            "feature_name": feature,
            "label": str(row.get("label", "")),
            "concept": str(row.get("concept", "")),
            "predicate_type": str(row.get("predicate_type", "")),
            "group": str(row.get("group", "")),
            "dictionary_file": str(row.get("dictionary_file", "")),
        }
        variable_entry = lookup.setdefault(variable, {"by_year": {}})
        variable_entry["by_year"][year] = year_entry
        if year >= int(variable_entry.get("year", -1)):
            variable_entry.update(year_entry)
    return lookup


def _create_acs_variable_features(con, acs_table_names: list[str], variable_lookup: dict[str, dict[str, str]]) -> None:
    rows = []
    for table_name in acs_table_names:
        raw_columns = _column_names(con, "raw", table_name)
        if "year" not in {column.lower() for column in raw_columns}:
            continue
        table_years = {
            int(row[0])
            for row in con.execute(
                f"SELECT DISTINCT try_cast(year AS INTEGER) FROM {_table('raw', table_name)} WHERE year IS NOT NULL"
            ).fetchall()
            if row[0] is not None
        }
        for column in raw_columns:
            variable = _acs_source_variable(column, variable_lookup)
            if variable is None:
                continue
            for year, row in variable_lookup[variable]["by_year"].items():
                if year not in table_years:
                    continue
                rows.append(
                    {
                        "source_table": table_name,
                        "year": year,
                        "variable": variable,
                        "raw_column": column,
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
            "year",
            "variable",
            "raw_column",
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


def _raw_table_exists(con, table_name: str) -> bool:
    return bool(
        con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'raw' AND table_name = ?",
            [table_name],
        ).fetchone()
    )


def _create_statsamerica_bea_cew_marts(con) -> None:
    """
    Build mart tables from the three StatsAmerica BEA/CEW raw tables.

    Tables created:
      mart.statsamerica_bea_per_capita_income_annual
      mart.statsamerica_bea_personal_income_annual
      mart.statsamerica_cew_county_annual
    All are filtered to county-level rows (Countyfips != '000') and the
    last 10 completed calendar years relative to the most recent year in the file.
    """
    # ------------------------------------------------------------------
    # BEA Per Capita Income
    # ------------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS mart.statsamerica_bea_per_capita_income_annual")
    if _raw_table_exists(con, "statsamerica_bea_per_capita_income"):
        con.execute(
            """
            CREATE TABLE mart.statsamerica_bea_per_capita_income_annual AS
            WITH county_rows AS (
                SELECT
                    lpad(trim(IBRC_GEO_ID), 5, '0') AS fips,
                    substr(lpad(trim(IBRC_GEO_ID), 5, '0'), 1, 2) AS state_fips,
                    TRY_CAST(Year AS INTEGER) AS year,
                    Description AS county_name,
                    TRY_CAST("BEA Per Capita Personal Income" AS INTEGER)
                        AS per_capita_personal_income_dollars
                FROM raw.statsamerica_bea_per_capita_income
                WHERE IBRC_GEO_ID IS NOT NULL
                  AND Countyfips IS NOT NULL
                  AND Countyfips != '000'
                  AND Year IS NOT NULL
            ),
            max_year AS (SELECT max(year) AS max_yr FROM county_rows)
            SELECT c.*
            FROM county_rows c, max_year m
            WHERE c.year >= m.max_yr - 9
            """
        )
    else:
        con.execute(
            """
            CREATE TABLE mart.statsamerica_bea_per_capita_income_annual (
                fips VARCHAR,
                state_fips VARCHAR,
                year INTEGER,
                county_name VARCHAR,
                per_capita_personal_income_dollars INTEGER
            )
            """
        )

    # ------------------------------------------------------------------
    # BEA Personal Income — pivoted wide, one column per selected linecode.
    # Linecodes: 0010 personal_income, 0020 population,
    #   0035 earnings_by_place_of_work, 0045 net_earnings_by_place_of_residence,
    #   0046 dividends_interest_rent, 0047 transfer_receipts,
    #   0050 wage_salary_disbursements, 0070 proprietors_income
    # ------------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS mart.statsamerica_bea_personal_income_annual")
    if _raw_table_exists(con, "statsamerica_bea_personal_income"):
        con.execute(
            """
            CREATE TABLE mart.statsamerica_bea_personal_income_annual AS
            WITH county_rows AS (
                SELECT
                    lpad(trim(IBRC_GEO_ID), 5, '0') AS fips,
                    substr(lpad(trim(IBRC_GEO_ID), 5, '0'), 1, 2) AS state_fips,
                    TRY_CAST(Year AS INTEGER) AS year,
                    Description AS county_name,
                    Linecode AS linecode,
                    -- Suppressed rows carry non-numeric Data; treat as NULL
                    CASE WHEN Disclosure = '0' THEN TRY_CAST(Data AS DOUBLE) ELSE NULL END AS data_value
                FROM raw.statsamerica_bea_personal_income
                WHERE IBRC_GEO_ID IS NOT NULL
                  AND Countyfips IS NOT NULL
                  AND Countyfips != '000'
                  AND Year IS NOT NULL
                  AND Linecode IN ('0010', '0020', '0035', '0045', '0046', '0047', '0050', '0070')
            ),
            max_year AS (SELECT max(year) AS max_yr FROM county_rows),
            recent AS (
                SELECT c.*
                FROM county_rows c, max_year m
                WHERE c.year >= m.max_yr - 9
            )
            SELECT
                fips,
                state_fips,
                year,
                county_name,
                max(data_value) FILTER (WHERE linecode = '0010') AS personal_income_thousands,
                max(data_value) FILTER (WHERE linecode = '0020') AS population,
                max(data_value) FILTER (WHERE linecode = '0035') AS earnings_by_place_of_work_thousands,
                max(data_value) FILTER (WHERE linecode = '0045') AS net_earnings_by_place_of_residence_thousands,
                max(data_value) FILTER (WHERE linecode = '0046') AS dividends_interest_rent_thousands,
                max(data_value) FILTER (WHERE linecode = '0047') AS transfer_receipts_thousands,
                max(data_value) FILTER (WHERE linecode = '0050') AS wage_salary_disbursements_thousands,
                max(data_value) FILTER (WHERE linecode = '0070') AS proprietors_income_thousands
            FROM recent
            GROUP BY fips, state_fips, year, county_name
            """
        )
    else:
        con.execute(
            """
            CREATE TABLE mart.statsamerica_bea_personal_income_annual (
                fips VARCHAR,
                state_fips VARCHAR,
                year INTEGER,
                county_name VARCHAR,
                personal_income_thousands DOUBLE,
                population DOUBLE,
                earnings_by_place_of_work_thousands DOUBLE,
                net_earnings_by_place_of_residence_thousands DOUBLE,
                dividends_interest_rent_thousands DOUBLE,
                transfer_receipts_thousands DOUBLE,
                wage_salary_disbursements_thousands DOUBLE,
                proprietors_income_thousands DOUBLE
            )
            """
        )

    # ------------------------------------------------------------------
    # CEW Total Ownership — 2-digit sector breakdown
    #   Ownership Code '0' (All), top-level sector NAICS codes only
    #   (2-digit codes + multi-range codes like 31-33, 44-45, 48-49)
    # ------------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS mart.statsamerica_cew_county_sector_annual")
    if _raw_table_exists(con, "statsamerica_cew_total_ownership"):
        con.execute(
            """
            CREATE TABLE mart.statsamerica_cew_county_sector_annual AS
            WITH county_rows AS (
                SELECT
                    lpad(trim(IBRC_GEO_ID), 5, '0') AS fips,
                    substr(lpad(trim(IBRC_GEO_ID), 5, '0'), 1, 2) AS state_fips,
                    TRY_CAST(Year AS INTEGER) AS year,
                    Description AS county_name,
                    trim("NAICS Code") AS naics_code,
                    trim("NAICS Description") AS naics_description,
                    TRY_CAST(Units AS DOUBLE) AS establishments,
                    TRY_CAST(Employment AS DOUBLE) AS employment,
                    TRY_CAST(Wages AS DOUBLE) AS total_wages_dollars,
                    TRY_CAST("Average Wage" AS DOUBLE) AS avg_annual_wage_dollars,
                    TRY_CAST("Average Weekly Wage" AS DOUBLE) AS avg_weekly_wage_dollars
                FROM raw.statsamerica_cew_total_ownership
                WHERE IBRC_GEO_ID IS NOT NULL
                  AND Countyfips IS NOT NULL
                  AND Countyfips != '000'
                  AND Year IS NOT NULL
                  AND "Ownership Code" = '0'
                  -- Top-level sector codes: 2-digit integers plus BLS multi-range codes
                  AND trim("NAICS Code") IN (
                      '11','21','22','23',
                      '31-33',
                      '42',
                      '44-45',
                      '48-49',
                      '51','52','53','54','55','56',
                      '61','62','71','72','81','92','99'
                  )
            ),
            max_year AS (SELECT max(year) AS max_yr FROM county_rows)
            SELECT c.*
            FROM county_rows c, max_year m
            WHERE c.year >= m.max_yr - 9
            """
        )
    else:
        con.execute(
            """
            CREATE TABLE mart.statsamerica_cew_county_sector_annual (
                fips VARCHAR,
                state_fips VARCHAR,
                year INTEGER,
                county_name VARCHAR,
                naics_code VARCHAR,
                naics_description VARCHAR,
                establishments DOUBLE,
                employment DOUBLE,
                total_wages_dollars DOUBLE,
                avg_annual_wage_dollars DOUBLE,
                avg_weekly_wage_dollars DOUBLE
            )
            """
        )

    # ------------------------------------------------------------------
    # CEW Total Ownership — aggregate totals only:
    #   Ownership Code '0' (All), NAICS Code '00' (Total all industries)
    # ------------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS mart.statsamerica_cew_county_annual")
    if _raw_table_exists(con, "statsamerica_cew_total_ownership"):
        con.execute(
            """
            CREATE TABLE mart.statsamerica_cew_county_annual AS
            WITH county_rows AS (
                SELECT
                    lpad(trim(IBRC_GEO_ID), 5, '0') AS fips,
                    substr(lpad(trim(IBRC_GEO_ID), 5, '0'), 1, 2) AS state_fips,
                    TRY_CAST(Year AS INTEGER) AS year,
                    Description AS county_name,
                    TRY_CAST(Units AS DOUBLE) AS establishments,
                    TRY_CAST(Employment AS DOUBLE) AS employment,
                    TRY_CAST(Wages AS DOUBLE) AS total_wages_dollars,
                    TRY_CAST("Average Wage" AS DOUBLE) AS avg_annual_wage_dollars,
                    TRY_CAST("Average Weekly Wage" AS DOUBLE) AS avg_weekly_wage_dollars
                FROM raw.statsamerica_cew_total_ownership
                WHERE IBRC_GEO_ID IS NOT NULL
                  AND Countyfips IS NOT NULL
                  AND Countyfips != '000'
                  AND Year IS NOT NULL
                  AND "Ownership Code" = '0'
                  AND "NAICS Code" = '00'
            ),
            max_year AS (SELECT max(year) AS max_yr FROM county_rows)
            SELECT c.*
            FROM county_rows c, max_year m
            WHERE c.year >= m.max_yr - 9
            """
        )
    else:
        con.execute(
            """
            CREATE TABLE mart.statsamerica_cew_county_annual (
                fips VARCHAR,
                state_fips VARCHAR,
                year INTEGER,
                county_name VARCHAR,
                establishments DOUBLE,
                employment DOUBLE,
                total_wages_dollars DOUBLE,
                avg_annual_wage_dollars DOUBLE,
                avg_weekly_wage_dollars DOUBLE
            )
            """
        )


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

    # Load insurance premiums from county_processed_data feather file
    # The JSON in DuckDB has numpy arrays stored as strings, so we parse in Python
    con.execute("DROP TABLE IF EXISTS mart.insurance_premiums_annual")
    county_processed_path = COUNTY_PROCESSED_DATA_PATH
    if county_processed_path.exists():
        import pandas as pd
        county_df = pd.read_feather(county_processed_path)

        premium_rows = []
        for _, row in county_df.iterrows():
            if pd.isna(row.get('insurance_premiums_14_to_24')):
                continue

            fips = str(row['fips']).zfill(5)
            premiums = row['insurance_premiums_14_to_24']

            if not isinstance(premiums, dict) or 'historical' not in premiums:
                continue

            hist = premiums['historical']
            years = hist.get('years', [])
            means = hist.get('mean', [])
            medians = hist.get('median', [])

            avg_data = premiums.get('averages', {})
            latest_data = premiums.get('latest', {})
            growth_data = premiums.get('growth_rates', {})

            for i, year in enumerate(years):
                if i < len(means) and i < len(medians):
                    premium_rows.append({
                        'fips': fips,
                        'year': int(year),
                        'mean_premium': float(means[i]),
                        'median_premium': float(medians[i]),
                        'avg_mean_premium': avg_data.get('mean'),
                        'avg_median_premium': avg_data.get('median'),
                        'latest_year': latest_data.get('year'),
                        'latest_mean_premium': latest_data.get('mean'),
                        'latest_median_premium': latest_data.get('median'),
                        'mean_cagr': growth_data.get('mean_cagr'),
                        'median_cagr': growth_data.get('median_cagr'),
                        'historical_start_year': avg_data.get('start_year'),
                        'historical_end_year': avg_data.get('end_year'),
                    })

        if premium_rows:
            premium_df = pd.DataFrame(premium_rows)
            con.register('_insurance_premiums_df', premium_df)
            con.execute("CREATE TABLE mart.insurance_premiums_annual AS SELECT * FROM _insurance_premiums_df")
            con.unregister('_insurance_premiums_df')
        else:
            con.execute("CREATE TABLE mart.insurance_premiums_annual (fips VARCHAR, year INTEGER, mean_premium DOUBLE, median_premium DOUBLE)")
    else:
        con.execute("CREATE TABLE mart.insurance_premiums_annual (fips VARCHAR, year INTEGER, mean_premium DOUBLE, median_premium DOUBLE)")

    # Load insurance non-renewal rates from county_processed_data feather file
    con.execute("DROP TABLE IF EXISTS mart.insurance_non_renewal_annual")
    if county_processed_path.exists():
        non_renewal_rows = []
        for _, row in county_df.iterrows():
            if pd.isna(row.get('insurance_non_renewal_rates')):
                continue

            fips = str(row['fips']).zfill(5)
            non_renewal = row['insurance_non_renewal_rates']

            if not isinstance(non_renewal, dict) or 'historical' not in non_renewal:
                continue

            hist = non_renewal['historical']
            years = hist.get('years', [])
            rates = hist.get('non_renewal_rate', [])
            total_policies = hist.get('num_policies_total', [])
            renewed = hist.get('num_policies_renewed', [])
            non_renewed = hist.get('num_policies_non_renewed', [])

            avg_data = non_renewal.get('averages', {})
            latest_data = non_renewal.get('latest', {})
            growth_data = non_renewal.get('growth_rates', {})

            for i, year in enumerate(years):
                if i < len(rates):
                    non_renewal_rows.append({
                        'fips': fips,
                        'year': int(year),
                        'non_renewal_rate': float(rates[i]) if i < len(rates) else None,
                        'total_policies': float(total_policies[i]) if i < len(total_policies) else None,
                        'renewed_policies': float(renewed[i]) if i < len(renewed) else None,
                        'non_renewed_policies': float(non_renewed[i]) if i < len(non_renewed) else None,
                        'avg_non_renewal_rate': avg_data.get('non_renewal_rate'),
                        'avg_total_policies': avg_data.get('num_policies_total'),
                        'avg_renewed_policies': avg_data.get('num_policies_renewed'),
                        'avg_non_renewed_policies': avg_data.get('num_policies_non_renewed'),
                        'latest_year': latest_data.get('year'),
                        'latest_non_renewal_rate': latest_data.get('non_renewal_rate'),
                        'latest_total_policies': latest_data.get('num_policies_total'),
                        'non_renewal_rate_cagr': growth_data.get('non_renewal_rate_cagr'),
                        'total_policies_cagr': growth_data.get('num_policies_total_cagr'),
                        'years_of_data': non_renewal.get('years_of_data'),
                        'historical_start_year': avg_data.get('start_year'),
                        'historical_end_year': avg_data.get('end_year'),
                    })

        if non_renewal_rows:
            non_renewal_df = pd.DataFrame(non_renewal_rows)
            con.register('_insurance_non_renewal_df', non_renewal_df)
            con.execute("CREATE TABLE mart.insurance_non_renewal_annual AS SELECT * FROM _insurance_non_renewal_df")
            con.unregister('_insurance_non_renewal_df')
        else:
            con.execute("CREATE TABLE mart.insurance_non_renewal_annual (fips VARCHAR, year INTEGER, non_renewal_rate DOUBLE)")
    else:
        con.execute("CREATE TABLE mart.insurance_non_renewal_annual (fips VARCHAR, year INTEGER, non_renewal_rate DOUBLE)")

    _create_statsamerica_bea_cew_marts(con)

    # Load StatsAmerica Components of Population Change (true net migration)
    con.execute("DROP TABLE IF EXISTS mart.statsamerica_population_components_annual")
    statsamerica_path = DATA_DIR / "statsamerica" / "Components of Population Change - U.S., States, and Counties.csv"
    if statsamerica_path.exists():
        con.execute(
            f"""
            CREATE TABLE mart.statsamerica_population_components_annual AS
            SELECT
                lpad(Statefips, 2, '0') || lpad(Countyfips, 3, '0') AS fips,
                lpad(Statefips, 2, '0') AS state_fips,
                TRY_CAST(Year AS INTEGER) AS year,
                Description AS county_name,
                TRY_CAST(Births AS INTEGER) AS births,
                TRY_CAST(Deaths AS INTEGER) AS deaths,
                TRY_CAST("Net International Migration" AS INTEGER) AS net_international_migration,
                TRY_CAST("Net Domestic Migration" AS INTEGER) AS net_domestic_migration,
                TRY_CAST(Residual AS INTEGER) AS residual,
                -- Computed columns
                TRY_CAST(Births AS INTEGER) - TRY_CAST(Deaths AS INTEGER) AS natural_increase,
                TRY_CAST("Net International Migration" AS INTEGER) + TRY_CAST("Net Domestic Migration" AS INTEGER) AS total_net_migration
            FROM read_csv_auto(
                {_quote_literal(statsamerica_path)},
                header = true,
                all_varchar = true,
                ignore_errors = true
            )
            WHERE Statefips IS NOT NULL
              AND Statefips != '0'
              AND Countyfips IS NOT NULL
              AND Countyfips != '000'
              AND Year IS NOT NULL
            """
        )
    else:
        con.execute(
            """
            CREATE TABLE mart.statsamerica_population_components_annual (
                fips VARCHAR,
                state_fips VARCHAR,
                year INTEGER,
                county_name VARCHAR,
                births INTEGER,
                deaths INTEGER,
                net_international_migration INTEGER,
                net_domestic_migration INTEGER,
                residual INTEGER,
                natural_increase INTEGER,
                total_net_migration INTEGER
            )
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


def _duplicate_acs_percent_columns(
    con,
    table_name: str,
    variable_lookup: dict[str, dict[str, str]],
) -> set[str]:
    """Return raw Census PE columns that add no information beyond paired E columns."""
    raw_columns = _column_names(con, "raw", table_name)
    columns_by_variable = {
        variable: column
        for column in raw_columns
        if (variable := _acs_source_variable(column, variable_lookup)) is not None
    }
    candidates: list[tuple[str, str]] = []
    for variable, percent_column in columns_by_variable.items():
        if not re.fullmatch(r"[A-Z]+\d+(?:_C\d+)?_\d+PE", variable):
            continue
        estimate_variable = f"{variable[:-2]}E"
        estimate_column = columns_by_variable.get(estimate_variable)
        if estimate_column is not None:
            candidates.append((percent_column, estimate_column))

    duplicate_columns: set[str] = set()
    special_values = ", ".join(str(value) for value in sorted(ACS_SPECIAL_NUMERIC_VALUES))
    for chunk_start in range(0, len(candidates), 100):
        chunk = candidates[chunk_start : chunk_start + 100]
        expressions = []
        for index, (percent_column, estimate_column) in enumerate(chunk):
            percent_value = (
                f"try_cast(replace(cast({_quote_ident(percent_column)} AS VARCHAR), ',', '') AS DOUBLE)"
            )
            estimate_value = (
                f"try_cast(replace(cast({_quote_ident(estimate_column)} AS VARCHAR), ',', '') AS DOUBLE)"
            )
            valid_percent = f"{percent_value} IS NOT NULL AND {percent_value} NOT IN ({special_values})"
            expressions.extend(
                [
                    f"count(*) FILTER (WHERE {valid_percent}) AS valid_{index}",
                    (
                        "count(*) FILTER (WHERE "
                        f"{valid_percent} AND {percent_value} IS DISTINCT FROM {estimate_value}"
                        f") AS different_{index}"
                    ),
                ]
            )
        if not expressions:
            continue
        result = con.execute(
            f"SELECT {', '.join(expressions)} FROM {_table('raw', table_name)}"
        ).fetchone()
        for index, (percent_column, _) in enumerate(chunk):
            valid_count = result[index * 2]
            different_count = result[index * 2 + 1]
            if valid_count > 0 and different_count == 0:
                duplicate_columns.add(percent_column)
    return duplicate_columns


def _acs_feature_aliases(
    con,
    table_name: str,
    variable_lookup: dict[str, dict[str, str]],
    *,
    excluded_columns: set[str] | None = None,
) -> list[tuple[str, str]]:
    excluded_columns = excluded_columns or set()
    raw_columns = [
        column
        for column in _column_names(con, "raw", table_name)
        if column.lower() not in ACS_KEY_COLUMNS
        and column.lower() not in ACS_EXCLUDED_FEATURE_COLUMNS
        and column not in excluded_columns
    ]
    aliases = []
    used_aliases: set[str] = set()
    used_alias_keys: set[str] = set()
    for column in raw_columns:
        variable = _acs_source_variable(column, variable_lookup)
        alias = variable_lookup.get(variable or "", {}).get("feature_name", column)
        alias = _dedupe_feature_alias(alias, variable or column, used_aliases)
        while alias.lower() in used_alias_keys:
            digest = hashlib.sha1(f"{table_name}:{column}".encode("utf-8")).hexdigest()[:8]
            suffix = f"_v{digest}"
            alias = f"{alias[: 140 - len(suffix)]}{suffix}"
        used_aliases.add(alias)
        used_alias_keys.add(alias.lower())
        aliases.append((column, alias))
    return aliases


def _acs_feature_selects(
    con,
    table_name: str,
    variable_lookup: dict[str, dict[str, str]],
    *,
    excluded_columns: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Build year-aware SQL expressions for stable ACS conceptual features."""
    excluded_columns = excluded_columns or set()
    table_years = {
        int(row[0])
        for row in con.execute(
            f"SELECT DISTINCT try_cast(year AS INTEGER) FROM {_table('raw', table_name)} WHERE year IS NOT NULL"
        ).fetchall()
        if row[0] is not None
    }
    feature_sources: dict[str, dict[int, list[str]]] = {}
    for column in _column_names(con, "raw", table_name):
        if (
            column.lower() in ACS_KEY_COLUMNS
            or column.lower() in ACS_EXCLUDED_FEATURE_COLUMNS
            or column in excluded_columns
        ):
            continue
        variable = _acs_source_variable(column, variable_lookup)
        if variable is None:
            for year in table_years:
                feature_sources.setdefault(column, {}).setdefault(year, []).append(column)
            continue
        for year, row in variable_lookup[variable]["by_year"].items():
            if year not in table_years:
                continue
            feature_sources.setdefault(row["feature_name"], {}).setdefault(year, []).append(column)

    selects = []
    for feature_name, sources_by_year in sorted(feature_sources.items()):
        cases = []
        for year, source_columns in sorted(sources_by_year.items()):
            values = [
                f"acs_source.{_quote_ident(column)}"
                for column in dict.fromkeys(source_columns)
            ]
            value_expression = values[0] if len(values) == 1 else f"coalesce({', '.join(values)})"
            cases.append(f"WHEN try_cast(acs_source.year AS INTEGER) = {year} THEN {value_expression}")
        selects.append((feature_name, f"CASE {' '.join(cases)} END"))
    return selects


def _acs_select_sql(
    con,
    table_name: str,
    variable_lookup: dict[str, dict[str, str]],
    *,
    excluded_columns: set[str] | None = None,
) -> str:
    select_columns = []
    for alias, expression in _acs_feature_selects(
        con,
        table_name,
        variable_lookup,
        excluded_columns=excluded_columns,
    ):
        select_columns.append(f"{expression} AS {_quote_ident(alias)}")
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
    excluded_columns_by_table = {
        table_name: _duplicate_acs_percent_columns(con, table_name, variable_lookup)
        for table_name in table_names
    }
    for table_name in table_names:
        for alias, _ in _acs_feature_selects(
            con,
            table_name,
            variable_lookup,
            excluded_columns=excluded_columns_by_table[table_name],
        ):
            alias_key = alias.lower()
            if alias_key not in column_keys:
                columns[alias] = "VARCHAR"
                column_keys.add(alias_key)

    column_defs = ",\n        ".join(f"{_quote_ident(column)} {column_type}" for column, column_type in columns.items())
    con.execute(f"CREATE TABLE {_table('mart', mart_name)} (\n        {column_defs}\n    )")
    for table_name in table_names:
        con.execute(
            f"INSERT INTO {_table('mart', mart_name)} BY NAME "
            f"{_acs_select_sql(con, table_name, variable_lookup, excluded_columns=excluded_columns_by_table[table_name])}"
        )
    _null_acs_special_values_in_mart(con, mart_name, [column for column in columns if column not in {"fips", "state_fips", "year", "source_table"}])


def _add_affordability_computed_columns(con) -> None:
    """Add concise affordability features derived from detailed ACS tables."""
    # B25103_001E/M report the county median real-estate taxes paid across
    # owner-occupied units, regardless of mortgage status.
    con.execute(
        """
        ALTER TABLE mart.acs_county_affordability_annual
        ADD COLUMN IF NOT EXISTS median_property_taxes DOUBLE
        """
    )
    con.execute(
        """
        ALTER TABLE mart.acs_county_affordability_annual
        ADD COLUMN IF NOT EXISTS median_property_taxes_moe DOUBLE
        """
    )
    affordability_columns = set(_column_names(con, "mart", "acs_county_affordability_annual"))
    b25103_estimate = "b25103_median_real_estate_taxes_paid_total_est"
    b25103_moe = "b25103_median_real_estate_taxes_paid_total_moe"
    if b25103_estimate in affordability_columns:
        moe_expression = f"try_cast({_quote_ident(b25103_moe)} AS DOUBLE)" if b25103_moe in affordability_columns else "NULL"
        con.execute(
            f"""
            UPDATE mart.acs_county_affordability_annual
            SET
                median_property_taxes = try_cast({_quote_ident(b25103_estimate)} AS DOUBLE),
                median_property_taxes_moe = {moe_expression}
            WHERE {_quote_ident(b25103_estimate)} IS NOT NULL
            """
        )

    # Add housing_cost_pct_income: median monthly housing costs as percentage of median household income
    # Uses S2503_C02_024E (median monthly housing costs for owner-occupied) and S2503_C02_013E (median household income for owner-occupied)
    con.execute(
        """
        ALTER TABLE mart.acs_county_affordability_annual
        ADD COLUMN IF NOT EXISTS housing_cost_pct_income DOUBLE
        """
    )
    con.execute(
        """
        UPDATE mart.acs_county_affordability_annual
        SET housing_cost_pct_income = (
            CAST(s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est AS DOUBLE) * 12.0 * 100.0 /
            NULLIF(CAST(s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est AS DOUBLE), 0)
        )
        WHERE s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est IS NOT NULL
          AND s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est IS NOT NULL
        """
    )


def _add_demographic_computed_columns(con) -> None:
    """Add computed columns to demographic mart derived from migration data."""
    # Add total_in_migration_rate: combines domestic and international in-migration
    # Uses domestic_in_migration_rate and moved_from_abroad_rate from B07001 migration data
    con.execute(
        """
        ALTER TABLE mart.acs_county_demographic_annual
        ADD COLUMN IF NOT EXISTS total_in_migration_rate DOUBLE
        """
    )
    con.execute(
        """
        UPDATE mart.acs_county_demographic_annual
        SET total_in_migration_rate = (
            CAST(domestic_in_migration_rate AS DOUBLE) + CAST(moved_from_abroad_rate AS DOUBLE)
        )
        WHERE domestic_in_migration_rate IS NOT NULL
          AND moved_from_abroad_rate IS NOT NULL
        """
    )


def _create_acs_marts(con, acs_table_names: list[str], variable_lookup: dict[str, dict[str, str]]) -> None:
    grouped = {"economic": [], "demographic": [], "affordability": []}
    for table_name in acs_table_names:
        category = _acs_category(table_name)
        if category:
            grouped[category].append(table_name)

    _create_acs_mart(con, mart_name="acs_county_economic_annual", table_names=grouped["economic"], variable_lookup=variable_lookup)
    _create_acs_mart(con, mart_name="acs_county_demographic_annual", table_names=grouped["demographic"], variable_lookup=variable_lookup)
    _create_acs_mart(con, mart_name="acs_county_affordability_annual", table_names=grouped["affordability"], variable_lookup=variable_lookup)
    _add_affordability_computed_columns(con)
    _add_demographic_computed_columns(con)


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
        ("idx_insurance_premiums_fips_year", "mart.insurance_premiums_annual", "fips, year"),
        ("idx_insurance_non_renewal_fips_year", "mart.insurance_non_renewal_annual", "fips, year"),
        ("idx_statsamerica_fips_year", "mart.statsamerica_population_components_annual", "fips, year"),
        ("idx_statsamerica_bea_pci_fips_year", "mart.statsamerica_bea_per_capita_income_annual", "fips, year"),
        ("idx_statsamerica_bea_pi_fips_year", "mart.statsamerica_bea_personal_income_annual", "fips, year"),
        ("idx_statsamerica_cew_fips_year", "mart.statsamerica_cew_county_annual", "fips, year"),
        ("idx_statsamerica_cew_sector_fips_year", "mart.statsamerica_cew_county_sector_annual", "fips, year"),
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
        create_feature_marts(con)
        create_analysis_marts(con)
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
