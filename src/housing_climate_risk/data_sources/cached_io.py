from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl


DEFAULT_CACHE_DIR = Path("data/cache")


FEMA_DISASTER_COLUMNS = [
    "disasterNumber",
    "state",
    "declarationDate",
    "incidentType",
    "declarationTitle",
    "ihProgramDeclared",
    "iaProgramDeclared",
    "paProgramDeclared",
    "hmProgramDeclared",
    "incidentBeginDate",
    "incidentEndDate",
    "designatedArea",
    "designatedIncidentTypes",
    "fipsStateCode",
    "fipsCountyCode",
]


REDFIN_COUNTY_COLUMNS = [
    "REGION",
    "STATE_CODE",
    "PARENT_METRO_REGION",
    "PROPERTY_TYPE",
    "PERIOD_BEGIN",
    "PERIOD_END",
    "MEDIAN_PPSF",
    "MEDIAN_PPSF_YOY",
    "MEDIAN_LIST_PPSF",
    "MEDIAN_LIST_PPSF_YOY",
    "HOMES_SOLD",
    "HOMES_SOLD_YOY",
    "PENDING_SALES",
    "PENDING_SALES_YOY",
    "INVENTORY",
    "INVENTORY_YOY",
    "MONTHS_OF_SUPPLY",
    "MONTHS_OF_SUPPLY_YOY",
    "MEDIAN_DOM",
    "MEDIAN_DOM_YOY",
    "AVG_SALE_TO_LIST",
    "AVG_SALE_TO_LIST_YOY",
    "PRICE_DROPS",
    "PRICE_DROPS_YOY",
]


def _fresh_cache(cache_path: Path, source_path: Path) -> bool:
    return cache_path.exists() and cache_path.stat().st_mtime >= source_path.stat().st_mtime


def _cache_path(source_path: Path, cache_dir: Path, suffix: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{source_path.stem}_{suffix}.parquet"


def _collect_or_read(cache_path: Path, source_path: Path, lf: pl.LazyFrame) -> pl.DataFrame:
    if _fresh_cache(cache_path, source_path):
        return pl.read_parquet(cache_path)
    df = lf.collect()
    df.write_parquet(cache_path)
    return df


def read_projected_csv_cached(
    source_path: str | Path,
    columns: list[str],
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    cache_suffix: str = "projected",
) -> pd.DataFrame:
    source_path = Path(source_path)
    cache_path = _cache_path(source_path, Path(cache_dir), cache_suffix)
    lf = pl.scan_csv(source_path, infer_schema_length=10_000, ignore_errors=True).select(columns)
    return _collect_or_read(cache_path, source_path, lf).to_pandas()


def read_fema_disasters_cached(
    source_path: str | Path,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    return read_projected_csv_cached(
        source_path,
        FEMA_DISASTER_COLUMNS,
        cache_dir=cache_dir,
        cache_suffix="projected_fema",
    )


def read_redfin_county_cached(
    source_path: str | Path,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    source_path = Path(source_path)
    cache_path = _cache_path(source_path, Path(cache_dir), "county_all_residential_projected")
    lf = (
        pl.scan_csv(source_path, infer_schema_length=10_000, ignore_errors=True)
        .select(REDFIN_COUNTY_COLUMNS)
        .filter(pl.col("PROPERTY_TYPE") == "All Residential")
    )
    return _collect_or_read(cache_path, source_path, lf).to_pandas()


def read_cew_totals_cached(
    source_path: str | Path,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    source_path = Path(source_path)
    cache_path = _cache_path(source_path, Path(cache_dir), "county_totals")
    lf = (
        pl.scan_csv(
            source_path,
            infer_schema_length=10_000,
            ignore_errors=True,
        )
        .select(
            [
                "Statefips",
                "Countyfips",
                "Description",
                "Year",
                "Ownership Code",
                "NAICS Code",
                "Employment",
                "Wages",
                "Average Wage",
                "Average Weekly Wage",
            ]
        )
        .with_columns(
            pl.col("Ownership Code").cast(pl.Int64, strict=False),
            pl.col("NAICS Code").cast(pl.Int64, strict=False),
            pl.col("Statefips").cast(pl.Int64, strict=False),
            pl.col("Countyfips").cast(pl.Int64, strict=False),
        )
        .filter((pl.col("Ownership Code") == 0) & (pl.col("NAICS Code") == 0))
        .filter(pl.col("Statefips").is_not_null() & pl.col("Countyfips").is_not_null())
        .with_columns(
            pl.col("Statefips").cast(pl.Int64).cast(pl.Utf8).str.pad_start(2, "0").alias("state_fips"),
            pl.col("Countyfips").cast(pl.Int64).cast(pl.Utf8).str.pad_start(3, "0").alias("county_fips"),
        )
        .filter((pl.col("state_fips") != "00") & (pl.col("county_fips") != "000"))
        .with_columns((pl.col("state_fips") + pl.col("county_fips")).alias("fips"))
        .select(["fips", "Description", "Year", "Employment", "Wages", "Average Wage", "Average Weekly Wage"])
        .unique(subset=["fips", "Year"], keep="first", maintain_order=True)
    )
    return _collect_or_read(cache_path, source_path, lf).to_pandas()


def read_cew_employment_wages_cached(
    source_path: str | Path,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    source_path = Path(source_path)
    cache_path = _cache_path(source_path, Path(cache_dir), "county_employment_wages")
    lf = (
        pl.scan_csv(source_path, infer_schema_length=10_000, ignore_errors=True)
        .select(
            [
                "IBRC_GEO_ID",
                "Year",
                "Employment",
                "Wages",
                "Average Wage",
                "Ownership Code Description",
                "NAICS Code",
            ]
        )
        .with_columns(
            pl.col("IBRC_GEO_ID")
            .cast(pl.Utf8)
            .str.replace(".0", "", literal=True)
            .str.pad_start(5, "0")
            .alias("county_fips"),
            pl.col("Year").cast(pl.Int64, strict=False).alias("cew_year"),
            pl.col("Employment").cast(pl.Float64, strict=False).alias("employment"),
            pl.col("Wages").cast(pl.Float64, strict=False).alias("wages"),
            pl.col("Average Wage").cast(pl.Float64, strict=False).alias("average_wage"),
            pl.col("Ownership Code Description").cast(pl.Utf8).str.strip_chars().alias("ownership_desc"),
            pl.col("NAICS Code").cast(pl.Utf8).str.strip_chars().alias("naics_code"),
        )
        .filter(
            (pl.col("county_fips").str.len_chars() == 5)
            & (pl.col("county_fips") != "00000")
            & (pl.col("ownership_desc") == "All")
            & (pl.col("naics_code").is_in(["00", "0"]))
        )
        .with_columns(
            pl.when(pl.col("average_wage").is_not_null())
            .then(pl.col("average_wage"))
            .otherwise(pl.col("wages") / pl.when(pl.col("employment") == 0).then(None).otherwise(pl.col("employment")))
            .alias("average_wage_per_job")
        )
        .select(["county_fips", "cew_year", "employment", "average_wage_per_job"])
        .unique(subset=["county_fips", "cew_year"], keep="first", maintain_order=True)
    )
    return _collect_or_read(cache_path, source_path, lf).to_pandas()
