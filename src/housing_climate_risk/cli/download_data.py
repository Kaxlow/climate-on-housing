from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from housing_climate_risk.cli import (
    acs_data,
    climate_damage_data,
    ncei_county_weather_data,
    statsamerica_bea_cew,
    statsamerica_population_components,
)
from housing_climate_risk.cli.federal_data import (
    _write_receipt_entry,
    download_census_boundaries,
    download_fema_disaster_declarations,
    download_fema_nri_counties,
)
from housing_climate_risk.paths import DATA_DIR


DIRECTORY_LAYOUT = (
    "20260401_county_processed_data",
    "acs",
    "archive",
    "cache",
    "climate",
    "climate_damage/raw/fema_web_disaster_summaries",
    "climate_damage/raw/noaa_storm_events",
    "fema",
    "fipsgeo",
    "housing",
    "statsamerica",
)

PRIVATE_INPUTS = {
    "redfin_housing": {
        "path": DATA_DIR / "housing" / "Redfin-Housing-Market-By-County.csv",
        "required": True,
        "columns": {
            "PERIOD_BEGIN", "PERIOD_END", "REGION_TYPE", "REGION", "STATE_CODE",
            "PROPERTY_TYPE", "MEDIAN_SALE_PRICE", "MEDIAN_PPSF", "MEDIAN_PPSF_YOY",
        },
        "instruction": (
            "Place the private Redfin county extract at "
            "data/housing/Redfin-Housing-Market-By-County.csv."
        ),
    },
    "fips_master": {
        "path": DATA_DIR / "fipsgeo" / "fips_master_v2.csv",
        "required": True,
        "columns": {
            "fips", "county_name", "state", "state_long", "msa_code", "msa_name",
            "msa_type", "csa_code", "csa_name",
        },
        "instruction": "Place the private county reference at data/fipsgeo/fips_master_v2.csv.",
    },
    "county_processed": {
        "path": DATA_DIR / "20260401_county_processed_data" / "county_processed_data.feather",
        "required": False,
        "columns": {
            "fips", "county_name", "state", "insurance_premiums_14_to_24",
            "insurance_non_renewal_rates",
        },
        "instruction": (
            "Optional: place county_processed_data.feather under "
            "data/20260401_county_processed_data/ to reproduce private insurance features."
        ),
    },
}

PUBLIC_CSV_SCHEMAS = {
    "fema/NRI_Table_Counties.csv": {
        "STATE", "COUNTY", "STCOFIPS", "RISK_SCORE", "RISK_RATNG", "NRI_VER",
    },
    "fema/FEMA_Disaster_Declarations.csv": {
        "disasterNumber", "state", "declarationType", "declarationDate",
        "incidentType", "fipsStateCode", "fipsCountyCode",
    },
    "climate_damage/noaa_storm_events_county_damage.csv": {
        "event_id", "state_fips", "cz_type", "cz_fips", "event_type", "total_damage",
    },
    "climate_damage/noaa_storm_events_zone_county_mapping.csv": {
        "state_fips", "cz_fips", "mapped_fips", "mapping_method",
    },
    "climate/ncei_climate_at_a_glance_county_monthly.csv": {
        "fips", "parameter", "year_month", "value", "source_url", "fetched_at",
    },
    "statsamerica/Components of Population Change - U.S., States, and Counties.csv": {
        "Statefips", "Countyfips", "Year", "Net International Migration",
        "Net Domestic Migration",
    },
    "statsamerica/BEA - US, States, Counties - Per Capita Income.csv": {
        "Statefips", "Countyfips", "Year", "BEA Per Capita Personal Income",
    },
    "statsamerica/BEA - US, States, Counties - Personal Income.csv": {
        "Statefips", "Countyfips", "Year", "Linecode", "Data",
    },
    "statsamerica/CEW - US, States, Counties - Total Ownership.csv": {
        "Statefips", "Countyfips", "Year", "NAICS Code", "Employment", "Wages",
    },
}


def create_directories() -> None:
    for relative in DIRECTORY_LAYOUT:
        (DATA_DIR / relative).mkdir(parents=True, exist_ok=True)


@contextmanager
def _cli_arguments(command: str, values: list[str]):
    previous = sys.argv
    sys.argv = [command, *values]
    try:
        yield
    finally:
        sys.argv = previous


def _run_cli(command: str, main: Callable[[], None], values: list[str]) -> None:
    with _cli_arguments(command, values):
        main()


def _columns(path: Path) -> set[str]:
    if path.suffix.lower() == ".feather":
        return set(pd.read_feather(path).columns)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return set(next(csv.reader(file)))


def validate_private_inputs() -> tuple[list[str], list[str]]:
    missing_required: list[str] = []
    notices: list[str] = []
    for name, spec in PRIVATE_INPUTS.items():
        path = Path(spec["path"])
        if not path.exists():
            message = str(spec["instruction"])
            (missing_required if spec["required"] else notices).append(message)
            continue
        missing_columns = sorted(set(spec["columns"]) - _columns(path))
        if missing_columns:
            missing_required.append(
                f"{path.relative_to(DATA_DIR.parent)} is present but missing columns: {missing_columns}"
            )
        else:
            size_mib = path.stat().st_size / (1024 * 1024)
            notices.append(f"Validated private input {name}: {path} ({size_mib:.2f} MiB)")
    return missing_required, notices


def _validate_public_outputs() -> list[str]:
    problems: list[str] = []
    for relative, expected in PUBLIC_CSV_SCHEMAS.items():
        path = DATA_DIR / relative
        if not path.exists():
            problems.append(
                f"Expected generated/downloaded file is missing: {path.relative_to(DATA_DIR.parent)}"
            )
            continue
        missing = sorted(expected - _columns(path))
        if missing:
            problems.append(
                f"{path.relative_to(DATA_DIR.parent)} is missing required columns: {missing}"
            )
    for path in (
        DATA_DIR / "fipsgeo" / "us_counties_boundaries_shapefile.json",
        DATA_DIR / "fipsgeo" / "census_state_boundaries" / "census_state_boundaries.shp",
    ):
        if not path.exists():
            problems.append(
                f"Expected generated/downloaded file is missing: {path.relative_to(DATA_DIR.parent)}"
            )
    return problems


def _record_bootstrap_source(
    key: str, source_url: str, destination: str, *, version_specific: bool
) -> None:
    _write_receipt_entry(
        key,
        {
            "source_url": source_url,
            "destination": destination,
            "version_specific_url": version_specific,
            "future_results_may_differ": not version_specific,
        },
    )


def bootstrap_all(*, force: bool = False, continue_on_error: bool = True) -> int:
    create_directories()
    failures: list[str] = []
    manual_required, notices = validate_private_inputs()

    def run(label: str, action: Callable[[], None]) -> bool:
        print(f"\n=== {label} ===", flush=True)
        try:
            action()
            return True
        except (Exception, SystemExit) as exc:
            message = f"{label}: {exc}"
            failures.append(message)
            print(f"[ERROR] {message}", file=sys.stderr, flush=True)
            if not continue_on_error:
                raise
            return False

    run("Census county/state boundaries", lambda: download_census_boundaries(force=force))
    run("FEMA National Risk Index counties", lambda: download_fema_nri_counties(force=force))
    run("FEMA disaster declarations", lambda: download_fema_disaster_declarations(force=force))

    acs_values = ["--include-population", "--include-migration"]
    if force:
        acs_values.append("--no-skip-existing")
    acs_ok = run(
        "Census ACS", lambda: _run_cli("download-data acs", acs_data.main, acs_values)
    )
    if acs_ok and any((DATA_DIR / "acs").glob("*.csv")):
        _record_bootstrap_source(
            "census_acs", "https://api.census.gov/data.html", "acs/",
            version_specific=True,
        )

    stats_values = ["--force"] if force else []
    stats_bea_ok = run(
        "StatsAmerica BEA and CEW",
        lambda: _run_cli(
            "download-data statsamerica-bea-cew", statsamerica_bea_cew.main, stats_values
        ),
    )
    if stats_bea_ok:
        _record_bootstrap_source(
            "statsamerica_bea_cew",
            "https://www.statsamerica.org/downloads/default.aspx",
            "statsamerica/",
            version_specific=False,
        )
    stats_population_ok = run(
        "StatsAmerica population components",
        lambda: _run_cli(
            "download-data statsamerica-population",
            statsamerica_population_components.main,
            stats_values,
        ),
    )
    if stats_population_ok:
        _record_bootstrap_source(
            "statsamerica_population_components",
            "https://www.statsamerica.org/downloads/default.aspx",
            "statsamerica/",
            version_specific=False,
        )

    if Path(PRIVATE_INPUTS["fips_master"]["path"]).exists():
        common_values = ["--force"] if force else []
        climate_damage_ok = run(
            "NOAA Storm Events, FEMA summaries, and derived NOAA county mappings",
            lambda: _run_cli(
                "download-data climate-damage", climate_damage_data.main, common_values
            ),
        )
        if climate_damage_ok:
            _record_bootstrap_source(
                "noaa_storm_events_and_fema_summaries",
                "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
                "climate_damage/",
                version_specific=True,
            )
        weather_ok = run(
            "NOAA NCEI county weather",
            lambda: _run_cli(
                "download-data ncei-weather", ncei_county_weather_data.main, common_values
            ),
        )
        if weather_ok:
            _record_bootstrap_source(
                "noaa_ncei_county_weather",
                "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/",
                "climate/ncei_climate_at_a_glance_county_monthly.csv",
                version_specific=False,
            )
    else:
        failures.append(
            "NOAA derived county outputs were skipped because the private "
            "data/fipsgeo/fips_master_v2.csv input is missing."
        )

    public_missing = _validate_public_outputs()
    print("\n=== Bootstrap summary ===")
    for notice in notices:
        print(f"[INFO] {notice}")
    if manual_required:
        print("\nManual retrieval required:")
        for item in manual_required:
            print(f"  - {item}")
    if failures or public_missing:
        print("\nDownload or validation problems:")
        for item in [*failures, *public_missing]:
            print(f"  - {item}")
    if not manual_required and not failures and not public_missing:
        print("[OK] All required public and private inputs are present and validated.")
        print(f"Retrieval metadata: {DATA_DIR / 'download_receipt.yaml'}")
        return 0
    print(
        "\nThe downloader uses the latest provider data. Mutable APIs and unversioned "
        "downloads may produce different results in the future; see "
        "data/download_receipt.yaml for resolved URLs and retrieval times."
    )
    return 2


DOWNLOADERS: dict[str, tuple[str, Callable[[], None]]] = {
    "acs": ("Census ACS county tables", acs_data.main),
    "climate-damage": (
        "NOAA Storm Events damage and FEMA financial summaries", climate_damage_data.main
    ),
    "ncei-weather": (
        "NOAA NCEI county monthly weather data", ncei_county_weather_data.main
    ),
    "statsamerica-bea-cew": (
        "StatsAmerica BEA and CEW datasets", statsamerica_bea_cew.main
    ),
    "statsamerica-population": (
        "StatsAmerica Components of Population Change", statsamerica_population_components.main
    ),
}


def parse_args(argv: list[str] | None = None) -> tuple[str, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    choices = [
        *DOWNLOADERS, "fema-nri", "fema-declarations", "census-boundaries", "all",
    ]
    if values and values[0] in choices:
        return values[0], values[1:]
    parser = argparse.ArgumentParser(
        description="Download source data used by the Quoll Intelligence pipeline."
    )
    parser.add_argument("source", choices=choices)
    args, source_args = parser.parse_known_args(values)
    return args.source, source_args


def main() -> None:
    source, source_args = parse_args()
    if source in DOWNLOADERS:
        _, downloader = DOWNLOADERS[source]
        _run_cli(f"download-data {source}", downloader, source_args)
        return

    parser = argparse.ArgumentParser(prog=f"download-data {source}")
    parser.add_argument("--force", action="store_true")
    if source == "all":
        parser.add_argument(
            "--fail-fast", action="store_true",
            help="Stop at the first provider error instead of completing the final report.",
        )
    args = parser.parse_args(source_args)
    if source == "fema-nri":
        download_fema_nri_counties(force=args.force)
    elif source == "fema-declarations":
        download_fema_disaster_declarations(force=args.force)
    elif source == "census-boundaries":
        download_census_boundaries(force=args.force)
    else:
        raise SystemExit(
            bootstrap_all(force=args.force, continue_on_error=not args.fail_fast)
        )


if __name__ == "__main__":
    main()
