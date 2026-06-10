from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from housing_climate_risk.paths import DATA_DIR


ACS_DIR = DATA_DIR / "acs"
REQUESTED_YEARS = list(range(2015, 2026))
TABLE_DESCRIPTIONS = {
    "S2503": "Financial characteristics for occupied housing units, including selected monthly owner costs, gross rent, household income, and affordability ratios.",
    "S2506": "Financial characteristics for owner-occupied housing units with a mortgage.",
    "S2507": "Financial characteristics for owner-occupied housing units without a mortgage.",
    "B25132": "Monthly housing costs by tenure and mortgage status.",
    "B25133": "Monthly housing costs by tenure and units in structure.",
    "B25134": "Monthly housing costs by household income in the past 12 months.",
    "B25135": "Median monthly housing costs by tenure and mortgage status.",
    "B25141": "Median gross rent by number of bedrooms.",
    "DP02": "Selected social characteristics, including households, relationships, marital status, education, veterans, disability, residence, place of birth, language, and ancestry.",
    "DP03": "Selected economic characteristics, including employment, commuting, occupation, industry, income, poverty, and health insurance.",
    "DP04": "Selected housing characteristics, including occupancy, structure type, rooms, tenure, housing value, rent, and housing costs.",
    "DP05": "Demographic and housing estimates, including age, sex, race, Hispanic origin, households, and housing occupancy.",
}


def _table_output_pattern(table: str) -> list[str]:
    patterns = [f"census_acs5_county_{table.lower()}_*.csv"]
    if table == "S2503":
        patterns.append("census_acs5_county_housing_financial_characteristics_*.csv")
    return patterns


def _years_from_filename(path: Path) -> set[int]:
    match = re.search(r"_(\d{4})_(\d{4})$", path.stem)
    if not match:
        return set()
    start, end = map(int, match.groups())
    return set(range(start, end + 1))


def _downloaded_years(output_dir: Path, table: str) -> set[int]:
    years: set[int] = set()
    for pattern in _table_output_pattern(table):
        for path in output_dir.glob(pattern):
            years.update(_years_from_filename(path))
    return years


def _failure_years(output_dir: Path, table: str) -> set[int]:
    years: set[int] = set()
    for path in output_dir.glob(f"census_acs5_{table.lower()}_failures_*.csv"):
        try:
            df = pd.read_csv(path, usecols=["year"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        years.update(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).tolist())
    return years


def build_coverage_summary(output_dir: Path, requested_years: list[int]) -> pd.DataFrame:
    requested = set(requested_years)
    rows: list[dict[str, str]] = []
    for table, description in TABLE_DESCRIPTIONS.items():
        downloaded = _downloaded_years(output_dir, table) & requested
        failures = _failure_years(output_dir, table) & requested
        missing = requested - downloaded
        rows.append(
            {
                "table": table,
                "description": description,
                "requested_years": _format_years(requested_years),
                "downloaded_years": _format_years(sorted(downloaded)),
                "missing_years": _format_years(sorted(missing)),
                "failure_years_recorded": _format_years(sorted(failures)),
                "status": "complete" if not missing else "partial" if downloaded else "missing",
            }
        )
    return pd.DataFrame(rows)


def _format_years(years: list[int] | set[int]) -> str:
    values = sorted(years)
    return ", ".join(str(year) for year in values) if values else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ACS housing table coverage in data/acs.")
    parser.add_argument("--output-dir", type=Path, default=ACS_DIR)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    summary = build_coverage_summary(args.output_dir, years)
    output_path = args.output_dir / f"census_acs5_housing_table_coverage_{args.start_year}_{args.end_year}.csv"
    summary.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(output_path)


if __name__ == "__main__":
    main()
