from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from housing_climate_risk.paths import DATA_DIR


ACS_DIR = DATA_DIR / "acs"
CENSUS_TOKEN_ENV_VARS = ["CENSUS_API_KEY", "CENSUS_KEY", "CENSUS_API_TOKEN"]
ACS_SPECIAL_VALUES = {"-222222222", "-333333333", "-555555555", "-666666666", "-888888888", "-999999999"}

DEFAULT_TABLES = {
    "S2503": "Financial Characteristics",
    "S2506": "Financial Characteristics for Housing Units With a Mortgage",
    "S2507": "Financial Characteristics for Housing Units Without a Mortgage",
    "B25132": "Monthly Housing Costs",
    "B25133": "Monthly Housing Costs by Units in Structure",
    "B25134": "Monthly Housing Costs by Household Income in the Past 12 Months",
    "B25135": "Median Monthly Housing Costs",
    "B25141": "Median Gross Rent by Bedrooms",
    "DP02": "Selected Social Characteristics in the United States",
    "DP03": "Selected Economic Characteristics",
    "DP04": "Selected Housing Characteristics",
    "DP05": "ACS Demographic and Housing Estimates",
}
POPULATION_VARIABLES = {"B01003_001E": "total_population", "B01003_001M": "total_population_moe"}
MIGRATION_VARIABLES = {
    "B07001_001E": "migration_population_1yr_plus",
    "B07001_001M": "migration_population_1yr_plus_moe",
    "B07001_017E": "same_house_1_year_ago",
    "B07001_017M": "same_house_1_year_ago_moe",
    "B07001_033E": "moved_within_same_county",
    "B07001_033M": "moved_within_same_county_moe",
    "B07001_049E": "moved_from_different_county_same_state",
    "B07001_049M": "moved_from_different_county_same_state_moe",
    "B07001_065E": "moved_from_different_state",
    "B07001_065M": "moved_from_different_state_moe",
    "B07001_081E": "moved_from_abroad",
    "B07001_081M": "moved_from_abroad_moe",
}
MIGRATION_PR_VARIABLES = {key.replace("B07001", "B07001PR"): value for key, value in MIGRATION_VARIABLES.items()}


def _first_env_value(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip().strip("\"'")
    return None


def _api_key(api_key: str | None) -> str:
    resolved = (api_key or _first_env_value(CENSUS_TOKEN_ENV_VARS) or "").strip().strip("\"'")
    if not resolved:
        raise RuntimeError(f"One of {', '.join(CENSUS_TOKEN_ENV_VARS)} is required by the Census API.")
    return resolved


def _request_json(url: str, *, timeout: int = 90) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "quoll-intelligence/acs-downloader", "Accept": "application/json,text/plain,*/*"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8-sig")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        snippet = " ".join(body[:500].split())
        raise RuntimeError(f"Census returned non-JSON response: {snippet}") from exc


def _census_url(year: int, dataset: str, params: dict[str, str]) -> str:
    return f"https://api.census.gov/data/{year}/{dataset}?{urllib.parse.urlencode(params, safe=',:*')}"


def _dataset_for_table(table_id: str) -> str:
    table_id = table_id.upper()
    if table_id.startswith("S"):
        return "acs/acs5/subject"
    if table_id.startswith("DP"):
        return "acs/acs5/profile"
    return "acs/acs5"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text in ACS_SPECIAL_VALUES else text


def _safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _rate(numerator: str, denominator: str) -> str:
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    return "" if num is None or den in (None, 0) else f"{num / den:.8f}"


def _fix_malformed_percentages(df: pd.DataFrame, variables: list[str], year: int, variable_metadata: list[dict[str, str]] | None = None) -> pd.DataFrame:
    """
    Fix Census API bug where PE (percent estimate) columns contain raw counts instead of percentages.

    This primarily affects DP (profile) tables in years 2015-2018 where the API returns estimate
    counts in PE columns instead of the calculated percentages.

    Detection: PE value > 100 or PE value equals its corresponding E value
    Fix: When PE == E, the value is a raw count. We cannot reliably infer the denominator from
         variable numbering (Census renumbered variables between years), so we leave these values
         as-is and rely on downstream processing to handle them (e.g., database negative value cleaning).

    Actually, a simpler fix: if PE > 100, it's definitely wrong. We can try common patterns:
    - Look for a "Total" variable in the same group by searching backwards for lower-numbered variables
    - If that fails, mark the value as null
    """
    pe_columns = [v for v in variables if v.endswith("PE")]
    if not pe_columns:
        return df

    # Build a lookup of variable labels if provided
    label_lookup = {}
    if variable_metadata:
        label_lookup = {row["variable"]: row.get("label", "") for row in variable_metadata}

    fixes_applied = 0
    nulled = 0

    for pe_col in pe_columns:
        base = pe_col[:-2]  # Remove 'PE' suffix
        e_col = f"{base}E"

        if e_col not in df.columns:
            continue

        # Detect malformed percentages: PE > 100 or PE == E (indicating raw count, not percentage)
        pe_numeric = pd.to_numeric(df[pe_col], errors='coerce')
        e_numeric = pd.to_numeric(df[e_col], errors='coerce')

        # Find rows where PE looks like a raw count instead of a percentage
        malformed_mask = (pe_numeric > 100) | (pe_numeric == e_numeric)
        malformed_count = malformed_mask.sum()

        if malformed_count == 0:
            continue

        # Try to find the parent/total column by looking for "Total" in labels
        # Parse variable number
        variable_parts = base.split('_')
        parent_found = False

        if len(variable_parts) >= 2:
            prefix = variable_parts[0]
            try:
                var_num = int(variable_parts[1])

                # Search backwards for a "Total" variable in same group (within 10 variables)
                for offset in range(1, min(var_num, 11)):
                    parent_e_col = f"{prefix}_{var_num-offset:04d}E"
                    if parent_e_col not in df.columns:
                        continue

                    # Check if this looks like a total (from label) AND parent value >= child value
                    parent_label = label_lookup.get(parent_e_col, "").lower()
                    is_total = "total" in parent_label

                    # Parent value must be >= child value (totals should be larger)
                    # Only check rows where both parent and child are non-null
                    parent_numeric = pd.to_numeric(df[parent_e_col], errors='coerce')
                    valid_comparison_mask = parent_numeric.notna() & e_numeric.notna()
                    is_plausible_parent = valid_comparison_mask.any() and (parent_numeric[valid_comparison_mask] >= e_numeric[valid_comparison_mask]).all()

                    # Both conditions must be true: label suggests it's a total AND values make sense
                    if is_total and is_plausible_parent:
                        # Calculate correct percentage: (E / parent_E) * 100
                        calculated_pct = (e_numeric / parent_numeric * 100).where(
                            (parent_numeric > 0) & malformed_mask
                        )

                        # Only apply fix where we detected malformed data and calculation succeeded
                        # and percentage is reasonable (0-100%)
                        fix_mask = malformed_mask & calculated_pct.notna() & (calculated_pct <= 100)
                        df.loc[fix_mask, pe_col] = calculated_pct[fix_mask].round(1).astype(str)
                        fixes_applied += fix_mask.sum()
                        parent_found = True
                        break

            except (ValueError, IndexError):
                pass

        # If no parent found, null out the malformed values
        if not parent_found and malformed_count > 0:
            df.loc[malformed_mask, pe_col] = None
            nulled += malformed_count

    # Final pass: null out any remaining PE values > 100 (these are definitively wrong)
    for pe_col in pe_columns:
        if pe_col not in df.columns:
            continue
        pe_numeric = pd.to_numeric(df[pe_col], errors='coerce')
        still_malformed = pe_numeric > 100
        if still_malformed.sum() > 0:
            df.loc[still_malformed, pe_col] = None
            nulled += still_malformed.sum()

    if fixes_applied > 0:
        print(f"  Fixed {fixes_applied} malformed percentage values in year {year}", flush=True)
    if nulled > 0:
        print(f"  Nulled {nulled} malformed percentage values (no parent found or still > 100%) in year {year}", flush=True)

    return df


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _years_from_output(path: Path) -> tuple[int, int] | None:
    pieces = path.stem.rsplit("_", 2)
    if len(pieces) < 3:
        return None
    try:
        return int(pieces[-2]), int(pieces[-1])
    except ValueError:
        return None


def _existing_output(output_dir: Path, stem_prefix: str, start_year: int, end_year: int, *, extra_prefixes: list[str] | None = None) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for prefix in [stem_prefix, *(extra_prefixes or [])]:
        for path in output_dir.glob(f"{prefix}_*.csv"):
            years = _years_from_output(path)
            if not years:
                continue
            file_start, file_end = years
            if file_start == start_year and start_year <= file_end <= end_year:
                candidates.append((file_end, path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["field", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_failures(path: Path, failures: list[dict[str, str]]) -> Path | None:
    if not failures:
        return None
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(failures[0]))
        writer.writeheader()
        writer.writerows(failures)
    return path


def _table_variables(year: int, table_id: str, api_key: str) -> tuple[list[str], list[dict[str, str]]]:
    dataset = _dataset_for_table(table_id)
    url = f"https://api.census.gov/data/{year}/{dataset}/groups/{table_id}.json?{urllib.parse.urlencode({'key': api_key})}"
    metadata = _request_json(url)
    variables = metadata.get("variables", {})
    selected: list[str] = []
    rows: list[dict[str, str]] = []
    for variable, info in sorted(variables.items()):
        if not variable.startswith(f"{table_id}_"):
            continue
        # Include E (estimate), M (margin of error), PE (percent estimate), PM (percent margin of error)
        if not (variable.endswith("E") or variable.endswith("M") or variable.endswith("PE") or variable.endswith("PM")):
            continue
        if variable.endswith("EA") or variable.endswith("MA"):
            continue
        selected.append(variable)
        rows.append(
            {
                "variable": variable,
                "label": str(info.get("label", "")),
                "concept": str(info.get("concept", "")),
                "predicate_type": str(info.get("predicateType", "")),
                "group": str(info.get("group", "")),
            }
        )
    if not selected:
        raise RuntimeError(f"No estimate or margin-of-error variables found for {table_id}.")
    return selected, rows


def download_table(
    table_id: str,
    start_year: int,
    end_year: int,
    output_dir: Path,
    *,
    api_key: str,
    chunk_size: int,
    sleep_seconds: float,
    skip_existing: bool,
) -> list[Path]:
    table_id = table_id.upper()
    prefix = f"census_acs5_county_{table_id.lower()}"
    extra_prefixes = ["census_acs5_county_housing_financial_characteristics"] if table_id == "S2503" else None
    if skip_existing and (existing := _existing_output(output_dir, prefix, start_year, end_year, extra_prefixes=extra_prefixes)):
        print(f"Skipping {table_id}: existing output found at {existing}", flush=True)
        return [existing]

    dataset = _dataset_for_table(table_id)
    frames: list[pd.DataFrame] = []
    variable_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for year in range(start_year, end_year + 1):
        print(f"Downloading ACS {year} {table_id} county data...", flush=True)
        try:
            variables, year_variable_rows = _table_variables(year, table_id, api_key)
            merged: pd.DataFrame | None = None
            for chunk in _chunks(variables, chunk_size):
                payload = _request_json(
                    _census_url(
                        year,
                        dataset,
                        {"get": ",".join(["NAME", *chunk]), "for": "county:*", "in": "state:*", "key": api_key},
                    )
                )
                chunk_df = pd.DataFrame(payload[1:], columns=payload[0])
                merged = chunk_df if merged is None else merged.merge(chunk_df, on=["NAME", "state", "county"], how="outer")
                time.sleep(sleep_seconds)
            if merged is None:
                raise RuntimeError("No rows returned.")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
            failures.append({"year": str(year), "dataset": dataset, "table": table_id, "error": str(exc)})
            print(f"Skipping ACS {year} {table_id}: {exc}", flush=True)
            continue

        merged["year"] = year
        merged["state"] = merged["state"].astype(str).str.zfill(2)
        merged["county"] = merged["county"].astype(str).str.zfill(3)
        merged["county_fips"] = merged["state"] + merged["county"]
        for col in variables:
            merged[col] = merged[col].map(_clean)

        # Fix malformed percentage columns (Census API bug in years 2015-2018 for DP tables)
        merged = _fix_malformed_percentages(merged, variables, year, year_variable_rows)

        frames.append(merged[["year", "county_fips", "state", "county", "NAME", *variables]])
        variable_rows.extend({"year": str(year), **row} for row in year_variable_rows)

    outputs: list[Path] = []
    if not frames:
        failure_path = output_dir / f"census_acs5_{table_id.lower()}_failures_{start_year}_{end_year}.csv"
        _write_failures(failure_path, failures)
        print(f"No rows downloaded for {table_id}; wrote failures to {failure_path}", flush=True)
        return [failure_path]

    output = pd.concat(frames, ignore_index=True, sort=False).sort_values(["year", "county_fips"])
    min_year = int(output["year"].min())
    max_year = int(output["year"].max())
    output_path = output_dir / f"{prefix}_{min_year}_{max_year}.csv"
    output.to_csv(output_path, index=False)
    outputs.append(output_path)

    variables_path = output_dir / f"census_acs5_{table_id.lower()}_variable_dictionary_{min_year}_{max_year}.csv"
    pd.DataFrame(variable_rows).drop_duplicates().to_csv(variables_path, index=False)
    outputs.append(variables_path)

    failure_path = _write_failures(output_dir / f"census_acs5_{table_id.lower()}_failures_{start_year}_{end_year}.csv", failures)
    if failure_path:
        outputs.append(failure_path)

    metadata_path = output_dir / f"{prefix}_metadata_{min_year}_{max_year}.csv"
    metadata_rows = [
        {"field": "dataset", "value": f"Census ACS 5-Year {table_id}"},
        {"field": "table_name", "value": DEFAULT_TABLES.get(table_id, table_id)},
        {"field": "api_dataset", "value": dataset},
        {"field": "geography", "value": "County and county-equivalent geographies from for=county:*&in=state:*"},
        {"field": "requested_years", "value": f"{start_year}-{end_year}"},
        {"field": "downloaded_years", "value": f"{min_year}-{max_year}"},
        {"field": "downloaded_on", "value": date.today().isoformat()},
        {"field": "row_count", "value": str(len(output))},
        {"field": "variable_dictionary", "value": str(variables_path)},
    ]
    if failure_path:
        metadata_rows.append({"field": "failures", "value": str(failure_path)})
    _write_metadata(metadata_path, metadata_rows)
    outputs.append(metadata_path)
    return outputs


def download_population(start_year: int, end_year: int, output_dir: Path, *, api_key: str, sleep_seconds: float, skip_existing: bool) -> list[Path]:
    prefix = "census_acs5_county_population"
    if skip_existing and (existing := _existing_output(output_dir, prefix, start_year, end_year)):
        print(f"Skipping population: existing output found at {existing}", flush=True)
        return [existing]
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    variables = list(POPULATION_VARIABLES)
    for year in range(start_year, end_year + 1):
        url = _census_url(year, "acs/acs5", {"get": ",".join(["NAME", *variables]), "for": "county:*", "in": "state:*", "key": api_key})
        try:
            payload = _request_json(url)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            failures.append({"year": str(year), "url": url, "error": str(exc)})
            continue
        for values in payload[1:]:
            record = dict(zip(payload[0], values, strict=True))
            state = record["state"].zfill(2)
            county = record["county"].zfill(3)
            rows.append({"year": str(year), "county_fips": f"{state}{county}", "state": state, "county": county, "name": record["NAME"], **{new: _clean(record[old]) for old, new in POPULATION_VARIABLES.items()}})
        time.sleep(sleep_seconds)
    if not rows:
        raise RuntimeError("No ACS county population data downloaded.")
    latest = max(int(row["year"]) for row in rows)
    output_path = output_dir / f"{prefix}_{start_year}_{latest}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["year"], row["county_fips"])))
    outputs = [output_path]
    failure_path = _write_failures(output_dir / f"{prefix}_failures_{start_year}_{end_year}.csv", failures)
    if failure_path:
        outputs.append(failure_path)
    return outputs


def _migration_row(year: str, state: str, county: str, name: str, cleaned: dict[str, str]) -> dict[str, str]:
    total_moved = sum(_safe_float(cleaned[col]) or 0 for col in ["moved_within_same_county", "moved_from_different_county_same_state", "moved_from_different_state", "moved_from_abroad"])
    domestic_in_migration = sum(_safe_float(cleaned[col]) or 0 for col in ["moved_from_different_county_same_state", "moved_from_different_state"])
    row = {"year": year, "county_fips": f"{state}{county}", "state": state, "county": county, "name": name, **cleaned, "total_moved": str(int(total_moved)) if total_moved.is_integer() else str(total_moved), "domestic_in_migration": str(int(domestic_in_migration)) if domestic_in_migration.is_integer() else str(domestic_in_migration)}
    total = cleaned["migration_population_1yr_plus"]
    row["same_house_rate"] = _rate(row["same_house_1_year_ago"], total)
    row["total_moved_rate"] = _rate(row["total_moved"], total)
    row["moved_within_same_county_rate"] = _rate(row["moved_within_same_county"], total)
    row["domestic_in_migration_rate"] = _rate(row["domestic_in_migration"], total)
    row["moved_from_abroad_rate"] = _rate(row["moved_from_abroad"], total)
    return row


def download_migration(start_year: int, end_year: int, output_dir: Path, *, api_key: str, sleep_seconds: float, skip_existing: bool) -> list[Path]:
    prefix = "census_acs5_county_migration"
    if skip_existing and (existing := _existing_output(output_dir, prefix, start_year, end_year)):
        print(f"Skipping migration: existing output found at {existing}", flush=True)
        return [existing]
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for year in range(start_year, end_year + 1):
        url = _census_url(year, "acs/acs5", {"get": ",".join(["NAME", *MIGRATION_VARIABLES]), "for": "county:*", "in": "state:*", "key": api_key})
        try:
            payload = _request_json(url)
            for values in payload[1:]:
                record = dict(zip(payload[0], values, strict=True))
                state = record["state"].zfill(2)
                if state == "72":
                    continue
                county = record["county"].zfill(3)
                cleaned = {new: _clean(record[old]) for old, new in MIGRATION_VARIABLES.items()}
                rows.append(_migration_row(str(year), state, county, record["NAME"], cleaned))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            failures.append({"year": str(year), "url": url, "error": str(exc)})
        pr_url = _census_url(year, "acs/acs5", {"get": ",".join(["NAME", *MIGRATION_PR_VARIABLES]), "for": "county:*", "in": "state:72", "key": api_key})
        try:
            pr_payload = _request_json(pr_url)
            for values in pr_payload[1:]:
                record = dict(zip(pr_payload[0], values, strict=True))
                state = record["state"].zfill(2)
                county = record["county"].zfill(3)
                cleaned = {new: _clean(record[old]) for old, new in MIGRATION_PR_VARIABLES.items()}
                rows.append(_migration_row(str(year), state, county, record["NAME"], cleaned))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            failures.append({"year": str(year), "url": pr_url, "error": str(exc)})
        time.sleep(sleep_seconds)
    if not rows:
        raise RuntimeError("No ACS county migration data downloaded.")
    latest = max(int(row["year"]) for row in rows)
    output_path = output_dir / f"{prefix}_{start_year}_{latest}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["year"], row["county_fips"])))
    outputs = [output_path]
    failure_path = _write_failures(output_dir / f"{prefix}_failures_{start_year}_{end_year}.csv", failures)
    if failure_path:
        outputs.append(failure_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ACS 5-year county data into data/acs.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=ACS_DIR)
    parser.add_argument("--census-api-key")
    parser.add_argument("--tables", nargs="+", default=list(DEFAULT_TABLES), help="ACS subject, detailed, or profile tables to download.")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--include-population", action="store_true")
    parser.add_argument("--include-migration", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=45)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_key = _api_key(args.census_api_key)
    outputs: list[Path] = []
    skip_existing = not args.no_skip_existing

    if not args.skip_tables:
        for table_id in args.tables:
            outputs.extend(
                download_table(
                    table_id,
                    args.start_year,
                    args.end_year,
                    args.output_dir,
                    api_key=api_key,
                    chunk_size=args.chunk_size,
                    sleep_seconds=args.sleep_seconds,
                    skip_existing=skip_existing,
                )
            )
    if args.include_population:
        outputs.extend(download_population(args.start_year, args.end_year, args.output_dir, api_key=api_key, sleep_seconds=args.sleep_seconds, skip_existing=skip_existing))
    if args.include_migration:
        outputs.extend(download_migration(args.start_year, args.end_year, args.output_dir, api_key=api_key, sleep_seconds=args.sleep_seconds, skip_existing=skip_existing))

    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
