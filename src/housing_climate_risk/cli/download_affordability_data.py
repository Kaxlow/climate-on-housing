from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from housing_climate_risk.paths import DATA_DIR, HOUSING_DIR


AFFORDABILITY_DIR = DATA_DIR / "affordability"
ACS_DIR = DATA_DIR / "acs"
HUD_CHAS_BASE_URL = "https://www.huduser.gov/hudapi/public/chas"

ACS_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B25064_001E": "median_gross_rent",
    "B25077_001E": "median_home_value",
    "B25088_002E": "median_owner_costs_mortgage",
    "B25088_003E": "median_owner_costs_no_mortgage",
    "B25091_001E": "median_owner_costs_pct_income",
    "B25070_007E": "renter_30_to_34pct_income",
    "B25070_008E": "renter_35_to_39pct_income",
    "B25070_009E": "renter_40_to_49pct_income",
    "B25070_010E": "renter_50pct_plus_income",
    "B25091_008E": "owner_mortgage_30_to_34pct_income",
    "B25091_009E": "owner_mortgage_35_to_39pct_income",
    "B25091_010E": "owner_mortgage_40_to_49pct_income",
    "B25091_011E": "owner_mortgage_50pct_plus_income",
}

CHAS_YEARS = [
    "2008-2012",
    "2009-2013",
    "2010-2014",
    "2011-2015",
    "2012-2016",
    "2013-2017",
    "2014-2018",
    "2015-2019",
    "2016-2020",
    "2017-2021",
    "2018-2022",
]

STATE_FIPS = [
    "01",
    "02",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "53",
    "54",
    "55",
    "56",
]

HUD_TOKEN_ENV_VARS = [
    "HUD_USER_TOKEN",
    "HUD_API_TOKEN",
    "HUD_API_KEY",
    "HUD_CHAS_TOKEN",
    "HUD_CHAS_API_KEY",
]

CENSUS_TOKEN_ENV_VARS = [
    "CENSUS_API_KEY",
    "CENSUS_KEY",
    "CENSUS_API_TOKEN",
]


def _first_env_value(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip().strip("\"'")
    return None


def _masked(value: str | None) -> str:
    if not value:
        return "missing"
    if len(value) <= 8:
        return "set"
    return f"set ({value[:4]}...{value[-4:]})"


def _request_text(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 90,
    retries: int = 0,
    retry_delay: float = 1.0,
) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "housing-climate-risk/0.1 data downloader",
            "Accept": "text/csv,application/json,text/plain,*/*",
            **(headers or {}),
        },
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8-sig")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt >= retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else retry_delay * (2**attempt)
            time.sleep(delay)
    raise RuntimeError("HTTP request retry loop exited unexpectedly.")


def _json_from_text(text: str, *, source: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = " ".join(text[:500].split())
        raise RuntimeError(f"{source} returned non-JSON response: {snippet}") from exc


def _dedupe_dicts(records: Iterable[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = tuple(record.get(field) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 90,
    retries: int = 0,
    retry_delay: float = 1.0,
) -> Any:
    text = _request_text(
        url,
        headers={
            "Accept": "application/json",
            **(headers or {}),
        },
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    return _json_from_text(text, source=url)


def _write_source_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "coverage", "source_url", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def _append_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def infer_housing_period() -> tuple[int, int]:
    redfin_path = HOUSING_DIR / "Redfin-Housing-Market-By-County.csv"
    df = pd.read_csv(redfin_path, usecols=["PERIOD_BEGIN", "PROPERTY_TYPE"], low_memory=False)
    df = df[df["PROPERTY_TYPE"].eq("All Residential")].copy()
    years = pd.to_datetime(df["PERIOD_BEGIN"], errors="coerce").dt.year.dropna().astype(int)
    return int(years.min()), int(years.max())


def download_acs_county(start_year: int, end_year: int, output_dir: Path, *, api_key: str | None = None) -> Path:
    rows: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    api_key = (api_key or _first_env_value(CENSUS_TOKEN_ENV_VARS) or "").strip().strip("\"'")
    if not api_key:
        raise RuntimeError(f"One of {', '.join(CENSUS_TOKEN_ENV_VARS)} is required by the Census API in this environment.")
    requested_years = range(start_year, min(end_year, 2024) + 1)
    variables = ["NAME", *ACS_VARIABLES.keys()]
    query = urllib.parse.urlencode(
        {
            "get": ",".join(variables),
            "for": "county:*",
            "in": "state:*",
            "key": api_key,
        },
        safe=",:*",
    )

    for year in requested_years:
        url = f"https://api.census.gov/data/{year}/acs/acs5?{query}"
        try:
            payload = _json_from_text(_request_text(url), source=f"ACS {year}")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"Skipping ACS {year}: {exc}")
            failures.append({"year": str(year), "error": str(exc)})
            continue
        if not payload:
            continue
        header, values = payload[0], payload[1:]
        year_df = pd.DataFrame(values, columns=header)
        year_df["year"] = year
        rows.append(year_df)
        time.sleep(0.2)

    if not rows:
        if failures:
            pd.DataFrame(failures).to_csv(output_dir / "census_acs5_api_failures.csv", index=False)
        raise RuntimeError("No ACS county data downloaded.")
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "census_acs5_api_failures.csv", index=False)

    acs = pd.concat(rows, ignore_index=True)
    acs["county_fips"] = acs["state"].astype(str).str.zfill(2) + acs["county"].astype(str).str.zfill(3)
    acs = acs.rename(columns=ACS_VARIABLES)
    numeric_cols = [*ACS_VARIABLES.values()]
    for col in numeric_cols:
        acs[col] = pd.to_numeric(acs[col], errors="coerce")

    acs["renter_cost_burden_30pct_plus"] = acs[
        [
            "renter_30_to_34pct_income",
            "renter_35_to_39pct_income",
            "renter_40_to_49pct_income",
            "renter_50pct_plus_income",
        ]
    ].sum(axis=1, min_count=1)
    acs["owner_mortgage_cost_burden_30pct_plus"] = acs[
        [
            "owner_mortgage_30_to_34pct_income",
            "owner_mortgage_35_to_39pct_income",
            "owner_mortgage_40_to_49pct_income",
            "owner_mortgage_50pct_plus_income",
        ]
    ].sum(axis=1, min_count=1)

    output_path = output_dir / f"census_acs5_county_affordability_{start_year}_{min(end_year, 2024)}.csv"
    acs[
        [
            "year",
            "county_fips",
            "state",
            "county",
            "NAME",
            *numeric_cols,
            "renter_cost_burden_30pct_plus",
            "owner_mortgage_cost_burden_30pct_plus",
        ]
    ].to_csv(output_path, index=False)
    return output_path


def _records_from_response(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "results", "result", "response", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
        if isinstance(value, dict):
            return [value]
    return [payload]


def _county_entity_id(record: dict[str, Any]) -> str | None:
    for key in ("entityId", "entityid", "entity_id", "countyId", "countyid", "id", "geoid", "fips"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def _county_name(record: dict[str, Any]) -> str | None:
    for key in ("entityName", "entityname", "entity_name", "countyName", "countyname", "name", "county"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def download_chas_county(
    output_dir: Path,
    *,
    years: list[str],
    state_fips: list[str],
    request_delay: float = 1.0,
    retries: int = 3,
    retry_delay: float = 5.0,
    checkpoint_every: int = 25,
    resume: bool = True,
    progress: bool = True,
) -> Path:
    token = _first_env_value(HUD_TOKEN_ENV_VARS)
    if not token:
        raise RuntimeError(f"One of {', '.join(HUD_TOKEN_ENV_VARS)} is required for the HUD CHAS API.")

    headers = {"Authorization": f"Bearer {token}"}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    output_path = output_dir / f"hud_chas_county_{years[0].replace('-', '_')}_{years[-1].replace('-', '_')}.csv"
    completed_keys: set[tuple[str, str, str]] = set()
    if resume and output_path.exists() and output_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(output_path, usecols=["chas_year", "state_id", "entity_id"], dtype=str)
            completed_keys = set(existing.drop_duplicates().itertuples(index=False, name=None))
        except (ValueError, pd.errors.EmptyDataError):
            completed_keys = set()
    completed_since_checkpoint = 0

    for state_index, state_id in enumerate(state_fips, start=1):
        state_param = str(int(state_id))
        list_url = f"{HUD_CHAS_BASE_URL}/listCounties/{state_param}"
        try:
            county_records = _records_from_response(
                _request_json(list_url, headers=headers, retries=retries, retry_delay=retry_delay)
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            failures.append({"state_id": state_id, "year": "", "entity_id": "", "error": str(exc)})
            continue

        counties = _dedupe_dicts(
            [
                {
                    "entity_id": entity_id,
                    "county_name": _county_name(record),
                }
                for record in county_records
                if (entity_id := _county_entity_id(record))
            ],
            ["entity_id"],
        )
        if progress:
            print(
                f"CHAS state {state_id} ({state_index}/{len(state_fips)}): "
                f"{len(counties)} counties, {len(years)} year range(s)",
                flush=True,
            )

        for year in years:
            for county in counties:
                county_key = (year, state_id, county["entity_id"])
                if county_key in completed_keys:
                    continue
                query = urllib.parse.urlencode(
                    {
                        "type": "3",
                        "year": year,
                        "stateId": state_param,
                        "entityId": county["entity_id"],
                    }
                )
                url = f"{HUD_CHAS_BASE_URL}?{query}"
                try:
                    records = _records_from_response(
                        _request_json(url, headers=headers, retries=retries, retry_delay=retry_delay)
                    )
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    failures.append(
                        {
                            "state_id": state_id,
                            "year": year,
                            "entity_id": county["entity_id"],
                            "error": str(exc),
                        }
                    )
                    continue

                request_rows: list[dict[str, Any]] = []
                for record in records:
                    row = {
                        "chas_year": year,
                        "state_id": state_id,
                        "entity_id": county["entity_id"],
                        "county_name": county["county_name"],
                    }
                    row.update(record)
                    request_rows.append(row)
                if request_rows:
                    request_rows = _dedupe_dicts(request_rows, list(request_rows[0].keys()))
                    rows.extend(request_rows)
                    completed_keys.add(county_key)
                    completed_since_checkpoint += 1
                if completed_since_checkpoint >= checkpoint_every:
                    _append_rows_csv(output_path, rows)
                    rows.clear()
                    completed_since_checkpoint = 0
                    if progress:
                        print(
                            f"Checkpoint: {len(completed_keys)} county/year rows saved to {output_path.name}",
                            flush=True,
                        )
                time.sleep(request_delay)

    if failures:
        failure_path = output_dir / "hud_chas_county_api_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    if not rows:
        if output_path.exists() and output_path.stat().st_size > 0:
            chas = pd.read_csv(output_path)
            chas.drop_duplicates().to_csv(output_path, index=False)
            return output_path
        raise RuntimeError("No HUD CHAS county data downloaded.")

    _append_rows_csv(output_path, rows)
    chas = pd.read_csv(output_path).drop_duplicates()
    chas.to_csv(output_path, index=False)
    return output_path


def write_chas_note(output_dir: Path, message: str) -> Path:
    path = output_dir / "hud_chas_county_download_note.txt"
    path.write_text(
        "\n".join(
            [
                "HUD CHAS county data coverage needed for the Redfin housing window:",
                ", ".join(CHAS_YEARS),
                "",
                f"Download status: {message}",
                "",
                "Set HUD_USER_TOKEN, or one of the supported HUD token environment variables, and rerun:",
                "python -m housing_climate_risk.cli.download_affordability_data",
                "",
                "Source: https://www.huduser.gov/portal/datasets/cp.html",
                "API docs: https://www.huduser.gov/portal/dataset/chas-api.html",
                "",
                "County summary level: 050",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_acs_note(output_dir: Path, message: str) -> Path:
    path = output_dir / "census_acs5_county_download_note.txt"
    path.write_text(
        "\n".join(
            [
                "Census ACS 5-Year county affordability data coverage needed for the Redfin housing window:",
                "2012 through 2024 ACS 5-year releases. The Redfin file runs through 2025, but 2025 ACS 5-year",
                "county data is not available yet as of this project run.",
                "",
                f"Download status: {message}",
                "",
                "Set CENSUS_API_KEY, or pass --census-api-key, and rerun:",
                "python -m housing_climate_risk.cli.download_affordability_data",
                "",
                "Source: https://api.census.gov/data.html",
            ]
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ACS and HUD CHAS county affordability data aligned with Redfin housing years.")
    parser.add_argument("--output-dir", type=Path, default=AFFORDABILITY_DIR)
    parser.add_argument("--acs-output-dir", type=Path, default=ACS_DIR)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--census-api-key", default=None, help="Optional Census API key override. Defaults to CENSUS_API_KEY.")
    parser.add_argument("--skip-acs", action="store_true")
    parser.add_argument("--skip-chas", action="store_true")
    parser.add_argument("--check-credentials", action="store_true", help="Print whether required API credentials are visible, without downloading.")
    parser.add_argument(
        "--chas-years",
        nargs="+",
        default=CHAS_YEARS,
        help="HUD CHAS year ranges to download. Default: all CHAS releases overlapping the housing window.",
    )
    parser.add_argument(
        "--chas-states",
        nargs="+",
        default=STATE_FIPS,
        help="State FIPS values to request from HUD CHAS. Default: all states and DC.",
    )
    parser.add_argument("--chas-request-delay", type=float, default=1.0)
    parser.add_argument("--chas-retries", type=int, default=3)
    parser.add_argument("--chas-retry-delay", type=float, default=5.0)
    parser.add_argument("--chas-checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true", help="Do not skip CHAS county/year requests already present in the output CSV.")
    parser.add_argument("--quiet", action="store_true", help="Suppress CHAS progress messages.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.acs_output_dir.mkdir(parents=True, exist_ok=True)
    inferred_start, inferred_end = infer_housing_period()
    start_year = args.start_year or inferred_start
    end_year = args.end_year or inferred_end

    if args.check_credentials:
        print(f"Census key: {_masked(args.census_api_key or _first_env_value(CENSUS_TOKEN_ENV_VARS))}")
        print(f"HUD token: {_masked(_first_env_value(HUD_TOKEN_ENV_VARS))}")
        return

    outputs: list[Path] = []
    if not args.skip_acs:
        try:
            outputs.append(download_acs_county(start_year, end_year, args.acs_output_dir, api_key=args.census_api_key))
        except Exception as exc:
            outputs.append(write_acs_note(args.acs_output_dir, str(exc)))
    if not args.skip_chas:
        try:
            outputs.append(
                download_chas_county(
                    args.output_dir,
                    years=args.chas_years,
                    state_fips=args.chas_states,
                    request_delay=args.chas_request_delay,
                    retries=args.chas_retries,
                    retry_delay=args.chas_retry_delay,
                    checkpoint_every=args.chas_checkpoint_every,
                    resume=not args.no_resume,
                    progress=not args.quiet,
                )
            )
        except Exception as exc:
            outputs.append(write_chas_note(args.output_dir, str(exc)))

    _write_source_manifest(
        args.output_dir / "affordability_sources.csv",
        [
            {
                "dataset": "Census ACS 5-Year county affordability indicators",
                "coverage": f"{start_year}-{min(end_year, 2024)}",
                "source_url": "https://api.census.gov/data.html",
                "notes": "County-level ACS 5-year variables for income, rent, home value, owner costs, and selected cost-burden counts.",
            },
            {
                "dataset": "HUD CHAS county data",
                "coverage": ", ".join(args.chas_years),
                "source_url": "https://www.huduser.gov/portal/datasets/cp.html",
                "notes": "County summary level type=3 from the HUD CHAS API.",
            },
        ],
    )

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
