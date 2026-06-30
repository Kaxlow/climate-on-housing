from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from housing_climate_risk.paths import CLIMATE_DIR, FIPSGEO_DIR


OUTPUT_PATH = CLIMATE_DIR / "ncei_climate_at_a_glance_county_monthly.csv"
CLIMDIV_INDEX_URL = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"

PARAMETERS = {
    "tavg": ("tmpccy", "average_temperature", "deg_f"),
    "tmin": ("tmincy", "minimum_temperature", "deg_f"),
    "tmax": ("tmaxcy", "maximum_temperature", "deg_f"),
    "pcp": ("pcpncy", "precipitation", "inch"),
}

FIELDNAMES = [
    "fips",
    "parameter",
    "parameter_label",
    "unit",
    "year_month",
    "date",
    "year",
    "month",
    "value",
    "anomaly",
    "rank",
    "source_url",
    "fetched_at",
]


def _default_years(today: date | None = None) -> tuple[int, int]:
    end_year = (today or date.today()).year - 1
    return end_year - 9, end_year


def _request_text(url: str, *, timeout: int = 180, retries: int = 4) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "quoll-intelligence/ncei-county-weather-downloader",
            "Accept": "text/plain,text/html,*/*",
        },
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if attempt >= retries or exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"Failed to download {url}: HTTP {exc.code}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 3.0 * (2**attempt))
            print(f"Retrying after HTTP {exc.code} in {delay:.0f}s: {url}", flush=True)
            time.sleep(delay)
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            delay = min(60.0, 3.0 * (2**attempt))
            print(f"Retrying after network error in {delay:.0f}s: {url}", flush=True)
            time.sleep(delay)
    raise RuntimeError("HTTP request retry loop exited unexpectedly.")


def _county_fips_from_file(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"fips": "string"})
    frame["fips"] = frame["fips"].dropna().astype(str).str.zfill(5)
    county_name = frame.get("county_name")
    valid_county = frame["fips"].str.match(r"^\d{5}$") & ~frame["fips"].str.endswith("000")
    if county_name is not None:
        valid_county &= county_name.notna() & county_name.astype(str).str.strip().ne("")
    return sorted(frame.loc[valid_county, "fips"].unique().tolist())


def _normalize_county_fips(values: list[str]) -> list[str]:
    fips = sorted({str(value).zfill(5) for value in values})
    invalid = [value for value in fips if not value.isdigit() or len(value) != 5 or value.endswith("000")]
    if invalid:
        raise ValueError(f"NCEI county weather requires 5-digit county FIPS codes, not aggregate/state FIPS: {invalid}")
    return fips


def _latest_climdiv_urls(parameters: list[str]) -> dict[str, str]:
    index_html = _request_text(CLIMDIV_INDEX_URL, timeout=60, retries=2)
    urls: dict[str, str] = {}
    for parameter in parameters:
        file_code = PARAMETERS[parameter][0]
        filenames = sorted(set(re.findall(rf"climdiv-{file_code}-v1\.0\.0-\d+", index_html)))
        if not filenames:
            raise RuntimeError(f"Could not find NCEI climdiv county file for parameter {parameter} ({file_code}).")
        urls[parameter] = CLIMDIV_INDEX_URL + filenames[-1]
    return urls


def _records_from_climdiv_text(
    *,
    text: str,
    parameter: str,
    source_url: str,
    fetched_at: str,
    start_year: int,
    end_year: int,
    selected_counties: set[str],
    completed_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    _, parameter_label, unit = PARAMETERS[parameter]
    rows = []
    for raw_line in text.splitlines():
        parts = raw_line.split()
        if len(parts) < 13:
            continue
        key = parts[0]
        if len(key) < 9 or not key.isdigit():
            continue
        fips = key[:5]
        year = int(key[-4:])
        if year < start_year or year > end_year or fips not in selected_counties:
            continue
        for month, raw_value in enumerate(parts[1:13], start=1):
            year_month = f"{year}{month:02d}"
            record_key = (fips, parameter, year_month)
            if record_key in completed_keys:
                continue
            rows.append(
                {
                    "fips": fips,
                    "parameter": parameter,
                    "parameter_label": parameter_label,
                    "unit": unit,
                    "year_month": year_month,
                    "date": f"{year:04d}-{month:02d}-01",
                    "year": year,
                    "month": month,
                    "value": _number_or_none(raw_value),
                    "anomaly": None,
                    "rank": None,
                    "source_url": source_url,
                    "fetched_at": fetched_at,
                }
            )
    return rows


def _existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for chunk in pd.read_csv(path, dtype=str, usecols=["fips", "parameter", "year_month"], chunksize=250_000):
        for row in chunk.dropna(subset=["fips", "parameter", "year_month"]).itertuples(index=False):
            keys.add((str(row.fips).zfill(5), str(row.parameter), str(row.year_month)))
    return keys


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "Missing", "NaN", "nan", "null", "-99.99"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def download_county_weather(
    *,
    output_path: Path = OUTPUT_PATH,
    county_fips: list[str] | None = None,
    parameters: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    sleep_seconds: float = 0.08,
    force: bool = False,
    limit_counties: int | None = None,
) -> Path:
    default_start, default_end = _default_years()
    start_year = start_year or default_start
    end_year = end_year or default_end
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    selected_parameters = parameters or list(PARAMETERS)
    unknown = sorted(set(selected_parameters) - set(PARAMETERS))
    if unknown:
        raise ValueError(f"Unsupported NCEI parameters: {unknown}. Expected one of {sorted(PARAMETERS)}")

    selected_counties = (
        _normalize_county_fips(county_fips)
        if county_fips
        else _county_fips_from_file(FIPSGEO_DIR / "fips_master_v2.csv")
    )
    if limit_counties is not None:
        selected_counties = selected_counties[:limit_counties]
    selected_county_set = set(selected_counties)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if force and output_path.exists():
        output_path.unlink()
    completed_keys = _existing_keys(output_path)
    write_header = not output_path.exists()

    row_count = 0
    source_urls = _latest_climdiv_urls(selected_parameters)
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for parameter in selected_parameters:
            expected_keys = {
                (fips, parameter, f"{year}{month:02d}")
                for fips in selected_counties
                for year in range(start_year, end_year + 1)
                for month in range(1, 13)
            }
            if expected_keys.issubset(completed_keys):
                continue

            url = source_urls[parameter]
            fetched_at = datetime.now(timezone.utc).isoformat()
            text = _request_text(url)
            rows = _records_from_climdiv_text(
                text=text,
                parameter=parameter,
                source_url=url,
                fetched_at=fetched_at,
                start_year=start_year,
                end_year=end_year,
                selected_counties=selected_county_set,
                completed_keys=completed_keys,
            )
            writer.writerows(rows)
            row_count += len(rows)
            completed_keys.update((row["fips"], row["parameter"], row["year_month"]) for row in rows)
            print(f"Loaded {len(rows):,} {parameter} rows from {url}", flush=True)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print(f"Wrote {row_count:,} new NCEI county weather rows to {output_path}", flush=True)
    return output_path


def main() -> None:
    default_start, default_end = _default_years()
    parser = argparse.ArgumentParser(description="Download NOAA NCEI Climate at a Glance county monthly weather data.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument("--start-year", type=int, default=default_start, help="First calendar year to download.")
    parser.add_argument("--end-year", type=int, default=default_end, help="Last calendar year to download.")
    parser.add_argument("--county-fips", nargs="*", help="Optional 5-digit county FIPS values. Defaults to all repo counties.")
    parser.add_argument("--parameters", nargs="+", default=list(PARAMETERS), help="NCEI CAG parameters to download.")
    parser.add_argument("--sleep-seconds", type=float, default=0.08, help="Delay between NCEI requests.")
    parser.add_argument("--limit-counties", type=int, help="Limit counties for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Delete the existing output and redownload from scratch.")
    args = parser.parse_args()

    download_county_weather(
        output_path=args.output,
        county_fips=args.county_fips,
        parameters=args.parameters,
        start_year=args.start_year,
        end_year=args.end_year,
        sleep_seconds=args.sleep_seconds,
        force=args.force,
        limit_counties=args.limit_counties,
    )


if __name__ == "__main__":
    main()
