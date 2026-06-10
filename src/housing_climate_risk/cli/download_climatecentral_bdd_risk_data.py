from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from housing_climate_risk.paths import CLIMATE_DIR


BASE_URL = "https://www.climatecentral.org/api/bdd/risk/county"
DEFAULT_OUTPUT_DIR = CLIMATE_DIR / "climatecentral_billion_dollar_disasters_risk"
COUNTY_FIPS_RE = re.compile(r"^\d{5}$")

ENDPOINTS: dict[str, list[str]] = {
    "weather-risk": [
        "all_disasters",
        "drought",
        "flooding",
        "freeze",
        "severe_storm",
        "tropical_cyclone",
        "wildfire",
        "winter_storm",
    ],
    "vulnerability": [
        "poverty",
        "minors",
        "seniors",
        "disabled",
        "single_parent",
        "minorities",
        "limited_english",
        "no_diploma",
        "mobile_homes",
        "no_vehicle",
        "veterans",
    ],
    "future-risk": [
        "agriculture",
        "mortality",
        "energy",
        "labor",
        "coastal_storms",
        "total_damage",
    ],
}


def _request_json(url: str, *, timeout: int, retries: int, retry_delay: float) -> tuple[dict[str, Any], int]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "quoll-intelligence/climatecentral-bdd-risk-downloader",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
            data = json.loads(body.decode("utf-8-sig"))
            if not isinstance(data, dict):
                raise RuntimeError(f"Expected JSON object from {url}, got {type(data).__name__}.")
            return data, len(body)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(retry_delay * (2**attempt))
    raise RuntimeError("HTTP request retry loop exited unexpectedly.")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "variable",
        "source_url",
        "raw_path",
        "records",
        "county_records",
        "bytes_downloaded",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _series_from_endpoint(group: str, variable: str, data: dict[str, Any], *, include_aggregates: bool) -> pd.Series:
    name = f"{group.replace('-', '_')}_{variable}"
    values = pd.Series(data, name=name)
    values.index = values.index.astype(str)
    values.index.name = "county_fips"
    if not include_aggregates:
        values = values[values.index.to_series().str.fullmatch(COUNTY_FIPS_RE)]
    return pd.to_numeric(values, errors="coerce")


def download(
    output_dir: Path,
    *,
    timeout: int,
    retries: int,
    retry_delay: float,
    include_aggregates: bool,
    force: bool,
) -> tuple[Path, Path]:
    raw_dir = output_dir / "raw"
    combined_csv = output_dir / "climatecentral_bdd_risk_county.csv"
    manifest_csv = output_dir / "climatecentral_bdd_risk_manifest.csv"
    if combined_csv.exists() and manifest_csv.exists() and not force:
        print(f"Using existing Climate Central outputs in {output_dir}", flush=True)
        return combined_csv, manifest_csv

    frames: list[pd.Series] = []
    manifest_rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for group, variables in ENDPOINTS.items():
        for variable in variables:
            url = f"{BASE_URL}/{group}/{variable}.json"
            print(f"Downloading {group}/{variable}...", flush=True)
            data, byte_count = _request_json(url, timeout=timeout, retries=retries, retry_delay=retry_delay)

            raw_path = raw_dir / group / f"{variable}.json"
            _write_json(raw_path, data)
            frames.append(_series_from_endpoint(group, variable, data, include_aggregates=include_aggregates))

            county_records = sum(1 for key in data if COUNTY_FIPS_RE.fullmatch(str(key)))
            manifest_rows.append(
                {
                    "group": group,
                    "variable": variable,
                    "source_url": url,
                    "raw_path": str(raw_path.relative_to(output_dir)),
                    "records": len(data),
                    "county_records": county_records,
                    "bytes_downloaded": byte_count,
                }
            )

    combined = pd.concat(frames, axis=1).reset_index()
    combined.to_csv(combined_csv, index=False)
    _write_manifest(manifest_csv, manifest_rows)
    total_bytes = sum(int(row["bytes_downloaded"]) for row in manifest_rows)
    print(
        f"Wrote {combined_csv} with {len(combined):,} rows and {len(combined.columns):,} columns. "
        f"Downloaded {total_bytes / 1024 / 1024:.2f} MB raw JSON.",
        flush=True,
    )
    return combined_csv, manifest_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Climate Central billion-dollar-disaster risk-map county data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--include-aggregates", action="store_true", help="Keep state and national aggregate keys such as 10 and 00.")
    parser.add_argument("--force", action="store_true", help="Redownload even if output files already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(
        args.output_dir,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        include_aggregates=args.include_aggregates,
        force=args.force,
    )


if __name__ == "__main__":
    main()
