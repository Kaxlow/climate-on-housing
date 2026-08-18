from __future__ import annotations

import argparse
import csv
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from housing_climate_risk.cli.federal_data import _write_receipt_entry
from housing_climate_risk.paths import DATA_DIR


DOWNLOAD_HUB = "https://www.redfin.com/news/data-center/downloads/"
METHODOLOGY_URL = "https://www.redfin.com/news/data-center/methodology/"
S3_BASE_URL = (
    "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_data_center"
)
RECEIPT_PATH = DATA_DIR / "download_receipt.yaml"
USER_AGENT = "quoll-intelligence/redfin-data-center-downloader"

# The prior county extract covered this exact interval. The raw provider files
# are retained as downloaded; build_database applies this window to the mart.
ANALYSIS_PERIOD_START = "2012-01-01"
ANALYSIS_PERIOD_END = "2025-12-31"

REDFIN_COUNTY_FILES = {
    "redfin_housing_market_monthly_counties": {
        "url": f"{S3_BASE_URL}/housing_market/monthly/all_counties.csv",
        "path": DATA_DIR
        / "housing"
        / "redfin_data_center"
        / "housing_market_monthly_counties.csv",
        "columns": {
            "FREQUENCY",
            "PERIOD BEGIN",
            "PERIOD END",
            "REGION TYPE",
            "REGION NAME",
            "HOMES SOLD",
            "MEDIAN SALE PRICE NSA ($)",
            "MEDIAN SALE PRICE PER SQ.FT. ($)",
            "MEDIAN SALE PRICE PER SQ.FT. YOY (%)",
            "AVERAGE SALE TO LIST RATIO (%)",
            "INVENTORY",
        },
    },
    "redfin_property_types_monthly_counties": {
        "url": f"{S3_BASE_URL}/property_types/monthly/all_counties.csv",
        "path": DATA_DIR
        / "housing"
        / "redfin_data_center"
        / "property_types_monthly_counties.csv",
        "columns": {
            "FREQUENCY",
            "PERIOD BEGIN",
            "PERIOD END",
            "REGION TYPE",
            "REGION NAME",
            "PROPERTY TYPE",
            "MEDIAN SALE PRICE PER SQ.FT. ($)",
            "MEDIAN SALE PRICE PER SQ.FT. YOY (%)",
        },
    },
    "redfin_price_drops_monthly_counties": {
        "url": f"{S3_BASE_URL}/price_drops/monthly/all_counties.csv",
        "path": DATA_DIR
        / "housing"
        / "redfin_data_center"
        / "price_drops_monthly_counties.csv",
        "columns": {
            "FREQUENCY",
            "PERIOD BEGIN",
            "PERIOD END",
            "REGION TYPE",
            "REGION NAME",
            "PERCENT ACTIVE WITH PRICE DROPS (%)",
            "PERCENT ACTIVE WITH PRICE DROPS YOY (PPTS)",
        },
    },
}


def _columns(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return set(next(csv.reader(file)))


def _validate(path: Path, expected: set[str]) -> None:
    missing = sorted(expected - _columns(path))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")


def _receipt_has_download(key: str) -> bool:
    if not RECEIPT_PATH.exists():
        return False
    receipt = yaml.safe_load(RECEIPT_PATH.read_text(encoding="utf-8")) or {}
    return key in receipt.get("downloads", {})


def _open_url(url: str, *, timeout: int = 900):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"},
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _download_csv(
    url: str, destination: Path, expected: set[str], *, force: bool
) -> tuple[Path, str, str | None, str | None, bool]:
    if destination.exists() and not force:
        _validate(destination, expected)
        return destination, url, None, None, False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with _open_url(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            resolved_url = response.geturl()
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
        _validate(temporary, expected)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, resolved_url, etag, last_modified, True


def download_redfin_county_monthly(*, force: bool = False) -> list[Path]:
    outputs: list[Path] = []
    for key, spec in REDFIN_COUNTY_FILES.items():
        destination, resolved_url, etag, last_modified, downloaded = _download_csv(
            str(spec["url"]),
            Path(spec["path"]),
            set(spec["columns"]),
            force=force,
        )
        outputs.append(destination)
        if downloaded or not _receipt_has_download(key):
            _write_receipt_entry(
                key,
                {
                    "source_url": resolved_url,
                    "source_page": DOWNLOAD_HUB,
                    "methodology_url": METHODOLOGY_URL,
                    "destination": str(destination.relative_to(DATA_DIR)),
                    "provider_etag": etag,
                    "provider_last_modified": last_modified,
                    "version_specific_url": False,
                    "future_results_may_differ": True,
                    "analysis_period_start": ANALYSIS_PERIOD_START,
                    "analysis_period_end": ANALYSIS_PERIOD_END,
                    "attribution": "Data provided by Redfin, a national real estate brokerage.",
                },
                RECEIPT_PATH,
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Redfin Data Center monthly county housing files."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in download_redfin_county_monthly(force=args.force):
        print(path)


if __name__ == "__main__":
    main()
