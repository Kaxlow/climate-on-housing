from __future__ import annotations

import argparse
import csv
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml

from housing_climate_risk.paths import DATA_DIR, FIPSGEO_DIR


USER_AGENT = "quoll-intelligence/federal-data-downloader"
FEMA_API = "https://www.fema.gov/api/open"
FEMA_NRI_SOURCE_PAGE = "https://hazards.fema.gov/nri/data-resources"
FEMA_NRI_CANDIDATES = (
    "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/Archive/v120_0/NRI_Table_Counties.zip",
    "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/NRI_Table_Counties.zip",
    "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/NRI_Table_Counties/NRI_Table_Counties.zip",
)
CENSUS_SOURCE_PAGE = (
    "https://www.census.gov/geographies/mapping-files/time-series/geo/"
    "cartographic-boundary.html"
)
RECEIPT_PATH = DATA_DIR / "download_receipt.yaml"

FEMA_NRI_REQUIRED_COLUMNS = {
    "STATE",
    "STATEABBRV",
    "STATEFIPS",
    "COUNTY",
    "COUNTYFIPS",
    "STCOFIPS",
    "RISK_SCORE",
    "RISK_RATNG",
    "NRI_VER",
}
FEMA_DECLARATION_REQUIRED_COLUMNS = {
    "femaDeclarationString",
    "disasterNumber",
    "state",
    "declarationType",
    "declarationDate",
    "incidentType",
    "incidentBeginDate",
    "fipsStateCode",
    "fipsCountyCode",
    "designatedArea",
    "lastRefresh",
}
CENSUS_BOUNDARY_REQUIRED_COLUMNS = {"STATEFP", "GEOID", "NAME", "geometry"}


def _request_bytes(url: str, *, timeout: int = 300) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.geturl()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _request_json(url: str, *, timeout: int = 300) -> tuple[dict[str, Any], str]:
    body, resolved_url = _request_bytes(url, timeout=timeout)
    return json.loads(body.decode("utf-8-sig")), resolved_url


def _csv_columns(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return set(next(csv.reader(file)))


def _validate_columns(path: Path, expected: set[str]) -> None:
    observed = _csv_columns(path)
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")


def _write_receipt_entry(
    key: str, values: dict[str, Any], path: Path | None = None
) -> None:
    path = path or RECEIPT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {}
    if path.exists():
        receipt = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    receipt.setdefault("downloads", {})[key] = {
        **values,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(yaml.safe_dump(receipt, sort_keys=False), encoding="utf-8")


def _safe_extract_csv(zip_bytes: bytes, output_dir: Path, preferred_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).name.lower().endswith(".csv")
        ]
        preferred = [name for name in members if Path(name).name.lower() == preferred_name.lower()]
        if not preferred:
            raise RuntimeError(
                f"Downloaded ZIP does not contain the expected {preferred_name}; "
                f"found {[Path(name).name for name in members]}"
            )
        source_name = preferred[0]
        destination = output_dir / preferred_name
        destination.write_bytes(archive.read(source_name))
        return destination


def _discover_nri_county_url() -> str:
    try:
        page, _ = _request_bytes(FEMA_NRI_SOURCE_PAGE, timeout=60)
        html = page.decode("utf-8", errors="replace")
        links = [
            urllib.parse.urljoin(FEMA_NRI_SOURCE_PAGE, value)
            for value in re.findall(r"""href=["']([^"']+)["']""", html, flags=re.IGNORECASE)
            if "count" in value.lower() and value.lower().endswith(".zip")
        ]
        current = [url for url in links if "/archive/" not in url.lower()]
        if current:
            return current[0]
        if links:
            return links[0]
    except RuntimeError:
        pass

    errors: list[str] = []
    for candidate in FEMA_NRI_CANDIDATES:
        try:
            _, resolved = _request_bytes(candidate, timeout=120)
            return resolved
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError(
        "Could not discover the current FEMA NRI county ZIP. "
        f"Checked {FEMA_NRI_SOURCE_PAGE} and known endpoints. {'; '.join(errors)}"
    )


def download_fema_nri_counties(*, force: bool = False) -> Path:
    destination = DATA_DIR / "fema" / "NRI_Table_Counties.csv"
    if destination.exists() and not force:
        _validate_columns(destination, FEMA_NRI_REQUIRED_COLUMNS)
        return destination

    requested_url = _discover_nri_county_url()
    body, resolved_url = _request_bytes(requested_url)
    destination = _safe_extract_csv(body, destination.parent, destination.name)
    _validate_columns(destination, FEMA_NRI_REQUIRED_COLUMNS)
    versions = (
        pd.read_csv(destination, usecols=["NRI_VER"], dtype=str)["NRI_VER"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    _write_receipt_entry(
        "fema_nri_counties",
        {
            "source_url": resolved_url,
            "destination": str(destination.relative_to(DATA_DIR)),
            "version": ", ".join(versions) if versions else "not reported",
            "version_specific_url": "/Archive/" in resolved_url,
            "future_results_may_differ": "/Archive/" not in resolved_url,
        },
    )
    return destination


def download_fema_disaster_declarations(*, force: bool = False) -> Path:
    destination = DATA_DIR / "fema" / "FEMA_Disaster_Declarations.csv"
    if destination.exists() and not force:
        _validate_columns(destination, FEMA_DECLARATION_REQUIRED_COLUMNS)
        return destination

    rows: list[dict[str, Any]] = []
    page_size = 5000
    skip = 0
    base_url = f"{FEMA_API}/v2/DisasterDeclarationsSummaries"
    last_metadata: dict[str, Any] = {}
    while True:
        query = urllib.parse.urlencode(
            {"$orderby": "disasterNumber", "$top": page_size, "$skip": skip}
        )
        payload, _ = _request_json(f"{base_url}?{query}")
        page = payload.get("DisasterDeclarationsSummaries", [])
        last_metadata = payload.get("metadata", {})
        if not page:
            break
        rows.extend(page)
        print(f"Fetched {len(rows):,} FEMA disaster declaration rows...", flush=True)
        if len(page) < page_size:
            break
        skip += page_size

    if not rows:
        raise RuntimeError("OpenFEMA returned no disaster declarations.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    _validate_columns(destination, FEMA_DECLARATION_REQUIRED_COLUMNS)
    _write_receipt_entry(
        "fema_disaster_declarations",
        {
            "source_url": base_url,
            "destination": str(destination.relative_to(DATA_DIR)),
            "api_version": "v2",
            "provider_rundate": last_metadata.get("rundate"),
            "row_count": len(rows),
            "version_specific_url": False,
            "future_results_may_differ": True,
        },
    )
    return destination


def _latest_census_boundary_year() -> int:
    current_year = datetime.now(timezone.utc).year
    for year in range(current_year, 2019, -1):
        url = (
            f"https://www2.census.gov/geo/tiger/GENZ{year}/shp/"
            f"cb_{year}_us_county_20m.zip"
        )
        try:
            _request_bytes(url, timeout=60)
            return year
        except RuntimeError:
            continue
    raise RuntimeError("Could not find a Census cartographic boundary release from 2020 onward.")


def _extract_shapefile(zip_bytes: bytes, destination_dir: Path, stable_stem: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        shp_members = [name for name in members if name.lower().endswith(".shp")]
        if len(shp_members) != 1:
            raise RuntimeError(f"Expected one shapefile in Census ZIP; found {shp_members}")
        source_stem = str(Path(shp_members[0]).with_suffix("")).replace("\\", "/")
        for member in members:
            normalized = member.replace("\\", "/")
            if not normalized.startswith(source_stem + "."):
                continue
            suffix = Path(normalized).suffix
            (destination_dir / f"{stable_stem}{suffix}").write_bytes(archive.read(member))
    shapefile = destination_dir / f"{stable_stem}.shp"
    if not shapefile.exists():
        raise RuntimeError(f"Failed to extract Census shapefile to {shapefile}")
    return shapefile


def download_census_boundaries(*, force: bool = False) -> tuple[Path, Path, Path]:
    county_dir = FIPSGEO_DIR / "census_county_boundaries"
    state_dir = FIPSGEO_DIR / "census_state_boundaries"
    county_shp = county_dir / "census_county_boundaries.shp"
    state_shp = state_dir / "census_state_boundaries.shp"
    county_geojson = FIPSGEO_DIR / "us_counties_boundaries_shapefile.json"
    if all(path.exists() for path in (county_shp, state_shp, county_geojson)) and not force:
        for path in (county_shp, state_shp):
            missing = CENSUS_BOUNDARY_REQUIRED_COLUMNS - set(gpd.read_file(path, rows=1).columns)
            if missing:
                raise RuntimeError(f"{path} is missing required fields: {sorted(missing)}")
        return county_shp, state_shp, county_geojson

    year = _latest_census_boundary_year()
    urls = {
        "county": (
            f"https://www2.census.gov/geo/tiger/GENZ{year}/shp/"
            f"cb_{year}_us_county_20m.zip"
        ),
        "state": (
            f"https://www2.census.gov/geo/tiger/GENZ{year}/shp/"
            f"cb_{year}_us_state_20m.zip"
        ),
    }
    county_body, county_url = _request_bytes(urls["county"])
    state_body, state_url = _request_bytes(urls["state"])
    county_shp = _extract_shapefile(county_body, county_dir, "census_county_boundaries")
    state_shp = _extract_shapefile(state_body, state_dir, "census_state_boundaries")

    counties = gpd.read_file(county_shp).to_crs("EPSG:4326")
    counties["id"] = counties["GEOID"].astype(str).str.zfill(5)
    county_geojson.parent.mkdir(parents=True, exist_ok=True)
    counties.to_file(county_geojson, driver="GeoJSON")
    _write_receipt_entry(
        "census_cartographic_boundaries",
        {
            "source_urls": [county_url, state_url],
            "destination": [
                str(county_shp.relative_to(DATA_DIR)),
                str(state_shp.relative_to(DATA_DIR)),
                str(county_geojson.relative_to(DATA_DIR)),
            ],
            "vintage": year,
            "resolution": "1:20,000,000",
            "version_specific_url": True,
            "future_results_may_differ": False,
        },
    )
    return county_shp, state_shp, county_geojson


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FEMA and Census federal source data.")
    parser.add_argument("source", choices=["fema-nri", "fema-declarations", "census-boundaries"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.source == "fema-nri":
        download_fema_nri_counties(force=args.force)
    elif args.source == "fema-declarations":
        download_fema_disaster_declarations(force=args.force)
    else:
        download_census_boundaries(force=args.force)


if __name__ == "__main__":
    main()
