from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from housing_climate_risk.paths import DATA_DIR


OUTPUT_DIR = DATA_DIR / "climate_damage"
NOAA_INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
FEMA_API = "https://www.fema.gov/api/open"
FEMA_WEB_DISASTER_SUMMARIES_URL = (
    f"{FEMA_API}/v1/FemaWebDisasterSummaries"
    "?$orderby=disasterNumber&$top=1000"
)
FIPS_MASTER_PATH = DATA_DIR / "fipsgeo" / "fips_master_v2.csv"
MANIFEST_COLUMNS = [
    "artifact",
    "source",
    "year",
    "url",
    "path",
    "bytes",
    "sha256",
    "derived_from",
    "downloaded_at",
]

def _last_complete_calendar_year(today: date | None = None) -> int:
    return (today or date.today()).year - 1


def _default_years() -> tuple[int, int]:
    end_year = _last_complete_calendar_year()
    return end_year - 9, end_year


def _request_bytes(url: str, *, timeout: int = 120, retries: int = 2, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "quoll-intelligence/climate-damage-downloader",
            "Accept": "*/*",
            **(headers or {}),
        },
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if attempt >= retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(90.0, 5.0 * (2**attempt))
            print(f"Retrying after HTTP {exc.code} in {delay:.0f}s: {url}", flush=True)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            delay = min(90.0, 5.0 * (2**attempt))
            print(f"Retrying after network error in {delay:.0f}s: {url}", flush=True)
            time.sleep(delay)
    raise RuntimeError("HTTP request retry loop exited unexpectedly.")


def _request_text(url: str, **kwargs: Any) -> str:
    return _request_bytes(url, **kwargs).decode("utf-8-sig")


def _head_size(url: str, *, timeout: int = 1) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "quoll-intelligence/climate-damage-downloader"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_row(
    *,
    artifact: Path,
    output_dir: Path,
    source: str,
    url: str,
    year: int | str = "",
    derived_from: str = "",
) -> dict[str, Any]:
    return {
        "artifact": artifact.name,
        "source": source,
        "year": year,
        "url": url,
        "path": str(artifact.relative_to(output_dir)),
        "bytes": artifact.stat().st_size,
        "sha256": _sha256(artifact),
        "derived_from": derived_from,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def _download_file(url: str, path: Path, *, force: bool) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path, path.stat().st_size
    body = _request_bytes(url)
    path.write_bytes(body)
    return path, len(body)


def _noaa_files(start_year: int, end_year: int) -> dict[int, tuple[str, int | None]]:
    html = _request_text(NOAA_INDEX_URL, timeout=10, retries=0)
    filenames = sorted(set(re.findall(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz", html)))
    by_year: dict[int, tuple[str, int | None]] = {}
    for year_text in filenames:
        year = int(year_text)
        if start_year <= year <= end_year:
            filename_match = re.search(rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz", html)
            if not filename_match:
                continue
            filename = filename_match.group(0)
            size_match = re.search(rf"{re.escape(filename)}.*?</a>\s+\S+\s+\S+\s+(\d+)", html, flags=re.DOTALL)
            size = int(size_match.group(1)) if size_match else None
            by_year[year] = (filename, size)
    missing = [year for year in range(start_year, end_year + 1) if year not in by_year]
    if missing:
        raise RuntimeError(f"NOAA Storm Events detail files not found for years: {missing}")
    return by_year


def _parse_noaa_damage(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().upper().replace(",", "")
    if not text:
        return 0.0
    multiplier = 1.0
    if text[-1:] == "K":
        multiplier = 1_000.0
        text = text[:-1]
    elif text[-1:] == "M":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text[-1:] == "B":
        multiplier = 1_000_000_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return 0.0


def _process_noaa(paths: list[Path], output_path: Path) -> Path:
    frames: list[pd.DataFrame] = []
    keep = [
        "BEGIN_YEARMONTH",
        "BEGIN_DATE_TIME",
        "END_DATE_TIME",
        "EPISODE_ID",
        "EVENT_ID",
        "STATE",
        "STATE_FIPS",
        "YEAR",
        "CZ_TYPE",
        "CZ_FIPS",
        "CZ_NAME",
        "EVENT_TYPE",
        "SOURCE",
        "MAGNITUDE",
        "TOR_F_SCALE",
        "DEATHS_DIRECT",
        "INJURIES_DIRECT",
        "DAMAGE_PROPERTY",
        "DAMAGE_CROPS",
    ]
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as file:
            frame = pd.read_csv(file, usecols=lambda col: col in keep, low_memory=False)
        frame.columns = [col.lower() for col in frame.columns]
        frame["state_fips"] = pd.to_numeric(frame["state_fips"], errors="coerce").astype("Int64")
        frame["cz_fips"] = pd.to_numeric(frame["cz_fips"], errors="coerce").astype("Int64")
        frame["county_fips"] = frame["state_fips"].astype(str).str.zfill(2) + frame["cz_fips"].astype(str).str.zfill(3)
        frame.loc[~frame["cz_type"].eq("C"), "county_fips"] = pd.NA
        frame["property_damage"] = frame["damage_property"].map(_parse_noaa_damage)
        frame["crop_damage"] = frame["damage_crops"].map(_parse_noaa_damage)
        frame["total_damage"] = frame["property_damage"] + frame["crop_damage"]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


def _normalize_county_name(value: Any) -> str:
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    suffixes = (
        " CITY AND BOROUGH",
        " CENSUS AREA",
        " MUNICIPALITY",
        " BOROUGH",
        " PARISH",
        " COUNTY",
        " CITY",
    )
    for suffix in suffixes:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    return re.sub(r"\s+", " ", text)


def _zone_mapping_rows(noaa: pd.DataFrame, counties: pd.DataFrame) -> list[dict[str, Any]]:
    zones = noaa.loc[noaa["cz_type"].eq("Z")].copy()
    zones["total_damage"] = pd.to_numeric(zones["total_damage"], errors="coerce").fillna(0)
    summaries = (
        zones.groupby(["state", "state_fips", "cz_fips", "cz_name"], dropna=False)
        .agg(
            source_row_count=("cz_name", "size"),
            source_total_damage=("total_damage", "sum"),
            max_row_damage=("total_damage", "max"),
        )
        .reset_index()
    )
    county_lookup: dict[str, list[tuple[str, str, str]]] = {}
    for county in counties.itertuples(index=False):
        fips = str(county.fips).zfill(5)
        county_lookup.setdefault(fips[:2], []).append(
            (fips, str(county.county_name), _normalize_county_name(county.county_name))
        )

    prefixes = re.compile(
        r"^(?:NORTH|SOUTH|EAST|WEST|CENTRAL|NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|"
        r"COASTAL|INTERIOR|UPPER|LOWER|EASTERN|WESTERN|NORTHERN|SOUTHERN)\s+"
    )
    rows: list[dict[str, Any]] = []
    for zone in summaries.itertuples(index=False):
        state_fips = str(zone.state_fips).zfill(2)
        zone_name = _normalize_county_name(zone.cz_name)
        candidates = county_lookup.get(state_fips, [])
        matches: list[tuple[str, str, str, str]] = []

        exact = [item for item in candidates if item[2] == zone_name]
        if exact:
            matches = [(fips, name, "exact_county_name", "high") for fips, name, _ in exact]
        else:
            stripped = prefixes.sub("", zone_name)
            directional = [item for item in candidates if item[2] == stripped]
            if directional:
                matches = [
                    (fips, name, "directional_or_coastal_prefix_stripped", "medium")
                    for fips, name, _ in directional
                ]
            else:
                contained = [
                    item
                    for item in candidates
                    if len(item[2]) >= 4
                    and re.search(rf"(?:^|\s){re.escape(item[2])}(?:$|\s)", zone_name)
                ]
                if len(contained) == 1:
                    matches = [
                        (contained[0][0], contained[0][1], "county_name_contained_in_zone", "low")
                    ]
                elif len(contained) > 1 and "COUNT" in str(zone.cz_name).upper():
                    matches = [
                        (fips, name, "multi_county_phrase", "medium")
                        for fips, name, _ in contained
                    ]

        common = {
            "state": zone.state,
            "state_fips": state_fips,
            "cz_fips": zone.cz_fips,
            "cz_name": zone.cz_name,
            "source_row_count": zone.source_row_count,
            "source_total_damage": zone.source_total_damage,
            "max_row_damage": zone.max_row_damage,
        }
        if matches:
            for mapped_fips, mapped_name, method, confidence in matches:
                rows.append(
                    {
                        **common,
                        "mapped_fips": mapped_fips,
                        "mapped_county_name": mapped_name,
                        "mapping_method": method,
                        "mapping_confidence": confidence,
                        "mapping_note": "Deterministic normalized-name match generated from fips_master_v2.csv.",
                    }
                )
        else:
            rows.append(
                {
                    **common,
                    "mapped_fips": "",
                    "mapped_county_name": "",
                    "mapping_method": "unmapped",
                    "mapping_confidence": "none",
                    "mapping_note": "No deterministic county-name match in the same state.",
                }
            )
    return rows


def _process_zone_county_mapping(
    noaa_path: Path,
    county_path: Path,
    output_path: Path,
) -> Path:
    noaa = pd.read_csv(
        noaa_path,
        usecols=["state", "state_fips", "cz_type", "cz_fips", "cz_name", "total_damage"],
        dtype={"state_fips": "string"},
        low_memory=False,
    )
    counties = pd.read_csv(county_path, usecols=["fips", "county_name"], dtype=str)
    rows = _zone_mapping_rows(noaa, counties)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _fema_get(endpoint: str, params: dict[str, str], *, timeout: int = 180, retries: int = 8) -> dict[str, Any]:
    url = f"{FEMA_API}/{endpoint}?{urllib.parse.urlencode(params, safe=',() ')}"
    return json.loads(_request_text(url, timeout=timeout, retries=retries))


def _fema_fetch_all(endpoint: str, root_key: str, params: dict[str, str], *, page_size: int = 5000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    skip = 0
    while True:
        payload = _fema_get(endpoint, {**params, "$top": str(page_size), "$skip": str(skip)})
        page = payload.get(root_key, [])
        if not page:
            break
        rows.extend(page)
        if len(rows) % 50_000 == 0:
            print(f"Fetched {len(rows):,} rows from {root_key}...", flush=True)
        if len(page) < page_size:
            break
        skip += page_size
    return pd.DataFrame(rows)


def _download_fema_web_disaster_summaries(output_dir: Path) -> Path:
    frame = _fema_fetch_all(
        "v1/FemaWebDisasterSummaries",
        "FemaWebDisasterSummaries",
        {"$orderby": "disasterNumber"},
        page_size=1000,
    )
    path = output_dir / "raw" / "fema_web_disaster_summaries" / "FemaWebDisasterSummaries.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def estimate(start_year: int, end_year: int, output_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    noaa = _noaa_files(start_year, end_year)
    for year, (filename, size) in noaa.items():
        url = urllib.parse.urljoin(NOAA_INDEX_URL, filename)
        rows.append(
            {
                "artifact": filename,
                "source": "noaa_storm_events",
                "year": year,
                "url": url,
                "bytes": size,
            }
        )
    rows.append(
        {
            "artifact": "FemaWebDisasterSummaries.csv",
            "source": "openfema",
            "url": FEMA_WEB_DISASTER_SUMMARIES_URL,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "climate_damage_download_estimate.csv"
    _write_manifest(path, rows)
    return path


def download(start_year: int, end_year: int, output_dir: Path, *, force: bool) -> None:
    raw_dir = output_dir / "raw"
    manifest_rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    noaa_paths = []
    for year, (filename, _) in _noaa_files(start_year, end_year).items():
        url = urllib.parse.urljoin(NOAA_INDEX_URL, filename)
        path, size = _download_file(url, raw_dir / "noaa_storm_events" / filename, force=force)
        noaa_paths.append(path)
        manifest_rows.append(
            _manifest_row(
                artifact=path,
                output_dir=output_dir,
                source="noaa_storm_events",
                year=year,
                url=url,
            )
        )

    noaa_output = _process_noaa(noaa_paths, output_dir / "noaa_storm_events_county_damage.csv")
    noaa_urls = ";".join(row["url"] for row in manifest_rows)
    manifest_rows.append(
        _manifest_row(
            artifact=noaa_output,
            output_dir=output_dir,
            source="noaa_storm_events",
            year=f"{start_year}-{end_year}",
            url=noaa_urls,
            derived_from=";".join(str(path.relative_to(output_dir)) for path in noaa_paths),
        )
    )

    zone_output = _process_zone_county_mapping(
        noaa_output,
        FIPS_MASTER_PATH,
        output_dir / "noaa_storm_events_zone_county_mapping.csv",
    )
    manifest_rows.append(
        _manifest_row(
            artifact=zone_output,
            output_dir=output_dir,
            source="generated_noaa_zone_county_mapping",
            year=f"{start_year}-{end_year}",
            url=noaa_urls,
            derived_from=(
                f"{noaa_output.relative_to(output_dir)};"
                f"{FIPS_MASTER_PATH.relative_to(DATA_DIR)}"
            ),
        )
    )

    fema_output = (
        _download_fema_web_disaster_summaries(output_dir)
        if force
        or not (
            output_dir
            / "raw"
            / "fema_web_disaster_summaries"
            / "FemaWebDisasterSummaries.csv"
        ).exists()
        else output_dir
        / "raw"
        / "fema_web_disaster_summaries"
        / "FemaWebDisasterSummaries.csv"
    )
    manifest_rows.append(
        _manifest_row(
            artifact=fema_output,
            output_dir=output_dir,
            source="openfema",
            url=FEMA_WEB_DISASTER_SUMMARIES_URL,
        )
    )
    _write_manifest(output_dir / "climate_damage_source_manifest.csv", manifest_rows)


def write_existing_manifest(output_dir: Path) -> Path:
    """Backfill lineage for existing processed artifacts without downloading data."""
    estimate_path = output_dir / "climate_damage_download_estimate.csv"
    noaa_urls: list[str] = []
    rows: list[dict[str, Any]] = []
    raw_noaa_paths = sorted((output_dir / "raw" / "noaa_storm_events").glob("*.csv.gz"))
    for artifact in raw_noaa_paths:
        match = re.search(r"_d(\d{4})_c\d+\.csv\.gz$", artifact.name)
        url = urllib.parse.urljoin(NOAA_INDEX_URL, artifact.name)
        noaa_urls.append(url)
        rows.append(
            _manifest_row(
                artifact=artifact,
                output_dir=output_dir,
                source="noaa_storm_events",
                year=int(match.group(1)) if match else "",
                url=url,
            )
        )
    if estimate_path.exists():
        with estimate_path.open(encoding="utf-8-sig", newline="") as file:
            noaa_urls.extend(
                row["url"]
                for row in csv.DictReader(file)
                if row.get("source") == "noaa_storm_events" and row.get("url")
            )
    upstream_noaa = ";".join(dict.fromkeys(noaa_urls)) or NOAA_INDEX_URL
    candidates = [
        (
            output_dir / "noaa_storm_events_county_damage.csv",
            "noaa_storm_events",
            upstream_noaa,
            "NOAA Storm Events annual detail CSV archives",
        ),
        (
            output_dir / "noaa_storm_events_zone_county_mapping.csv",
            "generated_noaa_zone_county_mapping",
            upstream_noaa,
            "noaa_storm_events_county_damage.csv;fipsgeo/fips_master_v2.csv",
        ),
        (
            output_dir
            / "raw"
            / "fema_web_disaster_summaries"
            / "FemaWebDisasterSummaries.csv",
            "openfema",
            FEMA_WEB_DISASTER_SUMMARIES_URL,
            "",
        ),
    ]
    rows.extend(
        [
            _manifest_row(
                artifact=artifact,
                output_dir=output_dir,
                source=source,
                url=url,
                derived_from=derived_from,
            )
            for artifact, source, url, derived_from in candidates
            if artifact.exists()
        ]
    )
    path = output_dir / "climate_damage_source_manifest.csv"
    _write_manifest(path, rows)
    return path


def parse_args() -> argparse.Namespace:
    start_year, end_year = _default_years()
    parser = argparse.ArgumentParser(
        description="Download NOAA Storm Events damage data and FEMA disaster financial summaries."
    )
    parser.add_argument("--start-year", type=int, default=start_year)
    parser.add_argument("--end-year", type=int, default=end_year)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write lineage and hashes for existing processed artifacts without downloading.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.estimate_only:
        path = estimate(args.start_year, args.end_year, args.output_dir)
        print(f"Wrote estimate to {path}")
        return
    if args.manifest_only:
        path = write_existing_manifest(args.output_dir)
        print(f"Wrote manifest to {path}")
        return
    download(args.start_year, args.end_year, args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
