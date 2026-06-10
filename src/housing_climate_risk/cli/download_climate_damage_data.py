from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from housing_climate_risk.paths import DATA_DIR


OUTPUT_DIR = DATA_DIR / "climate_damage"
NOAA_INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
FEMA_API = "https://www.fema.gov/api/open"
USDA_RMA_COL_PAGE_URL = "https://www.rma.usda.gov/tools-reports/summary-business/cause-loss/{year}"

WEATHER_INCIDENT_TYPES = [
    "Coastal Storm",
    "Drought",
    "Fire",
    "Flood",
    "Freezing",
    "Hurricane",
    "Mud/Landslide",
    "Severe Ice Storm",
    "Severe Storm",
    "Snowstorm",
    "Tornado",
    "Tropical Storm",
    "Typhoon",
    "Winter Storm",
]

USDA_WEATHER_CAUSE_TERMS = [
    "cold",
    "cyclone",
    "drought",
    "excess moisture",
    "excess precipitation",
    "flood",
    "freeze",
    "frost",
    "hail",
    "heat",
    "hurricane",
    "ice",
    "lightning",
    "moisture",
    "precipitation",
    "rain",
    "snow",
    "storm",
    "tornado",
    "weather",
    "wind",
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
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _rma_zip_url(year: int, *, timeout: int = 120) -> str:
    page_url = USDA_RMA_COL_PAGE_URL.format(year=year)
    html = _request_text(page_url, timeout=timeout, retries=0)
    match = re.search(r'href="([^"]*colsom_' + str(year) + r'\.zip)"', html, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Could not find USDA RMA colsom_{year}.zip link on {page_url}")
    return urllib.parse.urljoin(page_url, match.group(1))


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


def _fema_get(endpoint: str, params: dict[str, str], *, timeout: int = 180, retries: int = 8) -> dict[str, Any]:
    url = f"{FEMA_API}/{endpoint}?{urllib.parse.urlencode(params, safe=',() ')}"
    return json.loads(_request_text(url, timeout=timeout, retries=retries))


def _fema_metadata_fields(dataset_name: str, version: int) -> set[str]:
    params = {
        "$select": "name",
        "$filter": f"openFemaDataSet eq '{dataset_name}' and datasetVersion eq {version}",
        "$top": "10000",
    }
    payload = _fema_get("v1/OpenFemaDataSetFields", params)
    return {row["name"] for row in payload.get("OpenFemaDataSetFields", []) if row.get("name")}


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


def _openfema_filter(start_year: int, end_year: int) -> str:
    start = f"{start_year}-01-01T00:00:00.000Z"
    end = f"{end_year + 1}-01-01T00:00:00.000Z"
    incidents = " or ".join(f"incidentType eq '{value}'" for value in WEATHER_INCIDENT_TYPES)
    return f"declarationDate ge '{start}' and declarationDate lt '{end}' and ({incidents})"


def _process_fema_declarations(start_year: int, end_year: int, output_dir: Path) -> pd.DataFrame:
    fields = [
        "disasterNumber",
        "declarationDate",
        "incidentBeginDate",
        "incidentEndDate",
        "incidentType",
        "state",
        "designatedArea",
        "fipsStateCode",
        "fipsCountyCode",
        "declarationTitle",
        "ihProgramDeclared",
        "paProgramDeclared",
        "hmProgramDeclared",
    ]
    df = _fema_fetch_all(
        "v2/DisasterDeclarationsSummaries",
        "DisasterDeclarationsSummaries",
        {"$select": ",".join(fields), "$filter": _openfema_filter(start_year, end_year), "$orderby": "declarationDate"},
    )
    if not df.empty:
        df["county_fips"] = df["fipsStateCode"].astype(str).str.zfill(2) + df["fipsCountyCode"].astype(str).str.zfill(3)
    path = output_dir / "fema_disaster_declarations_weather_county.csv"
    df.to_csv(path, index=False)
    return df


def _or_chunks(values: Iterable[Any], field: str, chunk_size: int = 5) -> Iterable[str]:
    items = [int(value) for value in sorted(set(values)) if pd.notna(value)]
    for start in range(0, len(items), chunk_size):
        yield "(" + " or ".join(f"{field} eq {value}" for value in items[start : start + chunk_size]) + ")"


def _process_fema_ia(disaster_numbers: Iterable[Any], output_dir: Path) -> pd.DataFrame:
    dataset = "IndividualsAndHouseholdsProgramValidRegistrations"
    available = _fema_metadata_fields(dataset, 2)
    wanted = [
        "disasterNumber",
        "damagedStateAbbreviation",
        "damagedCounty",
        "censusBlockId",
        "ownRent",
        "ihpAmount",
        "haAmount",
        "onaAmount",
        "rpfvl",
        "ppfvl",
    ]
    fields = [field for field in wanted if field in available]
    frames = []
    for filt in _or_chunks(disaster_numbers, "disasterNumber"):
        frames.append(
            _fema_fetch_all(
                "v2/IndividualsAndHouseholdsProgramValidRegistrations",
                "IndividualsAndHouseholdsProgramValidRegistrations",
                {"$select": ",".join(fields), "$filter": filt},
            )
        )
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=fields)
    if raw.empty:
        out = raw
    else:
        if "censusBlockId" in raw.columns:
            raw["county_fips"] = raw["censusBlockId"].astype(str).str[:5]
        amount_cols = [col for col in ["ihpAmount", "haAmount", "onaAmount", "rpfvl", "ppfvl"] if col in raw.columns]
        for col in amount_cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)
        group_cols = [col for col in ["disasterNumber", "county_fips", "damagedStateAbbreviation", "damagedCounty"] if col in raw.columns]
        out = raw.groupby(group_cols, dropna=False).agg(valid_registrations=("disasterNumber", "size"), **{col: (col, "sum") for col in amount_cols}).reset_index()
    path = output_dir / "fema_individual_assistance_county.csv"
    out.to_csv(path, index=False)
    return out


def _process_fema_pa(start_year: int, end_year: int, output_dir: Path) -> pd.DataFrame:
    dataset = "PublicAssistanceGrantAwardActivities"
    available = _fema_metadata_fields(dataset, 2)
    wanted = [
        "disasterNumber",
        "declarationDate",
        "incidentType",
        "state",
        "county",
        "applicantName",
        "projectTitle",
        "federalShareObligated",
        "totalObligated",
        "projectAmount",
    ]
    fields = [field for field in wanted if field in available]
    raw = _fema_fetch_all(
        "v2/PublicAssistanceGrantAwardActivities",
        "PublicAssistanceGrantAwardActivities",
        {"$select": ",".join(fields), "$filter": _openfema_filter(start_year, end_year)},
    )
    if raw.empty:
        out = raw
    else:
        amount_cols = [col for col in ["federalShareObligated", "totalObligated", "projectAmount"] if col in raw.columns]
        for col in amount_cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)
        group_cols = [col for col in ["disasterNumber", "state", "county", "incidentType"] if col in raw.columns]
        out = raw.groupby(group_cols, dropna=False).agg(project_records=("disasterNumber", "size"), **{col: (col, "sum") for col in amount_cols}).reset_index()
    path = output_dir / "fema_public_assistance_county.csv"
    out.to_csv(path, index=False)
    return out


def _normalize_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _find_col(columns: Iterable[str], candidates: list[str]) -> str | None:
    lookup = {_normalize_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    for normalized, original in lookup.items():
        if any(candidate in normalized for candidate in candidates):
            return original
    return None


def _read_rma_zip(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".txt", ".csv", ".dat")):
                continue
            with zf.open(name) as member:
                raw = member.read()
            sample = raw[:4096].decode("utf-8", errors="replace")
            sep = "|" if "|" in sample else ","
            frame = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)
            frame.columns = [_normalize_name(col) for col in frame.columns]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _process_usda_rma(paths: list[Path], output_path: Path) -> pd.DataFrame:
    frames = []
    for path in paths:
        year = int(re.search(r"(\d{4})", path.name).group(1))
        frame = _read_rma_zip(path)
        if frame.empty:
            continue
        frame["source_year"] = year
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        raw.to_csv(output_path, index=False)
        return raw

    state_col = _find_col(raw.columns, ["state_code", "state"])
    county_col = _find_col(raw.columns, ["county_code", "county"])
    cause_col = _find_col(raw.columns, ["cause_of_loss_description", "cause_of_loss_desc", "cause"])
    indemnity_col = _find_col(raw.columns, ["indemnity_amount", "indemnity"])
    liability_col = _find_col(raw.columns, ["liability_amount", "liability"])
    year_col = _find_col(raw.columns, ["year_of_loss", "commodity_year", "crop_year", "source_year"]) or "source_year"

    for col in [indemnity_col, liability_col]:
        if col:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)
    if cause_col:
        terms = "|".join(re.escape(term) for term in USDA_WEATHER_CAUSE_TERMS)
        raw = raw[raw[cause_col].astype(str).str.lower().str.contains(terms, na=False)].copy()
    if state_col and county_col:
        raw["county_fips"] = raw[state_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(2) + raw[county_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(3)

    group_cols = [col for col in [year_col, "county_fips", state_col, county_col, cause_col] if col and col in raw.columns]
    agg: dict[str, tuple[str, str]] = {"records": (group_cols[0], "size")}
    if indemnity_col:
        agg["indemnity"] = (indemnity_col, "sum")
    if liability_col:
        agg["liability"] = (liability_col, "sum")
    out = raw.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    out.to_csv(output_path, index=False)
    return out


def estimate(start_year: int, end_year: int, output_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    noaa = _noaa_files(start_year, end_year)
    for year, (filename, size) in noaa.items():
        url = urllib.parse.urljoin(NOAA_INDEX_URL, filename)
        rows.append({"source": "noaa_storm_events", "year": year, "url": url, "estimated_bytes": size})
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "climate_damage_download_estimate.csv"
    _write_manifest(path, rows)
    return path


def download(start_year: int, end_year: int, output_dir: Path, *, force: bool) -> None:
    raw_dir = output_dir / "raw"
    manifest_rows: list[dict[str, Any]] = []
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    noaa_paths = []
    for year, (filename, _) in _noaa_files(start_year, end_year).items():
        url = urllib.parse.urljoin(NOAA_INDEX_URL, filename)
        path, size = _download_file(url, raw_dir / "noaa_storm_events" / filename, force=force)
        noaa_paths.append(path)
        manifest_rows.append({"source": "noaa_storm_events", "year": year, "url": url, "path": str(path.relative_to(output_dir)), "bytes": size})
    _process_noaa(noaa_paths, output_dir / "noaa_storm_events_county_damage.csv")

    manifest_rows.append({"source": "download_run", "year": f"{start_year}-{end_year}", "url": "", "path": "", "bytes": "", "elapsed_seconds": round(time.time() - started, 2)})
    _write_manifest(output_dir / "climate_damage_source_manifest.csv", manifest_rows)


def parse_args() -> argparse.Namespace:
    start_year, end_year = _default_years()
    parser = argparse.ArgumentParser(description="Download county-level weather/climate damage data from NOAA, FEMA, and USDA RMA.")
    parser.add_argument("--start-year", type=int, default=start_year)
    parser.add_argument("--end-year", type=int, default=end_year)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.estimate_only:
        path = estimate(args.start_year, args.end_year, args.output_dir)
        print(f"Wrote estimate to {path}")
        return
    download(args.start_year, args.end_year, args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
