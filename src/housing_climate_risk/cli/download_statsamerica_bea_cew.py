"""
Download BEA Personal Income and CEW Total Ownership datasets from StatsAmerica.

BEA: Personal Income by Major Component and Earnings by NAICS Industry
     from the Bureau of Economic Analysis.
CEW: Census of Employment and Wages from the Bureau of Labor Statistics.

Source: https://www.statsamerica.org/downloads/default.aspx
Coverage: annual, U.S. / states / counties
"""

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from housing_climate_risk.paths import DATA_DIR


STATSAMERICA_BASE_URL = "https://www.statsamerica.org/downloads"

DATASETS = {
    "bea": {
        "zip": "BEA-County.zip",
        "description": "BEA Personal Income (county level)",
    },
    "cew": {
        "zip": "CEW-County.zip",
        "description": "CEW Total Ownership (county level)",
    },
}

OUTPUT_DIR = DATA_DIR / "statsamerica"


def _download_and_extract(zip_name: str, description: str, output_dir: Path) -> list[Path]:
    """Download a StatsAmerica ZIP and extract all CSVs to output_dir."""
    download_url = f"{STATSAMERICA_BASE_URL}/{zip_name}"
    print(f"Downloading {description}...", flush=True)
    print(f"  URL: {download_url}", flush=True)

    try:
        with urllib.request.urlopen(download_url, timeout=300) as response:
            zip_data = response.read()
            file_size_mb = len(zip_data) / (1024 * 1024)
            print(f"[OK] Downloaded {file_size_mb:.1f} MB", flush=True)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to download {zip_name}: HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {zip_name}: {e.reason}") from e

    try:
        extracted: list[Path] = []
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            for name in zf.namelist():
                if not name.endswith(".csv"):
                    print(f"  Skipping non-CSV: {name}", flush=True)
                    continue
                print(f"  Extracting: {name}", flush=True)
                zf.extract(name, output_dir)
                extracted.append(output_dir / name)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {zip_name}") from e

    if not extracted:
        raise RuntimeError(f"No CSV files found in {zip_name}")

    print(f"[OK] Extracted {len(extracted)} file(s)", flush=True)
    return extracted


def _print_csv_info(path: Path) -> None:
    """Print column names and a few sample rows from a CSV."""
    print(f"\n{'='*72}")
    print(f"File: {path.name}")
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, encoding=encoding) as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = [next(reader, None) for _ in range(3)]
            print(f"Encoding: {encoding}")
            print(f"Columns ({len(header)}): {header}")
            for r in rows:
                if r:
                    print(f"  Sample: {r[:8]}")
            return
        except UnicodeDecodeError:
            continue
    print("  [WARN] Could not determine encoding")


def download_statsamerica_bea_cew(
    output_dir: Path,
    datasets: list[str],
    *,
    skip_existing: bool = True,
    info: bool = False,
) -> dict[str, list[Path]]:
    """
    Download BEA and/or CEW datasets from StatsAmerica.

    Args:
        output_dir: Directory to save extracted CSV files.
        datasets: Which datasets to download — any of 'bea', 'cew', or both.
        skip_existing: Skip a dataset if any matching CSV already exists in output_dir.
        info: Print column/sample diagnostics after each download.

    Returns:
        Dict mapping dataset key -> list of extracted file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[Path]] = {}

    for key in datasets:
        spec = DATASETS[key]
        zip_name = spec["zip"]
        description = spec["description"]

        # Detect existing files whose name contains the dataset prefix
        prefix = key.upper()  # 'BEA' or 'CEW'
        existing = [p for p in output_dir.glob("*.csv") if p.name.upper().startswith(prefix)]

        if skip_existing and existing:
            print(f"[OK] Found existing {description} files, skipping download", flush=True)
            print(f"  Files: {[p.name for p in existing]}", flush=True)
            results[key] = existing
            if info:
                for p in existing:
                    _print_csv_info(p)
            continue

        extracted = _download_and_extract(zip_name, description, output_dir)
        results[key] = extracted
        if info:
            for p in extracted:
                _print_csv_info(p)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BEA Personal Income and CEW Total Ownership datasets from StatsAmerica"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["bea", "cew", "all"],
        default=["all"],
        help="Which datasets to download: bea, cew, or all (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print column and sample-row diagnostics after download",
    )

    args = parser.parse_args()

    requested = set()
    for d in args.datasets:
        if d == "all":
            requested.update(DATASETS.keys())
        else:
            requested.add(d)

    try:
        results = download_statsamerica_bea_cew(
            args.output_dir,
            sorted(requested),
            skip_existing=not args.force,
            info=args.info,
        )
        print(f"\n[OK] Done. Files saved to: {args.output_dir}")
        for key, paths in results.items():
            print(f"  {key.upper()}: {[p.name for p in paths]}")
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
