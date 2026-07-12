"""
Download Components of Population Change dataset from StatsAmerica.

This dataset contains TRUE net migration data (in-migration minus out-migration)
along with births, deaths, and natural increase at county level.

Source: https://www.statsamerica.org/downloads/default.aspx
Coverage: 1990-2025 (U.S., states, counties, metros)
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
DATASET_NAME = "Components-of-Population-Change.zip"
OUTPUT_DIR = DATA_DIR / "statsamerica"


def download_components_of_population_change(output_dir: Path, *, skip_existing: bool = True) -> list[Path]:
    """
    Download and extract Components of Population Change dataset from StatsAmerica.

    Args:
        output_dir: Directory to save extracted CSV files
        skip_existing: Skip download if files already exist

    Returns:
        List of extracted file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    download_url = f"{STATSAMERICA_BASE_URL}/{DATASET_NAME}"

    # Check if already downloaded
    existing_files = list(output_dir.glob("*.csv"))
    if skip_existing and existing_files:
        print(f"[OK] Found existing files in {output_dir}, skipping download", flush=True)
        print(f"  Existing files: {[f.name for f in existing_files]}", flush=True)
        return existing_files

    print(f"Downloading {DATASET_NAME} from StatsAmerica...", flush=True)
    print(f"  URL: {download_url}", flush=True)

    try:
        # Download ZIP file to memory
        with urllib.request.urlopen(download_url, timeout=120) as response:
            zip_data = response.read()
            file_size_mb = len(zip_data) / (1024 * 1024)
            print(f"[OK] Downloaded {file_size_mb:.2f} MB", flush=True)

        # Extract ZIP contents
        extracted_files = []
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            print(f"\nExtracting ZIP contents:", flush=True)
            for name in zf.namelist():
                if name.endswith('.csv'):
                    print(f"  Extracting: {name}", flush=True)
                    zf.extract(name, output_dir)
                    extracted_path = output_dir / name
                    extracted_files.append(extracted_path)

                    # Print first few rows to verify (try multiple encodings)
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            with open(extracted_path, 'r', encoding=encoding) as f:
                                reader = csv.reader(f)
                                header = next(reader)
                                first_row = next(reader, None)
                                print(f"    Columns: {len(header)}")
                                print(f"    Sample columns: {', '.join(header[:8])}")
                                if first_row:
                                    row_count = sum(1 for _ in f) + 1  # +1 for the row we already read
                                    print(f"    Rows: ~{row_count:,}")
                                print(f"    Encoding: {encoding}")
                                break
                        except UnicodeDecodeError:
                            if encoding == 'cp1252':
                                print(f"    Warning: Could not determine encoding")
                            continue
                else:
                    print(f"  Skipping non-CSV: {name}", flush=True)

        if not extracted_files:
            raise RuntimeError("No CSV files found in ZIP archive")

        print(f"\n[OK] Extracted {len(extracted_files)} file(s) to {output_dir}", flush=True)
        return extracted_files

    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to download: HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download: {e.reason}") from e
    except zipfile.BadZipFile as e:
        raise RuntimeError("Downloaded file is not a valid ZIP archive") from e


def print_dataset_info(csv_path: Path) -> None:
    """Print information about the downloaded dataset."""
    print(f"\n{'='*80}")
    print("Components of Population Change - Dataset Information")
    print(f"{'='*80}\n")

    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                _print_dataset_info_from_file(f, csv_path, encoding)
                return
        except UnicodeDecodeError:
            continue
    print(f"[ERROR] Could not read {csv_path.name} with any encoding")


def _print_dataset_info_from_file(f, csv_path: Path, encoding: str) -> None:
    """Helper to print dataset info from an open file."""
    reader = csv.DictReader(f)
    header = reader.fieldnames

    print(f"File: {csv_path.name}")
    print(f"Encoding: {encoding}")
    print(f"Columns ({len(header)}):")
    for i, col in enumerate(header, 1):
        print(f"  {i:2d}. {col}")

    # Read a few sample rows
    rows = [row for _, row in zip(range(5), reader)]

    print(f"\nSample data (first 5 rows):")
    print("-" * 80)

    # Check if we have county-level data
    county_cols = [col for col in header if 'county' in col.lower() or 'fips' in col.lower()]
    migration_cols = [col for col in header if 'migration' in col.lower() or 'net' in col.lower()]

    print(f"\nCounty identifier columns: {county_cols}")
    print(f"Migration-related columns: {migration_cols}")

    if rows:
        print(f"\nFirst row sample:")
        for key, value in list(rows[0].items())[:10]:
            print(f"  {key}: {value}")

    # Try to identify the structure
    print(f"\n{'='*80}")
    print("Next steps:")
    print(f"{'='*80}")
    print("1. Inspect the CSV to identify FIPS code column")
    print("2. Identify net migration column (likely 'NetMigration' or similar)")
    print("3. Add to database build pipeline if needed")
    print("4. Compare with Census ACS migration data for validation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Components of Population Change dataset from StatsAmerica"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print detailed information about the dataset after download"
    )

    args = parser.parse_args()

    try:
        extracted_files = download_components_of_population_change(
            args.output_dir,
            skip_existing=not args.force
        )

        if args.info and extracted_files:
            print_dataset_info(extracted_files[0])

        print(f"\n[OK] Success! Files saved to: {args.output_dir}")

    except Exception as e:
        print(f"\n[ERROR] {e}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
