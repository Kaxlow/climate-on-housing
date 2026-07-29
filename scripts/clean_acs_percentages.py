"""
Clean malformed percentage values in ACS DP02 CSV file.

Sets any PE (percentage estimate) values > 100% to NULL, as these are definitively
incorrect (percentages cannot exceed 100%).

Usage:
    python scripts/clean_acs_percentages.py
"""
from pathlib import Path
import pandas as pd

# Paths
ROOT = Path(__file__).parent.parent
ACS_FILE = ROOT / "data" / "acs" / "census_acs5_county_dp02_2015_2024.csv"
BACKUP_FILE = ROOT / "data" / "acs" / "census_acs5_county_dp02_2015_2024_backup.csv"


def clean_percentage_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Clean percentage columns by setting values > 100% to NULL.

    Returns:
        Tuple of (cleaned_df, count_nulled)
    """
    pe_columns = [col for col in df.columns if col.endswith("PE")]
    nulled_count = 0

    for pe_col in pe_columns:
        # Convert to numeric, coercing errors to NaN
        pe_numeric = pd.to_numeric(df[pe_col], errors='coerce')

        # Find values > 100% (definitively wrong for percentages)
        malformed_mask = pe_numeric > 100
        count = malformed_mask.sum()

        if count > 0:
            df.loc[malformed_mask, pe_col] = None
            nulled_count += count
            print(f"  {pe_col}: nulled {count} values > 100%")

    return df, nulled_count


def main():
    print(f"Reading ACS data from: {ACS_FILE}")

    # Read CSV
    df = pd.read_csv(ACS_FILE, dtype=str)
    original_rows = len(df)
    print(f"Loaded {original_rows:,} rows with {len(df.columns)} columns")

    # Check for percentage columns
    pe_columns = [col for col in df.columns if col.endswith("PE")]
    print(f"Found {len(pe_columns)} percentage (PE) columns")

    # Count values > 100% before cleaning
    total_malformed = 0
    for pe_col in pe_columns:
        pe_numeric = pd.to_numeric(df[pe_col], errors='coerce')
        count = (pe_numeric > 100).sum()
        total_malformed += count

    print(f"\nTotal values > 100% before cleaning: {total_malformed:,}")

    if total_malformed == 0:
        print("\nNo malformed percentage values found. CSV is already clean!")
        return

    # Create backup
    print(f"\nCreating backup: {BACKUP_FILE}")
    df.to_csv(BACKUP_FILE, index=False)

    # Clean the data
    print("\nCleaning percentage columns...")
    cleaned_df, nulled_count = clean_percentage_columns(df)

    # Verify cleaning
    remaining_malformed = 0
    for pe_col in pe_columns:
        pe_numeric = pd.to_numeric(cleaned_df[pe_col], errors='coerce')
        count = (pe_numeric > 100).sum()
        remaining_malformed += count

    print(f"\nCleaning summary:")
    print(f"  Values nulled: {nulled_count:,}")
    print(f"  Remaining values > 100%: {remaining_malformed:,}")

    if remaining_malformed > 0:
        print(f"\nWARNING: {remaining_malformed} values > 100% still remain!")
    else:
        print("\n✓ All percentage values are now <= 100%")

    # Save cleaned data
    print(f"\nSaving cleaned data to: {ACS_FILE}")
    cleaned_df.to_csv(ACS_FILE, index=False)

    print("\nDone! Now rebuild the database with:")
    print("  python -m housing_climate_risk.cli.build_database")


if __name__ == "__main__":
    main()
