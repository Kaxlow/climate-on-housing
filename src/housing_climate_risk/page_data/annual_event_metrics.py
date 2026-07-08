"""
Annual event window metrics following event_window_variables.ipynb approach.

This module builds proper annual event windows for metrics that only have annual data,
avoiding the flat-line issues from mixing monthly event windows with annual data.

Uses fully vectorized operations for performance.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd


def build_annual_event_windows(
    events: pd.DataFrame,
    annual_data: pd.DataFrame,
    pre_years: int = 2,
    post_years: int = 3
) -> pd.DataFrame:
    """
    Build annual event windows for a given dataset using vectorized operations.

    Parameters:
    -----------
    events : DataFrame
        Event data with event_start_year, event_end_year, fips, event_key
    annual_data : DataFrame
        Annual county data with fips, year, and metric columns
    pre_years : int
        Number of years before event start to include
    post_years : int
        Number of years after event end to include

    Returns:
    --------
    DataFrame with event_window_year column (relative to event start year)
    """
    if events.empty or annual_data.empty:
        return pd.DataFrame()

    # Merge events with annual data on fips
    merged = events.merge(annual_data, on='fips', how='inner', suffixes=('_event', '_data'))

    # Calculate window boundaries
    merged['window_start_year'] = merged['event_start_year'] - pre_years
    merged['window_end_year'] = merged['event_end_year'] + post_years

    # Filter to only data within the event window
    in_window = merged['year'].between(merged['window_start_year'], merged['window_end_year'])
    result = merged[in_window].copy()

    # Calculate event window year (relative to event start)
    result['event_window_year'] = result['year'] - result['event_start_year']
    result['event_duration_years'] = result['event_end_year'] - result['event_start_year']
    result['line_id'] = result['event_key']

    # Drop the temporary window boundary columns
    result = result.drop(columns=['window_start_year', 'window_end_year'])

    return result


def filter_complete_annual_windows(
    affected: pd.DataFrame,
    metric_col: str,
    pre_years: int = 2,
    post_years: int = 3
) -> pd.DataFrame:
    """
    Filter to only include county-events with complete annual data across the event window.
    Uses vectorized operations for performance.
    """
    if affected.empty:
        return pd.DataFrame()

    # For each line_id, count how many required years have non-null data
    valid_data = affected.dropna(subset=[metric_col])

    # Get event duration for each line
    line_duration = affected.groupby('line_id')['event_duration_years'].first()

    # Count available years per line
    available_years_count = valid_data.groupby('line_id')['event_window_year'].nunique()

    # Calculate required years for each line (align with available_years_count index)
    required_years_count = line_duration.reindex(available_years_count.index).map(lambda d: pre_years + d + post_years + 1)

    # Keep only lines where available matches or exceeds required
    complete_mask = available_years_count >= required_years_count
    complete_lines = available_years_count[complete_mask].index

    return affected.loc[affected['line_id'].isin(complete_lines)].copy()


def aggregate_annual_lines(
    frame: pd.DataFrame,
    group_cols: list[str],
    metric: str
) -> list[dict[str, object]]:
    """
    Aggregate annual event window data by event_window_year.
    Returns data in format compatible with monthly aggregation (converts years to months * 12).
    """
    if frame.empty:
        return []

    q = (
        frame.dropna(subset=[metric, 'event_window_year'])
        .groupby(group_cols + ['event_window_year'], observed=False)[metric]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: 'q1', 0.5: 'median', 0.75: 'q3'})
    )

    return [
        {
            **{col: getattr(row, col) for col in group_cols},
            'month': int(row.event_window_year * 12),  # Convert to months for display compatibility
            'q1': round(float(row.q1), 5) if pd.notna(row.q1) else None,
            'median': round(float(row.median), 5) if pd.notna(row.median) else None,
            'q3': round(float(row.q3), 5) if pd.notna(row.q3) else None,
        }
        for row in q.itertuples(index=False)
    ]


def build_additional_annual_metrics(
    con: duckdb.DuckDBPyConnection,
    events: pd.DataFrame,
    nri: pd.DataFrame,
    pre_years: int = 2,
    post_years: int = 3
) -> list[dict[str, object]]:
    """
    Build annual event window metrics following event_window_variables.ipynb approach.
    Fully optimized with vectorized operations and minimal data loading.

    Parameters:
    -----------
    con : DuckDB connection
    events : DataFrame with event_start_month, event_end_month, fips, event_key
    nri : DataFrame with fips, riskRating
    """
    metrics = []

    if events.empty:
        return metrics

    # Extract event years
    events_annual = events[['fips', 'event_key', 'event_start_month', 'event_end_month']].copy()
    events_annual['event_start_year'] = events_annual['event_start_month'].dt.year
    events_annual['event_end_year'] = events_annual['event_end_month'].dt.year

    # Calculate year range needed based on events
    min_event_year = events_annual['event_start_year'].min()
    max_event_year = events_annual['event_end_year'].max()
    year_filter = f"year BETWEEN {min_event_year - pre_years} AND {max_event_year + post_years}"

    # Get unique fips codes from events to reduce data loading
    event_fips = set(events_annual['fips'].unique())
    fips_filter = "fips IN ('" + "','".join(event_fips) + "')"

    print(f"Loading annual data for {len(event_fips)} counties, years {min_event_year - pre_years} to {max_event_year + post_years}")

    # Load demographic data (filtered by event counties and years)
    demo_query = f"""
    SELECT
        fips,
        CAST(year AS INTEGER) AS year,
        TRY_CAST(domestic_in_migration_rate AS DOUBLE) AS net_migration_rate,
        TRY_CAST(dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct AS DOUBLE) AS communication_barrier_pct
    FROM mart.acs_county_demographic_annual
    WHERE {fips_filter} AND {year_filter}
    """
    demo_df = con.execute(demo_query).df()
    demo_df['fips'] = demo_df['fips'].astype(str).str.zfill(5)

    # Load employment data (filtered)
    employment_query = f"""
    SELECT
        fips,
        CAST(year AS INTEGER) AS year,
        TRY_CAST(dp03_population_16_plus_est AS DOUBLE) AS population_16_plus,
        TRY_CAST(dp03_occupation_civilian_employed_population_16_plus_est AS DOUBLE) AS total_employed
    FROM mart.acs_county_economic_annual
    WHERE {fips_filter} AND {year_filter}
    """
    employment_df = con.execute(employment_query).df()
    employment_df['fips'] = employment_df['fips'].astype(str).str.zfill(5)

    # Calculate employment rate as % of population 16+
    employment_df['employment_rate_pct'] = (
        employment_df['total_employed'] / employment_df['population_16_plus'] * 100
    ).where(employment_df['population_16_plus'] > 0)

    # Load insurance premium data from mart.insurance_premiums_annual (filtered)
    insurance_query = f"""
    SELECT
        fips,
        year,
        median_premium
    FROM mart.insurance_premiums_annual
    WHERE {fips_filter} AND {year_filter}
    """
    insurance_df = con.execute(insurance_query).df()
    insurance_df['fips'] = insurance_df['fips'].astype(str).str.zfill(5)

    # Load median household income for calculating insurance as % of income (filtered)
    income_query = f"""
    SELECT
        fips,
        CAST(year AS INTEGER) AS year,
        TRY_CAST(median_household_income AS DOUBLE) AS median_household_income
    FROM mart.acs_county_affordability_annual
    WHERE {fips_filter} AND {year_filter}
    """
    income_df = con.execute(income_query).df()
    income_df['fips'] = income_df['fips'].astype(str).str.zfill(5)

    # Merge insurance with income to calculate insurance as % of income
    insurance_income_df = insurance_df.merge(income_df, on=['fips', 'year'], how='inner')
    insurance_income_df['insurance_income_pct'] = (
        insurance_income_df['median_premium'] / insurance_income_df['median_household_income'] * 100
    ).where(insurance_income_df['median_household_income'] > 0)

    # Cap at reasonable maximum (insurance shouldn't exceed 50% of income)
    insurance_income_df['insurance_income_pct'] = insurance_income_df['insurance_income_pct'].clip(upper=50)

    print(f"Loaded data: demo={len(demo_df)}, employment={len(employment_df)}, insurance={len(insurance_income_df)}")

    # Build annual event windows for each metric
    metric_specs = [
        (demo_df, 'net_migration_rate', 'Net Migration Rate', 'Migration changes reveal whether people move to or from affected higher-risk counties after events.'),
        (demo_df, 'communication_barrier_pct', 'Share with Communication Barrier', 'If this remains high and unchanged for higher-risk groups, it indicates vulnerable populations that may not relocate easily.'),
        (employment_df, 'employment_rate_pct', 'Employment Rate (% of Pop 16+)', 'Employment declines after events can weaken household demand and reduce house price growth.'),
        (insurance_income_df, 'insurance_income_pct', 'Insurance as % of Income', 'A rising insurance burden can contribute to unaffordability and may indicate insurers pricing in climate risk.'),
    ]

    for i, (annual_data, metric_col, label, description) in enumerate(metric_specs):
        print(f"Processing metric {i+1}/{len(metric_specs)}: {label}")

        if annual_data.empty or metric_col not in annual_data.columns:
            print(f"  Skipping: no data")
            continue

        # Build annual event windows (vectorized)
        affected = build_annual_event_windows(events_annual, annual_data, pre_years, post_years)

        if affected.empty:
            print(f"  Skipping: no affected data")
            continue

        print(f"  Built {len(affected)} event-year records")

        # Add risk rating
        affected = affected.merge(nri, on='fips', how='left')

        # Filter to complete event windows only (vectorized)
        complete = filter_complete_annual_windows(affected, metric_col, pre_years, post_years)

        if complete.empty:
            print(f"  Skipping: no complete windows")
            continue

        n_lines = complete['line_id'].nunique()
        print(f"  {n_lines} complete county-events")

        # Skip if less than 10 complete county-events
        if n_lines < 10:
            print(f"  Skipping: too few complete windows")
            continue

        # Aggregate data
        aggregate = aggregate_annual_lines(
            complete.assign(series='All affected counties'),
            ['series'],
            metric_col
        )

        by_rating = aggregate_annual_lines(
            complete.dropna(subset=['riskRating']),
            ['riskRating'],
            metric_col
        )

        metrics.append({
            'key': metric_col,
            'label': label,
            'description': description,
            'frequency': 'annual',
            'isAnnual': True,
            'conclusion': f'{label} changes around climate events in a way that can alter local demand, affordability, or buyer confidence.',
            'aggregate': aggregate,
            'byRating': by_rating,
        })

        print(f"  Added metric with {len(aggregate)} aggregate points")

    print(f"Total metrics generated: {len(metrics)}")
    return metrics
