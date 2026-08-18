"""Reusable event-window builders for climate/disaster housing visualizations."""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import numpy as np
import pandas as pd

PRE_EVENT_START_PHASE = "12 months before through event start"
DEFAULT_PROPERTY_TYPE = "All Residential"
DEFAULT_NOAA_DAMAGE_THRESHOLD = 1_000_000_000
DEFAULT_EXCLUDED_FEMA_INCIDENT_TYPES = (
    "Biological",
    "Dam/Levee Break",
    "Chemical",
    "Terrorist",
    "Other",
    "Toxic Substances",
)
DEFAULT_HOUSING_INDEX_COMPONENTS = (
    "median_ppsf_yoy",
    "avg_sale_to_list_yoy",
    "homes_sold_yoy",
    "inventory_yoy",
)


def load_disaster_events(
    con: duckdb.DuckDBPyConnection,
    *,
    noaa_damage_threshold: float = DEFAULT_NOAA_DAMAGE_THRESHOLD,
    excluded_fema_incident_types: Sequence[str] = DEFAULT_EXCLUDED_FEMA_INCIDENT_TYPES,
    include_fema: bool = True,
    include_noaa: bool = True,
) -> pd.DataFrame:
    """Load the disaster event scope used by the housing market event-window views."""

    frames: list[pd.DataFrame] = []
    if include_fema:
        excluded_types = tuple(excluded_fema_incident_types)
        exclusion_clause = ""
        params: list[object] = []
        if excluded_types:
            placeholders = ", ".join("?" for _ in excluded_types)
            exclusion_clause = f"AND incidentType NOT IN ({placeholders})"
            params.extend(excluded_types)
        frames.append(
            con.execute(
                f"""
                SELECT
                    'fema' AS event_source,
                    disasterNumber AS source_event_id,
                    fips,
                    incidentType AS event_type,
                    declarationTitle AS event_name,
                    incident_begin_date AS event_start,
                    coalesce(incident_end_date, incident_begin_date) AS event_end,
                    CAST(NULL AS DOUBLE) AS total_damage_amount
                FROM mart.fema_disaster_declarations
                WHERE fips IS NOT NULL
                  AND incident_begin_date IS NOT NULL
                  {exclusion_clause}
                """,
                params,
            ).df()
        )

    if include_noaa:
        frames.append(
            con.execute(
                """
                SELECT
                    'noaa' AS event_source,
                    event_id AS source_event_id,
                    fips,
                    event_type,
                    event_type AS event_name,
                    begin_timestamp AS event_start,
                    coalesce(end_timestamp, begin_timestamp) AS event_end,
                    total_damage_amount
                FROM mart.noaa_storm_events
                WHERE fips IS NOT NULL
                  AND begin_timestamp IS NOT NULL
                  AND total_damage_amount >= ?
                """,
                [noaa_damage_threshold],
            ).df()
        )

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "event_source",
                "source_event_id",
                "fips",
                "event_type",
                "event_name",
                "event_start",
                "event_end",
                "total_damage_amount",
                "event_start_month",
                "event_end_month",
                "event_key",
            ]
        )

    events = pd.concat(frames, ignore_index=True)
    events["fips"] = events["fips"].astype(str).str.zfill(5)
    events["event_start"] = pd.to_datetime(events["event_start"])
    events["event_end"] = pd.to_datetime(events["event_end"]).fillna(events["event_start"])
    events = events.loc[events["event_end"].ge(events["event_start"])].copy()
    events["event_start_month"] = events["event_start"].dt.to_period("M").dt.to_timestamp()
    events["event_end_month"] = events["event_end"].dt.to_period("M").dt.to_timestamp()
    events["event_key"] = (
        events["event_source"].astype(str)
        + ":"
        + events["source_event_id"].fillna("").astype(str)
        + ":"
        + events["fips"].astype(str)
        + ":"
        + events["event_start_month"].astype(str)
    )
    return events


def load_redfin_county_monthly(
    con: duckdb.DuckDBPyConnection,
    *,
    property_type: str = DEFAULT_PROPERTY_TYPE,
) -> pd.DataFrame:
    """Load Redfin county monthly rows for one property type and add event-window-ready columns."""

    housing = con.execute(
        """
        SELECT
            fips,
            REGION AS county_label,
            STATE_CODE AS state_code,
            period_begin,
            period_end,
            try_cast(MEDIAN_PPSF_YOY AS DOUBLE) AS median_ppsf_yoy,
            try_cast(AVG_SALE_TO_LIST_YOY AS DOUBLE) AS avg_sale_to_list_yoy,
            try_cast(HOMES_SOLD_YOY AS DOUBLE) AS homes_sold_yoy,
            try_cast(INVENTORY_YOY AS DOUBLE) AS inventory_yoy,
            try_cast(NEW_LISTINGS_YOY AS DOUBLE) AS new_listings_yoy,
            try_cast(MEDIAN_DOM_YOY AS DOUBLE) AS median_dom_yoy,
            try_cast(PRICE_DROPS_YOY AS DOUBLE) AS price_drops_yoy
        FROM mart.redfin_county_monthly
        WHERE property_type = ?
          AND fips IS NOT NULL
          AND period_begin IS NOT NULL
        """,
        [property_type],
    ).df()
    if housing.empty:
        return housing

    housing["fips"] = housing["fips"].astype(str).str.zfill(5)
    housing["period_begin"] = pd.to_datetime(housing["period_begin"])
    housing["period_month"] = housing["period_begin"].dt.to_period("M").dt.to_timestamp()
    housing["year"] = housing["period_month"].dt.year
    housing["month"] = housing["period_month"].dt.month
    housing = housing.sort_values(["fips", "period_month"]).reset_index(drop=True)

    for component in DEFAULT_HOUSING_INDEX_COMPONENTS:
        values = pd.to_numeric(housing[component], errors="coerce")
        std = values.std(skipna=True)
        z_score = (values - values.mean(skipna=True)) / std if pd.notna(std) and std else np.nan
        if component == "inventory_yoy":
            z_score = -z_score
        housing[f"{component}_z"] = z_score

    housing["housing_market_index"] = housing[
        [f"{component}_z" for component in DEFAULT_HOUSING_INDEX_COMPONENTS]
    ].mean(axis=1, skipna=True)
    return housing.drop(columns=[f"{component}_z" for component in DEFAULT_HOUSING_INDEX_COMPONENTS])


def filter_complete_event_window_lines(
    frame: pd.DataFrame,
    *,
    x_col: str,
    line_col: str,
    metric_col: str,
    required_x_values: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Keep only lines with non-null metric values for every required event-window month."""

    filtered = frame.dropna(subset=[metric_col, x_col, line_col]).copy()
    required = set(required_x_values if required_x_values is not None else filtered[x_col].dropna().unique())
    if not required:
        return filtered.iloc[0:0].copy()

    line_months = filtered.groupby(line_col)[x_col].agg(lambda values: set(values.dropna()))
    complete_lines = line_months.loc[line_months.apply(lambda values: required.issubset(values))].index
    return filtered.loc[filtered[line_col].isin(complete_lines)].copy()


def event_window_months(pre_event_months: int, post_event_months: int) -> list[int]:
    """Return relative event-window months, including event start month as month 0."""

    return list(range(-pre_event_months, 1)) + list(range(1, post_event_months + 1))


def event_window_phase_sql(pre_event_months: int, post_event_months: int) -> str:
    cases = [
        f"WHEN ewm.event_window_month BETWEEN -{pre_event_months} AND 0 THEN '{PRE_EVENT_START_PHASE}'",
        "WHEN ewm.event_window_month BETWEEN 1 AND 12 THEN '1-12 months after event end'",
    ]
    for start in range(13, post_event_months + 1, 12):
        end = min(start + 11, post_event_months)
        cases.append(f"WHEN ewm.event_window_month BETWEEN {start} AND {end} THEN '{start}-{end} months after event end'")
    return "CASE " + " ".join(cases) + " END"


def add_event_window_columns(
    window: pd.DataFrame,
    *,
    event_start_month: pd.Timestamp,
    event_end_month: pd.Timestamp,
    pre_event_months: int,
    post_event_months: int,
) -> pd.DataFrame:
    """Add relative event-window month columns to one county-event housing slice."""

    if window.empty:
        return window.copy()

    result = window.copy()
    result["event_duration_months"] = (
        (event_end_month.year - event_start_month.year) * 12
        + (event_end_month.month - event_start_month.month)
    )
    result["months_from_event_start"] = (
        (result["period_month"].dt.year - event_start_month.year) * 12
        + (result["period_month"].dt.month - event_start_month.month)
    )
    result["months_after_event_end"] = (
        (result["period_month"].dt.year - event_end_month.year) * 12
        + (result["period_month"].dt.month - event_end_month.month)
    )
    pre_mask = result["months_from_event_start"].between(-pre_event_months, 0)
    post_mask = result["months_after_event_end"].between(1, post_event_months)
    result = result.loc[pre_mask | post_mask].copy()
    if result.empty:
        return result

    result["event_window_month"] = np.where(
        result["months_from_event_start"].le(0),
        result["months_from_event_start"],
        result["months_after_event_end"],
    )
    result["event_window_phase"] = np.select(
        [
            result["event_window_month"].between(-pre_event_months, 0),
            result["event_window_month"].between(1, 12),
            result["event_window_month"].between(13, 24),
            result["event_window_month"].between(25, 36),
            result["event_window_month"].between(37, 48),
            result["event_window_month"].between(49, 60),
        ],
        [
            PRE_EVENT_START_PHASE,
            "1-12 months after event end",
            "13-24 months after event end",
            "25-36 months after event end",
            "37-48 months after event end",
            "49-60 months after event end",
        ],
        default=pd.NA,
    )
    return result


def build_affected_event_windows(
    events: pd.DataFrame,
    housing: pd.DataFrame,
    *,
    pre_event_months: int = 12,
    post_event_months: int = 24,
) -> pd.DataFrame:
    """Build affected-county monthly event windows for event/housing dataframes."""

    if events.empty or housing.empty:
        return pd.DataFrame()

    rows = []
    housing_by_fips = {fips: group for fips, group in housing.groupby("fips")}
    for event in events.itertuples(index=False):
        county_housing = housing_by_fips.get(event.fips)
        if county_housing is None:
            continue

        start_window = event.event_start_month - pd.DateOffset(months=pre_event_months)
        end_window = event.event_end_month + pd.DateOffset(months=post_event_months)
        window = county_housing.loc[county_housing["period_month"].between(start_window, end_window)].copy()
        if window.empty:
            continue

        window["event_key"] = event.event_key
        window["event_source"] = event.event_source
        window["source_event_id"] = event.source_event_id
        window["event_type"] = event.event_type
        window["event_name"] = event.event_name
        window["event_start_month"] = event.event_start_month
        window["event_end_month"] = event.event_end_month
        window = add_event_window_columns(
            window,
            event_start_month=event.event_start_month,
            event_end_month=event.event_end_month,
            pre_event_months=pre_event_months,
            post_event_months=post_event_months,
        )
        if window.empty:
            continue
        window["line_id"] = window["event_key"]
        rows.append(window)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_affected_state_month_baseline_windows(
    events: pd.DataFrame,
    housing: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    pre_event_months: int = 12,
    post_event_months: int = 24,
) -> pd.DataFrame:
    """Return affected medians and same-state/month non-affected baselines by event window."""

    if events.empty or housing.empty:
        return pd.DataFrame()

    metrics = list(metric_columns)
    if not metrics:
        return pd.DataFrame()

    group_cols = ["event_source", "source_event_id", "event_start_month", "event_end_month"]
    event_groups = events[group_cols].drop_duplicates().reset_index(drop=True).copy()
    event_groups["event_group_id"] = np.arange(len(event_groups), dtype=np.int64)
    event_groups["event_key"] = event_groups[group_cols].astype(str).agg("|".join, axis=1)

    affected_counties = (
        events[group_cols + ["fips"]]
        .dropna(subset=["fips"])
        .merge(event_groups[group_cols + ["event_group_id"]], on=group_cols, how="inner")
        [["event_group_id", "fips"]]
        .drop_duplicates()
    )
    event_window_months_df = pd.DataFrame(
        {"event_window_month": event_window_months(pre_event_months, post_event_months)}
    )
    metric_selects = "\nUNION ALL\n".join(
        f"SELECT fips, state_code, period_month, '{metric}' AS metric_name, {metric} AS metric_value "
        f"FROM housing_df WHERE {metric} IS NOT NULL"
        for metric in metrics
    )
    required_month_count = len(event_window_months_df)
    phase_sql = event_window_phase_sql(pre_event_months, post_event_months)

    local_con = duckdb.connect()
    try:
        local_con.register("event_groups_df", event_groups)
        local_con.register("affected_counties_df", affected_counties)
        local_con.register("event_window_months_df", event_window_months_df)
        local_con.register("housing_df", housing)
        return local_con.execute(
            f"""
            WITH event_months AS (
                SELECT
                    eg.event_group_id,
                    eg.event_key,
                    eg.event_source,
                    eg.source_event_id,
                    eg.event_start_month,
                    eg.event_end_month,
                    ewm.event_window_month,
                    {phase_sql} AS event_window_phase,
                    CASE
                        WHEN ewm.event_window_month <= 0
                        THEN eg.event_start_month + ewm.event_window_month * INTERVAL 1 MONTH
                        ELSE eg.event_end_month + ewm.event_window_month * INTERVAL 1 MONTH
                    END AS period_month
                FROM event_groups_df AS eg
                CROSS JOIN event_window_months_df AS ewm
            ),
            housing_long AS (
                {metric_selects}
            ),
            affected_values AS (
                SELECT
                    em.event_group_id,
                    em.event_key,
                    em.event_source,
                    em.source_event_id,
                    em.event_start_month,
                    em.event_end_month,
                    em.event_window_month,
                    em.event_window_phase,
                    em.period_month,
                    ac.fips,
                    hl.state_code,
                    hl.metric_name,
                    hl.metric_value
                FROM event_months AS em
                INNER JOIN affected_counties_df AS ac
                    ON em.event_group_id = ac.event_group_id
                INNER JOIN housing_long AS hl
                    ON ac.fips = hl.fips
                   AND em.period_month = hl.period_month
            ),
            complete_affected_windows AS (
                SELECT event_group_id, fips, metric_name
                FROM affected_values
                GROUP BY event_group_id, fips, metric_name
                HAVING count(DISTINCT event_window_month) = {required_month_count}
            ),
            complete_affected_values AS (
                SELECT av.*
                FROM affected_values AS av
                INNER JOIN complete_affected_windows AS complete
                    ON av.event_group_id = complete.event_group_id
                   AND av.fips = complete.fips
                   AND av.metric_name = complete.metric_name
            ),
            affected_monthly AS (
                SELECT
                    event_window_month,
                    event_window_phase,
                    median(metric_value) AS metric_value,
                    metric_name,
                    count(DISTINCT fips) AS complete_county_count,
                    event_key,
                    'affected' AS affected_status,
                    event_source,
                    source_event_id,
                    event_start_month,
                    event_end_month,
                    event_key || ':affected:' || metric_name AS line_id
                FROM complete_affected_values
                GROUP BY
                    event_window_month,
                    event_window_phase,
                    metric_name,
                    event_key,
                    event_source,
                    source_event_id,
                    event_start_month,
                    event_end_month
            ),
            state_month_baseline_values AS (
                SELECT
                    cav.event_group_id,
                    cav.event_key,
                    cav.event_source,
                    cav.source_event_id,
                    cav.event_start_month,
                    cav.event_end_month,
                    cav.event_window_month,
                    cav.event_window_phase,
                    cav.fips AS affected_fips,
                    cav.metric_name,
                    median(control.metric_value) AS metric_value
                FROM complete_affected_values AS cav
                INNER JOIN housing_long AS control
                    ON cav.state_code = control.state_code
                   AND cav.period_month = control.period_month
                   AND cav.metric_name = control.metric_name
                   AND cav.fips <> control.fips
                LEFT JOIN affected_counties_df AS other_affected
                    ON cav.event_group_id = other_affected.event_group_id
                   AND control.fips = other_affected.fips
                WHERE other_affected.fips IS NULL
                GROUP BY
                    cav.event_group_id,
                    cav.event_key,
                    cav.event_source,
                    cav.source_event_id,
                    cav.event_start_month,
                    cav.event_end_month,
                    cav.event_window_month,
                    cav.event_window_phase,
                    cav.fips,
                    cav.metric_name
            ),
            complete_baseline_windows AS (
                SELECT event_group_id, affected_fips, metric_name
                FROM state_month_baseline_values
                WHERE metric_value IS NOT NULL
                GROUP BY event_group_id, affected_fips, metric_name
                HAVING count(DISTINCT event_window_month) = {required_month_count}
            ),
            baseline_monthly AS (
                SELECT
                    baseline.event_window_month,
                    baseline.event_window_phase,
                    median(baseline.metric_value) AS metric_value,
                    baseline.metric_name,
                    count(DISTINCT baseline.affected_fips) AS complete_county_count,
                    baseline.event_key,
                    'state_month_baseline' AS affected_status,
                    baseline.event_source,
                    baseline.source_event_id,
                    baseline.event_start_month,
                    baseline.event_end_month,
                    baseline.event_key || ':state_month_baseline:' || baseline.metric_name AS line_id
                FROM state_month_baseline_values AS baseline
                INNER JOIN complete_baseline_windows AS complete
                    ON baseline.event_group_id = complete.event_group_id
                   AND baseline.affected_fips = complete.affected_fips
                   AND baseline.metric_name = complete.metric_name
                GROUP BY
                    baseline.event_window_month,
                    baseline.event_window_phase,
                    baseline.metric_name,
                    baseline.event_key,
                    baseline.event_source,
                    baseline.source_event_id,
                    baseline.event_start_month,
                    baseline.event_end_month
            )
            SELECT * FROM affected_monthly
            UNION ALL
            SELECT * FROM baseline_monthly
            ORDER BY event_key, affected_status, metric_name, event_window_month
            """
        ).df()
    finally:
        local_con.close()
