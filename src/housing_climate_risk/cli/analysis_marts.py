"""Build persisted event-window analysis marts used by notebooks."""

from __future__ import annotations

import pandas as pd

from housing_climate_risk.page_data.event_windows import (
    DEFAULT_EXCLUDED_FEMA_INCIDENT_TYPES,
    DEFAULT_NOAA_DAMAGE_THRESHOLD,
    build_affected_event_windows,
    load_disaster_events,
)


PRE_EVENT_MONTHS = 12
POST_EVENT_HORIZONS = (12, 24, 36, 48, 60)
MAX_POST_EVENT_MONTHS = max(POST_EVENT_HORIZONS)
PROPERTY_TYPE = "All Residential"
METRICS = (
    "median_ppsf_yoy",
    "housing_market_index",
    "avg_sale_to_list_yoy",
    "homes_sold_yoy",
    "inventory_yoy",
    "new_listings_yoy",
    "median_dom_yoy",
    "price_drops_yoy",
)
RISK_ORDER = (
    "Very Low",
    "Relatively Low",
    "Relatively Moderate",
    "Relatively High",
    "Very High",
)


def _assert_unique_key(con, table_name: str, columns: tuple[str, ...]) -> None:
    key_sql = ", ".join(columns)
    duplicate_count = con.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT {key_sql}
            FROM {table_name}
            GROUP BY {key_sql}
            HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicate_count:
        raise RuntimeError(
            f"{table_name} violates its declared key ({key_sql}): "
            f"{duplicate_count:,} duplicate groups"
        )


def _empty_analysis_tables(con) -> None:
    con.execute(
        """
        CREATE TABLE analysis.extreme_events_county (
            event_source VARCHAR, source_event_id VARCHAR, fips VARCHAR, event_type VARCHAR,
            event_name VARCHAR, event_start TIMESTAMP, event_end TIMESTAMP,
            total_damage_amount DOUBLE, event_start_month DATE, event_end_month DATE,
            event_key VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analysis.housing_event_windows_monthly (
            event_key VARCHAR, fips VARCHAR, event_window_month INTEGER,
            event_window_phase VARCHAR, period_month DATE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analysis.housing_event_window_summary (
            aggregation_level VARCHAR, risk_rating VARCHAR, horizon_months INTEGER,
            event_window_month INTEGER, event_window_phase VARCHAR, metric_name VARCHAR,
            median_value DOUBLE, mean_value DOUBLE, q25_value DOUBLE, q75_value DOUBLE,
            county_event_count BIGINT, county_count BIGINT
        )
        """
    )


def create_analysis_marts(con) -> None:
    """Persist canonical county-event housing windows and aggregate summaries."""

    con.execute("CREATE SCHEMA IF NOT EXISTS analysis")
    for table in (
        "housing_event_window_summary",
        "housing_event_windows_monthly",
        "extreme_events_county",
        "event_window_config",
    ):
        con.execute(f"DROP TABLE IF EXISTS analysis.{table}")

    config_rows = [
        ("property_type", PROPERTY_TYPE),
        ("pre_event_months", str(PRE_EVENT_MONTHS)),
        ("post_event_horizons", ",".join(map(str, POST_EVENT_HORIZONS))),
        ("noaa_damage_threshold_usd", str(DEFAULT_NOAA_DAMAGE_THRESHOLD)),
        ("included_event_sources", "FEMA declarations and NOAA Storm Events"),
        ("excluded_fema_incident_types", ", ".join(DEFAULT_EXCLUDED_FEMA_INCIDENT_TYPES)),
        ("county_event_deduplication", "Exact event_key: source + source event id + county FIPS + event start month"),
        ("event_start_definition", "Calendar month containing incident/event begin timestamp"),
        ("event_end_definition", "Calendar month containing incident/event end timestamp; start used when absent"),
        ("window_axis", "Negative months and month 0 are relative to event start; positive months are relative to event end"),
        ("completeness_rule", "A county-event-metric must contain every required month from -12 through 0 and 1 through the selected horizon"),
        ("overlapping_events", "Retained as distinct county-events; one housing observation may contribute to multiple event windows"),
        ("multiple_events_per_county", "Retained as distinct county-event lines"),
        ("comparison_method", "Descriptive median across affected county-events; no external control baseline"),
        (
            "cohort_rule",
            "Eligibility is evaluated separately for each horizon; a county-event-metric "
            "must have complete -12 through 0 and +1 through the selected horizon",
        ),
    ]
    config = pd.DataFrame(config_rows, columns=["parameter", "value"])
    con.register("_analysis_config_df", config)
    try:
        con.execute("CREATE TABLE analysis.event_window_config AS SELECT * FROM _analysis_config_df")
    finally:
        con.unregister("_analysis_config_df")

    events = load_disaster_events(
        con,
        noaa_damage_threshold=DEFAULT_NOAA_DAMAGE_THRESHOLD,
        excluded_fema_incident_types=DEFAULT_EXCLUDED_FEMA_INCIDENT_TYPES,
    )
    housing = con.execute("SELECT * FROM feature.county_housing_monthly").df()
    if events.empty or housing.empty:
        _empty_analysis_tables(con)
        return

    housing = housing.rename(columns={"housing_month": "period_month", "county_name": "county_label"})
    housing["period_month"] = pd.to_datetime(housing["period_month"])
    housing["fips"] = housing["fips"].astype(str).str.zfill(5)
    events = events.sort_values(
        ["event_source", "source_event_id", "fips", "event_start_month", "event_end_month"]
    ).drop_duplicates(subset=["event_key"], keep="first")

    eligible_start = housing["period_month"].min() + pd.DateOffset(months=PRE_EVENT_MONTHS)
    minimum_post_event_months = min(POST_EVENT_HORIZONS)
    eligible_end = housing["period_month"].max() - pd.DateOffset(months=minimum_post_event_months)
    housing_fips = set(housing["fips"].dropna())
    events = events.loc[
        events["fips"].isin(housing_fips)
        & events["event_start_month"].ge(eligible_start)
        & events["event_end_month"].le(eligible_end)
    ].copy()

    if events.empty:
        _empty_analysis_tables(con)
        return

    windows = build_affected_event_windows(
        events,
        housing,
        pre_event_months=PRE_EVENT_MONTHS,
        post_event_months=MAX_POST_EVENT_MONTHS,
    )
    risk = con.execute("SELECT fips, risk_rating FROM feature.county_risk").df()
    risk["fips"] = risk["fips"].astype(str).str.zfill(5)
    windows = windows.merge(risk, on="fips", how="left")

    event_columns = [
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
    con.register("_analysis_events_df", events[event_columns])
    con.register("_analysis_windows_df", windows)
    try:
        con.execute("CREATE TABLE analysis.extreme_events_county AS SELECT * FROM _analysis_events_df")
        con.execute("CREATE TABLE analysis.housing_event_windows_monthly AS SELECT * FROM _analysis_windows_df")
    finally:
        con.unregister("_analysis_events_df")
        con.unregister("_analysis_windows_df")

    summaries: list[pd.DataFrame] = []
    for metric in METRICS:
        values = windows.loc[
            windows[metric].notna(),
            ["line_id", "event_key", "fips", "risk_rating", "event_window_month", "event_window_phase", metric],
        ].copy()
        values = values.rename(columns={metric: "metric_value"})
        for horizon in POST_EVENT_HORIZONS:
            required = set(range(-PRE_EVENT_MONTHS, 1)) | set(range(1, horizon + 1))
            scoped = values.loc[values["event_window_month"].isin(required)].copy()
            line_months = scoped.groupby("line_id")["event_window_month"].agg(lambda x: set(x))
            complete_ids = line_months.loc[line_months.apply(lambda x: required.issubset(x))].index
            complete = scoped.loc[scoped["line_id"].isin(complete_ids)].copy()
            if complete.empty:
                continue

            def aggregate(frame: pd.DataFrame, group_columns: list[str], level: str) -> pd.DataFrame:
                grouped = frame.groupby(group_columns, observed=True)["metric_value"]
                result = grouped.agg(
                    median_value="median",
                    mean_value="mean",
                    q25_value=lambda x: x.quantile(0.25),
                    q75_value=lambda x: x.quantile(0.75),
                ).reset_index()
                counts = (
                    frame.groupby(group_columns, observed=True)
                    .agg(county_event_count=("line_id", "nunique"), county_count=("fips", "nunique"))
                    .reset_index()
                )
                result = result.merge(counts, on=group_columns, how="left")
                result["aggregation_level"] = level
                result["horizon_months"] = horizon
                result["metric_name"] = metric
                return result

            all_summary = aggregate(
                complete,
                ["event_window_month", "event_window_phase"],
                "all_counties",
            )
            all_summary["risk_rating"] = pd.NA
            summaries.append(all_summary)

            by_risk = complete.loc[complete["risk_rating"].isin(RISK_ORDER)].copy()
            if not by_risk.empty:
                summaries.append(
                    aggregate(
                        by_risk,
                        ["risk_rating", "event_window_month", "event_window_phase"],
                        "nri_risk_rating",
                    )
                )

    summary_columns = [
        "aggregation_level",
        "risk_rating",
        "horizon_months",
        "event_window_month",
        "event_window_phase",
        "metric_name",
        "median_value",
        "mean_value",
        "q25_value",
        "q75_value",
        "county_event_count",
        "county_count",
    ]
    summary = pd.concat(summaries, ignore_index=True)[summary_columns] if summaries else pd.DataFrame(columns=summary_columns)
    con.register("_analysis_summary_df", summary)
    try:
        con.execute("CREATE TABLE analysis.housing_event_window_summary AS SELECT * FROM _analysis_summary_df")
    finally:
        con.unregister("_analysis_summary_df")

    con.execute("CREATE INDEX IF NOT EXISTS idx_analysis_events_fips_start ON analysis.extreme_events_county (fips, event_start_month)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_analysis_windows_line_month ON analysis.housing_event_windows_monthly (line_id, event_window_month)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_analysis_summary_scope ON analysis.housing_event_window_summary (aggregation_level, horizon_months, metric_name)")

    _assert_unique_key(con, "analysis.extreme_events_county", ("event_key",))
    _assert_unique_key(
        con,
        "analysis.housing_event_windows_monthly",
        ("line_id", "period_month"),
    )
    _assert_unique_key(
        con,
        "analysis.housing_event_window_summary",
        (
            "aggregation_level",
            "risk_rating",
            "horizon_months",
            "event_window_month",
            "metric_name",
        ),
    )
