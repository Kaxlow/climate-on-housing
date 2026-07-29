"""Build housing market and climate event EDA notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = ROOT / "scripts" / "eda"
NOTEBOOK_PATH = EDA_DIR / "housing_market_climate_event_eda.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


cells = [
    md(
        """# Housing Market And Climate Event EDA

Dataset: `mart.redfin_county_monthly`, `mart.fema_disaster_declarations`, and `mart.noaa_storm_events` in `data/quoll.duckdb`

Scope: Redfin observations are limited to `property_type = "All Residential"` from `mart.redfin_county_monthly`.

This notebook creates county-level monthly housing market plots and climate-event-centered response plots. Run cells from top to bottom after installing the project dependencies."""
    ),
    md("## Setup"),
    code(
        r"""
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 80)
sns.set_theme(style="whitegrid")

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent.parent

DB_PATH = ROOT / "data" / "quoll.duckdb"
OUTPUT_DIR = ROOT / "output" / "eda" / "housing_market_climate_event"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROPERTY_TYPE = "All Residential"
DAMAGE_THRESHOLD = 1_000_000_000
PRE_EVENT_MONTHS = 12
POST_EVENT_MONTHS = 24
EVENT_WINDOW_MONTHS = list(range(-PRE_EVENT_MONTHS, 0)) + list(range(1, POST_EVENT_MONTHS + 1))
YEARLY_PLOT_CAP_QUANTILES = (0.01, 0.99)
MAX_COUNTY_LINES = 500  # Set to None to draw every background county/event line.

con = duckdb.connect(str(DB_PATH), read_only=True)

def q(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()

DB_PATH
"""
    ),
    md("## Load Monthly Redfin Housing Data"),
    code(
        r"""
housing = q(f'''
    SELECT
        fips,
        REGION AS county_label,
        STATE_CODE AS state_code,
        period_begin,
        period_end,
        try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) AS median_ppsf_yoy,
        try_cast(replace(AVG_SALE_TO_LIST_YOY, ',', '') AS DOUBLE) AS avg_sale_to_list_yoy,
        try_cast(replace(HOMES_SOLD_YOY, ',', '') AS DOUBLE) AS homes_sold_yoy,
        try_cast(replace(INVENTORY_YOY, ',', '') AS DOUBLE) AS inventory_yoy,
        try_cast(replace(NEW_LISTINGS_YOY, ',', '') AS DOUBLE) AS new_listings_yoy,
        try_cast(replace(MEDIAN_DOM_YOY, ',', '') AS DOUBLE) AS median_dom_yoy,
        try_cast(replace(PRICE_DROPS_YOY, ',', '') AS DOUBLE) AS price_drops_yoy
    FROM mart.redfin_county_monthly
    WHERE property_type = '{PROPERTY_TYPE}'
      AND fips IS NOT NULL
      AND period_begin IS NOT NULL
''')

housing["period_begin"] = pd.to_datetime(housing["period_begin"])
housing["period_month"] = housing["period_begin"].dt.to_period("M").dt.to_timestamp()
housing["year"] = housing["period_month"].dt.year
housing["month"] = housing["period_month"].dt.month
housing = housing.sort_values(["fips", "period_month"]).reset_index(drop=True)

index_components = ["median_ppsf_yoy", "avg_sale_to_list_yoy", "homes_sold_yoy", "inventory_yoy"]
for component in index_components:
    values = pd.to_numeric(housing[component], errors="coerce")
    std = values.std(skipna=True)
    z = (values - values.mean(skipna=True)) / std if pd.notna(std) and std else np.nan
    if component == "inventory_yoy":
        z = -z
    housing[f"{component}_z"] = z

housing["housing_market_index"] = housing[[f"{c}_z" for c in index_components]].mean(axis=1, skipna=True)
housing = housing.drop(columns=[f"{c}_z" for c in index_components])

display(housing.head())
event_window_metric_columns = [
    "median_ppsf_yoy",
    "housing_market_index",
    "avg_sale_to_list_yoy",
    "homes_sold_yoy",
    "inventory_yoy",
    "new_listings_yoy",
    "median_dom_yoy",
    "price_drops_yoy",
]
display(housing[event_window_metric_columns].describe().T)
display(housing[["period_month", "fips"]].agg({"period_month": ["min", "max"], "fips": "nunique"}))
"""
    ),
    md(
        """## Plot Helpers

The housing market index is the simple average of standardized `MEDIAN_PPSF_YOY`, `AVG_SALE_TO_LIST_YOY`, `HOMES_SOLD_YOY`, and inverted `INVENTORY_YOY`. Inventory is inverted so higher values consistently indicate a tighter or stronger housing market."""
    ),
    code(
        r"""
def maybe_sample_lines(frame: pd.DataFrame, max_lines: int | None = MAX_COUNTY_LINES) -> pd.DataFrame:
    if max_lines is None:
        return frame
    ids = sorted(frame["line_id"].dropna().unique())
    if len(ids) <= max_lines:
        return frame
    rng = np.random.default_rng(42)
    keep = set(rng.choice(ids, size=max_lines, replace=False))
    return frame.loc[frame["line_id"].isin(keep)].copy()


def monthly_summary(frame: pd.DataFrame, x_col: str, metric_col: str) -> pd.DataFrame:
    return (
        frame.groupby(x_col)[metric_col]
        .quantile([0.25, 0.50, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "q25", 0.50: "median", 0.75: "q75"})
    )


def filter_complete_lines(
    frame: pd.DataFrame,
    *,
    x_col: str,
    line_col: str,
    metric_col: str,
    required_x_values: list | None = None,
) -> pd.DataFrame:
    filtered = frame.dropna(subset=[metric_col, x_col, line_col]).copy()
    required = set(required_x_values if required_x_values is not None else filtered[x_col].dropna().unique())
    if not required:
        return filtered.iloc[0:0].copy()
    line_months = filtered.groupby(line_col)[x_col].agg(lambda values: set(values.dropna()))
    complete_lines = line_months.loc[line_months.apply(lambda values: required.issubset(values))].index
    return filtered.loc[filtered[line_col].isin(complete_lines)].copy()


def add_capped_plot_metric(
    frame: pd.DataFrame,
    metric_col: str,
    *,
    lower_quantile: float = YEARLY_PLOT_CAP_QUANTILES[0],
    upper_quantile: float = YEARLY_PLOT_CAP_QUANTILES[1],
) -> tuple[pd.DataFrame, str, tuple[float, float]]:
    plot_frame = frame.copy()
    plot_col = f"{metric_col}_plot_capped"
    values = pd.to_numeric(plot_frame[metric_col], errors="coerce")
    lower = values.quantile(lower_quantile)
    upper = values.quantile(upper_quantile)
    plot_frame[plot_col] = values.clip(lower=lower, upper=upper)
    return plot_frame, plot_col, (lower, upper)


def plot_county_lines_with_iqr(
    frame: pd.DataFrame,
    *,
    metric_col: str,
    x_col: str,
    line_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path | None = None,
    event_band: tuple[float, float] | None = None,
    group_col: str | None = None,
    show_background_lines: bool = True,
    focus_iqr_scale: bool = False,
    phase_bands: bool = False,
    required_x_values: list | None = None,
):
    plot_frame = filter_complete_lines(
        frame,
        x_col=x_col,
        line_col=line_col,
        metric_col=metric_col,
        required_x_values=required_x_values,
    )
    if plot_frame.empty:
        print(f"No rows to plot for {title}")
        return None
    plot_frame["line_id"] = plot_frame[line_col]
    background = maybe_sample_lines(plot_frame)
    summary = monthly_summary(plot_frame, x_col, metric_col)

    fig, ax = plt.subplots(figsize=(12, 6))
    if group_col is None:
        if show_background_lines:
            for _, group in background.groupby(line_col):
                ax.plot(group[x_col], group[metric_col], color="#9aa4b2", alpha=0.08, linewidth=0.7)
        ax.fill_between(summary[x_col], summary["q25"], summary["q75"], color="#2f80ed", alpha=0.18, label="Interquartile range")
        ax.plot(summary[x_col], summary["median"], color="#0b3a75", linewidth=2.6, label="Median")
    else:
        palette = {"affected": "#c83f3f", "unaffected": "#2764c7"}
        for name, group in plot_frame.groupby(group_col):
            stats = monthly_summary(group, x_col, metric_col)
            color = palette.get(str(name), None)
            ax.fill_between(stats[x_col], stats["q25"], stats["q75"], color=color, alpha=0.14)
            ax.plot(stats[x_col], stats["median"], color=color, linewidth=2.6, label=f"{name} median")

    if focus_iqr_scale:
        if group_col is None:
            scale_values = summary[["q25", "median", "q75"]].to_numpy(dtype=float).ravel()
        else:
            scale_parts = []
            for _, group in plot_frame.groupby(group_col):
                stats = monthly_summary(group, x_col, metric_col)
                scale_parts.append(stats[["q25", "median", "q75"]].to_numpy(dtype=float).ravel())
            scale_values = np.concatenate(scale_parts) if scale_parts else np.array([])
        scale_values = scale_values[np.isfinite(scale_values)]
        if scale_values.size:
            lower, upper = np.nanpercentile(scale_values, [2, 98])
            if np.isclose(lower, upper):
                lower, upper = np.nanmin(scale_values), np.nanmax(scale_values)
            pad = (upper - lower) * 0.18 if upper > lower else max(abs(upper) * 0.1, 0.1)
            ax.set_ylim(lower - pad, upper + pad)

    if phase_bands:
        ax.axvspan(-12.5, -0.5, color="#2f80ed", alpha=0.08, label="12 months before start")
        ax.axvspan(0.5, 12.5, color="#167a5b", alpha=0.08, label="1-12 months after end")
        ax.axvspan(12.5, 24.5, color="#a15c00", alpha=0.08, label="13-24 months after end")
        ax.axvline(0, color="#344054", linewidth=1.0, alpha=0.55)
        ax.axvline(12.5, color="#344054", linewidth=0.8, alpha=0.35, linestyle="--")
        ax.set_xlim(-12.5, 24.5)
        ax.set_xticks([-12, -6, -1, 1, 6, 12, 13, 18, 24])
    if event_band is not None:
        ax.axvspan(event_band[0], event_band[1], color="#111827", alpha=0.08, label="Event duration")
    ax.axhline(0, color="#344054", linewidth=0.8, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig, ax
"""
    ),
    md("## 1. Monthly County Median PPSF YOY By Year"),
    code(
        r"""
for year, year_frame in housing.dropna(subset=["median_ppsf_yoy"]).groupby("year"):
    plot_frame, plot_col, caps = add_capped_plot_metric(year_frame, "median_ppsf_yoy")
    plot_county_lines_with_iqr(
        plot_frame,
        metric_col=plot_col,
        x_col="period_month",
        line_col="fips",
        title=f"County median PPSF YOY by month, {year} (plot capped at p1/p99)",
        xlabel="Month",
        ylabel="Median PPSF YOY, capped for plotting",
        output_path=OUTPUT_DIR / f"median_ppsf_yoy_by_county_{year}.png",
        show_background_lines=False,
        focus_iqr_scale=True,
    )
    plt.figtext(0.01, 0.01, f"Plot cap: [{caps[0]:.3f}, {caps[1]:.3f}]", ha="left", fontsize=9, color="#667085")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.show()
"""
    ),
    md("## 2. Monthly County Housing Market Index By Year"),
    code(
        r"""
for year, year_frame in housing.dropna(subset=["housing_market_index"]).groupby("year"):
    plot_frame, plot_col, caps = add_capped_plot_metric(year_frame, "housing_market_index")
    plot_county_lines_with_iqr(
        plot_frame,
        metric_col=plot_col,
        x_col="period_month",
        line_col="fips",
        title=f"County housing market index by month, {year} (plot capped at p1/p99)",
        xlabel="Month",
        ylabel="Housing market index, capped for plotting",
        output_path=OUTPUT_DIR / f"housing_market_index_by_county_{year}.png",
        show_background_lines=False,
        focus_iqr_scale=True,
    )
    plt.figtext(0.01, 0.01, f"Plot cap: [{caps[0]:.3f}, {caps[1]:.3f}]", ha="left", fontsize=9, color="#667085")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.show()
"""
    ),
    md("## Load Climate Events"),
    code(
        r"""
noaa_events = q(f'''
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
      AND total_damage_amount >= {DAMAGE_THRESHOLD}
''')

fema_events = q(f'''
    WITH noaa_billion AS (
        SELECT
            event_id AS noaa_event_id,
            fips,
            begin_timestamp,
            coalesce(end_timestamp, begin_timestamp) AS end_timestamp,
            total_damage_amount
        FROM mart.noaa_storm_events
        WHERE fips IS NOT NULL
          AND begin_timestamp IS NOT NULL
          AND total_damage_amount >= {DAMAGE_THRESHOLD}
    )
    SELECT DISTINCT
        'fema' AS event_source,
        f.disasterNumber AS source_event_id,
        f.fips,
        f.incidentType AS event_type,
        f.declarationTitle AS event_name,
        f.incident_begin_date AS event_start,
        coalesce(f.incident_end_date, f.incident_begin_date) AS event_end,
        n.total_damage_amount
    FROM mart.fema_disaster_declarations
    AS f
    JOIN noaa_billion AS n
     ON f.fips = n.fips
     AND f.incident_begin_date IS NOT NULL
     AND f.incident_begin_date <= n.end_timestamp + INTERVAL 30 DAY
     AND coalesce(f.incident_end_date, f.incident_begin_date) >= n.begin_timestamp - INTERVAL 30 DAY
    WHERE coalesce(f.incidentType, '') <> 'Biological'
''')

events = pd.concat([fema_events, noaa_events], ignore_index=True)
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

display(events["event_source"].value_counts(dropna=False).to_frame("county_event_rows"))
display(events.head())
"""
    ),
    md("## Build Event Windows"),
    code(
        r"""
def build_affected_event_windows(events: pd.DataFrame, housing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    housing_by_fips = {fips: group for fips, group in housing.groupby("fips")}
    for event in events.itertuples(index=False):
        county_housing = housing_by_fips.get(event.fips)
        if county_housing is None:
            continue
        start_window = event.event_start_month - pd.DateOffset(months=PRE_EVENT_MONTHS)
        end_window = event.event_end_month + pd.DateOffset(months=POST_EVENT_MONTHS)
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
        window["event_duration_months"] = (
            (event.event_end_month.year - event.event_start_month.year) * 12
            + (event.event_end_month.month - event.event_start_month.month)
        )
        window["months_from_event_start"] = (
            (window["period_month"].dt.year - event.event_start_month.year) * 12
            + (window["period_month"].dt.month - event.event_start_month.month)
        )
        window["months_after_event_end"] = (
            (window["period_month"].dt.year - event.event_end_month.year) * 12
            + (window["period_month"].dt.month - event.event_end_month.month)
        )
        pre_mask = window["months_from_event_start"].between(-PRE_EVENT_MONTHS, -1)
        post_mask = window["months_after_event_end"].between(1, POST_EVENT_MONTHS)
        window = window.loc[pre_mask | post_mask].copy()
        if window.empty:
            continue
        window["event_window_month"] = np.where(
            window["months_from_event_start"].lt(0),
            window["months_from_event_start"],
            window["months_after_event_end"],
        )
        window["event_window_phase"] = np.select(
            [
                window["event_window_month"].between(-PRE_EVENT_MONTHS, -1),
                window["event_window_month"].between(1, 12),
                window["event_window_month"].between(13, POST_EVENT_MONTHS),
            ],
            ["12 months before event start", "1-12 months after event end", "13-24 months after event end"],
            default=pd.NA,
        )
        window["line_id"] = window["event_key"]
        rows.append(window)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


affected_windows = build_affected_event_windows(events, housing)
display(affected_windows[["event_key", "fips"]].drop_duplicates().shape)
display(affected_windows.head())
"""
    ),
    md("## Housing Market Movement Around Extreme Climate Events"),
    code(
        r"""
event_window_metrics = [
    {"column": "median_ppsf_yoy", "label": "Median PPSF YOY", "slug": "median_ppsf_yoy"},
    {"column": "housing_market_index", "label": "Housing Market Index", "slug": "housing_market_index"},
    {"column": "avg_sale_to_list_yoy", "label": "Average Sale-to-List YOY", "slug": "avg_sale_to_list_yoy"},
    {"column": "homes_sold_yoy", "label": "Homes Sold YOY", "slug": "homes_sold_yoy"},
    {"column": "inventory_yoy", "label": "Inventory YOY", "slug": "inventory_yoy"},
    {"column": "new_listings_yoy", "label": "New Listings YOY", "slug": "new_listings_yoy"},
    {"column": "median_dom_yoy", "label": "Median Days on Market YOY", "slug": "median_dom_yoy"},
    {"column": "price_drops_yoy", "label": "Price Drops YOY", "slug": "price_drops_yoy"},
]

for metric in event_window_metrics:
    plot_county_lines_with_iqr(
        affected_windows,
        metric_col=metric["column"],
        x_col="event_window_month",
        line_col="line_id",
        title=f"Affected county {metric['label']} around extreme climate events",
        xlabel="Relative event-window month",
        ylabel=metric["label"],
        output_path=OUTPUT_DIR / f"{metric['slug']}_event_window_affected.png",
        show_background_lines=False,
        focus_iqr_scale=True,
        phase_bands=True,
        required_x_values=EVENT_WINDOW_MONTHS,
    )
    plt.show()
"""
    ),
    md("## Build Affected Versus Unaffected Event Windows"),
    code(
        r"""
affected_vs_unaffected_metrics = [
    {"column": "median_ppsf_yoy", "label": "Median PPSF YOY", "slug": "median_ppsf_yoy"},
    {"column": "housing_market_index", "label": "Housing Market Index", "slug": "housing_market_index"},
    {"column": "avg_sale_to_list_yoy", "label": "Average Sale-to-List YOY", "slug": "avg_sale_to_list_yoy"},
    {"column": "homes_sold_yoy", "label": "Homes Sold YOY", "slug": "homes_sold_yoy"},
    {"column": "inventory_yoy", "label": "Inventory YOY", "slug": "inventory_yoy"},
    {"column": "new_listings_yoy", "label": "New Listings YOY", "slug": "new_listings_yoy"},
    {"column": "median_dom_yoy", "label": "Median Days on Market YOY", "slug": "median_dom_yoy"},
]


def build_affected_unaffected_windows(events: pd.DataFrame, housing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [metric["column"] for metric in affected_vs_unaffected_metrics]
    all_fips = set(housing["fips"].dropna().unique())
    event_groups = events.groupby(["event_source", "source_event_id", "event_start_month", "event_end_month"], dropna=False)
    for event_id, group in event_groups:
        affected_fips = set(group["fips"].dropna().unique())
        unaffected_fips = all_fips - affected_fips
        representative = group.iloc[0]
        start_month = representative["event_start_month"]
        end_month = representative["event_end_month"]
        start_window = start_month - pd.DateOffset(months=PRE_EVENT_MONTHS)
        end_window = end_month + pd.DateOffset(months=POST_EVENT_MONTHS)
        aggregate_event_key = "|".join(str(part) for part in event_id)

        for status, fips_values in [("affected", affected_fips), ("unaffected", unaffected_fips)]:
            window = housing.loc[
                housing["fips"].isin(fips_values)
                & housing["period_month"].between(start_window, end_window)
            ].copy()
            if window.empty:
                continue
            window["months_from_event_start"] = (
                (window["period_month"].dt.year - start_month.year) * 12
                + (window["period_month"].dt.month - start_month.month)
            )
            window["months_after_event_end"] = (
                (window["period_month"].dt.year - end_month.year) * 12
                + (window["period_month"].dt.month - end_month.month)
            )
            pre_mask = window["months_from_event_start"].between(-PRE_EVENT_MONTHS, -1)
            post_mask = window["months_after_event_end"].between(1, POST_EVENT_MONTHS)
            window = window.loc[pre_mask | post_mask].copy()
            if window.empty:
                continue
            window["event_window_month"] = np.where(
                window["months_from_event_start"].lt(0),
                window["months_from_event_start"],
                window["months_after_event_end"],
            )
            window["event_window_phase"] = np.select(
                [
                    window["event_window_month"].between(-PRE_EVENT_MONTHS, -1),
                    window["event_window_month"].between(1, 12),
                    window["event_window_month"].between(13, POST_EVENT_MONTHS),
                ],
                ["12 months before event start", "1-12 months after event end", "13-24 months after event end"],
                default=pd.NA,
            )

            for metric in metrics:
                complete_window = filter_complete_lines(
                    window,
                    x_col="event_window_month",
                    line_col="fips",
                    metric_col=metric,
                    required_x_values=EVENT_WINDOW_MONTHS,
                )
                if complete_window.empty:
                    continue
                monthly = (
                    complete_window
                    .groupby(["event_window_month", "event_window_phase"], as_index=False)[metric]
                    .median()
                    .rename(columns={metric: "metric_value"})
                )
                monthly["metric_name"] = metric
                monthly["complete_county_count"] = complete_window["fips"].nunique()
                monthly["event_key"] = aggregate_event_key
                monthly["affected_status"] = status
                monthly["event_source"] = representative["event_source"]
                monthly["source_event_id"] = representative["source_event_id"]
                monthly["event_start_month"] = start_month
                monthly["event_end_month"] = end_month
                monthly["line_id"] = aggregate_event_key + ":" + status + ":" + metric
                rows.append(monthly)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


comparison_windows = build_affected_unaffected_windows(events, housing)
display(comparison_windows["affected_status"].value_counts(dropna=False).to_frame("rows"))
display(comparison_windows[["event_key", "affected_status"]].drop_duplicates().groupby("affected_status").size().to_frame("event_groups"))
display(comparison_windows.groupby(["metric_name", "affected_status"])["complete_county_count"].describe())
"""
    ),
    md("## 4. Housing Market Movement Around Events: Affected VS Unaffected Counties"),
    code(
        r"""
for metric in affected_vs_unaffected_metrics:
    plot_county_lines_with_iqr(
        comparison_windows.loc[comparison_windows["metric_name"].eq(metric["column"])],
        metric_col="metric_value",
        x_col="event_window_month",
        line_col="line_id",
        group_col="affected_status",
        title=f"{metric['label']} event-window response: affected vs unaffected counties",
        xlabel="Relative event-window month",
        ylabel=metric["label"],
        output_path=OUTPUT_DIR / f"{metric['slug']}_event_window_affected_vs_unaffected.png",
        phase_bands=True,
        required_x_values=EVENT_WINDOW_MONTHS,
    )
    plt.show()
"""
    ),
]


def main() -> None:
    NOTEBOOK_PATH.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
