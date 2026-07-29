"""Build grouped county event-window EDA notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = ROOT / "scripts" / "eda"
NOTEBOOK_PATH = EDA_DIR / "county_event_window_grouped_eda.ipynb"


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
        """# County Event-Window Grouped EDA

Dataset: `mart.redfin_county_monthly`, `mart.fema_disaster_declarations`, `mart.noaa_storm_events`, `mart.nri_county_risk`, and `mart.fema_disaster_financial_assistance` in `data/quoll.duckdb`.

This notebook plots affected county-event windows for `median_ppsf_yoy` and `housing_market_index`. Windows use months `-12` through `0` relative to the event start date, plus post-event horizons measured after the event end date."""
    ),
    md("## Setup"),
    code(
        r"""
from pathlib import Path
import sys

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", 160)
pd.set_option("display.max_rows", 100)
sns.set_theme(style="whitegrid")

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from housing_climate_risk.page_data.event_windows import (
    build_affected_event_windows,
    event_window_months,
    filter_complete_event_window_lines,
    load_disaster_events,
    load_redfin_county_monthly,
)

DB_PATH = ROOT / "data" / "quoll.duckdb"
OUTPUT_DIR = ROOT / "output" / "eda" / "county_event_window_grouped"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROPERTY_TYPE = "All Residential"
NOAA_DAMAGE_THRESHOLD = 1_000_000_000
PRE_EVENT_MONTHS = 12
POST_EVENT_HORIZONS = [12, 24, 36, 48, 60]
MAX_POST_EVENT_MONTHS = max(POST_EVENT_HORIZONS)
MIN_GROUP_LINE_COUNT = 10

METRICS = [
    {"column": "median_ppsf_yoy", "label": "Median PPSF YOY", "slug": "median_ppsf_yoy"},
    {"column": "housing_market_index", "label": "Housing Market Index", "slug": "housing_market_index"},
]

RISK_RATING_ORDER = ["Very Low", "Relatively Low", "Relatively Moderate", "Relatively High", "Very High"]

con = duckdb.connect(str(DB_PATH), read_only=True)
DB_PATH
"""
    ),
    md("## Load Housing, Events, And NRI Attributes"),
    code(
        r"""
housing = load_redfin_county_monthly(con, property_type=PROPERTY_TYPE)
events = load_disaster_events(
    con,
    noaa_damage_threshold=NOAA_DAMAGE_THRESHOLD,
    include_fema=True,
    include_noaa=True,
)

eligible_start_month = housing["period_month"].min() + pd.DateOffset(months=PRE_EVENT_MONTHS)
eligible_end_month = housing["period_month"].max() - pd.DateOffset(months=MAX_POST_EVENT_MONTHS)
housing_fips = set(housing["fips"].dropna().unique())
events = events.loc[
    events["fips"].isin(housing_fips)
    & events["event_start_month"].between(eligible_start_month, eligible_end_month)
].copy()

nri = con.execute(
    '''
    SELECT *
    FROM mart.nri_county_risk
    WHERE fips IS NOT NULL
    '''
).df()
nri["fips"] = nri["fips"].astype(str).str.zfill(5)

display(housing[["fips", "period_month", "median_ppsf_yoy", "housing_market_index"]].head())
display(events[["event_source", "source_event_id", "fips", "event_type", "event_start_month", "event_end_month"]].head())
display(
    pd.DataFrame(
        {
            "housing_counties": [housing["fips"].nunique()],
            "housing_months": [housing["period_month"].nunique()],
            "county_events": [events["event_key"].nunique()],
            "nri_counties": [nri["fips"].nunique()],
            "eligible_event_start_min": [eligible_start_month],
            "eligible_event_start_max": [eligible_end_month],
        }
    )
)
"""
    ),
    md("## Build Event Windows"),
    code(
        r"""
affected_windows = build_affected_event_windows(
    events,
    housing,
    pre_event_months=PRE_EVENT_MONTHS,
    post_event_months=MAX_POST_EVENT_MONTHS,
)

affected_windows = affected_windows.merge(
    nri[["fips", "risk_rating"]],
    on="fips",
    how="left",
)
affected_windows["risk_rating"] = pd.Categorical(
    affected_windows["risk_rating"],
    categories=RISK_RATING_ORDER,
    ordered=True,
)

display(affected_windows.head())
display(
    affected_windows[["event_key", "fips", "risk_rating"]]
    .drop_duplicates()
    .groupby("risk_rating", observed=True)
    .size()
    .to_frame("county_events")
)
"""
    ),
    md("## Plot Helpers"),
    code(
        r"""
def horizon_label(post_event_months: int) -> str:
    years = post_event_months // 12
    return f"{years} year{'s' if years != 1 else ''} after event end"


def window_for_horizon(frame: pd.DataFrame, post_event_months: int) -> pd.DataFrame:
    keep_months = event_window_months(PRE_EVENT_MONTHS, post_event_months)
    return frame.loc[frame["event_window_month"].isin(keep_months)].copy()


def summarize_by_group(
    frame: pd.DataFrame,
    *,
    metric_col: str,
    group_col: str,
    post_event_months: int,
    group_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_months = event_window_months(PRE_EVENT_MONTHS, post_event_months)
    scoped = window_for_horizon(frame, post_event_months)
    scoped = scoped.dropna(subset=[metric_col, group_col, "line_id", "event_window_month"]).copy()
    if scoped.empty:
        return scoped.iloc[0:0].copy(), scoped.iloc[0:0].copy()

    complete_parts = []
    for _, group in scoped.groupby(group_col, observed=True):
        complete = filter_complete_event_window_lines(
            group,
            x_col="event_window_month",
            line_col="line_id",
            metric_col=metric_col,
            required_x_values=keep_months,
        )
        if not complete.empty:
            complete_parts.append(complete)
    if not complete_parts:
        return scoped.iloc[0:0].copy(), scoped.iloc[0:0].copy()

    complete = pd.concat(complete_parts, ignore_index=True)
    line_counts = complete.groupby(group_col, observed=True)["line_id"].nunique()
    keep_groups = line_counts.loc[line_counts >= MIN_GROUP_LINE_COUNT].index
    complete = complete.loc[complete[group_col].isin(keep_groups)].copy()
    if complete.empty:
        return complete, complete.iloc[0:0].copy()

    summary = (
        complete.groupby([group_col, "event_window_month"], observed=True)[metric_col]
        .quantile([0.25, 0.50, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "q25", 0.50: "median", 0.75: "q75"})
    )
    if group_order is not None:
        summary[group_col] = pd.Categorical(summary[group_col], categories=group_order, ordered=True)
        complete[group_col] = pd.Categorical(complete[group_col], categories=group_order, ordered=True)
        summary = summary.sort_values([group_col, "event_window_month"])
    return complete, summary


def plot_grouped_event_window(
    frame: pd.DataFrame,
    *,
    metric_col: str,
    metric_label: str,
    group_col: str,
    group_label: str,
    post_event_months: int,
    output_slug: str,
    group_order: list[str] | None = None,
) -> None:
    complete, summary = summarize_by_group(
        frame,
        metric_col=metric_col,
        group_col=group_col,
        post_event_months=post_event_months,
        group_order=group_order,
    )
    if summary.empty:
        print(f"No complete windows for {metric_label}, {group_label}, {horizon_label(post_event_months)}")
        return

    groups = [g for g in summary[group_col].dropna().unique()]
    palette = sns.color_palette("tab10", n_colors=max(len(groups), 1))
    colors = dict(zip(groups, palette))

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axvspan(-12.5, 0.5, color="#4e79a7", alpha=0.08, label="12 months before through event start")
    ax.axvspan(0.5, post_event_months + 0.5, color="#59a14f", alpha=0.08, label=f"1-{post_event_months} months after event end")
    ax.axvline(0, color="#344054", linewidth=1.0, alpha=0.65)
    ax.axhline(0, color="#344054", linewidth=0.8, alpha=0.45)

    for group_name in groups:
        stats = summary.loc[summary[group_col].eq(group_name)].sort_values("event_window_month")
        color = colors[group_name]
        ax.fill_between(stats["event_window_month"], stats["q25"], stats["q75"], color=color, alpha=0.13)
        line_count = complete.loc[complete[group_col].eq(group_name), "line_id"].nunique()
        ax.plot(
            stats["event_window_month"],
            stats["median"],
            color=color,
            linewidth=2.4,
            label=f"{group_name} median (n={line_count:,})",
        )

    tick_values = sorted(set([-12, -9, -6, -3, 0, 1, 6, 12, post_event_months]))
    if post_event_months >= 24:
        tick_values.extend(range(24, post_event_months + 1, 12))
    tick_values = sorted(set(v for v in tick_values if -12 <= v <= post_event_months))
    ax.set_xlim(-12.5, post_event_months + 0.5)
    ax.set_xticks(tick_values)
    ax.set_title(f"{metric_label} by {group_label}: {horizon_label(post_event_months)}")
    ax.set_xlabel("Event-window month: negative months before start, positive months after end")
    ax.set_ylabel(metric_label)
    ax.legend(loc="best", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = OUTPUT_DIR / f"{output_slug}_{metric_col}_{post_event_months:02d}mo.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
"""
    ),
    md("## 1. Event Windows Grouped By NRI Risk Rating"),
    code(
        r"""
risk_windows = affected_windows.dropna(subset=["risk_rating"]).copy()

for post_event_months in POST_EVENT_HORIZONS:
    for metric in METRICS:
        plot_grouped_event_window(
            risk_windows,
            metric_col=metric["column"],
            metric_label=metric["label"],
            group_col="risk_rating",
            group_label="NRI risk rating",
            post_event_months=post_event_months,
            output_slug="nri_risk_rating",
            group_order=RISK_RATING_ORDER,
        )
"""
    ),
    md("## 2. Event Windows Grouped By Actual Event Type"),
    code(
        r"""
event_type_windows = affected_windows.dropna(subset=["event_type"]).copy()
event_type_windows["actual_event_type"] = event_type_windows["event_type"].astype(str).str.strip()
event_type_windows = event_type_windows.loc[event_type_windows["actual_event_type"].ne("")]

top_event_types = (
    event_type_windows[["event_key", "actual_event_type"]]
    .drop_duplicates()
    ["actual_event_type"]
    .value_counts()
    .head(5)
    .index
    .tolist()
)
event_type_windows = event_type_windows.loc[event_type_windows["actual_event_type"].isin(top_event_types)].copy()

display(pd.Series(top_event_types, name="top_5_actual_event_types").to_frame())
display(
    event_type_windows[["event_key", "actual_event_type"]]
    .drop_duplicates()
    .groupby("actual_event_type")
    .size()
    .sort_values(ascending=False)
    .to_frame("county_events")
)

for post_event_months in POST_EVENT_HORIZONS:
    for metric in METRICS:
        plot_grouped_event_window(
            event_type_windows,
            metric_col=metric["column"],
            metric_label=metric["label"],
            group_col="actual_event_type",
            group_label="actual event type",
            post_event_months=post_event_months,
            output_slug="actual_event_type",
            group_order=top_event_types,
        )
"""
    ),
    md("## 3. Event Windows Grouped By County Event Frequency In Event Year"),
    code(
        r"""
events_for_frequency = events.copy()
events_for_frequency["event_year"] = pd.to_datetime(events_for_frequency["event_start_month"]).dt.year
event_frequency = (
    events_for_frequency.groupby(["fips", "event_year"])["event_key"]
    .nunique()
    .reset_index(name="county_event_frequency")
)

event_year_lookup = events_for_frequency[["event_key", "event_year"]].drop_duplicates()
frequency_windows = affected_windows.merge(event_year_lookup, on="event_key", how="left").merge(
    event_frequency,
    on=["fips", "event_year"],
    how="left",
)
frequency_windows["event_frequency_group"] = frequency_windows["county_event_frequency"].map(
    lambda value: "5+" if pd.notna(value) and value >= 5 else str(int(value)) if pd.notna(value) else pd.NA
)
frequency_order = ["1", "2", "3", "4", "5+"]
frequency_windows["event_frequency_group"] = pd.Categorical(
    frequency_windows["event_frequency_group"],
    categories=frequency_order,
    ordered=True,
)

display(
    frequency_windows[["event_key", "fips", "event_year", "event_frequency_group"]]
    .drop_duplicates()
    .groupby("event_frequency_group", observed=True)
    .size()
    .to_frame("county_events")
)

for post_event_months in POST_EVENT_HORIZONS:
    for metric in METRICS:
        plot_grouped_event_window(
            frequency_windows,
            metric_col=metric["column"],
            metric_label=metric["label"],
            group_col="event_frequency_group",
            group_label="county event frequency in event year",
            post_event_months=post_event_months,
            output_slug="event_frequency",
            group_order=frequency_order,
        )
"""
    ),
    md("## 4. Event Windows Grouped By FEMA Assistance Or NOAA Damage Amount"),
    code(
        r"""
ASSISTANCE_BIN_ORDER = [
    "$0",
    "$1 to <$100K",
    "$100K to <$1M",
    "$1M to <$10M",
    "$10M to <$100M",
    "$100M to <$1B",
    "$1B+",
]

fema_assistance = con.execute(
    '''
    SELECT
        disaster_number,
        ihp_approved_amount,
        public_assistance_obligated_amount,
        hazard_mitigation_grant_obligated_amount,
        total_fema_financial_assistance_amount
    FROM mart.fema_disaster_financial_assistance
    WHERE disaster_number IS NOT NULL
    '''
).df()

fema_assistance["disaster_number"] = pd.to_numeric(fema_assistance["disaster_number"], errors="coerce").astype("Int64")

assistance_windows = affected_windows.loc[affected_windows["event_source"].isin(["fema", "noaa"])].copy()
event_amount_lookup = events[["event_key", "total_damage_amount"]].drop_duplicates()
assistance_windows = assistance_windows.merge(event_amount_lookup, on="event_key", how="left")
assistance_windows["disaster_number"] = pd.to_numeric(
    assistance_windows["source_event_id"].where(assistance_windows["event_source"].eq("fema")),
    errors="coerce",
).astype("Int64")
assistance_windows = assistance_windows.merge(fema_assistance, on="disaster_number", how="left")

assistance_windows["financial_group_amount"] = np.where(
    assistance_windows["event_source"].eq("noaa"),
    assistance_windows["total_damage_amount"],
    assistance_windows["total_fema_financial_assistance_amount"],
)
assistance_windows = assistance_windows.dropna(subset=["financial_group_amount"]).copy()
amount = assistance_windows["financial_group_amount"]
assistance_windows["assistance_bin"] = np.select(
    [
        amount.eq(0),
        amount.lt(100_000),
        amount.lt(1_000_000),
        amount.lt(10_000_000),
        amount.lt(100_000_000),
        amount.lt(1_000_000_000),
    ],
    [
        "$0",
        "$1 to <$100K",
        "$100K to <$1M",
        "$1M to <$10M",
        "$10M to <$100M",
        "$100M to <$1B",
    ],
    default="$1B+",
)
assistance_windows["assistance_bin"] = pd.Categorical(
    assistance_windows["assistance_bin"],
    categories=ASSISTANCE_BIN_ORDER,
    ordered=True,
)

assistance_event_summary = (
    assistance_windows[[
        "event_key",
        "event_source",
        "disaster_number",
        "assistance_bin",
        "financial_group_amount",
    ]]
    .drop_duplicates()
    .groupby("assistance_bin", observed=True)
    .agg(
        county_events=("event_key", "nunique"),
        disasters=("disaster_number", "nunique"),
        fema_county_events=("event_source", lambda values: values.eq("fema").sum()),
        noaa_county_events=("event_source", lambda values: values.eq("noaa").sum()),
        min_group_amount=("financial_group_amount", "min"),
        median_group_amount=("financial_group_amount", "median"),
        max_group_amount=("financial_group_amount", "max"),
    )
)
display(assistance_event_summary)

for post_event_months in POST_EVENT_HORIZONS:
    for metric in METRICS:
        plot_grouped_event_window(
            assistance_windows,
            metric_col=metric["column"],
            metric_label=metric["label"],
            group_col="assistance_bin",
            group_label="FEMA assistance or NOAA damage amount",
            post_event_months=post_event_months,
            output_slug="fema_financial_assistance",
            group_order=ASSISTANCE_BIN_ORDER,
        )
"""
    ),
]


NOTEBOOK_PATH.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
