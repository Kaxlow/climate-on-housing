from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from housing_climate_risk.page_data.event_windows import (
    build_affected_event_windows,
    event_window_months,
    filter_complete_event_window_lines,
    load_disaster_events,
    load_redfin_county_monthly,
)


ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "quoll.duckdb"
COUNTIES_PATH = ROOT / "data" / "fipsgeo" / "us_counties_boundaries_shapefile.json"
OUT_PATH = ROOT / "output" / "climate-risk-housing.html"
COUNTY_HISTORY_OUT_PATH = ROOT / "output" / "climate-risk-housing-county-history.js"
PLAYBOOK_OUT_PATH = ROOT / "output" / "climate-risk-housing-playbook.js"

RISK_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]
RISK_MAP = {
    "Very Low": "Very Low",
    "Relatively Low": "Low",
    "Relatively Moderate": "Medium",
    "Moderate": "Medium",
    "Relatively High": "High",
    "Very High": "Very High",
}
RISK_NUMERIC = {rating: index + 1 for index, rating in enumerate(RISK_ORDER)}
HAZARDS = [
    {"key": "overall", "label": "Overall NRI", "score": "risk_score", "rating": "risk_rating"},
    {"key": "river_flood", "label": "River Flood", "score": "IFLD_RISKS", "rating": "IFLD_RISKR"},
    {"key": "tornado", "label": "Tornado", "score": "TRND_RISKS", "rating": "TRND_RISKR"},
    {"key": "wildfire", "label": "Wildfire", "score": "WFIR_RISKS", "rating": "WFIR_RISKR"},
    {"key": "hail", "label": "Hail", "score": "HAIL_RISKS", "rating": "HAIL_RISKR"},
    {"key": "earthquake", "label": "Earthquake", "score": "ERQK_RISKS", "rating": "ERQK_RISKR"},
]

def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def serialize_number(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def rating_clean(value: object) -> str | None:
    if pd.isna(value):
        return None
    return RISK_MAP.get(str(value), str(value))


def feature_bucket_labels(values: pd.Series, fmt: str, count: int = 5) -> tuple[pd.Series, list[str]]:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=values.index, dtype="object"), []
    edges = valid.quantile(np.linspace(0, 1, count + 1)).to_numpy()
    edges = np.unique(edges)
    if len(edges) < 3:
        label = f"All values ({format_bucket_value(valid.median(), fmt)})"
        return pd.Series(label, index=values.index, dtype="object").where(values.notna()), [label]
    labels = [f"B{i + 1}: {format_bucket_value(edges[i], fmt)} to {format_bucket_value(edges[i + 1], fmt)}" for i in range(len(edges) - 1)]
    buckets = pd.cut(values, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    return buckets.astype("object"), labels


def format_bucket_value(value: float, fmt: str) -> str:
    if pd.isna(value):
        return "n/a"
    if fmt == "currency":
        return f"${value:,.0f}"
    if fmt == "percent":
        return f"{value:,.2f}%"
    if fmt == "pct":
        return f"{value * 100:,.2f}%"
    if fmt == "signed_pct":
        return f"{value * 100:+,.2f}%"
    if fmt == "temperature_f":
        return f"{value:,.1f} F"
    if fmt == "inches":
        return f"{value:,.2f} in"
    return f"{value:,.1f}"


def classify_bucket_position(bucket_order: int | None, bucket_count: int, corr: object) -> str:
    if bucket_order is None or not bucket_count or pd.isna(corr) or float(corr) == 0:
        return "neutral"
    midpoint = (bucket_count - 1) / 2
    if bucket_order == midpoint:
        return "neutral"
    higher_bin = bucket_order > midpoint
    positive_corr = float(corr) > 0
    if higher_bin == positive_corr:
        return "higher"
    return "lower"


def weighted_bucket_average(frame: pd.DataFrame, buckets: list[tuple[str, float]], *, zero_cols: list[str] | None = None) -> pd.Series:
    total = pd.Series(0.0, index=frame.index)
    weighted = pd.Series(0.0, index=frame.index)
    for column in zero_cols or []:
        if column in frame:
            total = total.add(pd.to_numeric(frame[column], errors="coerce").fillna(0), fill_value=0)
    for column, midpoint in buckets:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
            total = total.add(values, fill_value=0)
            weighted = weighted.add(values * midpoint, fill_value=0)
    return weighted.where(total > 0) / total.where(total > 0)


def mean_available(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index)
    return frame[available].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def build_price_risk(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    hazard_cols: list[str] = []
    for hazard in HAZARDS:
        if hazard["key"] == "overall":
            hazard_cols.extend(["risk_score", "risk_rating"])
        else:
            hazard_cols.extend([hazard["score"], hazard["rating"]])
    nri = con.execute(
        f"""
        SELECT fips, COUNTY, STATEABBRV, {", ".join(hazard_cols)}
        FROM mart.nri_county_risk
        WHERE fips IS NOT NULL
        """
    ).df()
    ppsf = con.execute(
        """
        SELECT
            fips,
            any_value(REGION) AS county_label,
            any_value(STATE_CODE) AS state_code,
            avg(CASE
                WHEN try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL
                ELSE try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE)
            END) AS avg_median_ppsf_yoy,
            count(*) FILTER (
                WHERE try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) IS NOT NULL
                  AND try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) > -888888000
            ) AS observed_months
        FROM mart.redfin_county_monthly
        WHERE property_type = 'All Residential'
          AND period_begin >= DATE '2025-01-01'
          AND period_begin < DATE '2026-01-01'
          AND fips IS NOT NULL
        GROUP BY fips
        """
    ).df()
    df = ppsf.merge(nri, on="fips", how="inner")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["avg_median_ppsf_yoy"] = pd.to_numeric(df["avg_median_ppsf_yoy"], errors="coerce")
    for hazard in HAZARDS:
        df[hazard["score"]] = clean_numeric(df[hazard["score"]])
        df[hazard["rating"]] = df[hazard["rating"]].map(rating_clean)
    df = df.dropna(subset=["avg_median_ppsf_yoy"]).copy()
    df["risk_rating_clean"] = df["risk_rating"].map(rating_clean)
    cap_lower = df["avg_median_ppsf_yoy"].quantile(0.01)
    cap_upper = df["avg_median_ppsf_yoy"].quantile(0.99)
    counties = []
    for row in df.itertuples(index=False):
        hazards = {}
        for hazard in HAZARDS:
            rating = getattr(row, hazard["rating"])
            hazards[hazard["key"]] = {
                "score": serialize_number(getattr(row, hazard["score"]), 3),
                "rating": rating,
                "ratingValue": RISK_NUMERIC.get(rating),
            }
        counties.append(
            {
                "fips": row.fips,
                "county": row.county_label if pd.notna(row.county_label) else f"{row.COUNTY}, {row.STATEABBRV}",
                "state": row.state_code if pd.notna(row.state_code) else row.STATEABBRV,
                "hazards": hazards,
            }
        )
    history = con.execute(
        """
        WITH monthly AS (
            SELECT
                r.fips,
                date_trunc('month', r.period_begin)::DATE AS month,
                any_value(r.REGION) AS county_label,
                any_value(r.STATE_CODE) AS state_code,
                avg(CASE
                    WHEN try_cast(replace(r.MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL
                    ELSE try_cast(replace(r.MEDIAN_PPSF_YOY, ',', '') AS DOUBLE)
                END) AS median_ppsf_yoy
            FROM mart.redfin_county_monthly AS r
            WHERE r.property_type = 'All Residential'
              AND r.period_begin >= DATE '2016-01-01'
              AND r.period_begin < DATE '2026-01-01'
              AND r.fips IS NOT NULL
            GROUP BY r.fips, date_trunc('month', r.period_begin)
        ),
        complete_counties AS (
            SELECT fips
            FROM monthly
            GROUP BY fips
            HAVING count(*) = 120
               AND count(median_ppsf_yoy) = 120
        )
        SELECT
            monthly.*
        FROM monthly
        INNER JOIN complete_counties USING (fips)
        ORDER BY fips, month
        """
    ).df()
    history["fips"] = history["fips"].astype(str).str.zfill(5)
    history["median_ppsf_yoy"] = pd.to_numeric(history["median_ppsf_yoy"], errors="coerce")

    # Merge all hazard columns to enable hazard-specific filtering
    history = history.merge(nri[["fips"] + hazard_cols], on="fips", how="left")
    history["riskRating"] = history["risk_rating"].map(rating_clean)

    # Clean hazard columns
    for hazard in HAZARDS:
        if hazard["key"] != "overall":
            history[hazard["rating"]] = history[hazard["rating"]].map(rating_clean)

    history = history.dropna(subset=["median_ppsf_yoy", "riskRating"]).copy()

    # Create hazard-specific histories for each hazard type
    rating_histories = {}
    for hazard in HAZARDS:
        hazard_key = hazard["key"]
        if hazard_key == "overall":
            hazard_history = history.copy()
            group_col = "riskRating"
        else:
            # Group by the hazard-specific rating, not overall
            rating_col = hazard["rating"]
            hazard_history = history[history[rating_col].notna()].copy()
            hazard_history["_hazard_rating"] = hazard_history[rating_col]
            group_col = "_hazard_rating"

        grouped = (
            hazard_history.groupby([group_col, "month"], observed=False)["median_ppsf_yoy"]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={group_col: "riskRating", 0.25: "q1", 0.5: "median", 0.75: "q3"})
        )
        rating_histories[hazard_key] = [
            {
                "riskRating": row.riskRating,
                "month": row.month.strftime("%Y-%m-%d"),
                "q1": serialize_number(row.q1, 5),
                "median": serialize_number(row.median, 5),
                "q3": serialize_number(row.q3, 5),
            }
            for row in grouped.itertuples(index=False)
            if row.riskRating in RISK_ORDER
        ]
    # Store the dense county histories as shared months plus one value array per
    # county. Repeating county and hazard metadata for every month made the
    # standalone HTML substantially larger and slower to parse.
    history_months = sorted(history["month"].dropna().unique())
    history_month_labels = [pd.Timestamp(month).strftime("%Y-%m-%d") for month in history_months]
    county_history_series = []
    for fips, county_history in history.groupby("fips", sort=False):
        county_history = county_history.set_index("month").reindex(history_months)
        first = county_history.iloc[0]
        county_history_series.append(
            {
                "fips": fips,
                "county": first.county_label,
                "state": first.state_code,
                "values": [serialize_number(value, 5) for value in county_history["median_ppsf_yoy"]],
            }
        )
    history_cap_lower = history["median_ppsf_yoy"].quantile(0.01)
    history_cap_upper = history["median_ppsf_yoy"].quantile(0.99)

    return {
        "hazards": [{"key": h["key"], "label": h["label"]} for h in HAZARDS],
        "counties": counties,
        "countyHistoryMonths": history_month_labels,
        "countyHistorySeries": county_history_series,
        "ratingHistoriesByHazard": rating_histories,
        "summary": {
            "countyCount": int(df["fips"].nunique()),
            "medianAvgPpsfYoy": serialize_number(df["avg_median_ppsf_yoy"].median(), 5),
            "ppsfCapLower": serialize_number(cap_lower, 5),
            "ppsfCapUpper": serialize_number(cap_upper, 5),
            "historyPpsfCapLower": serialize_number(history_cap_lower, 5),
            "historyPpsfCapUpper": serialize_number(history_cap_upper, 5),
        },
    }


def build_geojson(fips_set: set[str]) -> dict[str, object]:
    from shapely.geometry import mapping, shape
    from shapely.geometry.multipolygon import MultiPolygon
    from shapely.geometry.polygon import Polygon, orient

    raw = json.loads(COUNTIES_PATH.read_text(encoding="utf-8"))
    features = []
    for feature in raw["features"]:
        props = feature.get("properties", {})
        fips = str(props.get("GEOID") or props.get("GEOID10") or "").zfill(5)
        if fips in fips_set:
            tolerance = 0.08 if fips.startswith("02") else 0.025
            geometry = shape(feature["geometry"]).simplify(tolerance, preserve_topology=True)
            if not geometry.is_empty:
                if isinstance(geometry, Polygon):
                    geometry = orient(geometry, sign=-1.0)
                elif isinstance(geometry, MultiPolygon):
                    geometry = MultiPolygon([orient(part, sign=-1.0) for part in geometry.geoms])
                features.append({"type": "Feature", "properties": {"fips": fips}, "geometry": mapping(geometry)})
    return {"type": "FeatureCollection", "features": features}


def aggregate_lines(frame: pd.DataFrame, group_cols: list[str], metric: str, annual: bool = False) -> list[dict[str, object]]:
    if frame.empty:
        return []

    if annual:
        # For annual data, convert event_window_month to event_window_year
        frame_copy = frame.copy()
        frame_copy["event_window_year"] = (frame_copy["event_window_month"] / 12).round().astype(int)
        q = (
            frame_copy.dropna(subset=[metric, "event_window_year"])
            .groupby(group_cols + ["event_window_year"], observed=False)[metric]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
        )
        return [
            {
                **{col: getattr(row, col) for col in group_cols},
                "month": int(row.event_window_year * 12),  # Convert back to months for consistency
                "q1": serialize_number(row.q1, 5),
                "median": serialize_number(row.median, 5),
                "q3": serialize_number(row.q3, 5),
            }
            for row in q.itertuples(index=False)
        ]
    else:
        q = (
            frame.dropna(subset=[metric, "event_window_month"])
            .groupby(group_cols + ["event_window_month"], observed=False)[metric]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
        )
        return [
            {
                **{col: getattr(row, col) for col in group_cols},
                "month": int(row.event_window_month),
                "q1": serialize_number(row.q1, 5),
                "median": serialize_number(row.median, 5),
                "q3": serialize_number(row.q3, 5),
            }
            for row in q.itertuples(index=False)
        ]


def _build_window_data(
    affected: pd.DataFrame,
    nri: pd.DataFrame,
    metric: str,
    *,
    pre_months: int,
    post_months: int,
    anchor_col: str = "event_window_month",
    sample_per_group: int = 2,
    eligible_feature_fips_by_risk: dict[str, set[str]] | None = None,
) -> dict[str, object]:
    """Build by-rating aggregates + example lines for one event-window definition.

    Both time-window frames use the same raw ``affected`` dataframe but filter
    to counties that have complete monthly data across the frame's required months.
    """
    required = event_window_months(pre_months, post_months)
    complete = filter_complete_event_window_lines(
        affected,
        x_col=anchor_col,
        line_col="line_id",
        metric_col=metric,
        required_x_values=required,
    ).copy()
    complete = complete.loc[complete[anchor_col].isin(required)].copy()
    complete = complete.merge(nri[["fips", "riskRating"]], on="fips", how="left")
    complete_for_agg = complete.copy()
    if anchor_col != "event_window_month":
        complete_for_agg["event_window_month"] = complete_for_agg[anchor_col]
    by_rating = aggregate_lines(complete_for_agg.dropna(subset=["riskRating"]), ["riskRating"], metric)
    affected_counties = (
        complete.dropna(subset=["riskRating"])[["fips", "riskRating"]]
        .drop_duplicates()
        .groupby(["fips", "riskRating"], as_index=False)
        .size()
    )
    risk_counts = complete.drop_duplicates(["line_id", "riskRating"]).groupby("riskRating", dropna=True)["line_id"].nunique()

    # Compute per-county-event average metric over the window, then percentile within risk group.
    line_avg = (
        complete.dropna(subset=[metric, "riskRating"])
        .groupby(["line_id", "fips", "county_label", "state_code", "riskRating"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "avg_metric"})
    )
    line_avg["pct_rank"] = (
        line_avg.groupby("riskRating")["avg_metric"]
        .rank(method="average", pct=True, na_option="keep")
        .mul(100)
        .round(1)
    )

    bands = pd.DataFrame(by_rating)
    eligible_line_ids: set[str] = set()
    distance_join = pd.DataFrame()
    line_distance = pd.DataFrame(
        columns=[
            "line_id",
            "mean_line_gap",
            "mean_abs_line_gap",
            "median_line_gap",
            "median_standardized_gap",
            "mean_abs_standardized_gap",
            "above_median_share",
            "median_iqr_width",
            "distance_threshold",
            "significantly_separated",
            "directionally_consistent",
        ]
    )
    if not bands.empty:
        bands["iqr_width"] = bands["q3"] - bands["q1"]
        max_width_by_risk = bands.groupby("riskRating")["iqr_width"].max().to_dict()
        bands["lower_allowed"] = bands.apply(
            lambda row: row["q1"] - 0.5 * max_width_by_risk.get(row["riskRating"], np.nan),
            axis=1,
        )
        bands["upper_allowed"] = bands.apply(
            lambda row: row["q3"] + 0.5 * max_width_by_risk.get(row["riskRating"], np.nan),
            axis=1,
        )
        band_join = complete.dropna(subset=[metric, "riskRating"]).merge(
            bands[["riskRating", "month", "lower_allowed", "upper_allowed"]],
            left_on=["riskRating", anchor_col],
            right_on=["riskRating", "month"],
            how="inner",
        )
        band_join["inside_sample_band"] = band_join[metric].between(
            band_join["lower_allowed"],
            band_join["upper_allowed"],
            inclusive="both",
        )
        line_band_fit = (
            band_join.groupby("line_id", as_index=False)
            .agg(months=(anchor_col, "nunique"), all_inside=("inside_sample_band", "all"))
        )
        eligible_line_ids = set(
            line_band_fit.loc[
                line_band_fit["months"].eq(len(required)) & line_band_fit["all_inside"],
                "line_id",
            ].astype(str)
        )

        distance_join = complete.dropna(subset=[metric, "riskRating"]).merge(
            bands[["riskRating", "month", "median", "iqr_width"]],
            left_on=["riskRating", anchor_col],
            right_on=["riskRating", "month"],
            how="inner",
        )
        distance_join["line_gap"] = distance_join[metric] - distance_join["median"]
        distance_join["safe_iqr_width"] = distance_join["iqr_width"].abs().clip(lower=0.01)
        distance_join["standardized_line_gap"] = (
            distance_join["line_gap"] / distance_join["safe_iqr_width"]
        )
        distance_join["above_group_median"] = distance_join["line_gap"].gt(0)
        line_distance = (
            distance_join.groupby("line_id", as_index=False)
            .agg(
                mean_line_gap=("line_gap", "mean"),
                mean_abs_line_gap=("line_gap", lambda values: values.abs().mean()),
                median_line_gap=("line_gap", "median"),
                median_standardized_gap=("standardized_line_gap", "median"),
                mean_abs_standardized_gap=("standardized_line_gap", lambda values: values.abs().mean()),
                above_median_share=("above_group_median", "mean"),
                median_iqr_width=("iqr_width", "median"),
            )
        )
        line_distance["distance_threshold"] = np.maximum(
            0.01,
            line_distance["median_iqr_width"].fillna(0).mul(0.5),
        )
        line_distance["significantly_separated"] = line_distance["median_standardized_gap"].abs().ge(0.5)
        line_distance["directionally_consistent"] = (
            line_distance["above_median_share"].ge(0.7)
            | line_distance["above_median_share"].le(0.3)
        )

    # Compute per-line extrema to keep payload metadata and retain the older +/-100 guard.
    line_extremes = (
        complete.dropna(subset=[metric, "riskRating"])
        .groupby("line_id", as_index=False)[metric]
        .agg(
            min_metric="min",
            max_metric="max",
            max_abs=lambda s: s.abs().max(),
        )
    )
    line_avg = line_avg.merge(line_extremes, on="line_id", how="left")
    line_avg = line_avg.merge(line_distance, on="line_id", how="left")

    example_lines = []
    for risk in RISK_ORDER:
        group = line_avg.loc[line_avg["riskRating"].eq(risk)].dropna(subset=["pct_rank"]).copy()
        if eligible_feature_fips_by_risk is not None:
            eligible_fips = eligible_feature_fips_by_risk.get(risk, set())
            group = group.loc[group["fips"].astype(str).isin(eligible_fips)].copy()
        if group.empty:
            continue
        if eligible_line_ids:
            group = group.loc[group["line_id"].astype(str).isin(eligible_line_ids)].copy()
        if group.empty:
            continue
        # Sample counties shown in the "What Sets Apart..." plot should remain
        # visually interpretable: require all median PPSF YoY values to stay within +/-100%.
        group = group.loc[group["max_metric"].le(100) & group["min_metric"].ge(-100)].copy()
        if group.empty:
            continue
        group_median = group["avg_metric"].median()
        above_candidates = group.copy()
        below_candidates = group.copy()

        risk_trajectories = distance_join.loc[
            distance_join["riskRating"].eq(risk),
            ["line_id", anchor_col, "standardized_line_gap"],
        ].pivot_table(
            index="line_id",
            columns=anchor_col,
            values="standardized_line_gap",
            aggfunc="median",
        )
        above_candidates = above_candidates.loc[
            above_candidates["line_id"].isin(risk_trajectories.index)
        ].sort_values("line_id")
        below_candidates = below_candidates.loc[
            below_candidates["line_id"].isin(risk_trajectories.index)
        ].sort_values("line_id")

        # Optimize the two examples as a pair. Median standardized separation
        # rewards a sustained visual contrast without allowing high-variance
        # months to dominate merely because their raw IQR is wider.
        selected_pair: list[dict[str, object]] = []
        best_pair_score: tuple[bool, bool, float, float, float] | None = None
        for above_candidate in above_candidates.to_dict("records"):
            above_values = risk_trajectories.loc[above_candidate["line_id"]].to_numpy(dtype=float)
            for below_candidate in below_candidates.to_dict("records"):
                if (
                    str(below_candidate["line_id"]) == str(above_candidate["line_id"])
                    or str(below_candidate["fips"]) == str(above_candidate["fips"])
                ):
                    continue
                below_values = risk_trajectories.loc[below_candidate["line_id"]].to_numpy(dtype=float)
                pair_delta = above_values - below_values
                pair_contrast = float(np.nanmedian(np.abs(pair_delta)))
                positive_share = float(np.nanmean(pair_delta > 0))
                negative_share = float(np.nanmean(pair_delta < 0))
                pair_consistent_share = max(positive_share, negative_share)
                average_gap = float(abs(above_candidate["avg_metric"] - below_candidate["avg_metric"]))
                opposite_group_sides = (
                    float(above_candidate["median_standardized_gap"])
                    * float(below_candidate["median_standardized_gap"])
                    < 0
                )
                meets_consistency_target = pair_consistent_share >= 0.7 and pair_contrast >= 0.5
                pair_score = (
                    meets_consistency_target,
                    opposite_group_sides if meets_consistency_target else False,
                    pair_contrast if meets_consistency_target else pair_consistent_share,
                    pair_consistent_share if meets_consistency_target else pair_contrast,
                    average_gap,
                )
                if best_pair_score is None or pair_score > best_pair_score:
                    best_pair_score = pair_score
                    if float(np.nanmedian(pair_delta)) >= 0:
                        lower_candidate, higher_candidate = below_candidate, above_candidate
                    else:
                        lower_candidate, higher_candidate = above_candidate, below_candidate
                    strict_pair = (
                        float(lower_candidate["median_standardized_gap"]) <= -0.5
                        and float(higher_candidate["median_standardized_gap"]) >= 0.5
                        and float(lower_candidate["above_median_share"]) <= 0.3
                        and float(higher_candidate["above_median_share"]) >= 0.7
                    )
                    majority_direction_pair = (
                        float(lower_candidate["median_standardized_gap"]) < 0
                        and float(higher_candidate["median_standardized_gap"]) > 0
                        and float(lower_candidate["above_median_share"]) <= 0.5
                        and float(higher_candidate["above_median_share"]) >= 0.5
                    )
                    if strict_pair:
                        selection_tier = "strict"
                        lower_candidate["selection_position"] = "Below group median"
                        higher_candidate["selection_position"] = "Above group median"
                    elif majority_direction_pair:
                        selection_tier = "majority-direction fallback"
                        lower_candidate["selection_position"] = "Below group median"
                        higher_candidate["selection_position"] = "Above group median"
                    else:
                        selection_tier = "maximum-contrast fallback"
                        lower_candidate["selection_position"] = "Lower contrasting trajectory"
                        higher_candidate["selection_position"] = "Higher contrasting trajectory"
                    for candidate in [lower_candidate, higher_candidate]:
                        candidate["pair_contrast"] = pair_contrast
                        candidate["pair_consistent_share"] = pair_consistent_share
                        candidate["pair_average_gap"] = average_gap
                        candidate["selection_tier"] = selection_tier
                    selected_pair = [lower_candidate, higher_candidate]
        if len(selected_pair) != sample_per_group:
            continue
        candidates = pd.DataFrame(selected_pair)

        for candidate in candidates.itertuples(index=False):
            rows = complete.loc[complete["line_id"].eq(candidate.line_id)].sort_values(anchor_col)
            sample_position = candidate.selection_position
            example_lines.append(
                {
                    "riskRating": risk,
                    "lineId": candidate.line_id,
                    "fips": candidate.fips,
                    "county": candidate.county_label,
                    "state": candidate.state_code,
                    "pctRank": float(candidate.pct_rank),
                    "avgPpsfYoy": serialize_number(candidate.avg_metric, 5),
                    "groupMedianPpsfYoy": serialize_number(group_median, 5),
                    "samplePosition": sample_position,
                    "minPpsfYoy": serialize_number(candidate.min_metric, 3),
                    "maxPpsfYoy": serialize_number(candidate.max_metric, 3),
                    "meanLineGap": serialize_number(candidate.mean_line_gap, 5),
                    "meanAbsoluteLineGap": serialize_number(candidate.mean_abs_line_gap, 5),
                    "medianLineGap": serialize_number(candidate.median_line_gap, 5),
                    "aboveMedianShare": serialize_number(candidate.above_median_share, 3),
                    "medianStandardizedGap": serialize_number(candidate.median_standardized_gap, 5),
                    "distanceThreshold": serialize_number(candidate.distance_threshold, 5),
                    "pairContrast": serialize_number(candidate.pair_contrast, 5),
                    "pairConsistentShare": serialize_number(candidate.pair_consistent_share, 5),
                    "pairAverageGap": serialize_number(candidate.pair_average_gap, 5),
                    "selectionTier": candidate.selection_tier,
                    "values": [
                        {"month": int(getattr(row, anchor_col)), "value": serialize_number(getattr(row, metric), 5)}
                        for row in rows.itertuples(index=False)
                        if pd.notna(getattr(row, metric))
                    ],
                }
            )

    # Per-county percentile rank of average PPSF YoY within its risk group over this window.
    # Keyed by fips → percentile (0–100). Counties appearing in multiple events get their
    # best (highest avg_metric) line's rank.
    county_pct = (
        line_avg.sort_values("avg_metric", ascending=False)
        .drop_duplicates(subset=["fips"])
        [["fips", "pct_rank"]]
        .set_index("fips")["pct_rank"]
        .to_dict()
    )

    return {
        "byRating": by_rating,
        "affectedCounties": [
            {"fips": row.fips, "riskRating": row.riskRating}
            for row in affected_counties.itertuples(index=False)
        ],
        "riskCounts": {str(k): int(v) for k, v in risk_counts.items()},
        "exampleCountyLines": example_lines,
        "countyEventWindowPctRank": county_pct,
    }


def build_event_windows(
    con: duckdb.DuckDBPyConnection,
    eligible_feature_fips_by_risk: dict[str, set[str]] | None = None,
) -> dict[str, object]:
    events = load_disaster_events(con)
    events = events.loc[events["event_start_month"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-01"))].copy()
    housing = load_redfin_county_monthly(con)
    for column in ["median_ppsf_yoy", "avg_sale_to_list_yoy", "homes_sold_yoy", "inventory_yoy", "housing_market_index"]:
        if column in housing:
            housing.loc[pd.to_numeric(housing[column], errors="coerce").le(-888888000), column] = np.nan
    metric = "median_ppsf_yoy"

    # Window A: 1 year before event start → 3 years after event end (pre=12, post=36)
    # Window B: 1 year before event end → 5 years after event end (pre=12, post=60, anchored at event end)
    # We use pre_event_months=24 for the raw build to cover both windows.
    affected = build_affected_event_windows(events, housing, pre_event_months=24, post_event_months=60)
    if affected.empty:
        empty = {"byRating": [], "affectedCounties": [], "riskCounts": {}, "exampleCountyLines": []}
        return {
            "windowA": empty,
            "windowB": empty,
            "summary": {"events": 0},
        }

    nri = con.execute("SELECT fips, risk_rating FROM mart.nri_county_risk WHERE fips IS NOT NULL").df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)

    # Window A uses event_window_month (relative to event start), pre=12, post=36
    window_a = _build_window_data(
        affected,
        nri,
        metric,
        pre_months=12,
        post_months=36,
        eligible_feature_fips_by_risk=eligible_feature_fips_by_risk,
    )

    # Window B is anchored at event END: use months_after_event_end for post, months_from_event_start for pre.
    # We derive a combined "end-anchored" month column: negative = months before event end, positive = after.
    affected_b = affected.copy()
    affected_b["ewm_end"] = np.where(
        affected_b["months_from_event_start"].le(0),
        # pre-event period: distance from event start, negative
        affected_b["months_from_event_start"],
        # post-event period: months_after_event_end
        affected_b["months_after_event_end"],
    )
    affected_b["line_id"] = affected_b["event_key"]
    window_b = _build_window_data(
        affected_b,
        nri,
        metric,
        pre_months=12,
        post_months=60,
        anchor_col="ewm_end",
        eligible_feature_fips_by_risk=eligible_feature_fips_by_risk,
    )

    return {
        "windowA": window_a,
        "windowB": window_b,
        "summary": {
            "events": int(events["event_key"].nunique()),
        },
    }


def ten_year_avg_by_fips(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> pd.DataFrame:
    quoted = ", ".join(f'"{column}"' for column in columns)
    df = con.execute(
        f"""
        SELECT {quoted}
        FROM {table}
        WHERE fips IS NOT NULL
          AND year IS NOT NULL
          AND year >= (SELECT max(year) FROM {table}) - 9
        """
    ).df()
    if df.empty:
        return df
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    value_columns = [column for column in columns if column not in {"fips", "year"}]
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column].le(-888888000), column] = np.nan
    return df.groupby("fips", as_index=False)[value_columns].mean()


def build_county_playbook_data(
    con: duckdb.DuckDBPyConnection,
) -> dict[str, object]:
    """Build county hazard ratings, monthly PPSF history, and event periods."""
    hazard_cols: list[str] = []
    for hazard in HAZARDS:
        if hazard["key"] == "overall":
            hazard_cols.extend(["risk_score", "risk_rating"])
        else:
            hazard_cols.extend([hazard["score"], hazard["rating"]])
    nri = con.execute(
        f"""
        SELECT fips, COUNTY, STATEABBRV, {", ".join(hazard_cols)}
        FROM mart.nri_county_risk
        WHERE fips IS NOT NULL
        """
    ).df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    labels = con.execute(
        """
        SELECT lpad(fips, 5, '0') AS fips,
               any_value(REGION) AS county_label,
               any_value(STATE_CODE) AS state_code
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
        GROUP BY fips
        """
    ).df()
    labels["fips"] = labels["fips"].astype(str).str.zfill(5)
    nri = nri.merge(labels, on="fips", how="left")
    for hazard in HAZARDS:
        nri[hazard["score"]] = clean_numeric(nri[hazard["score"]])
        nri[hazard["rating"]] = nri[hazard["rating"]].map(rating_clean)
    counties = []
    for row in nri.itertuples(index=False):
        state = (
            str(row.state_code)
            if pd.notna(row.state_code)
            else str(row.STATEABBRV)
            if pd.notna(row.STATEABBRV)
            else ""
        )
        county = (
            str(row.county_label)
            if pd.notna(row.county_label)
            else f"{row.COUNTY}, {state}"
            if pd.notna(row.COUNTY)
            else f"County FIPS {row.fips}"
        )
        hazards = {}
        for hazard in HAZARDS:
            rating = getattr(row, hazard["rating"])
            hazards[hazard["key"]] = {
                "score": serialize_number(getattr(row, hazard["score"]), 3),
                "rating": rating,
                "ratingValue": RISK_NUMERIC.get(rating),
            }
        counties.append(
            {
                "fips": row.fips,
                "county": county,
                "state": state,
                "riskRating": hazards["overall"]["rating"],
                "riskScore": hazards["overall"]["score"],
                "hazards": hazards,
            }
        )
    if not counties:
        return {"available": False, "message": "County housing and NRI data are unavailable"}

    history = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            date_trunc('month', period_begin)::DATE AS month,
            avg(CASE
                WHEN try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL
                ELSE try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE)
            END) AS median_ppsf_yoy
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
          AND period_begin IS NOT NULL
          AND coalesce(property_type, PROPERTY_TYPE_1) = 'All Residential'
          AND extract(year FROM period_begin) >= (
              SELECT max(extract(year FROM period_begin)) - 9
              FROM mart.redfin_county_monthly
              WHERE period_begin IS NOT NULL
          )
        GROUP BY fips, date_trunc('month', period_begin)
        ORDER BY fips, month
        """
    ).df()
    history["fips"] = history["fips"].astype(str).str.zfill(5)
    history["median_ppsf_yoy"] = pd.to_numeric(history["median_ppsf_yoy"], errors="coerce")
    history = history.dropna(subset=["median_ppsf_yoy"])

    events = load_disaster_events(con)
    events = events.loc[
        events["event_start_month"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-01"))
    ].drop_duplicates("event_key")
    county_fips = {str(county["fips"]).zfill(5) for county in counties}
    events = events.loc[events["fips"].isin(county_fips)].sort_values(["fips", "event_start_month"])

    history_months = pd.date_range(history["month"].min(), history["month"].max(), freq="MS")
    history_month_labels = [month.strftime("%Y-%m") for month in history_months]
    monthly_history_values_by_fips: dict[str, list[float | None]] = {}
    for fips, county_history in history.groupby("fips", sort=False):
        values = county_history.set_index("month")["median_ppsf_yoy"].reindex(history_months)
        monthly_history_values_by_fips[fips] = [serialize_number(value, 5) for value in values]
    events_by_fips: dict[str, list[list[object]]] = {}
    for row in events.itertuples(index=False):
        events_by_fips.setdefault(row.fips, []).append(
            [
                row.event_key,
                row.event_source,
                row.event_type,
                row.event_name,
                row.event_start_month.strftime("%Y-%m"),
                row.event_end_month.strftime("%Y-%m"),
            ]
        )

    return {
        "available": True,
        "hazards": [{"key": hazard["key"], "label": hazard["label"]} for hazard in HAZARDS],
        "counties": counties,
        "monthlyHistoryMonths": history_month_labels,
        "monthlyHistoryValuesByFips": monthly_history_values_by_fips,
        "eventsByFips": events_by_fips,
        "historyStart": history["month"].min().strftime("%Y-%m"),
        "historyEnd": history["month"].max().strftime("%Y-%m"),
    }


def build_feature_payload(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    nri = con.execute("SELECT fips, risk_rating, risk_score FROM mart.nri_county_risk WHERE fips IS NOT NULL").df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)
    nri["riskValue"] = nri["riskRating"].map(RISK_NUMERIC)

    econ_cols = [
        "fips",
        "year",
        "dp03_income_and_benefits_total_households_median_household_income_est",
        "dp03_civilian_labor_force_unemployment_rate_pct",
    ]
    demo_cols = [
        "fips",
        "year",
        "domestic_in_migration_rate",
        "dp05_total_population_65_plus_pct",
        "dp02_households_by_type_total_households_households_with_one_or_more_people_65_plus_pct",
        "dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct",
        "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct",
        "dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct",
    ]
    affordability_cols = [
        "fips",
        "year",
        "s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est",
        "s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_median_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_30_0_to_34_9_percent_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_35_0_percent_or_more_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_not_computed_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_30_0_to_34_9_percent_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_35_0_percent_or_more_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_not_computed_est",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_30_0_to_34_9_percent_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_35_0_percent_or_more_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_30_0_to_34_9_percent_pct",
        "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_35_0_percent_or_more_pct",
        "median_owner_costs_mortgage",
        "housing_cost_pct_income",
        "owner_mortgage_cost_burden_30pct_plus",
        "b25132_monthly_electricity_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_less_than_dollars_50_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_50_to_dollars_99_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_100_to_dollars_149_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_150_to_dollars_199_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_200_to_dollars_249_est",
        "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_250_or_more_est",
        "b25133_monthly_gas_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_less_than_dollars_25_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_25_to_dollars_49_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_50_to_dollars_74_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_75_to_dollars_99_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_100_to_dollars_149_est",
        "b25133_monthly_gas_costs_total_charged_for_gas_dollars_150_or_more_est",
        "b25134_annual_water_and_sewer_costs_total_not_charged_or_payment_included_in_other_fees_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_less_than_dollars_125_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_125_to_dollars_249_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_250_to_dollars_499_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_500_to_dollars_749_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_750_to_dollars_999_est",
        "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_1_000_or_more_est",
        "b25135_annual_other_fuel_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est",
        "b25135_annual_other_fuel_costs_total_charged_for_other_fuels_less_than_dollars_250_est",
        "b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_250_to_dollars_749_est",
        "b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_750_or_more_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_no_real_estate_taxes_paid_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_less_than_dollars_800_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_dollars_800_to_dollars_1_499_est",
        "s2506_owner_occupied_units_mortgage_real_estate_taxes_dollars_1_500_or_more_est",
        "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_no_real_estate_taxes_paid_est",
        "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_less_than_dollars_800_est",
        "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_dollars_800_to_dollars_1_499_est",
        "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_dollars_1_500_or_more_est",
    ]
    insurance_cols = [
        f"b25141_homeowners_insurance_costs_by_mortgage_status_total_{status}_{suffix}_est"
        for status in ["mortgage", "not_mortgaged"]
        for suffix in [
            "less_than_dollars_100",
            "dollars_100_to_dollars_299",
            "dollars_300_to_dollars_499",
            "dollars_500_to_dollars_799",
            "dollars_800_to_dollars_999",
            "dollars_1000_to_dollars_1499",
            "dollars_1500_to_dollars_1999",
            "dollars_2000_to_dollars_2499",
            "dollars_2500_to_dollars_2999",
            "dollars_3000_to_dollars_3499",
            "dollars_3500_to_dollars_3999",
            "dollars_4000_or_more",
        ]
    ]
    affordability_cols.extend(insurance_cols)

    econ = ten_year_avg_by_fips(con, "mart.acs_county_economic_annual", econ_cols)
    demo = ten_year_avg_by_fips(con, "mart.acs_county_demographic_annual", demo_cols)
    afford = ten_year_avg_by_fips(con, "mart.acs_county_affordability_annual", affordability_cols)
    weather = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            avg(avg_temperature_f) AS avg_temperature_f,
            avg(precipitation_inches) AS precipitation_inches
        FROM mart.ncei_county_weather_monthly
        WHERE fips IS NOT NULL
          AND weather_month IS NOT NULL
          AND extract(year FROM weather_month) >= (
              SELECT max(extract(year FROM weather_month)) - 9
              FROM mart.ncei_county_weather_monthly
              WHERE weather_month IS NOT NULL
          )
        GROUP BY fips
        """
    ).df()
    weather["fips"] = weather["fips"].astype(str).str.zfill(5)
    migration = con.execute(
        """
        WITH net AS (
          SELECT lpad(fips, 5, '0') AS fips, avg(CAST(total_net_migration AS DOUBLE)) AS avg_total_net_migration
          FROM mart.statsamerica_population_components_annual
          WHERE fips IS NOT NULL
            AND year >= (SELECT max(year) FROM mart.statsamerica_population_components_annual) - 9
          GROUP BY fips
        ),
        pop AS (
          SELECT lpad(fips, 5, '0') AS fips,
                 avg(try_cast(replace(nullif(trim(cast(total_population AS VARCHAR)), ''), ',', '') AS DOUBLE)) AS avg_population
          FROM mart.acs_county_demographic_annual
          WHERE fips IS NOT NULL
            AND year >= (SELECT max(year) FROM mart.acs_county_demographic_annual) - 9
          GROUP BY fips
        )
        SELECT net.fips, avg_total_net_migration / nullif(avg_population, 0) AS net_migration_rate
        FROM net
        LEFT JOIN pop ON net.fips = pop.fips
        """
    ).df()
    migration["fips"] = migration["fips"].astype(str).str.zfill(5)

    for frame in [econ, demo, afford, weather]:
        for column in frame.columns:
            if column != "fips":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    afford["estimated_annual_home_insurance"] = weighted_bucket_average(
        afford,
        [
            (f"b25141_homeowners_insurance_costs_by_mortgage_status_total_{status}_{suffix}_est", midpoint)
            for status in ["mortgage", "not_mortgaged"]
            for suffix, midpoint in [
                ("less_than_dollars_100", 50),
                ("dollars_100_to_dollars_299", 200),
                ("dollars_300_to_dollars_499", 400),
                ("dollars_500_to_dollars_799", 650),
                ("dollars_800_to_dollars_999", 900),
                ("dollars_1000_to_dollars_1499", 1250),
                ("dollars_1500_to_dollars_1999", 1750),
                ("dollars_2000_to_dollars_2499", 2250),
                ("dollars_2500_to_dollars_2999", 2750),
                ("dollars_3000_to_dollars_3499", 3250),
                ("dollars_3500_to_dollars_3999", 3750),
                ("dollars_4000_or_more", 4250),
            ]
        ],
    )
    afford["estimated_annual_property_tax"] = weighted_bucket_average(
        afford,
        [
            (f"{prefix}_{suffix}_est", midpoint)
            for prefix in ["s2506_owner_occupied_units_mortgage_real_estate_taxes", "s2507_owner_occupied_units_no_mortgage_real_estate_taxes"]
            for suffix, midpoint in [
                ("less_than_dollars_800", 400),
                ("dollars_800_to_dollars_1_499", 1150),
                ("dollars_1_500_or_more", 2000),
            ]
        ],
        zero_cols=[
            "s2506_owner_occupied_units_mortgage_real_estate_taxes_no_real_estate_taxes_paid_est",
            "s2507_owner_occupied_units_no_mortgage_real_estate_taxes_no_real_estate_taxes_paid_est",
        ],
    )
    electricity = weighted_bucket_average(
        afford,
        [
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_less_than_dollars_50_est", 25),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_50_to_dollars_99_est", 75),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_100_to_dollars_149_est", 125),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_150_to_dollars_199_est", 175),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_200_to_dollars_249_est", 225),
            ("b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_250_or_more_est", 275),
        ],
        zero_cols=["b25132_monthly_electricity_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"],
    ) * 12
    gas = weighted_bucket_average(
        afford,
        [
            ("b25133_monthly_gas_costs_total_charged_for_gas_less_than_dollars_25_est", 12.5),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_25_to_dollars_49_est", 37.5),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_50_to_dollars_74_est", 62.5),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_75_to_dollars_99_est", 87.5),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_100_to_dollars_149_est", 125),
            ("b25133_monthly_gas_costs_total_charged_for_gas_dollars_150_or_more_est", 175),
        ],
        zero_cols=["b25133_monthly_gas_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"],
    ) * 12
    water = weighted_bucket_average(
        afford,
        [
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_less_than_dollars_125_est", 62.5),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_125_to_dollars_249_est", 187.5),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_250_to_dollars_499_est", 375),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_500_to_dollars_749_est", 625),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_750_to_dollars_999_est", 875),
            ("b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_1_000_or_more_est", 1125),
        ],
        zero_cols=["b25134_annual_water_and_sewer_costs_total_not_charged_or_payment_included_in_other_fees_est"],
    )
    other_fuel = weighted_bucket_average(
        afford,
        [
            ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_less_than_dollars_250_est", 125),
            ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_250_to_dollars_749_est", 500),
            ("b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_750_or_more_est", 875),
        ],
        zero_cols=["b25135_annual_other_fuel_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"],
    )
    afford["estimated_annual_utilities"] = electricity + gas + water + other_fuel
    afford["income_median_household_usd"] = afford["s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est"]
    afford["insurance_homeowners_pct_income"] = afford["estimated_annual_home_insurance"] / afford["income_median_household_usd"].replace(0, np.nan) * 100
    afford["property_taxes_pct_income"] = afford["s2506_owner_occupied_units_mortgage_real_estate_taxes_median_est"] / afford["income_median_household_usd"].replace(0, np.nan) * 100
    afford["utilities_pct_income"] = afford["estimated_annual_utilities"] / afford["income_median_household_usd"].replace(0, np.nan) * 100
    burdened_owner_households = sum(
        (
            afford[column]
            for column in [
                "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_30_0_to_34_9_percent_est",
                "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_35_0_percent_or_more_est",
                "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_30_0_to_34_9_percent_est",
                "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_35_0_percent_or_more_est",
            ]
        ),
        start=pd.Series(0.0, index=afford.index),
    )
    owner_households_with_computable_burden = (
        afford["dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_est"]
        + afford["dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_est"]
        - afford["dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_not_computed_est"].fillna(0)
        - afford["dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_not_computed_est"].fillna(0)
    )
    afford["housing_burden_30pct_plus_share"] = (
        burdened_owner_households
        / owner_households_with_computable_burden.replace(0, np.nan)
        * 100
    )
    afford["homeownership_cost_pct_income"] = (
        afford["s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est"] * 12
        / afford["income_median_household_usd"].replace(0, np.nan)
        * 100
    )

    features = (
        nri[["fips", "riskRating", "riskValue", "risk_score"]]
        .merge(econ[["fips", *econ_cols[2:]]], on="fips", how="left")
        .merge(demo[["fips", *demo_cols[2:]]], on="fips", how="left")
        .merge(migration, on="fips", how="left")
        .merge(afford[[
            "fips",
            "median_owner_costs_mortgage",
            "housing_cost_pct_income",
            "owner_mortgage_cost_burden_30pct_plus",
            "estimated_annual_home_insurance",
            "estimated_annual_property_tax",
            "estimated_annual_utilities",
            "income_median_household_usd",
            "insurance_homeowners_pct_income",
            "property_taxes_pct_income",
            "utilities_pct_income",
            "housing_burden_30pct_plus_share",
            "homeownership_cost_pct_income",
        ]], on="fips", how="left")
        .merge(weather, on="fips", how="left")
    )
    bea_features = con.execute(
        """
        WITH bea AS (
          SELECT
            lpad(fips, 5, '0') AS fips,
            avg(net_earnings_by_place_of_residence_thousands * 1000.0 / nullif(population, 0)) AS net_earnings_per_capita,
            avg(dividends_interest_rent_thousands * 1000.0 / nullif(population, 0)) AS dividends_interest_rent_per_capita,
            avg(transfer_receipts_thousands * 1000.0 / nullif(population, 0)) AS transfer_receipts_per_capita
          FROM mart.statsamerica_bea_personal_income_annual
          WHERE fips IS NOT NULL
            AND year >= (SELECT max(year) FROM mart.statsamerica_bea_personal_income_annual) - 9
            AND population > 0
          GROUP BY fips
        ),
        cew AS (
          SELECT
            lpad(s.fips, 5, '0') AS fips,
            avg(s.total_wages_dollars / nullif(t.total_wages_dollars, 0) * 100) AS accom_food_wages_pct_total_wages
          FROM mart.statsamerica_cew_county_sector_annual s
          JOIN mart.statsamerica_cew_county_annual t
            ON lpad(s.fips, 5, '0') = lpad(t.fips, 5, '0')
           AND s.year = t.year
          WHERE s.naics_code = '72'
            AND s.year >= (SELECT max(year) FROM mart.statsamerica_cew_county_sector_annual) - 9
            AND t.total_wages_dollars > 0
          GROUP BY s.fips
        )
        SELECT
          coalesce(bea.fips, cew.fips) AS fips,
          net_earnings_per_capita,
          dividends_interest_rent_per_capita,
          transfer_receipts_per_capita,
          accom_food_wages_pct_total_wages
        FROM bea
        FULL OUTER JOIN cew ON bea.fips = cew.fips
        """
    ).df()
    if not bea_features.empty:
        bea_features["fips"] = bea_features["fips"].astype(str).str.zfill(5)
        features = features.merge(bea_features, on="fips", how="left")
    redfin_features = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            avg(CASE WHEN try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(MEDIAN_PPSF_YOY, ',', '') AS DOUBLE) END) AS median_ppsf_yoy,
            avg(CASE WHEN try_cast(replace(AVG_SALE_TO_LIST_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(AVG_SALE_TO_LIST_YOY, ',', '') AS DOUBLE) END) AS avg_sale_to_list_yoy,
            avg(CASE WHEN try_cast(replace(HOMES_SOLD_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(HOMES_SOLD_YOY, ',', '') AS DOUBLE) END) AS homes_sold_yoy,
            avg(CASE WHEN try_cast(replace(INVENTORY_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(INVENTORY_YOY, ',', '') AS DOUBLE) END) AS inventory_yoy,
            avg(CASE WHEN try_cast(replace(NEW_LISTINGS_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(NEW_LISTINGS_YOY, ',', '') AS DOUBLE) END) AS new_listings_yoy,
            avg(CASE WHEN try_cast(replace(MEDIAN_DOM_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(MEDIAN_DOM_YOY, ',', '') AS DOUBLE) END) AS median_dom_yoy,
            avg(CASE WHEN try_cast(replace(PRICE_DROPS_YOY, ',', '') AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(replace(PRICE_DROPS_YOY, ',', '') AS DOUBLE) END) AS price_drops_yoy
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
          AND period_begin IS NOT NULL
          AND extract(year FROM period_begin) >= (
              SELECT max(extract(year FROM period_begin)) - 9
              FROM mart.redfin_county_monthly
              WHERE period_begin IS NOT NULL
          )
          AND coalesce(property_type, PROPERTY_TYPE_1) = 'All Residential'
        GROUP BY fips
        """
    ).df()
    redfin_features["fips"] = redfin_features["fips"].astype(str).str.zfill(5)
    features = features.merge(redfin_features, on="fips", how="left")
    features["no_broadband_pct"] = 100 - features["dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct"]
    feature_defs = [
        ("Economic", "Income", "income_median_household_usd", "currency", "mart.acs_county_affordability_annual"),
        ("Economic", "Insurance Share of Income", "insurance_homeowners_pct_income", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Property Tax Share of Income", "property_taxes_pct_income", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Utilities Share of Income", "utilities_pct_income", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Cost-Burdened Households", "housing_burden_30pct_plus_share", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Homeownership Cost Share", "homeownership_cost_pct_income", "percent", "mart.acs_county_affordability_annual"),
        ("Economic", "Unemployment", "dp03_civilian_labor_force_unemployment_rate_pct", "percent", "mart.acs_county_economic_annual"),
        ("Economic", "Net Earnings per Capita", "net_earnings_per_capita", "currency", "mart.statsamerica_bea_personal_income_annual"),
        ("Economic", "Dividends/Interest/Rent per Capita", "dividends_interest_rent_per_capita", "currency", "mart.statsamerica_bea_personal_income_annual"),
        ("Economic", "Transfer Receipts per Capita", "transfer_receipts_per_capita", "currency", "mart.statsamerica_bea_personal_income_annual"),
        ("Economic", "Accom. & Food Wages Share of Total Wages", "accom_food_wages_pct_total_wages", "percent", "mart.statsamerica_cew_county_sector_annual"),
        ("Demographic", "Net Migration Rate", "net_migration_rate", "signed_pct", "mart.statsamerica_population_components_annual"),
        ("Demographic", "Age >= 65 Years", "dp05_total_population_65_plus_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Demographic", "Disability Status", "dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Demographic", "Communication Barrier", "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Demographic", "No Internet Access", "no_broadband_pct", "percent", "mart.acs_county_demographic_annual"),
        ("Housing Market", "Median PPSF YOY", "median_ppsf_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "Average Sale-to-List YOY", "avg_sale_to_list_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "Homes Sold YOY", "homes_sold_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "Inventory YOY", "inventory_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "New Listings YOY", "new_listings_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "Median Days on Market YOY", "median_dom_yoy", "number", "mart.redfin_county_monthly"),
        ("Housing Market", "Price Drops YOY", "price_drops_yoy", "pct", "mart.redfin_county_monthly"),
        ("Climate", "Temperature", "avg_temperature_f", "temperature_f", "mart.ncei_county_weather_monthly"),
        ("Climate", "Precipitation", "precipitation_inches", "inches", "mart.ncei_county_weather_monthly"),
    ]
    excluded_feature_labels = {
        "Median PPSF YOY",
        "Homeownership Cost Share",
        "Accom. & Food Wages Share of Total Wages",
    }
    for _, _, column, _, _ in feature_defs:
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    candidate_feature_defs = [definition for definition in feature_defs if definition[1] not in excluded_feature_labels]
    feature_display_meta = {
        label: {"category": category, "column": column, "format": fmt, "source": source}
        for category, label, column, fmt, source in feature_defs
    }
    # --- Option B: within-group feature correlations ---
    # For each risk rating, Spearman correlation of each feature with the continuous
    # NRI risk_score (0–100) restricted to counties in that tier. Uses the raw score
    # rather than the integer riskValue so there is meaningful variance within each tier.
    events_for_position = load_disaster_events(con)
    events_for_position = events_for_position.loc[
        events_for_position["event_start_month"].between(pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-01"))
    ].copy()
    housing_for_position = load_redfin_county_monthly(con)
    housing_for_position.loc[pd.to_numeric(housing_for_position["median_ppsf_yoy"], errors="coerce").le(-888888000), "median_ppsf_yoy"] = np.nan
    affected_for_position = build_affected_event_windows(
        events_for_position,
        housing_for_position,
        pre_event_months=24,
        post_event_months=60,
    )
    required_position_months = event_window_months(12, 36)
    complete_position = filter_complete_event_window_lines(
        affected_for_position,
        x_col="event_window_month",
        line_col="line_id",
        metric_col="median_ppsf_yoy",
        required_x_values=required_position_months,
    ).copy()
    complete_position = complete_position.loc[complete_position["event_window_month"].isin(required_position_months)].copy()
    complete_position = complete_position.merge(nri[["fips", "riskRating"]], on="fips", how="left")
    line_position = (
        complete_position.dropna(subset=["riskRating", "median_ppsf_yoy"])
        .groupby(["line_id", "fips", "riskRating"], as_index=False)["median_ppsf_yoy"]
        .mean()
        .rename(columns={"median_ppsf_yoy": "avg_ppsf_yoy"})
    )
    line_position["group_median_ppsf_yoy"] = line_position.groupby("riskRating", observed=False)["avg_ppsf_yoy"].transform("median")
    line_position["relative_position"] = line_position["avg_ppsf_yoy"] - line_position["group_median_ppsf_yoy"]
    position_analysis = line_position.merge(features, on="fips", how="inner", suffixes=("", "_feature"))

    within_group_top_features: dict[str, list[dict]] = {}
    within_group_feature_bins: dict[str, dict[str, dict[str, object]]] = {}
    feature_lookup = {
        label: {"category": category, "column": column, "format": fmt, "source": source}
        for category, label, column, fmt, source in feature_defs
    }
    for rating in RISK_ORDER:
        group = position_analysis[position_analysis["riskRating"] == rating].copy()
        rating_corrs = []
        for _, label, column, _, _ in candidate_feature_defs:
            valid_group = group.dropna(subset=[column, "relative_position"])
            if len(valid_group) < 10:
                corr = None
            else:
                corr = serialize_number(
                    valid_group[["relative_position", column]].corr(method="spearman").iloc[0, 1], 3
                )
            rating_corrs.append({"feature": label, "corr": corr})
        selected_corrs = [item for item in rating_corrs if item["corr"] is not None]
        selected_corrs = sorted(selected_corrs, key=lambda item: abs(item["corr"] or 0), reverse=True)[:10]
        within_group_top_features[rating] = selected_corrs
        within_group_feature_bins[rating] = {}
        group_position_baseline = group["relative_position"].median()
        for selected in selected_corrs:
            label = selected["feature"]
            meta = feature_lookup[label]
            column = meta["column"]
            valid_group = group.dropna(subset=[column, "relative_position"]).copy()
            if valid_group.empty:
                continue
            valid_group["bucket"], bucket_order = feature_bucket_labels(valid_group[column], meta["format"])
            valid_group = valid_group.dropna(subset=["bucket"]).copy()
            county_feature_values = valid_group[["fips", column]].drop_duplicates("fips").copy()
            county_feature_values["feature_percentile"] = (
                county_feature_values[column]
                .rank(method="average", pct=True, na_option="keep")
                .mul(100)
            )
            valid_group = valid_group.merge(
                county_feature_values[["fips", "feature_percentile"]],
                on="fips",
                how="left",
            )
            valid_group["feature_contribution"] = (
                (valid_group["feature_percentile"] / 100 * 2 - 1) * float(selected["corr"])
            )
            bucket_summaries = (
                valid_group.groupby("bucket", observed=False)
                .agg(
                    median_relative_position=("relative_position", "median"),
                    median_avg_ppsf_yoy=("avg_ppsf_yoy", "median"),
                    count=("line_id", "nunique"),
                )
                .reset_index()
            )
            bucket_summary_by_name = {}
            for summary in bucket_summaries.itertuples(index=False):
                bucket_order_index = bucket_order.index(str(summary.bucket)) if str(summary.bucket) in bucket_order else None
                ppsf_association = classify_bucket_position(bucket_order_index, len(bucket_order), selected["corr"])
                bucket_summary_by_name[str(summary.bucket)] = {
                    "bucket": str(summary.bucket),
                    "bucketOrder": bucket_order_index,
                    "bucketCount": len(bucket_order),
                    "medianRelativePosition": serialize_number(summary.median_relative_position, 5),
                    "medianAvgPpsfYoy": serialize_number(summary.median_avg_ppsf_yoy, 5),
                    "relativePpsfAssociation": ppsf_association,
                    "ppsfAssociation": ppsf_association,
                    "relativePpsfCorrelation": selected["corr"],
                    "ppsfCorrelation": selected["corr"],
                    "count": int(summary.count),
                    "baselineRelativePosition": serialize_number(group_position_baseline, 5),
                }
            for row in valid_group[
                ["fips", column, "bucket", "feature_percentile", "feature_contribution"]
            ].drop_duplicates("fips").itertuples(index=False):
                summary = bucket_summary_by_name.get(str(row.bucket))
                if not summary:
                    continue
                within_group_feature_bins[rating].setdefault(str(row.fips), {})[label] = {
                    "value": serialize_number(row[1], 4),
                    "valuePercentile": serialize_number(row.feature_percentile, 2),
                    "contribution": serialize_number(row.feature_contribution, 4),
                    "bucket": str(row.bucket),
                    "bucketOrder": summary["bucketOrder"],
                    "bucketCount": summary["bucketCount"],
                    "corr": selected["corr"],
                    "relativePpsfAssociation": summary["relativePpsfAssociation"],
                    "ppsfAssociation": summary["ppsfAssociation"],
                    "relativePpsfCorrelation": summary["relativePpsfCorrelation"],
                    "ppsfCorrelation": summary["ppsfCorrelation"],
                    "medianRelativePosition": summary["medianRelativePosition"],
                    "medianAvgPpsfYoy": summary["medianAvgPpsfYoy"],
                    "baselineRelativePosition": summary["baselineRelativePosition"],
                }

    return {
        "riskOrder": RISK_ORDER,
        "featureMeta": feature_display_meta,
        "withinGroupTopFeatures": within_group_top_features,
        "withinGroupFeatureBins": within_group_feature_bins,
    }


def make_html(data: dict[str, object]) -> str:
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    return HTML_TEMPLATE.replace("__PAYLOAD__", payload)


def make_deferred_data_script(global_name: str, data: object) -> str:
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    return f"window.{global_name}={payload};\n"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Which Way the Wind Blows: Climate Risk and U.S. Housing Markets</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Rubik:ital,wght@0,300..900;1,300..900&display=swap" rel="stylesheet">
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    :root {
      --paper: #f1f5f2;
      --ink: #17332d;
      --muted: #60736d;
      --panel: #ffffff;
      --line: #cad8d2;
      --teal: #11796d;
      --forest: #173f37;
      --water: #287da1;
      --hazard: #c4523d;
      --sun: #d6a52f;
      --shadow: 0 0 0 1px rgba(23, 51, 45, 0.04), 0 14px 34px rgba(23, 51, 45, 0.12);
    }
    .control-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 10px 0 12px; }
    .toggle-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .control-bar .sidebar-label { font-size: 11px; font-weight: 800; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-right: 4px; }
    .control-bar button { border-radius: 999px; padding: 8px 11px; font-size: 12px; text-align: left; border: 1px solid var(--line); background: #fff; cursor: pointer; font-weight: 800; display: inline-flex; align-items: center; gap: 6px; }
    .control-bar button.active { border-color: transparent; color: white; }
    .pricing-viz-grid { align-items: center; margin-bottom: 32px; }
    .map-with-legend { position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 0; }
    .map-with-legend .legend { position: absolute; top: 100%; left: 0; justify-content: center; width: 100%; }
    #within-group-correlations { overflow-x: auto; }
    .feature-detail-list { display: grid; grid-template-columns: minmax(135px, .8fr) 68px minmax(180px, 1.1fr) minmax(205px, 1.25fr) 68px minmax(180px, 1.1fr) minmax(205px, 1.25fr); column-gap: 12px; align-items: center; min-width: 1050px; }
    .feature-detail-header, .feature-detail-row { display: contents; }
    .feature-detail-header > div { padding: 0 0 7px; color: var(--muted); font-size: 10px; font-weight: 800; text-transform: uppercase; }
    .feature-detail-header .county-group { padding: 7px 8px; border-bottom: 2px solid var(--line); color: var(--ink); font-size: 12px; text-align: center; text-transform: none; }
    .feature-detail-row > div { min-width: 0; padding: 9px 0; border-top: 1px solid rgba(216,208,196,.7); font-size: 13px; }
    .feature-scale { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 6px; align-items: end; color: var(--muted); font-size: 10px; }
    .feature-scale > :first-child { grid-column: 1; grid-row: 1; }
    .feature-scale > :last-child { grid-column: 2; grid-row: 1; }
    .feature-scale > .feature-scale-bar { grid-column: 1 / -1; grid-row: 2; }
    .feature-scale-bar { position: relative; height: 10px; border-radius: 999px; border: 1px solid rgba(23,32,38,.14); background: linear-gradient(90deg, #16803c 0%, #e0b33b 50%, #b42318 100%); }
    .feature-scale-bar::after { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: rgba(23,32,38,.42); }
    .feature-scale-bar.percentile { background: linear-gradient(90deg, #e8f1f5 0%, #6fa3b8 50%, #22566b 100%); }
    .feature-scale-bar.percentile::after { display: none; }
    .feature-scale-arrow { position: absolute; top: -8px; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 8px solid #172026; transform: translateX(-5px); }
    .county-count { margin-top: 8px; text-align: center; color: var(--ink); font-size: 20px; font-weight: 850; }
    #risk-play-button { margin-left: 4px; visibility: hidden; opacity: 0; pointer-events: none; transition: opacity 180ms ease; }
    #risk-play-button.visible { visibility: visible; opacity: 1; pointer-events: auto; }
    .playbook-map-wrap { position: relative; height: 380px; overflow: hidden; }
    .playbook-map-wrap .chart { height: 380px; }
    .playbook-map-controls { position: absolute; top: 10px; right: 10px; z-index: 3; display: flex; gap: 6px; }
    .playbook-map-controls button { min-width: 34px; justify-content: center; box-shadow: 0 2px 8px rgba(23,32,38,.16); }
    .hazard-icon { display: inline-flex; width: 22px; justify-content: center; margin-right: 6px; }
    .hazard-rating-grid { margin: 10px 0 18px; }
    .hazard-rating-overall { margin-bottom: 8px; border-bottom: 2px solid var(--line); }
    .hazard-rating-overall .hazard-rating-item { font-size: 16px; font-weight: 800; padding: 11px 0; border-bottom: 0; }
    .hazard-rating-specific { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px 16px; }
    .hazard-rating-item { display: flex; justify-content: space-between; gap: 10px; align-items: center; padding: 9px 0; border-bottom: 1px solid var(--line); font-size: 13px; }
    .playbook-selected-county { display: none; margin: 10px 0 2px; text-align: center; color: var(--ink); font-size: 18px; font-weight: 850; }
    .playbook-history-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px 22px; min-height: 38px; margin: 8px auto 0; padding: 10px 14px; border: 1px solid var(--line); background: #f8fbf9; color: var(--ink); font-size: 12px; font-weight: 750; }
    .playbook-history-legend:empty { display: none; }
    .playbook-history-legend-item { display: inline-flex; align-items: center; gap: 8px; }
    .playbook-history-legend-swatch { width: 24px; height: 12px; border: 1px solid rgba(23,51,45,.16); }
    .playbook-commentary { margin-top: 14px; padding: 18px 20px; border: 1px solid var(--line); border-left: 5px solid var(--teal); border-radius: 0; background: #edf7f4; box-shadow: 0 10px 26px rgba(23,51,45,.10); font-size: 16px; line-height: 1.58; }
    .playbook-commentary.neutral { border-left-color: var(--water); background: #edf5f8; }
    .playbook-event-list { list-style: none; margin: 14px 0 0; padding: 0; overflow: visible; }
    .playbook-event-item { display: grid; grid-template-columns: 30px minmax(0,1fr) auto; gap: 9px; align-items: center; margin-top: 8px; padding: 11px 12px; border: 1px solid rgba(23,51,45,.12); border-left-width: 5px; border-radius: 0; background: #fff; box-shadow: 0 6px 18px rgba(23,51,45,.07); }
    .playbook-summary-detail { display: block; margin: 14px -20px -18px; padding: 14px 20px; border-top: 1px solid rgba(23,51,45,.18); background: rgba(255,255,255,.42); }
    .playbook-event-item.aligned { border-color: #83b99f; border-left-color: #16804c; background: #eaf7ef; }
    .playbook-event-item.misaligned { border-color: #d99b90; border-left-color: #b83d2f; background: #fff0ed; }
    .event-expectation { display: block; margin-top: 6px; color: var(--ink); font-size: 13px; line-height: 1.4; }
    .event-expectation strong { font-weight: 850; }
    .playbook-event-item.aligned .event-expectation strong { color: #167044; }
    .playbook-event-item.misaligned .event-expectation strong { color: #b5352b; }
    .playbook-event-icon { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; background: rgba(255,255,255,.72); border: 1px solid rgba(23,51,45,.12); border-radius: 50%; font-size: 16px; }
    .event-change { display: inline-flex; align-items: center; justify-content: flex-end; gap: 5px; white-space: nowrap; font-weight: 850; }
    .event-change.up { color: #167044; }
    .event-change.down { color: #b5352b; }
    .event-change.flat { color: var(--muted); }
    .event-change-arrow { display: inline-block; line-height: 1; text-align: center; }
    /* ---- window frame transition ---- */
    .window-frame { transition: opacity 380ms ease, transform 380ms ease; }
    .window-frame.sliding-left { opacity: 0; transform: translateX(-32px); pointer-events: none; }
    .window-frame.sliding-right { opacity: 0; transform: translateX(32px); pointer-events: none; }
    .window-frame.sliding-in-left { animation: slideInLeft 380ms ease both; }
    .window-frame.sliding-in-right { animation: slideInRight 380ms ease both; }
    @keyframes slideInLeft { from { opacity: 0; transform: translateX(-32px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes slideInRight { from { opacity: 0; transform: translateX(32px); } to { opacity: 1; transform: translateX(0); } }
    /* Arrow nav for event window */
    .window-arrow { position: absolute; top: 50%; transform: translateY(-50%); width: 36px; height: 36px; border-radius: 50%; background: rgba(23,51,45,.82); color: white; display: none; align-items: center; justify-content: center; font-size: 18px; cursor: pointer; opacity: 0; transition: opacity 200ms; z-index: 5; border: none; padding: 0; }
    .window-arrow.available { display: flex; }
    .window-arrow:hover { background: rgba(23,32,38,.92); }
    .window-arrow-left { left: 8px; }
    .window-arrow-right { right: 8px; }
    .event-chart-wrap:hover .window-arrow.available { opacity: 1; }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      overflow-x: clip;
      background: var(--paper);
      color: var(--ink);
      font-family: "Rubik", sans-serif;
      font-optical-sizing: auto;
      font-weight: <weight>;
      font-style: normal;
      letter-spacing: 0;
    }
    main { width: min(1180px, calc(100vw - 28px)); margin: 0 auto; padding: 0 0 56px; overflow-x: clip; }
    .hero { position: relative; min-height: 72vh; display: flex; align-items: center; color: white; background: var(--forest); box-shadow: 0 0 0 100vmax var(--forest); clip-path: inset(0 -100vmax); border-bottom: 6px solid var(--sun); padding: 44px 0 34px; }
    .hero::after { content: ""; position: absolute; left: 0; right: 0; bottom: 18px; height: 1px; background: rgba(255,255,255,.22); }
    h1 { margin: 0; max-width: 1080px; font-size: clamp(44px, 7.8vw, 98px); line-height: .95; letter-spacing: 0; }
    .dek { margin: 24px 0 0; max-width: 1020px; color: #dcebe6; font-size: clamp(20px, 2.4vw, 30px); line-height: 1.32; }
    .slide { position: relative; min-height: 96vh; padding: 64px 0; border-bottom: 1px solid var(--line); opacity: 0; transform: translateY(18px); transition: opacity 520ms ease, transform 520ms ease; }
    .slide.visible { opacity: 1; transform: translateY(0); }
    .slide.transition-out { opacity: 0; transform: translateX(-40px); }
    .slide.transition-in { opacity: 1; transform: translateX(0); animation: slideFade 420ms ease both; }
    @keyframes slideFade { from { opacity: 0; transform: translateX(48px); } to { opacity: 1; transform: translateX(0); } }
    h2 { margin: 0; max-width: 1040px; font-size: clamp(30px, 4vw, 52px); line-height: 1.02; letter-spacing: 0; }
    h2::before { content: ""; display: block; width: 58px; height: 5px; margin-bottom: 14px; background: var(--water); }
    #pricing-grouping h2::before, #features h2::before { background: var(--teal); }
    #events h2::before { background: var(--hazard); }
    #playbook h2::before { background: var(--sun); }
    .section-copy { color: #46545f; font-size: 17px; line-height: 1.55; margin: 14px 0 0; max-width: 900px; }
    .toolbar, .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
    button { font: inherit; font-weight: 800; font-size: 12px; border: 1px solid var(--line); background: #fff; color: #41505a; border-radius: 999px; padding: 8px 11px; cursor: pointer; }
    button.active { background: var(--ink); border-color: var(--ink); color: white; }
    button:disabled { cursor: default; opacity: .7; }
    .viz-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-top: 12px; }
    .timeseries-grid { grid-template-columns: minmax(0, 1.15fr) minmax(0, .85fr); }
    .feature-grid { grid-template-columns: 1fr; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-top: 4px solid var(--water); border-radius: 0; padding: 18px; box-shadow: var(--shadow); min-width: 0; }
    #pricing-grouping .panel, #features .panel { border-top-color: var(--teal); }
    #events .panel { border-top-color: var(--hazard); }
    #playbook .panel { border-top-color: var(--sun); }
    .panel h3 { margin: 0 0 4px; font-size: 17px; }
    .sub, .note { color: var(--muted); font-size: 13px; line-height: 1.4; margin: 0 0 8px; }
    .footnote { color: var(--muted); font-size: 11px; line-height: 1.4; margin: 6px 0 0; font-style: italic; }
    .chart { width: 100%; height: 430px; display: block; overflow: hidden; }
    .canvas-chart { position: relative; cursor: crosshair; }
    .canvas-chart canvas, .canvas-chart > svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
    .canvas-chart > svg { overflow: visible; pointer-events: none; }
    .chart.tall { height: 520px; }
    .chart.map-companion-line { height: 516px; }
    .chart.rating-risk-line { height: 671px; }
    .chart.map-companion-map { height: 430px; }
    .chart.xtall { height: 620px; }
    .chart.compressing { transform-origin: left center; animation: compressLeft 360ms ease both; }
    @keyframes compressLeft { 0% { transform: scaleX(1); } 45% { transform: scaleX(.94); } 100% { transform: scaleX(1); } }
    .axis text { fill: var(--muted); font-size: 11px; }
    .axis path, .axis line, .grid line { stroke: #d4dfda; }
    .grid path { display: none; }
    .event-line { stroke: #172026; stroke-width: 1.5; stroke-dasharray: 5 5; }
    .event-line { pointer-events: none; }
    .line { fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
    .example-county { pointer-events: stroke; }
    .line.background { opacity: .18; }
    .band { opacity: .18; }
    .band.background { opacity: .05; }
    .band { pointer-events: none; }
    .county { stroke: white; stroke-width: .22; vector-effect: non-scaling-stroke; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 12px; color: var(--muted); font-size: 12px; align-items: center; margin-top: 8px; }
    .scale-legend { width: min(100%, 320px); display: grid; grid-template-columns: 70px 1fr 70px; gap: 8px; align-items: center; }
    .scale-bar { height: 10px; border-radius: 999px; border: 1px solid rgba(23,32,38,.12); }
    .scale-legend span:last-child { text-align: right; }
    .swatch { width: 16px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 5px; vertical-align: -1px; }
    .takeaway { margin-top: 16px; border: 1px solid #b9d8ce; border-left: 5px solid var(--teal); border-radius: 0; background: #edf7f3; padding: 17px 19px; box-shadow: 0 10px 26px rgba(23,51,45,.09); font-size: 18px; line-height: 1.5; font-weight: 750; }
    .takeaway.segmented { padding: 0; }
    .takeaway-section { display: block; padding: 15px 19px; }
    .takeaway-section + .takeaway-section { border-top: 1px solid #b9d8ce; background: rgba(255,255,255,.34); }
    .sources { font-size: 12px; color: var(--muted); line-height: 1.5; margin-top: 14px; }
    .sources a { color: #205f90; }
    .tooltip { position: fixed; display: none; max-width: 300px; background: #172026; color: white; padding: 9px 10px; border-radius: 0; box-shadow: 0 8px 22px rgba(23,32,38,.28); font-size: 12px; line-height: 1.35; pointer-events: none; z-index: 10; }
    #county-results { border-radius: 0 !important; box-shadow: 0 8px 20px rgba(23,51,45,.10); }
    .county-line-label { font-size: 10px; fill: var(--muted); pointer-events: none; }
    .feature-line-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px 18px; min-height: 24px; margin: -4px 0 8px; color: var(--ink); font-size: 11px; font-weight: 700; }
    .feature-line-legend-item { display: inline-flex; align-items: center; gap: 7px; }
    .feature-line-key { width: 28px; border-top: 3px solid; }
    .feature-line-key.lower { border-top-style: dashed; }
    @media (max-width: 900px) {
      .viz-grid { grid-template-columns: 1fr; }
      .timeseries-grid { grid-template-columns: 1fr; }
      .hero { min-height: auto; }
      h1 { font-size: clamp(42px, 12vw, 66px); }
      .dek { font-size: 19px; line-height: 1.42; }
      .chart, .chart.tall, .chart.xtall { height: 360px; }
      .chart.map-companion-line { height: 384px; }
      .chart.rating-risk-line { height: 499px; }
      .chart.map-companion-map { height: 320px; }
      .hazard-rating-grid { grid-template-columns: 1fr; }
      .hazard-rating-specific { grid-template-columns: 1fr; }
      .playbook-event-item { grid-template-columns: 30px minmax(0,1fr); }
      .event-change { grid-column: 2; justify-content: flex-start; }
    }
  </style>
</head>
<body>
<main>
  <section class="hero" id="top">
    <div>
      <h1 id="t-hero-h1"></h1>
      <p class="dek" id="t-hero-dek"></p>
    </div>
  </section>

  <section class="slide" id="pricing">
    <h2 id="t-pricing-h2"></h2>

    <!-- County PPSF YoY histories — full width, footnotes below chart -->
    <div class="panel" style="margin-top:12px;">
      <h3 id="t-scatter-title"></h3>
      <p class="sub" id="t-scatter-sub"></p>
      <div id="score-scatter" class="chart canvas-chart" style="height:340px;">
        <canvas id="score-scatter-canvas" aria-label="Monthly Median PPSF YoY lines for U.S. counties"></canvas>
        <svg id="score-scatter-axes" aria-hidden="true"></svg>
      </div>
      <p class="footnote" id="t-scatter-fn1"></p>
      <p class="footnote" id="t-scatter-fn2"></p>
    </div>
    <div class="takeaway" id="score-scatter-takeaway"></div>
  </section>

  <section class="slide" id="pricing-grouping">
    <h2 id="t-pricing-grouping-subtitle"></h2>
    <p class="section-copy" id="t-pricing-nri-placeholder" style="margin-top:6px;"></p>

    <!-- PPSF YoY grouped by risk rating + map — side by side, hazard toggle on right -->
    <div class="panel" style="margin-top:16px;">
      <h3 id="t-pricing-card-title"></h3>
      <p class="sub" id="t-pricing-card-text"></p>
      <div class="control-bar" id="rating-hazard-sidebar-right">
        <div class="sidebar-label" id="t-hazard-sidebar-label"></div>
      </div>
      <div class="viz-grid pricing-viz-grid">
        <div>
          <svg id="rating-scatter" class="chart map-companion-line rating-risk-line"></svg>
        </div>
        <div class="map-with-legend">
          <svg id="rating-map" class="chart map-companion-map"></svg>
          <div class="legend" id="rating-map-legend"></div>
        </div>
      </div>
    </div>
    <div class="takeaway" id="pricing-takeaway"></div>
    <div class="sources" id="t-pricing-sources"></div>
  </section>

  <section class="slide" id="events">
    <h2 id="t-events-h2"></h2>
    <p class="section-copy" id="t-events-copy"></p>

    <!-- Event window + affected map in one card, risk toggle on right, arrow nav for windows -->
    <div class="panel" style="margin-top:12px;">
      <h3 id="t-events-card-title"></h3>
      <div class="control-bar" id="risk-frame-sidebar-right">
        <div class="sidebar-label" id="t-risk-sidebar-label"></div>
        <div class="toggle-row" id="risk-rating-toggles"></div>
        <button id="risk-play-button" type="button">&#9654; Play</button>
      </div>
      <div class="viz-grid timeseries-grid">
        <div class="event-chart-wrap" style="position:relative;">
          <div id="event-window-frame" class="window-frame">
            <svg id="event-window" class="chart map-companion-line"></svg>
          </div>
          <button class="window-arrow window-arrow-left" id="event-arrow-left" type="button">&#8592;</button>
          <button class="window-arrow window-arrow-right" id="event-arrow-right" type="button">&#8594;</button>
        </div>
        <div>
          <svg id="affected-map" class="chart map-companion-map"></svg>
          <div class="county-count" id="affected-county-count"></div>
        </div>
      </div>
      <div class="takeaway" id="event-window-takeaway" style="margin-top:10px;"></div>
    </div>
    <div class="takeaway" id="event-takeaway"></div>
    <div class="sources" id="t-events-sources"></div>
  </section>

  <section class="slide" id="features">
    <h2 id="t-features-h2"></h2>
    <p class="section-copy" id="t-features-copy"></p>

    <!-- Chart + map side by side, risk group toggle on right -->
    <div class="panel" style="margin-top:12px;">
      <div class="control-bar" id="feature-risk-sidebar">
        <div class="sidebar-label" id="t-feature-sidebar-label"></div>
      </div>
      <div class="viz-grid timeseries-grid">
        <div>
          <svg id="feature-event-window" class="chart map-companion-line"></svg>
          <div id="feature-line-legend" class="feature-line-legend"></div>
        </div>
        <div>
          <svg id="feature-county-map" class="chart map-companion-map"></svg>
        </div>
      </div>
      <div id="county-features-card" style="margin-top:16px; display:none; border-top:1px solid var(--line); padding-top:14px;">
        <h3 id="county-features-title"></h3>
        <p class="sub" id="t-feature-corr-label" style="font-weight:700; margin-bottom:6px;"></p>
        <div id="within-group-correlations"></div>
      </div>
    </div>

    <div class="takeaway" id="feature-takeaway"></div>
    <div class="sources" id="t-features-sources"></div>
  </section>

  <section class="slide" id="playbook">
    <h2 id="t-playbook-h2"></h2>

    <!-- County search, fixed map viewport, hazard profile, and housing history -->
    <div class="panel" style="margin-top:12px;">
      <div style="margin-bottom:10px;">
        <input type="text" id="county-search" style="padding:9px 14px; border:1px solid var(--line); border-radius:999px; font-size:13px; width:min(400px,100%); background:#fff;">
      </div>
      <div id="county-results" style="max-height:200px; overflow-y:auto; margin-bottom:10px; border:1px solid var(--line); border-radius:6px; display:none;"></div>
      <div class="playbook-map-wrap">
        <svg id="county-selection-map" class="chart"></svg>
        <div class="playbook-map-controls">
          <button id="playbook-map-zoom-in" type="button" title="Zoom in" aria-label="Zoom in">+</button>
          <button id="playbook-map-zoom-minus" type="button" title="Zoom out" aria-label="Zoom out">&#8722;</button>
          <button id="playbook-map-zoom-toggle" type="button" style="display:none;"></button>
        </div>
      </div>
      <div class="playbook-selected-county" id="playbook-selected-county-name"></div>

      <div id="playbook-display" style="display:none; margin-top:20px; border-top:1px solid var(--line); padding-top:16px;">
        <div class="hazard-rating-grid" id="playbook-hazard-ratings"></div>
        <h3 id="t-playbook-history-title" style="margin-top:18px;"></h3>
        <svg id="playbook-ppsf-history" class="chart tall"></svg>
        <div class="playbook-history-legend" id="playbook-history-legend"></div>
        <div class="playbook-commentary" id="playbook-event-commentary"></div>
      </div>
    </div>

    <div class="sources" id="t-playbook-sources"></div>
  </section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
/* ============================================================
   TEXT — Every visible string on the page lives here.
   Edit any value below and rebuild to update the page.
   ============================================================ */
const TEXT = {
  // ---- Hero ----
  heroH1: "Are Climate Risks Priced Into Housing Markets?",
  heroDek: "Climate change results in more severe weather events and natural disasters that cause substantial damage to properties and in extreme cases, devastate local communities. What does all this mean to you as a homeowner?",

  // ---- Pricing section ----
  pricingH2: "To Begin: What Does Growth in Housing Markets Look Like Across the United States?",
  scatterTitle: "Median Price-Per-Square-Foot (PPSF) Year-Over-Year (YoY) by County",
  scatterSub: "Each line represents a county's monthly Median PPSF YoY from 2016 through 2025.",
  scatterFootnote1: "* Outliers beyond the 1st–99th percentile are excluded.",
  scatterFootnote2: "* Only counties with a valid observation in every month from January 2016 through December 2025 are included.",
  pricingScoreScatterTakeaway: "<span class=\"takeaway-section\">From county-level median house price growth over the last 10 years, there is significant variation and there doesn't seem to be a clear pattern.</span><span class=\"takeaway-section\">However, the impact of climate change is uneven across the country, so looking from a climate angle might reveal a more meaningful pattern.</span>",
  pricingGroupingSubtitle: "A Climate Perspective: What Does House Price Growth Look Like When Grouping Counties by Climate Risk?",
  pricingNriPlaceholder: "The FEMA National Risk Index (NRI) serves as a measure of climate-related risk exposure. It summarizes a county's expected annual loss, social vulnerability, and community resilience across natural hazards. Counties are assigned a risk rating along a scale from \"Very Low\" to \"Very High\".",
  pricingCardTitle: "Median PPSF YoY by Climate Risk",
  pricingCardText: "House price growth of counties grouped by their FEMA National Risk Index (NRI) risk rating. Risk ratings are available for specific hazards and for overall climate risk.",
  hazardSidebarLabel: "Hazard type",
  pricingTakeaway: "<span class=\"takeaway-section\">A pattern now emerges: Counties with higher risk tend to show lower levels of house price growth.</span><span class=\"takeaway-section\">Housing markets are influenced by events across time. With respect to climate events, is there a shift in house price growth?</span>",
  pricingSources: 'Sources: <a href="https://hazards.fema.gov/nri/" target="_blank" rel="noopener">FEMA National Risk Index</a>, local mart <code>data/quoll.duckdb: mart.nri_county_risk</code>; <a href="https://www.redfin.com/news/data-center/" target="_blank" rel="noopener">Redfin Data Center</a>, local mart <code>mart.redfin_county_monthly</code>. The charts use monthly <code>MEDIAN_PPSF_YOY</code> observations from January 2016 through December 2025 and include only counties with complete data throughout that period.',

  // ---- Events section ----
  eventsH2: "Observing Climate Events: What Happens to House Price Growth?",
  eventsCopy: "Severe weather incidents and natural disasters can change the housing market outlook in its area as buyers and sellers re-evaluate their positions based on the perceived increased risk.",
  eventsCardTitle: "Median PPSF YoY Around Extreme Climate Events",
  riskSidebarLabel: "Risk rating",
  eventWindowATakeaway: "Past the 2-year mark post-event, house price growth momentum diverges across different risk groups. Growth weakening is more pronounced in higher risk groups.",
  eventWindowBTakeaway: "Around the 4-year mark post-event, house price growth across different risk groups begin to converge to the same level. It appears that the event’s impact fades out from view eventually.",
  eventsTakeaway: "<span class=\"takeaway-section\">In higher-risk counties, there is a time lag after an event before house price growth declines significantly. Homeowners who made it through the period of weakness then experienced some subsequent recovery.</span><span class=\"takeaway-section\">The wide band within each risk group shows that counties are hardly uniform, even within the same risk category. Why does this variation exist?</span>",
  eventsSources: "Sources: local marts <code>mart.fema_disaster_declarations</code>, <code>mart.noaa_storm_events</code>, <code>mart.redfin_county_monthly</code>, and <code>mart.nri_county_risk</code>.",

  // ---- Features section ----
  featuresH2: "What Sets Apart Counties Within the Same Risk Group?",
  featuresCopy: "A county's features can make its housing market more vulnerable or resilient to destructive weather events, and also influence its housing market.",
  featureSidebarLabel: "Risk Rating",
  featureCorrLabel: "County features most correlated with relative position within its risk group",
  featuresTakeaway: "<span class=\"takeaway-section\">Comparing counties with higher and lower house price growth levels within the same risk group, we can see that they have distinctly different features. A county's unique combination of features matter to its housing market growth in the face of climate risks.</span><span class=\"takeaway-section\">Given the significance of climate risk to housing markets, we can paint a picture of a county's climate risk that will be invaluable to homeowners.</span>",
  featuresSources: "Sources: local marts <code>mart.acs_county_economic_annual</code>, <code>mart.acs_county_demographic_annual</code>, <code>mart.acs_county_affordability_annual</code>, <code>mart.ncei_county_weather_monthly</code>, and <code>mart.nri_county_risk</code>. Cost components are midpoint estimates from ACS cost buckets.",

  // ---- Playbook section ----
  playbookH2: "Climate Playbook: What to Know About Your County's Climate Exposure",
  playbookSearchPlaceholder: "Search for a county by name, state, or FIPS…",
  playbookHistoryTitle: "Monthly Median PPSF YoY Over the Past 10 Years",
  playbookZoomOut: "Zoom out",
  playbookZoomIn: "Zoom to county",
  playbookEventLegend: "Extreme event period",
  playbookMissingDataLegend: "Missing county data",
  playbookSeriesLegend: "County Median PPSF YoY",
  playbookTakeaways: {
    noEvents: "<strong>{county} had zero extreme climate events in the last 10 years.</strong><span class=\"playbook-summary-detail\">{expectation}</span>",
    eventSummary: "<strong>{county} had {eventCount} extreme climate {eventNoun} in the last 10 years.</strong><ul class=\"playbook-event-list\">{details}</ul>",
    insufficientHistory: "{county} had insufficient housing data around the time of its past extreme climate events to measure a change.",
    eventDetail: "<li class=\"playbook-event-item {alignmentState}\"><span class=\"playbook-event-icon\">{icon}</span><span><strong>{name}</strong><br><span class=\"event-date\">{start} to {end}</span><span class=\"event-expectation\"><strong>{comment}</strong></span></span>{result}</li>",
    eventChange: "{direction} {magnitude} pp post-event",
    eventNoChange: "0.0 pp post-event",
    groupExpectation: "By year 3, counties with {risk} climate risk typically {groupBehavior} versus their pre-event-year level.",
    groupBehaviorChange: "{direction} by about {magnitude} percentage points",
    groupBehaviorFlat: "remain broadly steady ({magnitude} percentage-point change)",
    expectedBehavior: "This is expected for {risk} climate risk because counties in this group typically {groupBehavior} by year 3.",
    unexpectedBehavior: "This is not expected for {risk} climate risk because counties in this group typically {groupBehavior} by year 3, while this county's house-price growth {observedBehavior}.",
    unavailableExpectation: "A group-level post-event expectation is unavailable for this NRI rating.",
  },
  playbookTakeawayTerms: {
    event: "event",
    events: "events",
    declined: "declined",
    decline: "decline",
    increased: "increased",
    increase: "increase",
    unchanged: "was broadly unchanged",
    aligned: "broadly in line",
    notAligned: "not clearly in line",
    down: "down",
    up: "up",
    littleChanged: "little changed",
    insufficient: "insufficient pre/post data",
  },
  playbookSources: "Sources: FEMA National Risk Index and local mart <code>mart.nri_county_risk</code>; Redfin Data Center and local mart <code>mart.redfin_county_monthly</code>; local marts <code>mart.fema_disaster_declarations</code> and <code>mart.noaa_storm_events</code>.",
  riskImpacts: {
    "Very Low": "Counties with Very Low climate risk tend to maintain steady house price growth around climate events, with minimal disruption to market momentum.",
    "Low": "Counties with Low climate risk typically see modest softening of house price growth about two years after the event, but generally recover within three years.",
    "Medium": "Counties with Medium climate risk experience noticeable softening of house price growth around the two-year mark after the event.",
    "High": "Counties with High climate risk see significant deceleration in house price growth following an event, with the decline primarily occurring 18-24 months after event end.",
    "Very High": "Counties with Very High climate risk face substantial impacts on house price growth, with softening trends that can persist for several years after events.",
  },
};

/* Populate all text elements from the TEXT object above.
   Each entry maps a TEXT key to an element id. Uses innerHTML
   so sources with <a>/<code> tags render correctly. */
function hydrateText() {
  const map = {
    heroH1: "t-hero-h1",
    heroDek: "t-hero-dek",
    pricingH2: "t-pricing-h2",
    scatterTitle: "t-scatter-title",
    scatterSub: "t-scatter-sub",
    scatterFootnote1: "t-scatter-fn1",
    scatterFootnote2: "t-scatter-fn2",
    pricingScoreScatterTakeaway: "score-scatter-takeaway",
    pricingTakeaway: "pricing-takeaway",
    pricingGroupingSubtitle: "t-pricing-grouping-subtitle",
    pricingNriPlaceholder: "t-pricing-nri-placeholder",
    pricingCardTitle: "t-pricing-card-title",
    pricingCardText: "t-pricing-card-text",
    hazardSidebarLabel: "t-hazard-sidebar-label",
    pricingSources: "t-pricing-sources",
    eventsH2: "t-events-h2",
    eventsCopy: "t-events-copy",
    eventsCardTitle: "t-events-card-title",
    riskSidebarLabel: "t-risk-sidebar-label",
    eventsTakeaway: "event-takeaway",
    eventsSources: "t-events-sources",
    featuresH2: "t-features-h2",
    featuresCopy: "t-features-copy",
    featureSidebarLabel: "t-feature-sidebar-label",
    featureCorrLabel: "t-feature-corr-label",
    featuresTakeaway: "feature-takeaway",
    featuresSources: "t-features-sources",
    playbookH2: "t-playbook-h2",
    playbookHistoryTitle: "t-playbook-history-title",
    playbookSources: "t-playbook-sources",
  };
  for (const [key, id] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (el && TEXT[key] != null) {
      el.innerHTML = TEXT[key];
      el.classList.toggle("segmented", Boolean(el.querySelector(".takeaway-section")));
    }
  }
  document.getElementById("county-search").placeholder = TEXT.playbookSearchPlaceholder;
}

const HAZARD_ICONS = {
  overall: "\u{1F30E}",
  river_flood: "\u{1F30A}",
  tornado: "\u{1F32A}",
  wildfire: "\u{1F525}",
  hail: "\u{1F9CA}",
  earthquake: "\u{1F4A5}",
};

const DATA = __PAYLOAD__;
const RISK_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"];
const RISK_COLORS = {"Very Low":"#16803c","Low":"#79b851","Medium":"#e0b33b","High":"#df7d2f","Very High":"#b42318"};
const COUNTY_LINE_COLOR = "#5b7a8a";
const FEATURE_HIGHER_LINE_COLOR = "#175d8f";
const FEATURE_LOWER_LINE_COLOR = "#8a4f7d";
const fmtPct = d3.format("+.1%");
const fmtShare = d3.format(".0%");
const fmtAxisPct = value => Math.abs(value) >= 10 ? `${value > 0 ? "+" : ""}${d3.format(".2s")(value * 100)}%` : fmtPct(value);
const fmtNum = d3.format(",.1f");
const fmtMoney = d3.format("$,.0f");
const parsePriceMonth = d3.utcParse("%Y-%m-%d");
const tooltip = d3.select("#tooltip");
const countyByFips = new Map(DATA.priceRisk.counties.map(d => [d.fips, d]));
let playbookCountyByFips = new Map();
let scoreHistoryData = null;
const deferredDataPromises = new Map();
let ratingHazard = "overall";
let scoreScatterState = null;
let scoreScatterRendered = false;
let scoreTooltipFrame = null;
let scoreRenderGeneration = 0;
let ratingSectionRendered = false;
let eventSectionRendered = false;
let featureSectionRendered = false;
let playbookSectionRendered = false;
// Event section state
let selectedRisk = "Very Low";
let riskTimer = null;
let riskAutoPaused = false;
let activeEventWindow = "A"; // "A" or "B"
// Features section state
let selectedFeatureRisk = "Medium";
let selectedFeatureCounty = null;
// Playbook state
let selectedCountyFips = null;
let playbookMapZoomed = false;
let playbookZoomBehavior = null;
let playbookMapTransform = d3.zoomIdentity;
let playbookSelectedTransform = d3.zoomIdentity;

function loadDeferredData(filename, globalName) {
  if (window[globalName]) return Promise.resolve(window[globalName]);
  if (deferredDataPromises.has(globalName)) return deferredDataPromises.get(globalName);
  const promise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = new URL(filename, document.baseURI).href;
    script.async = true;
    script.onload = () => window[globalName]
      ? resolve(window[globalName])
      : reject(new Error(`Deferred data file did not define ${globalName}`));
    script.onerror = () => reject(new Error(`Unable to load deferred data file: ${filename}`));
    document.head.appendChild(script);
  });
  deferredDataPromises.set(globalName, promise);
  return promise;
}

function hazardLabel(key) { return DATA.priceRisk.hazards.find(h => h.key === key)?.label || key; }
function hazardCounty(county, key) { return county?.hazards?.[key] || {}; }
function capValue(value, domain) {
  if (value == null || !Number.isFinite(value)) return null;
  return Math.max(domain[0], Math.min(domain[1], value));
}
function robustDomain(values) {
  const valid = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  if (!valid.length) return [0, 1];
  const lo = d3.quantileSorted(valid, .01);
  const hi = d3.quantileSorted(valid, .99);
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}
function outlierFreeValues(values) {
  const sorted = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  if (!sorted.length) return sorted;
  const lo = d3.quantileSorted(sorted, .01);
  const hi = d3.quantileSorted(sorted, .99);
  return sorted.filter(v => v >= lo && v <= hi);
}
function colorScale(values, color, robust = false) {
  const valid = values.filter(v => v != null).sort(d3.ascending);
  const domain = robust ? robustDomain(valid) : d3.extent(valid);
  return d3.scaleSequentialSymlog(domain[0] === domain[1] ? [domain[0] - 1, domain[1] + 1] : domain, color).constant(.05);
}
function scaleLegendHtml(domain, colorA, colorB, label, formatter = fmtNum) {
  return `<div class="scale-legend"><span>${formatter(domain[0])}</span><div class="scale-bar" style="background:linear-gradient(90deg,${colorA},${colorB})"></div><span>${formatter(domain[1])}</span></div><span>${label}</span>`;
}
function pctText(value) { return value == null ? "n/a" : fmtPct(value); }
function ratingValue(rating) { return RISK_ORDER.indexOf(rating) + 1; }
function sparsePctTicks(domain) {
  const [lo, hi] = domain;
  return [lo, 0, 0.1, 1, hi]
    .filter(v => v != null && Number.isFinite(v) && v >= lo && v <= hi)
    .filter((v, i, arr) => arr.findIndex(x => Math.abs(x - v) < 1e-9) === i);
}

function drawMap(svgId, fillFn, tooltipFn, legendId, legendHtml, clickFn) {
  const svg = d3.select(svgId);
  const width = svg.node().clientWidth || 520;
  const height = svg.node().clientHeight || 430;
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);
  svg.on("mouseleave", () => tooltip.style("display","none"));
  svg.append("g").selectAll("path")
    .data(DATA.geojson.features)
    .join("path")
    .attr("class","county")
    .attr("d", path)
    .attr("fill", d => fillFn(countyByFips.get(d.properties.fips), d.properties.fips))
    .style("cursor", clickFn ? "pointer" : null)
    .on("mousemove", (event, d) => {
      const html = tooltipFn(countyByFips.get(d.properties.fips), d.properties.fips);
      if (!html) {
        tooltip.style("display", "none");
        return;
      }
      tooltip.style("display","block").style("left", `${event.clientX+12}px`).style("top", `${event.clientY+12}px`).html(html);
    })
    .on("click", clickFn ? (event, d) => clickFn(d.properties.fips) : null);
  if (legendId) d3.select(legendId).html(legendHtml || "");
}

function drawScoreScatter() {
  const container = document.querySelector("#score-scatter");
  const canvas = document.querySelector("#score-scatter-canvas");
  const series = scoreHistoryData?.series || [];
  const months = (scoreHistoryData?.months || []).map(parsePriceMonth);
  if (!container || !canvas || !series.length || !months.length) return;
  const generation = ++scoreRenderGeneration;
  delete canvas.dataset.rendered;
  const width = container.clientWidth || 960, height = container.clientHeight || 340;
  const margin = {top: 18, right: 18, bottom: 46, left: 68};
  const p1 = DATA.priceRisk.summary.historyPpsfCapLower;
  const p99 = DATA.priceRisk.summary.historyPpsfCapUpper;
  const yDomain = [p1, p99];
  const x = d3.scaleUtc().domain(d3.extent(months)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);

  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  const svg = d3.select("#score-scatter-axes");
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const yTicks = sparsePctTicks(y.domain());
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(d3.utcYear.every(2)).tickFormat(d3.utcFormat("%Y")));
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickFormat(fmtAxisPct));
  svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Month");
  svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",20).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Monthly Median PPSF YoY");
  scoreScatterState = {container, series, months, x, y, yDomain, margin, width, height, renderedCount: 0};
  scoreScatterRendered = true;
  const batchSize = 120;
  let seriesIndex = 0;
  function drawCountyBatch() {
    if (generation !== scoreRenderGeneration) return;
    const batchEnd = Math.min(series.length, seriesIndex + batchSize);
    context.save();
    context.beginPath();
    context.rect(margin.left, margin.top, width - margin.left - margin.right, height - margin.top - margin.bottom);
    context.clip();
    context.beginPath();
    for (; seriesIndex < batchEnd; seriesIndex += 1) {
      const county = series[seriesIndex];
      let started = false;
      county.values.forEach((value, index) => {
        if (value == null || !Number.isFinite(value) || !months[index]) {
          started = false;
          return;
        }
        const px = x(months[index]);
        const py = y(capValue(value, yDomain));
        if (started) context.lineTo(px, py);
        else {
          context.moveTo(px, py);
          started = true;
        }
      });
    }
    context.strokeStyle = "rgba(91, 122, 138, 0.12)";
    context.lineWidth = 1.1;
    context.stroke();
    context.restore();
    scoreScatterState.renderedCount = batchEnd;
    canvas.dataset.progress = String(batchEnd);
    if (seriesIndex < series.length) requestAnimationFrame(drawCountyBatch);
    else canvas.dataset.rendered = "true";
  }
  requestAnimationFrame(drawCountyBatch);
}

function initScoreScatter() {
  const container = document.querySelector("#score-scatter");
  if (!container) return;
  container.addEventListener("mousemove", event => {
    if (!scoreScatterState) return;
    const clientX = event.clientX, clientY = event.clientY;
    if (scoreTooltipFrame != null) cancelAnimationFrame(scoreTooltipFrame);
    scoreTooltipFrame = requestAnimationFrame(() => {
      scoreTooltipFrame = null;
      const {series, months, x, y, yDomain, margin, width, height} = scoreScatterState;
      const rect = container.getBoundingClientRect();
      const px = clientX - rect.left, py = clientY - rect.top;
      if (px < margin.left || px > width - margin.right || py < margin.top || py > height - margin.bottom) {
        tooltip.style("display", "none");
        return;
      }
      const monthIndex = d3.bisector(d => d).center(months, x.invert(px));
      let nearest = null, nearestDistance = Infinity;
      for (const county of series.slice(0, scoreScatterState.renderedCount)) {
        const value = county.values[monthIndex];
        if (value == null || !Number.isFinite(value)) continue;
        const distance = Math.abs(y(capValue(value, yDomain)) - py);
        if (distance < nearestDistance) {
          nearest = county;
          nearestDistance = distance;
        }
      }
      if (!nearest || nearestDistance > 10) {
        tooltip.style("display", "none");
        return;
      }
      tooltip.style("display","block").style("left",`${clientX+12}px`).style("top",`${clientY+12}px`).html(`<strong>${countyDisplayName(nearest)}</strong>`);
    });
  });
  container.addEventListener("mouseleave", () => tooltip.style("display", "none"));
}

function drawRatingScatter() {
  const ratingHistory = DATA.priceRisk.ratingHistoriesByHazard[ratingHazard] || [];
  const data = ratingHistory.filter(d => d.median != null);
  data.forEach(d => { d.monthDate = parsePriceMonth(d.month); });
  const svg = d3.select("#rating-scatter");
  const width = svg.node().clientWidth || 520, height = svg.node().clientHeight || 620;
  const margin = {top: 18, right: 100, bottom: 54, left: 68};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const bandValues = data.flatMap(d => [d.q1, d.q3]).filter(v => v != null && Number.isFinite(v));
  const bandExtent = bandValues.length ? d3.extent(bandValues) : [0, 1];
  const bandSpan = bandExtent[1] - bandExtent[0];
  const bandPadding = bandSpan > 0 ? bandSpan * 0.05 : 0.01;
  const yDomain = [bandExtent[0] - bandPadding, bandExtent[1] + bandPadding];
  const x = d3.scaleUtc().domain(d3.extent(data, d => d.monthDate)).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
  const yTicks = sparsePctTicks(y.domain());
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(d3.utcYear.every(2)).tickFormat(d3.utcFormat("%Y")));
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).tickValues(yTicks).tickFormat(fmtAxisPct));
  svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",20).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text("Monthly Median PPSF YoY");
  const area = d3.area().x(d => x(d.monthDate)).y0(d => y(capValue(d.q1, yDomain))).y1(d => y(capValue(d.q3, yDomain)));
  const line = d3.line().x(d => x(d.monthDate)).y(d => y(capValue(d.median, yDomain)));
  const endLabels = [];
  for (const risk of RISK_ORDER) {
    const rows = data.filter(d => d.riskRating === risk).sort((a,b)=>d3.ascending(a.monthDate,b.monthDate));
    if (!rows.length) continue;
    svg.append("path").datum(rows).attr("class","band").attr("fill",RISK_COLORS[risk]).attr("d",area);
    svg.append("path").datum(rows).attr("class","line").attr("stroke",RISK_COLORS[risk]).attr("d",line);
    const last = rows.at(-1);
    endLabels.push({risk, last, targetY: y(capValue(last.median, yDomain))});
  }
  endLabels.sort((a, b) => d3.ascending(a.targetY, b.targetY));
  const labelGap = 19;
  const labelMin = margin.top + 8;
  const labelMax = height - margin.bottom - 8;
  endLabels.forEach((label, index) => {
    label.labelY = Math.max(label.targetY, index ? endLabels[index - 1].labelY + labelGap : labelMin);
  });
  if (endLabels.length && endLabels.at(-1).labelY > labelMax) {
    endLabels.at(-1).labelY = labelMax;
    for (let index = endLabels.length - 2; index >= 0; index -= 1) {
      endLabels[index].labelY = Math.min(endLabels[index].labelY, endLabels[index + 1].labelY - labelGap);
    }
  }
  for (const label of endLabels) {
    const labelX = x(label.last.monthDate) + 10;
    const text = label.risk;
    svg.append("line")
      .attr("x1", x(label.last.monthDate) + 2).attr("x2", labelX - 3)
      .attr("y1", label.targetY).attr("y2", label.labelY)
      .attr("stroke", RISK_COLORS[label.risk]).attr("stroke-width", 1.2);
    svg.append("rect").attr("x", labelX - 3).attr("y", label.labelY - 10)
      .attr("width", text.length * 6.7 + 7).attr("height", 17)
      .attr("fill", "white").attr("fill-opacity", 0.94).attr("rx", 3);
    svg.append("text").attr("x", labelX).attr("y", label.labelY + 3)
      .attr("fill", RISK_COLORS[label.risk]).attr("font-size", 12).attr("font-weight", 850).text(text);
  }
}

function drawRatingMap() {
  drawMap("#rating-map",
    (county, fips) => {
      if (!county) return "#ece7df";
      return RISK_COLORS[hazardCounty(county, ratingHazard).rating] || "#ece7df";
    },
    (county, fips) => {
      if (!county) return "";
      const label = hazardLabel(ratingHazard).replace(/\s+NRI$/i, "");
      return `<strong>${countyDisplayName(county)}</strong><br>${label} NRI rating: ${hazardCounty(county, ratingHazard).rating ?? "n/a"}`;
    },
    "#rating-map-legend",
    RISK_ORDER.map(r => `<span><span class="swatch" style="background:${RISK_COLORS[r]}"></span>${r}</span>`).join("")
  );
}

function drawLineChart(svgId, source, groupKey, horizonLimit, activeRisk = null, minMonth = -12, opts = {}) {
  const data = source.filter(d => d.month >= minMonth && d.month <= horizonLimit);
  const svg = d3.select(svgId);
  const width = svg.node().clientWidth || 700, height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: opts.marginRight ? Math.min(opts.marginRight, width * 0.3) : 96, bottom: 42, left: 58};
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const x = d3.scaleLinear().domain([minMonth, horizonLimit]).range([margin.left, width - margin.right]);
  const hideOthers = opts.hideOtherGroups;
  const domainData = hideOthers && activeRisk ? data.filter(d => d[groupKey] === activeRisk) : data;
  const values = domainData.flatMap(d => [d.q1, d.median, d.q3]).filter(v => v != null);
  if (opts.extraDomainValues) values.push(...opts.extraDomainValues);
  const y = d3.scaleLinear().domain(d3.extent(values)).nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  const useYears = horizonLimit > 12;
  const yearTicks = [minMonth, 0, ...d3.range(12, horizonLimit + 1, 12)];
  const axis = useYears
    ? d3.axisBottom(x).tickValues(yearTicks).tickFormat(d => d < 0 ? `${Math.abs(d / 12)}y pre` : d === 0 ? "event" : `${d / 12}y`)
    : d3.axisBottom(x).ticks(8);
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(axis);
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  svg.append("line").attr("class","event-line").attr("x1",x(0)).attr("x2",x(0)).attr("y1",margin.top).attr("y2",height-margin.bottom);
  const grouped = d3.group(data, d => groupKey ? d[groupKey] : "All affected counties");
  for (const [key, rows] of grouped) {
    rows.sort((a,b)=>d3.ascending(a.month,b.month));
    if (hideOthers && key !== activeRisk) continue;
    const isBackground = !hideOthers && activeRisk && RISK_ORDER.indexOf(key) < RISK_ORDER.indexOf(activeRisk);
    const isHidden = !hideOthers && activeRisk && RISK_ORDER.indexOf(key) > RISK_ORDER.indexOf(activeRisk);
    if (isHidden) continue;
    const color = groupKey === "riskRating" ? RISK_COLORS[key] : "#0f766e";
    const areaFn = d3.area().x(d=>x(d.month)).y0(d=>y(d.q1)).y1(d=>y(d.q3));
    const lineFn = d3.line().x(d=>x(d.month)).y(d=>y(d.median));
    svg.append("path").datum(rows).attr("class",`band ${isBackground ? "background" : ""}`).attr("fill",color).attr("d",areaFn);
    svg.append("path").datum(rows).attr("class",`line ${isBackground ? "background" : ""}`).attr("stroke",color).attr("d",lineFn);
    const last = rows.at(-1);
    if (last && !isBackground) svg.append("text").attr("x",x(last.month)+5).attr("y",y(last.median)+4).attr("fill",color).attr("font-size",12).attr("font-weight",800).text(key);
  }
  svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text(useYears ? "Years from event start / after event end" : "Months from event start / after event end");
  return {x, y, margin, width, height};
}

function initButtons() {
  // Hazard sidebar (right) for pricing section — vertical, uniform size, with icons
  const hazardContainer = d3.select("#rating-hazard-sidebar-right");
  hazardContainer.selectAll("button")
    .data(DATA.priceRisk.hazards).join("button")
    .html(d => `<span style="font-size:14px;">${HAZARD_ICONS[d.key]||""}</span> ${d.label}`)
    .classed("active", d => d.key === ratingHazard)
    .style("background", d => d.key === ratingHazard ? "var(--ink)" : null)
    .on("click", (event, d) => {
      ratingHazard = d.key;
      hazardContainer.selectAll("button").classed("active", x => x.key === ratingHazard)
        .style("background", x => x.key === ratingHazard ? "var(--ink)" : null);
      drawRatingScatter(); drawRatingMap();
    });

  // Event section: arrow nav for windows
  d3.select("#event-arrow-left").on("click", () => switchEventWindow("left"));
  d3.select("#event-arrow-right").on("click", () => switchEventWindow("right"));

  // Event section: risk rating sidebar (right, color-coded)
  const riskSidebar = d3.select("#risk-rating-toggles");
  riskSidebar.selectAll("button.risk-toggle")
    .data(RISK_ORDER).join("button")
    .attr("class", "risk-toggle")
    .text(d=>d)
    .style("border-left", d=>`4px solid ${RISK_COLORS[d]}`)
    .classed("active", d=>d===selectedRisk)
    .style("background", d=>d===selectedRisk ? RISK_COLORS[d] : null)
    .on("click",(event,d)=>{
      selectedRisk=d; clearInterval(riskTimer); riskAutoPaused=true;
      riskSidebar.selectAll("button.risk-toggle").classed("active",x=>x===selectedRisk)
        .style("background", x=>x===selectedRisk ? RISK_COLORS[x] : null);
      d3.select("#risk-play-button").classed("visible", true);
      renderEventSection();
    });
  d3.select("#risk-play-button").on("click", startRiskTimer);

  // Features risk group sidebar (right, color-coded)
  const featureSidebar = d3.select("#feature-risk-sidebar");
  featureSidebar.selectAll("button")
    .data(RISK_ORDER).join("button")
    .text(d=>d)
    .style("border-left", d=>`4px solid ${RISK_COLORS[d]}`)
    .classed("active", d=>d===selectedFeatureRisk)
    .style("background", d=>d===selectedFeatureRisk ? RISK_COLORS[d] : null)
    .on("click",(event,d)=>{
      selectedFeatureRisk=d; selectedFeatureCounty=null;
      featureSidebar.selectAll("button").classed("active",x=>x===selectedFeatureRisk)
        .style("background", x=>x===selectedFeatureRisk ? RISK_COLORS[x] : null);
      drawFeatureHeatmaps();
    });
}

function switchEventWindow(direction) {
  const newWindow = direction === "right" ? "B" : "A";
  if (newWindow === activeEventWindow) return;
  const frame = d3.select("#event-window-frame");
  const slideOut = direction === "right" ? "sliding-left" : "sliding-right";
  const slideIn = direction === "right" ? "sliding-in-right" : "sliding-in-left";
  frame.classed(slideOut, true);
  setTimeout(() => {
    activeEventWindow = newWindow;
    renderEventSection();
    frame.classed(slideOut, false).classed(slideIn, true);
    setTimeout(() => frame.classed(slideIn, false), 400);
  }, 200);
}

function median(values) {
  const valid = values.filter(v => v != null && Number.isFinite(v)).sort(d3.ascending);
  return valid.length ? d3.median(valid) : null;
}

function countyDisplayName(county) {
  if (!county) return "";
  const name = county.county || "";
  const suffix = county.state ? `, ${county.state}` : "";
  return suffix && name.endsWith(suffix) ? name : `${name}${suffix}`;
}

function featureExampleIsHigher(county) {
  return county.samplePosition.startsWith("Above") || county.samplePosition.startsWith("Higher");
}

function featureExampleRole(county) {
  return featureExampleIsHigher(county)
    ? (county.samplePosition.startsWith("Above") ? "above group median" : "higher trajectory")
    : (county.samplePosition.startsWith("Below") ? "below group median" : "lower trajectory");
}

function featureExampleColor(county) {
  return featureExampleIsHigher(county) ? FEATURE_HIGHER_LINE_COLOR : FEATURE_LOWER_LINE_COLOR;
}

function featureScaleHtml(contribution) {
  const score = contribution == null || !Number.isFinite(+contribution)
    ? null
    : Math.max(-1, Math.min(1, +contribution));
  const p = score == null ? null : 50 + score * 50;
  const markerPosition = p == null ? null : Math.max(3, Math.min(97, p));
  const marker = markerPosition == null ? "" : `<span class="feature-scale-arrow" style="left:${markerPosition}%;"></span>`;
  return `<div class="feature-scale"><span>Lower PPSF YoY</span><span class="feature-scale-bar">${marker}</span><span style="text-align:right;">Higher PPSF YoY</span></div>`;
}

function featurePercentileScaleHtml(percentile) {
  const p = percentile == null || !Number.isFinite(+percentile)
    ? null
    : Math.max(0, Math.min(100, +percentile));
  const markerPosition = p == null ? null : Math.max(3, Math.min(97, p));
  const marker = markerPosition == null ? "" : `<span class="feature-scale-arrow" style="left:${markerPosition}%;"></span>`;
  return `<div class="feature-scale"><span>Low value</span><span class="feature-scale-bar percentile">${marker}</span><span style="text-align:right;">High value</span></div>`;
}

// ---- render event section with two windows ----
function activeWindowData() {
  return activeEventWindow === "A" ? DATA.eventWindows.windowA : DATA.eventWindows.windowB;
}

function renderEventSection() {
  const wd = activeWindowData();
  d3.select("#event-arrow-left")
    .classed("available", activeEventWindow === "B")
    .attr("aria-hidden", activeEventWindow === "B" ? null : "true")
    .property("disabled", activeEventWindow !== "B");
  d3.select("#event-arrow-right")
    .classed("available", activeEventWindow === "A")
    .attr("aria-hidden", activeEventWindow === "A" ? null : "true")
    .property("disabled", activeEventWindow !== "A");
  d3.select("#risk-rating-toggles").selectAll("button.risk-toggle").classed("active", d => d === selectedRisk)
    .style("background", d => d === selectedRisk ? RISK_COLORS[d] : null);
  d3.select("#risk-play-button").classed("visible", riskAutoPaused);
  const ta = activeEventWindow === "A" ? TEXT.eventWindowATakeaway : TEXT.eventWindowBTakeaway;
  d3.select("#event-window-takeaway").text(ta);
  drawLineChart("#event-window", wd.byRating, "riskRating",
    activeEventWindow === "A" ? 36 : 60,
    selectedRisk,
    -12
  );
  drawAffectedMap(wd);
}

function drawAffectedMap(wd) {
  const affected = new Map((wd.affectedCounties || []).map(d => [d.fips, d.riskRating]));
  const selectedCount = new Set((wd.affectedCounties || []).filter(d => d.riskRating === selectedRisk).map(d => d.fips)).size;
  d3.select("#affected-county-count").text(`${d3.format(",d")(selectedCount)} counties`);
  drawMap("#affected-map",
    (county, fips) => affected.get(fips) === selectedRisk ? RISK_COLORS[selectedRisk] : "#e6dfd5",
    (county, fips) => affected.get(fips) === selectedRisk ? `<strong>${countyDisplayName(county) || fips}</strong><br>NRI rating: ${selectedRisk}` : "",
    null, null
  );
}

function startRiskTimer() {
  clearInterval(riskTimer);
  riskAutoPaused = false;
  d3.select("#risk-play-button").classed("visible", false);
  riskTimer = setInterval(() => {
    selectedRisk = RISK_ORDER[(RISK_ORDER.indexOf(selectedRisk) + 1) % RISK_ORDER.length];
    d3.select("#risk-rating-toggles").selectAll("button.risk-toggle").classed("active", d => d === selectedRisk)
      .style("background", d => d === selectedRisk ? RISK_COLORS[d] : null);
    renderEventSection();
  }, 3200);
}

// ---- features section ----
function drawFeatureHeatmaps() {
  d3.select("#feature-risk-sidebar").selectAll("button").classed("active", d => d === selectedFeatureRisk)
    .style("background", d => d === selectedFeatureRisk ? RISK_COLORS[d] : null);
  const wd = DATA.eventWindows.windowA;
  const examples = (wd.exampleCountyLines || []).filter(d => d.riskRating === selectedFeatureRisk);
  // Collect county line values so the y-domain encompasses them
  const countyVals = examples.flatMap(d => d.values.filter(v => v.month >= -12 && v.month <= 36 && v.value != null).map(v => v.value));
  // Only show the selected risk group (hideOtherGroups=true)
  const {x, y, margin, width, height} = drawLineChart("#feature-event-window", wd.byRating, "riskRating", 36, selectedFeatureRisk, -12, {hideOtherGroups: true, extraDomainValues: countyVals, marginRight: 150});

  // Draw stratified sample county lines (different color from median, with labels)
  // Reuse the same y scale from drawLineChart so county lines align with the IQR band
  const svg = d3.select("#feature-event-window");
  const lineFn = d3.line().defined(d => d.value != null).x(d => x(d.month)).y(d => y(d.value));
  const isHigherExample = d => d.samplePosition.startsWith("Above") || d.samplePosition.startsWith("Higher");
  const exampleRole = d => isHigherExample(d)
    ? (d.samplePosition.startsWith("Above") ? "above group median" : "higher trajectory")
    : (d.samplePosition.startsWith("Below") ? "below group median" : "lower trajectory");
  const exampleColor = d => isHigherExample(d) ? FEATURE_HIGHER_LINE_COLOR : FEATURE_LOWER_LINE_COLOR;
  const exampleByFips = new Map(examples.map(example => [example.fips, example]));

  d3.select("#feature-line-legend").html([
    ...examples.map(d => `<span class="feature-line-legend-item"><span class="feature-line-key ${isHigherExample(d) ? "higher" : "lower"}" style="border-color:${exampleColor(d)}"></span>${d.county} (${exampleRole(d)})</span>`),
    `<span class="feature-line-legend-item"><span class="feature-line-key" style="border-color:${RISK_COLORS[selectedFeatureRisk]}"></span>${selectedFeatureRisk} group median</span>`,
  ].join(""));

  const g = svg.append("g");
  g.selectAll("path.example-county").data(examples).join("path")
    .attr("class","line example-county")
    .attr("stroke", exampleColor)
    .attr("stroke-dasharray", d => isHigherExample(d) ? null : "7 4")
    .attr("stroke-width", d => (selectedFeatureCounty && d.lineId === selectedFeatureCounty.lineId) ? 3.6 : 2.4)
    .attr("opacity", d => (selectedFeatureCounty && d.lineId === selectedFeatureCounty.lineId) ? 1 : 0.86)
    .attr("d", d => lineFn(d.values.filter(v => v.month >= -12 && v.month <= 36)))
    .style("cursor","pointer")
    .on("click",(event,d)=>{ selectedFeatureCounty=d; drawFeatureHeatmaps(); })
    .on("mousemove",(event,d)=>tooltip.style("display","block").style("left",`${event.clientX+12}px`).style("top",`${event.clientY+12}px`).html(`<strong>${countyDisplayName(d)}</strong>`))
    .on("mouseleave",()=>tooltip.style("display","none"));

  // Highlight selected county on map
  drawMap("#feature-county-map",
    (county, fips) => {
      const example = exampleByFips.get(fips);
      return example ? exampleColor(example) : "#d8d0c4";
    },
    (county, fips) => {
      const example = exampleByFips.get(fips);
      return example ? `<strong>${countyDisplayName(example)}</strong>` : "";
    },
    null, null
  );
  d3.select("#feature-county-map").selectAll("path.county")
    .classed("feature-tooltip-county", d => exampleByFips.has(d.properties.fips))
    .style("cursor", d => exampleByFips.has(d.properties.fips) ? "help" : "default")
    .attr("aria-label", d => {
      const example = exampleByFips.get(d.properties.fips);
      return example ? countyDisplayName(example) : null;
    });
  drawCountyFeaturePanelV2(examples);
}

function drawCountyFeaturePanelV2(examples) {
  if (!examples || examples.length !== 2) {
    d3.select("#county-features-card").style("display","none");
    return;
  }
  d3.select("#county-features-card").style("display","block");
  d3.select("#county-features-title").text(`${selectedFeatureRisk} Risk County Feature Comparison`);

  const featureMeta = DATA.features.featureMeta || {};
  const riskBins = (DATA.features.withinGroupFeatureBins || {})[selectedFeatureRisk] || {};
  const wgTop = ((DATA.features.withinGroupTopFeatures || {})[selectedFeatureRisk] || [])
    .slice(0, 10);
  if (wgTop.length) {
    const [countyA, countyB] = examples;
    const html2 = `<div class="feature-detail-list">
      <div class="feature-detail-header">
        <div>Feature</div>
        <div class="county-group" style="grid-column:span 3;"><span class="swatch" style="background:${featureExampleColor(countyA)}"></span>${countyDisplayName(countyA)} · ${featureExampleRole(countyA)}</div>
        <div class="county-group" style="grid-column:span 3;"><span class="swatch" style="background:${featureExampleColor(countyB)}"></span>${countyDisplayName(countyB)} · ${featureExampleRole(countyB)}</div>
        <div></div><div>Value</div><div>Percentile within risk group</div><div>Contribution to Median PPSF YoY</div><div>Value</div><div>Percentile within risk group</div><div>Contribution to Median PPSF YoY</div>
      </div>
      ${wgTop.map(c => {
        const fmt = featureMeta[c.feature]?.format || "number";
        const countyCells = examples.flatMap(county => {
          const bin = (riskBins[county.fips] || {})[c.feature];
          return [
            `<div style="text-align:right;">${formatFeatureVal(bin?.value, fmt)}</div>`,
            `<div>${featurePercentileScaleHtml(bin?.valuePercentile)}</div>`,
            `<div>${featureScaleHtml(bin?.contribution)}</div>`,
          ];
        }).join("");
        return `<div class="feature-detail-row"><div style="font-weight:700;">${c.feature}</div>${countyCells}</div>`;
      }).join("")}
    </div>`;
    d3.select("#within-group-correlations").html(html2);
  } else {
    d3.select("#within-group-correlations").text("Within-group bin data not available.");
  }
}

function formatFeatureVal(v, fmt) {
  if (v == null) return "n/a";
  if (fmt === "currency") return fmtMoney(v);
  if (fmt === "percent") return `${d3.format(",.2f")(v)}%`;
  if (fmt === "pct" || fmt === "signed_pct") return d3.format("+.2%")(v);
  if (fmt === "temperature_f") return `${d3.format(",.1f")(v)} °F`;
  if (fmt === "inches") return `${d3.format(",.2f")(v)} in`;
  return fmtNum(v);
}

// ---- playbook section ----
function updatePlaybookZoomControl() {
  const county = playbookCountyByFips.get(selectedCountyFips);
  const zoomed = playbookMapTransform.k > 1.01;
  d3.select("#playbook-map-zoom-toggle")
    .style("display", county || zoomed ? "inline-flex" : "none")
    .text(zoomed ? TEXT.playbookZoomOut : TEXT.playbookZoomIn);
}

function drawPlaybookMap(county = null, autoZoom = false) {
  const svg = d3.select("#county-selection-map");
  const width = svg.node().clientWidth || 1100;
  const height = svg.node().clientHeight || 380;
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);
  const group = svg.append("g");

  group.selectAll("path")
    .data(DATA.geojson.features)
    .join("path")
    .attr("class", "county")
    .attr("d", path)
    .attr("fill", d => d.properties.fips === county?.fips ? "#172026" : "#d8d0c4")
    .style("cursor", "pointer")
    .on("mousemove", (event, d) => {
      const profile = playbookCountyByFips.get(d.properties.fips);
      if (!profile) return;
      tooltip.style("display", "block")
        .style("left", `${event.clientX + 12}px`)
        .style("top", `${event.clientY + 12}px`)
        .html(`<strong>${countyDisplayName(profile)}</strong>`);
    })
    .on("mouseleave", () => tooltip.style("display", "none"))
    .on("click", (event, d) => {
      const profile = playbookCountyByFips.get(d.properties.fips);
      if (profile) selectPlaybookCounty(profile);
    });

  playbookZoomBehavior = d3.zoom()
    .scaleExtent([1, 12])
    .translateExtent([[-width * .35, -height * .35], [width * 1.35, height * 1.35]])
    .on("zoom", event => {
      playbookMapTransform = event.transform;
      playbookMapZoomed = event.transform.k > 1.01;
      group.attr("transform", event.transform);
      updatePlaybookZoomControl();
    });
  svg.call(playbookZoomBehavior).on("dblclick.zoom", null);
  svg.call(playbookZoomBehavior.transform, d3.zoomIdentity);

  if (county) {
    const selectedFeature = DATA.geojson.features.find(d => d.properties.fips === county.fips);
    if (selectedFeature) {
      const [cx, cy] = path.centroid(selectedFeature);
      const scale = county.state === "AK" ? 3.2 : county.state === "HI" ? 4.2 : 5.2;
      playbookSelectedTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(scale).translate(-cx, -cy);
      if (autoZoom) svg.transition().duration(520).call(playbookZoomBehavior.transform, playbookSelectedTransform);
    }
  } else {
    playbookSelectedTransform = d3.zoomIdentity;
    playbookMapTransform = d3.zoomIdentity;
    playbookMapZoomed = false;
    updatePlaybookZoomControl();
  }
}

function interpolateInternalHistory(source) {
  const history = source.map(d => ({...d, interpolated: Boolean(d.interpolated)}));
  const firstObservedIndex = history.findIndex(d => d.value != null);
  const lastObservedIndex = history.findLastIndex(d => d.value != null);
  if (firstObservedIndex < 0 || lastObservedIndex <= firstObservedIndex) return history;
  let index = firstObservedIndex + 1;
  while (index < lastObservedIndex) {
    if (history[index].value != null) {
      index += 1;
      continue;
    }
    const previousIndex = index - 1;
    let nextIndex = index;
    while (nextIndex <= lastObservedIndex && history[nextIndex].value == null) nextIndex += 1;
    if (nextIndex > lastObservedIndex) break;
    const monthSpan = d3.utcMonth.count(history[previousIndex].date, history[nextIndex].date);
    for (let gapIndex = index; gapIndex < nextIndex; gapIndex += 1) {
      const elapsed = d3.utcMonth.count(history[previousIndex].date, history[gapIndex].date);
      history[gapIndex].value = history[previousIndex].value
        + (history[nextIndex].value - history[previousIndex].value) * (elapsed / monthSpan);
      history[gapIndex].interpolated = true;
    }
    index = nextIndex + 1;
  }
  return history;
}

function drawPlaybookHistory(county) {
  const parseMonth = d3.utcParse("%Y-%m");
  const historyMonths = DATA.playbook.monthlyHistoryMonths || [];
  const historyValues = (DATA.playbook.monthlyHistoryValuesByFips || {})[county.fips] || [];
  const observedHistory = historyMonths.map((month, index) => ({
    month,
    value: historyValues[index] ?? null,
    date: parseMonth(month),
  })).filter(d => d.value != null);
  const historyStart = parseMonth(DATA.playbook.historyStart);
  const historyEnd = parseMonth(DATA.playbook.historyEnd);
  const historyDomainEnd = d3.utcDay.offset(d3.utcMonth.offset(historyEnd, 1), -1);
  const observedByMonth = new Map(observedHistory.map(d => [d.month, d.value]));
  const history = interpolateInternalHistory(d3.utcMonth.range(historyStart, d3.utcMonth.offset(historyEnd, 1)).map(date => {
    const month = d3.utcFormat("%Y-%m")(date);
    return {date, month, value: observedByMonth.has(month) ? observedByMonth.get(month) : null, interpolated: false};
  }));
  const observed = history.filter(d => d.value != null);
  const events = ((DATA.playbook.eventsByFips || {})[county.fips] || [])
    .map(d => ({
      eventKey: d[0], source: d[1], type: d[2], name: d[3], start: d[4], end: d[5],
      startDate: parseMonth(d[4]), endDate: parseMonth(d[5]),
    }));
  const svg = d3.select("#playbook-ppsf-history");
  const width = svg.node().clientWidth || 1050;
  const height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: 24, bottom: 42, left: 62};
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();

  const x = d3.scaleUtc().domain([historyStart, historyDomainEnd]).range([margin.left, width - margin.right]);
  const valueExtent = observed.length ? d3.extent(observed, d => d.value) : [-0.1, 0.1];
  const padding = Math.max((valueExtent[1] - valueExtent[0]) * 0.12, 0.01);
  const y = d3.scaleLinear().domain([valueExtent[0] - padding, valueExtent[1] + padding]).nice()
    .range([height - margin.bottom, margin.top]);

  svg.append("g").attr("class", "grid").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(""));

  const chartStart = x.domain()[0], chartEnd = x.domain()[1];
  svg.append("g").selectAll("rect.event-period")
    .data(events.filter(d => d.endDate >= chartStart && d.startDate <= chartEnd))
    .join("rect")
    .attr("class", "event-period")
    .attr("x", d => x(d3.max([d.startDate, chartStart])))
    .attr("y", margin.top)
    .attr("width", d => Math.max(3, x(d3.min([d.endDate, chartEnd])) - x(d3.max([d.startDate, chartStart]))))
    .attr("height", height - margin.top - margin.bottom)
    .attr("fill", "#df7d2f").attr("opacity", .16)
    .on("mousemove", (event, d) => tooltip.style("display", "block")
      .style("left", `${event.clientX + 12}px`).style("top", `${event.clientY + 12}px`)
      .html(`<strong>${normalCase(d.name || d.type)}</strong><br>${d3.utcFormat("%b %Y")(d.startDate)} to ${d3.utcFormat("%b %Y")(d.endDate)}<br>${normalCase(d.source)}`))
    .on("mouseleave", () => tooltip.style("display", "none"));

  svg.append("path").datum(history).attr("class", "line")
    .attr("stroke", "#0f766e").attr("stroke-width", 2.4)
    .attr("d", d3.line().defined(d => d.value != null).x(d => x(d.date)).y(d => y(d.value)));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(d3.utcYear.every(1)).tickFormat(d3.utcFormat("%Y")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  svg.append("text").attr("x", margin.left).attr("y", 13).attr("fill", "#66717b").attr("font-size", 11)
    .text("Median PPSF YoY");
  const legendItems = [{label: TEXT.playbookSeriesLegend, color: "#0f766e", opacity: 1, line: true}];
  if (events.length) legendItems.push({label: TEXT.playbookEventLegend, color: "#df7d2f", opacity: .22});
  d3.select("#playbook-history-legend").html(legendItems.map(item =>
    `<span class="playbook-history-legend-item"><span class="playbook-history-legend-swatch" style="background:${item.color};opacity:${item.opacity};${item.line ? "height:3px;border:none;" : ""}"></span>${item.label}</span>`
  ).join(""));
  if (!observed.length) {
    svg.append("text").attr("x", width / 2).attr("y", height / 2)
      .attr("text-anchor", "middle").attr("fill", "#66717b")
      .text(TEXT.playbookMissingDataLegend);
  }
  renderPlaybookCommentary(county, history, events);
}

function fillTextTemplate(template, values) {
  return Object.entries(values).reduce(
    (text, [key, value]) => text.replaceAll(`{${key}}`, value == null ? "" : String(value)),
    template || ""
  );
}

function normalCase(value) {
  const text = String(value || "").trim();
  if (!text) return "Unnamed event";
  if (text !== text.toUpperCase()) return text.charAt(0).toUpperCase() + text.slice(1);
  return text.toLowerCase().replace(/\b\w/g, letter => letter.toUpperCase())
    .replace(/\bFema\b/g, "FEMA").replace(/\bNoaa\b/g, "NOAA");
}

function eventIcon(event) {
  const label = `${event.type || ""} ${event.name || ""}`.toLowerCase();
  if (/earthquake|seismic/.test(label)) return "💥";
  if (/tornado|funnel/.test(label)) return "🌪️";
  if (/wildfire|forest fire|fire/.test(label)) return "🔥";
  if (/flood|surge|coastal/.test(label)) return "🌊";
  if (/hurricane|typhoon|tropical/.test(label)) return "🌀";
  if (/hail|ice/.test(label)) return "🧊";
  if (/winter|snow|blizzard|freeze/.test(label)) return "❄️";
  if (/drought|heat/.test(label)) return "☀️";
  if (/wind|storm|thunder|lightning/.test(label)) return "⛈️";
  return "⚠️";
}

function eventChangeHtml(delta, copy, terms) {
  if (delta == null) return `<span class="event-change flat">${terms.insufficient}</span>`;
  const magnitude = Math.abs(delta * 100);
  const intensity = Math.min(1, magnitude / 8);
  const opacity = 0.45 + intensity * 0.55;
  const size = 14 + Math.min(12, magnitude * 0.75);
  if (delta === 0) return `<span class="event-change flat"><span class="event-change-arrow">→</span>${copy.eventNoChange}</span>`;
  const isUp = delta > 0;
  const changeText = fillTextTemplate(copy.eventChange, {
    direction: isUp ? terms.up : terms.down,
    magnitude: magnitude.toFixed(1),
  });
  return `<span class="event-change ${isUp ? "up" : "down"}"><span class="event-change-arrow" style="font-size:${size}px;opacity:${opacity};">${isUp ? "↑" : "↓"}</span>${changeText}</span>`;
}

function eventChangeDirection(delta) {
  if (delta == null || !Number.isFinite(+delta)) return null;
  if (delta < -0.005) return "down";
  if (delta > 0.005) return "up";
  return "flat";
}

function riskGroupEventExpectation(risk, copy = TEXT.playbookTakeaways, terms = TEXT.playbookTakeawayTerms) {
  const rows = (DATA.eventWindows?.windowA?.byRating || [])
    .filter(row => row.riskRating === risk && row.median != null);
  const before = rows.filter(row => row.month >= -12 && row.month < 0).map(row => row.median);
  const after = rows.filter(row => row.month >= 25 && row.month <= 36).map(row => row.median);
  if (!before.length || !after.length) {
    return {delta: null, direction: null, description: copy.unavailableExpectation};
  }
  const delta = d3.median(after) - d3.median(before);
  const direction = eventChangeDirection(delta);
  const groupBehavior = fillTextTemplate(
    direction === "flat" ? copy.groupBehaviorFlat : copy.groupBehaviorChange,
    {
      direction: direction === "down" ? terms.decline : terms.increase,
      magnitude: Math.abs(delta * 100).toFixed(1),
    },
  );
  return {
    delta,
    direction,
    description: fillTextTemplate(copy.groupExpectation, {
      risk,
      groupBehavior,
    }),
    groupBehavior,
  };
}

function renderPlaybookCommentary(county, history, events) {
  const container = d3.select("#playbook-event-commentary");
  const risk = county.hazards?.overall?.rating || county.riskRating || "Unknown";
  const copy = TEXT.playbookTakeaways;
  const terms = TEXT.playbookTakeawayTerms;
  const countyLabel = countyDisplayName(county);
  const groupExpectation = riskGroupEventExpectation(risk, copy, terms);
  const expectation = groupExpectation.description;
  const alignmentForDelta = delta => {
    const observedDirection = eventChangeDirection(delta);
    if (!observedDirection || !groupExpectation.direction) return "neutral";
    const aligned = observedDirection === groupExpectation.direction;
    return aligned ? "aligned" : "misaligned";
  };
  if (!events.length) {
    container.attr("class", "playbook-commentary neutral");
    container.html(fillTextTemplate(copy.noEvents, {county: countyLabel, expectation}));
    return;
  }

  const assessments = events.map(event => {
    const preStart = d3.utcMonth.offset(event.startDate, -12);
    const thirdYearStart = d3.utcMonth.offset(event.endDate, 24);
    const postEnd = d3.utcMonth.offset(event.endDate, 36);
    const before = history.filter(d => d.value != null && !d.interpolated && d.date >= preStart && d.date < event.startDate).map(d => d.value);
    const after = history.filter(d => d.value != null && !d.interpolated && d.date > thirdYearStart && d.date <= postEnd).map(d => d.value);
    const beforeMedian = before.length >= 3 ? d3.median(before) : null;
    const afterMedian = after.length >= 6 ? d3.median(after) : null;
    const delta = beforeMedian == null || afterMedian == null ? null : afterMedian - beforeMedian;
    return {...event, delta, postMonths: after.length};
  });
  const measured = assessments.filter(d => d.delta != null);
  if (!measured.length) {
    container.attr("class", "playbook-commentary neutral");
    container.html(fillTextTemplate(copy.insufficientHistory, {county: countyLabel}));
    return;
  }
  const details = measured.map(d => {
    const alignmentState = alignmentForDelta(d.delta);
    const commentTemplate = alignmentState === "aligned" ? copy.expectedBehavior : copy.unexpectedBehavior;
    const observedDirection = eventChangeDirection(d.delta);
    const observedBehavior = observedDirection === "down" ? terms.declined : observedDirection === "up" ? terms.increased : terms.unchanged;
    return fillTextTemplate(copy.eventDetail, {
      icon: eventIcon(d),
      name: normalCase(d.name || d.type),
      start: d3.utcFormat("%b %Y")(d.startDate),
      end: d3.utcFormat("%b %Y")(d.endDate),
      result: eventChangeHtml(d.delta, copy, terms),
      alignmentState,
      comment: alignmentState === "neutral"
        ? copy.unavailableExpectation
        : fillTextTemplate(commentTemplate, {risk, groupBehavior: groupExpectation.groupBehavior, observedBehavior}),
    });
  }).join("");
  container.attr("class", "playbook-commentary");
  container.html(fillTextTemplate(copy.eventSummary, {
    county: countyLabel,
    eventCount: events.length,
    eventNoun: events.length === 1 ? terms.event : terms.events,
    details,
  }));
}

function renderPlaybookHazards(county) {
  const hazards = DATA.playbook.hazards || DATA.priceRisk.hazards || [];
  const renderRating = hazard => {
    const rating = county.hazards?.[hazard.key]?.rating || "No rating";
    const color = RISK_COLORS[rating] || "#66717b";
    return `<div class="hazard-rating-item"><span><span class="hazard-icon">${HAZARD_ICONS[hazard.key] || ""}</span>${hazard.label}</span><strong style="color:${color};">${rating}</strong></div>`;
  };
  const overall = hazards.find(hazard => hazard.key === "overall");
  const specific = hazards.filter(hazard => hazard.key !== "overall");
  d3.select("#playbook-hazard-ratings").html(
    `<div class="hazard-rating-overall">${overall ? renderRating(overall) : ""}</div>`
    + `<div class="hazard-rating-specific">${specific.map(renderRating).join("")}</div>`
  );
}

function selectPlaybookCounty(county) {
  selectedCountyFips = county.fips;
  playbookMapZoomed = true;
  d3.select("#playbook-display").style("display", "block");
  d3.select("#playbook-selected-county-name").style("display", "block").text(countyDisplayName(county));
  drawPlaybookMap(county, true);
  renderPlaybookHazards(county);
  drawPlaybookHistory(county);
}

function initPlaybook() {
  if (!DATA.playbook?.available) {
    d3.select("#county-search").property("disabled", true).property("placeholder", DATA.playbook?.message || "County data unavailable");
    return;
  }
  drawPlaybookMap();
  const input = d3.select("#county-search");
  const results = d3.select("#county-results");
  input.on("input", function() {
    const query = this.value.toLowerCase().trim();
    if (query.length < 2) { results.style("display", "none").html(""); return; }
    const matches = DATA.playbook.counties.filter(c =>
      c.county.toLowerCase().includes(query) || c.state.toLowerCase().includes(query) || c.fips.includes(query)
    ).slice(0, 20);
    results.style("display", matches.length ? "block" : "none").html("")
      .selectAll("div").data(matches).join("div")
      .style("padding", "8px 12px").style("cursor", "pointer")
      .style("border-bottom", "1px solid var(--line)").style("font-size", "13px")
      .html(d => `<strong>${countyDisplayName(d)}</strong> <span style="color:var(--muted);">(${d.hazards?.overall?.rating || "Unknown risk"})</span>`)
      .on("click", (event, d) => { selectPlaybookCounty(d); input.property("value", ""); results.style("display", "none").html(""); });
  });
  d3.select("#playbook-map-zoom-in").on("click", () => {
    if (playbookZoomBehavior) d3.select("#county-selection-map").transition().duration(220).call(playbookZoomBehavior.scaleBy, 1.5);
  });
  d3.select("#playbook-map-zoom-minus").on("click", () => {
    if (playbookZoomBehavior) d3.select("#county-selection-map").transition().duration(220).call(playbookZoomBehavior.scaleBy, 1 / 1.5);
  });
  d3.select("#playbook-map-zoom-toggle").on("click", () => {
    if (!playbookZoomBehavior) return;
    const target = playbookMapTransform.k > 1.01 ? d3.zoomIdentity : playbookSelectedTransform;
    d3.select("#county-selection-map").transition().duration(420).call(playbookZoomBehavior.transform, target);
  });
}

// ---- bootstrap ----
hydrateText();
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) entry.target.classList.add("visible","transition-in");
    else if (entry.boundingClientRect.top < 0) entry.target.classList.add("transition-out");
  });
}, { threshold: .18 });
document.querySelectorAll(".slide").forEach(el => observer.observe(el));
initButtons();
initScoreScatter();
function renderWhenNear(selector, render, rootMargin = "350px 0px") {
  const target = document.querySelector(selector);
  if (!target) return;
  const sectionObserver = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    Promise.resolve(render()).catch(error => {
      console.error(error);
      target.setAttribute("data-load-error", "true");
    });
    sectionObserver.disconnect();
  }, {rootMargin});
  sectionObserver.observe(target);
}
renderWhenNear("#score-scatter", async () => {
  scoreHistoryData = await loadDeferredData(
    "climate-risk-housing-county-history.js",
    "CLIMATE_RISK_HOUSING_COUNTY_HISTORY"
  );
  drawScoreScatter();
});
renderWhenNear("#pricing-grouping", () => {
  drawRatingScatter();
  drawRatingMap();
  ratingSectionRendered = true;
});
renderWhenNear("#events", () => {
  renderEventSection();
  startRiskTimer();
  eventSectionRendered = true;
});
renderWhenNear("#features", () => {
  drawFeatureHeatmaps();
  featureSectionRendered = true;
});
renderWhenNear("#playbook", async () => {
  DATA.playbook = await loadDeferredData(
    "climate-risk-housing-playbook.js",
    "CLIMATE_RISK_HOUSING_PLAYBOOK"
  );
  playbookCountyByFips = new Map(DATA.playbook.counties.map(d => [d.fips, d]));
  initPlaybook();
  playbookSectionRendered = true;
}, "1200px 0px");
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (scoreScatterRendered) drawScoreScatter();
    if (ratingSectionRendered) {
      drawRatingScatter();
      drawRatingMap();
    }
    if (eventSectionRendered) renderEventSection();
    if (featureSectionRendered) drawFeatureHeatmaps();
    if (playbookSectionRendered) {
      const playbookCounty = DATA.playbook?.counties?.find(c => c.fips === selectedCountyFips);
      drawPlaybookMap(playbookCounty || null, playbookCounty ? playbookMapZoomed : false);
      if (playbookCounty) drawPlaybookHistory(playbookCounty);
    }
  }, 150);
});
</script>
</body>
</html>"""


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        price_risk = build_price_risk(con)
        features = build_feature_payload(con)
        eligible_feature_fips_by_risk: dict[str, set[str]] = {}
        for risk in RISK_ORDER:
            top_labels = {
                item["feature"]
                for item in features.get("withinGroupTopFeatures", {}).get(risk, [])
            }
            county_bins = features.get("withinGroupFeatureBins", {}).get(risk, {})
            eligible_feature_fips_by_risk[risk] = {
                str(fips)
                for fips, values in county_bins.items()
                if top_labels
                and all(
                    label in values and values[label].get("value") is not None
                    for label in top_labels
                )
            }
        event_windows = build_event_windows(con, eligible_feature_fips_by_risk)
        playbook = build_county_playbook_data(con)

    # The feature comparison displays only the two window-A example counties
    # per risk group. Keep their details and discard thousands of unused county
    # records before embedding the standalone page.
    displayed_feature_fips: dict[str, set[str]] = {risk: set() for risk in RISK_ORDER}
    for example in event_windows.get("windowA", {}).get("exampleCountyLines", []):
        risk = example.get("riskRating")
        if risk in displayed_feature_fips and example.get("fips"):
            displayed_feature_fips[risk].add(str(example["fips"]).zfill(5))
    for risk, county_bins in features.get("withinGroupFeatureBins", {}).items():
        selected_fips = displayed_feature_fips.get(risk, set())
        features["withinGroupFeatureBins"][risk] = {
            fips: values for fips, values in county_bins.items() if str(fips).zfill(5) in selected_fips
        }

    geojson = build_geojson({county["fips"] for county in playbook["counties"]})
    county_history = {
        "months": price_risk.pop("countyHistoryMonths"),
        "series": price_risk.pop("countyHistorySeries"),
    }
    data = {
        "priceRisk": price_risk,
        "eventWindows": event_windows,
        "features": features,
        "playbook": None,
        "geojson": geojson,
    }
    OUT_PATH.write_text(make_html(data), encoding="utf-8")
    COUNTY_HISTORY_OUT_PATH.write_text(
        make_deferred_data_script("CLIMATE_RISK_HOUSING_COUNTY_HISTORY", county_history),
        encoding="utf-8",
    )
    PLAYBOOK_OUT_PATH.write_text(
        make_deferred_data_script("CLIMATE_RISK_HOUSING_PLAYBOOK", playbook),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {COUNTY_HISTORY_OUT_PATH}")
    print(f"Wrote {PLAYBOOK_OUT_PATH}")


if __name__ == "__main__":
    main()
