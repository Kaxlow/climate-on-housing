"""
Group county-incident observations by pre-incident housing-market strength.

The page-data pipeline passes cleaned FEMA disaster rows and the prepared
monthly county housing panel into ``build_all_pre_incident_market_strength_tiers``. The module
builds complete county-incident windows, derives features only from the 12
months before incident start, computes a robust pre-incident market-strength
score, and assigns weak/middle/strong quantile tiers per incident type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump


MONTHS_BEFORE = 12
MONTHS_AFTER = 24
PRE_OFFSETS = list(range(-MONTHS_BEFORE, 0))
POST_1_12_OFFSETS = list(range(1, 13))
POST_13_24_OFFSETS = list(range(13, MONTHS_AFTER + 1))
REQUIRED_OFFSETS = PRE_OFFSETS + POST_1_12_OFFSETS + POST_13_24_OFFSETS

EXCLUDED_INCIDENT_TYPES = {
    "Biological",
    "Chemical",
    "Other",
    "Human Cause",
    "Terrorist",
    "Fishing Losses",
    "Dam/Levee Break",
    "Toxic Substances",
}

FEATURE_COLUMNS = [
    "pre_index_mean",
    "pre_index_slope",
    "pre_index_volatility",
    "pre_index_last_minus_first",
    "pre_median_ppsf_yoy_mean",
    "pre_avg_sale_to_list_yoy_mean",
    "pre_homes_sold_yoy_mean",
    "pre_inventory_yoy_mean",
]

STRENGTH_SCORE_WEIGHTS = {
    "pre_index_mean": 1.35,
    "pre_index_slope": 0.65,
    "pre_index_last_minus_first": 0.55,
    "pre_median_ppsf_yoy_mean": 0.9,
    "pre_avg_sale_to_list_yoy_mean": 0.85,
    "pre_homes_sold_yoy_mean": 0.65,
    "pre_inventory_yoy_mean": -0.75,
    "pre_index_volatility": -0.25,
}

FEATURE_LABELS = {
    "pre_index_mean": "pre-incident market index",
    "pre_index_slope": "pre-incident index trend",
    "pre_index_volatility": "pre-incident index volatility",
    "pre_index_last_minus_first": "late pre-window index change",
    "pre_median_ppsf_yoy_mean": "price growth",
    "pre_avg_sale_to_list_yoy_mean": "sale-to-list pressure",
    "pre_homes_sold_yoy_mean": "sales growth",
    "pre_inventory_yoy_mean": "inventory growth",
}

TIER_LABELS = {
    0: "Weak pre-incident market",
    1: "Middle pre-incident market",
    2: "Strong pre-incident market",
}

STORY_COLUMNS = [
    "incident_type",
    "incident_num",
    "incident_event_id",
    "incident_disaster_number",
    "incident_begin_date",
    "incident_end_date",
    "incident_year",
    "fips",
    "county_name",
    "REGION",
    "per_capita_income",
    "per_capita_income_bin",
    "pre_market_strength_tier",
    "pre_market_strength_tier_name",
    "pre_market_strength_tier_interpretation",
    "pre_market_strength_score",
    "pre_market_strength_percentile",
    "pre_market_strength_tier_method",
    "pre_index_mean",
    "pre_index_slope",
    "pre_index_volatility",
    "pre_index_last_minus_first",
    "pre_median_ppsf_yoy_mean",
    "pre_avg_sale_to_list_yoy_mean",
    "pre_homes_sold_yoy_mean",
    "pre_inventory_yoy_mean",
    "post_1_12_index_mean",
    "post_13_24_index_mean",
    "impact_pre_to_12",
    "impact_12_to_24",
]


def _incident_type_slug(incident_type: str) -> str:
    return (
        pd.Series([incident_type])
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
        .iloc[0]
        or "unknown"
    )


def _prepare_incidents(natural_disasters_df: pd.DataFrame, incident_type: str) -> pd.DataFrame:
    incident_df = natural_disasters_df.loc[
        (natural_disasters_df["incidentType"] == incident_type)
        & (~natural_disasters_df["incidentType"].isin(EXCLUDED_INCIDENT_TYPES))
    ].copy()
    if incident_df.empty:
        return incident_df

    incident_duration = incident_df["incidentEndDate"] - incident_df["incidentBeginDate"]
    median_duration = incident_duration[incident_duration.notna()].median()
    if pd.isna(median_duration):
        median_duration = pd.Timedelta(days=0)
    incident_df["incidentEndDate"] = incident_df["incidentEndDate"].fillna(
        incident_df["incidentBeginDate"] + median_duration
    )
    incident_df = incident_df.dropna(subset=["incidentBeginDate", "incidentEndDate", "fips"]).copy()
    incident_df["incident_begin_month"] = incident_df["incidentBeginDate"].dt.to_period("M")
    incident_df["incident_end_month"] = incident_df["incidentEndDate"].dt.to_period("M")
    incident_df["incident_event_id"] = np.arange(len(incident_df))
    return incident_df


def build_complete_county_incident_market_windows(
    natural_disasters_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    incident_type: str,
) -> pd.DataFrame:
    incident_df = _prepare_incidents(natural_disasters_df, incident_type)
    if incident_df.empty:
        return pd.DataFrame()

    housing_with_keys = housing_df.copy()
    housing_with_keys["fips_normalized"] = housing_with_keys["fips"].astype(str).str.zfill(5)
    housing_with_keys["state_prefix"] = housing_with_keys["fips_normalized"].str[:2]
    county_housing = housing_with_keys.loc[~housing_with_keys["fips_normalized"].str.endswith("000")].copy()

    event_month_rows = []
    for event in incident_df.itertuples(index=False):
        event_fips = str(event.fips).zfill(5)
        offset_lookup = {event.incident_begin_month + offset: offset for offset in PRE_OFFSETS}
        offset_lookup.update({event.incident_end_month + offset: offset for offset in POST_1_12_OFFSETS})
        offset_lookup.update({event.incident_end_month + offset: offset for offset in POST_13_24_OFFSETS})
        for month, offset in offset_lookup.items():
            event_month_rows.append(
                {
                    "event_fips": event_fips,
                    "event_state_prefix": event_fips[:2],
                    "is_statewide_event": event_fips.endswith("000"),
                    "MONTH": month,
                    "month_offset_from_incident": offset,
                    "incident_type": incident_type,
                    "incident_event_id": event.incident_event_id,
                    "incident_disaster_number": event.disasterNumber,
                    "incident_begin_date": event.incidentBeginDate,
                    "incident_end_date": event.incidentEndDate,
                    "incident_year": event.incidentBeginDate.year,
                }
            )

    event_months = pd.DataFrame(event_month_rows)
    if event_months.empty:
        return pd.DataFrame()

    matched_frames = []
    county_events = event_months.loc[~event_months["is_statewide_event"]]
    state_events = event_months.loc[event_months["is_statewide_event"]]
    if not county_events.empty:
        matched_frames.append(
            county_housing.merge(
                county_events,
                left_on=["fips_normalized", "MONTH"],
                right_on=["event_fips", "MONTH"],
                how="inner",
            )
        )
    if not state_events.empty:
        matched_frames.append(
            county_housing.merge(
                state_events,
                left_on=["state_prefix", "MONTH"],
                right_on=["event_state_prefix", "MONTH"],
                how="inner",
            )
        )
    if not matched_frames:
        return pd.DataFrame()

    matched = pd.concat(matched_frames, ignore_index=True)
    matched = matched.drop(
        columns=["fips_normalized", "state_prefix", "event_fips", "event_state_prefix", "is_statewide_event"],
        errors="ignore",
    )
    matched["fips"] = matched["fips"].astype(str).str.zfill(5)
    matched["HOUSING_MARKET_INDEX"] = pd.to_numeric(matched["HOUSING_MARKET_INDEX"], errors="coerce")

    complete_counts = (
        matched.dropna(subset=["HOUSING_MARKET_INDEX"])
        .groupby(["fips", "incident_event_id"], dropna=False)["month_offset_from_incident"]
        .agg(lambda offsets: set(pd.to_numeric(offsets, errors="coerce").dropna().astype(int)))
    )
    complete_keys = complete_counts.loc[complete_counts.map(lambda offsets: set(REQUIRED_OFFSETS).issubset(offsets))].index
    if complete_keys.empty:
        return pd.DataFrame()
    key_frame = complete_keys.to_frame(index=False)
    return matched.merge(key_frame, on=["fips", "incident_event_id"], how="inner")


def _linear_slope(values: pd.Series, offsets: list[int]) -> float:
    y = pd.to_numeric(values.reindex(offsets), errors="coerce").to_numpy(dtype=float)
    x = np.array(offsets, dtype=float)
    if np.isnan(y).any():
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def build_county_incident_feature_frame(window_rows: pd.DataFrame) -> pd.DataFrame:
    if window_rows.empty:
        return pd.DataFrame()

    id_cols = [
        "incident_type",
        "incident_event_id",
        "incident_disaster_number",
        "incident_begin_date",
        "incident_end_date",
        "incident_year",
        "fips",
        "county_name",
        "REGION",
    ]
    optional_id_cols = [col for col in ["per_capita_income", "per_capita_income_bin"] if col in window_rows.columns]
    metric_cols = [
        "HOUSING_MARKET_INDEX",
        "MEDIAN_PPSF_YOY",
        "AVG_SALE_TO_LIST_YOY",
        "HOMES_SOLD_YOY",
        "INVENTORY_YOY",
    ]
    work = window_rows[id_cols + optional_id_cols + ["month_offset_from_incident"] + metric_cols].copy()
    work["month_offset_from_incident"] = pd.to_numeric(work["month_offset_from_incident"], errors="coerce")
    work = work.dropna(subset=["month_offset_from_incident"])
    work["month_offset_from_incident"] = work["month_offset_from_incident"].astype(int)
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    group_cols = [
        "incident_type",
        "incident_event_id",
        "incident_disaster_number",
        "incident_begin_date",
        "incident_end_date",
        "incident_year",
        "fips",
    ]
    monthly = (
        work.groupby(group_cols + ["month_offset_from_incident"], dropna=False, sort=False)[metric_cols]
        .mean()
        .reset_index()
    )
    base = (
        work.sort_values(group_cols)
        .drop_duplicates(group_cols)[group_cols + ["county_name", "REGION"] + optional_id_cols]
        .set_index(group_cols)
    )
    feature_df = base.copy()
    hmi_wide = (
        monthly.set_index(group_cols + ["month_offset_from_incident"])["HOUSING_MARKET_INDEX"]
        .unstack("month_offset_from_incident")
        .reindex(index=base.index, columns=REQUIRED_OFFSETS)
    )
    complete_mask = hmi_wide.notna().all(axis=1)

    feature_df["pre_index_mean"] = hmi_wide[PRE_OFFSETS].mean(axis=1)
    feature_df["pre_index_slope"] = hmi_wide[PRE_OFFSETS].apply(lambda row: _linear_slope(row, PRE_OFFSETS), axis=1)
    feature_df["pre_index_volatility"] = hmi_wide[PRE_OFFSETS].std(axis=1, ddof=0)
    feature_df["pre_index_last_minus_first"] = hmi_wide[-1] - hmi_wide[-12]
    feature_df["post_1_12_index_mean"] = hmi_wide[POST_1_12_OFFSETS].mean(axis=1)
    feature_df["post_13_24_index_mean"] = hmi_wide[POST_13_24_OFFSETS].mean(axis=1)
    feature_df["impact_pre_to_12"] = feature_df["post_1_12_index_mean"] - feature_df["pre_index_mean"]
    feature_df["impact_12_to_24"] = feature_df["post_13_24_index_mean"] - feature_df["post_1_12_index_mean"]

    pre_metric_map = {
        "MEDIAN_PPSF_YOY": "pre_median_ppsf_yoy_mean",
        "AVG_SALE_TO_LIST_YOY": "pre_avg_sale_to_list_yoy_mean",
        "HOMES_SOLD_YOY": "pre_homes_sold_yoy_mean",
        "INVENTORY_YOY": "pre_inventory_yoy_mean",
    }
    for metric_col, feature_col in pre_metric_map.items():
        wide = (
            monthly.set_index(group_cols + ["month_offset_from_incident"])[metric_col]
            .unstack("month_offset_from_incident")
            .reindex(index=base.index, columns=PRE_OFFSETS)
        )
        feature_df[feature_col] = wide.mean(axis=1)

    feature_df = feature_df.loc[complete_mask].reset_index()
    if "per_capita_income" in feature_df.columns:
        feature_df["per_capita_income"] = pd.to_numeric(feature_df["per_capita_income"], errors="coerce")
    if "per_capita_income_bin" in feature_df.columns:
        feature_df["per_capita_income_bin"] = pd.to_numeric(
            feature_df["per_capita_income_bin"], errors="coerce"
        ).astype("Int64")
    elif "per_capita_income" in feature_df.columns:
        income_values = pd.to_numeric(feature_df["per_capita_income"], errors="coerce")
        feature_df["per_capita_income_bin"] = pd.Series(pd.NA, index=feature_df.index, dtype="Int64")
        valid_income = income_values.notna()
        if valid_income.sum() >= 3:
            try:
                income_bins = pd.qcut(income_values.loc[valid_income], q=3, labels=False, duplicates="drop")
                feature_df.loc[valid_income, "per_capita_income_bin"] = (
                    pd.Series(income_bins, index=income_values.loc[valid_income].index).astype("Int64") + 1
                )
            except ValueError:
                pass
    feature_df["incident_num"] = feature_df["incident_event_id"] + 1
    return feature_df


def _robust_feature_score(feature_df: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    score_parts = []
    feature_metadata = {}
    for column, weight in STRENGTH_SCORE_WEIGHTS.items():
        values = pd.to_numeric(feature_df[column], errors="coerce")
        median = values.median(skipna=True)
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        scale = q3 - q1
        scale_source = "iqr"
        if pd.isna(scale) or scale == 0:
            scale = values.std(skipna=True)
            scale_source = "std"
        if pd.isna(scale) or scale == 0:
            normalized = pd.Series(0.0, index=feature_df.index)
            scale = 1.0
            scale_source = "constant"
        else:
            normalized = (values.fillna(median) - median) / scale
        score_parts.append(weight * normalized.clip(-4, 4))
        feature_metadata[column] = {
            "label": FEATURE_LABELS[column],
            "weight": weight,
            "median": None if pd.isna(median) else float(median),
            "scale": float(scale),
            "scale_source": scale_source,
        }
    return sum(score_parts), feature_metadata


def _assign_strength_tiers(score: pd.Series) -> pd.Series:
    if len(score) < 3:
        return pd.Series(pd.NA, index=score.index, dtype="Int64")
    ranked = score.rank(method="first")
    return pd.qcut(ranked, q=3, labels=False, duplicates="drop").astype("Int64")


def _tier_interpretation(tier: int, group: pd.DataFrame) -> str:
    pre_index = pd.to_numeric(group["pre_index_mean"], errors="coerce").mean()
    score = pd.to_numeric(group["pre_market_strength_score"], errors="coerce").mean()
    pre_to_12 = pd.to_numeric(group["impact_pre_to_12"], errors="coerce").median()
    count = len(group)
    return (
        f"{TIER_LABELS[tier]} contains {count:,} complete county-incident observations. "
        f"The tier's average pre-incident market score is {score:+.2f}, with average pre-incident index {pre_index:+.2f}. "
        f"Its median index shift from the pre-window to months 1-12 after the incident is {pre_to_12:+.2f}."
    )


def build_pre_incident_market_strength_tiers(
    *,
    natural_disasters_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    incident_type: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    window_rows = build_complete_county_incident_market_windows(natural_disasters_df, housing_df, incident_type)
    if window_rows.empty:
        return None, None, {"incident_type": incident_type, "status": "no complete county-incident windows"}

    feature_df = build_county_incident_feature_frame(window_rows)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS, how="all").copy()
    if len(feature_df) < 3:
        return feature_df, None, {
            "incident_type": incident_type,
            "status": "not enough complete county-incident observations",
            "observations_grouped": len(feature_df),
        }

    score, score_metadata = _robust_feature_score(feature_df)
    annotations = feature_df[
        [
            "incident_type",
            "incident_num",
            "incident_event_id",
            "incident_disaster_number",
            "incident_begin_date",
            "incident_end_date",
            "incident_year",
            "fips",
            "county_name",
            "REGION",
            *[col for col in ["per_capita_income", "per_capita_income_bin"] if col in feature_df.columns],
            *FEATURE_COLUMNS,
            "post_1_12_index_mean",
            "post_13_24_index_mean",
            "impact_pre_to_12",
            "impact_12_to_24",
        ]
    ].copy()
    annotations["pre_market_strength_score"] = score
    annotations["pre_market_strength_percentile"] = score.rank(pct=True, method="average")
    annotations["pre_market_strength_tier"] = _assign_strength_tiers(score)
    annotations = annotations.dropna(subset=["pre_market_strength_tier"]).copy()
    annotations["pre_market_strength_tier"] = annotations["pre_market_strength_tier"].astype(int)
    annotations["pre_market_strength_tier_name"] = annotations["pre_market_strength_tier"].map(TIER_LABELS)
    annotations["pre_market_strength_tier_method"] = "within_incident_type_score_tercile"

    interpretation_rows = []
    interpretation_map = {}
    for tier, group in annotations.groupby("pre_market_strength_tier", sort=True):
        interpretation = _tier_interpretation(int(tier), group)
        interpretation_map[int(tier)] = interpretation
        interpretation_rows.append(
            {
                "incident_type": incident_type,
                "tier": int(tier),
                "tier_name": TIER_LABELS[int(tier)],
                "interpretation": interpretation,
                "observations": len(group),
                "method": "within_incident_type_score_tercile",
            }
        )
    annotations["pre_market_strength_tier_interpretation"] = annotations["pre_market_strength_tier"].map(interpretation_map)
    for column in STORY_COLUMNS:
        if column not in annotations.columns:
            annotations[column] = pd.NA
    annotations = annotations[STORY_COLUMNS]
    interpretations_df = pd.DataFrame(interpretation_rows)
    summary = {
        "incident_type": incident_type,
        "status": "grouped",
        "observations_grouped": len(annotations),
        "method": "within_incident_type_score_tercile",
        "score_metadata": score_metadata,
    }
    return annotations, interpretations_df, summary


def build_all_pre_incident_market_strength_tiers(
    *,
    natural_disasters_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    incident_types = sorted(
        natural_disasters_df.loc[
            ~natural_disasters_df["incidentType"].isin(EXCLUDED_INCIDENT_TYPES), "incidentType"
        ]
        .dropna()
        .unique()
    )

    annotation_frames = []
    interpretation_frames = []
    summary_rows = []
    score_metadata = {}
    story_files = {}

    for incident_type in incident_types:
        annotations, interpretations_df, summary = build_pre_incident_market_strength_tiers(
            natural_disasters_df=natural_disasters_df,
            housing_df=housing_df,
            incident_type=incident_type,
        )
        score_metadata[incident_type] = summary.pop("score_metadata", {})
        summary_rows.append(summary)
        slug = _incident_type_slug(incident_type)
        story_path = output_dir / f"{slug}_story_4_pre_market_tiers.csv"
        if annotations is not None and not annotations.empty:
            annotations.to_csv(story_path, index=False)
            story_files[incident_type] = story_path
            annotation_frames.append(annotations)
        else:
            pd.DataFrame(columns=STORY_COLUMNS).to_csv(story_path, index=False)
            story_files[incident_type] = story_path
        if interpretations_df is not None and not interpretations_df.empty:
            interpretation_frames.append(interpretations_df)

    assignments_df = pd.concat(annotation_frames, ignore_index=True) if annotation_frames else pd.DataFrame(columns=STORY_COLUMNS)
    interpretations_df = pd.concat(interpretation_frames, ignore_index=True) if interpretation_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    paths = {
        "assignments": output_dir / "pre_market_strength_tier_assignments.csv",
        "interpretations": output_dir / "pre_market_strength_tier_interpretations.csv",
        "summary": output_dir / "pre_market_strength_tier_summary.csv",
        "summaries_json": output_dir / "pre_market_strength_tier_summaries.json",
        "score_metadata": output_dir / "pre_market_strength_tier_score_metadata.joblib",
    }
    assignments_df.to_csv(paths["assignments"], index=False)
    interpretations_df.to_csv(paths["interpretations"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    dump(
        {
            "score_weights": STRENGTH_SCORE_WEIGHTS,
            "feature_columns": FEATURE_COLUMNS,
            "feature_labels": FEATURE_LABELS,
            "incident_type_metadata": score_metadata,
        },
        paths["score_metadata"],
    )

    summaries_json = {}
    for incident_type, group in interpretations_df.groupby("incident_type") if not interpretations_df.empty else []:
        summary_row = summary_df.loc[summary_df["incident_type"] == incident_type].iloc[0]
        summaries_json[incident_type] = {
            "incident_type": incident_type,
            "status": summary_row["status"],
            "method": summary_row.get("method", "within_incident_type_score_tercile"),
            "observations_grouped": int(summary_row.get("observations_grouped", 0)),
            "tiers": [
                {
                    "tier": int(row.tier),
                    "name": row.tier_name,
                    "interpretation": row.interpretation,
                    "observations": int(row.observations),
                }
                for row in group.sort_values("tier").itertuples(index=False)
            ],
        }
    paths["summaries_json"].write_text(json.dumps(summaries_json, indent=2), encoding="utf-8")

    return {
        "pre_market_strength_assignments_df": assignments_df,
        "pre_market_strength_interpretations_df": interpretations_df,
        "pre_market_strength_summary_df": summary_df,
        "story_files": story_files,
        "paths": paths,
    }
