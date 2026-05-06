"""
Cluster counties by housing-market YOY responses for each incident type.

The notebook passes cleaned FEMA disaster rows and the prepared county housing
panel into ``build_all_housing_market_response_clusters``. The function chooses
the best model by silhouette score across Ward agglomerative and KMeans
candidates for each incident type, assigns labels to counties, writes annotation
artifacts, and returns the DataFrames needed by the visualization export step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PPSF_RESPONSE_MONTHS_BEFORE = 12
PPSF_RESPONSE_MONTHS_AFTER = 24
PPSF_PRE_OFFSETS = list(range(-PPSF_RESPONSE_MONTHS_BEFORE, 0))
PPSF_POST_1_12_OFFSETS = list(range(1, 13))
PPSF_POST_13_24_OFFSETS = list(range(13, PPSF_RESPONSE_MONTHS_AFTER + 1))
PPSF_RESPONSE_OFFSETS = PPSF_PRE_OFFSETS + PPSF_POST_1_12_OFFSETS + PPSF_POST_13_24_OFFSETS
PPSF_RESPONSE_K_VALUES = [2, 3, 4, 5]
PPSF_RESPONSE_MIN_CLUSTER_SHARE = 0.05
PPSF_RESPONSE_MIN_CLUSTER_SIZE = 3
PPSF_RESPONSE_BALANCE_PENALTY_WEIGHT = 0.35
PPSF_RESPONSE_METRICS = {
    "median_ppsf": "MEDIAN_PPSF_YOY_MOM",
    "avg_sale_to_list": "AVG_SALE_TO_LIST_YOY_MOM",
    "homes_sold": "HOMES_SOLD_YOY_MOM",
    "inventory": "INVENTORY_YOY_MOM",
}
PPSF_RESPONSE_METRIC_LABELS = {
    "median_ppsf": "Median PPSF",
    "avg_sale_to_list": "Avg Sale to List",
    "homes_sold": "Homes Sold",
    "inventory": "Inventory",
}
PPSF_RESPONSE_BASE_FEATURES = [
    "pre_period_mean",
    "post_months_1_12_mean",
    "post_months_13_24_mean",
    "post_minus_pre_change",
    "volatility",
    "max_drawdown",
    "recovery_slope",
]
PPSF_RESPONSE_FEATURES = [
    f"{metric}_{feature}" for metric in PPSF_RESPONSE_METRICS for feature in PPSF_RESPONSE_BASE_FEATURES
]
PPSF_RESPONSE_EXCLUDED_INCIDENT_TYPES = {
    "Biological",
    "Chemical",
    "Other",
    "Human Cause",
    "Terrorist",
    "Fishing Losses",
    "Dam/Levee Break",
    "Toxic Substances",
}


def _minimum_cluster_size(n_rows: int, k: int) -> int:
    share_floor = int(np.ceil(n_rows * PPSF_RESPONSE_MIN_CLUSTER_SHARE))
    half_expected_cluster_size = max(1, int(np.floor((n_rows / k) * 0.5)))
    return max(PPSF_RESPONSE_MIN_CLUSTER_SIZE, min(share_floor, half_expected_cluster_size))


def _cluster_balance_metrics(labels: np.ndarray, n_rows: int, k: int) -> dict[str, Any]:
    sizes = pd.Series(labels).value_counts().sort_index()
    expected_size = n_rows / k
    min_cluster_size = int(sizes.min())
    max_cluster_share = float(sizes.max() / n_rows)
    smallest_cluster_share = float(min_cluster_size / n_rows)
    size_cv = float(sizes.std(ddof=0) / expected_size) if expected_size else np.nan
    required_min_cluster_size = _minimum_cluster_size(n_rows, k)
    tiny_cluster_count = int((sizes < required_min_cluster_size).sum())
    has_tiny_cluster = bool(tiny_cluster_count > 0)
    balance_penalty = size_cv + max(0.0, max_cluster_share - 0.75)
    return {
        "cluster_sizes": sizes.to_dict(),
        "min_cluster_size": min_cluster_size,
        "tiny_cluster_count": tiny_cluster_count,
        "smallest_cluster_share": smallest_cluster_share,
        "largest_cluster_share": max_cluster_share,
        "cluster_size_cv": size_cv,
        "required_min_cluster_size": required_min_cluster_size,
        "has_tiny_cluster": has_tiny_cluster,
        "balance_penalty": balance_penalty,
    }


def _build_complete_housing_yoy_change_incident_response_rows(
    natural_disasters_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    incident_type: str,
) -> pd.DataFrame:
    incident_df = natural_disasters_df.loc[
        (natural_disasters_df["incidentType"] == incident_type)
        & (~natural_disasters_df["incidentType"].isin(PPSF_RESPONSE_EXCLUDED_INCIDENT_TYPES))
    ].copy()
    if incident_df.empty:
        return pd.DataFrame()

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
    incident_df["event_id"] = np.arange(len(incident_df))

    housing_with_keys = housing_df.copy()
    housing_with_keys["fips_normalized"] = housing_with_keys["fips"].astype(str).str.zfill(5)
    housing_with_keys["state_prefix"] = housing_with_keys["fips_normalized"].str[:2]
    housing_with_keys["is_county"] = ~housing_with_keys["fips_normalized"].str.endswith("000")
    county_housing = housing_with_keys.loc[housing_with_keys["is_county"]].copy()

    county_by_fips = {fips: group.copy() for fips, group in county_housing.groupby("fips_normalized", sort=False)}
    counties_by_state = {state: group.copy() for state, group in county_housing.groupby("state_prefix", sort=False)}

    response_rows = []
    for event in incident_df.itertuples(index=False):
        event_fips = str(event.fips).zfill(5)
        affected_housing = counties_by_state.get(event_fips[:2]) if event_fips.endswith("000") else county_by_fips.get(event_fips)
        if affected_housing is None or affected_housing.empty:
            continue

        offset_lookup = {event.incident_begin_month + offset: offset for offset in PPSF_PRE_OFFSETS}
        offset_lookup.update(
            {event.incident_end_month + offset: offset for offset in PPSF_POST_1_12_OFFSETS + PPSF_POST_13_24_OFFSETS}
        )

        event_rows = affected_housing.loc[affected_housing["MONTH"].isin(offset_lookup)].copy()
        if event_rows.empty:
            continue

        event_rows["month_offset_from_incident"] = event_rows["MONTH"].map(offset_lookup).astype("Int64")
        event_rows["incident_type"] = incident_type
        event_rows["incident_event_id"] = event.event_id
        event_rows["incident_disaster_number"] = event.disasterNumber
        event_rows["incident_begin_date"] = event.incidentBeginDate
        response_rows.append(event_rows)

    return pd.concat(response_rows, ignore_index=True) if response_rows else pd.DataFrame()


def _fit_recovery_slope(values_by_offset: pd.Series) -> float:
    post_offsets = PPSF_POST_1_12_OFFSETS + PPSF_POST_13_24_OFFSETS
    post_x = np.array(post_offsets, dtype=float)
    post_values = values_by_offset.loc[post_offsets].to_numpy(dtype=float)
    return np.polyfit(post_x, post_values, 1)[0]


def _summarize_metric_response(values_by_offset: pd.Series) -> dict[str, float]:
    pre_values = values_by_offset.loc[PPSF_PRE_OFFSETS]
    post_1_12_values = values_by_offset.loc[PPSF_POST_1_12_OFFSETS]
    post_13_24_values = values_by_offset.loc[PPSF_POST_13_24_OFFSETS]
    post_values = values_by_offset.loc[PPSF_POST_1_12_OFFSETS + PPSF_POST_13_24_OFFSETS]
    pre_mean = pre_values.mean()
    post_mean = post_values.mean()
    return {
        "pre_period_mean": pre_mean,
        "post_months_1_12_mean": post_1_12_values.mean(),
        "post_months_13_24_mean": post_13_24_values.mean(),
        "post_minus_pre_change": post_mean - pre_mean,
        "volatility": values_by_offset.std(ddof=0),
        "max_drawdown": post_values.min() - pre_mean,
        "recovery_slope": _fit_recovery_slope(values_by_offset),
    }


def _summarize_complete_incident_vectors(response_rows: pd.DataFrame) -> pd.DataFrame:
    required_metric_columns = list(PPSF_RESPONSE_METRICS.values())
    missing_metric_columns = [column for column in required_metric_columns if column not in response_rows.columns]
    if missing_metric_columns:
        raise KeyError(f"Missing required YOY monthly-change columns: {missing_metric_columns}")

    index_cols = ["fips", "county_name", "incident_event_id", "incident_begin_date"]
    feature_rows = []
    for keys, group in response_rows.groupby(index_cols, dropna=False):
        feature_row = dict(zip(index_cols, keys))
        complete = True
        for metric_key, metric_col in PPSF_RESPONSE_METRICS.items():
            values_by_offset = group.groupby("month_offset_from_incident")[metric_col].mean().reindex(PPSF_RESPONSE_OFFSETS)
            values_by_offset = pd.to_numeric(values_by_offset, errors="coerce")
            if values_by_offset.isna().any():
                complete = False
                break
            metric_summary = _summarize_metric_response(values_by_offset.astype(float))
            for feature, value in metric_summary.items():
                feature_row[f"{metric_key}_{feature}"] = value
        if complete:
            feature_rows.append(feature_row)

    return pd.DataFrame(feature_rows)


def _weighted_average_incident_features(event_features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    event_features = event_features.copy()
    event_features["incident_recency_rank"] = event_features["incident_begin_date"].rank(
        method="dense", ascending=True
    ).astype(float)

    averaged_rows = []
    for (fips, county_name), county_events in event_features.groupby(["fips", "county_name"], dropna=False):
        weights = county_events["incident_recency_rank"].to_numpy(dtype=float)
        averaged = {"fips": fips, "county_name": county_name}
        for column in feature_columns:
            values = pd.to_numeric(county_events[column], errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(values)
            averaged[column] = np.average(values[valid], weights=weights[valid]) if valid.any() else np.nan
        averaged["incident_count"] = county_events["incident_event_id"].nunique()
        averaged["latest_incident_begin_date"] = county_events["incident_begin_date"].max()
        averaged_rows.append(averaged)

    return pd.DataFrame(averaged_rows)


def summarize_ppsf_response_cluster(cluster_means: pd.Series) -> tuple[str, str]:
    def metric_change(metric: str) -> float:
        return cluster_means[f"{metric}_post_minus_pre_change"]

    def metric_volatility(metric: str) -> float:
        return cluster_means[f"{metric}_volatility"]

    price_change = metric_change("median_ppsf")
    sale_to_list_change = metric_change("avg_sale_to_list")
    sales_change = metric_change("homes_sold")
    inventory_change = metric_change("inventory")
    average_volatility = np.mean([metric_volatility(metric) for metric in PPSF_RESPONSE_METRICS])
    metric_changes = {PPSF_RESPONSE_METRIC_LABELS[metric]: metric_change(metric) for metric in PPSF_RESPONSE_METRICS}
    dominant_metric, dominant_change = max(metric_changes.items(), key=lambda item: abs(item[1]))

    if price_change > 0 and sale_to_list_change > 0 and inventory_change < 0:
        name = "Demand-tightening response"
        interpretation = "Price growth and sale-to-list pressure strengthen while inventory growth cools, suggesting tighter post-event market conditions."
    elif price_change < 0 and sales_change < 0 and inventory_change > 0:
        name = "Broad market softening"
        interpretation = "Price growth and sales momentum weaken while inventory growth rises, indicating a softer post-event market reaction."
    elif sales_change < 0 and inventory_change > 0:
        name = "Liquidity slowdown"
        interpretation = "Homes sold weaken while inventory growth rises, pointing to slower market clearing after the event."
    elif price_change > 0 and sales_change > 0:
        name = "Broad market acceleration"
        interpretation = "Price growth and transaction momentum both strengthen after the event."
    elif average_volatility > 0.08:
        name = "Volatile mixed response"
        interpretation = "The cluster is defined more by unstable month-to-month reactions than by one consistent direction across metrics."
    elif abs(dominant_change) < 0.01:
        name = "Stable market response"
        interpretation = "The selected housing market indicators remain close to their pre-event monthly-change pattern."
    else:
        direction = "strengthening" if dominant_change > 0 else "weakening"
        name = f"{dominant_metric} {direction}"
        interpretation = f"The clearest post-event shift is {direction} in {dominant_metric} monthly YOY changes."

    change_text = "; ".join(f"{metric}: {change:+.1%}" for metric, change in metric_changes.items())
    interpretation += f" Post-minus-pre monthly-change shifts are {change_text}."
    return name, interpretation


def build_housing_market_response_clusters(
    *,
    natural_disasters_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    incident_type: str,
    k_values: list[int] = PPSF_RESPONSE_K_VALUES,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    response_rows = _build_complete_housing_yoy_change_incident_response_rows(natural_disasters_df, housing_df, incident_type)
    if response_rows.empty:
        return None, None, {"incident_type": incident_type, "status": "no housing rows in complete incident window"}

    event_features = _summarize_complete_incident_vectors(response_rows)
    if event_features.empty:
        return None, None, {"incident_type": incident_type, "status": "no complete county-incident housing market response windows"}

    county_features = _weighted_average_incident_features(event_features, PPSF_RESPONSE_FEATURES)
    county_features = county_features.replace([np.inf, -np.inf], np.nan).dropna(subset=PPSF_RESPONSE_FEATURES).copy()
    if len(county_features) < min(k_values) + 1:
        return county_features, None, {
            "incident_type": incident_type,
            "status": "not enough counties with complete housing market response features",
            "counties_clustered": len(county_features),
            "complete_county_incident_vectors": len(event_features),
        }

    x_response = StandardScaler().fit_transform(county_features[PPSF_RESPONSE_FEATURES])
    comparison_rows = []

    def add_clustering_result(algorithm: str, k: int, labels: np.ndarray, cluster_col: str) -> None:
        county_features[cluster_col] = labels
        silhouette = silhouette_score(
            x_response, labels, sample_size=min(10_000, len(county_features)), random_state=42
        )
        balance_metrics = _cluster_balance_metrics(labels, len(county_features), k)
        comparison_rows.append(
            {
                "incident_type": incident_type,
                "algorithm": algorithm,
                "k": k,
                "status": "clustered",
                "counties_clustered": len(county_features),
                "complete_county_incident_vectors": len(event_features),
                "feature_count": len(PPSF_RESPONSE_FEATURES),
                "metrics_used": ", ".join(PPSF_RESPONSE_METRIC_LABELS.values()),
                "silhouette_score": silhouette,
                "balance_penalty": balance_metrics["balance_penalty"],
                "balanced_selection_score": silhouette
                - (PPSF_RESPONSE_BALANCE_PENALTY_WEIGHT * balance_metrics["balance_penalty"]),
                "min_cluster_size": balance_metrics["min_cluster_size"],
                "tiny_cluster_count": balance_metrics["tiny_cluster_count"],
                "smallest_cluster_share": balance_metrics["smallest_cluster_share"],
                "largest_cluster_share": balance_metrics["largest_cluster_share"],
                "cluster_size_cv": balance_metrics["cluster_size_cv"],
                "required_min_cluster_size": balance_metrics["required_min_cluster_size"],
                "has_tiny_cluster": balance_metrics["has_tiny_cluster"],
                "cluster_col": cluster_col,
                "cluster_sizes": balance_metrics["cluster_sizes"],
            }
        )

    for k in k_values:
        if k >= len(county_features):
            for algorithm in ["ward_agglomerative", "kmeans_benchmark"]:
                comparison_rows.append(
                    {
                        "incident_type": incident_type,
                        "algorithm": algorithm,
                        "k": k,
                        "status": "skipped",
                        "counties_clustered": len(county_features),
                        "complete_county_incident_vectors": len(event_features),
                        "feature_count": len(PPSF_RESPONSE_FEATURES),
                        "metrics_used": ", ".join(PPSF_RESPONSE_METRIC_LABELS.values()),
                        "silhouette_score": np.nan,
                        "balance_penalty": np.nan,
                        "balanced_selection_score": np.nan,
                        "min_cluster_size": np.nan,
                        "tiny_cluster_count": np.nan,
                        "smallest_cluster_share": np.nan,
                        "largest_cluster_share": np.nan,
                        "cluster_size_cv": np.nan,
                        "required_min_cluster_size": _minimum_cluster_size(len(county_features), k),
                        "has_tiny_cluster": pd.NA,
                        "cluster_col": pd.NA,
                        "cluster_sizes": {},
                    }
                )
            continue

        ward_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x_response)
        add_clustering_result("ward_agglomerative", k, ward_labels, f"median_ppsf_response_cluster_k{k}")

        kmeans_labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(x_response)
        add_clustering_result("kmeans_benchmark", k, kmeans_labels, f"median_ppsf_response_kmeans_cluster_k{k}")

    comparison_df = pd.DataFrame(comparison_rows)
    ward_results = comparison_df.loc[comparison_df["algorithm"] == "ward_agglomerative"].copy()
    kmeans_results = comparison_df.loc[comparison_df["algorithm"] == "kmeans_benchmark"].copy()
    valid_results = comparison_df.dropna(subset=["silhouette_score"]).copy()
    selectable_results = valid_results.loc[~valid_results["has_tiny_cluster"].astype(bool)].copy()
    selection_relaxed = False
    if selectable_results.empty:
        selectable_results = valid_results
        selection_relaxed = True
    best_ward_row = ward_results.loc[ward_results["silhouette_score"].idxmax()] if ward_results["silhouette_score"].notna().any() else None
    best_kmeans_row = kmeans_results.loc[kmeans_results["silhouette_score"].idxmax()] if kmeans_results["silhouette_score"].notna().any() else None
    best_row = (
        selectable_results.loc[selectable_results["balanced_selection_score"].idxmax()]
        if not selection_relaxed and not selectable_results.empty
        else None
    )
    if best_row is None and not selectable_results.empty:
        fallback_results = selectable_results.sort_values(
            ["tiny_cluster_count", "largest_cluster_share", "balanced_selection_score", "silhouette_score"],
            ascending=[True, True, False, False],
        )
        best_row = fallback_results.iloc[0]
    summary = {
        "incident_type": incident_type,
        "status": "clustered" if best_row is not None else "no valid k",
        "metrics_used": ", ".join(PPSF_RESPONSE_METRIC_LABELS.values()),
        "feature_count": len(PPSF_RESPONSE_FEATURES),
        "counties_clustered": len(county_features),
        "complete_county_incident_vectors": len(event_features),
        "best_algorithm_by_balanced_score": best_row["algorithm"] if best_row is not None else pd.NA,
        "best_k_by_balanced_score": int(best_row["k"]) if best_row is not None else pd.NA,
        "best_silhouette_score": float(best_row["silhouette_score"]) if best_row is not None else np.nan,
        "best_balanced_selection_score": float(best_row["balanced_selection_score"]) if best_row is not None else np.nan,
        "best_min_cluster_size": int(best_row["min_cluster_size"]) if best_row is not None else pd.NA,
        "best_tiny_cluster_count": int(best_row["tiny_cluster_count"]) if best_row is not None else pd.NA,
        "best_largest_cluster_share": float(best_row["largest_cluster_share"]) if best_row is not None else np.nan,
        "best_has_tiny_cluster": bool(best_row["has_tiny_cluster"]) if best_row is not None else pd.NA,
        "selection_relaxed_due_to_all_candidates_tiny": selection_relaxed,
        "best_ward_k_by_silhouette": int(best_ward_row["k"]) if best_ward_row is not None else pd.NA,
        "best_ward_silhouette_score": float(best_ward_row["silhouette_score"]) if best_ward_row is not None else np.nan,
        "best_kmeans_k_by_silhouette": int(best_kmeans_row["k"]) if best_kmeans_row is not None else pd.NA,
        "best_kmeans_silhouette_score": float(best_kmeans_row["silhouette_score"]) if best_kmeans_row is not None else np.nan,
    }
    return county_features, comparison_df, summary


def build_all_housing_market_response_clusters(
    *,
    natural_disasters_df: pd.DataFrame,
    housing_df: pd.DataFrame,
    output_dir: Path | str,
    k_values: list[int] = PPSF_RESPONSE_K_VALUES,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    incident_types = sorted(
        natural_disasters_df.loc[
            ~natural_disasters_df["incidentType"].isin(PPSF_RESPONSE_EXCLUDED_INCIDENT_TYPES), "incidentType"
        ]
        .dropna()
        .unique()
    )

    cluster_results = {}
    cluster_comparisons = []
    cluster_summaries = []
    for incident_type in incident_types:
        county_features, comparison_df, summary = build_housing_market_response_clusters(
            natural_disasters_df=natural_disasters_df,
            housing_df=housing_df,
            incident_type=incident_type,
            k_values=k_values,
        )
        cluster_results[incident_type] = county_features
        if comparison_df is not None:
            cluster_comparisons.append(comparison_df)
        cluster_summaries.append(summary)

    comparison_df = pd.concat(cluster_comparisons, ignore_index=True) if cluster_comparisons else pd.DataFrame()
    summary_df = pd.DataFrame(cluster_summaries)
    best_cluster_rows = (
        comparison_df.dropna(subset=["silhouette_score"])
        .assign(
            is_selectable=lambda df: ~df["has_tiny_cluster"].astype(bool),
            effective_selection_score=lambda df: np.where(
                df["is_selectable"], df["balanced_selection_score"], -np.inf
            ),
        )
        .sort_values(
            [
                "incident_type",
                "is_selectable",
                "effective_selection_score",
                "tiny_cluster_count",
                "largest_cluster_share",
                "balanced_selection_score",
                "silhouette_score",
            ],
            ascending=[True, False, False, True, True, False, False],
        )
        .groupby("incident_type", as_index=False, group_keys=False)
        .head(1)
        .copy()
        if not comparison_df.empty
        else pd.DataFrame()
    )
    if not best_cluster_rows.empty:
        missing_best_rows = set(comparison_df["incident_type"]) - set(best_cluster_rows["incident_type"])
        if missing_best_rows:
            relaxed_best_rows = (
                comparison_df.loc[comparison_df["incident_type"].isin(missing_best_rows)]
                .dropna(subset=["silhouette_score"])
                .sort_values(
                    ["incident_type", "balanced_selection_score", "silhouette_score"],
                    ascending=[True, False, False],
                )
                .groupby("incident_type", as_index=False, group_keys=False)
                .head(1)
            )
            best_cluster_rows = pd.concat([best_cluster_rows, relaxed_best_rows], ignore_index=True)

    annotation_frames = []
    interpretation_rows = []
    for best_row in best_cluster_rows.itertuples(index=False):
        incident_type = best_row.incident_type
        county_features = cluster_results.get(incident_type)
        if county_features is None or county_features.empty:
            continue
        cluster_col = best_row.cluster_col
        if cluster_col not in county_features.columns:
            continue

        annotated = county_features[["fips", "county_name", "incident_count", cluster_col] + PPSF_RESPONSE_FEATURES].copy()
        annotated = annotated.rename(
            columns={
                cluster_col: "median_ppsf_response_cluster",
                "incident_count": "median_ppsf_response_incident_count",
            }
        )
        annotated["incident_type"] = incident_type
        annotated["median_ppsf_response_cluster_algorithm"] = best_row.algorithm
        annotated["median_ppsf_response_cluster_k"] = int(best_row.k)
        annotated["median_ppsf_response_cluster_silhouette"] = float(best_row.silhouette_score)

        cluster_means = annotated.groupby("median_ppsf_response_cluster")[PPSF_RESPONSE_FEATURES].mean()
        cluster_sizes = annotated["median_ppsf_response_cluster"].value_counts().sort_index()
        interpretation_map = {}
        label_map = {}
        for cluster_id, means in cluster_means.iterrows():
            label, interpretation = summarize_ppsf_response_cluster(means)
            label_map[cluster_id] = label
            interpretation_map[cluster_id] = interpretation
            interpretation_rows.append(
                {
                    "incident_type": incident_type,
                    "cluster": int(cluster_id),
                    "cluster_name": label,
                    "interpretation": interpretation,
                    "counties": int(cluster_sizes.loc[cluster_id]),
                    "algorithm": best_row.algorithm,
                    "k": int(best_row.k),
                    "silhouette_score": float(best_row.silhouette_score),
                    "metrics_used": ", ".join(PPSF_RESPONSE_METRIC_LABELS.values()),
                }
            )

        annotated["median_ppsf_response_cluster_name"] = annotated["median_ppsf_response_cluster"].map(label_map)
        annotated["median_ppsf_response_cluster_interpretation"] = annotated["median_ppsf_response_cluster"].map(interpretation_map)
        annotation_frames.append(annotated)

    annotations_df = pd.concat(annotation_frames, ignore_index=True) if annotation_frames else pd.DataFrame()
    interpretations_df = pd.DataFrame(interpretation_rows)

    paths = {
        "assignments": output_dir / "ppsf_response_cluster_assignments.csv",
        "interpretations": output_dir / "ppsf_response_cluster_interpretations.csv",
        "comparison": output_dir / "ppsf_response_cluster_comparison.csv",
        "summary": output_dir / "ppsf_response_cluster_summary.csv",
    }
    annotations_df.to_csv(paths["assignments"], index=False)
    interpretations_df.to_csv(paths["interpretations"], index=False)
    comparison_df.to_csv(paths["comparison"], index=False)
    summary_df.to_csv(paths["summary"], index=False)

    return {
        "ppsf_response_cluster_results": cluster_results,
        "ppsf_response_cluster_comparison_df": comparison_df,
        "ppsf_response_cluster_summary_df": summary_df,
        "ppsf_response_best_cluster_rows": best_cluster_rows,
        "ppsf_response_cluster_annotations_df": annotations_df,
        "ppsf_response_cluster_interpretations_df": interpretations_df,
        "paths": paths,
    }
