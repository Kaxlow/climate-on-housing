"""County migration trend profiles around FEMA incident years.

This module builds feature-based time-series clusters from annual county net
migration per 1,000 residents in the two years before through two years after
FEMA incident occurrence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer


OFFSETS = [-2, -1, 0, 1, 2]
RATE_COLUMNS = {offset: f"mig_rate_{offset:+d}" for offset in OFFSETS}
FEATURE_COLUMNS = [
    "mig_pre_avg",
    "mig_incident_change",
    "mig_first_year_change",
    "mig_second_year_change",
    "mig_post_avg",
    "mig_overall_change",
    "mig_volatility",
    "mig_linear_slope",
]


def build_migration_trend_profile_outputs(
    *,
    county_windows: pd.DataFrame,
    profile_inputs: dict[str, pd.DataFrame],
    output_dir: Path | str,
    risk_order: list[str],
    candidate_ks: range | list[int] = range(4, 7),
    random_state: int = 42,
) -> dict[str, Any]:
    """Cluster counties by migration trend shape and summarize by NRI rating."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = build_migration_feature_matrix(county_windows, profile_inputs)
    x_df = winsorize_frame(feature_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all"))
    if x_df.empty:
        raise ValueError("No usable migration trend features were available for clustering.")

    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=min(1000, len(x_df)),
                    output_distribution="normal",
                    random_state=random_state,
                    subsample=None,
                ),
            ),
        ]
    )
    x_scaled = preprocess.fit_transform(x_df)
    pca = PCA(n_components=min(6, x_scaled.shape[1]), random_state=random_state)
    x_model = pca.fit_transform(x_scaled)

    scores = []
    artifacts = {}
    for k in candidate_ks:
        model = GaussianMixture(
            n_components=int(k),
            covariance_type="diag",
            reg_covar=1e-3,
            n_init=20,
            random_state=random_state,
        )
        labels = model.fit_predict(x_model)
        probabilities = model.predict_proba(x_model)
        row = score_candidate(x_model, labels, probabilities)
        row["k"] = int(k)
        scores.append(row)
        artifacts[int(k)] = {"model": model, "labels": labels, "probabilities": probabilities}

    score_df = rank_candidates(pd.DataFrame(scores))
    best_k = int(score_df.iloc[0]["k"])
    best = artifacts[best_k]
    profile_labels = label_profiles(feature_df, best["labels"], best["probabilities"])
    interpretations = build_profile_interpretations(feature_df, best["labels"], profile_labels)
    assignments = build_assignment_table(feature_df, best["labels"], best["probabilities"], profile_labels)
    risk_profile_summary = build_risk_profile_summary(assignments, risk_order)
    profile_summary = build_profile_summary(feature_df, assignments, interpretations)
    series_summary = build_profile_series(feature_df, assignments)

    assignments.to_csv(output_dir / "stormhouse_migration_trend_assignments.csv", index=False)
    risk_profile_summary.to_csv(output_dir / "stormhouse_migration_trend_risk_summary.csv", index=False)
    profile_summary.to_csv(output_dir / "stormhouse_migration_trend_labels.csv", index=False)
    series_summary.to_csv(output_dir / "stormhouse_migration_trend_series.csv", index=False)
    score_df.to_csv(output_dir / "stormhouse_migration_trend_scores.csv", index=False)
    dump(
        {
            "preprocess": preprocess,
            "pca": pca,
            "model": best["model"],
            "feature_columns": x_df.columns.tolist(),
            "best_k": best_k,
            "profile_labels": profile_labels,
            "interpretations": interpretations,
            "scores": score_df.iloc[0].to_dict(),
        },
        output_dir / "stormhouse_migration_trend_model.joblib",
    )

    return {
        "assignments": assignments,
        "risk_profile_summary": risk_profile_summary,
        "profile_summary": profile_summary,
        "series_summary": series_summary,
        "scores": score_df,
        "best_k": best_k,
    }


def build_migration_feature_matrix(county_windows: pd.DataFrame, profile_inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build county-level migration rate features around incident years."""

    change = profile_inputs["population_change_df"][
        ["fips", "Year", "Net International Migration", "Net Domestic Migration"]
    ].copy()
    population = profile_inputs["population_estimates_df"][["fips", "Year", "Population"]].copy()
    change["fips"] = change["fips"].astype(str).str.zfill(5)
    change["Year"] = pd.to_numeric(change["Year"], errors="coerce").astype("Int64")
    population["fips"] = population["fips"].astype(str).str.zfill(5)
    population["Year"] = pd.to_numeric(population["Year"], errors="coerce").astype("Int64")
    population["population"] = pd.to_numeric(population["Population"], errors="coerce")
    change["net_migration"] = (
        pd.to_numeric(change["Net International Migration"], errors="coerce")
        + pd.to_numeric(change["Net Domestic Migration"], errors="coerce")
    )
    change = change.merge(population[["fips", "Year", "population"]], on=["fips", "Year"], how="left")
    change["net_migration_per_1000"] = np.where(
        change["population"] > 0,
        change["net_migration"] / change["population"] * 1000,
        np.nan,
    )

    events = (
        county_windows[["fips", "county_name", "STATE_CODE", "incident_event_id", "incident_begin_dt", "nri_risk_rating"]]
        .drop_duplicates(["fips", "incident_event_id"])
        .dropna(subset=["incident_begin_dt"])
        .copy()
    )
    events["incident_year"] = events["incident_begin_dt"].dt.year.astype("Int64")
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        for year_offset in OFFSETS:
            rows.append(
                {
                    "fips": str(event.fips).zfill(5),
                    "county_name": event.county_name,
                    "state": event.STATE_CODE,
                    "incident_event_id": event.incident_event_id,
                    "nri_risk_rating": event.nri_risk_rating,
                    "year_offset": year_offset,
                    "Year": int(event.incident_year) + year_offset,
                }
            )
    work = pd.DataFrame(rows).merge(change[["fips", "Year", "net_migration_per_1000"]], on=["fips", "Year"], how="left")
    work = work.dropna(subset=["net_migration_per_1000"]).sort_values(["fips", "year_offset", "incident_event_id"]).copy()
    lower = float(work["net_migration_per_1000"].quantile(0.01))
    upper = float(work["net_migration_per_1000"].quantile(0.99))
    work["plot_rate"] = work["net_migration_per_1000"].clip(lower=lower, upper=upper)
    work["recency_weight"] = work.groupby(["fips", "year_offset"]).cumcount() + 1
    work["weighted_rate"] = work["plot_rate"] * work["recency_weight"]
    weighted = (
        work.groupby(["fips", "year_offset"], as_index=False)
        .agg(
            net_migration_per_1000=("weighted_rate", "sum"),
            total_weight=("recency_weight", "sum"),
            incident_count=("incident_event_id", "nunique"),
            county_name=("county_name", "last"),
            state=("state", "last"),
            nri_risk_rating=("nri_risk_rating", "last"),
        )
        .assign(net_migration_per_1000=lambda df: df["net_migration_per_1000"] / df["total_weight"])
    )
    pivot = weighted.pivot(index="fips", columns="year_offset", values="net_migration_per_1000").rename(columns=RATE_COLUMNS)
    complete = pivot.dropna(subset=list(RATE_COLUMNS.values())).reset_index()
    meta = (
        weighted.sort_values(["fips", "year_offset"])
        .groupby("fips", as_index=False)
        .agg(
            county_name=("county_name", "last"),
            state=("state", "last"),
            nri_risk_rating=("nri_risk_rating", "last"),
            incident_count=("incident_count", "max"),
        )
    )
    out = complete.merge(meta, on="fips", how="left")
    rates = out[list(RATE_COLUMNS.values())].to_numpy(dtype=float)
    out["mig_pre_avg"] = out[[RATE_COLUMNS[-2], RATE_COLUMNS[-1]]].mean(axis=1)
    out["mig_incident_change"] = out[RATE_COLUMNS[0]] - out["mig_pre_avg"]
    out["mig_first_year_change"] = out[RATE_COLUMNS[1]] - out["mig_pre_avg"]
    out["mig_second_year_change"] = out[RATE_COLUMNS[2]] - out[RATE_COLUMNS[1]]
    out["mig_post_avg"] = out[[RATE_COLUMNS[1], RATE_COLUMNS[2]]].mean(axis=1)
    out["mig_overall_change"] = out["mig_post_avg"] - out["mig_pre_avg"]
    out["mig_volatility"] = np.nanstd(rates, axis=1)
    out["mig_linear_slope"] = np.polyfit(np.array(OFFSETS, dtype=float), rates.T, deg=1)[0]
    return out


def winsorize_frame(x_df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    clipped = x_df.copy()
    for column in clipped.columns:
        values = clipped[column]
        if values.notna().sum() < 20:
            continue
        lo = values.quantile(lower)
        hi = values.quantile(upper)
        if pd.notna(lo) and pd.notna(hi) and lo < hi:
            clipped[column] = values.clip(lo, hi)
    return clipped


def score_candidate(x_model: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    sizes = pd.Series(labels).value_counts().sort_index()
    sorted_probs = np.sort(probabilities, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return {
        "n_clusters": int(len(sizes)),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_share": float(sizes.max() / len(labels)),
        "cluster_size_cv": float(sizes.std(ddof=0) / sizes.mean()),
        "mean_assignment_confidence": float(confidence.mean()),
        "median_assignment_confidence": float(np.median(confidence)),
        "low_confidence_under_0_60_rate": float((confidence < 0.60).mean()),
        "mean_assignment_margin": float(margin.mean()),
        "median_assignment_margin": float(np.median(margin)),
        "silhouette_score": float(silhouette_score(x_model, labels, sample_size=min(10_000, len(labels)), random_state=42)),
        "calinski_harabasz_score": float(calinski_harabasz_score(x_model, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(x_model, labels)),
        "cluster_sizes": json.dumps({int(key): int(value) for key, value in sizes.items()}, sort_keys=True),
    }


def rank_candidates(score_df: pd.DataFrame) -> pd.DataFrame:
    score_df = score_df.copy()
    score_df["passes_balance_check"] = (
        (score_df["min_cluster_size"] >= 80)
        & (score_df["max_cluster_share"] <= 0.50)
        & (score_df["low_confidence_under_0_60_rate"] <= 0.40)
    )
    score_df["silhouette_rank"] = score_df["silhouette_score"].rank(ascending=False, method="min")
    score_df["davies_bouldin_rank"] = score_df["davies_bouldin_index"].rank(ascending=True, method="min")
    score_df["confidence_rank"] = score_df["mean_assignment_confidence"].rank(ascending=False, method="min")
    score_df["balance_penalty"] = score_df["max_cluster_share"] * 8 + score_df["cluster_size_cv"]
    score_df["usable_rank_score"] = (
        score_df[["silhouette_rank", "davies_bouldin_rank", "confidence_rank"]].mean(axis=1)
        + score_df["balance_penalty"]
    )
    return score_df.sort_values(["passes_balance_check", "usable_rank_score"], ascending=[False, True]).reset_index(drop=True)


def label_profiles(feature_df: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> dict[int, str]:
    seed = {cluster: str(cluster) for cluster in sorted(set(labels))}
    assignments = build_assignment_table(feature_df, labels, probabilities, seed)
    merged = assignments.merge(feature_df[["fips", *FEATURE_COLUMNS]], on="fips", how="left")
    out: dict[int, str] = {}
    used: set[str] = set()
    for profile, group in merged.groupby("migration_trend_profile"):
        stats = group[FEATURE_COLUMNS].median()
        label = derive_migration_label(stats)
        if label in used:
            label = f"{label} ({int(profile)})"
        used.add(label)
        out[int(profile)] = label
    return out


def derive_migration_label(stats: pd.Series) -> str:
    pre = float(stats.get("mig_pre_avg", 0.0))
    post = float(stats.get("mig_post_avg", 0.0))
    overall = float(stats.get("mig_overall_change", 0.0))
    first = float(stats.get("mig_first_year_change", 0.0))
    second = float(stats.get("mig_second_year_change", 0.0))
    volatility = float(stats.get("mig_volatility", 0.0))
    if first <= -1 and second >= 1:
        return "Dip-and-recover migration counties"
    if first >= 1 and second <= -1:
        return "Spike-and-fade migration counties"
    if volatility >= 6:
        return "Volatile migration counties"
    if pre >= 1 and post >= 1 and abs(overall) < 1:
        return "Stable in-migration counties"
    if pre <= -1 and post <= -1 and abs(overall) < 1:
        return "Stable out-migration counties"
    if overall >= 3 or first >= 3:
        return "Strong post-incident migration gain counties"
    if overall >= 1:
        return "Gradual post-incident migration gain counties"
    if overall <= -1:
        return "Post-incident migration loss counties"
    return "Mostly stable migration counties"


def build_profile_interpretations(feature_df: pd.DataFrame, labels: np.ndarray, profile_labels: dict[int, str]) -> dict[int, str]:
    temp = feature_df[["fips", *FEATURE_COLUMNS]].copy()
    temp["migration_trend_profile"] = labels.astype(int)
    out: dict[int, str] = {}
    for profile, group in temp.groupby("migration_trend_profile"):
        stats = group[FEATURE_COLUMNS].median()
        out[int(profile)] = describe_profile(profile_labels[int(profile)], stats)
    return out


def describe_profile(label: str, stats: pd.Series) -> str:
    pre = float(stats.get("mig_pre_avg", 0.0))
    post = float(stats.get("mig_post_avg", 0.0))
    incident = float(stats.get("mig_incident_change", 0.0))
    first = float(stats.get("mig_first_year_change", 0.0))
    second = float(stats.get("mig_second_year_change", 0.0))
    level = "net in-migration" if pre > 1 else "net out-migration" if pre < -1 else "roughly balanced migration"
    overall = "higher" if post - pre > 1 else "lower" if post - pre < -1 else "about the same"
    if "Dip-and-recover" in label:
        return "Migration weakens around the first post-incident year, then recovers by the second year."
    if "Spike-and-fade" in label:
        return "Migration rises shortly after the incident, then gives back part of that gain by year two."
    if "Volatile" in label:
        return "Migration swings materially across the five-year window rather than following one steady direction."
    return (
        f"Counties start with {level}; by the two-year post-incident window, migration is {overall}. "
        f"The incident-year movement is {'upward' if incident > 1 else 'downward' if incident < -1 else 'muted'}, "
        f"with {'a rise' if first > 1 else 'a decline' if first < -1 else 'little movement'} in year one and "
        f"{'a rise' if second > 1 else 'a decline' if second < -1 else 'little movement'} in year two."
    )


def build_assignment_table(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    profile_labels: dict[int, str],
) -> pd.DataFrame:
    sorted_indices = np.argsort(probabilities, axis=1)
    second_best = sorted_indices[:, -2]
    confidence = probabilities.max(axis=1)
    margin = confidence - probabilities[np.arange(len(probabilities)), second_best]
    out = feature_df[["fips", "county_name", "state", "nri_risk_rating", "incident_count"]].copy()
    out["migration_trend_profile"] = labels.astype(int)
    out["migration_trend_profile_label"] = out["migration_trend_profile"].map(profile_labels)
    out["assignment_confidence"] = confidence
    out["assignment_margin"] = margin
    out["second_best_profile"] = second_best.astype(int)
    out["second_best_profile_label"] = out["second_best_profile"].map(profile_labels)
    return out


def build_risk_profile_summary(assignments: pd.DataFrame, risk_order: list[str]) -> pd.DataFrame:
    counts = (
        assignments.groupby(["nri_risk_rating", "migration_trend_profile", "migration_trend_profile_label"], as_index=False)
        .agg(counties=("fips", "size"), mean_assignment_confidence=("assignment_confidence", "mean"))
    )
    totals = assignments.groupby("nri_risk_rating")["fips"].size().rename("risk_group_count")
    counts = counts.merge(totals, on="nri_risk_rating", how="left")
    counts["share"] = counts["counties"] / counts["risk_group_count"]
    counts["nri_risk_rating"] = pd.Categorical(counts["nri_risk_rating"], categories=risk_order, ordered=True)
    return counts.sort_values(["nri_risk_rating", "share"], ascending=[True, False]).reset_index(drop=True)


def build_profile_summary(
    feature_df: pd.DataFrame,
    assignments: pd.DataFrame,
    interpretations: dict[int, str],
) -> pd.DataFrame:
    merged = assignments.merge(feature_df[["fips", *FEATURE_COLUMNS]], on="fips", how="left")
    rows = []
    for profile, group in merged.groupby("migration_trend_profile"):
        stats = group[FEATURE_COLUMNS].median()
        rows.append(
            {
                "migration_trend_profile": int(profile),
                "migration_trend_profile_label": group["migration_trend_profile_label"].iloc[0],
                "county_count": int(len(group)),
                "mean_assignment_confidence": float(group["assignment_confidence"].mean()),
                "pre_avg": float(stats["mig_pre_avg"]),
                "post_avg": float(stats["mig_post_avg"]),
                "overall_change": float(stats["mig_overall_change"]),
                "first_year_change": float(stats["mig_first_year_change"]),
                "second_year_change": float(stats["mig_second_year_change"]),
                "volatility": float(stats["mig_volatility"]),
                "interpretation": interpretations[int(profile)],
            }
        )
    return pd.DataFrame(rows).sort_values("migration_trend_profile")


def build_profile_series(feature_df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    merged = assignments[["fips", "migration_trend_profile", "migration_trend_profile_label"]].merge(
        feature_df[["fips", *RATE_COLUMNS.values()]], on="fips", how="left"
    )
    long = merged.melt(
        id_vars=["fips", "migration_trend_profile", "migration_trend_profile_label"],
        value_vars=list(RATE_COLUMNS.values()),
        var_name="offset_column",
        value_name="net_migration_per_1000",
    )
    offset_lookup = {value: key for key, value in RATE_COLUMNS.items()}
    long["year_offset"] = long["offset_column"].map(offset_lookup)
    return (
        long.groupby(["migration_trend_profile", "migration_trend_profile_label", "year_offset"], as_index=False)["net_migration_per_1000"]
        .agg(
            median=lambda values: values.quantile(0.5),
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            n="count",
        )
        .sort_values(["migration_trend_profile", "year_offset"])
    )
