"""Interpret fitted county clusters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from housing_climate_risk.modeling.county_clustering.features import ID_COLUMNS


def build_profiles_for_all_outputs(
    feature_sets: dict[str, pd.DataFrame],
    labels_dir: Path | str,
    output_dir: Path | str,
    top_n: int = 8,
) -> pd.DataFrame:
    """Build readable profile summaries for every saved label file."""

    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = []
    lifts = []

    for labels_path in sorted(labels_dir.glob("*.parquet")):
        labels_df = pd.read_parquet(labels_path)
        if labels_df.empty:
            continue
        feature_set = str(labels_df["feature_set"].iloc[0])
        feature_df = feature_sets[feature_set]
        profile_df, lift_df = build_cluster_profile(feature_df, labels_df, top_n=top_n)
        profiles.append(profile_df)
        lifts.append(lift_df)

    profile_all = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    lift_all = pd.concat(lifts, ignore_index=True) if lifts else pd.DataFrame()
    profile_all.to_csv(output_dir / "cluster_profiles.csv", index=False)
    lift_all.to_csv(output_dir / "cluster_feature_lifts.csv", index=False)
    return profile_all


def build_cluster_profile(
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    top_n: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary rows and per-feature standardized lifts for one result."""

    identity_columns = [column for column in ID_COLUMNS if column in feature_df.columns]
    numeric_columns = [
        column
        for column in feature_df.select_dtypes(include=[np.number]).columns
        if column not in identity_columns
    ]
    merged = labels_df[["fips", "experiment", "feature_set", "model_name", "cluster"]].merge(
        feature_df[["fips", *numeric_columns]], on="fips", how="left"
    )
    merged = merged.loc[merged["cluster"] != -1].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    national_median = merged[numeric_columns].median()
    national_std = merged[numeric_columns].std().replace(0, np.nan)
    cluster_medians = merged.groupby("cluster")[numeric_columns].median()
    standardized_lifts = (cluster_medians - national_median) / national_std

    lift_rows = []
    profile_rows = []
    for cluster_id, zscores in standardized_lifts.iterrows():
        county_rows = labels_df.loc[labels_df["cluster"] == cluster_id]
        county_count = int(len(county_rows))
        high = zscores.dropna().sort_values(ascending=False).head(top_n)
        low = zscores.dropna().sort_values(ascending=True).head(top_n)
        examples = ", ".join(
            county_rows[["county_name", "state"]]
            .dropna()
            .drop_duplicates()
            .head(8)
            .apply(lambda row: f"{row['county_name']}, {row['state']}", axis=1)
        )
        profile_rows.append(
            {
                "experiment": labels_df["experiment"].iloc[0],
                "feature_set": labels_df["feature_set"].iloc[0],
                "model_name": labels_df["model_name"].iloc[0],
                "cluster": int(cluster_id),
                "county_count": county_count,
                "top_high_features": "; ".join(f"{feature} ({value:.2f} SD)" for feature, value in high.items()),
                "top_low_features": "; ".join(f"{feature} ({value:.2f} SD)" for feature, value in low.items()),
                "example_counties": examples,
            }
        )
        for feature, value in zscores.dropna().items():
            lift_rows.append(
                {
                    "experiment": labels_df["experiment"].iloc[0],
                    "feature_set": labels_df["feature_set"].iloc[0],
                    "model_name": labels_df["model_name"].iloc[0],
                    "cluster": int(cluster_id),
                    "feature": feature,
                    "standardized_median_lift": float(value),
                }
            )

    return pd.DataFrame(profile_rows), pd.DataFrame(lift_rows)

