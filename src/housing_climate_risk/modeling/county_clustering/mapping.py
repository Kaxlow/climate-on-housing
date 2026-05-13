"""Join cluster labels to county boundaries for map review."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


def build_cluster_geojsons(
    labels_dir: Path | str,
    boundaries_path: Path | str,
    output_dir: Path | str,
    limit: int | None = None,
) -> list[Path]:
    """Create one GeoJSON per clustering output.

    The GeoJSON files are intentionally simple: they include the original county
    geometry plus the assigned cluster, feature set, and model name.  They can be
    loaded in QGIS, kepler.gl, or the project's web visualizations.
    """

    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counties = _read_county_boundaries(boundaries_path)
    paths = []

    label_paths = sorted(labels_dir.glob("*.parquet"))
    if limit is not None:
        label_paths = label_paths[:limit]

    for labels_path in label_paths:
        labels_df = pd.read_parquet(labels_path)
        mapped = counties.merge(
            labels_df[["fips", "county_name", "state", "experiment", "feature_set", "model_name", "cluster"]],
            on="fips",
            how="left",
        )
        output_path = output_dir / f"{labels_path.stem}.geojson"
        mapped.to_file(output_path, driver="GeoJSON")
        paths.append(output_path)
    return paths


def _read_county_boundaries(boundaries_path: Path | str) -> gpd.GeoDataFrame:
    counties = gpd.read_file(boundaries_path)
    fips_column = _find_fips_column(counties)
    counties["fips"] = counties[fips_column].astype(str).str.zfill(5)
    return counties


def _find_fips_column(counties: gpd.GeoDataFrame) -> str:
    for column in ["GEOID", "geoid", "FIPS", "fips", "COUNTY_ID", "county_id"]:
        if column in counties.columns:
            return column
    raise KeyError("Could not find a county FIPS/GEOID column in the boundary file.")

