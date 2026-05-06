"""
Add a composite housing-market index to climate housing visualization exports.

The index is an equal-weight average of standardized year-over-year housing
market metrics:

- MEDIAN_PPSF_YOY
- AVG_SALE_TO_LIST_YOY
- HOMES_SOLD_YOY
- INVENTORY_YOY

Each component is z-scored over the rows in a source export so metrics with
different units contribute on the same scale. The composite index is the mean
of the available component z-scores for a row. Its monthly change is calculated
within each county as the month-over-month difference in the composite score.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("output/visualizations")
MANIFEST_PATH = OUTPUT_DIR / "incident_housing_manifest.json"
HOUSING_SOURCE_PATH = Path("data/housing/Redfin-Housing-Market-By-County.csv")

INDEX_COMPONENTS = [
    "MEDIAN_PPSF_YOY",
    "AVG_SALE_TO_LIST_YOY",
    "HOMES_SOLD_YOY",
    "INVENTORY_YOY",
]
INDEX_COL = "HOUSING_MARKET_INDEX"
INDEX_MOM_COL = "HOUSING_MARKET_INDEX_MOM"
INDEX_CHANGE_12_TO_24_COL = "HOUSING_MARKET_INDEX_change_in_yoy_12_to_24"


def load_component_source() -> pd.DataFrame:
    source = pd.read_csv(
        HOUSING_SOURCE_PATH,
        usecols=["REGION", "PERIOD_BEGIN", *INDEX_COMPONENTS],
        low_memory=False,
    )
    source["PERIOD_BEGIN"] = pd.to_datetime(source["PERIOD_BEGIN"], errors="coerce").dt.strftime("%Y-%m-%d")
    source["MONTH"] = pd.to_datetime(source["PERIOD_BEGIN"], errors="coerce").dt.strftime("%Y-%m")
    source = source.drop_duplicates(["REGION", "PERIOD_BEGIN", "MONTH"])
    return add_index_columns(source)


def add_missing_component_columns(df: pd.DataFrame, component_source: pd.DataFrame) -> pd.DataFrame:
    missing_components = [col for col in INDEX_COMPONENTS if col not in df.columns]
    if not missing_components or "REGION" not in df.columns:
        return df

    out = df.copy()
    merge_keys = ["REGION"]
    if "PERIOD_BEGIN" in out.columns:
        out["PERIOD_BEGIN"] = pd.to_datetime(out["PERIOD_BEGIN"], errors="coerce").dt.strftime("%Y-%m-%d")
        merge_keys.append("PERIOD_BEGIN")
    elif "MONTH" in out.columns:
        merge_keys.append("MONTH")
    else:
        return out

    source_cols = merge_keys + missing_components
    return out.merge(component_source[source_cols], on=merge_keys, how="left")


def merge_full_source_index(df: pd.DataFrame, component_source: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "REGION" not in out.columns:
        return out

    merge_keys = ["REGION"]
    if "PERIOD_BEGIN" in out.columns:
        out["PERIOD_BEGIN"] = pd.to_datetime(out["PERIOD_BEGIN"], errors="coerce").dt.strftime("%Y-%m-%d")
        merge_keys.append("PERIOD_BEGIN")
    elif "MONTH" in out.columns:
        merge_keys.append("MONTH")
    else:
        return out

    stale_cols = [INDEX_COL, INDEX_MOM_COL]
    source_cols = merge_keys + [INDEX_COL, INDEX_MOM_COL]
    return out.drop(columns=stale_cols, errors="ignore").merge(
        component_source[source_cols],
        on=merge_keys,
        how="left",
    )


def normalize_fips(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split(".").str[0].str.zfill(5)


def add_index_columns(df: pd.DataFrame, recompute_index: bool = True) -> pd.DataFrame:
    drop_cols = [INDEX_CHANGE_12_TO_24_COL]
    if recompute_index:
        drop_cols.extend([INDEX_COL, INDEX_MOM_COL])
    out = df.drop(columns=drop_cols, errors="ignore").copy()
    zscore_cols = []

    if recompute_index or INDEX_COL not in out.columns:
        present_components = [col for col in INDEX_COMPONENTS if col in out.columns]
        if not present_components:
            return out

        for col in present_components:
            values = pd.to_numeric(out[col], errors="coerce")
            std = values.std(skipna=True)
            z_col = f"{col}_Z"
            out[z_col] = (values - values.mean(skipna=True)) / std if pd.notna(std) and std else pd.NA
            zscore_cols.append(z_col)

        out[INDEX_COL] = out[zscore_cols].mean(axis=1, skipna=True)
    sort_cols = [col for col in ["fips", "REGION", "incident_num", "PERIOD_BEGIN", "MONTH"] if col in out.columns]
    if "fips" in out.columns:
        out["fips"] = normalize_fips(out["fips"])
    out = out.sort_values(sort_cols)
    diff_group_cols = [col for col in ["fips", "incident_num"] if col in out.columns]
    if not diff_group_cols and "REGION" in out.columns:
        diff_group_cols = ["REGION"]
    if recompute_index or INDEX_MOM_COL not in out.columns:
        out[INDEX_MOM_COL] = out.groupby(diff_group_cols, dropna=False)[INDEX_COL].diff()
    if {"fips", "incident_num", "month_offset_from_incident"}.issubset(out.columns):
        grouped = (
            out.groupby(["fips", "incident_num"], dropna=False)
            .apply(
                lambda group: pd.Series(
                    {
                        "months_1_12": pd.to_numeric(
                            group.loc[group["month_offset_from_incident"].between(1, 12), INDEX_MOM_COL],
                            errors="coerce",
                        ).mean(),
                        "months_13_24": pd.to_numeric(
                            group.loc[group["month_offset_from_incident"].between(13, 24), INDEX_MOM_COL],
                            errors="coerce",
                        ).mean(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        if {"months_1_12", "months_13_24"}.issubset(grouped.columns):
            grouped[INDEX_CHANGE_12_TO_24_COL] = grouped["months_13_24"] - grouped["months_1_12"]
            out = out.merge(
                grouped[["fips", "incident_num", INDEX_CHANGE_12_TO_24_COL]],
                on=["fips", "incident_num"],
                how="left",
            )
        else:
            out[INDEX_CHANGE_12_TO_24_COL] = pd.NA
    return out.drop(columns=zscore_cols, errors="ignore")


def merge_index_columns(target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    keys = [col for col in ["fips", "incident_num", "MONTH"] if col in out.columns and col in source.columns]
    if len(keys) != 3 or INDEX_COL not in source.columns:
        return out
    if "fips" in out.columns:
        out["fips"] = normalize_fips(out["fips"])
    index_cols = keys + [
        col
        for col in [INDEX_COL, INDEX_MOM_COL, INDEX_CHANGE_12_TO_24_COL, "county_profile"]
        if col in source.columns
    ]
    source_subset = source[index_cols].drop_duplicates(keys)
    stale_cols = [
        col
        for col in out.columns
        if col in [INDEX_COL, INDEX_MOM_COL, INDEX_CHANGE_12_TO_24_COL]
        or col.startswith(f"{INDEX_COL}_")
        or col.startswith(f"{INDEX_MOM_COL}_")
        or col.startswith(f"{INDEX_CHANGE_12_TO_24_COL}_")
    ]
    return out.drop(columns=stale_cols, errors="ignore").merge(source_subset, on=keys, how="left")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    component_source = load_component_source()
    for entry in manifest.get("incident_types", []):
        index_name = entry.get("index_housing_csv")
        index_24m_name = entry.get("index_housing_24mths_csv")
        if not index_name or not index_24m_name:
            continue

        index_path = OUTPUT_DIR / index_name
        index_24m_path = OUTPUT_DIR / index_24m_name
        if not index_path.exists() or not index_24m_path.exists():
            continue

        index_df = add_index_columns(
            merge_full_source_index(
                add_missing_component_columns(
                    pd.read_csv(index_path, dtype={"fips": str}, low_memory=False),
                    component_source,
                ),
                component_source,
            ),
            recompute_index=False,
        )
        index_24m_df = add_index_columns(
            merge_full_source_index(
                add_missing_component_columns(
                    pd.read_csv(index_24m_path, dtype={"fips": str}, low_memory=False),
                    component_source,
                ),
                component_source,
            ),
            recompute_index=False,
        )
        write_csv(index_df, index_path)
        write_csv(index_24m_df, index_24m_path)

        story_1_name = entry.get("story_1_housing_csv")
        if story_1_name:
            story_1_path = OUTPUT_DIR / story_1_name
            if story_1_path.exists():
                story_1_df = pd.read_csv(story_1_path, dtype={"fips": str}, low_memory=False)
                story_1_df = merge_index_columns(story_1_df, index_df)
                story_1_df = story_1_df.drop(columns=["MEDIAN_PPSF_YOY_MOM"], errors="ignore")
                write_csv(story_1_df, story_1_path)

        story_2_name = entry.get("story_2_housing_24mths_csv")
        if story_2_name:
            story_2_path = OUTPUT_DIR / story_2_name
            if story_2_path.exists():
                story_2_df = pd.read_csv(story_2_path, dtype={"fips": str}, low_memory=False)
                story_2_df = merge_index_columns(story_2_df, index_24m_df)
                story_2_df = story_2_df.drop(columns=["MEDIAN_PPSF_YOY"], errors="ignore")
                write_csv(story_2_df, story_2_path)


if __name__ == "__main__":
    main()
