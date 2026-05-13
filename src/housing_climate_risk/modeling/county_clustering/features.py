"""Feature engineering for county clustering.

The processed county file stores most variables as nested dictionaries and
monthly or yearly arrays.  This module turns those records into flat numeric
feature matrices that can be shared by KMeans, Gaussian mixtures, hierarchical
clustering, and HDBSCAN-style methods.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


ID_COLUMNS = ["fips", "county_name", "state", "state_long", "msa_type", "msa_name"]

HOUSING_TIME_SERIES_METRICS = [
    "MEDIAN_SALE_PRICE",
    "MEDIAN_PPSF",
    "INVENTORY",
    "MONTHS_OF_SUPPLY",
    "MEDIAN_DOM",
    "SOLD_ABOVE_LIST",
    "AVG_SALE_TO_LIST",
    "NEW_LISTINGS",
    "HOMES_SOLD",
    "PENDING_SALES",
]

HOUSING_SUMMARY_METRICS = [
    "MEDIAN_SALE_PRICE",
    "MEDIAN_PPSF",
    "INVENTORY",
    "MONTHS_OF_SUPPLY",
    "MEDIAN_DOM",
    "SOLD_ABOVE_LIST",
    "AVG_SALE_TO_LIST",
    "NEW_LISTINGS",
    "HOMES_SOLD",
]

SIZE_DOMINATED_TOKENS = [
    "population",
    "buildvalue",
    "agrivalue",
    "risk_value",
    "property_value",
    "replacement_cost",
    "coverage",
    "policy_count",
    "policies_total",
    "num_policies",
    "event_count_total",
    "deaths_total",
    "injuries_total",
    "damage_total",
    "taxes_paid",
]


def build_feature_sets(counties: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the default first-run feature sets.

    The returned keys match the experiment definitions in ``experiments.py``.
    Each DataFrame includes county identity columns plus numeric model features.
    """

    all_features = build_all_features(counties)
    return {
        "all_features_without_size_variables": drop_size_dominated_columns(all_features),
        "climate_insurance_only": build_climate_insurance_features(counties),
        "housing_time_series_features": build_housing_time_series_features(counties),
    }


def build_all_features(counties: pd.DataFrame) -> pd.DataFrame:
    """Flatten all available county feature families into one table."""

    rows = []
    for row in counties.itertuples(index=False):
        features = _identity_features(row)
        features.update(extract_housing_summary_features(_get(row, "metrics")))
        features.update(extract_climate_features(_get(row, "nri_climate")))
        features.update(extract_insurance_premium_features(_get(row, "insurance_premiums_14_to_24")))
        features.update(extract_insurance_nonrenewal_features(_get(row, "insurance_non_renewal_rates")))
        features.update(extract_property_tax_features(_get(row, "property_tax")))
        features.update(extract_nfip_features(_get(row, "nfip_claims")))
        features.update(extract_storm_features(_get(row, "storm_events")))
        features.update(extract_disaster_features(_get(row, "fema_disaster_declarations")))
        features.update(extract_temperature_features(_get(row, "temp_max_min")))
        rows.append(features)
    return _clean_feature_frame(pd.DataFrame(rows))


def build_climate_insurance_features(counties: pd.DataFrame) -> pd.DataFrame:
    """Build a table focused on hazard exposure and insurance pressure."""

    rows = []
    for row in counties.itertuples(index=False):
        features = _identity_features(row)
        features.update(extract_climate_features(_get(row, "nri_climate")))
        features.update(extract_insurance_premium_features(_get(row, "insurance_premiums_14_to_24")))
        features.update(extract_insurance_nonrenewal_features(_get(row, "insurance_non_renewal_rates")))
        features.update(extract_nfip_features(_get(row, "nfip_claims")))
        features.update(extract_storm_features(_get(row, "storm_events")))
        features.update(extract_disaster_features(_get(row, "fema_disaster_declarations")))
        features.update(extract_temperature_features(_get(row, "temp_max_min")))
        rows.append(features)
    return _clean_feature_frame(pd.DataFrame(rows))


def build_housing_time_series_features(counties: pd.DataFrame) -> pd.DataFrame:
    """Summarize Redfin monthly arrays into trajectory features."""

    rows = []
    for row in counties.itertuples(index=False):
        features = _identity_features(row)
        features.update(extract_housing_time_series_features(_get(row, "metrics")))
        rows.append(features)
    return _clean_feature_frame(pd.DataFrame(rows))


def drop_size_dominated_columns(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns likely to group counties by absolute size alone.

    This keeps ratio, percentile, score, trend, and rate variables while removing
    raw population, property value, coverage, policy count, and similar scale
    variables from the broad all-feature experiment.
    """

    keep_columns = []
    for column in feature_df.columns:
        if column in ID_COLUMNS:
            keep_columns.append(column)
            continue
        lowered = column.lower()
        if any(token in lowered for token in SIZE_DOMINATED_TOKENS):
            if any(token in lowered for token in ["rate", "ratio", "share", "percentile", "per_", "score"]):
                keep_columns.append(column)
            continue
        keep_columns.append(column)
    return feature_df[keep_columns].copy()


def extract_housing_summary_features(metrics: Any) -> dict[str, float]:
    """Extract level, trend, and volatility features from core housing metrics."""

    if not isinstance(metrics, Mapping):
        return {}
    out = {}
    for metric in HOUSING_SUMMARY_METRICS:
        values = _numeric_array(metrics.get(metric))
        prefix = f"housing_{metric.lower()}"
        out.update(_series_summary(prefix, values))
    return out


def extract_housing_time_series_features(metrics: Any) -> dict[str, float]:
    """Extract detailed trajectory features from Redfin monthly arrays."""

    if not isinstance(metrics, Mapping):
        return {}
    out = {}
    for metric in HOUSING_TIME_SERIES_METRICS:
        values = _numeric_array(metrics.get(metric))
        prefix = f"housing_ts_{metric.lower()}"
        out.update(_series_summary(prefix, values))
        out[f"{prefix}_indexed_slope"] = _indexed_slope(values)
        out[f"{prefix}_indexed_volatility"] = _indexed_volatility(values)
    return out


def extract_climate_features(nri_climate: Any) -> dict[str, float]:
    """Extract national risk index scores and hazard-level risk details."""

    if not isinstance(nri_climate, Mapping):
        return {}
    out = {
        "nri_risk_score": _to_float(nri_climate.get("RISK_SCORE")),
        "nri_risk_value": _to_float(nri_climate.get("RISK_VALUE")),
        "nri_population": _to_float(nri_climate.get("POPULATION")),
        "nri_area": _to_float(nri_climate.get("AREA")),
        "nri_buildvalue": _to_float(nri_climate.get("BUILDVALUE")),
        "nri_agrivalue": _to_float(nri_climate.get("AGRIVALUE")),
    }
    population = out["nri_population"]
    if np.isfinite(population) and population > 0:
        out["nri_risk_value_per_capita"] = out["nri_risk_value"] / population

    for item in _iter_records(nri_climate.get("RISKS_BREAKDOWN")):
        if not isinstance(item, Mapping):
            continue
        risk = str(item.get("risk") or item.get("code") or "").lower().replace(" ", "_")
        if not risk:
            continue
        out[f"nri_hazard_{risk}_score"] = _to_float(item.get("risk_score"))
        percentile = item.get("percentile")
        if isinstance(percentile, Mapping):
            out[f"nri_hazard_{risk}_pct_nation"] = _to_float(percentile.get("nation"))
            out[f"nri_hazard_{risk}_pct_state"] = _to_float(percentile.get("state"))
    return out


def extract_insurance_premium_features(premiums: Any) -> dict[str, float]:
    """Extract current insurance premium levels and growth rates."""

    if not isinstance(premiums, Mapping):
        return {}
    out = {}
    out.update(_flatten_numeric_dict("insurance_premium_latest", premiums.get("latest")))
    out.update(_flatten_numeric_dict("insurance_premium_growth", premiums.get("growth_rates")))
    out.update(_flatten_numeric_dict("insurance_premium_average", premiums.get("averages")))
    historical = premiums.get("historical")
    if isinstance(historical, Mapping):
        out.update(_series_summary("insurance_premium_historical_mean", _numeric_array(historical.get("mean"))))
        out.update(_series_summary("insurance_premium_historical_median", _numeric_array(historical.get("median"))))
    return out


def extract_insurance_nonrenewal_features(nonrenewals: Any) -> dict[str, float]:
    """Extract current non-renewal levels and policy pressure trends."""

    if not isinstance(nonrenewals, Mapping):
        return {}
    out = {"insurance_nonrenewal_years_of_data": _to_float(nonrenewals.get("years_of_data"))}
    out.update(_flatten_numeric_dict("insurance_nonrenewal_latest", nonrenewals.get("latest")))
    out.update(_flatten_numeric_dict("insurance_nonrenewal_growth", nonrenewals.get("growth_rates")))
    out.update(_flatten_numeric_dict("insurance_nonrenewal_average", nonrenewals.get("averages")))
    historical = nonrenewals.get("historical")
    if isinstance(historical, Mapping):
        for key in ["non_renewal_rate", "num_policies_total"]:
            out.update(_series_summary(f"insurance_nonrenewal_historical_{key}", _numeric_array(historical.get(key))))
    return out


def extract_property_tax_features(property_tax: Any) -> dict[str, float]:
    """Extract county property tax levels and percentile ranks."""

    if not isinstance(property_tax, Mapping):
        return {}
    out = {}
    for key, value in property_tax.items():
        prefix = f"property_tax_{key}"
        if isinstance(value, Mapping):
            out[f"{prefix}_value"] = _to_float(value.get("value"))
            out.update(_flatten_numeric_dict(f"{prefix}_percentile", value.get("percentiles")))
        else:
            out[prefix] = _to_float(value)
    return out


def extract_nfip_features(nfip_claims: Any) -> dict[str, float]:
    """Summarize NFIP claims and coverage histories."""

    if not isinstance(nfip_claims, Mapping):
        return {}
    out = {}
    for key, values in nfip_claims.items():
        if key == "yearOfLoss":
            continue
        arr = _numeric_array(values)
        prefix = f"nfip_{key}"
        out[f"{prefix}_sum"] = np.nansum(arr) if arr.size else np.nan
        out[f"{prefix}_latest"] = _latest_valid(arr)
        out[f"{prefix}_mean"] = np.nanmean(arr) if _has_valid(arr) else np.nan
    total_payment = np.nansum(
        [
            out.get("nfip_netBuildingPaymentAmount_sum", np.nan),
            out.get("nfip_netContentsPaymentAmount_sum", np.nan),
        ]
    )
    policy_count = out.get("nfip_policyCount_sum")
    if np.isfinite(policy_count) and policy_count > 0:
        out["nfip_total_payment_per_policy"] = total_payment / policy_count
    return out


def extract_storm_features(storm_events: Any) -> dict[str, float]:
    """Summarize NOAA storm event records by county."""

    records = list(_iter_records(storm_events))
    if not records:
        return {}
    out = {}
    scalar_keys = [
        "total_damage_crops",
        "total_damage_property",
        "total_damage_total",
        "total_deaths_total",
        "total_injuries_total",
        "total_event_count",
        "total_damage_total_percentile_nationally",
        "total_event_count_percentile_nationally",
        "total_deaths_total_percentile_nationally",
        "total_injuries_total_percentile_nationally",
    ]
    for key in scalar_keys:
        arr = _numeric_array([record.get(key) for record in records if isinstance(record, Mapping)])
        out[f"storm_{key}_sum"] = np.nansum(arr) if arr.size else np.nan
        out[f"storm_{key}_mean"] = np.nanmean(arr) if _has_valid(arr) else np.nan
        out[f"storm_{key}_max"] = np.nanmax(arr) if _has_valid(arr) else np.nan

    event_type_totals: dict[str, float] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("total_event_count_by_type"), Mapping):
            continue
        for event_type, value in record["total_event_count_by_type"].items():
            event_type_totals[event_type] = event_type_totals.get(event_type, 0.0) + _zero_if_nan(value)
    for event_type, value in event_type_totals.items():
        out[f"storm_event_type_{event_type}_sum"] = value
    return out


def extract_disaster_features(disasters: Any) -> dict[str, float]:
    """Summarize FEMA disaster declarations by incident type and year."""

    if not isinstance(disasters, Mapping):
        return {}
    out = {}
    breakdown = disasters.get("breakdown")
    if isinstance(breakdown, Mapping):
        for key, value in breakdown.items():
            out[f"fema_disaster_{key}_count"] = _to_float(value)
        out["fema_disaster_total_count"] = np.nansum([_to_float(value) for value in breakdown.values()])
    summary = _numeric_array(disasters.get("summary"))
    years = _numeric_array(disasters.get("fyDeclared"))
    out["fema_disaster_years_with_declarations"] = float(np.sum(summary > 0)) if summary.size else np.nan
    out["fema_disaster_latest_year"] = np.nanmax(years) if _has_valid(years) else np.nan
    return out


def extract_temperature_features(temp_records: Any) -> dict[str, float]:
    """Summarize annual and monthly temperature distributions and trends."""

    records = list(_iter_records(temp_records))
    if not records:
        return {}
    out = {}
    for key in [
        "tmax_average_temp_f",
        "tmax_max_temp_f",
        "tmax_min_temp_f",
        "tmax_trend_slope_temp_f",
        "tmin_average_temp_f",
        "tmin_max_temp_f",
        "tmin_min_temp_f",
        "tmin_trend_slope_temp_f",
    ]:
        arr = _numeric_array([record.get(key) for record in records if isinstance(record, Mapping)])
        out[f"temperature_{key}_mean"] = np.nanmean(arr) if _has_valid(arr) else np.nan
        out[f"temperature_{key}_max"] = np.nanmax(arr) if _has_valid(arr) else np.nan
        out[f"temperature_{key}_min"] = np.nanmin(arr) if _has_valid(arr) else np.nan
        out[f"temperature_{key}_volatility"] = np.nanstd(arr) if _has_valid(arr) else np.nan
    return out


def _identity_features(row: Any) -> dict[str, Any]:
    return {column: _get(row, column) for column in ID_COLUMNS if hasattr(row, column)}


def _clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if column in ID_COLUMNS:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def _series_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_latest": np.nan,
            f"{prefix}_mean_last_12": np.nan,
            f"{prefix}_change_12": np.nan,
            f"{prefix}_change_36": np.nan,
            f"{prefix}_slope": np.nan,
            f"{prefix}_volatility": np.nan,
            f"{prefix}_missing_rate": np.nan,
        }
    return {
        f"{prefix}_latest": _latest_valid(values),
        f"{prefix}_mean_last_12": _trailing_mean(values, 12),
        f"{prefix}_change_12": _pct_change_over(values, 12),
        f"{prefix}_change_36": _pct_change_over(values, 36),
        f"{prefix}_slope": _linear_slope(values),
        f"{prefix}_volatility": np.nanstd(values) if _has_valid(values) else np.nan,
        f"{prefix}_missing_rate": float(np.isnan(values).mean()),
    }


def _flatten_numeric_dict(prefix: str, value: Any) -> dict[str, float]:
    out = {}
    if not isinstance(value, Mapping):
        return out
    for key, nested_value in value.items():
        if isinstance(nested_value, Mapping):
            out.update(_flatten_numeric_dict(f"{prefix}_{key}", nested_value))
        elif _is_array_like(nested_value):
            out.update(_series_summary(f"{prefix}_{key}", _numeric_array(nested_value)))
        else:
            out[f"{prefix}_{key}"] = _to_float(nested_value)
    return out


def _numeric_array(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    if isinstance(values, np.ndarray):
        return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, Mapping)):
        return pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return np.array([_to_float(values)], dtype=float)


def _iter_records(values: Any) -> Iterable[Any]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, Mapping)):
        return values
    return []


def _latest_valid(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(valid[-1]) if valid.size else np.nan


def _trailing_mean(values: np.ndarray, periods: int) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan
    return float(np.mean(valid[-periods:]))


def _pct_change_over(values: np.ndarray, periods: int) -> float:
    valid = values[np.isfinite(values)]
    if valid.size <= periods:
        return np.nan
    start = valid[-periods - 1]
    end = valid[-1]
    if start == 0 or not np.isfinite(start):
        return np.nan
    return float((end - start) / abs(start))


def _linear_slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(values.size, dtype=float)[mask]
    y = values[mask]
    return float(np.polyfit(x, y, 1)[0])


def _indexed_slope(values: np.ndarray) -> float:
    indexed = _indexed_values(values)
    return _linear_slope(indexed)


def _indexed_volatility(values: np.ndarray) -> float:
    indexed = _indexed_values(values)
    return float(np.nanstd(indexed)) if _has_valid(indexed) else np.nan


def _indexed_values(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=True)
    valid = np.where(np.isfinite(values) & (values != 0))[0]
    if valid.size == 0:
        return np.full(values.shape, np.nan, dtype=float)
    return values / values[valid[0]] * 100.0


def _has_valid(values: np.ndarray) -> bool:
    return values.size > 0 and bool(np.isfinite(values).any())


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _zero_if_nan(value: Any) -> float:
    number = _to_float(value)
    return 0.0 if not np.isfinite(number) else number


def _is_array_like(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))


def _get(row: Any, key: str) -> Any:
    return getattr(row, key, None)

