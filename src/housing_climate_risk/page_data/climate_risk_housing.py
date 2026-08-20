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
_STABLE_STATES_PATH = (
    ROOT
    / "data"
    / "fipsgeo"
    / "census_state_boundaries"
    / "census_state_boundaries.shp"
)
_LEGACY_STATE_PATHS = sorted(
    (ROOT / "data" / "fipsgeo").glob("cb_*_us_state_20m/cb_*_us_state_20m.shp"),
    reverse=True,
)
STATES_PATH = (
    _STABLE_STATES_PATH
    if _STABLE_STATES_PATH.exists() or not _LEGACY_STATE_PATHS
    else _LEGACY_STATE_PATHS[0]
)
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
FEATURE_FOCUS_EVENTS = {
    "Very Low": [
        {
            "fips": "41031",
            "source_event_id": "5126",
            "position": "Above",
            "display": "Jefferson County, OR — Akawana Fire, Jun 2016",
        },
        {
            "fips": "24009",
            "source_event_id": "4261",
            "position": "Below",
            "display": "Calvert County, MD — Winter Storm and Snowstorm, Jan 2016",
        },
    ],
    "Low": [
        {
            "fips": "16027",
            "source_event_id": "4342",
            "position": "Above",
            "display": "Canyon County, ID — Flooding, Mar–Jun 2017",
        },
        {
            "fips": "51550",
            "source_event_id": "4291",
            "position": "Below",
            "display": "Chesapeake, VA — Hurricane Matthew, Oct 2016",
        },
    ],
    "Medium": [
        {
            "fips": "34015",
            "source_event_id": "4614",
            "position": "Above",
            "display": "Gloucester County, NJ — Remnants of Hurricane Ida, Sep 2021",
        },
        {
            "fips": "24003",
            "source_event_id": "4261",
            "position": "Below",
            "display": "Anne Arundel County, MD — Snowstorm, Jan 2016",
        },
    ],
    "High": [
        {
            "fips": "48157",
            "source_event_id": "4332",
            "position": "Below",
            "display": "Fort Bend County, TX — Hurricane Harvey Flooding, Aug 2017",
        },
        {
            "fips": "12083",
            "source_event_id": "3377",
            "position": "Above",
            "display": "Marion County, FL — Hurricane Matthew, Oct 2016",
        },
    ],
    "Very High": [
        {
            "fips": "06059",
            "source_event_id": "4305",
            "position": "Below",
            "display": "Orange County, CA — Winter Storms and Flooding, Jan 2017",
        },
        {
            "fips": "12086",
            "source_event_id": "3561",
            "position": "Above",
            "display": "Miami-Dade County, FL — Tropical Storm Elsa, Jul 2021",
        },
    ],
}
WITHIN_GROUP_FEATURES = [
    ("Economic", "Income factors", "net_earnings_per_capita_usd", "currency"),
    ("Economic", "Income factors", "dividends_interest_rent_per_capita_usd", "currency"),
    ("Economic", "Income factors", "transfer_receipts_per_capita_usd", "currency"),
    ("Economic", "Cost factors", "homeowners_insurance_pct_income", "percent"),
    ("Economic", "Cost factors", "property_taxes_pct_income", "percent"),
    ("Economic", "Cost factors", "utilities_pct_income", "percent"),
    ("Economic", "Cost factors", "owner_cost_burden_30pct_plus_pct", "percent"),
    ("Economic", "Employment", "unemployment_rate_pct", "percent"),
    ("Demographic", "Population trend", "net_migration_rate_pct", "percent"),
    ("Demographic", "Population vulnerability factors", "age_65_plus_pct", "percent"),
    ("Demographic", "Population vulnerability factors", "communication_barrier_pct", "percent"),
    ("Demographic", "Population vulnerability factors", "disability_pct", "percent"),
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


def latest_complete_calendar_window(
    con: duckdb.DuckDBPyConnection,
    *,
    years: int = 10,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return [start, end) for the latest complete calendar years in Redfin."""

    latest_year = con.execute(
        """
        SELECT max(calendar_year)
        FROM (
            SELECT
                year(period_begin) AS calendar_year
            FROM mart.redfin_county_monthly
            WHERE property_type = 'All Residential'
              AND period_begin IS NOT NULL
            GROUP BY year(period_begin)
            HAVING count(DISTINCT month(period_begin)) = 12
        )
        """
    ).fetchone()[0]
    if latest_year is None:
        raise ValueError("Redfin data has no complete calendar year")
    end = pd.Timestamp(year=int(latest_year) + 1, month=1, day=1)
    start = pd.Timestamp(year=int(latest_year) - years + 1, month=1, day=1)
    return start, end


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


def build_price_risk(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    analysis_start, analysis_end = latest_complete_calendar_window(con)
    latest_year_start = analysis_end - pd.DateOffset(years=1)
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
                WHEN try_cast(MEDIAN_PPSF_YOY AS DOUBLE) <= -888888000 THEN NULL
                ELSE try_cast(MEDIAN_PPSF_YOY AS DOUBLE)
            END) AS avg_median_ppsf_yoy,
            count(*) FILTER (
                WHERE try_cast(MEDIAN_PPSF_YOY AS DOUBLE) IS NOT NULL
                  AND try_cast(MEDIAN_PPSF_YOY AS DOUBLE) > -888888000
            ) AS observed_months
        FROM mart.redfin_county_monthly
        WHERE property_type = 'All Residential'
          AND period_begin >= ?
          AND period_begin < ?
          AND fips IS NOT NULL
        GROUP BY fips
        """,
        [latest_year_start, analysis_end],
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
                    WHEN try_cast(r.MEDIAN_PPSF_YOY AS DOUBLE) <= -888888000 THEN NULL
                    ELSE try_cast(r.MEDIAN_PPSF_YOY AS DOUBLE)
                END) AS median_ppsf_yoy
            FROM mart.redfin_county_monthly AS r
            WHERE r.property_type = 'All Residential'
              AND r.period_begin >= ?
              AND r.period_begin < ?
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
        """,
        [analysis_start, analysis_end],
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
    history_cap_lower = history["median_ppsf_yoy"].quantile(0.10)
    history_cap_upper = history["median_ppsf_yoy"].quantile(0.90)

    return {
        "hazards": [{"key": h["key"], "label": h["label"]} for h in HAZARDS],
        "counties": counties,
        "countyHistoryMonths": history_month_labels,
        "countyHistorySeries": county_history_series,
        "ratingHistoriesByHazard": rating_histories,
        "summary": {
            "analysisStart": analysis_start.strftime("%Y-%m-%d"),
            "analysisEnd": (analysis_end - pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
            "countyCount": int(df["fips"].nunique()),
            "medianAvgPpsfYoy": serialize_number(df["avg_median_ppsf_yoy"].median(), 5),
            "ppsfCapLower": serialize_number(cap_lower, 5),
            "ppsfCapUpper": serialize_number(cap_upper, 5),
            "historyPpsfCapLower": serialize_number(history_cap_lower, 5),
            "historyPpsfCapUpper": serialize_number(history_cap_upper, 5),
        },
    }


def load_state_geometries() -> dict[str, tuple[str, object]]:
    """Load the cartographic state land geometries used by every county map."""
    if not STATES_PATH.exists():
        raise FileNotFoundError(
            f"State boundary shapefile is required at {STATES_PATH}. "
            "Run `download-data census-boundaries` before rebuilding."
        )

    import geopandas as gpd

    states = gpd.read_file(STATES_PATH).to_crs("EPSG:4326")
    states = states.loc[
        states["STUSPS"].isin(
            [
                "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
                "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
                "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
                "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
                "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
                "WY",
            ]
        )
    ]
    return {
        str(row.STATEFP).zfill(2): (
            row.STUSPS,
            row.geometry.simplify(0.02, preserve_topology=True),
        )
        for row in states.itertuples(index=False)
        if not row.geometry.is_empty
    }


def build_geojson(
    fips_set: set[str],
    state_geometries: dict[str, tuple[str, object]],
) -> dict[str, object]:
    from shapely.geometry import mapping, shape
    from shapely.geometry.collection import GeometryCollection
    from shapely.geometry.multipolygon import MultiPolygon
    from shapely.geometry.polygon import Polygon, orient
    from shapely.ops import unary_union

    raw = json.loads(COUNTIES_PATH.read_text(encoding="utf-8"))
    features = []
    for feature in raw["features"]:
        props = feature.get("properties", {})
        fips = str(props.get("GEOID") or props.get("GEOID10") or "").zfill(5)
        if fips in fips_set:
            state_record = state_geometries.get(fips[:2])
            if state_record is None:
                continue
            tolerance = 0.08 if fips.startswith("02") else 0.025
            geometry = shape(feature["geometry"]).simplify(tolerance, preserve_topology=True)
            geometry = geometry.intersection(state_record[1])
            if isinstance(geometry, GeometryCollection):
                geometry = unary_union(
                    [part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon))]
                )
            if not geometry.is_empty:
                if isinstance(geometry, Polygon):
                    geometry = orient(geometry, sign=-1.0)
                elif isinstance(geometry, MultiPolygon):
                    geometry = MultiPolygon([orient(part, sign=-1.0) for part in geometry.geoms])
                features.append({"type": "Feature", "properties": {"fips": fips}, "geometry": mapping(geometry)})
    return {"type": "FeatureCollection", "features": features}


def build_state_geojson(
    fips_set: set[str],
    state_geometries: dict[str, tuple[str, object]],
    county_geojson: dict[str, object] | None = None,
) -> dict[str, object]:
    """Dissolve unsimplified counties, then simplify only completed state shells."""
    from shapely.geometry import MultiPolygon, Polygon, mapping, shape
    from shapely.ops import unary_union

    raw = county_geojson or json.loads(COUNTIES_PATH.read_text(encoding="utf-8"))
    features = []
    counties_by_state: dict[str, list[object]] = {}
    for feature in raw.get("features", []):
        properties = feature.get("properties", {})
        fips = str(
            properties.get("fips")
            or properties.get("GEOID")
            or properties.get("GEOID10")
            or ""
        ).zfill(5)
        if fips not in fips_set or not feature.get("geometry"):
            continue
        counties_by_state.setdefault(fips[:2], []).append(shape(feature["geometry"]))
    for state_fips, county_geometries in counties_by_state.items():
        state_record = state_geometries.get(state_fips)
        if state_record is None or not county_geometries:
            continue
        state_abbr = state_record[0]
        geometry = unary_union(county_geometries)
        if geometry.is_empty:
            continue
        if isinstance(geometry, Polygon):
            geometry = Polygon(geometry.exterior)
        elif isinstance(geometry, MultiPolygon):
            geometry = MultiPolygon(
                [Polygon(polygon.exterior) for polygon in geometry.geoms]
            )
        tolerance = 0.08 if state_fips == "02" else 0.025
        geometry = geometry.simplify(tolerance, preserve_topology=True)
        features.append(
            {
                "type": "Feature",
                "properties": {"state": state_abbr, "stateFips": state_fips},
                "geometry": mapping(geometry),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _select_story_peer_candidates(
    background: pd.DataFrame,
    eligible_line_ids: set[str],
    *,
    count: int = 8,
) -> list[dict[str, object]]:
    """Select percentile-spaced peers, preferring trajectories inside the IQR tolerance."""
    selected: list[dict[str, object]] = []
    selected_fips: set[str] = set()
    selected_line_ids: set[str] = set()
    for target in np.linspace(5, 95, count):
        available = background.loc[
            ~background["fips"].astype(str).isin(selected_fips)
            & ~background["line_id"].astype(str).isin(selected_line_ids)
        ].copy()
        if available.empty:
            break

        strict = available.loc[
            available["line_id"].astype(str).isin(eligible_line_ids)
        ].copy()
        if not strict.empty:
            candidate = strict.iloc[
                (strict["pct_rank"] - target).abs().argsort()[:1]
            ].iloc[0]
        else:
            available["target_distance"] = (available["pct_rank"] - target).abs()
            candidate = available.sort_values(
                [
                    "max_normalized_band_deviation",
                    "mean_normalized_band_deviation",
                    "target_distance",
                    "line_id",
                ],
                na_position="last",
            ).iloc[0]

        record = candidate.to_dict()
        selected.append(record)
        selected_fips.add(str(record["fips"]))
        selected_line_ids.add(str(record["line_id"]))
    return selected


def _build_story_example_lines(
    complete: pd.DataFrame,
    line_avg: pd.DataFrame,
    *,
    anchor_col: str,
    metric: str,
    eligible_line_ids: set[str],
) -> list[dict[str, object]]:
    """Select two fixed focus events plus eight IQR-constrained context lines."""
    output: list[dict[str, object]] = []
    for risk in RISK_ORDER:
        group = line_avg.loc[line_avg["riskRating"].eq(risk)].copy()
        group = group.loc[
            group["max_metric"].le(100)
            & group["min_metric"].ge(-100)
            & group["pct_rank"].notna()
        ].copy()
        if group.empty:
            continue

        selected: list[tuple[dict[str, object], dict[str, object] | None]] = []
        selected_fips: set[str] = set()
        selected_line_ids: set[str] = set()
        for specification in FEATURE_FOCUS_EVENTS[risk]:
            match = group.loc[
                group["fips"].astype(str).eq(specification["fips"])
                & group["line_id"].astype(str).str.startswith(
                    f"fema:{specification['source_event_id']}:"
                )
            ]
            if match.empty:
                raise ValueError(
                    f"Requested focus event is unavailable in the complete window: "
                    f"{specification['display']}"
                )
            candidate = match.iloc[0].to_dict()
            selected.append((candidate, specification))
            selected_fips.add(str(candidate["fips"]))
            selected_line_ids.add(str(candidate["line_id"]))

        background = group.loc[
            ~group["fips"].astype(str).isin(selected_fips)
            & ~group["line_id"].astype(str).isin(selected_line_ids)
        ].copy()
        for candidate in _select_story_peer_candidates(
            background,
            eligible_line_ids,
        ):
            selected.append((candidate, None))
            selected_fips.add(str(candidate["fips"]))
            selected_line_ids.add(str(candidate["line_id"]))

        group_median = group["avg_metric"].median()
        for candidate, specification in selected:
            rows = complete.loc[complete["line_id"].eq(candidate["line_id"])].sort_values(anchor_col)
            output.append(
                {
                    "riskRating": risk,
                    "lineId": candidate["line_id"],
                    "fips": str(candidate["fips"]).zfill(5),
                    "county": candidate["county_label"],
                    "state": candidate["state_code"],
                    "pctRank": serialize_number(candidate["pct_rank"], 2),
                    "avgPpsfYoy": serialize_number(candidate["avg_metric"], 5),
                    "groupMedianPpsfYoy": serialize_number(group_median, 5),
                    "samplePosition": (
                        f"{specification['position']} group median"
                        if specification
                        else "Context county"
                    ),
                    "displayLabel": (
                        specification["display"]
                        if specification
                        else str(candidate["county_label"])
                    ),
                    "isFocus": specification is not None,
                    "focusPosition": specification["position"] if specification else None,
                    "withinPeerIqrTolerance": str(candidate["line_id"]) in eligible_line_ids,
                    "maxNormalizedBandDeviation": serialize_number(
                        candidate.get("max_normalized_band_deviation"), 5
                    ),
                    "values": [
                        {
                            "month": int(getattr(row, anchor_col)),
                            "value": serialize_number(getattr(row, metric), 5),
                        }
                        for row in rows.itertuples(index=False)
                        if pd.notna(getattr(row, metric))
                    ],
                }
            )
    return output


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
    story_examples: bool = False,
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
    line_band_fit = pd.DataFrame(
        columns=[
            "line_id",
            "months",
            "all_inside",
            "max_normalized_band_deviation",
            "mean_normalized_band_deviation",
        ]
    )
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
        band_join["outside_sample_band_distance"] = pd.concat(
            [
                band_join["lower_allowed"] - band_join[metric],
                band_join[metric] - band_join["upper_allowed"],
                pd.Series(0.0, index=band_join.index),
            ],
            axis=1,
        ).max(axis=1)
        band_join["sample_band_scale"] = (
            band_join["riskRating"]
            .map(max_width_by_risk)
            .abs()
            .clip(lower=0.01)
        )
        band_join["normalized_band_deviation"] = (
            band_join["outside_sample_band_distance"]
            / band_join["sample_band_scale"]
        )
        line_band_fit = (
            band_join.groupby("line_id", as_index=False)
            .agg(
                months=(anchor_col, "nunique"),
                all_inside=("inside_sample_band", "all"),
                max_normalized_band_deviation=(
                    "normalized_band_deviation",
                    "max",
                ),
                mean_normalized_band_deviation=(
                    "normalized_band_deviation",
                    "mean",
                ),
            )
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
    line_avg = line_avg.merge(line_band_fit, on="line_id", how="left")

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

    # The story view uses two explicitly requested focus events and eight
    # IQR-constrained, percentile-spaced context trajectories per risk group.
    if story_examples:
        example_lines = _build_story_example_lines(
            complete,
            line_avg,
            anchor_col=anchor_col,
            metric=metric,
            eligible_line_ids=eligible_line_ids,
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
    analysis_start, analysis_end = latest_complete_calendar_window(con)
    events = load_disaster_events(con)
    events = events.loc[
        events["event_start_month"].ge(analysis_start)
        & events["event_start_month"].lt(analysis_end)
    ].copy()
    housing = load_redfin_county_monthly(con)
    for column in ["median_ppsf_yoy", "avg_sale_to_list_yoy", "homes_sold_yoy", "inventory_yoy", "housing_market_index"]:
        if column in housing:
            housing.loc[pd.to_numeric(housing[column], errors="coerce").le(-888888000), column] = np.nan
    metric = "median_ppsf_yoy"

    # Both windows use the split-anchored event_window_month: pre-event months
    # relative to event start and post-event months relative to event end.
    affected = build_affected_event_windows(events, housing, pre_event_months=12, post_event_months=60)
    if affected.empty:
        empty = {"byRating": [], "affectedCounties": [], "riskCounts": {}, "exampleCountyLines": []}
        return {
            "windowA": empty,
            "windowB": empty,
            "summary": {
                "events": 0,
                "analysisStart": analysis_start.strftime("%Y-%m-%d"),
                "analysisEnd": (analysis_end - pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
            },
        }

    nri = con.execute("SELECT fips, risk_rating FROM mart.nri_county_risk WHERE fips IS NOT NULL").df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)

    window_a = _build_window_data(
        affected,
        nri,
        metric,
        pre_months=12,
        post_months=36,
        eligible_feature_fips_by_risk=eligible_feature_fips_by_risk,
        story_examples=True,
    )

    window_b = _build_window_data(
        affected,
        nri,
        metric,
        pre_months=12,
        post_months=60,
        eligible_feature_fips_by_risk=eligible_feature_fips_by_risk,
    )

    return {
        "windowA": window_a,
        "windowB": window_b,
        "summary": {
            "events": int(events["event_key"].nunique()),
            "analysisStart": analysis_start.strftime("%Y-%m-%d"),
            "analysisEnd": (analysis_end - pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
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
    analysis_start, analysis_end = latest_complete_calendar_window(con)
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
                WHEN try_cast(MEDIAN_PPSF_YOY AS DOUBLE) <= -888888000 THEN NULL
                ELSE try_cast(MEDIAN_PPSF_YOY AS DOUBLE)
            END) AS median_ppsf_yoy
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
          AND period_begin IS NOT NULL
          AND coalesce(property_type, PROPERTY_TYPE_1) = 'All Residential'
          AND period_begin >= ?
          AND period_begin < ?
        GROUP BY fips, date_trunc('month', period_begin)
        ORDER BY fips, month
        """,
        [analysis_start, analysis_end],
    ).df()
    history["fips"] = history["fips"].astype(str).str.zfill(5)
    history["median_ppsf_yoy"] = pd.to_numeric(history["median_ppsf_yoy"], errors="coerce")
    history = history.dropna(subset=["median_ppsf_yoy"])

    events = load_disaster_events(con)
    events = events.loc[
        events["event_start_month"].ge(analysis_start)
        & events["event_start_month"].lt(analysis_end)
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
        "eventCountyFips": sorted(events_by_fips),
        "historyStart": history["month"].min().strftime("%Y-%m"),
        "historyEnd": history["month"].max().strftime("%Y-%m"),
    }


def _build_legacy_feature_payload(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
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
        "b25103_median_real_estate_taxes_paid_total_est",
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
    afford["estimated_annual_property_tax"] = afford[
        "b25103_median_real_estate_taxes_paid_total_est"
    ]
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
    afford["property_taxes_pct_income"] = afford["estimated_annual_property_tax"] / afford["income_median_household_usd"].replace(0, np.nan) * 100
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
        """
    ).df()
    if not bea_features.empty:
        bea_features["fips"] = bea_features["fips"].astype(str).str.zfill(5)
        features = features.merge(bea_features, on="fips", how="left")
    redfin_features = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            avg(CASE WHEN try_cast(MEDIAN_PPSF_YOY AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(MEDIAN_PPSF_YOY AS DOUBLE) END) AS median_ppsf_yoy,
            avg(CASE WHEN try_cast(AVG_SALE_TO_LIST_YOY AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(AVG_SALE_TO_LIST_YOY AS DOUBLE) END) AS avg_sale_to_list_yoy,
            avg(CASE WHEN try_cast(HOMES_SOLD_YOY AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(HOMES_SOLD_YOY AS DOUBLE) END) AS homes_sold_yoy,
            avg(CASE WHEN try_cast(INVENTORY_YOY AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(INVENTORY_YOY AS DOUBLE) END) AS inventory_yoy,
            avg(CASE WHEN try_cast(NEW_LISTINGS_YOY AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(NEW_LISTINGS_YOY AS DOUBLE) END) AS new_listings_yoy,
            avg(CASE WHEN try_cast(MEDIAN_DOM_YOY AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(MEDIAN_DOM_YOY AS DOUBLE) END) AS median_dom_yoy,
            avg(CASE WHEN try_cast(PRICE_DROPS_YOY AS DOUBLE) <= -888888000 THEN NULL ELSE try_cast(PRICE_DROPS_YOY AS DOUBLE) END) AS price_drops_yoy
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
        ("Housing Market", "Median Days on Market YOY", "median_dom_yoy", "pct", "mart.redfin_county_monthly"),
        ("Housing Market", "Active Listings with Price Drops YOY", "price_drops_yoy", "pct", "mart.redfin_county_monthly"),
        ("Climate", "Temperature", "avg_temperature_f", "temperature_f", "mart.ncei_county_weather_monthly"),
        ("Climate", "Precipitation", "precipitation_inches", "inches", "mart.ncei_county_weather_monthly"),
    ]
    for _, _, column, _, _ in feature_defs:
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    feature_display_meta = {
        label: {"category": category, "column": column, "format": fmt, "source": source}
        for category, label, column, fmt, source in feature_defs
    }

    return {
        "riskOrder": RISK_ORDER,
        "featureMeta": feature_display_meta,
    }


def _spearman_correlation(x: pd.Series, y: pd.Series) -> float:
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(paired.iloc[:, 0].rank().corr(paired.iloc[:, 1].rank()))


FEATURE_TARGET_COLUMN = "event_window_avg_ppsf_yoy"


def _county_average_event_window_target(
    complete: pd.DataFrame,
    *,
    metric: str = "median_ppsf_yoy",
) -> pd.DataFrame:
    """Return one average PPSF YoY level across complete event windows per county."""
    event_rows = complete[["fips", metric]].dropna().copy()
    if event_rows.empty:
        return pd.DataFrame(columns=["fips", FEATURE_TARGET_COLUMN])
    return (
        event_rows.groupby("fips", as_index=False)[metric]
        .mean()
        .rename(columns={metric: FEATURE_TARGET_COLUMN})
    )


def _bootstrap_spearman_ci(
    frame: pd.DataFrame,
    feature: str,
    *,
    iterations: int = 160,
    seed: int,
) -> tuple[float, float]:
    paired = frame[[feature, FEATURE_TARGET_COLUMN]].dropna().reset_index(drop=True)
    if len(paired) < 12:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ranked_x = paired[feature].rank().to_numpy(dtype=float)
    ranked_y = paired[FEATURE_TARGET_COLUMN].rank().to_numpy(dtype=float)
    values: list[float] = []
    for _ in range(iterations):
        sample_index = rng.integers(0, len(paired), len(paired))
        correlation = float(np.corrcoef(ranked_x[sample_index], ranked_y[sample_index])[0, 1])
        if np.isfinite(correlation):
            values.append(correlation)
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def build_feature_payload(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """Build the within-risk feature story from the DuckDB feature layer."""
    feature_columns = [item[2] for item in WITHIN_GROUP_FEATURES]
    economic_columns = feature_columns[:8]
    demographic_columns = feature_columns[8:]

    economic = con.execute(
        f"""
        SELECT lpad(fips, 5, '0') AS fips,
               {', '.join(f'avg({column}) AS {column}' for column in economic_columns)}
        FROM feature.county_economic_annual
        WHERE fips IS NOT NULL
          AND year >= (SELECT max(year) - 9 FROM feature.county_economic_annual)
        GROUP BY fips
        """
    ).df()
    demographic = con.execute(
        f"""
        SELECT lpad(fips, 5, '0') AS fips,
               {', '.join(f'avg({column}) AS {column}' for column in demographic_columns)}
        FROM feature.county_demographic_annual
        WHERE fips IS NOT NULL
          AND year >= (SELECT max(year) - 9 FROM feature.county_demographic_annual)
        GROUP BY fips
        """
    ).df()
    nri = con.execute(
        "SELECT lpad(fips, 5, '0') AS fips, risk_rating FROM feature.county_risk WHERE fips IS NOT NULL"
    ).df()
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)

    analysis_start, analysis_end = latest_complete_calendar_window(con)
    events = load_disaster_events(con)
    events = events.loc[
        events["event_start_month"].ge(analysis_start)
        & events["event_start_month"].lt(analysis_end)
    ].copy()
    housing = load_redfin_county_monthly(con)
    for column in ["median_ppsf_yoy"]:
        housing[column] = pd.to_numeric(housing[column], errors="coerce")
        housing.loc[housing[column].le(-888888000), column] = np.nan
    affected = build_affected_event_windows(events, housing, pre_event_months=12, post_event_months=36)
    required_months = event_window_months(12, 36)
    complete = filter_complete_event_window_lines(
        affected,
        x_col="event_window_month",
        line_col="line_id",
        metric_col="median_ppsf_yoy",
        required_x_values=required_months,
    ).copy()
    complete = complete.loc[complete["event_window_month"].isin(required_months)].copy()
    complete = complete.merge(nri[["fips", "riskRating"]], on="fips", how="left")

    county_target = _county_average_event_window_target(complete)
    counties = (
        nri[["fips", "riskRating"]]
        .merge(economic, on="fips", how="left")
        .merge(demographic, on="fips", how="left")
        .merge(county_target, on="fips", how="inner")
    )
    for column in [FEATURE_TARGET_COLUMN, *feature_columns]:
        counties[column] = pd.to_numeric(counties[column], errors="coerce")

    minimum_effect = 0.10
    importance_by_risk: dict[str, list[dict[str, object]]] = {}
    county_rows_by_risk: dict[str, list[dict[str, object]]] = {}
    subgroup_payload: dict[str, dict[str, object]] = {}
    subgroup_by_fips: dict[str, int] = {}

    for risk_index, risk in enumerate(RISK_ORDER):
        group = counties.loc[counties["riskRating"].eq(risk)].copy()
        metrics: list[dict[str, object]] = []
        for feature_index, (_, _, feature, _) in enumerate(WITHIN_GROUP_FEATURES):
            paired = group[[feature, FEATURE_TARGET_COLUMN]].dropna()
            rho = _spearman_correlation(paired[feature], paired[FEATURE_TARGET_COLUMN])
            ci_low, ci_high = _bootstrap_spearman_ci(
                paired,
                feature,
                seed=20260814 + risk_index * 100 + feature_index,
            )
            ci_effect = (
                np.isfinite(ci_low)
                and np.isfinite(ci_high)
                and (ci_low > minimum_effect or ci_high < -minimum_effect)
            )
            metrics.append(
                {
                    "feature": feature,
                    "rho": serialize_number(rho, 4),
                    "absRho": serialize_number(abs(rho), 4),
                    "ciLow": serialize_number(ci_low, 4),
                    "ciHigh": serialize_number(ci_high, 4),
                    "passesThreshold": bool(ci_effect),
                    "n": int(len(paired)),
                }
            )
        importance_by_risk[risk] = metrics

        county_rows_by_risk[risk] = [
            {
                "fips": row.fips,
                "target": serialize_number(getattr(row, FEATURE_TARGET_COLUMN), 5),
                "values": {
                    feature: serialize_number(getattr(row, feature), 5)
                    for feature in feature_columns
                },
            }
            for row in group.itertuples(index=False)
            if pd.notna(getattr(row, FEATURE_TARGET_COLUMN))
        ]

        performance_group = group.dropna(subset=[FEATURE_TARGET_COLUMN]).copy()
        if len(performance_group) < 9:
            subgroup_payload[risk] = {"groups": [], "excludedOutliers": 0}
            continue
        ranked_metrics = sorted(
            [item for item in metrics if item["rho"] is not None],
            key=lambda item: float(item["absRho"] or 0),
            reverse=True,
        )
        strong_metrics = [item for item in ranked_metrics if float(item["absRho"] or 0) >= 0.3]
        distribution_metrics = strong_metrics or ranked_metrics[:1]
        distribution_features = [str(item["feature"]) for item in distribution_metrics]

        subgroup_count = 3 if risk == "Very High" else 4
        performance_group = performance_group.sort_values(
            [FEATURE_TARGET_COLUMN, "fips"], ascending=[False, True]
        ).reset_index(drop=True)
        performance_group["subgroup"] = np.minimum(
            np.floor(
                np.arange(len(performance_group)) * subgroup_count / len(performance_group)
            ).astype(int),
            subgroup_count - 1,
        )
        subgroup_map = performance_group.set_index("fips")["subgroup"].to_dict()
        subgroup_by_fips.update(
            {
                str(fips).zfill(5): int(subgroup)
                for fips, subgroup in subgroup_map.items()
                if pd.notna(subgroup)
            }
        )
        line_frame = complete.loc[
            complete["riskRating"].eq(risk)
            & complete["fips"].isin(performance_group["fips"])
        ].copy()
        line_frame["subgroup"] = line_frame["fips"].map(subgroup_map)
        county_month_lines = (
            line_frame.groupby(["fips", "subgroup", "event_window_month"], as_index=False)["median_ppsf_yoy"]
            .median()
        )
        group_entries: list[dict[str, object]] = []
        for subgroup_index in sorted(performance_group["subgroup"].dropna().unique()):
            subgroup_index = int(subgroup_index)
            members = performance_group.loc[performance_group["subgroup"].eq(subgroup_index)]
            monthly = (
                county_month_lines.loc[county_month_lines["subgroup"].eq(subgroup_index)]
                .groupby("event_window_month", as_index=False)["median_ppsf_yoy"]
                .median()
                .sort_values("event_window_month")
            )
            traits = []
            for feature in distribution_features:
                subgroup_q1, subgroup_q3 = members[feature].quantile([0.25, 0.75])
                traits.append(
                    {
                        "feature": feature,
                        "median": serialize_number(members[feature].median(), 5),
                        "q1": serialize_number(subgroup_q1, 5),
                        "q3": serialize_number(subgroup_q3, 5),
                    }
                )
            group_entries.append(
                {
                    "index": subgroup_index,
                    "count": int(members["fips"].nunique()),
                    "targetMedian": serialize_number(members[FEATURE_TARGET_COLUMN].median(), 5),
                    "traits": traits,
                    "values": [
                        {
                            "month": int(row.event_window_month),
                            "value": serialize_number(row.median_ppsf_yoy, 5),
                        }
                        for row in monthly.itertuples(index=False)
                    ],
                }
            )
        subgroup_payload[risk] = {
            "features": distribution_features,
            "hasStrongFeatures": bool(strong_metrics),
            "groups": group_entries,
            "excludedOutliers": 0,
        }

    return {
        "riskOrder": RISK_ORDER,
        "minimumEffect": minimum_effect,
        "featureOrder": feature_columns,
        "featureMeta": {
            feature: {"category": category, "subcategory": subcategory, "format": fmt}
            for category, subcategory, feature, fmt in WITHIN_GROUP_FEATURES
        },
        "importanceByRisk": importance_by_risk,
        "countyRowsByRisk": county_rows_by_risk,
        "subgroupsByRisk": subgroup_payload,
        "subgroupByFips": subgroup_by_fips,
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
  <link rel="icon" href="data:,">
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
    #rating-hazard-sidebar-right { margin-top: 16px; padding-top: 4px; }
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
    .playbook-map-wrap { position: sticky; top: 0; height: 350px; overflow: hidden; align-self: start; }
    .playbook-map-wrap .chart { height: 350px; }
    .playbook-map-controls { position: absolute; top: 10px; right: 10px; z-index: 3; display: flex; gap: 6px; }
    .playbook-map-controls button { min-width: 34px; justify-content: center; box-shadow: 0 2px 8px rgba(23,32,38,.16); }
    .hazard-icon { display: inline-flex; width: 22px; justify-content: center; margin-right: 6px; }
    .hazard-rating-grid { margin: 4px 0 10px; }
    .hazard-rating-overall { margin-bottom: 4px; border-bottom: 2px solid var(--line); }
    .hazard-rating-overall .hazard-rating-item { font-size: 14px; font-weight: 800; padding: 7px 0; border-bottom: 0; }
    .hazard-rating-specific { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px 16px; }
    .hazard-rating-item { display: flex; justify-content: space-between; gap: 8px; align-items: center; padding: 5px 0; border-bottom: 1px solid var(--line); font-size: 11px; }
    .playbook-selected-county { display: none; margin: 10px 0 2px; text-align: center; color: var(--ink); font-size: 18px; font-weight: 850; }
    .playbook-history-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px 22px; min-height: 38px; margin: 8px auto 0; padding: 10px 14px; border: 1px solid var(--line); background: #f8fbf9; color: var(--ink); font-size: 12px; font-weight: 750; }
    .playbook-history-legend:empty { display: none; }
    .playbook-history-legend-item { display: inline-flex; align-items: center; gap: 8px; }
    .playbook-history-legend-swatch { width: 24px; height: 12px; border: 1px solid rgba(23,51,45,.16); }
    .playbook-commentary { margin-top: 14px; padding: 18px 20px; border: 1px solid var(--line); border-left: 5px solid var(--teal); border-radius: 0; background: #edf7f4; box-shadow: 0 10px 26px rgba(23,51,45,.10); font-size: 16px; line-height: 1.58; }
    .playbook-commentary.neutral { border-left-color: var(--water); background: #edf5f8; }
    .playbook-event-list { list-style: none; margin: 8px 0 0; padding: 0; overflow: visible; }
    .playbook-event-item { display: grid; grid-template-columns: 24px minmax(0,1fr) auto; gap: 6px; align-items: center; margin-top: 5px; padding: 7px 8px; border: 1px solid rgba(23,51,45,.12); border-left-width: 4px; border-radius: 0; background: #fff; box-shadow: 0 4px 12px rgba(23,51,45,.06); font-size: 12px; }
    .playbook-summary-detail { display: block; margin: 14px -20px -18px; padding: 14px 20px; border-top: 1px solid rgba(23,51,45,.18); background: rgba(255,255,255,.42); }
    .playbook-event-item.aligned { border-color: #83b99f; border-left-color: #16804c; background: #eaf7ef; }
    .playbook-event-item.misaligned { border-color: #d99b90; border-left-color: #b83d2f; background: #fff0ed; }
    .event-expectation { display: block; margin-top: 4px; color: var(--ink); font-size: 11px; line-height: 1.35; }
    .event-expectation strong { font-weight: 850; }
    .playbook-event-item.aligned .event-expectation strong { color: #167044; }
    .playbook-event-item.misaligned .event-expectation strong { color: #b5352b; }
    .playbook-event-icon { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; background: rgba(255,255,255,.72); border: 1px solid rgba(23,51,45,.12); border-radius: 50%; font-size: 13px; }
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
    .panel { position: relative; background: var(--panel); border: 1px solid var(--line); border-top: 4px solid var(--water); border-radius: 0; padding: 18px; box-shadow: var(--shadow); min-width: 0; }
    .panel h3, .panel p, .panel .sub, .panel .note, .panel .sources { margin-left: 0; margin-right: 0; }
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
    .county { stroke: #ffffff; stroke-width: .45; stroke-linejoin: round; stroke-linecap: round; vector-effect: non-scaling-stroke; }
    .state-boundary { fill: none; stroke: #173f37; stroke-width: 1.4; stroke-linejoin: round; stroke-linecap: round; pointer-events: none; vector-effect: non-scaling-stroke; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 12px; color: var(--muted); font-size: 12px; align-items: center; margin-top: 8px; }
    .scale-legend { width: min(100%, 320px); display: grid; grid-template-columns: 70px 1fr 70px; gap: 8px; align-items: center; }
    .scale-bar { height: 10px; border-radius: 999px; border: 1px solid rgba(23,32,38,.12); }
    .scale-legend span:last-child { text-align: right; }
    .swatch { width: 16px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 5px; vertical-align: -1px; }
    .takeaway { margin-top: 16px; border: 1px solid #b9d8ce; border-left: 5px solid var(--teal); border-radius: 0; background: #edf7f3; padding: 24px 28px; box-shadow: 0 10px 26px rgba(23,51,45,.09); font-size: 18px; line-height: 1.5; font-weight: 750; }
    .takeaway.segmented { padding: 0; }
    .takeaway-section { display: block; padding: 24px 28px; }
    .takeaway-section + .takeaway-section { border-top: 1px solid #b9d8ce; background: rgba(255,255,255,.34); }
    .sources { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); line-height: 1.5; margin-top: 14px; }
    .panel > .sources { margin-top: 12px; padding-top: 8px; border-top: 1px solid var(--line); }
    .sources a { color: #205f90; }
    .info-tooltip-trigger { display: inline-flex; align-items: center; justify-content: center; width: 17px; height: 17px; min-width: 17px; border: 1px solid currentColor; border-radius: 50%; padding: 0; background: white; color: var(--teal); font-size: 10px; font-weight: 900; line-height: 1; cursor: pointer; }
    .tooltip { position: fixed; display: none; max-width: min(300px, calc(100vw - 16px)); max-height: calc(100svh - 16px); overflow: auto; background: #172026; color: white; padding: 9px 10px; border-radius: 0; box-shadow: 0 8px 22px rgba(23,32,38,.28); font-size: 12px; line-height: 1.35; pointer-events: none; z-index: 1000; }
    .tooltip.persistent { pointer-events: auto; }
    #county-results { border-radius: 0 !important; box-shadow: 0 8px 20px rgba(23,51,45,.10); }
    .county-line-label { font-size: 10px; fill: var(--muted); pointer-events: none; }
    .feature-line-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px 18px; min-height: 18px; margin: -4px 0 6px; color: var(--ink); font-size: 11px; font-weight: 700; }
    .feature-line-legend-item { display: inline-flex; align-items: center; gap: 7px; }
    .feature-line-key { width: 28px; border-top: 3px solid; }
    #feature-risk-sidebar { flex-wrap: nowrap; align-items: center; justify-content: center; text-align: left; }
    #feature-risk-sidebar .sidebar-label { flex: 0 0 auto; width: auto; margin-right: 8px; }
    .feature-importance-chart { display: grid; gap: 4px; margin-top: 6px; align-content: start; min-height: 0; overflow-y: auto; padding-right: 4px; }
    .feature-story-grid { align-items: start; }
    .feature-line-pane { position: relative; z-index: 20; isolation: isolate; }
    .feature-line-pane.scatter-negative .feature-relationship { border-left-color: #b42318; background: #fff0ed; color: #6f2119; }
    #feature-chart-title .info-tooltip-trigger { margin: 0 5px; vertical-align: 2px; }
    .feature-plot-shell { position: relative; min-width: 0; isolation: isolate; }
    #feature-event-window { position: relative; z-index: 0; }
    .feature-detail-stack { position: relative; min-height: min(52svh, 480px); max-height: min(58svh, 520px); overflow: hidden; padding-right: 5px; }
    .feature-detail-heading { position: relative; z-index: 3; min-height: 25px; margin: 0 0 6px; }
    .feature-frame { position: absolute; inset: var(--feature-frame-top, 42px) 0 auto; width: 100%; height: calc(100% - var(--feature-frame-top, 42px)); opacity: 0; pointer-events: none; transition: opacity 320ms ease, transform 380ms ease; }
    .feature-frame[data-frame="1"] { display: flex; flex-direction: column; overflow: hidden; }
    #features .story-stage[data-story-state="feature-frame-1"] .feature-frame[data-frame="2"],
    #features .story-stage[data-story-state="feature-frame-1"] .feature-frame[data-frame="3"],
    #features .story-stage[data-story-state="feature-frame-2"] .feature-frame[data-frame="3"] { transform: translateY(42px); }
    #features .story-stage[data-story-state="feature-frame-2"] .feature-frame[data-frame="1"],
    #features .story-stage[data-story-state="feature-frame-3"] .feature-frame[data-frame="1"],
    #features .story-stage[data-story-state="feature-frame-3"] .feature-frame[data-frame="2"] { transform: translateY(-42px); }
    #features .story-stage[data-story-state="feature-frame-1"] .feature-frame[data-frame="1"],
    #features .story-stage[data-story-state="takeaway-feature"] .feature-frame[data-frame="1"],
    #features .story-stage[data-story-state="feature-frame-2"] .feature-frame[data-frame="2"],
    #features .story-stage[data-story-state="feature-frame-3"] .feature-frame[data-frame="3"] { opacity: 1; transform: translateY(0); pointer-events: auto; }
    #features .story-stage[data-story-state="feature-frame-2"] .feature-detail-heading,
    #features .story-stage[data-story-state="feature-frame-3"] .feature-detail-heading { display: none; }
    #features .story-stage[data-story-state="feature-frame-2"] .feature-frame[data-frame="2"],
    #features .story-stage[data-story-state="feature-frame-3"] .feature-frame[data-frame="3"] { inset: 0; height: 100%; }
    .feature-order-controls { display: flex; gap: 6px; margin: 2px 0 6px; }
    .feature-order-controls button { flex: 1 1 0; padding: 6px 8px; font-size: 10px; }
    .feature-click-hint { margin: 0 0 5px; color: var(--muted); font-size: 9px; }
    .importance-group-label { margin: 5px 0 1px; color: var(--muted); font-size: 9px; font-weight: 850; letter-spacing: .05em; text-transform: uppercase; }
    .importance-row { display: grid; grid-template-columns: 9px minmax(164px, 1.18fr) minmax(108px, .82fr); gap: 7px; align-items: center; width: 100%; border: 1px solid transparent; border-radius: 0; padding: 3px 4px; background: transparent; color: var(--ink); text-align: left; font-size: 10px; }
    .importance-row:hover, .importance-row.active { background: #e8f4ef; color: #17332d; }
    .importance-row.active .importance-label { color: #17332d; }
    .importance-row.active { box-shadow: inset 3px 0 0 var(--teal); }
    .importance-row.active.negative-active { background: #fff0ed; box-shadow: inset 3px 0 0 #b42318; }
    .importance-strong-group { display: grid; gap: 4px; border: 2px solid #9a6b00; padding: 3px; background: rgba(184,134,11,.045); }
    .correlation-marker { width: 8px; height: 8px; border-radius: 2px; }
    .correlation-marker.positive { background: #16803c; }
    .correlation-marker.negative { background: #b42318; }
    .importance-label { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .importance-bar-track { position: relative; height: 10px; border: 1px solid var(--line); background: #edf2ef; overflow: hidden; }
    .importance-bar { height: 100%; background: linear-gradient(90deg, #b9ddd4, #11796d); transform-origin: left center; transition: width 360ms ease; }
    .feature-footnote-tabs { display: none; gap: 18px; margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--line); color: var(--muted); font-size: 9px; font-weight: 800; }
    #features .story-stage[data-story-state="feature-frame-1"] .feature-footnote-tabs { display: flex; }
    .feature-footnote-topic { display: inline-flex; align-items: center; gap: 4px; }
    .feature-relationship { min-height: 48px; margin-top: 8px; padding: 10px 12px; border-left: 4px solid var(--teal); background: #edf7f3; color: var(--ink); font-size: 12px; line-height: 1.42; }
    .feature-distribution-controls { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 7px; align-items: center; margin: 4px 0 8px; }
    .feature-distribution-controls button { min-width: 34px; padding: 5px 8px; font-size: 12px; }
    .feature-distribution-current { min-width: 0; color: var(--ink); font-size: 17px; font-weight: 800; line-height: 1.2; overflow-wrap: anywhere; text-align: center; }
    .feature-distribution-controls.single-feature .feature-distribution-current { grid-column: 1 / -1; }
    .feature-distribution-title { display: flex; align-items: center; justify-content: center; gap: 5px; min-height: 24px; color: var(--ink); font-size: 12px; font-weight: 800; text-align: center; }
    .feature-distribution-plot { display: block; width: 100%; }
    .feature-distribution-chart { width: 100%; height: min(24svh, 210px); display: block; overflow: visible; }
    .feature-subgroup-takeaway { margin-top: 8px; padding: 10px 12px; border-left: 4px solid #7651a8; background: #f2eef8; font-size: 12px; line-height: 1.42; }
    .feature-subgroup-summary { display: flex; flex-direction: column; gap: 8px; height: 100%; min-height: 0; overflow: hidden; }
    .feature-subgroup-summary-intro { margin: 0 0 2px; color: var(--muted); font-size: 12px; line-height: 1.4; }
    .feature-subgroup-summary-rows { display: grid; align-content: start; gap: 8px; min-height: 0; overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable; padding-right: 4px; }
    .feature-subgroup-summary-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 10px 11px; border: 1px solid var(--line); background: #f8fbf9; font-size: 12px; }
    .feature-peer-relation { padding: 4px 7px; border-radius: 999px; font-size: 10px; font-weight: 850; white-space: nowrap; }
    .feature-peer-relation.higher { background: #e8f4ef; color: #166147; }
    .feature-peer-relation.lower { background: #fff0ed; color: #9b3026; }
    .feature-peer-relation.close { background: #edf2f5; color: #50636d; }
    .feature-sequence-resume { display: none; margin: 0 auto 8px; }
    .feature-sequence-resume.visible { display: inline-flex; }
    .feature-story-copy { display: grid; margin-top: 4px; }
    .feature-story-copy p { margin: 0; padding: 22px 24px; border-left: 6px solid var(--line); background: #f6f8f6; font-size: clamp(17px, 1.7vw, 23px); line-height: 1.55; }
    .subgroup-list { display: grid; gap: 8px; margin-top: 10px; }
    .subgroup-card { border: 1px solid var(--line); padding: 18px; background: #f7faf8; }
    .subgroup-card.active { border-color: var(--teal); box-shadow: inset 4px 0 0 var(--teal); background: #edf7f3; }
    .subgroup-traits { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 7px; }
    .subgroup-trait { padding: 4px 7px; background: white; border: 1px solid var(--line); font-size: 10px; }
    .subgroup-iqr { display: grid; gap: 8px; margin-top: 12px; }
    .subgroup-iqr-row { display: grid; grid-template-columns: minmax(150px, 1fr) auto; gap: 14px; align-items: center; padding: 10px 12px; background: white; border: 1px solid var(--line); font-size: 12px; }
    .subgroup-iqr-value { color: var(--ink); font-weight: 800; white-space: nowrap; }
    .feature-subgroup-control-stack { position: relative; z-index: 100; width: 146px; align-self: center; isolation: isolate; pointer-events: auto; }
    .feature-subgroup-controls { display: none; width: 100%; gap: 9px; pointer-events: auto; }
    .feature-subgroup-controls.visible { display: grid; }
    #features .story-stage[data-story-state="feature-frame-2"] .feature-plot-shell,
    #features .story-stage[data-story-state="feature-frame-3"] .feature-plot-shell { display: grid; grid-template-columns: minmax(0, 1fr) 146px; gap: 8px; align-items: center; }
    #features .story-stage[data-story-state="feature-frame-2"] .feature-subgroup-controls,
    #features .story-stage[data-story-state="feature-frame-3"] .feature-subgroup-controls { position: relative; z-index: 101; align-self: center; width: 100%; }
    .feature-subgroup-control, .feature-subgroup-control * { cursor: pointer !important; user-select: none; }
    .feature-subgroup-control { position: relative; z-index: 102; width: 100%; min-height: 30px; padding: 6px 10px 6px 31px; border: 2px solid var(--subgroup-color); border-radius: 999px; background: rgba(255,255,255,.96); color: var(--subgroup-color); text-align: left; font-size: 11px; cursor: pointer !important; pointer-events: auto; touch-action: manipulation; user-select: none; }
    .feature-subgroup-control::before { content: ""; position: absolute; left: 10px; top: 50%; width: 10px; height: 10px; border: 2px solid var(--subgroup-color); border-radius: 50%; background: white; transform: translateY(-50%); cursor: pointer; pointer-events: none; }
    .feature-subgroup-control-label { display: block; pointer-events: none; user-select: none; }
    .feature-subgroup-control.active { background: var(--subgroup-color); color: white; }
    .feature-subgroup-control.active::before { border-color: white; background: var(--subgroup-color); box-shadow: inset 0 0 0 2px var(--subgroup-color), inset 0 0 0 4px white; }
    .feature-subgroup-control:focus-visible { outline: 3px solid rgba(17,121,109,.32); outline-offset: 2px; }
    .subgroup-note { margin-top: 8px; color: var(--muted); font-size: 10px; line-height: 1.4; }
    #pricing-grouping .panel { position: relative; }
    .rating-line-pane { position: relative; display: flex; flex-direction: column; min-width: 0; }
    .sequence-callout { position: relative; z-index: 7; align-self: flex-start; width: calc(100% - 82px); min-height: 0; margin: 4px 24px 0 58px; padding: 7px 10px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(23,51,45,.18); background: rgba(255,255,255,.94); box-shadow: 0 5px 14px rgba(23,51,45,.12); color: var(--ink); font-size: 12px; line-height: 1.3; font-weight: 750; text-align: center; opacity: 0; pointer-events: none; transform: translateY(7px); transition: opacity 240ms ease, transform 240ms ease; }
    .sequence-callout.visible { opacity: 1; transform: translateY(0); }
    .event-horizon-number { display: inline-block; min-width: 1.2em; margin: 0 .08em; padding: 0 .18em; border: 2px solid #9a6b00; background: #f4d77c; color: #5b4300; text-align: center; }
    .event-horizon-number.changed { animation: eventHorizonPulse 720ms ease both; }
    @keyframes eventHorizonPulse { 0% { transform: scale(.72); background: #fff5c2; } 55% { transform: scale(1.2); background: #e9bb36; } 100% { transform: scale(1); background: #f4d77c; } }
    .percentile-comparison { display: grid; grid-template-columns: minmax(145px, .9fr) 1fr; gap: 10px 16px; align-items: center; }
    .percentile-comparison-row { display: contents; }
    .comparison-scale { position: relative; height: 24px; border: 1px solid var(--line); background: linear-gradient(90deg, #e8f4ed, #f0cf75, #e77662); }
    .comparison-marker { position: absolute; top: 2px; width: 12px; height: 12px; transform: translateX(-50%) rotate(45deg); border: 2px solid white; box-shadow: 0 0 0 1px rgba(23,51,45,.35); }
    .comparison-marker.second { top: 10px; border-radius: 50%; transform: translateX(-50%); }
    .comparison-values { display: flex; justify-content: space-between; gap: 12px; margin-top: 3px; color: var(--muted); font-size: 10px; }
    .playbook-frame-stack { position: relative; flex: 1 1 auto; min-height: 0; overflow: hidden; }
    .playbook-frame { position: absolute; inset: 0; opacity: 0; transform: translateY(48px); pointer-events: none; transition: opacity 340ms ease, transform 420ms ease; }
    .playbook-search-frame { display: flex; flex-direction: column; }
    .playbook-search-shell { position: relative; z-index: 9; flex: 0 0 auto; width: min(520px, 72%); margin: 0 auto 10px; text-align: center; }
    #county-search { width: 100%; padding: 10px 15px; border: 1px solid var(--line); border-radius: 999px; font-size: 13px; background: #fff; }
    #county-results { position: absolute; left: 0; right: 0; top: calc(100% + 4px); max-height: 200px; overflow-y: auto; background: white; text-align: left; }
    .playbook-search-map { position: relative; flex: 1 1 auto; min-height: 0; overflow: hidden; }
    .playbook-search-map .chart { height: 100%; }
    .playbook-selected-layout { display: grid; grid-template-columns: minmax(270px, .8fr) minmax(0, 1.35fr) minmax(225px, .72fr); gap: 14px; height: 100%; min-height: 0; align-items: stretch; }
    .playbook-profile-panel { min-width: 0; padding: 12px; border: 1px solid var(--line); background: #f8fbf9; overflow: hidden; }
    .playbook-profile-map-pane, .playbook-history-pane, .playbook-events-pane, .playbook-commentary-pane { min-width: 0; min-height: 0; }
    .playbook-profile-map-pane { grid-column: 2 / 4; position: relative; overflow: hidden; }
    .playbook-profile-map-pane .chart { height: 100%; }
    .playbook-history-pane { grid-column: 2; display: flex; flex-direction: column; }
    .playbook-history-pane .chart { flex: 1 1 auto; min-height: 0; height: 100%; }
    .playbook-events-pane, .playbook-commentary-pane { grid-column: 3; overflow: hidden; }
    .playbook-events-pane { display: flex; flex-direction: column; }
    .playbook-event-column { display: grid; flex: 1 1 auto; align-content: start; gap: 7px; min-height: 0; margin-top: 8px; padding-right: 6px; overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable; }
    .playbook-event-card { padding: 9px 10px; border: 1px solid var(--line); border-left: 4px solid #df7d2f; background: #fff8f1; font-size: 11px; line-height: 1.35; }
    .playbook-back-button { align-self: flex-start; margin-bottom: 8px; border-radius: 0; padding: 5px 8px; font-size: 10px; line-height: 1.15; }
    .playbook-subgroup-badge { margin: 10px 0; padding: 9px 10px; border-left: 4px solid var(--teal); background: #eaf5f1; font-size: 12px; line-height: 1.35; }
    #playbook-profile-details { min-width: 0; }
    .playbook-feature-summary { display: grid; gap: 6px; max-height: min(26svh, 210px); overflow-x: hidden; overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable; }
    .playbook-feature-row { display: grid; grid-template-columns: minmax(110px, 1fr) auto; gap: 8px; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--line); font-size: 10px; }
    .playbook-feature-insufficient { padding: 14px; border: 1px solid var(--line); background: #f5f7f5; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .feature-data-unavailable { grid-column: 2 / -1; color: var(--muted); font-style: italic; }
    .playbook-commentary { height: 100%; overflow: hidden; padding: 16px; font-size: 13px; }
    .playbook-event-item { display: block; }
    .playbook-event-item summary { display: grid; grid-template-columns: 24px minmax(0,1fr); gap: 6px; align-items: center; cursor: pointer; list-style: none; }
    .playbook-event-item summary::-webkit-details-marker { display: none; }
    .playbook-event-item[open] { box-shadow: 0 9px 22px rgba(23,51,45,.13); }
    .event-period { transition: opacity 180ms ease, filter 180ms ease; }
    .event-period.event-muted { opacity: .025 !important; }
    .event-period.event-focused { opacity: .52 !important; filter: saturate(1.35); }
    #playbook .story-stage > .panel { height: calc(100svh - 88px); display: flex; flex-direction: column; overflow: hidden; }
    .story-nav { position: fixed; inset: 0; z-index: 20; pointer-events: none; }
    .story-nav button { position: absolute; left: 50%; width: 42px; height: 34px; display: flex; align-items: center; justify-content: center; border-radius: 0; padding: 0; background: rgba(255,255,255,.96); box-shadow: 0 7px 20px rgba(23,51,45,.16); font-size: 20px; opacity: 0; pointer-events: none; transform: translateX(-50%); transition: opacity 160ms ease, transform 160ms ease; }
    #story-prev { top: 8px; transform: translate(-50%, -8px); }
    #story-next { bottom: 8px; transform: translate(-50%, 8px); }
    .story-nav button.edge-visible:not(:disabled) { opacity: 1; pointer-events: auto; transform: translate(-50%, 0); }
    html { scroll-snap-type: y mandatory; }
    .hero { min-height: 100svh; scroll-snap-align: start; }
    .slide { min-height: 0; height: calc(var(--story-steps, 3) * 100svh); padding: 0; border-bottom: 0; opacity: 1; transform: none; transition: none; scroll-snap-align: start; }
    .slide.visible, .slide.transition-in, .slide.transition-out { opacity: 1; transform: none; animation: none; }
    .story-stage { position: sticky; top: 0; z-index: 2; width: 100%; height: 100svh; padding: 28px 0; display: flex; flex-direction: column; justify-content: center; overflow: hidden; }
    .story-stage > h2 { transition: top 520ms ease, transform 520ms ease, font-size 520ms ease, opacity 360ms ease; }
    .story-stage > .section-copy, .story-stage > .panel, .story-stage > .sources { transition: opacity 360ms ease, transform 420ms ease, filter 360ms ease; }
    .story-stage > h2 { position: absolute; z-index: 5; left: 0; top: 50%; width: min(1040px, 100%); transform: translateY(-50%); }
    .story-stage > .section-copy { position: absolute; z-index: 4; left: 0; top: 52%; width: min(900px, 100%); opacity: 0; transform: translateY(28px); }
    .story-stage > .panel { position: relative; width: 100%; max-height: calc(100svh - 88px); margin-top: 72px !important; padding: 14px; overflow: hidden; opacity: 0; transform: translateY(42px); }
    .story-stage > .panel:has(> .sources) { padding-bottom: 78px; }
    .story-stage > .panel > .sources { position: absolute; z-index: 2; left: 14px; right: 14px; bottom: 14px; margin: 0; padding: 10px 0 0; background: var(--panel); }
    .story-stage > .panel.inner-scroll-locked,
    .story-stage[data-story-state^="takeaway"] > .panel,
    #playbook .panel:not(.has-county-selection) { overflow-y: hidden; }
    .story-stage > .sources { position: absolute; left: 18px; right: 18px; bottom: 5px; margin: 0; opacity: 0; }
    .story-stage[data-story-state="copy"] > h2 { top: 29%; transform: translateY(-50%) scale(.82); transform-origin: left center; }
    .story-stage[data-story-state="copy"] > .section-copy { opacity: 1; transform: translateY(0); }
    .story-stage[data-story-state^="card"] > h2,
    .story-stage[data-story-state^="comparison"] > h2,
    .story-stage[data-story-state="search"] > h2,
    .story-stage[data-story-state^="profile"] > h2,
    .story-stage[data-story-state^="history"] > h2,
    .story-stage[data-story-state^="feature-frame"] > h2,
    .story-stage[data-story-state^="takeaway"] > h2 {
      top: 20px; transform: none; font-size: clamp(20px, 2.2vw, 30px); max-width: calc(100% - 20px);
    }
    .story-stage[data-story-state^="card"] > h2::before,
    .story-stage[data-story-state^="comparison"] > h2::before,
    .story-stage[data-story-state="search"] > h2::before,
    .story-stage[data-story-state^="profile"] > h2::before,
    .story-stage[data-story-state^="history"] > h2::before,
    .story-stage[data-story-state^="feature-frame"] > h2::before,
    .story-stage[data-story-state^="takeaway"] > h2::before { width: 34px; height: 3px; margin-bottom: 6px; }
    .story-stage[data-story-state^="card"] > .panel,
    .story-stage[data-story-state^="comparison"] > .panel,
    .story-stage[data-story-state="search"] > .panel,
    .story-stage[data-story-state^="profile"] > .panel,
    .story-stage[data-story-state^="history"] > .panel,
    .story-stage[data-story-state^="feature-frame"] > .panel,
    .story-stage[data-story-state^="takeaway"] > .panel { opacity: 1; transform: translateY(0); }
    .story-stage[data-story-state^="card"] > .sources,
    .story-stage[data-story-state^="comparison"] > .sources,
    .story-stage[data-story-state="search"] > .sources,
    .story-stage[data-story-state^="profile"] > .sources,
    .story-stage[data-story-state^="history"] > .sources,
    .story-stage[data-story-state^="feature-frame"] > .sources { opacity: 1; }
    .story-stage .takeaway { opacity: 0; max-height: 0; margin: 0; padding-top: 0; padding-bottom: 0; overflow: hidden; transition: opacity 320ms ease, translate 380ms ease; }
    .story-stage > .panel > *:not(.takeaway) { transition: opacity 320ms ease, filter 320ms ease; }
    .story-stage[data-story-state^="takeaway"] > .panel { --takeaway-space: min(17svh, 132px); --takeaway-bottom: 8px; --takeaway-footnote-space: 0px; padding-bottom: calc(var(--takeaway-space) + var(--takeaway-footnote-space) + 12px) !important; }
    .story-stage[data-story-state^="takeaway"] > .panel:has(> .sources) { --takeaway-bottom: 52px; --takeaway-footnote-space: 44px; }
    .story-stage[data-story-state^="takeaway"] > .panel > *:not(.takeaway) { opacity: 1; filter: none; }
    .story-stage[data-story-state^="takeaway"] > .panel > .sources { bottom: 8px; }
    .story-stage[data-story-state^="takeaway"] .takeaway.story-active-takeaway,
    .story-stage .takeaway.story-outgoing-takeaway { position: absolute; z-index: 8; left: 8px; right: 8px; top: auto; bottom: var(--takeaway-bottom, 8px); width: auto; max-width: none; max-height: none; margin: 0; padding: 10px 14px; opacity: 1; overflow: hidden; transform: none; font-size: clamp(13px, 1.15vw, 17px); line-height: 1.3; }
    #features .story-stage[data-story-state="takeaway-feature"] > .panel { --takeaway-bottom: 40px; --takeaway-footnote-space: 32px; }
    #features .story-stage[data-story-state="takeaway-feature"] .feature-footnote-tabs { display: flex; position: absolute; z-index: 9; left: 14px; right: 14px; bottom: 8px; margin: 0; background: var(--panel); }
    .story-stage[data-story-state^="takeaway"] .takeaway.story-active-takeaway.segmented,
    .story-stage .takeaway.story-outgoing-takeaway.segmented { padding: 0; }
    .story-stage[data-story-state^="takeaway"] .takeaway.story-active-takeaway .takeaway-section { display: none; }
    .story-stage[data-story-state^="takeaway"] .takeaway.story-active-takeaway .takeaway-section.story-active-segment { display: block; border-top: 0; padding: 10px 14px; }
    .story-stage .takeaway.story-outgoing-takeaway .takeaway-section { display: none; }
    .story-stage .takeaway.story-outgoing-takeaway .takeaway-section.story-active-segment { display: block; border-top: 0; padding: 10px 14px; }
    .story-slide-out-up { animation: takeawayOutUp 420ms ease both; }
    .story-slide-in-up { animation: takeawayInUp 420ms ease both; }
    .story-slide-out-down { animation: takeawayOutDown 420ms ease both; }
    .story-slide-in-down { animation: takeawayInDown 420ms ease both; }
    @keyframes takeawayOutUp { to { opacity: 0; translate: 0 -70px; } }
    @keyframes takeawayInUp { from { opacity: 0; translate: 0 70px; } to { opacity: 1; translate: 0 0; } }
    @keyframes takeawayOutDown { to { opacity: 0; translate: 0 70px; } }
    @keyframes takeawayInDown { from { opacity: 0; translate: 0 -70px; } to { opacity: 1; translate: 0 0; } }
    #playbook .story-stage[data-story-state="search"] .playbook-search-frame,
    #playbook .story-stage[data-story-state="profile"] .playbook-selected-frame,
    #playbook .story-stage[data-story-state="history-events"] .playbook-selected-frame,
    #playbook .story-stage[data-story-state="history-compare"] .playbook-selected-frame { opacity: 1; transform: translateY(0); pointer-events: auto; }
    #playbook .story-stage[data-story-state="profile"] .playbook-selected-layout { grid-template-columns: minmax(290px, 1fr) minmax(0, 1fr); }
    #playbook .story-stage[data-story-state="profile"] .playbook-profile-map-pane { display: block; grid-column: 2; }
    #playbook .story-stage[data-story-state="profile"] .playbook-history-pane,
    #playbook .story-stage[data-story-state="profile"] .playbook-events-pane,
    #playbook .story-stage[data-story-state="profile"] .playbook-commentary-pane { display: none; }
    #playbook .story-stage[data-story-state^="history-"] .playbook-profile-map-pane { display: none; }
    #playbook .story-stage[data-story-state="history-events"] .playbook-commentary-pane { display: none; }
    #playbook .story-stage[data-story-state="history-compare"] .playbook-events-pane { display: none; }
    #playbook .story-stage[data-story-state^="history-"] .playbook-profile-panel { display: flex; flex-direction: column; min-height: 0; }
    #playbook .story-stage[data-story-state^="history-"] #playbook-selected-county-name { display: none !important; }
    #playbook .story-stage[data-story-state^="history-"] .playbook-feature-summary { flex: 1 1 auto; min-height: 0; max-height: none; }
    #playbook .story-stage[data-story-state^="history-"] .playbook-subgroup-badge { flex: 0 0 auto; margin: auto 0 0; }
    .story-stage .chart.rating-risk-line { height: min(48svh, 470px); }
    .story-stage .chart.map-companion-line { height: min(43svh, 405px); }
    .story-stage .chart.map-companion-map { height: min(38svh, 340px); }
    .story-stage #score-scatter { height: min(41svh, 320px) !important; }
    .story-stage[data-story-state^="takeaway"] .chart.rating-risk-line { height: min(48svh, 470px, max(190px, calc(100svh - var(--takeaway-space) - 265px))); }
    .story-stage[data-story-state^="takeaway"] .chart.map-companion-line { height: min(43svh, 405px, max(180px, calc(100svh - var(--takeaway-space) - 275px))); }
    .story-stage[data-story-state^="takeaway"] .chart.map-companion-map { height: min(38svh, 340px, max(170px, calc(100svh - var(--takeaway-space) - 300px))); }
    .story-stage[data-story-state^="takeaway"] #score-scatter { height: min(41svh, 320px, max(180px, calc(100svh - var(--takeaway-space) - 245px))) !important; }
    #features .story-stage[data-story-state^="takeaway"] .feature-detail-stack { min-height: min(52svh, 480px, max(190px, calc(100svh - var(--takeaway-space) - 260px))); max-height: min(52svh, 480px, max(190px, calc(100svh - var(--takeaway-space) - 260px))); }
    #features .feature-line-pane.scatter-active #feature-event-window { height: min(40svh, 360px); }
    #features .story-stage[data-story-state="feature-frame-2"] #feature-event-window,
    #features .story-stage[data-story-state="feature-frame-3"] #feature-event-window { height: min(47svh, 435px); }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      .story-stage > *, .story-stage .takeaway { transition-duration: 1ms !important; }
    }
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
      .playbook-selected-layout { grid-template-columns: minmax(220px, .8fr) minmax(0, 1.2fr) minmax(190px, .7fr); gap: 8px; }
      .importance-row { grid-template-columns: 9px minmax(125px, 1fr) minmax(90px, 1fr); }
      .story-stage { padding: 18px 0; }
      .story-stage > .panel { margin-top: 82px !important; max-height: calc(100svh - 96px); }
      .story-stage > .panel:has(> .sources) { padding-bottom: 122px; }
      .story-stage[data-story-state^="takeaway"] .takeaway.story-active-takeaway,
      .story-stage .takeaway.story-outgoing-takeaway { width: auto; margin: 0; font-size: 16px; }
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
      <div class="takeaway" id="score-scatter-takeaway"></div>
    </div>
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
        <div class="rating-line-pane">
          <svg id="rating-scatter" class="chart map-companion-line rating-risk-line"></svg>
          <div id="rating-sequence-callout" class="sequence-callout"></div>
        </div>
        <div class="map-with-legend">
          <svg id="rating-map" class="chart map-companion-map"></svg>
          <div class="legend" id="rating-map-legend"></div>
        </div>
      </div>
      <div class="takeaway" id="pricing-takeaway"></div>
      <div class="sources" id="t-pricing-sources"></div>
    </div>
  </section>

  <section class="slide" id="events">
    <h2 id="t-events-h2"></h2>
    <p class="section-copy" id="t-events-copy"></p>

    <!-- Event window + affected map in one card, risk toggle on right, arrow nav for windows -->
    <div class="panel" style="margin-top:12px;">
      <h3 id="t-events-card-title"></h3>
      <p class="sub" id="events-window-subtitle"></p>
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
        </div>
        <div>
          <svg id="affected-map" class="chart map-companion-map"></svg>
          <div class="county-count" id="affected-county-count"></div>
        </div>
      </div>
      <div class="takeaway" id="event-window-takeaway"></div>
      <div class="takeaway" id="event-future-prompt"></div>
      <div class="takeaway" id="event-takeaway"></div>
      <div class="sources" id="t-events-sources"></div>
    </div>
  </section>

  <section class="slide" id="features">
    <h2 id="t-features-h2"></h2>
    <p class="section-copy" id="t-features-copy"></p>

    <!-- Fixed analysis chart on the left; scroll-driven feature frames on the right. -->
    <div class="panel" style="margin-top:12px;">
      <div class="control-bar" id="feature-risk-sidebar">
        <div class="sidebar-label" id="t-feature-sidebar-label"></div>
      </div>
      <div class="viz-grid timeseries-grid feature-story-grid">
        <div class="feature-line-pane">
          <h3 id="feature-chart-title"></h3>
          <div class="feature-plot-shell">
            <svg id="feature-event-window" class="chart map-companion-line"></svg>
            <div class="feature-subgroup-control-stack">
              <button id="feature-sequence-resume" class="feature-sequence-resume" type="button"></button>
              <div id="feature-subgroup-toggles" class="feature-subgroup-controls" aria-label="Feature subgroup lines"></div>
            </div>
          </div>
          <div id="feature-line-legend" class="feature-line-legend"></div>
          <div id="feature-relationship" class="feature-relationship" hidden></div>
        </div>
        <div class="feature-detail-stack">
          <h3 id="feature-detail-title" class="feature-detail-heading"></h3>
          <div class="feature-frame" data-frame="1">
            <div id="feature-order-controls" class="feature-order-controls" aria-label="Feature ordering">
              <button type="button" data-order="category"></button>
              <button type="button" data-order="significance"></button>
            </div>
            <p id="feature-click-hint" class="feature-click-hint"></p>
            <div id="feature-importance-chart" class="feature-importance-chart"></div>
          </div>
          <div class="feature-frame" data-frame="2">
            <div id="feature-distribution-controls" class="feature-distribution-controls"></div>
            <div id="feature-distribution-title" class="feature-distribution-title"></div>
            <div class="feature-distribution-plot">
              <svg id="feature-distribution-chart" class="feature-distribution-chart"></svg>
            </div>
            <div id="feature-subgroup-takeaway" class="feature-subgroup-takeaway"></div>
          </div>
          <div class="feature-frame" data-frame="3">
            <div id="feature-subgroup-summary" class="feature-subgroup-summary"></div>
          </div>
        </div>
      </div>
      <div id="feature-footnote-tabs" class="feature-footnote-tabs"></div>
      <div class="takeaway" id="feature-takeaway"></div>
    </div>
  </section>

  <section class="slide" id="playbook">
    <h2 id="t-playbook-h2"></h2>

    <!-- Four-frame county search, risk profile, history, and comparison playbook. -->
    <div class="panel" style="margin-top:12px;">
      <div class="playbook-frame-stack">
        <div class="playbook-frame playbook-search-frame">
          <div class="playbook-search-shell">
            <input type="text" id="county-search">
            <div id="county-results" style="display:none;"></div>
          </div>
          <div class="playbook-search-map">
            <svg id="county-selection-map" class="chart"></svg>
            <div class="playbook-map-controls">
              <button id="playbook-map-zoom-in" type="button" title="Zoom in" aria-label="Zoom in">+</button>
              <button id="playbook-map-zoom-minus" type="button" title="Zoom out" aria-label="Zoom out">&#8722;</button>
              <button id="playbook-map-zoom-toggle" type="button" style="display:none;"></button>
            </div>
          </div>
        </div>

        <div class="playbook-frame playbook-selected-frame">
          <div class="playbook-selected-layout">
            <aside class="playbook-profile-panel" id="playbook-profile-details">
              <button id="playbook-back-to-search" class="playbook-back-button" type="button">&#8592; Back to county search</button>
              <div class="playbook-selected-county" id="playbook-selected-county-name"></div>
              <div class="hazard-rating-grid" id="playbook-hazard-ratings"></div>
              <h3 id="playbook-feature-title"></h3>
              <div id="playbook-feature-summary" class="playbook-feature-summary"></div>
              <div id="playbook-subgroup-summary" class="playbook-subgroup-badge"></div>
            </aside>
            <div class="playbook-profile-map-pane">
              <svg id="playbook-profile-map" class="chart"></svg>
            </div>
            <div class="playbook-history-pane">
              <h3 id="t-playbook-history-title"></h3>
              <svg id="playbook-ppsf-history" class="chart"></svg>
              <div class="playbook-history-legend" id="playbook-history-legend"></div>
            </div>
            <aside class="playbook-events-pane">
              <h3 id="playbook-events-title"></h3>
              <div id="playbook-event-column" class="playbook-event-column"></div>
            </aside>
            <aside class="playbook-commentary-pane">
              <div class="playbook-commentary" id="playbook-event-commentary"></div>
            </aside>
          </div>
        </div>
      </div>
      <div class="sources" id="t-playbook-sources"></div>
    </div>
  </section>
</main>
<nav class="story-nav" aria-label="Section navigation">
  <button id="story-prev" type="button">&#8593;</button>
  <button id="story-next" type="button">&#8595;</button>
</nav>
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
  storyPrevLabel: "Prev",
  storyNextLabel: "Next",
  sourcesLabel: "Sources",
  informationTooltipLabel: "More information",

  // ---- Pricing section ----
  pricingH2: "To Begin: What Does Growth in Housing Markets Look Like Across the United States?",
  scatterTitle: "Median Price-Per-Square-Foot (PPSF) Year-Over-Year (YoY) by County",
  scatterSub: "Each line represents a county's monthly Median PPSF YoY over the latest 10 complete calendar years.",
  scatterFootnotesTooltip: "Values beyond the 10th–90th percentile are capped to keep extreme observations from compressing the visible pattern. Only counties with a valid observation in every month of the latest 10 complete calendar years are included.",
  pricingScoreScatterTakeaway: "<span class=\"takeaway-section\">From county-level median house price growth over the last 10 years, there is significant variation and there doesn't seem to be a clear pattern.</span><span class=\"takeaway-section\">However, the impact of climate change is uneven across the country, so looking from a climate angle might reveal a more meaningful pattern.</span>",
  pricingGroupingSubtitle: "A Climate Perspective: What Happens When Counties are Grouped by Climate Risk?",
  pricingNriPlaceholder: "The <a href='https://www.fema.gov/flood-maps/products-tools/national-risk-index' target='_blank' rel='noopener'>FEMA National Risk Index (NRI)</a> serves as a measure of climate-related risk exposure. It summarizes a county's expected annual loss, social vulnerability, and community resilience across natural hazards. Counties are assigned a risk rating along a scale from \"Very Low\" to \"Very High\".",
  pricingCardTitle: "Median PPSF YoY by Climate Risk",
  pricingCardText: "House price growth of counties grouped by their FEMA National Risk Index (NRI) risk rating. Risk ratings are available for specific hazards and for overall climate risk.",
  hazardSidebarLabel: "Hazard type",
  ratingSequenceCallouts: {
    "Very Low": "Very Low-risk counties establish the baseline housing-growth trajectory.",
    Low: "Low-risk counties add a slightly softer housing-growth path than the Very Low group.",
    Medium: "Medium-risk counties generally sit below the lower-risk trajectories.",
    High: "High-risk counties deepen the pattern of weaker median PPSF growth.",
    "Very High": "Very High-risk counties show the lowest housing-growth trajectory in the sequence.",
    All: "Together, the five layers reveal a broad decline in housing-price growth as climate risk rises.",
  },
  pricingTakeaway: "<span class=\"takeaway-section\">Looking at the bands pertaining to different risk levels, a pattern now emerges: Counties with higher risk tend to show lower levels of house price growth.</span><span class=\"takeaway-section\">Housing markets are influenced by events across time. Does this apply to extreme climate events?</span>",
  pricingSources: 'Sources: <a href="https://hazards.fema.gov/nri/" target="_blank" rel="noopener">FEMA National Risk Index</a>, local mart <code>data/quoll.duckdb: mart.nri_county_risk</code>. Housing data provided by <a href="https://www.redfin.com/news/data-center/downloads/" target="_blank" rel="noopener">Redfin, a national real estate brokerage</a>; see Redfin\'s <a href="https://www.redfin.com/news/data-center/methodology/" target="_blank" rel="noopener">Data Center methodology</a>. Local housing mart: <code>mart.redfin_county_monthly</code>. The charts use monthly <code>MEDIAN_PPSF_YOY</code> observations from the latest 10 complete calendar years and include only counties with complete data throughout that period.',

  // ---- Events section ----
  eventsH2: "What Do Housing Market Reactions to Extreme Climate Events Look Like?",
  eventsCopy: "Taking <a href='https://www.fema.gov/disaster/declarations' target='_blank' rel='noopener'>FEMA disaster declarations</a> and <a href='https://www.ncei.noaa.gov/stormevents/' target='_blank' rel='noopener'>NOAA storm events</a> that cost at least 1 billion dollars from the last 10 years as a reference point, let's examine the state of housing markets around the time of these events.",
  eventsCardTitle: "Median PPSF YoY Around Extreme Climate Events",
  eventsShortTitle: "Median PPSF YoY in 3 years after event",
  eventsShortSubtitle: "Within the short term:",
  eventsLongTitle: "Median PPSF YoY in 5 years after event",
  eventsLongSubtitle: "Within the longer term:",
  eventHorizonYears: {A: "3", B: "5"},
  riskSidebarLabel: "Risk rating",
  eventWindowATakeaway: "Past the 2-year mark post-event, house price growth momentum diverges across the different risk bands. Growth weakening is more pronounced in higher risk groups.",
  eventFuturePrompt: "What does it look like further into the future?",
  eventWindowBTakeaway: "Around the 4-year mark post-event, house price growth across the different risk bands begin to converge to the same level. It appears that the event’s impact fades from view eventually.",
  eventsTakeaway: "<span class=\"takeaway-section\">In higher-risk counties, there is a time lag after an event before house price growth declines significantly. Homeowners who made it through the period of weakness then experienced some subsequent recovery.</span><span class=\"takeaway-section\">The risk bands have significant width, indicating that counties' housing markets are hardly uniform, even within the same risk category. What is behind this variation?</span>",
  eventsSources: "Sources: local marts <code>mart.fema_disaster_declarations</code>, <code>mart.noaa_storm_events</code>, <code>mart.redfin_county_monthly</code>, and <code>mart.nri_county_risk</code>. Housing data provided by Redfin; see the linked Download Hub and methodology in the first housing-data source note.",

  // ---- Features section ----
  featuresH2: "What Factors Influence a County's Housing Market Performance within Risk Groups?",
  featuresCopy: "From the vast data on counties, a model to understand counties' housing market <span id=\"feature-performance-term\">performance</span> can be built upon the most significant data types.",
  featurePerformanceTooltip: "Performance is defined in terms of different Median PPSF YoY over the same time period.",
  featureSidebarLabel: "Risk rating",
  featureLineTitle: "Median PPSF YoY around events",
  featureScatterTitle: "Median PPSF YoY around events vs. {feature}",
  featureOutcomeTerm: "Median PPSF YoY around events",
  featureOutcomeTooltip: "This value is each county's average monthly Median PPSF YoY across its complete event windows, from month -12 through the event start and months 1–36 after the event end. Counties with multiple qualifying events contribute one observation.",
  featureFrame1Title: "Which data types matter most to {risk} Risk counties?",
  featureFrame2Title: "What types of counties exist within the {risk} Risk group?",
  featureFrame3Title: "What features define {subgroup} in the {risk} Risk group?",
  featureGroupByCategory: "Group by Category",
  featureOrderBySignificance: "Order by Significance",
  featureClickHint: "Select any feature to reveal its relationship with Median PPSF YoY around events.",
  featureStrongTooltip: "Strongest correlation: |ρ| ≥ 0.30",
  featureSourcesTopic: "Sources",
  featureRankingTopic: "Ranking",
  featureSourcesNote: "DuckDB feature-layer tables feature.county_economic_annual, feature.county_demographic_annual, and feature.county_risk. Economic and demographic features use ten-year county averages.",
  featureRankingNote: "Data types are ranked by descending absolute Spearman correlation (|ρ|). A correlation is significant if its bootstrapped 95% confidence intervals meet the minimum-effect threshold of |ρ| ≥ {threshold}.",
  featureCategories: {
    Economic: "Economic Data",
    Demographic: "Demographic Data",
  },
  featureLabels: {
    net_earnings_per_capita_usd: "Net Earnings per Resident (Place of Residence)",
    dividends_interest_rent_per_capita_usd: "Dividends, Interest & Rent per Resident",
    transfer_receipts_per_capita_usd: "Transfer Receipts per Resident",
    homeowners_insurance_pct_income: "Home Insurance as % of Median Household Income",
    property_taxes_pct_income: "Property Tax as % of Median Household Income",
    utilities_pct_income: "Utilities Cost as % of Median Household Income",
    owner_cost_burden_30pct_plus_pct: "Share of Cost-Burdened Households",
    unemployment_rate_pct: "Unemployment Rate",
    net_migration_rate_pct: "Net Migration Rate",
    age_65_plus_pct: "Aged Population Share",
    communication_barrier_pct: "Share of Population with Language Barrier",
    disability_pct: "Share of Population with Disabled Status",
  },
  featureRelationship: "{feature} has a {strength} {direction} relationship with Median PPSF YoY around events.",
  featureRelationshipStrength: {weak: "weak", moderate: "moderate", strong: "strong"},
  featureRelationshipDirection: {positive: "positive", negative: "negative"},
  featureTakeaway: "A location's NRI Risk Rating by itself doesn't determine everything. Many different attributes can also influence house price growth.",
  subgroupNamesFour: ["Strong Overperformers", "Mild Overperformers", "Mild Underperformers", "Strong Underperformers"],
  subgroupNamesThree: ["Overperformers", "Average Performers", "Underperformers"],
  subgroupFallback: "Subgroup {number}",
  subgroupCount: "{count} counties",
  featureDistributionFallback: "No feature reaches |ρ| ≥ 0.30 for this risk group; the strongest available feature is shown.",
  featureDistributionTitle: "County Distribution",
  featureDistributionOutlierTooltip: "Outlier values beyond 1.5 times the interquartile range are not shown in this plot.",
  featureDistributionVeryHighTooltip: "All available values are shown for the Very High Risk group because this group has relatively few counties.",
  featureDistributionPrevious: "Previous feature",
  featureDistributionNext: "Next feature",
  featureDistributionSelected: "Selected subgroup",
  featureDistributionLevel: {higher: "higher", lower: "lower"},
  featureResumeSequence: "Resume",
  featureSubgroupTakeaway: "Counties with {level} {feature} values are more likely to be {subgroup} in the {risk} Risk group.",
  featureSubgroupSummaryIntro: "Compared with other {risk} Risk counties, counties in {subgroup} tend to have:",
  featurePeerRelation: {
    higher: "above average",
    lower: "below average",
    close: "average",
  },
  featureSubgroupSummaryUnavailable: "A feature summary is not available for this subgroup.",
  featureGroupMedianLegend: "{risk} group median",
  featureEventMarker: "Event",
  featureXAxis: "Months from event",
  featureYAxis: "Median PPSF YoY",
  featureScatterYAxis: "Median PPSF YoY around events",

  // ---- Playbook section ----
  playbookH2: "Climate Playbook: What to Know About Your County's Climate Exposure",
  playbookSearchPlaceholder: "Search for a county by name, state, or FIPS…",
  playbookInsufficientFeatureData: "Insufficient feature data available for {county}.",
  playbookInsufficientEventWindowData: "The most important data types for {county} could not be determined because there were insufficient housing data to form a complete event window for analysis.",
  playbookInsufficientFeatureValue: "Insufficient data",
  playbookHistoryTitle: "Monthly Median PPSF YoY Over the Past 10 Years",
  playbookFeatureTitle: "Most important data types for {risk} Risk counties",
  playbookSubgroupFeatureTitle: "County Traits",
  playbookSubgroup: "{county}'s house price growth rate around extreme climate events makes it a {subgroup} among {risk} Risk counties.",
  playbookPastEventsTitle: "Past extreme weather events",
  playbookNoPastEvents: "No qualifying extreme weather events occurred during this 10-year period.",
  playbookZoomOut: "Zoom out",
  playbookZoomIn: "Zoom to county",
  playbookEventLegend: "Extreme event period",
  playbookMissingDataLegend: "Missing county data",
  playbookSeriesLegend: "County Median PPSF YoY",
  playbookRiskSeriesLegend: "{risk} Risk median and IQR",
  playbookTakeaways: {
    noEvents: "<strong>{county} had no extreme climate events in the last 10 years.</strong><span class=\"playbook-summary-detail\">Given its profile as {countyContext}, its Median PPSF YoY would be expected to {expectation} after an extreme climate event.</span>",
    eventAlignmentSummary: "<strong>{county}'s post-event change {riskAlignment} with the {risk} Risk expectation.</strong><span class=\"playbook-summary-detail\">The county {countyChange}; its risk group typically {groupChange}.<br>Its Median PPSF YoY level relative to its risk group {subgroupAlignment} with the {subgroup} within the {risk} Risk category.<br>The percentage-point change covers one year before each event began through three years after it ended.</span>",
    eventAlignmentWithoutSubgroup: "<strong>{county}'s post-event change {riskAlignment} with the {risk} Risk expectation.</strong><span class=\"playbook-summary-detail\">The county {countyChange}; its risk group typically {groupChange}.<br>Its relative performance within its risk group could not be assessed because sufficient feature data were unavailable.<br>The percentage-point change covers one year before each event began through three years after it ended.</span>",
    volatileEventSummary: "<strong>{county}'s Median PPSF YoY was too volatile to assess whether its post-event change aligned with the {risk} Risk expectation.</strong><span class=\"playbook-summary-detail\">Its level relative to its risk group also could not be compared meaningfully with the {subgroupContext}.</span>",
    insufficientHistory: "{county} had insufficient housing data from one year before its events through three years after they ended to compare with its risk group.",
    groupExpectation: "By year 3, counties with {risk} climate risk typically {groupBehavior} versus their pre-event-year level.",
    groupBehaviorChange: "{direction} by about {magnitude} percentage points",
    groupBehaviorFlat: "remain broadly steady ({magnitude} percentage-point change)",
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
  playbookSources: "Sources: FEMA National Risk Index and local mart <code>mart.nri_county_risk</code>; housing data provided by <a href=\"https://www.redfin.com/news/data-center/downloads/\" target=\"_blank\" rel=\"noopener\">Redfin, a national real estate brokerage</a>, with definitions in Redfin's <a href=\"https://www.redfin.com/news/data-center/methodology/\" target=\"_blank\" rel=\"noopener\">methodology</a>, and local mart <code>mart.redfin_county_monthly</code>; local marts <code>mart.fema_disaster_declarations</code> and <code>mart.noaa_storm_events</code>.",
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
    eventFuturePrompt: "event-future-prompt",
    eventsTakeaway: "event-takeaway",
    eventsSources: "t-events-sources",
    featuresH2: "t-features-h2",
    featuresCopy: "t-features-copy",
    featureSidebarLabel: "t-feature-sidebar-label",
    featureTakeaway: "feature-takeaway",
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
  document.getElementById("story-prev").setAttribute("aria-label", TEXT.storyPrevLabel);
  document.getElementById("story-prev").title = TEXT.storyPrevLabel;
  document.getElementById("story-next").setAttribute("aria-label", TEXT.storyNextLabel);
  document.getElementById("story-next").title = TEXT.storyNextLabel;
  const scatterSub = document.getElementById("t-scatter-sub");
  scatterSub?.append(" ", makeInfoButton(TEXT.scatterFootnotesTooltip, {label: TEXT.informationTooltipLabel}));
  const performanceTerm = document.getElementById("feature-performance-term");
  performanceTerm?.after(makeInfoButton(TEXT.featurePerformanceTooltip, {label: TEXT.informationTooltipLabel}));
  condenseSourceDisclosures();
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
const RATING_SEQUENCE_FRAMES = [
  ...RISK_ORDER.flatMap(risk => [
    {risk, callout: false},
    {risk, callout: true},
  ]),
  {risk: null, callout: false, key: "All"},
  {risk: null, callout: true, key: "All"},
];
const RISK_COLORS = {"Very Low":"#16803c","Low":"#79b851","Medium":"#e0b33b","High":"#df7d2f","Very High":"#b42318"};
const COUNTY_LINE_COLOR = "#2456a6";
const FEATURE_HIGHER_LINE_COLOR = "#175d8f";
const FEATURE_LOWER_LINE_COLOR = "#8a4f7d";
const fmtPct = d3.format("+.1%");
const fmtShare = d3.format(".0%");
const fmtAxisPct = value => Math.abs(value) >= 10 ? `${value > 0 ? "+" : ""}${d3.format(".2s")(value * 100)}%` : fmtPct(value);
const fmtNum = d3.format(",.1f");
const fmtMoney = d3.format("$,.0f");
const parsePriceMonth = d3.utcParse("%Y-%m-%d");
const tooltip = d3.select("#tooltip");
let activeInfoTooltipTrigger = null;

function hideTooltip(force = false) {
  if (activeInfoTooltipTrigger && !force) return;
  tooltip.style("display", "none").style("visibility", "hidden").classed("persistent", false);
}

function closeInfoTooltip() {
  if (activeInfoTooltipTrigger) activeInfoTooltipTrigger.setAttribute("aria-expanded", "false");
  activeInfoTooltipTrigger = null;
  hideTooltip(true);
}

function showTooltip(event, content, {html = true, anchor = null} = {}) {
  if (activeInfoTooltipTrigger && anchor !== activeInfoTooltipTrigger) return;
  if (html) tooltip.html(content); else tooltip.text(content);
  tooltip.style("display", "block").style("visibility", "hidden");
  const anchorRect = anchor?.getBoundingClientRect();
  const originX = Number.isFinite(event?.clientX) ? event.clientX : (anchorRect?.right || 8);
  const originY = Number.isFinite(event?.clientY) ? event.clientY : (anchorRect?.top || 8);
  const node = tooltip.node();
  const width = node.offsetWidth, height = node.offsetHeight;
  const gap = 12, edge = 8;
  let left = originX + gap;
  let top = originY + gap;
  if (left + width > window.innerWidth - edge) left = originX - width - gap;
  if (top + height > window.innerHeight - edge) top = originY - height - gap;
  left = Math.max(edge, Math.min(left, window.innerWidth - width - edge));
  top = Math.max(edge, Math.min(top, window.innerHeight - height - edge));
  tooltip.style("left", `${left}px`).style("top", `${top}px`).style("visibility", "visible");
}

function attachInfoTooltip(element, content, {html = false} = {}) {
  element.setAttribute("aria-expanded", "false");
  element.addEventListener("click", event => {
    event.preventDefault();
    event.stopPropagation();
    if (activeInfoTooltipTrigger === element) {
      closeInfoTooltip();
      return;
    }
    closeInfoTooltip();
    activeInfoTooltipTrigger = element;
    element.setAttribute("aria-expanded", "true");
    tooltip.classed("persistent", true);
    showTooltip(event, content, {html, anchor: element});
  });
}

document.addEventListener("pointerdown", event => {
  if (!activeInfoTooltipTrigger) return;
  if (event.target === activeInfoTooltipTrigger || tooltip.node().contains(event.target)) return;
  closeInfoTooltip();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") closeInfoTooltip();
});

function makeInfoButton(content, options = {}) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "info-tooltip-trigger";
  button.textContent = "?";
  button.setAttribute("aria-label", options.label || TEXT.informationTooltipLabel);
  attachInfoTooltip(button, content, options);
  return button;
}

function condenseSourceDisclosures() {
  document.querySelectorAll(".panel > .sources").forEach(source => {
    const content = source.innerHTML.replace(/^\s*Sources:\s*/i, "");
    source.innerHTML = "";
    const label = document.createElement("span");
    label.textContent = TEXT.sourcesLabel;
    source.append(label, makeInfoButton(content, {html: true, label: `${TEXT.sourcesLabel}: ${TEXT.informationTooltipLabel}`}));
  });
}
const countyByFips = new Map(DATA.priceRisk.counties.map(d => [d.fips, d]));
let playbookCountyByFips = new Map();
let scoreHistoryData = null;
const deferredDataPromises = new Map();
let ratingHazard = "overall";
let ratingSequenceIndex = 0;
let ratingSequenceTimer = null;
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
let selectedFeatureKey = null;
let selectedFeatureSubgroup = null;
let featureOrderMode = "category";
let selectedDistributionFeature = null;
let featureSubgroupSequenceTimer = null;
let featureSubgroupSequencePaused = false;
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

function drawStateBoundaries(target, path) {
  if (!DATA.stateGeojson?.features?.length) return;
  target.append("g").attr("class", "state-boundaries")
    .selectAll("path")
    .data(DATA.stateGeojson.features)
    .join("path")
    .attr("class", "state-boundary")
    .attr("d", path);
}

function drawMap(svgId, fillFn, tooltipFn, legendId, legendHtml, clickFn) {
  const svg = d3.select(svgId);
  const width = svg.node().clientWidth || 520;
  const height = svg.node().clientHeight || 430;
  svg.attr("viewBox", [0,0,width,height]).selectAll("*").remove();
  const projection = d3.geoAlbersUsa().fitSize([width, height], DATA.geojson);
  const path = d3.geoPath(projection);
  svg.on("mouseleave", () => hideTooltip());
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
        hideTooltip();
        return;
      }
      showTooltip(event, html);
    })
    .on("click", clickFn ? (event, d) => clickFn(d.properties.fips) : null);
  drawStateBoundaries(svg, path);
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
  const p10 = DATA.priceRisk.summary.historyPpsfCapLower;
  const p90 = DATA.priceRisk.summary.historyPpsfCapUpper;
  const clipDomain = [p10, p90];
  const cappedSpan = Math.max(p90 - p10, 0.01);
  const yDomain = [p10 - cappedSpan * 0.08, p90 + cappedSpan * 0.08];
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
  scoreScatterState = {container, series, months, x, y, clipDomain, margin, width, height, renderedCount: 0};
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
        const py = y(capValue(value, clipDomain));
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
      const {series, months, x, y, clipDomain, margin, width, height} = scoreScatterState;
      const rect = container.getBoundingClientRect();
      const px = clientX - rect.left, py = clientY - rect.top;
      if (px < margin.left || px > width - margin.right || py < margin.top || py > height - margin.bottom) {
        hideTooltip();
        return;
      }
      const monthIndex = d3.bisector(d => d).center(months, x.invert(px));
      let nearest = null, nearestDistance = Infinity;
      for (const county of series.slice(0, scoreScatterState.renderedCount)) {
        const value = county.values[monthIndex];
        if (value == null || !Number.isFinite(value)) continue;
        const distance = Math.abs(y(capValue(value, clipDomain)) - py);
        if (distance < nearestDistance) {
          nearest = county;
          nearestDistance = distance;
        }
      }
      if (!nearest || nearestDistance > 10) {
        hideTooltip();
        return;
      }
      showTooltip({clientX, clientY}, `<strong>${countyDisplayName(nearest)}</strong>`);
    });
  });
  container.addEventListener("mouseleave", () => hideTooltip());
}

function currentRatingSequenceFrame() {
  return RATING_SEQUENCE_FRAMES[
    Math.max(0, Math.min(ratingSequenceIndex, RATING_SEQUENCE_FRAMES.length - 1))
  ];
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
  const activeSequenceRisk = currentRatingSequenceFrame().risk;
  const activeRiskIndex = activeSequenceRisk == null
    ? RISK_ORDER.length
    : RISK_ORDER.indexOf(activeSequenceRisk);
  for (const risk of RISK_ORDER) {
    const riskIndex = RISK_ORDER.indexOf(risk);
    if (activeSequenceRisk && riskIndex > activeRiskIndex) continue;
    const rows = data.filter(d => d.riskRating === risk).sort((a,b)=>d3.ascending(a.monthDate,b.monthDate));
    if (!rows.length) continue;
    const background = activeSequenceRisk && riskIndex < activeRiskIndex;
    svg.append("path").datum(rows).attr("class",`band ${background ? "background" : ""}`).attr("fill",RISK_COLORS[risk]).attr("d",area);
    svg.append("path").datum(rows).attr("class",`line ${background ? "background" : ""}`).attr("stroke",RISK_COLORS[risk]).attr("d",line);
    const last = rows.at(-1);
    if (!background) endLabels.push({risk, last, targetY: y(capValue(last.median, yDomain))});
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
  const activeRisk = currentRatingSequenceFrame().risk;
  drawMap("#rating-map",
    (county, fips) => {
      if (!county) return "#ece7df";
      const rating = hazardCounty(county, ratingHazard).rating;
      if (activeRisk && rating !== activeRisk) return "#e6e8e5";
      return RISK_COLORS[rating] || "#ece7df";
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

function renderRatingSequence() {
  drawRatingScatter();
  drawRatingMap();
  const frame = currentRatingSequenceFrame();
  const key = frame.key || frame.risk;
  const callout = TEXT.ratingSequenceCallouts[key];
  d3.select("#rating-sequence-callout")
    .classed("visible", Boolean(callout) && frame.callout)
    .text(callout || "");
}

function pauseRatingSequence() {
  clearInterval(ratingSequenceTimer);
  ratingSequenceTimer = null;
}

function startRatingSequence() {
  pauseRatingSequence();
  ratingSequenceIndex = 0;
  renderRatingSequence();
  ratingSequenceTimer = setInterval(() => {
    ratingSequenceIndex += 1;
    renderRatingSequence();
    if (ratingSequenceIndex >= RATING_SEQUENCE_FRAMES.length - 1) {
      pauseRatingSequence();
    }
  }, 2000);
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
  const valueExtent = d3.extent(values);
  const valueSpan = Math.max((valueExtent[1] || 0) - (valueExtent[0] || 0), 0.01);
  const yDomain = [
    valueExtent[0] - valueSpan * (opts.lowerDomainPadding || 0),
    valueExtent[1] + valueSpan * (opts.upperDomainPadding || 0),
  ];
  const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class","grid").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
  const useYears = horizonLimit > 12;
  const yearTicks = [minMonth, 0, ...d3.range(12, horizonLimit + 1, 12)];
  const axis = useYears
    ? d3.axisBottom(x).tickValues(yearTicks).tickFormat(d => d < 0 ? `${Math.abs(d / 12)}y pre` : d === 0 ? (opts.eventLabel || "event") : `${d / 12}y`)
    : d3.axisBottom(x).ticks(8);
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${height-margin.bottom})`).call(axis);
  svg.append("g").attr("class","axis").attr("transform",`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  if (opts.yAxisLabel) svg.append("text").attr("transform","rotate(-90)").attr("x",-height/2).attr("y",14).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",11).text(opts.yAxisLabel);
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
    if (last && !isBackground && !opts.hideEndLabel) svg.append("text").attr("x",x(last.month)+5).attr("y",y(last.median)+4).attr("fill",color).attr("font-size",12).attr("font-weight",800).text(key);
  }
  svg.append("text").attr("x",width/2).attr("y",height-8).attr("text-anchor","middle").attr("fill","#66717b").attr("font-size",12).text(opts.xAxisLabel || (useYears ? "Years from event start / after event end" : "Months from event start / after event end"));
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
      startRatingSequence();
    });

  // Event section: arrow nav for windows

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

  // Features risk group toggle (centered above the card)
  const featureSidebar = d3.select("#feature-risk-sidebar");
  featureSidebar.selectAll("button")
    .data(RISK_ORDER).join("button")
    .text(d=>d)
    .style("border-left", d=>`4px solid ${RISK_COLORS[d]}`)
    .classed("active", d=>d===selectedFeatureRisk)
    .style("background", d=>d===selectedFeatureRisk ? RISK_COLORS[d] : null)
    .on("click",(event,d)=>{
      selectedFeatureRisk=d;
      selectedFeatureKey=null;
      selectedFeatureSubgroup=null;
      selectedDistributionFeature=null;
      featureSubgroupSequencePaused=false;
      clearInterval(featureSubgroupSequenceTimer);
      featureSidebar.selectAll("button").classed("active",x=>x===selectedFeatureRisk)
        .style("background", x=>x===selectedFeatureRisk ? RISK_COLORS[x] : null);
      drawFeatureHeatmaps();
    });
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
  const eventTitle = activeEventWindow === "A" ? TEXT.eventsShortTitle : TEXT.eventsLongTitle;
  const horizonYears = TEXT.eventHorizonYears[activeEventWindow];
  const highlightedYears = `<span class="event-horizon-number${activeEventWindow === "B" ? " changed" : ""}">${horizonYears}</span>`;
  const eventTitleElement = document.getElementById("t-events-card-title");
  if (eventTitleElement.dataset.window !== activeEventWindow) {
    eventTitleElement.dataset.window = activeEventWindow;
    eventTitleElement.innerHTML = eventTitle.replace(horizonYears, highlightedYears);
  }
  d3.select("#events-window-subtitle").text(
    activeEventWindow === "A" ? TEXT.eventsShortSubtitle : TEXT.eventsLongSubtitle
  );
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
function featureLabel(feature) {
  return TEXT.featureLabels[feature] || feature;
}

function replaceFeatureText(template, values) {
  return Object.entries(values).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, String(value)),
    template,
  );
}

function activeFeatureMetric(feature) {
  return (DATA.features.importanceByRisk[selectedFeatureRisk] || []).find(d => d.feature === feature);
}

function drawFeatureImportanceV2() {
  const metrics = new Map((DATA.features.importanceByRisk[selectedFeatureRisk] || []).map(d => [d.feature, d]));
  d3.select("#feature-order-controls").selectAll("button")
    .text(function() { return this.dataset.order === "category" ? TEXT.featureGroupByCategory : TEXT.featureOrderBySignificance; })
    .classed("active", function() { return this.dataset.order === featureOrderMode; })
    .attr("aria-pressed", function() { return this.dataset.order === featureOrderMode ? "true" : "false"; })
    .on("click", function() {
      featureOrderMode = this.dataset.order;
      drawFeatureImportanceV2();
    });
  d3.select("#feature-click-hint").text(TEXT.featureClickHint);
  const chart = d3.select("#feature-importance-chart");
  chart.selectAll("*").remove();
  let lastCategory = null;
  const features = featureOrderMode === "significance"
    ? [...DATA.features.featureOrder].sort((a, b) => (metrics.get(b)?.absRho || 0) - (metrics.get(a)?.absRho || 0))
    : DATA.features.featureOrder;
  const strongContainer = featureOrderMode === "significance" && features.some(feature => (metrics.get(feature)?.absRho || 0) >= 0.3)
    ? chart.append("div").attr("class", "importance-strong-group")
    : null;
  features.forEach(feature => {
    const meta = DATA.features.featureMeta[feature];
    if (featureOrderMode === "category" && meta.category !== lastCategory) {
      chart.append("div").attr("class", "importance-group-label").text(TEXT.featureCategories[meta.category] || meta.category);
      lastCategory = meta.category;
    }
    const metric = metrics.get(feature) || {};
    const width = Math.min(100, Math.max(0, (metric.absRho || 0) * 200));
    const strong = featureOrderMode === "significance" && (metric.absRho || 0) >= 0.3;
    const negativeActive = selectedFeatureKey === feature && (metric.rho || 0) < 0;
    const button = (strong ? strongContainer : chart).append("button")
      .attr("type", "button")
      .attr("class", `importance-row${selectedFeatureKey === feature ? " active" : ""}${negativeActive ? " negative-active" : ""}`)
      .attr("aria-pressed", selectedFeatureKey === feature ? "true" : "false")
      .attr("title", strong ? TEXT.featureStrongTooltip : `${featureLabel(feature)} · |ρ| ${d3.format(".2f")(metric.absRho || 0)}`)
      .on("click", () => {
        selectedFeatureKey = selectedFeatureKey === feature ? null : feature;
        drawFeatureHeatmaps();
      });
    button.append("span").attr("class", `correlation-marker ${(metric.rho || 0) < 0 ? "negative" : "positive"}`);
    button.append("span").attr("class", "importance-label").text(featureLabel(feature));
    button.append("span").attr("class", "importance-bar-track")
      .append("span").attr("class", "importance-bar").style("display", "block").style("width", `${width}%`);
  });
  const tabs = document.getElementById("feature-footnote-tabs");
  tabs.innerHTML = "";
  [
    [TEXT.featureSourcesTopic, TEXT.featureSourcesNote],
    [TEXT.featureRankingTopic, replaceFeatureText(TEXT.featureRankingNote, {threshold: d3.format(".2f")(DATA.features.minimumEffect)})],
  ].forEach(([labelText, content]) => {
    const topic = document.createElement("span");
    topic.className = "feature-footnote-topic";
    topic.append(document.createTextNode(labelText), makeInfoButton(content, {label: `${labelText}: ${TEXT.informationTooltipLabel}`}));
    tabs.appendChild(topic);
  });
}

function drawFeatureMedianLine() {
  d3.select(".feature-line-pane").classed("scatter-active", false).classed("scatter-negative", false);
  d3.select("#feature-chart-title").text(TEXT.featureLineTitle);
  d3.select("#feature-relationship").attr("hidden", true);
  drawLineChart("#feature-event-window", DATA.eventWindows.windowA.byRating, "riskRating", 36, selectedFeatureRisk, -12, {hideOtherGroups: true, marginRight: 82, xAxisLabel: TEXT.featureXAxis, yAxisLabel: TEXT.featureYAxis, eventLabel: TEXT.featureEventMarker});
  d3.select("#feature-line-legend").html(`<span class="feature-line-legend-item"><span class="feature-line-key" style="border-color:${RISK_COLORS[selectedFeatureRisk]}"></span>${replaceFeatureText(TEXT.featureGroupMedianLegend, {risk: selectedFeatureRisk})}</span>`);
}

function featureTickFormatter(feature) {
  const format = DATA.features.featureMeta[feature]?.format;
  if (format === "currency") return d3.format("$,.2s");
  if (format === "pct") return fmtPct;
  if (format === "percent") return value => `${d3.format(".1f")(value)}%`;
  return d3.format(".2s");
}

function drawFeatureScatter(feature) {
  const allRows = (DATA.features.countyRowsByRisk[selectedFeatureRisk] || [])
    .map(d => ({fips: d.fips, x: d.values[feature], y: d.target}))
    .filter(d => d.x != null && d.y != null);
  const iqrBounds = values => {
    const sorted = values.filter(Number.isFinite).sort(d3.ascending);
    const q1 = d3.quantileSorted(sorted, .25), q3 = d3.quantileSorted(sorted, .75);
    const spread = q3 - q1;
    return [q1 - 1.5 * spread, q3 + 1.5 * spread];
  };
  const [xLow, xHigh] = iqrBounds(allRows.map(d => d.x));
  const [yLow, yHigh] = iqrBounds(allRows.map(d => d.y));
  const rows = allRows.filter(d => d.x >= xLow && d.x <= xHigh && d.y >= yLow && d.y <= yHigh);
  const metric = activeFeatureMetric(feature) || {};
  const rho = metric.rho || 0;
  d3.select(".feature-line-pane").classed("scatter-active", true).classed("scatter-negative", rho < 0);
  const svg = d3.select("#feature-event-window");
  const width = svg.node().clientWidth || 700, height = svg.node().clientHeight || 500;
  const margin = {top: 24, right: 24, bottom: 58, left: 68};
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  const x = d3.scaleLinear().domain(d3.extent(rows, d => d.x)).nice().range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain(d3.extent(rows, d => d.y)).nice().range([height - margin.bottom, margin.top]);
  svg.append("g").attr("class", "grid").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(6).tickFormat(featureTickFormatter(feature)));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  svg.append("g").selectAll("circle").data(rows).join("circle")
    .attr("cx", d => x(d.x)).attr("cy", d => y(d.y)).attr("r", 3.2)
    .attr("fill", RISK_COLORS[selectedFeatureRisk]).attr("opacity", .55)
    .on("mousemove", (event, d) => showTooltip(event, `<strong>${d.fips}</strong><br>${featureLabel(feature)}: ${featureTickFormatter(feature)(d.x)}<br>${TEXT.featureScatterYAxis}: ${fmtPct(d.y)}`))
    .on("mouseleave", () => hideTooltip());
  if (rows.length >= 2) {
    const meanX = d3.mean(rows, d => d.x), meanY = d3.mean(rows, d => d.y);
    const denominator = d3.sum(rows, d => (d.x - meanX) ** 2);
    const slope = denominator ? d3.sum(rows, d => (d.x - meanX) * (d.y - meanY)) / denominator : 0;
    const intercept = meanY - slope * meanX;
    const [trendStart, trendEnd] = x.domain();
    svg.append("line")
      .attr("x1", x(trendStart)).attr("y1", y(intercept + slope * trendStart))
      .attr("x2", x(trendEnd)).attr("y2", y(intercept + slope * trendEnd))
      .attr("stroke", "#17332d").attr("stroke-width", 2).attr("stroke-dasharray", "5 5")
      .attr("pointer-events", "none");
  }
  svg.append("text").attr("x", width / 2).attr("y", height - 9).attr("text-anchor", "middle").attr("fill", "#66717b").attr("font-size", 12).text(featureLabel(feature));
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", 18).attr("text-anchor", "middle").attr("fill", "#66717b").attr("font-size", 12).text(TEXT.featureScatterYAxis);
  const chartTitle = document.getElementById("feature-chart-title");
  chartTitle.innerHTML = "";
  const outcomeLabel = document.createElement("span");
  outcomeLabel.textContent = TEXT.featureOutcomeTerm;
  const comparisonLabel = document.createElement("span");
  comparisonLabel.textContent = ` vs. ${featureLabel(feature)}`;
  chartTitle.append(outcomeLabel, makeInfoButton(TEXT.featureOutcomeTooltip), comparisonLabel);
  d3.select("#feature-line-legend").html("");
  const strengthKey = Math.abs(rho) > 0.5 ? "strong" : Math.abs(rho) >= 0.3 ? "moderate" : "weak";
  const directionKey = rho >= 0 ? "positive" : "negative";
  d3.select("#feature-relationship").attr("hidden", null).text(replaceFeatureText(TEXT.featureRelationship, {
    feature: featureLabel(feature),
    strength: TEXT.featureRelationshipStrength[strengthKey],
    direction: TEXT.featureRelationshipDirection[directionKey],
  }));
}

const FEATURE_SUBGROUP_COLORS = ["#54278f", "#807dba", "#6baed6", "#2171b5"];

function subgroupName(group, count) {
  const names = count === 3 ? TEXT.subgroupNamesThree : TEXT.subgroupNamesFour;
  return names[group.index] || replaceFeatureText(TEXT.subgroupFallback, {number: group.index + 1});
}

function subgroupProseName(label) {
  return String(label || "").toLowerCase();
}

function subgroupPerformanceName(label) {
  return String(label || "")
    .replace(/^(Strong|Mild)\s+/i, "")
    .toLowerCase();
}

function playbookPerformanceName(label) {
  const names = {
    "Strong Overperformers": "strong overperformer",
    "Mild Overperformers": "mild overperformer",
    "Mild Underperformers": "mild underperformer",
    "Strong Underperformers": "strong underperformer",
    "Overperformers": "overperformer",
    "Average Performers": "average performer",
    "Underperformers": "underperformer",
  };
  return names[label] || subgroupProseName(label).replace(/s$/, "");
}

function mostImportantFeatureMetrics(risk) {
  const metrics = [...(DATA.features.importanceByRisk[risk] || [])]
    .filter(metric => metric.rho != null)
    .sort((a, b) => (b.absRho || 0) - (a.absRho || 0));
  const strongest = metrics.filter(metric => (metric.absRho || 0) >= 0.3);
  return strongest.length ? strongest : metrics.slice(0, 1);
}

function subgroupFeatureRelations(risk, subgroup) {
  if (!subgroup) return [];
  const peerRows = DATA.features.countyRowsByRisk[risk] || [];
  return (subgroup.traits || []).map(trait => {
    const peerValues = peerRows
      .map(row => row.values?.[trait.feature])
      .filter(Number.isFinite)
      .sort(d3.ascending);
    if (!Number.isFinite(trait.median) || !peerValues.length) return null;
    const peerMedian = d3.quantileSorted(peerValues, .5);
    const peerQ1 = d3.quantileSorted(peerValues, .25);
    const peerQ3 = d3.quantileSorted(peerValues, .75);
    const tolerance = Math.max((peerQ3 - peerQ1) * .15, Math.abs(peerMedian) * .02, 1e-6);
    const relation = trait.median > peerMedian + tolerance
      ? "higher"
      : trait.median < peerMedian - tolerance
        ? "lower"
        : "close";
    return {...trait, relation};
  }).filter(Boolean);
}

function drawFeatureSubgroupSummary() {
  const payload = DATA.features.subgroupsByRisk[selectedFeatureRisk] || {groups: []};
  const group = payload.groups.find(d => d.index === selectedFeatureSubgroup);
  const groupLabel = group ? subgroupName(group, payload.groups.length) : "";
  const relations = subgroupFeatureRelations(selectedFeatureRisk, group);
  const summary = d3.select("#feature-subgroup-summary").html("");
  summary.append("h3").text(replaceFeatureText(TEXT.featureFrame3Title, {
    subgroup: groupLabel,
    risk: selectedFeatureRisk,
  }));
  if (!relations.length) {
    summary.append("p").attr("class", "feature-subgroup-summary-intro").text(TEXT.featureSubgroupSummaryUnavailable);
    return;
  }
  summary.append("p").attr("class", "feature-subgroup-summary-intro").text(replaceFeatureText(TEXT.featureSubgroupSummaryIntro, {
    subgroup: groupLabel,
    risk: selectedFeatureRisk,
  }));
  const rowScroller = summary.append("div").attr("class", "feature-subgroup-summary-rows");
  const rows = rowScroller.selectAll("div.feature-subgroup-summary-row").data(relations, d => d.feature).join("div")
    .attr("class", "feature-subgroup-summary-row");
  rows.append("strong").text(d => featureLabel(d.feature));
  rows.append("span").attr("class", d => `feature-peer-relation ${d.relation}`).text(d => TEXT.featurePeerRelation[d.relation]);
}

function featureDistributionOptions(payload) {
  return mostImportantFeatureMetrics(selectedFeatureRisk);
}

function drawFeatureSubgroupPanel() {
  const payload = DATA.features.subgroupsByRisk[selectedFeatureRisk] || {groups: [], excludedOutliers: 0};
  if (selectedFeatureSubgroup == null || !payload.groups.some(d => d.index === selectedFeatureSubgroup)) {
    selectedFeatureSubgroup = [...payload.groups].sort((a, b) => b.index - a.index)[0]?.index ?? null;
  }
  const options = featureDistributionOptions(payload);
  if (!options.some(metric => metric.feature === selectedDistributionFeature)) selectedDistributionFeature = options[0]?.feature || null;
  const controls = d3.select("#feature-distribution-controls");
  controls.classed("single-feature", options.length <= 1).selectAll("*").remove();
  if (!payload.hasStrongFeatures) controls.append("span").attr("class", "feature-click-hint").style("grid-column", "1 / -1").text(TEXT.featureDistributionFallback);
  const rotateFeature = direction => {
    const current = Math.max(0, options.findIndex(metric => metric.feature === selectedDistributionFeature));
    selectedDistributionFeature = options[(current + direction + options.length) % options.length]?.feature || null;
    drawFeatureSubgroupPanel();
  };
  if (options.length > 1) controls.append("button").attr("type", "button").attr("aria-label", TEXT.featureDistributionPrevious)
    .attr("title", TEXT.featureDistributionPrevious).text("←").on("click", () => rotateFeature(-1));
  controls.append("span").attr("class", "feature-distribution-current").text(featureLabel(selectedDistributionFeature));
  if (options.length > 1) controls.append("button").attr("type", "button").attr("aria-label", TEXT.featureDistributionNext)
    .attr("title", TEXT.featureDistributionNext).text("→").on("click", () => rotateFeature(1));
  const group = payload.groups.find(d => d.index === selectedFeatureSubgroup);
  if (!group || !selectedDistributionFeature) return;
  const rows = DATA.features.countyRowsByRisk[selectedFeatureRisk] || [];
  const allValues = rows.map(row => row.values[selectedDistributionFeature]).filter(Number.isFinite).sort(d3.ascending);
  const rawSubgroupPoints = rows
    .filter(row => DATA.features.subgroupByFips[row.fips] === selectedFeatureSubgroup)
    .map(row => ({fips: row.fips, value: row.values[selectedDistributionFeature]}))
    .filter(point => Number.isFinite(point.value)).sort((a, b) => d3.ascending(a.value, b.value));
  const rawSubgroupValues = rawSubgroupPoints.map(point => point.value);
  const q1All = d3.quantileSorted(allValues, .25), q3All = d3.quantileSorted(allValues, .75);
  const spread = q3All - q1All;
  const lowerBound = spread > 0 ? q1All - 1.5 * spread : d3.min(allValues);
  const upperBound = spread > 0 ? q3All + 1.5 * spread : d3.max(allValues);
  const removeOutliers = selectedFeatureRisk !== "Very High";
  const displayedValues = removeOutliers
    ? allValues.filter(value => value >= lowerBound && value <= upperBound)
    : allValues;
  const subgroupPoints = removeOutliers
    ? rawSubgroupPoints.filter(point => point.value >= lowerBound && point.value <= upperBound)
    : rawSubgroupPoints;
  const subgroupValues = subgroupPoints.map(point => point.value);
  const svg = d3.select("#feature-distribution-chart");
  const width = svg.node().clientWidth || 320, height = svg.node().clientHeight || 210;
  const margin = {top: 8, right: 18, bottom: 34, left: 10};
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  if (!displayedValues.length || !subgroupValues.length) return;
  const x = d3.scaleLinear().domain(d3.extent(displayedValues)).nice().range([margin.left, width - margin.right]);
  const selectedColor = FEATURE_SUBGROUP_COLORS[group.index];
  const drawDistribution = (points, y, color, opacity) => {
    const values = points.map(point => point.value).sort(d3.ascending);
    svg.append("g").selectAll("circle").data(points).join("circle")
      .attr("cx", d => x(d.value)).attr("cy", (d, i) => y + ((i % 7) - 3) * 1.6)
      .attr("r", 2.8).attr("fill", color).attr("opacity", opacity)
      .style("cursor", "pointer")
      .on("mousemove", (event, d) => {
        const county = countyByFips.get(d.fips);
        showTooltip(event, `<strong>${countyDisplayName(county) || d.fips}</strong><br>${featureLabel(selectedDistributionFeature)}: ${featureTickFormatter(selectedDistributionFeature)(d.value)}`);
      })
      .on("mouseleave", () => hideTooltip());
    const q1 = d3.quantileSorted(values, .25), medianValue = d3.quantileSorted(values, .5), q3 = d3.quantileSorted(values, .75);
    svg.append("rect").attr("x", x(q1)).attr("y", y - 11).attr("width", Math.max(1, x(q3) - x(q1))).attr("height", 22).attr("fill", color).attr("opacity", .2).attr("stroke", color);
    svg.append("line").attr("x1", x(medianValue)).attr("x2", x(medianValue)).attr("y1", y - 13).attr("y2", y + 13).attr("stroke", color).attr("stroke-width", 3);
  };
  drawDistribution(subgroupPoints, (margin.top + height - margin.bottom) / 2, selectedColor, .48);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(5).tickFormat(featureTickFormatter(selectedDistributionFeature)));
  const distributionTitle = document.getElementById("feature-distribution-title");
  const distributionTitleKey = `${selectedFeatureRisk}:${selectedDistributionFeature}`;
  if (distributionTitle.dataset.feature !== distributionTitleKey) {
    if (activeInfoTooltipTrigger && distributionTitle.contains(activeInfoTooltipTrigger)) closeInfoTooltip();
    distributionTitle.innerHTML = "";
    distributionTitle.dataset.feature = distributionTitleKey;
    distributionTitle.append(
      document.createTextNode(replaceFeatureText(TEXT.featureDistributionTitle, {feature: featureLabel(selectedDistributionFeature)})),
      makeInfoButton(removeOutliers ? TEXT.featureDistributionOutlierTooltip : TEXT.featureDistributionVeryHighTooltip, {label: TEXT.informationTooltipLabel}),
    );
  }
  const levelKey = d3.median(rawSubgroupValues) >= d3.median(allValues) ? "higher" : "lower";
  const level = TEXT.featureDistributionLevel[levelKey];
  d3.select("#feature-subgroup-takeaway").text(replaceFeatureText(TEXT.featureSubgroupTakeaway, {
    level,
    feature: featureLabel(selectedDistributionFeature),
    subgroup: subgroupPerformanceName(subgroupName(group, payload.groups.length)),
    risk: selectedFeatureRisk,
  }));
  d3.select("#feature-sequence-resume").text(TEXT.featureResumeSequence).classed("visible", featureSubgroupSequencePaused)
    .on("click", () => {
      featureSubgroupSequencePaused = false;
      startFeatureSubgroupSequence();
      drawFeatureSubgroupDetail();
    });
}

function drawFeatureSubgroupDetail() {
  const state = document.querySelector("#features .story-stage")?.dataset.storyState;
  if (state === "feature-frame-3") drawFeatureSubgroupSummary();
  else drawFeatureSubgroupPanel();
  d3.select("#feature-sequence-resume").text(TEXT.featureResumeSequence).classed("visible", featureSubgroupSequencePaused)
    .on("click", () => {
      featureSubgroupSequencePaused = false;
      startFeatureSubgroupSequence();
      drawFeatureSubgroupDetail();
    });
}

function selectFeatureSubgroup(index, userInitiated = true) {
  selectedFeatureSubgroup = index;
  if (userInitiated) {
    featureSubgroupSequencePaused = true;
    clearInterval(featureSubgroupSequenceTimer);
  }
  drawFeatureSubgroupDetail();
  drawFeatureSubgroupLines();
}

function startFeatureSubgroupSequence() {
  clearInterval(featureSubgroupSequenceTimer);
  const payload = DATA.features.subgroupsByRisk[selectedFeatureRisk] || {groups: []};
  const sequence = [...payload.groups].sort((a, b) => b.index - a.index);
  if (!sequence.length) return;
  if (selectedFeatureSubgroup == null) selectedFeatureSubgroup = sequence[0].index;
  if (featureSubgroupSequencePaused) return;
  featureSubgroupSequenceTimer = setInterval(() => {
    const current = sequence.findIndex(group => group.index === selectedFeatureSubgroup);
    selectedFeatureSubgroup = sequence[(current + 1) % sequence.length].index;
    drawFeatureSubgroupDetail();
    drawFeatureSubgroupLines();
  }, 3200);
}

function drawFeatureSubgroupLines() {
  const payload = DATA.features.subgroupsByRisk[selectedFeatureRisk] || {groups: []};
  const domainValues = payload.groups.flatMap(group => group.values.map(d => d.value).filter(Number.isFinite));
  d3.select(".feature-line-pane").classed("scatter-active", false).classed("scatter-negative", false);
  const chart = drawLineChart("#feature-event-window", DATA.eventWindows.windowA.byRating, "riskRating", 36, selectedFeatureRisk, -12, {hideOtherGroups: true, hideEndLabel: true, extraDomainValues: domainValues, upperDomainPadding: 0.16, marginRight: 24, xAxisLabel: TEXT.featureXAxis, yAxisLabel: TEXT.featureYAxis, eventLabel: TEXT.featureEventMarker});
  const svg = d3.select("#feature-event-window");
  const line = d3.line().defined(d => d.value != null).x(d => chart.x(d.month)).y(d => chart.y(d.value));
  svg.selectAll("path.feature-subgroup-line").data(payload.groups).join("path")
    .attr("class", "line feature-subgroup-line").attr("stroke", d => FEATURE_SUBGROUP_COLORS[d.index])
    .attr("stroke-width", d => d.index === selectedFeatureSubgroup ? 4.5 : 2.5)
    .attr("opacity", d => selectedFeatureSubgroup == null || d.index === selectedFeatureSubgroup ? 1 : .16)
    .attr("d", d => line(d.values)).style("cursor", "pointer")
    .on("click", (event, d) => selectFeatureSubgroup(d.index, true));
  const orderedGroups = [...payload.groups].sort((a, b) => a.index - b.index);
  const subgroupButtons = d3.select("#feature-subgroup-toggles")
    .classed("visible", true)
    .selectAll("button.feature-subgroup-control")
    .data(orderedGroups, d => d.index)
    .join("button")
    .attr("type", "button")
    .attr("class", d => `feature-subgroup-control${d.index === selectedFeatureSubgroup ? " active" : ""}`)
    .attr("aria-pressed", d => d.index === selectedFeatureSubgroup ? "true" : "false")
    .attr("aria-label", d => `${subgroupName(d, payload.groups.length, selectedFeatureRisk)} subgroup${d.index === selectedFeatureSubgroup ? ", selected" : ""}`)
    .style("--subgroup-color", d => FEATURE_SUBGROUP_COLORS[d.index])
    .on("pointerdown", (event, d) => {
      if (event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      selectFeatureSubgroup(Number(d.index), true);
    })
    .on("click", (event, d) => {
      if (event.detail !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      selectFeatureSubgroup(Number(d.index), true);
  });
  subgroupButtons.selectAll("span.feature-subgroup-control-label")
    .data(d => [d], d => d.index)
    .join("span")
    .attr("class", "feature-subgroup-control-label")
    .text(d => subgroupName(d, payload.groups.length, selectedFeatureRisk));
  d3.select("#feature-chart-title").text(TEXT.featureLineTitle);
  d3.select("#feature-relationship").attr("hidden", true);
  d3.select("#feature-line-legend").html("");
}

function drawFeatureHeatmaps() {
  d3.select("#feature-risk-sidebar").selectAll("button").classed("active", d => d === selectedFeatureRisk)
    .style("background", d => d === selectedFeatureRisk ? RISK_COLORS[d] : null);
  drawFeatureImportanceV2();
  const state = document.querySelector("#features .story-stage")?.dataset.storyState || "feature-frame-1";
  d3.select("#feature-detail-title").text(replaceFeatureText(
    state === "feature-frame-2" ? TEXT.featureFrame2Title : state === "feature-frame-3" ? TEXT.featureFrame3Title : TEXT.featureFrame1Title,
    {risk: selectedFeatureRisk, subgroup: ""},
  ));
  const detailStack = document.querySelector(".feature-detail-stack");
  const detailTitle = document.getElementById("feature-detail-title");
  detailStack?.style.setProperty("--feature-frame-top", `${(detailTitle?.offsetHeight || 25) + 8}px`);
  if (state === "feature-frame-2" || state === "feature-frame-3") {
    if (selectedFeatureSubgroup == null) {
      const groups = DATA.features.subgroupsByRisk[selectedFeatureRisk]?.groups || [];
      selectedFeatureSubgroup = [...groups].sort((a, b) => b.index - a.index)[0]?.index ?? null;
    }
    drawFeatureSubgroupDetail();
    drawFeatureSubgroupLines();
    startFeatureSubgroupSequence();
  }
  else {
    clearInterval(featureSubgroupSequenceTimer);
    d3.select("#feature-subgroup-toggles").classed("visible", false).selectAll("*").remove();
    if ((state === "feature-frame-1" || state === "takeaway-feature") && selectedFeatureKey) drawFeatureScatter(selectedFeatureKey);
    else drawFeatureMedianLine();
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

function drawPlaybookMap(svgSelector = "#county-selection-map", county = null, autoZoom = false, interactive = true) {
  const svg = d3.select(svgSelector);
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
    .style("cursor", interactive ? "pointer" : "default")
    .on("mousemove", (event, d) => {
      const profile = playbookCountyByFips.get(d.properties.fips);
      if (!profile) return;
      showTooltip(event, `<strong>${countyDisplayName(profile)}</strong>`);
    })
    .on("mouseleave", () => hideTooltip())
    .on("click", (event, d) => {
      if (!interactive) return;
      const profile = playbookCountyByFips.get(d.properties.fips);
      if (profile) {
        selectPlaybookCounty(profile);
        goToPlaybookProfile();
      }
    });
  drawStateBoundaries(group, path);

  if (!interactive) {
    svg.on(".zoom", null);
    if (county) {
      const selectedFeature = DATA.geojson.features.find(d => d.properties.fips === county.fips);
      if (selectedFeature) {
        const [cx, cy] = path.centroid(selectedFeature);
        const scale = county.state === "AK" ? 3.2 : county.state === "HI" ? 4.2 : 5.2;
        group.attr("transform", d3.zoomIdentity.translate(width / 2, height / 2).scale(scale).translate(-cx, -cy));
      }
    }
    return;
  }

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

function playbookFeatureProfile(county) {
  const metrics = mostImportantFeatureMetrics(county.riskRating);
  const row = (DATA.features.countyRowsByRisk[county.riskRating] || []).find(d => d.fips === county.fips);
  const subgroupIndex = DATA.features.subgroupByFips?.[county.fips];
  const payload = DATA.features.subgroupsByRisk[county.riskRating] || {groups: []};
  const subgroup = payload.groups.find(d => d.index === subgroupIndex);
  return {metrics, row, subgroup, subgroupName: subgroup ? subgroupName(subgroup, payload.groups.length, county.riskRating) : null};
}

function renderPlaybookFeatureSummary(county, summarizeSubgroup = false) {
  const profile = playbookFeatureProfile(county);
  const subgroupRelations = subgroupFeatureRelations(county.riskRating, profile.subgroup);
  const featureTitle = summarizeSubgroup && profile.subgroupName
    ? replaceFeatureText(TEXT.playbookSubgroupFeatureTitle, {
      subgroup: profile.subgroupName,
      risk: county.riskRating || "Unknown",
    })
    : replaceFeatureText(TEXT.playbookFeatureTitle, {risk: county.riskRating || "Unknown"});
  d3.select("#playbook-feature-title").text(featureTitle);
  const hasFeatureData = summarizeSubgroup
    ? Boolean(profile.subgroupName && subgroupRelations.length)
    : Boolean(
      profile.row
      && profile.metrics.length
      && profile.subgroupName
      && profile.metrics.every(metric => Number.isFinite(profile.row.values?.[metric.feature]))
    );
  if (!hasFeatureData) {
    const insufficientCopy = !profile.row
      ? TEXT.playbookInsufficientEventWindowData
      : TEXT.playbookInsufficientFeatureData;
    d3.select("#playbook-feature-summary").html(`<div class="playbook-feature-insufficient">${replaceFeatureText(insufficientCopy, {county: countyDisplayName(county)})}</div>`);
    d3.select("#playbook-subgroup-summary").style("display", "none").text("");
  } else if (summarizeSubgroup) {
    const summary = d3.select("#playbook-feature-summary").html("");
    const rows = summary.selectAll("div.playbook-feature-row").data(subgroupRelations, d => d.feature).join("div")
      .attr("class", "playbook-feature-row");
    rows.append("strong").text(d => featureLabel(d.feature));
    rows.append("span").attr("class", d => `feature-peer-relation ${d.relation}`).text(d => TEXT.featurePeerRelation[d.relation]);
    d3.select("#playbook-subgroup-summary")
      .style("display", null)
      .text(replaceFeatureText(TEXT.playbookSubgroup, {
        county: countyDisplayName(county),
        subgroup: playbookPerformanceName(profile.subgroupName),
        risk: county.riskRating || "Unknown",
      }));
  } else {
    d3.select("#playbook-feature-summary").html(profile.metrics.map(metric => {
      const value = profile.row.values?.[metric.feature];
      const format = DATA.features.featureMeta[metric.feature]?.format;
      return `<div class="playbook-feature-row"><strong>${featureLabel(metric.feature)}</strong><span>${value == null ? TEXT.playbookInsufficientFeatureValue : formatFeatureVal(value, format)}</span></div>`;
    }).join(""));
    d3.select("#playbook-subgroup-summary")
      .style("display", null)
      .text(replaceFeatureText(TEXT.playbookSubgroup, {
        county: countyDisplayName(county),
        subgroup: playbookPerformanceName(profile.subgroupName),
        risk: county.riskRating || "Unknown",
      }));
  }
}

function buildRiskGroupSeries(county) {
  const months = DATA.playbook.monthlyHistoryMonths || [];
  const histories = DATA.playbook.monthlyHistoryValuesByFips || {};
  const selected = (DATA.playbook.counties || []).filter(peer => peer.riskRating === county.riskRating && histories[peer.fips]);
  return months.map((month, index) => {
    const valid = selected.map(peer => histories[peer.fips]?.[index]).filter(Number.isFinite).sort(d3.ascending);
    return valid.length
      ? {month, q1: d3.quantileSorted(valid, .25), median: d3.quantileSorted(valid, .5), q3: d3.quantileSorted(valid, .75)}
      : {month, q1: null, median: null, q3: null};
  });
}

function playbookEvents(county) {
  const parseMonth = d3.utcParse("%Y-%m");
  return ((DATA.playbook.eventsByFips || {})[county.fips] || []).map(d => ({
    eventKey: d[0], source: d[1], type: d[2], name: d[3], start: d[4], end: d[5],
    startDate: parseMonth(d[4]), endDate: parseMonth(d[5]),
  }));
}

function renderPlaybookEventList(county) {
  const events = playbookEvents(county);
  d3.select("#playbook-events-title").text(TEXT.playbookPastEventsTitle);
  const list = d3.select("#playbook-event-column");
  if (!events.length) {
    list.html(`<div class="playbook-event-card">${TEXT.playbookNoPastEvents}</div>`);
    return;
  }
  list.html(events.map(event => `<div class="playbook-event-card" data-event-key="${event.eventKey}"><strong>${eventIcon(event)} ${normalCase(event.name || event.type)}</strong><br>${d3.utcFormat("%b %Y")(event.startDate)} to ${d3.utcFormat("%b %Y")(event.endDate)}</div>`).join(""));
  list.selectAll("[data-event-key]")
    .on("mouseenter", function() {
      const eventKey = this.dataset.eventKey;
      d3.select("#playbook-ppsf-history").selectAll(".event-period")
        .classed("event-focused", d => d.eventKey === eventKey)
        .classed("event-muted", d => d.eventKey !== eventKey);
    })
    .on("mouseleave", () => d3.select("#playbook-ppsf-history").selectAll(".event-period").classed("event-focused", false).classed("event-muted", false));
}

function drawPlaybookHistory(county, compareRisk = false) {
  d3.select("#t-playbook-history-title").text(countyDisplayName(county));
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
  const riskSeries = buildRiskGroupSeries(county);
  const events = playbookEvents(county);
  const svg = d3.select("#playbook-ppsf-history");
  const width = svg.node().clientWidth || 1050;
  const height = svg.node().clientHeight || 430;
  const margin = {top: 22, right: 24, bottom: 42, left: 62};
  svg.attr("viewBox", [0, 0, width, height]).selectAll("*").remove();
  const clipId = `playbook-history-clip-${county.fips}`;
  svg.append("defs").append("clipPath").attr("id", clipId).append("rect")
    .attr("x", margin.left).attr("y", margin.top)
    .attr("width", width - margin.left - margin.right).attr("height", height - margin.top - margin.bottom);
  const plot = svg.append("g").attr("clip-path", `url(#${clipId})`);

  const x = d3.scaleUtc().domain([historyStart, historyDomainEnd]).range([margin.left, width - margin.right]);
  const riskValues = compareRisk ? riskSeries.flatMap(d => [d.q1, d.q3]).filter(Number.isFinite) : [];
  const extentValues = observed.map(d => d.value).concat(riskValues);
  const valueExtent = extentValues.length ? d3.extent(extentValues) : [-0.1, 0.1];
  const padding = Math.max((valueExtent[1] - valueExtent[0]) * 0.12, 0.01);
  const y = d3.scaleLinear().domain([valueExtent[0] - padding, valueExtent[1] + padding]).nice()
    .range([height - margin.bottom, margin.top]);

  const grid = svg.append("g").attr("class", "grid").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(""));

  const chartStart = x.domain()[0], chartEnd = x.domain()[1];
  plot.selectAll("rect.event-period")
    .data(events.filter(d => d.endDate >= chartStart && d.startDate <= chartEnd))
    .join("rect")
    .attr("class", "event-period")
    .attr("data-event-key", d => d.eventKey)
    .attr("x", d => x(d3.max([d.startDate, chartStart])))
    .attr("y", margin.top)
    .attr("width", d => Math.max(3, x(d3.min([d.endDate, chartEnd])) - x(d3.max([d.startDate, chartStart]))))
    .attr("height", height - margin.top - margin.bottom)
    .attr("fill", "#df7d2f").attr("opacity", .16)
    .on("mousemove", (event, d) => showTooltip(event, `<strong>${normalCase(d.name || d.type)}</strong><br>${d3.utcFormat("%b %Y")(d.startDate)} to ${d3.utcFormat("%b %Y")(d.endDate)}<br>${normalCase(d.source)}`))
    .on("mouseleave", () => hideTooltip());

  const riskArea = d3.area().defined(d => d.q1 != null && d.q3 != null)
    .x(d => x(parseMonth(d.month))).y0(d => y(d.q1)).y1(d => y(d.q3));
  const riskLine = d3.line().defined(d => d.median != null)
    .x(d => x(parseMonth(d.month))).y(d => y(d.median));
  let riskBand = null, riskMedian = null;
  if (compareRisk) {
    riskBand = plot.append("path").datum(riskSeries).attr("class", "band playbook-risk-band")
      .attr("fill", RISK_COLORS[county.riskRating] || "#66717b").attr("opacity", .14).attr("d", riskArea);
    riskMedian = plot.append("path").datum(riskSeries).attr("class", "line playbook-risk-median")
      .attr("stroke", RISK_COLORS[county.riskRating] || "#66717b").attr("stroke-width", 2)
      .attr("stroke-dasharray", "6 4").attr("opacity", .9).attr("d", riskLine);
  }
  const countyLinePath = d3.line().defined(d => d.value != null).x(d => x(d.date)).y(d => y(d.value));
  const countyLine = plot.append("path").datum(history).attr("class", "line playbook-county-line")
    .attr("stroke", COUNTY_LINE_COLOR).attr("stroke-width", 2.4)
    .attr("d", countyLinePath);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(d3.utcYear.every(1)).tickFormat(d3.utcFormat("%Y")));
  const yAxis = svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6).tickFormat(fmtPct));
  svg.append("text").attr("x", margin.left).attr("y", 13).attr("fill", "#66717b").attr("font-size", 11)
    .text("Median PPSF YoY");
  const legendItems = [
    {label: TEXT.playbookSeriesLegend, color: COUNTY_LINE_COLOR, opacity: 1, line: true},
  ];
  if (compareRisk) legendItems.push({label: replaceFeatureText(TEXT.playbookRiskSeriesLegend, {risk: county.riskRating}), color: RISK_COLORS[county.riskRating] || "#66717b", opacity: .85, line: true});
  if (events.length) legendItems.push({label: TEXT.playbookEventLegend, color: "#df7d2f", opacity: .22});
  d3.select("#playbook-history-legend").html(legendItems.map(item =>
    `<span class="playbook-history-legend-item"><span class="playbook-history-legend-swatch" style="background:${item.color};opacity:${item.opacity};${item.line ? "height:3px;border:none;" : ""}"></span>${item.label}</span>`
  ).join(""));
  if (!observed.length) {
    svg.append("text").attr("x", width / 2).attr("y", height / 2)
      .attr("text-anchor", "middle").attr("fill", "#66717b")
      .text(TEXT.playbookMissingDataLegend);
  }
  if (compareRisk) {
    const validRisk = riskSeries.filter(d => Number.isFinite(d.q1) && Number.isFinite(d.q3));
    const limitsByMonth = new Map(validRisk.map(d => [d.month, {low: d.q1 - .5 * (d.q3 - d.q1), high: d.q3 + .5 * (d.q3 - d.q1)}]));
    const outside = observed.some(d => {
      const limits = limitsByMonth.get(d.month);
      return limits && (d.value < limits.low || d.value > limits.high);
    });
    const zoomLow = d3.min(validRisk, d => d.q1 - .5 * (d.q3 - d.q1));
    const zoomHigh = d3.max(validRisk, d => d.q3 + .5 * (d.q3 - d.q1));
    if (outside && Number.isFinite(zoomLow) && Number.isFinite(zoomHigh) && zoomHigh > zoomLow) {
      const zoomY = y.copy().domain([zoomLow, zoomHigh]).nice();
      const transition = svg.transition().delay(180).duration(900).ease(d3.easeCubicInOut);
      yAxis.transition(transition).call(d3.axisLeft(zoomY).ticks(6).tickFormat(fmtPct));
      grid.transition(transition).call(d3.axisLeft(zoomY).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
      countyLine.transition(transition).attr("d", d3.line().defined(d => d.value != null).x(d => x(d.date)).y(d => zoomY(d.value)));
      riskBand?.transition(transition).attr("d", d3.area().defined(d => d.q1 != null && d.q3 != null).x(d => x(parseMonth(d.month))).y0(d => zoomY(d.q1)).y1(d => zoomY(d.q3)));
      riskMedian?.transition(transition).attr("d", d3.line().defined(d => d.median != null).x(d => x(parseMonth(d.month))).y(d => zoomY(d.median)));
    }
  }
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

function eventWindowTrendStats(points) {
  const valid = points.filter(d => Number.isFinite(d.month) && Number.isFinite(d.value));
  if (valid.length < 3) return {slope: null, r2: null};
  const meanMonth = d3.mean(valid, d => d.month);
  const meanValue = d3.mean(valid, d => d.value);
  const monthVariance = d3.sum(valid, d => (d.month - meanMonth) ** 2);
  const valueVariance = d3.sum(valid, d => (d.value - meanValue) ** 2);
  if (!monthVariance || !valueVariance) return {slope: 0, r2: 0};
  const covariance = d3.sum(valid, d => (d.month - meanMonth) * (d.value - meanValue));
  return {
    slope: covariance / monthVariance,
    r2: (covariance * covariance) / (monthVariance * valueVariance),
  };
}

function qualitativeRelation(difference, threshold) {
  if (Math.abs(difference) <= threshold) return "similar to";
  return difference > 0 ? "higher than" : "lower than";
}

function alignmentExtent(observed, expected, threshold) {
  if (!Number.isFinite(observed) || !Number.isFinite(expected)) return "could not be compared";
  const distance = Math.abs(observed - expected);
  if (distance <= threshold) return "closely aligns";
  if (distance <= threshold * 2) return "partially aligns";
  return "does not align";
}

function describePpsfChange(delta, terms) {
  if (Math.abs(delta) <= .005) return "remained broadly steady";
  const direction = delta > 0 ? terms.increased : terms.declined;
  return `${direction} by about ${Math.abs(delta * 100).toFixed(1)} percentage points`;
}

function renderPlaybookCommentary(county, history, events) {
  const container = d3.select("#playbook-event-commentary");
  const risk = county.hazards?.overall?.rating || county.riskRating || "Unknown";
  const copy = TEXT.playbookTakeaways;
  const terms = TEXT.playbookTakeawayTerms;
  const countyLabel = countyDisplayName(county);
  const groupExpectation = riskGroupEventExpectation(risk, copy, terms);
  const profile = playbookFeatureProfile(county);
  if (!events.length) {
    container.attr("class", "playbook-commentary neutral");
    container.html(fillTextTemplate(copy.noEvents, {
      county: countyLabel,
      countyContext: profile.subgroupName
        ? `a ${risk} Risk county in the ${subgroupProseName(profile.subgroupName)} range`
        : `a ${risk} Risk county`,
      expectation: groupExpectation.groupBehavior || "follow the broader risk-group pattern",
    }));
    return;
  }

  const eventWindowPoints = events.flatMap(event => {
    const preStart = d3.utcMonth.offset(event.startDate, -12);
    const postEnd = d3.utcMonth.offset(event.endDate, 36);
    return history
      .filter(d => d.value != null && !d.interpolated && d.date >= preStart && d.date <= postEnd)
      .map(d => ({month: d3.utcMonth.count(event.startDate, d.date), value: d.value}));
  });
  const groupRows = (DATA.eventWindows?.windowA?.byRating || [])
    .filter(row => row.riskRating === risk && row.month >= -12 && row.month <= 36 && row.median != null);
  const eventWindowValues = eventWindowPoints.map(d => d.value);
  const groupValues = groupRows.map(row => row.median);
  const countyBefore = eventWindowPoints.filter(d => d.month >= -12 && d.month < 0).map(d => d.value);
  const countyAfter = eventWindowPoints.filter(d => d.month >= 25 && d.month <= 36).map(d => d.value);
  if (!eventWindowValues.length || !groupValues.length || !countyBefore.length || !countyAfter.length || groupExpectation.delta == null) {
    container.attr("class", "playbook-commentary neutral");
    container.html(fillTextTemplate(copy.insufficientHistory, {county: countyLabel}));
    return;
  }
  const countyMedian = d3.median(eventWindowValues);
  const groupMedian = d3.median(groupValues);
  const typicalGroupIqr = d3.median(groupRows.map(row => Math.max(0, (row.q3 ?? row.median) - (row.q1 ?? row.median)))) || 0;
  const countyTrend = eventWindowTrendStats(eventWindowPoints);
  const countyDeviation = d3.deviation(eventWindowValues) || 0;
  const groupDeviation = d3.deviation(groupValues) || 0;
  const tooVolatile = (
    eventWindowValues.length >= 12
    && (countyTrend.r2 ?? 0) < .08
    && countyDeviation > Math.max(.04, groupDeviation * 1.75, typicalGroupIqr * 1.25)
  );
  container.attr("class", "playbook-commentary");
  if (tooVolatile) {
    container.html(fillTextTemplate(copy.volatileEventSummary, {
      county: countyLabel,
      risk,
      subgroupContext: profile.subgroupName
        ? `${subgroupProseName(profile.subgroupName)} subgroup shown at left`
        : "feature subgroup because sufficient feature data were unavailable",
    }));
    return;
  }
  const levelThreshold = Math.max(.01, typicalGroupIqr * .5);
  const countyChange = d3.median(countyAfter) - d3.median(countyBefore);
  const riskAlignment = alignmentExtent(countyChange, groupExpectation.delta, levelThreshold);
  const riskTargets = (DATA.features.countyRowsByRisk[risk] || []).map(d => d.target).filter(Number.isFinite);
  const riskTargetMedian = riskTargets.length ? d3.median(riskTargets) : null;
  const subgroupExpectedGap = profile.subgroup && Number.isFinite(profile.subgroup.targetMedian) && Number.isFinite(riskTargetMedian)
    ? profile.subgroup.targetMedian - riskTargetMedian
    : null;
  const observedGap = countyMedian - groupMedian;
  const template = Number.isFinite(subgroupExpectedGap)
    ? copy.eventAlignmentSummary
    : copy.eventAlignmentWithoutSubgroup;
  container.html(fillTextTemplate(template, {
    county: countyLabel,
    risk,
    riskAlignment,
    countyChange: describePpsfChange(countyChange, terms),
    groupChange: groupExpectation.groupBehavior,
    subgroup: subgroupProseName(profile.subgroupName),
    subgroupAlignment: alignmentExtent(observedGap, subgroupExpectedGap, levelThreshold),
    countyLevel: qualitativeRelation(observedGap, levelThreshold),
    subgroupLevel: qualitativeRelation(subgroupExpectedGap, levelThreshold),
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
  const playbookPanel = document.querySelector("#playbook .panel");
  playbookPanel?.classList.add("has-county-selection");
  d3.select("#playbook-selected-county-name").style("display", "block").text(countyDisplayName(county));
  renderPlaybookHazards(county);
  renderPlaybookFeatureSummary(county);
  drawPlaybookMap("#playbook-profile-map", county, true, false);
}

function goToPlaybookProfile() {
  const section = document.querySelector("#playbook");
  if (!section) return;
  const viewport = window.innerHeight || 1;
  window.scrollTo({
    top: section.offsetTop + viewport * 2,
    behavior: "smooth",
  });
}

function goToPlaybookSearch() {
  const section = document.querySelector("#playbook");
  if (!section) return;
  window.scrollTo({top: section.offsetTop + (window.innerHeight || 1), behavior: "smooth"});
}

function playbookHistoryRows(county) {
  const parseMonth = d3.utcParse("%Y-%m");
  const months = DATA.playbook.monthlyHistoryMonths || [];
  const values = (DATA.playbook.monthlyHistoryValuesByFips || {})[county.fips] || [];
  return interpolateInternalHistory(months.map((month, index) => ({date: parseMonth(month), month, value: values[index] ?? null, interpolated: false})));
}

function renderPlaybookFrame() {
  const state = document.querySelector("#playbook .story-stage")?.dataset.storyState || "search";
  const county = playbookCountyByFips.get(selectedCountyFips);
  if (state === "search" || !county) {
    drawPlaybookMap("#county-selection-map", county || null, false, true);
    return;
  }
  d3.select("#playbook-selected-county-name").style("display", "block").text(countyDisplayName(county));
  renderPlaybookHazards(county);
  renderPlaybookFeatureSummary(county, state === "history-events" || state === "history-compare");
  if (state === "profile") {
    drawPlaybookMap("#playbook-profile-map", county, true, false);
    return;
  }
  const compareRisk = state === "history-compare";
  drawPlaybookHistory(county, compareRisk);
  if (compareRisk) renderPlaybookCommentary(county, playbookHistoryRows(county), playbookEvents(county));
  else renderPlaybookEventList(county);
}

function initPlaybook() {
  if (!DATA.playbook?.available) {
    d3.select("#county-search").property("disabled", true).property("placeholder", DATA.playbook?.message || "County data unavailable");
    return;
  }
  drawPlaybookMap("#county-selection-map", null, false, true);
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
      .on("click", (event, d) => {
        selectPlaybookCounty(d);
        goToPlaybookProfile();
        input.property("value", "");
        results.style("display", "none").html("");
      });
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
  d3.select("#playbook-back-to-search").on("click", goToPlaybookSearch);
}

const STORY_CONFIG = {
  pricing: [
    {state: "title"},
    {state: "card-main"},
    {state: "takeaway-0", takeaway: "#score-scatter-takeaway", segment: 0},
    {state: "takeaway-1", takeaway: "#score-scatter-takeaway", segment: 1},
  ],
  "pricing-grouping": [
    {state: "title"},
    {state: "copy"},
    {state: "card-main", ratingSequence: true},
    {state: "takeaway-0", takeaway: "#pricing-takeaway", segment: 0},
    {state: "takeaway-1", takeaway: "#pricing-takeaway", segment: 1},
  ],
  events: [
    {state: "title"},
    {state: "copy"},
    {state: "card-short", eventWindow: "A"},
    {state: "takeaway-short", takeaway: "#event-window-takeaway", eventWindow: "A"},
    {state: "takeaway-future", takeaway: "#event-future-prompt", eventWindow: "A"},
    {state: "card-long", eventWindow: "B"},
    {state: "takeaway-long", takeaway: "#event-window-takeaway", eventWindow: "B"},
    {state: "takeaway-0", takeaway: "#event-takeaway", segment: 0, eventWindow: "B"},
    {state: "takeaway-1", takeaway: "#event-takeaway", segment: 1, eventWindow: "B"},
  ],
  features: [
    {state: "title"},
    {state: "copy"},
    {state: "feature-frame-1"},
    {state: "takeaway-feature", takeaway: "#feature-takeaway"},
    {state: "feature-frame-2"},
    {state: "feature-frame-3"},
  ],
  playbook: [
    {state: "title"},
    {state: "search"},
    {state: "profile"},
    {state: "history-events"},
    {state: "history-compare"},
  ],
};

const takeawayTransitionTimers = new WeakMap();

function syncTakeawaySpace(section, takeaway) {
  const panel = section.querySelector(":scope > .story-stage > .panel");
  if (!panel) return;
  if (!takeaway) {
    panel.style.removeProperty("--takeaway-space");
    return;
  }
  requestAnimationFrame(() => {
    const height = Math.ceil(takeaway.getBoundingClientRect().height);
    if (height > 0) panel.style.setProperty("--takeaway-space", `${height}px`);
  });
}

function activateStoryTakeaway(section, step, direction) {
  clearTimeout(takeawayTransitionTimers.get(section));
  section.querySelectorAll(".takeaway, .takeaway-section").forEach(element => {
    element.classList.remove(
      "story-outgoing-takeaway",
      "story-slide-out-up",
      "story-slide-in-up",
      "story-slide-out-down",
      "story-slide-in-down",
    );
  });
  const previousTakeaway = section.querySelector(".takeaway.story-active-takeaway");
  const previousSegment = previousTakeaway?.querySelector(".takeaway-section.story-active-segment");
  const nextTakeaway = step.takeaway ? section.querySelector(step.takeaway) : null;
  const nextSegments = nextTakeaway?.querySelectorAll(".takeaway-section") || [];
  const nextSegment = nextSegments.length
    ? nextSegments[Math.min(step.segment || 0, nextSegments.length - 1)]
    : null;
  const previousContent = previousSegment || previousTakeaway;
  const nextContent = nextSegment || nextTakeaway;

  if (!nextTakeaway) {
    if (!previousContent) return;
    previousTakeaway.classList.add("story-outgoing-takeaway");
    previousContent.classList.add(direction >= 0 ? "story-slide-out-up" : "story-slide-out-down");
    const transitionTimer = setTimeout(() => {
      previousContent.classList.remove("story-active-segment", "story-slide-out-up", "story-slide-out-down");
      previousTakeaway.classList.remove("story-active-takeaway", "story-outgoing-takeaway");
      syncTakeawaySpace(section, null);
    }, 430);
    takeawayTransitionTimers.set(section, transitionTimer);
    return;
  }

  if (!previousContent) {
    section.querySelectorAll(".takeaway").forEach(takeaway => takeaway.classList.remove("story-active-takeaway"));
    section.querySelectorAll(".takeaway-section").forEach(segment => segment.classList.remove("story-active-segment"));
    nextTakeaway.classList.add("story-active-takeaway");
    if (nextSegment) nextSegment.classList.add("story-active-segment");
    syncTakeawaySpace(section, nextTakeaway);
    nextContent.classList.add(direction >= 0 ? "story-slide-in-up" : "story-slide-in-down");
    const transitionTimer = setTimeout(() => {
      nextContent.classList.remove("story-slide-in-up", "story-slide-in-down");
    }, 430);
    takeawayTransitionTimers.set(section, transitionTimer);
    return;
  }

  if (previousTakeaway === nextTakeaway && previousContent !== nextContent) {
    previousContent.classList.remove("story-active-segment");
    nextTakeaway.classList.add("story-active-takeaway");
    if (nextSegment) nextSegment.classList.add("story-active-segment");
    syncTakeawaySpace(section, nextTakeaway);
    nextContent.classList.add(direction >= 0 ? "story-slide-in-up" : "story-slide-in-down");
    const transitionTimer = setTimeout(() => {
      nextContent.classList.remove("story-slide-in-up", "story-slide-in-down");
      syncTakeawaySpace(section, nextTakeaway);
    }, 430);
    takeawayTransitionTimers.set(section, transitionTimer);
    return;
  }

  nextTakeaway.classList.add("story-active-takeaway");
  if (nextSegment) nextSegment.classList.add("story-active-segment");
  syncTakeawaySpace(section, nextTakeaway);
  if (previousContent === nextContent) return;
  previousTakeaway.classList.add("story-outgoing-takeaway");
  const movingDownPage = direction >= 0;
  previousContent.classList.add(movingDownPage ? "story-slide-out-up" : "story-slide-out-down");
  nextContent.classList.add(movingDownPage ? "story-slide-in-up" : "story-slide-in-down");
  const transitionTimer = setTimeout(() => {
    previousContent.classList.remove(
      "story-active-segment",
      "story-slide-out-up",
      "story-slide-out-down",
    );
    if (previousTakeaway !== nextTakeaway) {
      previousTakeaway.classList.remove("story-active-takeaway");
    }
    previousTakeaway.classList.remove("story-outgoing-takeaway");
    nextContent.classList.remove("story-slide-in-up", "story-slide-in-down");
    syncTakeawaySpace(section, nextTakeaway);
  }, 430);
  takeawayTransitionTimers.set(section, transitionTimer);
}

function applyStoryStep(section, step, index) {
  const stage = section.querySelector(".story-stage");
  if (!stage) return;
  const effectiveStep = (
    section.id === "playbook"
    && step.state !== "title"
    && step.state !== "search"
    && !selectedCountyFips
  )
    ? {...step, state: "search"}
    : step;
  if (
    stage.dataset.storyIndex === String(index)
    && stage.dataset.storyState === effectiveStep.state
  ) return;
  const previousState = stage.dataset.storyState;
  const previousIndex = Number(stage.dataset.storyIndex);
  const direction = Number.isFinite(previousIndex) ? Math.sign(index - previousIndex) : 1;
  stage.dataset.storyIndex = String(index);
  stage.dataset.storyState = effectiveStep.state;
  stage.dataset.storyDirection = direction < 0 ? "backward" : "forward";
  stage.classList.remove("story-step-forward", "story-step-backward");
  void stage.offsetWidth;
  stage.classList.add(direction < 0 ? "story-step-backward" : "story-step-forward");
  activateStoryTakeaway(section, effectiveStep, direction);
  const panel = stage.querySelector(".panel");
  const lockInnerScroll = (
    effectiveStep.state.startsWith("takeaway")
    || section.id === "features"
    || section.id === "playbook"
  );
  panel?.classList.toggle("inner-scroll-locked", lockInnerScroll);
  if (lockInnerScroll && panel) panel.scrollTop = 0;

  if (effectiveStep.eventWindow && activeEventWindow !== effectiveStep.eventWindow) {
    activeEventWindow = effectiveStep.eventWindow;
    if (eventSectionRendered) renderEventSection();
  }
  if (effectiveStep.ratingSequence && previousState !== effectiveStep.state && ratingSectionRendered) {
    startRatingSequence();
  }
  if (
    section.id === "pricing-grouping"
    && effectiveStep.state.startsWith("takeaway")
  ) {
    pauseRatingSequence();
    d3.select("#rating-sequence-callout").classed("visible", false);
  }
  if (section.id === "features" && featureSectionRendered) {
    drawFeatureHeatmaps();
  }
  if (section.id === "playbook" && playbookSectionRendered) {
    renderPlaybookFrame();
  }
}

function storyNavigationStops() {
  const viewport = window.innerHeight || 1;
  const stops = [document.querySelector(".hero")?.offsetTop || 0];
  for (const [id, config] of Object.entries(STORY_CONFIG)) {
    const section = document.getElementById(id);
    if (!section) continue;
    config.forEach((step, index) => {
      stops.push(section.offsetTop + index * viewport);
    });
  }
  return [...new Set(stops.map(stop => Math.round(stop)))].sort((a, b) => a - b);
}

function updateStoryNavigation() {
  const stops = storyNavigationStops();
  const current = window.scrollY;
  const tolerance = (window.innerHeight || 1) * 0.2;
  document.getElementById("story-prev").disabled = !stops.some(stop => stop < current - tolerance);
  document.getElementById("story-next").disabled = !stops.some(stop => stop > current + tolerance);
}

function navigateStory(direction) {
  const stops = storyNavigationStops();
  const current = window.scrollY;
  const tolerance = (window.innerHeight || 1) * 0.2;
  const target = direction > 0
    ? stops.find(stop => stop > current + tolerance)
    : [...stops].reverse().find(stop => stop < current - tolerance);
  if (target == null) return;
  window.scrollTo({top: target, behavior: "smooth"});
}

function initStoryEdgeNavigation() {
  const previous = document.getElementById("story-prev");
  const next = document.getElementById("story-next");
  const hide = () => {
    previous.classList.remove("edge-visible");
    next.classList.remove("edge-visible");
  };
  document.addEventListener("pointermove", event => {
    const edgeSize = 64;
    previous.classList.toggle("edge-visible", event.clientY <= edgeSize);
    next.classList.toggle(
      "edge-visible",
      event.clientY >= (window.innerHeight || 1) - edgeSize,
    );
  }, {passive: true});
  document.documentElement.addEventListener("mouseleave", hide);
  window.addEventListener("blur", hide);
}

function initPanelScrollRouting() {
  const standardPanels = [...document.querySelectorAll(".story-stage > .panel")]
    .filter(panel => !panel.closest("#playbook"));
  standardPanels.forEach(scrollContainer => {
    scrollContainer.addEventListener("wheel", event => {
      const ownerPanel = scrollContainer.closest(".panel");
      if (ownerPanel?.classList.contains("inner-scroll-locked")) return;
      const canScroll = scrollContainer.scrollHeight > scrollContainer.clientHeight + 1;
      if (!canScroll) return;
      const atTop = scrollContainer.scrollTop <= 0;
      const atBottom = (
        scrollContainer.scrollTop + scrollContainer.clientHeight
        >= scrollContainer.scrollHeight - 1
      );
      const leavingCard = (event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom);
      if (leavingCard) {
        event.preventDefault();
        window.scrollBy({top: event.deltaY, behavior: "auto"});
      } else {
        event.stopPropagation();
      }
    }, {passive: false});
  });
}

function updateStoryFromScroll() {
  const viewport = window.innerHeight || 1;
  document.querySelectorAll(".slide[data-story-ready='true']").forEach(section => {
    const config = STORY_CONFIG[section.id];
    const relative = (window.scrollY - section.offsetTop + viewport * .42) / viewport;
    const index = Math.max(0, Math.min(config.length - 1, Math.floor(relative)));
    applyStoryStep(section, config[index], index);
  });
  updateStoryNavigation();
}

function initScrollStory() {
  Object.entries(STORY_CONFIG).forEach(([id, config]) => {
    const section = document.getElementById(id);
    if (!section || section.dataset.storyReady) return;
    const stage = document.createElement("div");
    stage.className = "story-stage";
    while (section.firstChild) stage.appendChild(section.firstChild);
    section.appendChild(stage);
    section.style.setProperty("--story-steps", config.length);
    section.dataset.storyReady = "true";
  });
  let scheduled = false;
  const requestUpdate = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      updateStoryFromScroll();
    });
  };
  window.addEventListener("scroll", requestUpdate, {passive: true});
  window.addEventListener("resize", requestUpdate);
  document.getElementById("story-prev").addEventListener("click", () => navigateStory(-1));
  document.getElementById("story-next").addEventListener("click", () => navigateStory(1));
  initStoryEdgeNavigation();
  initPanelScrollRouting();
  updateStoryFromScroll();
}

// ---- bootstrap ----
hydrateText();
initScrollStory();
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
  startRatingSequence();
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
  renderPlaybookFrame();
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
      renderPlaybookFrame();
    }
    document.querySelectorAll(".slide[data-story-ready='true']").forEach(section => {
      syncTakeawaySpace(section, section.querySelector(".takeaway.story-active-takeaway"));
    });
  }, 150);
});
</script>
</body>
</html>"""


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        price_risk = build_price_risk(con)
        features = build_feature_payload(con)
        event_windows = build_event_windows(con)
        playbook = build_county_playbook_data(con)

    state_geometries = load_state_geometries()
    playbook_fips = {county["fips"] for county in playbook["counties"]}
    geojson = build_geojson(
        playbook_fips,
        state_geometries,
    )
    state_geojson = build_state_geojson(playbook_fips, state_geometries)
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
        "stateGeojson": state_geojson,
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
