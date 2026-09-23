from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
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
STATE_AND_DC_FIPS = {
    "01",
    "02",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "53",
    "54",
    "55",
    "56",
}


@dataclass(frozen=True)
class PageEventWindowContext:
    """Maximum page event window and the inputs used to derive its display subsets."""

    analysis_start: pd.Timestamp
    analysis_end: pd.Timestamp
    events: pd.DataFrame
    nri: pd.DataFrame
    affected: pd.DataFrame


HAZARDS = [
    {
        "key": "overall",
        "label": "Overall NRI",
        "score": "risk_score",
        "rating": "risk_rating",
    },
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
    (
        "Economic",
        "Income factors",
        "dividends_interest_rent_per_capita_usd",
        "currency",
    ),
    ("Economic", "Income factors", "transfer_receipts_per_capita_usd", "currency"),
    ("Economic", "Cost factors", "homeowners_insurance_pct_income", "percent"),
    ("Economic", "Cost factors", "property_taxes_pct_income", "percent"),
    ("Economic", "Cost factors", "utilities_pct_income", "percent"),
    ("Economic", "Cost factors", "owner_cost_burden_30pct_plus_pct", "percent"),
    ("Economic", "Employment", "unemployment_rate_pct", "percent"),
    ("Demographic", "Population trend", "net_migration_rate_pct", "percent"),
    ("Demographic", "Population vulnerability factors", "age_65_plus_pct", "percent"),
    (
        "Demographic",
        "Population vulnerability factors",
        "communication_barrier_pct",
        "percent",
    ),
    ("Demographic", "Population vulnerability factors", "disability_pct", "percent"),
]


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False), errors="coerce"
    )


def serialize_number(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def rating_clean(value: object) -> str | None:
    if pd.isna(value):
        return None
    return RISK_MAP.get(str(value), str(value))


@lru_cache(maxsize=1)
def current_county_fips() -> frozenset[str]:
    """Return present-day state/DC county FIPS from the page's boundary file."""

    payload = json.loads(COUNTIES_PATH.read_text(encoding="utf-8"))
    return frozenset(
        str(feature.get("properties", {}).get("GEOID", "")).zfill(5)
        for feature in payload.get("features", [])
        if str(feature.get("properties", {}).get("GEOID", ""))[:2] in STATE_AND_DC_FIPS
        and not str(feature.get("properties", {}).get("GEOID", "")).endswith("000")
    )


def filter_current_state_county_events(
    events: pd.DataFrame,
    current_nri_fips: pd.Series | set[str] | list[str],
) -> pd.DataFrame:
    """Keep events assigned to present-day counties in the 50 states or D.C."""

    if events.empty:
        return events.copy()
    valid_nri_fips = {
        str(fips).zfill(5) for fips in current_nri_fips if pd.notna(fips)
    } & current_county_fips()
    event_fips = events["fips"].astype(str).str.zfill(5)
    eligible = (
        event_fips.str[:2].isin(STATE_AND_DC_FIPS)
        & ~event_fips.str.endswith("000")
        & event_fips.isin(valid_nri_fips)
    )
    filtered = events.loc[eligible].copy()
    filtered["fips"] = event_fips.loc[eligible]
    return filtered


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


def build_max_affected_event_context(
    con: duckdb.DuckDBPyConnection,
) -> PageEventWindowContext:
    """Match events to housing once for the page's maximum 60-month horizon."""

    analysis_start, analysis_end = latest_complete_calendar_window(con)
    events = load_disaster_events(con)
    events = events.loc[
        events["event_start_month"].ge(analysis_start)
        & events["event_start_month"].lt(analysis_end)
    ].copy()
    nri = con.execute(
        "SELECT fips, risk_rating FROM mart.nri_county_risk WHERE fips IS NOT NULL"
    ).df()
    nri["fips"] = nri["fips"].astype(str).str.zfill(5)
    nri = nri.loc[
        nri["fips"].str[:2].isin(STATE_AND_DC_FIPS) & ~nri["fips"].str.endswith("000")
    ].copy()
    nri["riskRating"] = nri["risk_rating"].map(rating_clean)
    events = filter_current_state_county_events(events, nri["fips"])

    housing = load_redfin_county_monthly(con)
    for column in [
        "median_ppsf_yoy",
        "avg_sale_to_list_yoy",
        "homes_sold_yoy",
        "inventory_yoy",
        "housing_market_index",
    ]:
        if column in housing:
            values = pd.to_numeric(housing[column], errors="coerce")
            housing[column] = values.mask(values.le(-888888000))
    affected = build_affected_event_windows(
        events,
        housing,
        pre_event_months=12,
        post_event_months=60,
    )
    return PageEventWindowContext(
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        events=events,
        nri=nri,
        affected=affected,
    )


def weighted_bucket_average(
    frame: pd.DataFrame,
    buckets: list[tuple[str, float]],
    *,
    zero_cols: list[str] | None = None,
) -> pd.Series:
    total = pd.Series(0.0, index=frame.index)
    weighted = pd.Series(0.0, index=frame.index)
    for column in zero_cols or []:
        if column in frame:
            total = total.add(
                pd.to_numeric(frame[column], errors="coerce").fillna(0), fill_value=0
            )
    for column, midpoint in buckets:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
            total = total.add(values, fill_value=0)
            weighted = weighted.add(values * midpoint, fill_value=0)
    return weighted.where(total > 0) / total.where(total > 0)


def build_price_risk(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    analysis_start, analysis_end = latest_complete_calendar_window(con)
    latest_year_start = analysis_end - pd.DateOffset(years=1)
    state_fips_sql = ", ".join(f"'{code}'" for code in sorted(STATE_AND_DC_FIPS))
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
    df["avg_median_ppsf_yoy"] = pd.to_numeric(
        df["avg_median_ppsf_yoy"], errors="coerce"
    )
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
                "county": row.county_label
                if pd.notna(row.county_label)
                else f"{row.COUNTY}, {row.STATEABBRV}",
                "state": row.state_code if pd.notna(row.state_code) else row.STATEABBRV,
                "hazards": hazards,
            }
        )
    history = con.execute(
        f"""
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
              AND substr(lpad(cast(r.fips AS VARCHAR), 5, '0'), 1, 2) IN ({state_fips_sql})
              AND substr(lpad(cast(r.fips AS VARCHAR), 5, '0'), 3, 3) <> '000'
            GROUP BY r.fips, date_trunc('month', r.period_begin)
        )
        SELECT monthly.*
        FROM monthly
        ORDER BY fips, month
        """,
        [analysis_start, analysis_end],
    ).df()
    history["fips"] = history["fips"].astype(str).str.zfill(5)
    history["median_ppsf_yoy"] = pd.to_numeric(
        history["median_ppsf_yoy"], errors="coerce"
    )
    history = history.loc[history["fips"].isin(current_county_fips())].copy()

    # Overall NRI is the only risk measure used by the story.
    history_for_counties = history.copy()
    history = history.merge(nri[["fips"] + hazard_cols], on="fips", how="left")
    history["riskRating"] = history["risk_rating"].map(rating_clean)
    required_history_months = pd.date_range(
        analysis_start, analysis_end - pd.DateOffset(months=1), freq="MS"
    )
    history_for_rating = filter_complete_event_window_lines(
        history,
        x_col="month",
        line_col="fips",
        metric_col="median_ppsf_yoy",
        required_x_values=required_history_months,
    ).dropna(subset=["riskRating"])

    grouped = (
        history_for_rating.groupby(["riskRating", "month"], observed=False)[
            "median_ppsf_yoy"
        ]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3"})
    )
    rating_history = [
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
    history_months = list(
        pd.date_range(analysis_start, analysis_end - pd.DateOffset(months=1), freq="MS")
    )
    history_month_labels = [
        pd.Timestamp(month).strftime("%Y-%m-%d") for month in history_months
    ]
    county_history_series = []
    for fips, county_history in history_for_counties.groupby("fips", sort=False):
        if not county_history["median_ppsf_yoy"].notna().any():
            continue
        first = county_history.iloc[0]
        county_history = county_history.set_index("month").reindex(history_months)
        county_history_series.append(
            {
                "fips": fips,
                "county": first.county_label,
                "state": first.state_code,
                "values": [
                    serialize_number(value, 5)
                    for value in county_history["median_ppsf_yoy"]
                ],
            }
        )
    history_cap_lower = history_for_counties["median_ppsf_yoy"].quantile(0.10)
    history_cap_upper = history_for_counties["median_ppsf_yoy"].quantile(0.90)

    return {
        "hazards": [{"key": h["key"], "label": h["label"]} for h in HAZARDS],
        "counties": counties,
        "countyHistoryMonths": history_month_labels,
        "countyHistorySeries": county_history_series,
        "ratingHistory": rating_history,
        "summary": {
            "analysisStart": analysis_start.strftime("%Y-%m-%d"),
            "analysisEnd": (analysis_end - pd.DateOffset(months=1)).strftime(
                "%Y-%m-%d"
            ),
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
                "AL",
                "AK",
                "AZ",
                "AR",
                "CA",
                "CO",
                "CT",
                "DE",
                "DC",
                "FL",
                "GA",
                "HI",
                "ID",
                "IL",
                "IN",
                "IA",
                "KS",
                "KY",
                "LA",
                "ME",
                "MD",
                "MA",
                "MI",
                "MN",
                "MS",
                "MO",
                "MT",
                "NE",
                "NV",
                "NH",
                "NJ",
                "NM",
                "NY",
                "NC",
                "ND",
                "OH",
                "OK",
                "OR",
                "PA",
                "RI",
                "SC",
                "SD",
                "TN",
                "TX",
                "UT",
                "VT",
                "VA",
                "WA",
                "WV",
                "WI",
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
            geometry = shape(feature["geometry"]).simplify(
                tolerance, preserve_topology=True
            )
            geometry = geometry.intersection(state_record[1])
            if isinstance(geometry, GeometryCollection):
                geometry = unary_union(
                    [
                        part
                        for part in geometry.geoms
                        if isinstance(part, (Polygon, MultiPolygon))
                    ]
                )
            if not geometry.is_empty:
                if isinstance(geometry, Polygon):
                    geometry = orient(geometry, sign=-1.0)
                elif isinstance(geometry, MultiPolygon):
                    geometry = MultiPolygon(
                        [orient(part, sign=-1.0) for part in geometry.geoms]
                    )
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"fips": fips},
                        "geometry": mapping(geometry),
                    }
                )
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
                & group["line_id"]
                .astype(str)
                .str.startswith(f"fema:{specification['source_event_id']}:")
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
            rows = complete.loc[
                complete["line_id"].eq(candidate["line_id"])
            ].sort_values(anchor_col)
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
                    "focusPosition": specification["position"]
                    if specification
                    else None,
                    "withinPeerIqrTolerance": str(candidate["line_id"])
                    in eligible_line_ids,
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


def aggregate_lines(
    frame: pd.DataFrame, group_cols: list[str], metric: str, annual: bool = False
) -> list[dict[str, object]]:
    if frame.empty:
        return []

    if annual:
        # For annual data, convert event_window_month to event_window_year
        frame_copy = frame.copy()
        frame_copy["event_window_year"] = (
            (frame_copy["event_window_month"] / 12).round().astype(int)
        )
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
                "month": int(
                    row.event_window_year * 12
                ),  # Convert back to months for consistency
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

    Each frame includes only affected county-event trajectories with a valid
    observation in every required month.
    """
    required = event_window_months(pre_months, post_months)
    window_rows = affected.loc[
        affected[anchor_col].isin(required) & affected["line_id"].notna()
    ].copy()
    complete = filter_complete_event_window_lines(
        window_rows,
        x_col=anchor_col,
        line_col="line_id",
        metric_col=metric,
        required_x_values=required,
    )
    complete = complete.merge(nri[["fips", "riskRating"]], on="fips", how="left")
    complete_for_agg = complete.copy()
    if anchor_col != "event_window_month":
        complete_for_agg["event_window_month"] = complete_for_agg[anchor_col]
    by_rating = aggregate_lines(
        complete_for_agg.dropna(subset=["riskRating"]), ["riskRating"], metric
    )
    affected_counties = (
        complete.dropna(subset=["riskRating"])[["fips", "riskRating"]]
        .drop_duplicates()
        .groupby(["fips", "riskRating"], as_index=False)
        .size()
    )
    risk_counts = (
        complete.dropna(subset=["riskRating"])
        .groupby("riskRating", dropna=True)["fips"]
        .nunique()
    )

    # Compute per-county-event average metric over the window, then percentile within risk group.
    line_avg = (
        complete.dropna(subset=[metric, "riskRating"])
        .groupby(
            ["line_id", "fips", "county_label", "state_code", "riskRating"],
            as_index=False,
        )[metric]
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
            lambda row: row["q1"]
            - 0.5 * max_width_by_risk.get(row["riskRating"], np.nan),
            axis=1,
        )
        bands["upper_allowed"] = bands.apply(
            lambda row: row["q3"]
            + 0.5 * max_width_by_risk.get(row["riskRating"], np.nan),
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
            band_join["riskRating"].map(max_width_by_risk).abs().clip(lower=0.01)
        )
        band_join["normalized_band_deviation"] = (
            band_join["outside_sample_band_distance"] / band_join["sample_band_scale"]
        )
        line_band_fit = band_join.groupby("line_id", as_index=False).agg(
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
        distance_join["safe_iqr_width"] = (
            distance_join["iqr_width"].abs().clip(lower=0.01)
        )
        distance_join["standardized_line_gap"] = (
            distance_join["line_gap"] / distance_join["safe_iqr_width"]
        )
        distance_join["above_group_median"] = distance_join["line_gap"].gt(0)
        line_distance = distance_join.groupby("line_id", as_index=False).agg(
            mean_line_gap=("line_gap", "mean"),
            mean_abs_line_gap=("line_gap", lambda values: values.abs().mean()),
            median_line_gap=("line_gap", "median"),
            median_standardized_gap=("standardized_line_gap", "median"),
            mean_abs_standardized_gap=(
                "standardized_line_gap",
                lambda values: values.abs().mean(),
            ),
            above_median_share=("above_group_median", "mean"),
            median_iqr_width=("iqr_width", "median"),
        )
        line_distance["distance_threshold"] = np.maximum(
            0.01,
            line_distance["median_iqr_width"].fillna(0).mul(0.5),
        )
        line_distance["significantly_separated"] = (
            line_distance["median_standardized_gap"].abs().ge(0.5)
        )
        line_distance["directionally_consistent"] = line_distance[
            "above_median_share"
        ].ge(0.7) | line_distance["above_median_share"].le(0.3)

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
        group = (
            line_avg.loc[line_avg["riskRating"].eq(risk)]
            .dropna(subset=["pct_rank"])
            .copy()
        )
        if eligible_feature_fips_by_risk is not None:
            eligible_fips = eligible_feature_fips_by_risk.get(risk, set())
            group = group.loc[group["fips"].astype(str).isin(eligible_fips)].copy()
        if group.empty:
            continue
        if eligible_line_ids:
            group = group.loc[
                group["line_id"].astype(str).isin(eligible_line_ids)
            ].copy()
        if group.empty:
            continue
        # Sample counties shown in the "What Sets Apart..." plot should remain
        # visually interpretable: require all median PPSF YoY values to stay within +/-100%.
        group = group.loc[
            group["max_metric"].le(100) & group["min_metric"].ge(-100)
        ].copy()
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
            above_values = risk_trajectories.loc[above_candidate["line_id"]].to_numpy(
                dtype=float
            )
            for below_candidate in below_candidates.to_dict("records"):
                if str(below_candidate["line_id"]) == str(
                    above_candidate["line_id"]
                ) or str(below_candidate["fips"]) == str(above_candidate["fips"]):
                    continue
                below_values = risk_trajectories.loc[
                    below_candidate["line_id"]
                ].to_numpy(dtype=float)
                pair_delta = above_values - below_values
                pair_contrast = float(np.nanmedian(np.abs(pair_delta)))
                positive_share = float(np.nanmean(pair_delta > 0))
                negative_share = float(np.nanmean(pair_delta < 0))
                pair_consistent_share = max(positive_share, negative_share)
                average_gap = float(
                    abs(above_candidate["avg_metric"] - below_candidate["avg_metric"])
                )
                opposite_group_sides = (
                    float(above_candidate["median_standardized_gap"])
                    * float(below_candidate["median_standardized_gap"])
                    < 0
                )
                meets_consistency_target = (
                    pair_consistent_share >= 0.7 and pair_contrast >= 0.5
                )
                pair_score = (
                    meets_consistency_target,
                    opposite_group_sides if meets_consistency_target else False,
                    pair_contrast
                    if meets_consistency_target
                    else pair_consistent_share,
                    pair_consistent_share
                    if meets_consistency_target
                    else pair_contrast,
                    average_gap,
                )
                if best_pair_score is None or pair_score > best_pair_score:
                    best_pair_score = pair_score
                    if float(np.nanmedian(pair_delta)) >= 0:
                        lower_candidate, higher_candidate = (
                            below_candidate,
                            above_candidate,
                        )
                    else:
                        lower_candidate, higher_candidate = (
                            above_candidate,
                            below_candidate,
                        )
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
                        lower_candidate["selection_position"] = (
                            "Lower contrasting trajectory"
                        )
                        higher_candidate["selection_position"] = (
                            "Higher contrasting trajectory"
                        )
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
            rows = complete.loc[complete["line_id"].eq(candidate.line_id)].sort_values(
                anchor_col
            )
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
                    "meanAbsoluteLineGap": serialize_number(
                        candidate.mean_abs_line_gap, 5
                    ),
                    "medianLineGap": serialize_number(candidate.median_line_gap, 5),
                    "aboveMedianShare": serialize_number(
                        candidate.above_median_share, 3
                    ),
                    "medianStandardizedGap": serialize_number(
                        candidate.median_standardized_gap, 5
                    ),
                    "distanceThreshold": serialize_number(
                        candidate.distance_threshold, 5
                    ),
                    "pairContrast": serialize_number(candidate.pair_contrast, 5),
                    "pairConsistentShare": serialize_number(
                        candidate.pair_consistent_share, 5
                    ),
                    "pairAverageGap": serialize_number(candidate.pair_average_gap, 5),
                    "selectionTier": candidate.selection_tier,
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
        .drop_duplicates(subset=["fips"])[["fips", "pct_rank"]]
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
    *,
    event_context: PageEventWindowContext | None = None,
) -> dict[str, object]:
    context = event_context or build_max_affected_event_context(con)
    analysis_start = context.analysis_start
    analysis_end = context.analysis_end
    events = context.events
    nri = context.nri
    metric = "median_ppsf_yoy"

    # All display windows are derived from the shared maximum affected-event table.
    affected = context.affected
    if affected.empty:
        empty = {
            "byRating": [],
            "affectedCounties": [],
            "riskCounts": {},
            "exampleCountyLines": [],
        }
        return {
            "windowBefore": empty,
            "windowA": empty,
            "windowB": empty,
            "summary": {
                "events": 0,
                "analysisStart": analysis_start.strftime("%Y-%m-%d"),
                "analysisEnd": (analysis_end - pd.DateOffset(months=1)).strftime(
                    "%Y-%m-%d"
                ),
            },
        }

    event_counties = (
        events[["fips"]]
        .drop_duplicates()
        .merge(nri[["fips", "riskRating"]], on="fips", how="left")
    )
    overview_counts = (
        event_counties.dropna(subset=["riskRating"])
        .groupby("riskRating")["fips"]
        .nunique()
        .to_dict()
    )

    window_before = _build_window_data(
        affected,
        nri,
        metric,
        pre_months=12,
        post_months=0,
        eligible_feature_fips_by_risk=eligible_feature_fips_by_risk,
    )

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
        "windowBefore": window_before,
        "windowA": window_a,
        "windowB": window_b,
        "summary": {
            "events": int(events["event_key"].nunique()),
            "affectedCounties": int(event_counties["fips"].nunique()),
            "totalCounties": len(current_county_fips()),
            "riskCounts": {
                risk: int(overview_counts.get(risk, 0)) for risk in RISK_ORDER
            },
            "analysisStart": analysis_start.strftime("%Y-%m-%d"),
            "analysisEnd": (analysis_end - pd.DateOffset(months=1)).strftime(
                "%Y-%m-%d"
            ),
        },
    }


def ten_year_avg_by_fips(
    con: duckdb.DuckDBPyConnection, table: str, columns: list[str]
) -> pd.DataFrame:
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
    current_nri_fips = set(nri["fips"])
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
    labels = labels.loc[labels["fips"].isin(current_county_fips())].copy()
    # Keep housing counties even when the NRI has no matching risk record so the
    # playbook can explain that climate-risk guidance is unavailable.
    nri = labels.merge(nri, on="fips", how="left")
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
        return {
            "available": False,
            "message": "County housing and NRI data are unavailable",
        }

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
    history["median_ppsf_yoy"] = pd.to_numeric(
        history["median_ppsf_yoy"], errors="coerce"
    )
    county_fips = {str(county["fips"]).zfill(5) for county in counties}
    history = history.loc[history["fips"].isin(county_fips)].dropna(
        subset=["median_ppsf_yoy"]
    )

    events = load_disaster_events(con)
    events = events.loc[
        events["event_start_month"].ge(analysis_start)
        & events["event_start_month"].lt(analysis_end)
    ].copy()
    events = filter_current_state_county_events(
        events, current_nri_fips
    ).drop_duplicates("event_key")
    events = events.loc[events["fips"].isin(county_fips)].sort_values(
        ["fips", "event_start_month"]
    )

    history_months = pd.date_range(
        analysis_start, analysis_end - pd.DateOffset(months=1), freq="MS"
    )
    history_month_labels = [month.strftime("%Y-%m") for month in history_months]
    monthly_history_values_by_fips: dict[str, list[float | None]] = {}
    for fips, county_history in history.groupby("fips", sort=False):
        values = county_history.set_index("month")["median_ppsf_yoy"].reindex(
            history_months
        )
        monthly_history_values_by_fips[fips] = [
            serialize_number(value, 5) for value in values
        ]
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
        "hazards": [
            {"key": hazard["key"], "label": hazard["label"]} for hazard in HAZARDS
        ],
        "counties": counties,
        "monthlyHistoryMonths": history_month_labels,
        "monthlyHistoryValuesByFips": monthly_history_values_by_fips,
        "eventsByFips": events_by_fips,
        "eventCountyFips": sorted(events_by_fips),
        "historyStart": analysis_start.strftime("%Y-%m"),
        "historyEnd": (analysis_end - pd.DateOffset(months=1)).strftime("%Y-%m"),
    }


def _build_legacy_feature_payload(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    nri = con.execute(
        "SELECT fips, risk_rating, risk_score FROM mart.nri_county_risk WHERE fips IS NOT NULL"
    ).df()
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
    afford = ten_year_avg_by_fips(
        con, "mart.acs_county_affordability_annual", affordability_cols
    )
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
            (
                f"b25141_homeowners_insurance_costs_by_mortgage_status_total_{status}_{suffix}_est",
                midpoint,
            )
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
    electricity = (
        weighted_bucket_average(
            afford,
            [
                (
                    "b25132_monthly_electricity_costs_total_charged_for_electricity_less_than_dollars_50_est",
                    25,
                ),
                (
                    "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_50_to_dollars_99_est",
                    75,
                ),
                (
                    "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_100_to_dollars_149_est",
                    125,
                ),
                (
                    "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_150_to_dollars_199_est",
                    175,
                ),
                (
                    "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_200_to_dollars_249_est",
                    225,
                ),
                (
                    "b25132_monthly_electricity_costs_total_charged_for_electricity_dollars_250_or_more_est",
                    275,
                ),
            ],
            zero_cols=[
                "b25132_monthly_electricity_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"
            ],
        )
        * 12
    )
    gas = (
        weighted_bucket_average(
            afford,
            [
                (
                    "b25133_monthly_gas_costs_total_charged_for_gas_less_than_dollars_25_est",
                    12.5,
                ),
                (
                    "b25133_monthly_gas_costs_total_charged_for_gas_dollars_25_to_dollars_49_est",
                    37.5,
                ),
                (
                    "b25133_monthly_gas_costs_total_charged_for_gas_dollars_50_to_dollars_74_est",
                    62.5,
                ),
                (
                    "b25133_monthly_gas_costs_total_charged_for_gas_dollars_75_to_dollars_99_est",
                    87.5,
                ),
                (
                    "b25133_monthly_gas_costs_total_charged_for_gas_dollars_100_to_dollars_149_est",
                    125,
                ),
                (
                    "b25133_monthly_gas_costs_total_charged_for_gas_dollars_150_or_more_est",
                    175,
                ),
            ],
            zero_cols=[
                "b25133_monthly_gas_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"
            ],
        )
        * 12
    )
    water = weighted_bucket_average(
        afford,
        [
            (
                "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_less_than_dollars_125_est",
                62.5,
            ),
            (
                "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_125_to_dollars_249_est",
                187.5,
            ),
            (
                "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_250_to_dollars_499_est",
                375,
            ),
            (
                "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_500_to_dollars_749_est",
                625,
            ),
            (
                "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_750_to_dollars_999_est",
                875,
            ),
            (
                "b25134_annual_water_and_sewer_costs_total_charged_for_water_and_sewer_dollars_1_000_or_more_est",
                1125,
            ),
        ],
        zero_cols=[
            "b25134_annual_water_and_sewer_costs_total_not_charged_or_payment_included_in_other_fees_est"
        ],
    )
    other_fuel = weighted_bucket_average(
        afford,
        [
            (
                "b25135_annual_other_fuel_costs_total_charged_for_other_fuels_less_than_dollars_250_est",
                125,
            ),
            (
                "b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_250_to_dollars_749_est",
                500,
            ),
            (
                "b25135_annual_other_fuel_costs_total_charged_for_other_fuels_dollars_750_or_more_est",
                875,
            ),
        ],
        zero_cols=[
            "b25135_annual_other_fuel_costs_total_not_charged_not_used_or_payment_included_in_other_fees_est"
        ],
    )
    afford["estimated_annual_utilities"] = electricity + gas + water + other_fuel
    afford["income_median_household_usd"] = afford[
        "s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est"
    ]
    afford["insurance_homeowners_pct_income"] = (
        afford["estimated_annual_home_insurance"]
        / afford["income_median_household_usd"].replace(0, np.nan)
        * 100
    )
    afford["property_taxes_pct_income"] = (
        afford["estimated_annual_property_tax"]
        / afford["income_median_household_usd"].replace(0, np.nan)
        * 100
    )
    afford["utilities_pct_income"] = (
        afford["estimated_annual_utilities"]
        / afford["income_median_household_usd"].replace(0, np.nan)
        * 100
    )
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
        afford[
            "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_est"
        ]
        + afford[
            "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_est"
        ]
        - afford[
            "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_units_mortgage_not_computed_est"
        ].fillna(0)
        - afford[
            "dp04_selected_monthly_owner_costs_as_a_pct_of_household_income_housing_unit_no_mortgage_not_computed_est"
        ].fillna(0)
    )
    afford["housing_burden_30pct_plus_share"] = (
        burdened_owner_households
        / owner_households_with_computable_burden.replace(0, np.nan)
        * 100
    )
    afford["homeownership_cost_pct_income"] = (
        afford[
            "s2503_owner_occupied_units_occupied_housing_units_monthly_housing_costs_median_est"
        ]
        * 12
        / afford["income_median_household_usd"].replace(0, np.nan)
        * 100
    )

    features = (
        nri[["fips", "riskRating", "riskValue", "risk_score"]]
        .merge(econ[["fips", *econ_cols[2:]]], on="fips", how="left")
        .merge(demo[["fips", *demo_cols[2:]]], on="fips", how="left")
        .merge(migration, on="fips", how="left")
        .merge(
            afford[
                [
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
                ]
            ],
            on="fips",
            how="left",
        )
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
    features["no_broadband_pct"] = (
        100
        - features[
            "dp02_computers_and_internet_use_total_households_with_a_broadband_internet_subscription_pct"
        ]
    )
    feature_defs = [
        (
            "Economic",
            "Income",
            "income_median_household_usd",
            "currency",
            "mart.acs_county_affordability_annual",
        ),
        (
            "Economic",
            "Insurance Share of Income",
            "insurance_homeowners_pct_income",
            "percent",
            "mart.acs_county_affordability_annual",
        ),
        (
            "Economic",
            "Property Tax Share of Income",
            "property_taxes_pct_income",
            "percent",
            "mart.acs_county_affordability_annual",
        ),
        (
            "Economic",
            "Utilities Share of Income",
            "utilities_pct_income",
            "percent",
            "mart.acs_county_affordability_annual",
        ),
        (
            "Economic",
            "Cost-Burdened Households",
            "housing_burden_30pct_plus_share",
            "percent",
            "mart.acs_county_affordability_annual",
        ),
        (
            "Economic",
            "Homeownership Cost Share",
            "homeownership_cost_pct_income",
            "percent",
            "mart.acs_county_affordability_annual",
        ),
        (
            "Economic",
            "Unemployment",
            "dp03_civilian_labor_force_unemployment_rate_pct",
            "percent",
            "mart.acs_county_economic_annual",
        ),
        (
            "Economic",
            "Net Earnings per Capita",
            "net_earnings_per_capita",
            "currency",
            "mart.statsamerica_bea_personal_income_annual",
        ),
        (
            "Economic",
            "Dividends/Interest/Rent per Capita",
            "dividends_interest_rent_per_capita",
            "currency",
            "mart.statsamerica_bea_personal_income_annual",
        ),
        (
            "Economic",
            "Transfer Receipts per Capita",
            "transfer_receipts_per_capita",
            "currency",
            "mart.statsamerica_bea_personal_income_annual",
        ),
        (
            "Demographic",
            "Net Migration Rate",
            "net_migration_rate",
            "signed_pct",
            "mart.statsamerica_population_components_annual",
        ),
        (
            "Demographic",
            "Age >= 65 Years",
            "dp05_total_population_65_plus_pct",
            "percent",
            "mart.acs_county_demographic_annual",
        ),
        (
            "Demographic",
            "Disability Status",
            "dp02_disability_status_of_the_civilian_noninstitutionalized_population_total_civilian_noninstitutionalized_population_with_a_disability_pct",
            "percent",
            "mart.acs_county_demographic_annual",
        ),
        (
            "Demographic",
            "Communication Barrier",
            "dp02_language_spoken_at_home_population_5_years_and_over_language_other_than_english_speak_english_less_than_very_well_pct",
            "percent",
            "mart.acs_county_demographic_annual",
        ),
        (
            "Demographic",
            "No Internet Access",
            "no_broadband_pct",
            "percent",
            "mart.acs_county_demographic_annual",
        ),
        (
            "Housing Market",
            "Median PPSF YOY",
            "median_ppsf_yoy",
            "pct",
            "mart.redfin_county_monthly",
        ),
        (
            "Housing Market",
            "Average Sale-to-List YOY",
            "avg_sale_to_list_yoy",
            "pct",
            "mart.redfin_county_monthly",
        ),
        (
            "Housing Market",
            "Homes Sold YOY",
            "homes_sold_yoy",
            "pct",
            "mart.redfin_county_monthly",
        ),
        (
            "Housing Market",
            "Inventory YOY",
            "inventory_yoy",
            "pct",
            "mart.redfin_county_monthly",
        ),
        (
            "Housing Market",
            "New Listings YOY",
            "new_listings_yoy",
            "pct",
            "mart.redfin_county_monthly",
        ),
        (
            "Housing Market",
            "Median Days on Market YOY",
            "median_dom_yoy",
            "pct",
            "mart.redfin_county_monthly",
        ),
        (
            "Housing Market",
            "Active Listings with Price Drops YOY",
            "price_drops_yoy",
            "pct",
            "mart.redfin_county_monthly",
        ),
        (
            "Climate",
            "Temperature",
            "avg_temperature_f",
            "temperature_f",
            "mart.ncei_county_weather_monthly",
        ),
        (
            "Climate",
            "Precipitation",
            "precipitation_inches",
            "inches",
            "mart.ncei_county_weather_monthly",
        ),
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
    if (
        len(paired) < 3
        or paired.iloc[:, 0].nunique() < 2
        or paired.iloc[:, 1].nunique() < 2
    ):
        return float("nan")
    return float(paired.iloc[:, 0].rank().corr(paired.iloc[:, 1].rank()))


FEATURE_TARGET_COLUMN = "event_window_avg_ppsf_yoy"
FEATURE_PERFORMANCE_TARGET_COLUMN = "complete_event_window_trajectory_median_ppsf_yoy"


def _select_significant_feature_metrics(metrics: list[dict]) -> list[dict]:
    """Keep threshold matches, topping up to three finite correlations by rank."""
    ranked = sorted(
        [item for item in metrics if item.get("rho") is not None and np.isfinite(item["rho"])],
        key=lambda item: abs(item["rho"]), reverse=True,
    )
    threshold_count = sum(abs(item["rho"]) >= 0.30 for item in ranked)
    return ranked[:max(3, threshold_count)]


def _county_median_trajectory_target(complete: pd.DataFrame) -> pd.DataFrame:
    """Median across complete events at each relative month, then across months."""
    monthly = (
        complete[["fips", "event_window_month", "median_ppsf_yoy"]].dropna()
        .groupby(["fips", "event_window_month"], as_index=False)["median_ppsf_yoy"]
        .median()
    )
    return (
        monthly.groupby("fips", as_index=False)["median_ppsf_yoy"].median()
        .rename(columns={"median_ppsf_yoy": FEATURE_PERFORMANCE_TARGET_COLUMN})
    )


def _county_average_event_window_target(
    complete: pd.DataFrame,
    *,
    metric: str = "median_ppsf_yoy",
) -> pd.DataFrame:
    """Average every available event-window PPSF YoY observation per county."""
    event_rows = complete[["fips", metric]].dropna().copy()
    if event_rows.empty:
        return pd.DataFrame(columns=["fips", FEATURE_TARGET_COLUMN])
    return (
        event_rows.groupby("fips", as_index=False)[metric]
        .mean()
        .rename(columns={metric: FEATURE_TARGET_COLUMN})
    )


def assign_performer_subgroup_from_ranges(
    value: float,
    groups: object,
) -> int | None:
    """Map a historical county average to event-window subgroup value ranges."""
    if not np.isfinite(value) or not isinstance(groups, list):
        return None
    valid_groups = [
        group
        for group in groups
        if isinstance(group, dict)
        and isinstance(group.get("index"), int)
        and group.get("targetMin") is not None
        and group.get("targetMax") is not None
    ]
    if not valid_groups:
        return None

    for group in valid_groups:
        if float(group["targetMin"]) <= value <= float(group["targetMax"]):
            return int(group["index"])

    strongest = min(valid_groups, key=lambda group: int(group["index"]))
    weakest = max(valid_groups, key=lambda group: int(group["index"]))
    if value > float(strongest["targetMax"]):
        return int(strongest["index"])
    if value < float(weakest["targetMin"]):
        return int(weakest["index"])

    def distance_to_range(group: dict[str, object]) -> tuple[float, int]:
        lower = float(group["targetMin"])
        upper = float(group["targetMax"])
        distance = lower - value if value < lower else value - upper
        return round(distance, 12), int(group["index"])

    return int(min(valid_groups, key=distance_to_range)["index"])


def _bootstrap_spearman_ci(
    frame: pd.DataFrame,
    feature: str,
    *,
    target_column: str = FEATURE_TARGET_COLUMN,
    iterations: int = 160,
    seed: int,
) -> tuple[float, float]:
    paired = frame[[feature, target_column]].dropna().reset_index(drop=True)
    if len(paired) < 12:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ranked_x = paired[feature].rank().to_numpy(dtype=float)
    ranked_y = paired[target_column].rank().to_numpy(dtype=float)
    values: list[float] = []
    for _ in range(iterations):
        sample_index = rng.integers(0, len(paired), len(paired))
        correlation = float(
            np.corrcoef(ranked_x[sample_index], ranked_y[sample_index])[0, 1]
        )
        if np.isfinite(correlation):
            values.append(correlation)
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def build_feature_payload(
    con: duckdb.DuckDBPyConnection,
    *,
    event_context: PageEventWindowContext | None = None,
) -> dict[str, object]:
    """Build the within-risk feature story from the DuckDB feature layer."""
    feature_columns = [item[2] for item in WITHIN_GROUP_FEATURES]
    economic_columns = feature_columns[:8]
    demographic_columns = feature_columns[8:]

    economic = con.execute(
        f"""
        SELECT lpad(fips, 5, '0') AS fips,
               {", ".join(f"avg({column}) AS {column}" for column in economic_columns)}
        FROM feature.county_economic_annual
        WHERE fips IS NOT NULL
          AND year >= (SELECT max(year) - 9 FROM feature.county_economic_annual)
        GROUP BY fips
        """
    ).df()
    demographic = con.execute(
        f"""
        SELECT lpad(fips, 5, '0') AS fips,
               {", ".join(f"avg({column}) AS {column}" for column in demographic_columns)}
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

    context = event_context or build_max_affected_event_context(con)
    analysis_start = context.analysis_start
    analysis_end = context.analysis_end
    history = con.execute(
        """
        SELECT
            lpad(fips, 5, '0') AS fips,
            any_value(REGION) AS county,
            any_value(STATE_CODE) AS state,
            avg(CASE
                WHEN try_cast(MEDIAN_PPSF_YOY AS DOUBLE) <= -888888000 THEN NULL
                ELSE try_cast(MEDIAN_PPSF_YOY AS DOUBLE)
            END) AS historical_average_ppsf_yoy
        FROM mart.redfin_county_monthly
        WHERE fips IS NOT NULL
          AND coalesce(property_type, PROPERTY_TYPE_1) = 'All Residential'
          AND period_begin >= ?
          AND period_begin < ?
        GROUP BY fips
        """,
        [analysis_start, analysis_end],
    ).df()
    history["fips"] = history["fips"].astype(str).str.zfill(5)
    history = history.loc[history["fips"].isin(current_county_fips())].copy()
    history["historical_average_ppsf_yoy"] = pd.to_numeric(
        history["historical_average_ppsf_yoy"], errors="coerce"
    )
    affected = context.affected
    required_months = event_window_months(12, 36)
    available = affected.loc[
        affected["event_window_month"].isin(required_months)
        & affected["median_ppsf_yoy"].notna()
        & affected["line_id"].notna()
    ].copy()
    complete = filter_complete_event_window_lines(
        affected.loc[
            affected["event_window_month"].isin(required_months)
            & affected["line_id"].notna()
        ].copy(),
        x_col="event_window_month",
        line_col="line_id",
        metric_col="median_ppsf_yoy",
        required_x_values=required_months,
    )
    available = available.merge(nri[["fips", "riskRating"]], on="fips", how="left")
    complete = complete.merge(nri[["fips", "riskRating"]], on="fips", how="left")

    county_target = _county_average_event_window_target(available)
    performance_target = _county_median_trajectory_target(complete)
    counties = (
        nri[["fips", "riskRating"]]
        .merge(economic, on="fips", how="left")
        .merge(demographic, on="fips", how="left")
        .merge(history, on="fips", how="left")
        .merge(county_target, on="fips", how="left")
        .merge(performance_target, on="fips", how="left")
    )
    counties = counties.loc[
        counties["fips"].isin(current_county_fips())
        & counties["riskRating"].isin(RISK_ORDER)
    ].copy()
    for column in [
        FEATURE_TARGET_COLUMN,
        FEATURE_PERFORMANCE_TARGET_COLUMN,
        "historical_average_ppsf_yoy",
        *feature_columns,
    ]:
        counties[column] = pd.to_numeric(counties[column], errors="coerce")

    minimum_effect = 0.10
    importance_by_risk: dict[str, list[dict[str, object]]] = {}
    scatter_rows_by_risk: dict[str, list[dict[str, object]]] = {}
    county_rows_by_risk: dict[str, list[dict[str, object]]] = {}
    subgroup_payload: dict[str, dict[str, object]] = {}
    subgroup_by_fips: dict[str, int] = {}

    for risk_index, risk in enumerate(RISK_ORDER):
        risk_counties = counties.loc[counties["riskRating"].eq(risk)].copy()
        analysis_group = risk_counties.dropna(
            subset=[FEATURE_PERFORMANCE_TARGET_COLUMN]
        ).copy()
        metrics: list[dict[str, object]] = []
        for feature_index, (_, _, feature, _) in enumerate(WITHIN_GROUP_FEATURES):
            paired = analysis_group[
                [feature, FEATURE_PERFORMANCE_TARGET_COLUMN]
            ].dropna()
            rho = _spearman_correlation(
                paired[feature], paired[FEATURE_PERFORMANCE_TARGET_COLUMN]
            )
            ci_low, ci_high = _bootstrap_spearman_ci(
                paired,
                feature,
                target_column=FEATURE_PERFORMANCE_TARGET_COLUMN,
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

        scatter_rows_by_risk[risk] = [
            {
                "fips": row.fips,
                "county": row.county,
                "state": row.state,
                "target": serialize_number(
                    getattr(row, FEATURE_PERFORMANCE_TARGET_COLUMN), 5
                ),
                "values": {
                    feature: serialize_number(getattr(row, feature), 5)
                    for feature in feature_columns
                },
            }
            for row in analysis_group.itertuples(index=False)
        ]

        performance_group = analysis_group.copy()
        county_rows_by_risk[risk] = [
            {
                "fips": row.fips,
                "county": row.county,
                "state": row.state,
                "target": serialize_number(
                    getattr(row, FEATURE_PERFORMANCE_TARGET_COLUMN), 5
                ),
                "values": {
                    feature: serialize_number(getattr(row, feature), 5)
                    for feature in feature_columns
                },
            }
            for row in performance_group.itertuples(index=False)
        ]

        if performance_group.empty:
            subgroup_payload[risk] = {"groups": [], "excludedOutliers": 0}
            continue
        distribution_metrics = _select_significant_feature_metrics(metrics)
        distribution_features = [str(item["feature"]) for item in distribution_metrics]

        subgroup_count = 4
        performance_group = performance_group.sort_values(
            [FEATURE_PERFORMANCE_TARGET_COLUMN, "fips"], ascending=[False, True]
        ).reset_index(drop=True)
        performance_group["subgroup"] = np.minimum(
            np.floor(
                np.arange(len(performance_group))
                * subgroup_count
                / len(performance_group)
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
        county_month_lines = line_frame.groupby(
            ["fips", "subgroup", "event_window_month"], as_index=False
        )["median_ppsf_yoy"].median()
        group_entries: list[dict[str, object]] = []
        for subgroup_index in sorted(performance_group["subgroup"].dropna().unique()):
            subgroup_index = int(subgroup_index)
            members = performance_group.loc[
                performance_group["subgroup"].eq(subgroup_index)
            ]
            monthly = (
                county_month_lines.loc[
                    county_month_lines["subgroup"].eq(subgroup_index)
                ]
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
                    "targetMedian": serialize_number(
                        members[FEATURE_PERFORMANCE_TARGET_COLUMN].median(), 5
                    ),
                    "targetMin": serialize_number(
                        members[FEATURE_PERFORMANCE_TARGET_COLUMN].min(), 5
                    ),
                    "targetMax": serialize_number(
                        members[FEATURE_PERFORMANCE_TARGET_COLUMN].max(), 5
                    ),
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
            "hasStrongFeatures": bool(distribution_metrics),
            "groups": group_entries,
            "excludedOutliers": 0,
        }

    playbook_subgroup_by_fips = dict(subgroup_by_fips)
    playbook_subgroup_source_by_fips = {
        fips: "event-window" for fips in subgroup_by_fips
    }
    # Historical fallback is calculated from the deferred ten-year histories in
    # the browser, using county medians and monthly risk-group quartiles.

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
        "allCountyRowsByRisk": {
            risk: [
                {"fips": row.fips, "values": {
                    feature: serialize_number(getattr(row, feature), 5)
                    for feature in feature_columns
                }}
                for row in counties.loc[counties["riskRating"].eq(risk)].itertuples(index=False)
            ] for risk in RISK_ORDER
        },
        "subgroupsByRisk": subgroup_payload,
        "subgroupByFips": subgroup_by_fips,
        "playbookSubgroupByFips": playbook_subgroup_by_fips,
        "playbookSubgroupSourceByFips": playbook_subgroup_source_by_fips,
    }


def make_html(data: dict[str, object]) -> str:
    initial = {key: value for key, value in data.items()
               if key not in {"features", "eventWindows", "geojson", "stateGeojson"}}
    payload = json.dumps(initial, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return HTML_TEMPLATE.replace("__PAYLOAD__", payload)


def make_deferred_data_script(global_name: str, data: object) -> str:
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    return f"window.{global_name}={payload};\n"


WEB_ASSET_DIR = Path(__file__).with_name("web")
HTML_TEMPLATE = (WEB_ASSET_DIR / "page.html").read_text(encoding="utf-8")


def write_page_assets(data: dict[str, object], output_dir: Path = OUT_PATH.parent) -> None:
    """Write the static site and deferred bundles without changing analysis data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(make_html(data), encoding="utf-8")
    for name in ("climate-risk-housing.css", "climate-risk-housing.js"):
        (output_dir / name).write_text((WEB_ASSET_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    features = dict(data["features"])
    features.pop("scatterRowsByRisk", None)
    bundles = {
        "features": ("FEATURES", features),
        "events": ("EVENTS", data["eventWindows"]),
        "geography": ("GEOGRAPHY", {key: data[key] for key in ("geojson", "stateGeojson")}),
    }
    for name, (suffix, payload) in bundles.items():
        (output_dir / f"climate-risk-housing-{name}.js").write_text(
            make_deferred_data_script(f"CLIMATE_RISK_HOUSING_{suffix}", payload), encoding="utf-8")
    for relative_path, destination in (("climate-risk-housing.html", "index.html"),
                                       ("output/climate-risk-housing.html", "../index.html")):
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Climate Risk &amp; Housing</title>'
            f'<meta http-equiv="refresh" content="0;url={destination}">'
            f'<link rel="canonical" href="{destination}">'
            f'<script>location.replace("{destination}" + location.search + location.hash);</script>'
            f'<a href="{destination}">Continue to Climate Risk &amp; Housing</a></html>', encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        price_risk = build_price_risk(con)
        event_context = build_max_affected_event_context(con)
        features = build_feature_payload(con, event_context=event_context)
        event_windows = build_event_windows(con, event_context=event_context)
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
    write_page_assets(data)
    COUNTY_HISTORY_OUT_PATH.write_text(
        make_deferred_data_script(
            "CLIMATE_RISK_HOUSING_COUNTY_HISTORY", county_history
        ),
        encoding="utf-8",
    )
    PLAYBOOK_OUT_PATH.write_text(
        make_deferred_data_script("CLIMATE_RISK_HOUSING_PLAYBOOK", playbook),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_PATH.parent / 'index.html'} and static assets (legacy URL redirects retained)")
    print(f"Wrote {COUNTY_HISTORY_OUT_PATH}")
    print(f"Wrote {PLAYBOOK_OUT_PATH}")


if __name__ == "__main__":
    main()
