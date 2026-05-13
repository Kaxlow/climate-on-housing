"""Build the stormhouse county housing response visualization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from housing_climate_risk.data_sources.processed import prepare_housing_df, prepare_natural_disasters_df
from housing_climate_risk.paths import CLIMATE_DIR, GEOGRAPHIC_DIR, OUTPUT_DIR

DATA_JS = OUTPUT_DIR / "stormhouse_data.js"
HTML_PATH = OUTPUT_DIR / "stormhouse.html"

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

VALID_RISK_RATINGS = ["Very Low", "Low", "Moderate", "High", "Very High"]
RISK_RATING_MAP = {
    "Very Low": "Very Low",
    "Relatively Low": "Low",
    "Relatively Moderate": "Moderate",
    "Relatively High": "High",
    "Very High": "Very High",
}
OFFSETS = list(range(-12, 25))
US_STATE_FIPS = {
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


def _num(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _round_coords(coords: object, digits: int = 3) -> object:
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        return [round(float(coords[0]), digits), round(float(coords[1]), digits)]
    if isinstance(coords, (list, tuple)):
        return [_round_coords(item, digits) for item in coords]
    return coords


def build_county_risk_map() -> dict[str, object]:
    from shapely.geometry import mapping, shape

    nri = pd.read_csv(CLIMATE_DIR / "NRI_Table_Counties.csv", usecols=["STCOFIPS", "RISK_RATNG"])
    risk_by_fips = (
        nri.assign(
            fips=nri["STCOFIPS"].astype(str).str.zfill(5),
            riskRating=nri["RISK_RATNG"].map(RISK_RATING_MAP),
        )
        .drop_duplicates("fips")
        .set_index("fips")["riskRating"]
        .to_dict()
    )

    with (GEOGRAPHIC_DIR / "us_counties_boundaries_shapefile.json").open(encoding="utf-8") as file:
        counties = json.load(file)

    features: list[dict[str, object]] = []
    counts = {rating: 0 for rating in VALID_RISK_RATINGS}
    for feature in counties["features"]:
        props = feature["properties"]
        state_fips = str(props.get("STATEFP", "")).zfill(2)
        if state_fips not in US_STATE_FIPS:
            continue
        fips = str(props.get("GEOID", "")).zfill(5)
        risk_rating = risk_by_fips.get(fips)
        if risk_rating not in counts:
            continue
        counts[risk_rating] += 1

        tolerance = 0.08 if state_fips == "02" else 0.025
        geom = shape(feature["geometry"]).simplify(tolerance, preserve_topology=True)
        if geom.is_empty:
            continue
        geom_mapping = mapping(geom)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": geom_mapping["type"],
                    "coordinates": _round_coords(geom_mapping["coordinates"]),
                },
                "properties": {
                    "fips": fips,
                    "stateFips": state_fips,
                    "name": props.get("NAMELSAD") or props.get("NAME") or fips,
                    "riskRating": risk_rating,
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "counts": counts,
    }


def build_complete_windows() -> pd.DataFrame:
    disasters = prepare_natural_disasters_df()
    housing = prepare_housing_df(include_profiles=False).copy()

    nri = pd.read_csv(CLIMATE_DIR / "NRI_Table_Counties.csv", usecols=["STCOFIPS", "RISK_RATNG"])
    nri = (
        nri.assign(
            fips=nri["STCOFIPS"].astype(str).str.zfill(5),
            nri_risk_rating=nri["RISK_RATNG"].map(RISK_RATING_MAP),
        )[["fips", "nri_risk_rating"]]
        .drop_duplicates("fips")
    )

    housing["fips"] = housing["fips"].astype(str).str.zfill(5)
    housing["fips_normalized"] = housing["fips"]
    housing["state_prefix"] = housing["fips"].str[:2]
    housing["MONTH"] = housing["MONTH"].astype("period[M]")
    housing = housing[
        [
            "fips",
            "fips_normalized",
            "state_prefix",
            "REGION",
            "county_name",
            "STATE_CODE",
            "MONTH",
            "HOUSING_MARKET_INDEX",
        ]
    ].copy()

    events = disasters.loc[~disasters["incidentType"].isin(EXCLUDED_INCIDENT_TYPES)].copy()
    events = events.dropna(subset=["fips", "incidentBeginDate"])
    duration = events["incidentEndDate"] - events["incidentBeginDate"]
    median_duration = duration[duration.notna()].median()
    events["incidentEndDate"] = events["incidentEndDate"].fillna(events["incidentBeginDate"] + median_duration)
    events = events.dropna(subset=["incidentEndDate"])
    events["fips"] = events["fips"].astype(str).str.zfill(5)
    events["event_id"] = np.arange(len(events), dtype=int) + 1
    events["incident_begin_month"] = events["incidentBeginDate"].dt.to_period("M")
    events["incident_end_month"] = events["incidentEndDate"].dt.to_period("M")

    event_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        event_fips = str(event.fips).zfill(5)
        months = {event.incident_begin_month + offset: offset for offset in range(-12, 0)}
        months[event.incident_end_month] = 0
        months.update({event.incident_end_month + offset: offset for offset in range(1, 25)})
        for month, offset in months.items():
            event_rows.append(
                {
                    "event_fips": event_fips,
                    "event_state_prefix": event_fips[:2],
                    "is_statewide_event": event_fips.endswith("000"),
                    "MONTH": month,
                    "month_offset": int(offset),
                    "incident_event_id": int(event.event_id),
                    "incident_type": event.incidentType,
                    "incident_title": event.declarationTitle,
                    "incident_begin": event.incidentBeginDate.date().isoformat(),
                    "incident_end": event.incidentEndDate.date().isoformat(),
                }
            )

    event_months = pd.DataFrame(event_rows)
    county_events = event_months.loc[~event_months["is_statewide_event"]]
    state_events = event_months.loc[event_months["is_statewide_event"]]
    frames = []
    if not county_events.empty:
        frames.append(housing.merge(county_events, left_on=["fips_normalized", "MONTH"], right_on=["event_fips", "MONTH"], how="inner"))
    if not state_events.empty:
        frames.append(housing.merge(state_events, left_on=["state_prefix", "MONTH"], right_on=["event_state_prefix", "MONTH"], how="inner"))
    matched = pd.concat(frames, ignore_index=True) if frames else housing.iloc[0:0].copy()

    matched = matched.merge(nri, on="fips", how="left")
    matched = matched.dropna(subset=["HOUSING_MARKET_INDEX"])
    matched["series_id"] = (
        matched["fips"].astype(str)
        + "|"
        + matched["incident_type"].astype(str)
        + "|"
        + matched["incident_event_id"].astype(str)
    )
    expected = set(OFFSETS)
    complete_ids = (
        matched.groupby("series_id", sort=False)["month_offset"]
        .agg(lambda values: set(pd.to_numeric(values, errors="coerce").dropna().astype(int)) == expected)
    )
    complete_ids = complete_ids[complete_ids].index
    matched = matched.loc[matched["series_id"].isin(complete_ids)].copy()
    return matched


def build_payload(windows: pd.DataFrame) -> dict[str, object]:
    county_risk_map = build_county_risk_map()
    county_windows = windows.loc[windows["nri_risk_rating"].isin(VALID_RISK_RATINGS)].copy()
    county_windows["incident_begin_dt"] = pd.to_datetime(county_windows["incident_begin"], errors="coerce")
    county_windows = county_windows.sort_values(["fips", "month_offset", "incident_begin_dt", "incident_event_id"]).copy()
    county_windows["recency_weight"] = county_windows.groupby(["fips", "month_offset"]).cumcount() + 1
    county_windows["weighted_index"] = county_windows["HOUSING_MARKET_INDEX"] * county_windows["recency_weight"]

    weighted = (
        county_windows.groupby(["fips", "month_offset"], as_index=False)
        .agg(
            housing_market_yoy_index=("weighted_index", "sum"),
            total_weight=("recency_weight", "sum"),
            incident_count=("incident_event_id", "nunique"),
        )
        .assign(housing_market_yoy_index=lambda df: df["housing_market_yoy_index"] / df["total_weight"])
    )
    meta = (
        county_windows.sort_values(["fips", "incident_begin_dt", "incident_event_id"])
        .groupby("fips", as_index=False)
        .agg(
            county=("county_name", "last"),
            state=("STATE_CODE", "last"),
            riskRating=("nri_risk_rating", "last"),
            latestIncident=("incident_begin", "last"),
            incidentCount=("incident_event_id", "nunique"),
        )
    )
    weighted = weighted.merge(meta, on="fips", how="left")

    group_stats = (
        weighted.groupby(["riskRating", "month_offset"], sort=True)["housing_market_yoy_index"]
        .agg(
            median=lambda values: values.quantile(0.5),
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            n="count",
        )
        .reset_index()
    )

    risk_ratings = VALID_RISK_RATINGS
    group_payload: dict[str, list[dict[str, object]]] = {}
    for rating in risk_ratings:
        rows = group_stats.loc[group_stats["riskRating"] == rating]
        group_payload[rating] = [
            {
                "offset": int(row.month_offset),
                "median": _num(row.median),
                "q1": _num(row.q1),
                "q3": _num(row.q3),
                "n": int(row.n),
            }
            for row in rows.itertuples(index=False)
        ]

    period_means = (
        weighted.assign(
            period=np.select(
                [
                    weighted["month_offset"].between(-12, -1),
                    weighted["month_offset"].between(1, 12),
                    weighted["month_offset"].between(13, 24),
                ],
                ["pre", "months_1_12", "months_13_24"],
                default="other",
            )
        )
        .loc[lambda df: df["period"].isin(["pre", "months_1_12", "months_13_24"])]
        .groupby(["riskRating", "fips", "period"], sort=False)["housing_market_yoy_index"]
        .mean()
        .unstack("period")
        .reset_index()
    )
    period_means["change_pre_to_1_12"] = period_means["months_1_12"] - period_means["pre"]
    period_means["change_1_12_to_13_24"] = period_means["months_13_24"] - period_means["months_1_12"]
    group_summaries: dict[str, dict[str, object]] = {}
    for rating in risk_ratings:
        group = period_means.loc[period_means["riskRating"] == rating]
        group_summaries[rating] = {
            "avgPreToMonths1To12": _num(group["change_pre_to_1_12"].mean()),
            "avgMonths1To12To13To24": _num(group["change_1_12_to_13_24"].mean()),
            "countyCount": int(group["fips"].nunique()),
        }

    rated_summaries = {rating: group_summaries[rating] for rating in VALID_RISK_RATINGS if rating in group_summaries}
    lower_risk = [rating for rating in ["Very Low", "Low"] if rating in rated_summaries]
    higher_risk = [rating for rating in ["High", "Very High"] if rating in rated_summaries]
    lower_late = np.mean([rated_summaries[rating]["avgMonths1To12To13To24"] for rating in lower_risk])
    higher_late = np.mean([rated_summaries[rating]["avgMonths1To12To13To24"] for rating in higher_risk])
    lower_initial = np.mean([rated_summaries[rating]["avgPreToMonths1To12"] for rating in lower_risk])
    higher_initial = np.mean([rated_summaries[rating]["avgPreToMonths1To12"] for rating in higher_risk])
    commentary = [
        f"In the first year after an incident, higher-risk counties moved only modestly differently from lower-risk counties on average ({higher_initial:+.3f} vs. {lower_initial:+.3f}).",
        f"The longer-run pattern is clearer: higher-risk counties had a larger drop from months 1-12 to months 13-24 ({higher_late:+.3f}) than lower-risk counties ({lower_late:+.3f}), suggesting a more persistent weakening in higher-risk housing markets.",
    ]

    incident_types = sorted(windows["incident_type"].dropna().astype(str).unique())
    return {
        "meta": {
            "completeCountyIncidentWindows": int(windows["series_id"].nunique()),
            "countyCount": int(windows["fips"].nunique()),
            "plottedCountyCount": int(weighted["fips"].nunique()),
            "incidentTypes": incident_types,
            "excludedIncidentTypes": sorted(EXCLUDED_INCIDENT_TYPES),
            "riskRatings": risk_ratings,
            "offsets": OFFSETS,
        },
        "byRiskRating": group_payload,
        "groupSummaries": group_summaries,
        "commentary": commentary,
        "countyRiskMap": county_risk_map,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stormhouse: Housing Market Index Around FEMA Incidents</title>
  <script src="stormhouse_data.js"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/topojson/3.0.2/topojson.min.js"></script>
  <style>
    :root {
      --bg: #f7f5ef;
      --ink: #172026;
      --muted: #5e6872;
      --line: #c7c1b5;
      --panel: #ffffff;
      --accent: #0f766e;
      --band: rgba(15, 118, 110, 0.18);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 42px; }
    header { margin-bottom: 18px; }
    h1 { margin: 0; font-size: 30px; line-height: 1.1; letter-spacing: 0; }
    .dek { margin: 10px 0 0; color: var(--muted); max-width: none; font-size: 15px; line-height: 1.45; }
    section { background: var(--panel); border: 1px solid #ddd8cf; border-radius: 8px; padding: 16px; margin-top: 16px; box-shadow: 0 1px 2px rgba(23, 32, 38, 0.04); }
    .section-head { display: flex; justify-content: space-between; gap: 16px; align-items: start; margin-bottom: 8px; }
    h2 { margin: 0; font-size: 18px; line-height: 1.25; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; max-width: none; }
    .chart { width: 100%; height: 470px; }
    .chart text { fill: var(--muted); font-size: 12px; }
    .axis path, .axis line, .grid line { stroke: #d8d3ca; }
    .grid path { display: none; }
    .period-band-early { fill: rgba(15, 118, 110, 0.045); stroke: none; }
    .period-band-late { fill: rgba(15, 118, 110, 0.085); stroke: none; }
    .period-label { fill: var(--muted); font-size: 11px; }
    .zero-line { stroke: #111827; stroke-width: 2.25; }
    .risk-line { fill: none; stroke-width: 2.6; stroke-linejoin: round; stroke-linecap: round; }
    .risk-hit-line { fill: none; stroke: transparent; stroke-width: 14; stroke-linejoin: round; stroke-linecap: round; cursor: pointer; }
    .risk-band { stroke: none; opacity: 0; pointer-events: none; transition: opacity 120ms ease; }
    .risk-band.active { opacity: 0.2; }
    .incident-line { stroke: #2f3941; stroke-width: 1.5; stroke-dasharray: 5 5; }
    .incident-label { fill: #2f3941; font-size: 12px; font-weight: 700; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px 16px; color: var(--muted); font-size: 12px; margin-top: 8px; }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
    .map-wrap { border-top: 1px solid #e6e0d6; margin-top: 16px; padding-top: 14px; }
    h3 { margin: 0; font-size: 16px; line-height: 1.25; }
    .map { width: 100%; height: 560px; margin-top: 10px; display: block; background: #fbfaf7; border: 1px solid #ebe5dc; border-radius: 6px; }
    .county { stroke: #ffffff; stroke-width: 0.22; vector-effect: non-scaling-stroke; cursor: default; }
    .county:hover { stroke: #172026; stroke-width: 0.8; }
    .map-label { fill: var(--muted); font-size: 12px; font-weight: 700; }
    .map-frame { fill: none; stroke: #d8d3ca; stroke-width: 1; }
    .note { color: var(--muted); font-size: 12px; line-height: 1.4; margin-top: 10px; max-width: none; }
    .commentary { border-top: 1px solid #e6e0d6; margin-top: 12px; padding-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .commentary p { margin: 0 0 6px; }
    .tooltip {
      position: fixed;
      display: none;
      max-width: 300px;
      background: #172026;
      color: #fff;
      border-radius: 6px;
      padding: 9px 10px;
      font-size: 12px;
      line-height: 1.35;
      pointer-events: none;
      box-shadow: 0 6px 18px rgba(23, 32, 38, 0.22);
      z-index: 10;
    }
    .tooltip strong { display: block; margin-bottom: 4px; }
    @media (max-width: 820px) {
      main { width: min(100vw - 20px, 1180px); padding-top: 18px; }
      .chart { height: 390px; }
      .map { height: 430px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Climate Risk to Housing Markets</h1>
      <p class="dek">Housing markets across the US are affected by extreme climate events. The impact is uneven across different regions of the country. What climate risk does your home face?</p>
    </div>
  </header>

  <section>
    <div class="section-head">
      <div>
        <h2>Housing Market Response to Extreme Climate Incidents</h2>
        <div class="sub">This chart compares counties by FEMA risk level. Each line shows the typical housing market path for counties in that risk group, from one year before an incident to two years after it ends. Hover over a line to see the middle range for that group and a short before-and-after summary.</div>
      </div>
    </div>
    <svg id="riskChart" class="chart" role="img" aria-label="Housing market index around FEMA incidents grouped by NRI risk rating"></svg>
    <div id="riskLegend" class="legend"></div>
    <p class="note">Housing market index: this score combines prices, sale-to-list ratios, homes sold, and inventory into one number. The inputs use year-over-year changes so normal seasonal swings, like busier spring markets, have less influence.</p>
    <p class="note">Incident weighting: when a county was hit by multiple incidents, its values are averaged by month, with more recent incidents counted more heavily.</p>
    <div id="riskCommentary" class="commentary" aria-label="Commentary on NRI risk group responses"></div>
    <div class="map-wrap">
      <h3>County Risk Map</h3>
      <div class="sub">The same FEMA risk levels are mapped county by county across the US, including Alaska and Hawaii. Green marks the lowest-risk counties and red marks the highest-risk counties.</div>
      <svg id="riskMap" class="map" role="img" aria-label="US county map colored by FEMA National Risk Index rating"></svg>
      <div id="mapLegend" class="legend"></div>
    </div>
  </section>
  <div id="riskTooltip" class="tooltip" role="status" aria-live="polite"></div>
  <div id="mapTooltip" class="tooltip" role="status" aria-live="polite"></div>
</main>

<script>
const data = window.STORMHOUSE_DATA;
let usAtlasPromise = null;
let usCountyFeatures = null;
const colors = {
  "Very Low": "#16a34a",
  "Low": "#84cc16",
  "Moderate": "#facc15",
  "High": "#f97316",
  "Very High": "#dc2626"
};
const allOffsets = data.meta.offsets;
function extent(values) {
  let min = Infinity, max = -Infinity;
  values.forEach(v => {
    if (v == null || Number.isNaN(v)) return;
    min = Math.min(min, v);
    max = Math.max(max, v);
  });
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1, 1];
  if (min === max) return [min - 1, max + 1];
  const pad = (max - min) * 0.12;
  return [min - pad, max + pad];
}

function makeScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return value => r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

function symmetricExtent(values) {
  const [min, max] = extent(values);
  const bound = Math.max(Math.abs(min), Math.abs(max), 0.01);
  return [-bound, bound];
}

function pathFor(rows, x, y, key = "median") {
  return rows.map((d, i) => `${i === 0 ? "M" : "L"}${x(d.offset).toFixed(2)},${y(d[key]).toFixed(2)}`).join(" ");
}

function bandPath(rows, x, y) {
  const top = rows.map((d, i) => `${i === 0 ? "M" : "L"}${x(d.offset).toFixed(2)},${y(d.q3).toFixed(2)}`);
  const bottom = rows.slice().reverse().map(d => `L${x(d.offset).toFixed(2)},${y(d.q1).toFixed(2)}`);
  return `${top.join(" ")} ${bottom.join(" ")} Z`;
}

function clear(svg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}

function add(tag, parent, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  parent.appendChild(node);
  return node;
}

function drawAxes(svg, dims, x, y, xDomain, yDomain) {
  const g = add("g", svg);
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const value = yDomain[0] + ((yDomain[1] - yDomain[0]) * i / yTicks);
    const yy = y(value);
    add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: yy, y2: yy, stroke: "#e5e0d7" });
    add("text", g, { x: dims.left - 8, y: yy + 4, "text-anchor": "end" }).textContent = value.toFixed(2);
  }
  const tickValues = [-12, -6, -1, 1, 6, 12, 18, 24].filter(d => d >= xDomain[0] && d <= xDomain[1]);
  tickValues.forEach(value => {
    const xx = x(value);
    add("line", g, { x1: xx, x2: xx, y1: dims.top, y2: dims.height - dims.bottom, stroke: "#eee9df" });
    add("text", g, { x: xx, y: dims.height - dims.bottom + 22, "text-anchor": "middle" }).textContent = value;
  });
  add("line", g, { x1: dims.left, x2: dims.width - dims.right, y1: y(0), y2: y(0), class: "zero-line" });
  add("text", g, { x: dims.left, y: dims.height - 8 }).textContent = "Month offset";
  add("text", g, { x: 4, y: dims.top - 8 }).textContent = "Housing market YOY index";
}

function chartDims(svg) {
  const box = svg.getBoundingClientRect();
  return { width: Math.max(320, box.width), height: Math.max(300, box.height), top: 26, right: 22, bottom: 46, left: 58 };
}

function drawPeriodShading(svg, dims, x) {
  const y = dims.top;
  const height = dims.height - dims.bottom - dims.top;
  add("rect", svg, { x: x(1), y, width: x(12) - x(1), height, class: "period-band-early" });
  add("rect", svg, { x: x(13), y, width: x(24) - x(13), height, class: "period-band-late" });
  add("text", svg, { x: x(6.5), y: dims.top + 13, class: "period-label", "text-anchor": "middle" }).textContent = "Months 1-12";
  add("text", svg, { x: x(18.5), y: dims.top + 13, class: "period-label", "text-anchor": "middle" }).textContent = "Months 13-24";
}

function formatChange(value) {
  if (value == null || Number.isNaN(value)) return "n/a";
  const fixed = Math.abs(value).toFixed(3);
  return `${value > 0 ? "+" : value < 0 ? "-" : ""}${fixed}`;
}

function showTooltip(event, rating) {
  const tooltip = document.getElementById("riskTooltip");
  const summary = data.groupSummaries[rating] || {};
  tooltip.innerHTML = `<strong>${rating} risk</strong>` +
    `<div>Counties: ${summary.countyCount ?? "n/a"}</div>` +
    `<div>Avg. shift from before to year 1: ${formatChange(summary.avgPreToMonths1To12)}</div>` +
    `<div>Avg. shift from year 1 to year 2: ${formatChange(summary.avgMonths1To12To13To24)}</div>`;
  tooltip.style.display = "block";
  moveTooltip(event);
}

function moveTooltip(event) {
  const tooltip = document.getElementById("riskTooltip");
  const offset = 14;
  tooltip.style.left = `${Math.min(event.clientX + offset, window.innerWidth - tooltip.offsetWidth - offset)}px`;
  tooltip.style.top = `${Math.min(event.clientY + offset, window.innerHeight - tooltip.offsetHeight - offset)}px`;
}

function hideTooltip() {
  document.getElementById("riskTooltip").style.display = "none";
}

function drawRiskChart() {
  const svg = document.getElementById("riskChart");
  clear(svg);
  const dims = chartDims(svg);
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const offsets = allOffsets;
  const groups = data.meta.riskRatings.map(rating => ({
    rating,
    rows: (data.byRiskRating[rating] || []).filter(d => offsets.includes(d.offset))
  })).filter(group => group.rows.length);
  const yValues = groups.flatMap(group => group.rows.flatMap(d => [d.q1, d.q3, d.median]));
  const xDomain = [Math.min(...offsets), Math.max(...offsets)];
  const yDomain = symmetricExtent(yValues);
  const x = makeScale(xDomain, [dims.left, dims.width - dims.right]);
  const y = makeScale(yDomain, [dims.height - dims.bottom, dims.top]);
  drawPeriodShading(svg, dims, x);
  drawAxes(svg, dims, x, y, xDomain, yDomain);
  const incidentX = x(0);
  add("line", svg, { x1: incidentX, x2: incidentX, y1: dims.top, y2: dims.height - dims.bottom, class: "incident-line" });
  add("text", svg, { x: incidentX + 7, y: dims.top + 14, class: "incident-label" }).textContent = "Incident";
  groups.forEach(group => {
    const linePath = pathFor(group.rows, x, y);
    const band = add("path", svg, { d: bandPath(group.rows, x, y), class: "risk-band", fill: colors[group.rating] || "#5e6872", "data-rating": group.rating });
    add("path", svg, { d: linePath, class: "risk-line", stroke: colors[group.rating] || "#5e6872", "data-rating": group.rating });
    const hitLine = add("path", svg, { d: linePath, class: "risk-hit-line", "data-rating": group.rating });
    hitLine.addEventListener("mouseenter", event => {
      band.classList.add("active");
      showTooltip(event, group.rating);
    });
    hitLine.addEventListener("mousemove", moveTooltip);
    hitLine.addEventListener("mouseleave", () => {
      band.classList.remove("active");
      hideTooltip();
    });
  });
}

function drawLegend() {
  const legend = document.getElementById("riskLegend");
  legend.innerHTML = "";
  data.meta.riskRatings.forEach(rating => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(rating));
    legend.appendChild(item);
  });
}

function coordWalker(coords, callback) {
  if (!Array.isArray(coords)) return;
  if (typeof coords[0] === "number" && typeof coords[1] === "number") {
    callback(coords);
    return;
  }
  coords.forEach(item => coordWalker(item, callback));
}

function mapFeatureGroup(feature) {
  const stateFips = feature.properties.stateFips;
  if (stateFips === "02") return "alaska";
  if (stateFips === "15") return "hawaii";
  return "conus";
}

function normalizedLon(lon, group) {
  if (group === "alaska" && lon > 0) return lon - 360;
  return lon;
}

function albersFactory({ parallels, lat0, lon0 }) {
  const toRad = Math.PI / 180;
  const phi1 = parallels[0] * toRad;
  const phi2 = parallels[1] * toRad;
  const phi0 = lat0 * toRad;
  const lambda0 = lon0 * toRad;
  const n = 0.5 * (Math.sin(phi1) + Math.sin(phi2));
  const c = Math.cos(phi1) ** 2 + 2 * n * Math.sin(phi1);
  const rho0 = Math.sqrt(c - 2 * n * Math.sin(phi0)) / n;
  return (lon, lat) => {
    const lambda = lon * toRad;
    const phi = lat * toRad;
    const rho = Math.sqrt(Math.max(0, c - 2 * n * Math.sin(phi))) / n;
    const theta = n * (lambda - lambda0);
    return [rho * Math.sin(theta), rho0 - rho * Math.cos(theta)];
  };
}

const projectors = {
  conus: albersFactory({ parallels: [29.5, 45.5], lat0: 37.5, lon0: -96 }),
  alaska: albersFactory({ parallels: [55, 65], lat0: 50, lon0: -154 }),
  hawaii: albersFactory({ parallels: [8, 18], lat0: 20, lon0: -157 })
};

function projectedCoord(coord, group) {
  const lon = normalizedLon(coord[0], group);
  return projectors[group](lon, coord[1]);
}

function mapBounds(features) {
  const bounds = {
    conus: [Infinity, Infinity, -Infinity, -Infinity],
    alaska: [Infinity, Infinity, -Infinity, -Infinity],
    hawaii: [Infinity, Infinity, -Infinity, -Infinity]
  };
  features.forEach(feature => {
    const group = mapFeatureGroup(feature);
    coordWalker(feature.geometry.coordinates, coord => {
      const [x, y] = projectedCoord(coord, group);
      bounds[group][0] = Math.min(bounds[group][0], x);
      bounds[group][1] = Math.min(bounds[group][1], y);
      bounds[group][2] = Math.max(bounds[group][2], x);
      bounds[group][3] = Math.max(bounds[group][3], y);
    });
  });
  return bounds;
}

function makeMapProjector(features, dims) {
  const bounds = mapBounds(features);
  const panels = {
    conus: { x: 34, y: 24, width: dims.width - 68, height: dims.height - 145 },
    alaska: { x: 58, y: dims.height - 124, width: 260, height: 92 },
    hawaii: { x: 360, y: dims.height - 105, width: 178, height: 62 }
  };
  const fitted = {};
  Object.entries(panels).forEach(([group, panel]) => {
    const [minX, minY, maxX, maxY] = bounds[group];
    const xSpan = Math.max(maxX - minX, 0.001);
    const ySpan = Math.max(maxY - minY, 0.001);
    const scale = Math.min(panel.width / xSpan, panel.height / ySpan);
    const drawnWidth = xSpan * scale;
    const drawnHeight = ySpan * scale;
    fitted[group] = {
      x: panel.x + (panel.width - drawnWidth) / 2,
      y: panel.y + (panel.height - drawnHeight) / 2,
      scale,
      minX,
      maxY
    };
  });
  return (coord, group) => {
    const [px, py] = projectedCoord(coord, group);
    const panel = fitted[group];
    const x = panel.x + (px - panel.minX) * panel.scale;
    const y = panel.y + (panel.maxY - py) * panel.scale;
    return [x, y];
  };
}

function pathFromRing(ring, project, group) {
  return ring.map((coord, index) => {
    const [x, y] = project(coord, group);
    return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ") + " Z";
}

function pathFromGeometry(feature, project) {
  const group = mapFeatureGroup(feature);
  const geometry = feature.geometry;
  if (geometry.type === "Polygon") {
    return geometry.coordinates.map(ring => pathFromRing(ring, project, group)).join(" ");
  }
  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.flatMap(poly => poly.map(ring => pathFromRing(ring, project, group))).join(" ");
  }
  return "";
}

function showMapTooltip(event, feature) {
  const tooltip = document.getElementById("mapTooltip");
  const props = feature.properties;
  tooltip.innerHTML = `<strong>${props.name}</strong>` +
    `<div>NRI risk rating: ${props.riskRating}</div>`;
  tooltip.style.display = "block";
  moveMapTooltip(event);
}

function moveMapTooltip(event) {
  const tooltip = document.getElementById("mapTooltip");
  const offset = 14;
  tooltip.style.left = `${Math.min(event.clientX + offset, window.innerWidth - tooltip.offsetWidth - offset)}px`;
  tooltip.style.top = `${Math.min(event.clientY + offset, window.innerHeight - tooltip.offsetHeight - offset)}px`;
}

function hideMapTooltip() {
  document.getElementById("mapTooltip").style.display = "none";
}

function padFips(value) {
  return String(value ?? "").padStart(5, "0");
}

async function loadUsCountyFeatures() {
  if (usCountyFeatures) return usCountyFeatures;
  if (!usAtlasPromise) {
    usAtlasPromise = d3.json("https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json");
  }
  const usTopo = await usAtlasPromise;
  const riskByFips = new Map(data.countyRiskMap.features.map(feature => [feature.properties.fips, feature.properties]));
  usCountyFeatures = topojson
    .feature(usTopo, usTopo.objects.counties)
    .features
    .filter(feature => riskByFips.has(padFips(feature.id)))
    .map(feature => ({
      ...feature,
      properties: {
        ...feature.properties,
        ...riskByFips.get(padFips(feature.id)),
      },
    }));
  return usCountyFeatures;
}

async function drawCountyRiskMap() {
  const svg = document.getElementById("riskMap");
  clear(svg);
  const box = svg.getBoundingClientRect();
  const dims = { width: Math.max(360, box.width), height: Math.max(360, box.height) };
  svg.setAttribute("viewBox", `0 0 ${dims.width} ${dims.height}`);
  const features = await loadUsCountyFeatures();
  const projection = d3.geoAlbersUsa();
  projection.fitExtent(
    [[34, 24], [dims.width - 34, dims.height - 34]],
    { type: "FeatureCollection", features }
  );
  const geoPath = d3.geoPath(projection);
  const layer = add("g", svg);
  features.forEach(feature => {
    const rating = feature.properties.riskRating;
    const countyPath = add("path", layer, {
      d: geoPath(feature),
      class: "county",
      fill: colors[rating] || "transparent",
      "data-rating": rating
    });
    countyPath.addEventListener("mouseenter", event => showMapTooltip(event, feature));
    countyPath.addEventListener("mousemove", moveMapTooltip);
    countyPath.addEventListener("mouseleave", hideMapTooltip);
  });
}

function drawMapLegend() {
  const legend = document.getElementById("mapLegend");
  const counts = data.countyRiskMap.counts || {};
  legend.innerHTML = "";
  data.meta.riskRatings.forEach(rating => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colors[rating] || colors.Unrated;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(`${rating} (${counts[rating] ?? 0})`));
    legend.appendChild(item);
  });
}

function redraw() {
  drawRiskChart();
  drawCountyRiskMap();
}

window.addEventListener("resize", redraw);
drawLegend();
drawMapLegend();
document.getElementById("riskCommentary").innerHTML = data.commentary.map(text => `<p>${text}</p>`).join("");
redraw();
</script>
</body>
</html>
"""


def build_stormhouse_page() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = build_complete_windows()
    payload = build_payload(windows)
    DATA_JS.write_text("window.STORMHOUSE_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    HTML_PATH.write_text(HTML, encoding="utf-8")
    return {
        "page": "stormhouse",
        "html_file": HTML_PATH.name,
        "data_file": DATA_JS.name,
        "written_paths": [HTML_PATH, DATA_JS],
        "meta": payload["meta"],
    }
