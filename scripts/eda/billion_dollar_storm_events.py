"""Visualize county-year storm records with more than $1B in damage.

This script reads
``data/20260401_county_processed_data/county_processed_data.feather``, flattens the nested
``storm_events`` records, filters county-year observations whose
``total_damage_total`` exceeds $1 billion, and writes a standalone HTML
dashboard plus a CSV extract.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    ROOT
    / "data"
    / "20260401_county_processed_data"
    / "county_processed_data.feather"
)
DEFAULT_HTML_PATH = Path(__file__).with_name("billion_dollar_storm_events.html")
DEFAULT_CSV_PATH = Path(__file__).with_name("billion_dollar_storm_events.csv")
DEFAULT_THRESHOLD = 1_000_000_000

ID_COLUMNS = ["fips", "county_name", "state", "state_long"]
DAMAGE_COLUMN = "total_damage_total"
EVENT_COUNT_BY_TYPE_COLUMN = "total_event_count_by_type"


def iter_records(values: Any) -> Iterable[Mapping[str, Any]]:
    """Yield dict-like records from Arrow, NumPy, list, or scalar objects."""

    if values is None:
        return
    if isinstance(values, float) and np.isnan(values):
        return
    if isinstance(values, Mapping):
        yield values
        return
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        for item in values:
            if isinstance(item, Mapping):
                yield item


def flatten_storm_events(counties: pd.DataFrame) -> pd.DataFrame:
    """Return one row per county-year storm event summary."""

    if "storm_events" not in counties.columns:
        raise ValueError("Expected a 'storm_events' column in the county data.")

    available_id_columns = [column for column in ID_COLUMNS if column in counties.columns]
    rows: list[dict[str, Any]] = []
    for county in counties.to_dict(orient="records"):
        base = {column: county.get(column) for column in available_id_columns}
        for record in iter_records(county.get("storm_events")):
            row = dict(base)
            row.update(dict(record))
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=available_id_columns)

    storm_events = pd.DataFrame(rows)
    if DAMAGE_COLUMN not in storm_events.columns:
        raise ValueError(f"Expected '{DAMAGE_COLUMN}' inside storm_events records.")
    storm_events[DAMAGE_COLUMN] = pd.to_numeric(storm_events[DAMAGE_COLUMN], errors="coerce").fillna(0)
    if "year" in storm_events.columns:
        storm_events["year"] = pd.to_numeric(storm_events["year"], errors="coerce").astype("Int64")
    return storm_events


def event_types_from_counts(value: Any) -> list[dict[str, Any]]:
    """Return event types with a positive count for a county-year storm record."""

    if not isinstance(value, Mapping):
        return []
    event_types = []
    for event_type, count in value.items():
        if count is None:
            continue
        numeric_count = pd.to_numeric(count, errors="coerce")
        if pd.notna(numeric_count) and numeric_count > 0:
            event_types.append({"event_type": str(event_type), "event_count": float(numeric_count)})
    return event_types


def build_payload(storm_events: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Create the JSON payload consumed by the standalone HTML dashboard."""

    filtered = storm_events.loc[storm_events[DAMAGE_COLUMN] > threshold].copy()
    filtered["damage_billions"] = filtered[DAMAGE_COLUMN] / 1_000_000_000
    filtered["county_label"] = filtered.apply(format_county_label, axis=1)
    filtered = filtered.sort_values([DAMAGE_COLUMN, "year"], ascending=[False, True])

    records = []
    type_records = []
    for row in filtered.to_dict(orient="records"):
        slim = {
            "fips": clean_scalar(row.get("fips")),
            "county_name": clean_scalar(row.get("county_name")),
            "state": clean_scalar(row.get("state")),
            "state_long": clean_scalar(row.get("state_long")),
            "county_label": clean_scalar(row.get("county_label")),
            "year": clean_scalar(row.get("year")),
            "damage": clean_scalar(row.get(DAMAGE_COLUMN)),
            "damage_billions": clean_scalar(row.get("damage_billions")),
            "property_damage": clean_scalar(row.get("total_damage_property")),
            "crop_damage": clean_scalar(row.get("total_damage_crops")),
            "total_event_count": clean_scalar(row.get("total_event_count")),
        }
        event_types = event_types_from_counts(row.get(EVENT_COUNT_BY_TYPE_COLUMN))
        slim["event_types"] = [item["event_type"] for item in event_types]
        records.append(slim)
        for item in event_types:
            typed = dict(slim)
            typed["event_type"] = item["event_type"]
            typed["event_count_for_type"] = item["event_count"]
            type_records.append(typed)

    years = sorted({record["year"] for record in records if record["year"] is not None})
    event_types = sorted({record["event_type"] for record in type_records})
    totals_by_year = summarize(records, "year")
    totals_by_type = summarize(type_records, "event_type")

    return {
        "threshold": threshold,
        "records": records,
        "typeRecords": type_records,
        "years": years,
        "eventTypes": event_types,
        "totalsByYear": totals_by_year,
        "totalsByType": totals_by_type,
        "notes": {
            "damage": "Damage is the county-year total_damage_total field.",
            "eventTypes": "Event types are taken from total_event_count_by_type counts greater than zero.",
        },
    }


def summarize(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        item = grouped.setdefault(value, {"key": value, "damage": 0.0, "count": 0})
        item["damage"] += float(record.get("damage") or 0)
        item["count"] += 1
    return sorted(grouped.values(), key=lambda item: item["damage"], reverse=True)


def clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def format_county_label(row: pd.Series) -> str:
    county = row.get("county_name")
    state = row.get("state")
    if pd.notna(county) and pd.notna(state):
        return f"{county}, {state}"
    if pd.notna(county):
        return str(county)
    if pd.notna(row.get("fips")):
        return str(row.get("fips"))
    return "Unknown county"


def write_outputs(payload: dict[str, Any], csv_path: Path, html_path: Path) -> None:
    records = pd.DataFrame(payload["records"])
    records.to_csv(csv_path, index=False)
    html_path.write_text(render_html(payload), encoding="utf-8")


def render_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Billion-dollar county storm damage</title>
  <link rel="icon" href="data:,">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #1d2433;
      --muted: #667085;
      --line: #d9dee8;
      --blue: #2764c7;
      --red: #c83f3f;
      --green: #167a5b;
      --gold: #a15c00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; line-height: 1.1; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 24px 32px 36px; }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
      align-items: end;
    }}
    label {{ display: grid; gap: 6px; font-size: 12px; font-weight: 700; color: #344054; }}
    select, input {{
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 10px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .stat, .chart, .table-wrap {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .stat {{ padding: 14px 16px; }}
    .stat strong {{ display: block; font-size: 24px; line-height: 1.1; }}
    .stat span {{ color: var(--muted); font-size: 12px; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      margin-bottom: 16px;
    }}
    .chart {{ padding: 16px; min-height: 360px; }}
    .chart h2, .table-wrap h2 {{ margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }}
    svg {{ width: 100%; height: 300px; display: block; }}
    .axis {{ stroke: var(--line); stroke-width: 1; }}
    .tick text, .label {{ fill: var(--muted); font-size: 11px; }}
    .bar {{ fill: var(--blue); }}
    .bar.alt {{ fill: var(--green); }}
    .dot {{ fill: var(--red); stroke: #ffffff; stroke-width: 1.5; }}
    .legend {{ display: flex; gap: 14px; color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .legend b {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }}
    .table-wrap {{ padding: 16px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 8px; border-bottom: 1px solid #edf0f5; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: var(--panel); font-size: 12px; color: #344054; }}
    td.numeric, th.numeric {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .empty {{ padding: 40px; text-align: center; color: var(--muted); }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .controls, .stats, .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Billion-dollar county storm damage</h1>
    <p>County-year observations from <code>storm_events</code> where total damage exceeds $1 billion.</p>
  </header>
  <main>
    <section class="controls" aria-label="Filters">
      <label>Event type
        <select id="eventType"></select>
      </label>
      <label>Year
        <select id="year"></select>
      </label>
      <label>State
        <select id="state"></select>
      </label>
      <label>Minimum damage, $B
        <input id="minDamage" type="number" min="1" step="0.25" value="1">
      </label>
    </section>
    <section class="stats" id="stats"></section>
    <section class="grid">
      <article class="chart">
        <h2>Damage by year</h2>
        <svg id="yearChart" role="img" aria-label="Damage by year"></svg>
      </article>
      <article class="chart">
        <h2>Damage by event type</h2>
        <svg id="typeChart" role="img" aria-label="Damage by event type"></svg>
        <div class="legend"><span><b style="background: var(--green)"></b>Damage is county-year total for records containing the event type</span></div>
      </article>
    </section>
    <section class="chart">
      <h2>County-year events</h2>
      <svg id="scatterChart" role="img" aria-label="County-year damage points"></svg>
    </section>
    <section class="table-wrap">
      <h2>Filtered county-year observations</h2>
      <div id="table"></div>
    </section>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const records = payload.records;
    const typeRecords = payload.typeRecords;
    const eventTypeSelect = document.getElementById("eventType");
    const yearSelect = document.getElementById("year");
    const stateSelect = document.getElementById("state");
    const minDamageInput = document.getElementById("minDamage");

    function unique(values) {{
      return [...new Set(values.filter((value) => value !== null && value !== undefined && value !== ""))].sort();
    }}

    function addOptions(select, values, allLabel) {{
      select.innerHTML = "";
      const all = document.createElement("option");
      all.value = "All";
      all.textContent = allLabel;
      select.appendChild(all);
      values.forEach((value) => {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }});
    }}

    addOptions(eventTypeSelect, payload.eventTypes, "All event types");
    addOptions(yearSelect, payload.years, "All years");
    addOptions(stateSelect, unique(records.map((record) => record.state)), "All states");

    [eventTypeSelect, yearSelect, stateSelect, minDamageInput].forEach((element) => element.addEventListener("input", render));

    function filteredRecords() {{
      const eventType = eventTypeSelect.value;
      const year = yearSelect.value;
      const state = stateSelect.value;
      const minDamage = Number(minDamageInput.value || 1);
      return records.filter((record) => {{
        const matchesType = eventType === "All" || record.event_types.includes(eventType);
        const matchesYear = year === "All" || String(record.year) === year;
        const matchesState = state === "All" || record.state === state;
        return matchesType && matchesYear && matchesState && record.damage_billions >= minDamage;
      }});
    }}

    function filteredTypeRecords(baseRecords) {{
      const keys = new Set(baseRecords.map((record) => `${{record.fips}}|${{record.year}}|${{record.damage}}`));
      const selectedType = eventTypeSelect.value;
      return typeRecords.filter((record) => {{
        const key = `${{record.fips}}|${{record.year}}|${{record.damage}}`;
        return keys.has(key) && (selectedType === "All" || record.event_type === selectedType);
      }});
    }}

    function currencyBillions(value) {{
      return `$${{Number(value).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}B`;
    }}

    function compactNumber(value) {{
      return Number(value).toLocaleString(undefined, {{maximumFractionDigits: 0}});
    }}

    function groupBy(records, key) {{
      const grouped = new Map();
      records.forEach((record) => {{
        const value = record[key];
        if (value === null || value === undefined) return;
        const item = grouped.get(value) || {{key: value, damage: 0, count: 0}};
        item.damage += Number(record.damage || 0);
        item.count += 1;
        grouped.set(value, item);
      }});
      return [...grouped.values()].sort((a, b) => b.damage - a.damage);
    }}

    function renderStats(current) {{
      const totalDamage = current.reduce((sum, record) => sum + Number(record.damage || 0), 0) / 1e9;
      const counties = unique(current.map((record) => record.county_label)).length;
      const years = unique(current.map((record) => record.year)).length;
      const states = unique(current.map((record) => record.state)).length;
      document.getElementById("stats").innerHTML = `
        <div class="stat"><strong>${{current.length}}</strong><span>county-year observations</span></div>
        <div class="stat"><strong>${{currencyBillions(totalDamage)}}</strong><span>total damage</span></div>
        <div class="stat"><strong>${{counties}}</strong><span>counties</span></div>
        <div class="stat"><strong>${{years}}</strong><span>years across ${{states}} states</span></div>
      `;
    }}

    function chartFrame(svg, title) {{
      svg.innerHTML = "";
      const width = svg.clientWidth || 700;
      const height = 300;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      if (!title) return {{width, height, margin: {{top: 16, right: 18, bottom: 44, left: 58}}}};
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", width / 2);
      text.setAttribute("y", height / 2);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("class", "label");
      text.textContent = title;
      svg.appendChild(text);
      return null;
    }}

    function renderBarChart(svgId, data, options = {{}}) {{
      const svg = document.getElementById(svgId);
      if (!data.length) return chartFrame(svg, "No records match the current filters");
      const frame = chartFrame(svg);
      const {{width, height, margin}} = frame;
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const maxDamage = Math.max(...data.map((item) => item.damage / 1e9));
      const barGap = 4;
      const barWidth = Math.max(5, plotWidth / data.length - barGap);
      axis(svg, margin.left, margin.top + plotHeight, margin.left + plotWidth, margin.top + plotHeight);
      axis(svg, margin.left, margin.top, margin.left, margin.top + plotHeight);
      data.forEach((item, index) => {{
        const value = item.damage / 1e9;
        const heightValue = maxDamage ? (value / maxDamage) * plotHeight : 0;
        const x = margin.left + index * (plotWidth / data.length) + barGap / 2;
        const y = margin.top + plotHeight - heightValue;
        rect(svg, x, y, barWidth, heightValue, options.alt ? "bar alt" : "bar", `${{item.key}}: ${{currencyBillions(value)}}`);
        if (data.length <= 18 || index % Math.ceil(data.length / 14) === 0) {{
          text(svg, x + barWidth / 2, margin.top + plotHeight + 18, String(item.key).replaceAll("_", " "), "middle", "label", -35);
        }}
      }});
      text(svg, margin.left - 8, margin.top + 8, currencyBillions(maxDamage), "end", "label");
    }}

    function renderScatter(current) {{
      const svg = document.getElementById("scatterChart");
      if (!current.length) return chartFrame(svg, "No records match the current filters");
      const frame = chartFrame(svg);
      const {{width, height}} = frame;
      const margin = {{top: 16, right: 18, bottom: 44, left: 185}};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const years = unique(current.map((record) => record.year)).map(Number);
      const counties = unique(current.map((record) => record.county_label));
      const minYear = Math.min(...years);
      const maxYear = Math.max(...years);
      const maxDamage = Math.max(...current.map((record) => record.damage_billions));
      axis(svg, margin.left, margin.top + plotHeight, margin.left + plotWidth, margin.top + plotHeight);
      axis(svg, margin.left, margin.top, margin.left, margin.top + plotHeight);
      current.forEach((record) => {{
        const x = margin.left + ((Number(record.year) - minYear) / Math.max(1, maxYear - minYear)) * plotWidth;
        const rank = counties.indexOf(record.county_label);
        const y = margin.top + (rank / Math.max(1, counties.length - 1)) * plotHeight;
        const radius = 4 + Math.sqrt(record.damage_billions / maxDamage) * 10;
        circle(svg, x, y, radius, `${{record.county_label}} ${{record.year}}: ${{currencyBillions(record.damage_billions)}}`);
      }});
      years.forEach((year) => {{
        const x = margin.left + ((year - minYear) / Math.max(1, maxYear - minYear)) * plotWidth;
        text(svg, x, margin.top + plotHeight + 22, year, "middle", "label");
      }});
      counties.slice(0, 10).forEach((county, index) => {{
        const y = margin.top + (index / Math.max(1, counties.length - 1)) * plotHeight;
        text(svg, margin.left - 8, y + 4, county, "end", "label");
      }});
    }}

    function axis(svg, x1, y1, x2, y2) {{
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
      line.setAttribute("class", "axis");
      svg.appendChild(line);
    }}

    function rect(svg, x, y, width, height, className, titleText) {{
      const item = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      item.setAttribute("x", x);
      item.setAttribute("y", y);
      item.setAttribute("width", width);
      item.setAttribute("height", Math.max(1, height));
      item.setAttribute("class", className);
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = titleText;
      item.appendChild(title);
      svg.appendChild(item);
    }}

    function circle(svg, x, y, radius, titleText) {{
      const item = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      item.setAttribute("cx", x);
      item.setAttribute("cy", y);
      item.setAttribute("r", radius);
      item.setAttribute("class", "dot");
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = titleText;
      item.appendChild(title);
      svg.appendChild(item);
    }}

    function text(svg, x, y, value, anchor = "start", className = "label", rotate = 0) {{
      const item = document.createElementNS("http://www.w3.org/2000/svg", "text");
      item.setAttribute("x", x);
      item.setAttribute("y", y);
      item.setAttribute("text-anchor", anchor);
      item.setAttribute("class", className);
      if (rotate) item.setAttribute("transform", `rotate(${{rotate}} ${{x}} ${{y}})`);
      item.textContent = value;
      svg.appendChild(item);
    }}

    function renderTable(current) {{
      const rows = current.slice().sort((a, b) => b.damage - a.damage).map((record) => `
        <tr>
          <td>${{record.county_label}}</td>
          <td>${{record.year ?? ""}}</td>
          <td>${{record.event_types.join(", ").replaceAll("_", " ")}}</td>
          <td class="numeric">${{currencyBillions(record.damage_billions)}}</td>
          <td class="numeric">${{compactNumber(record.total_event_count || 0)}}</td>
        </tr>
      `).join("");
      document.getElementById("table").innerHTML = rows ? `
        <table>
          <thead><tr><th>County</th><th>Year</th><th>Event types present</th><th class="numeric">Damage</th><th class="numeric">Event count</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      ` : `<div class="empty">No records match the current filters.</div>`;
    }}

    function render() {{
      const current = filteredRecords();
      const typed = filteredTypeRecords(current);
      renderStats(current);
      renderBarChart("yearChart", groupBy(current, "year").sort((a, b) => a.key - b.key));
      renderBarChart("typeChart", groupBy(typed, "event_type").slice(0, 24), {{alt: true}});
      renderScatter(current);
      renderTable(current);
    }}

    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--html-path", type=Path, default=DEFAULT_HTML_PATH)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counties = pd.read_feather(args.data_path)
    storm_events = flatten_storm_events(counties)
    payload = build_payload(storm_events, args.threshold)
    write_outputs(payload, args.csv_path, args.html_path)
    print(f"Wrote {len(payload['records']):,} county-year records to {args.csv_path}")
    print(f"Wrote dashboard to {args.html_path}")


if __name__ == "__main__":
    main()
