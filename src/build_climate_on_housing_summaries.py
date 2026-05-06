"""
Build county-level visualization summaries from notebook-exported housing data.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = {
    "AVG_SALE_TO_LIST": {
        "value_col": "AVG_SALE_TO_LIST",
        "change_col": "AVG_SALE_TO_LIST_change_in_yoy_12_to_24",
    },
    "HOMES_SOLD": {
        "value_col": "HOMES_SOLD",
        "change_col": "HOMES_SOLD_change_in_yoy_12_to_24",
    },
    "INVENTORY": {
        "value_col": "INVENTORY",
        "change_col": "INVENTORY_change_in_yoy_12_to_24",
    },
    "MEDIAN_PPSF": {
        "value_col": "MEDIAN_PPSF",
        "change_col": "MEDIAN_PPSF_change_in_yoy_12_to_24",
    },
    "HOUSING_MARKET_INDEX": {
        "value_col": "HOUSING_MARKET_INDEX",
        "change_col": "HOUSING_MARKET_INDEX_change_in_yoy_12_to_24",
    },
}
REQUIRED_BEFORE = {-(i + 1) for i in range(12)}
REQUIRED_AFTER = {i + 1 for i in range(24)}
CLUSTER_COLUMNS = [
    "median_ppsf_response_cluster",
    "median_ppsf_response_cluster_name",
    "median_ppsf_response_cluster_interpretation",
    "median_ppsf_response_cluster_algorithm",
    "median_ppsf_response_cluster_k",
    "median_ppsf_response_cluster_silhouette",
    "median_ppsf_response_incident_count",
]


def parse_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: object) -> int | None:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else None


def normalize_fips(value: object) -> str:
    return str(value or "").split(".")[0].zfill(5)


def is_county_observation(row: dict[str, str]) -> bool:
    fips = normalize_fips(row.get("fips"))
    return len(fips) == 5 and fips[2:] != "000"


def has_complete_metric_coverage(rows: list[dict[str, str]], metric_value_col: str) -> bool:
    offsets = set()
    for row in rows:
        offset = parse_int(row.get("month_offset_from_incident"))
        metric_value = parse_float(row.get(metric_value_col))
        if offset is None or offset == 0 or metric_value is None:
            continue
        offsets.add(offset)
    return REQUIRED_BEFORE.issubset(offsets) and REQUIRED_AFTER.issubset(offsets)


def weighted_change(incidents: list[dict[str, object]]) -> tuple[float | None, int]:
    if not incidents:
        return None, 0
    incidents = sorted(incidents, key=lambda incident: incident["incident_num"])
    total_weight = len(incidents) * (len(incidents) + 1) / 2
    value = sum((idx + 1) * incident["change_value"] for idx, incident in enumerate(incidents)) / total_weight
    return value, len(incidents)


def cluster_fields_from_row(row: dict[str, str]) -> dict[str, str]:
    return {
        column: row.get(column, "")
        for column in CLUSTER_COLUMNS
        if row.get(column, "") not in ("", None)
    }


def build_county_summary_rows(input_path: Path) -> list[dict[str, object]]:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    incident_rows: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not is_county_observation(row):
            continue
        fips = normalize_fips(row.get("fips"))
        incident_num = parse_int(row.get("incident_num"))
        if incident_num is None:
            continue
        row["fips"] = fips
        incident_rows[(fips, incident_num)].append(row)

    incidents_by_county: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (fips, incident_num), grouped_rows in incident_rows.items():
        latest_row = grouped_rows[-1]
        incident_summary: dict[str, object] = {
            "fips": fips,
            "incident_num": incident_num,
            "county_name": latest_row.get("REGION") or latest_row.get("county_name") or fips,
            "county_profile": parse_int(latest_row.get("county_profile")),
            "cluster_fields": cluster_fields_from_row(latest_row),
            "metrics": {},
        }

        for metric_name, metric_config in METRICS.items():
            change_col = metric_config["change_col"]
            metric_value_col = metric_config["value_col"]
            representative = next(
                (row for row in grouped_rows if parse_float(row.get(change_col)) is not None),
                None,
            )
            if representative is None:
                continue

            incident_summary["metrics"][metric_name] = {
                "change_value": parse_float(representative.get(change_col)),
                "complete": has_complete_metric_coverage(grouped_rows, metric_value_col),
            }

        incidents_by_county[fips].append(incident_summary)

    summary_rows: list[dict[str, object]] = []
    for fips, incidents in sorted(incidents_by_county.items()):
        incidents.sort(key=lambda incident: incident["incident_num"])
        representative = incidents[-1]
        row: dict[str, object] = {
            "fips": fips,
            "county_name": representative["county_name"],
            "county_profile": representative["county_profile"],
        }
        row.update(representative["cluster_fields"])

        has_any_metric = False
        for metric_name in METRICS:
            all_metric_incidents = [
                {
                    "incident_num": incident["incident_num"],
                    "change_value": incident["metrics"][metric_name]["change_value"],
                }
                for incident in incidents
                if metric_name in incident["metrics"]
                and incident["metrics"][metric_name]["change_value"] is not None
            ]
            complete_metric_incidents = [
                {
                    "incident_num": incident["incident_num"],
                    "change_value": incident["metrics"][metric_name]["change_value"],
                }
                for incident in incidents
                if metric_name in incident["metrics"]
                and incident["metrics"][metric_name]["change_value"] is not None
                and incident["metrics"][metric_name]["complete"]
            ]

            all_change, all_count = weighted_change(all_metric_incidents)
            complete_change, complete_count = weighted_change(complete_metric_incidents)

            row[f"{metric_name}_change_all"] = "" if all_change is None else all_change
            row[f"{metric_name}_incident_count_all"] = all_count
            row[f"{metric_name}_change_complete"] = "" if complete_change is None else complete_change
            row[f"{metric_name}_incident_count_complete"] = complete_count
            has_any_metric = has_any_metric or all_count > 0 or complete_count > 0

        if has_any_metric:
            summary_rows.append(row)

    return summary_rows


def write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "fips",
        "county_name",
        "county_profile",
    ]
    fieldnames.extend(CLUSTER_COLUMNS)
    for metric_name in METRICS:
        fieldnames.extend(
            [
                f"{metric_name}_change_all",
                f"{metric_name}_incident_count_all",
                f"{metric_name}_change_complete",
                f"{metric_name}_incident_count_complete",
            ]
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = Path("output/visualizations")
    manifest_path = output_dir / "incident_housing_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cluster_summary_path = output_dir / "ppsf_response_cluster_summaries.json"

    for entry in manifest.get("incident_types", []):
        input_name = entry.get("housing_24mths_csv")
        if not input_name:
            continue
        input_path = output_dir / input_name
        if not input_path.exists():
            continue

        summary_name = input_name.replace("_housing_24mths.csv", "_county_summary.csv")
        summary_path = output_dir / summary_name
        summary_rows = build_county_summary_rows(input_path)
        write_summary_csv(summary_path, summary_rows)
        entry["county_summary_csv"] = summary_name
        if cluster_summary_path.exists():
            entry["ppsf_response_cluster_summary_json"] = cluster_summary_path.name

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
