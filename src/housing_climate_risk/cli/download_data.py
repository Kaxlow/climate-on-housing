from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from housing_climate_risk.cli import (
    acs_data,
    climate_damage_data,
    ncei_county_weather_data,
    statsamerica_bea_cew,
    statsamerica_population_components,
)


DOWNLOADERS: dict[str, tuple[str, Callable[[], None]]] = {
    "acs": ("Census ACS county tables", acs_data.main),
    "climate-damage": ("NOAA Storm Events county damage data", climate_damage_data.main),
    "ncei-weather": ("NOAA NCEI county monthly weather data", ncei_county_weather_data.main),
    "statsamerica-bea-cew": ("StatsAmerica BEA and CEW datasets", statsamerica_bea_cew.main),
    "statsamerica-population": (
        "StatsAmerica Components of Population Change",
        statsamerica_population_components.main,
    ),
}


def parse_args(argv: list[str] | None = None) -> tuple[str, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in DOWNLOADERS:
        return values[0], values[1:]

    parser = argparse.ArgumentParser(
        description="Download raw source data used by the Quoll Intelligence DuckDB pipeline.",
    )
    parser.add_argument(
        "source",
        choices=DOWNLOADERS,
        help="Data source to download. Run `download-data SOURCE --help` for source-specific options.",
    )
    args, source_args = parser.parse_known_args(values)
    return args.source, source_args


def main() -> None:
    source, source_args = parse_args()
    _, downloader = DOWNLOADERS[source]
    sys.argv = [f"download-data {source}", *source_args]
    downloader()


if __name__ == "__main__":
    main()
