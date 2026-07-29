from __future__ import annotations

from housing_climate_risk.cli.download_data import parse_args


def test_parse_args_accepts_new_public_sources_and_all() -> None:
    assert parse_args(["fema-nri", "--force"]) == ("fema-nri", ["--force"])
    assert parse_args(["fema-declarations"]) == ("fema-declarations", [])
    assert parse_args(["census-boundaries"]) == ("census-boundaries", [])
    assert parse_args(["all", "--fail-fast"]) == ("all", ["--fail-fast"])
