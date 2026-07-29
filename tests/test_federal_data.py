from __future__ import annotations

import io
import zipfile

import pandas as pd
import yaml

from housing_climate_risk.cli import federal_data


def _zip_with(name: str, content: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


def test_safe_extract_csv_flattens_expected_member(tmp_path) -> None:
    body = _zip_with("nested/NRI_Table_Counties.csv", "STATE,NRI_VER\nAlabama,1.20\n")
    output = federal_data._safe_extract_csv(
        body, tmp_path, "NRI_Table_Counties.csv"
    )
    assert output == tmp_path / "NRI_Table_Counties.csv"
    assert output.read_text(encoding="utf-8").startswith("STATE,NRI_VER")


def test_write_receipt_entry_preserves_existing_downloads(tmp_path) -> None:
    receipt = tmp_path / "receipt.yaml"
    federal_data._write_receipt_entry(
        "first", {"source_url": "https://example.test/one"}, receipt
    )
    federal_data._write_receipt_entry(
        "second", {"source_url": "https://example.test/two"}, receipt
    )
    values = yaml.safe_load(receipt.read_text(encoding="utf-8"))
    assert set(values["downloads"]) == {"first", "second"}
    assert values["downloads"]["first"]["retrieved_at"].endswith("+00:00")


def test_download_disaster_declarations_paginates_and_records_receipt(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "data" / "fema" / "FEMA_Disaster_Declarations.csv"
    monkeypatch.setattr(federal_data, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(federal_data, "RECEIPT_PATH", tmp_path / "data" / "receipt.yaml")
    pages = [
        {
            "metadata": {"rundate": "2026-01-01T00:00:00Z"},
            "DisasterDeclarationsSummaries": [
                {
                    "femaDeclarationString": "DR-1-AA",
                    "disasterNumber": 1,
                    "state": "AA",
                    "declarationType": "DR",
                    "declarationDate": "2026-01-01",
                    "incidentType": "Storm",
                    "incidentBeginDate": "2025-12-31",
                    "fipsStateCode": "01",
                    "fipsCountyCode": "001",
                    "designatedArea": "Example",
                    "lastRefresh": "2026-01-02",
                }
            ],
        },
        {"metadata": {}, "DisasterDeclarationsSummaries": []},
    ]

    def fake_request(url: str, *, timeout: int = 300):
        return pages.pop(0), url

    monkeypatch.setattr(federal_data, "_request_json", fake_request)
    monkeypatch.setattr(federal_data, "FEMA_DECLARATION_REQUIRED_COLUMNS", {
        "femaDeclarationString", "disasterNumber", "state", "declarationType",
        "declarationDate", "incidentType", "incidentBeginDate", "fipsStateCode",
        "fipsCountyCode", "designatedArea", "lastRefresh",
    })

    result = federal_data.download_fema_disaster_declarations(force=True)
    assert result == destination
    assert pd.read_csv(result).shape[0] == 1
    receipt = yaml.safe_load((tmp_path / "data" / "receipt.yaml").read_text())
    assert receipt["downloads"]["fema_disaster_declarations"]["api_version"] == "v2"
