"""Static publishing contract, independent of local source databases."""
import json
from pathlib import Path
import re
import tempfile
import unittest

from housing_climate_risk.page_data import climate_risk_housing as page


class PageBundleTests(unittest.TestCase):
    def test_split_bundle_and_legacy_redirects(self):
        rows = {"Low": [{"fips": "01001", "target": 0.1}]}
        data = {
            "priceRisk": {"label": "</script>"}, "playbook": None,
            "features": {"countyRowsByRisk": rows, "scatterRowsByRisk": rows},
            "eventWindows": {"windowA": {}},
            "geojson": {"features": []}, "stateGeojson": {"features": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page.write_page_assets(data, root)
            html = (root / "index.html").read_text(encoding="utf-8")
            payload = json.loads(re.search(r'type="application/json">(.*?)</script>', html).group(1))
            self.assertEqual(set(payload), {"priceRisk", "playbook"})
            self.assertEqual(payload["priceRisk"]["label"], "</script>")
            self.assertNotIn("function draw", html)
            self.assertNotIn("<style>", html)
            for asset in ("climate-risk-housing.js", "climate-risk-housing.css"):
                self.assertEqual((root / asset).read_text(encoding="utf-8"),
                                 (page.WEB_ASSET_DIR / asset).read_text(encoding="utf-8"))
            features = (root / "climate-risk-housing-features.js").read_text()
            self.assertNotIn("scatterRowsByRisk", features)
            self.assertIn("countyRowsByRisk", features)
            self.assertIn("scatterRowsByRisk", data["features"])  # No mutation.
            for name, destination in (("climate-risk-housing.html", "index.html"),
                                      ("output/climate-risk-housing.html", "../index.html")):
                redirect = (root / name).read_text()
                self.assertIn(f'location.replace("{destination}" + location.search + location.hash)', redirect)

    def test_committed_assets_match_sources(self):
        root = page.OUT_PATH.parent
        for name in ("climate-risk-housing.js", "climate-risk-housing.css"):
            self.assertEqual((root / name).read_text(encoding="utf-8"),
                             (page.WEB_ASSET_DIR / name).read_text(encoding="utf-8"))

    def test_sections_await_shared_dependencies(self):
        script = (page.WEB_ASSET_DIR / "climate-risk-housing.js").read_text(encoding="utf-8")
        self.assertIn("await Promise.all([ensureGeography(), ensureEvents()])", script)
        self.assertIn("await Promise.all([ensureEvents(), ensureFeatures()])", script)
        self.assertIn("ensureGeography(), ensureEvents(), ensureFeatures(),", script)
        self.assertIn("deferredDataPromises.delete(globalName)", script)
        self.assertIn('retry.addEventListener("click", attempt)', script)


if __name__ == "__main__":
    unittest.main()
