from __future__ import annotations

from collections.abc import Callable
from typing import Final

from housing_climate_risk.page_data.climate_housing import PAGE_ALIASES as CLIMATE_PAGE_ALIASES
from housing_climate_risk.page_data.climate_housing import PAGE_HTML_FILES as CLIMATE_PAGE_HTML_FILES
from housing_climate_risk.page_data.climate_housing import build_page as build_climate_housing_page
from housing_climate_risk.page_data.server import serve_visualization
from housing_climate_risk.page_data.stormhouse import build_stormhouse_page


PAGE_HTML_FILES: Final[dict[str, str]] = {
    **CLIMATE_PAGE_HTML_FILES,
    "stormhouse": "stormhouse.html",
}
PAGE_ALIASES: Final[dict[str, str]] = {
    **CLIMATE_PAGE_ALIASES,
    "stormhouse": "stormhouse",
}


def normalize_page(page: str) -> str:
    normalized = PAGE_ALIASES.get(page.lower().strip(), page.lower().strip())
    if normalized not in PAGE_HTML_FILES:
        valid = ", ".join(PAGE_HTML_FILES)
        raise ValueError(f"Unknown page: {page}. Expected one of: {valid}")
    return normalized


def _build_climate_page(page: str) -> dict[str, object]:
    return build_climate_housing_page(page)


PAGE_BUILDERS: Final[dict[str, Callable[[str], dict[str, object]]]] = {
    **{page: _build_climate_page for page in CLIMATE_PAGE_HTML_FILES},
    "stormhouse": lambda page: build_stormhouse_page(),
}


def build_page(page: str, *, serve: bool = False, host: str = "127.0.0.1", port: int | None = None) -> dict[str, object]:
    normalized = normalize_page(page)
    result = PAGE_BUILDERS[normalized](normalized)
    result["page"] = normalized
    result["html_file"] = PAGE_HTML_FILES[normalized]
    if serve:
        result["url"] = serve_visualization(result["html_file"], host=host, port=port)
    return result


def build_all_pages(*, progress: bool = False) -> list[dict[str, object]]:
    results = []
    for page in PAGE_HTML_FILES:
        if progress:
            print(f"Building {page}...")
        results.append(build_page(page))
    return results
