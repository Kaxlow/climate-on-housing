from __future__ import annotations

from typing import Final


PAGE_HTML_FILES: Final[dict[str, str]] = {
    "index": "index.html",
    "climate-on-housing": "climate-on-housing.html",
    "story-1": "climate-housing-story-1.html",
    "story-2": "climate-housing-story-2.html",
    "story-3": "climate-housing-story-3.html",
    "story-4": "climate-housing-story-4.html",
    "story-5": "climate-housing-story-5.html",
}

PAGE_ALIASES: Final[dict[str, str]] = {
    "home": "index",
    "index": "index",
    "climate-on-housing": "climate-on-housing",
    "story-1": "story-1",
    "story-2": "story-2",
    "story-3": "story-3",
    "story-4": "story-4",
    "story-5": "story-5",
}


def normalize_page(page: str) -> str:
    normalized = PAGE_ALIASES.get(page.lower().strip(), page.lower().strip())
    if normalized not in PAGE_HTML_FILES:
        valid = ", ".join(PAGE_HTML_FILES)
        raise ValueError(f"Unknown page: {page}. Expected one of: {valid}")
    return normalized


def export_page_data(page: str) -> dict[str, object]:
    from housing_climate_risk.page_data.climate_housing_utils import export_page_data as _export_page_data

    return _export_page_data(normalize_page(page))


def build_page(page: str, *, serve: bool = False, host: str = "127.0.0.1", port: int | None = None) -> dict[str, object]:
    result = export_page_data(page)
    normalized = result["page"]
    result["html_file"] = PAGE_HTML_FILES[normalized]
    if serve:
        from housing_climate_risk.page_data.server import serve_visualization

        result["url"] = serve_visualization(result["html_file"], host=host, port=port)
    return result


def build_all_pages() -> list[dict[str, object]]:
    return [build_page(page) for page in PAGE_HTML_FILES]


def export_index_data() -> dict[str, object]:
    return export_page_data("index")


def export_story_1_data() -> dict[str, object]:
    return export_page_data("story-1")


def export_story_2_data() -> dict[str, object]:
    return export_page_data("story-2")


def export_story_3_data() -> dict[str, object]:
    return export_page_data("story-3")


def export_story_4_data() -> dict[str, object]:
    return export_page_data("story-4")


def export_story_5_data() -> dict[str, object]:
    return export_page_data("story-5")
