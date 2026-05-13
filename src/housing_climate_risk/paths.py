from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    return current


ROOT = find_project_root()
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "visualizations"
CACHE_DIR = DATA_DIR / "cache"
CLIMATE_DIR = DATA_DIR / "climate"
ECONOMIC_DIR = DATA_DIR / "economic"
GEOGRAPHIC_DIR = DATA_DIR / "geographic"
HOUSING_DIR = DATA_DIR / "housing"
POPULATION_DIR = DATA_DIR / "population"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

