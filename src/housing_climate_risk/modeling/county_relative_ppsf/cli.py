from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from housing_climate_risk.paths import ROOT

from .data import build_county_modeling_dataset
from .train import TrainingConfig, train_all_risk_groups


DEFAULT_DB_PATH = ROOT / "data" / "quoll.duckdb"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "models" / "county_relative_ppsf"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train county-level relative Median PPSF YoY models, pooling "
            "High and Very High NRI counties."
        )
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-housing-months", type=int, default=60)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--max-outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--gradient-search-iterations", type=int, default=12)
    parser.add_argument("--maximum-absolute-correlation", type=float, default=0.85)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with duckdb.connect(str(args.db_path), read_only=True) as con:
        dataset = build_county_modeling_dataset(
            con,
            minimum_housing_months=args.minimum_housing_months,
        )
    config = TrainingConfig(
        output_dir=args.output_dir,
        outer_repeats=args.outer_repeats,
        max_outer_splits=args.max_outer_splits,
        inner_splits=args.inner_splits,
        gradient_search_iterations=args.gradient_search_iterations,
        maximum_absolute_correlation=args.maximum_absolute_correlation,
        n_jobs=args.n_jobs,
    )
    manifest = train_all_risk_groups(dataset, config)
    print(f"Wrote county model artifacts to {args.output_dir}")
    for risk_group, model in manifest["models"].items():
        print(f"{risk_group}: {model['model_name']} ({model['county_count']} counties)")


if __name__ == "__main__":
    main()
