"""
CLI command to train climate risk prediction models.

Usage:
    train-climate-risk-model [--tune] [--model MODEL_NAME] [--hazard HAZARD_TYPE]
    train-climate-risk-model --all-hazards [--tune]
"""

import argparse
import sys

import numpy as np

from housing_climate_risk.modeling.climate_risk_prediction import (
    ClimateRiskPredictor,
    HAZARD_TYPES,
    RISK_ORDER,
    RISK_ORDER_INVERSE,
    train_all_hazards
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train climate risk prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Hazard Types:
  overall     Overall climate risk (default)
{chr(10).join(f'  {code:<12}{name}' for code, name in HAZARD_TYPES.items())}

Examples:
  # Train overall risk model
  train-climate-risk-model

  # Train specific hazard model
  train-climate-risk-model --hazard TRND

  # Train all hazards (overall + 5 hazard-specific models)
  train-climate-risk-model --all-hazards

  # Train with hyperparameter tuning
  train-climate-risk-model --all-hazards --tune
        """
    )

    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning (slower but better results)'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic_regression', 'random_forest', 'gradient_boosting', 'neural_network', 'all'],
        default='all',
        help='Specific model to train (default: all)'
    )

    parser.add_argument(
        '--hazard',
        type=str,
        choices=['overall'] + list(HAZARD_TYPES.keys()),
        default='overall',
        help='Hazard type to train (default: overall)'
    )

    parser.add_argument(
        '--all-hazards',
        action='store_true',
        help='Train models for overall risk and all 5 hazard types'
    )

    parser.add_argument(
        '--min-year',
        type=int,
        default=None,
        help='Minimum year for data (default: max-year minus 9)'
    )

    parser.add_argument(
        '--max-year',
        type=int,
        default=None,
        help='Maximum year for data (default: last complete calendar year)'
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()

    print("=" * 70)
    print("Climate Risk Prediction Model Training")
    print("=" * 70)

    try:
        # Train all hazards if requested
        if args.all_hazards:
            print(f"\nConfiguration:")
            print(f"  Training: Overall + All 5 hazard types")
            print(f"  Model(s): {args.model}")
            min_yr = args.min_year or "auto (max-9)"
            max_yr = args.max_year or "auto (last complete year)"
            print(f"  Data years: {min_yr} - {max_yr}")
            print(f"  Hyperparameter tuning: {args.tune}")

            all_results = train_all_hazards(
                tune_hyperparams=args.tune,
                min_year=args.min_year,
                max_year=args.max_year
            )

            print("\n" + "=" * 70)
            print("Training Complete - Summary")
            print("=" * 70)

            for hazard_key, (predictor, results) in all_results.items():
                hazard_name = HAZARD_TYPES.get(hazard_key, 'Overall')
                best_name, _, best_mae = predictor.get_best_model()
                print(f"{hazard_name:20} -> Best: {best_name:20} (Ordinal MAE: {best_mae:.4f})")

            return 0

        # Single hazard training
        hazard_type = None if args.hazard == 'overall' else args.hazard
        hazard_display = HAZARD_TYPES.get(hazard_type, 'Overall') if hazard_type else 'Overall'

        print(f"\nConfiguration:")
        print(f"  Hazard type: {hazard_display}")
        print(f"  Model(s): {args.model}")
        print(f"  Data years: {args.min_year} - {args.max_year}")
        print(f"  Hyperparameter tuning: {args.tune}")

        predictor = ClimateRiskPredictor(hazard_type=hazard_type)

        print("\nLoading data...")
        df = predictor.load_data(min_year=args.min_year, max_year=args.max_year)

        if args.model == 'all':
            print("\nTraining all models...")
            results = predictor.train_all_models(df, tune_hyperparams=args.tune)
        else:
            # Single-model path: delegate to train_all_models then discard other models
            # so that ordinal encoding and spatial CV are applied consistently.
            print(f"\nTraining {args.model}...")
            all_results = predictor.train_all_models(df, tune_hyperparams=args.tune)

            # Keep only the requested model
            predictor.models = {args.model: predictor.models[args.model]}
            predictor.results['models'] = {args.model: predictor.results['models'][args.model]}
            results = predictor.results

            m = results['models'][args.model]
            print(f"\n{args.model} Results:")
            print(f"  Spatial CV F1:     {m['spatial_cv_f1_mean']:.4f} (+/- {m['spatial_cv_f1_std']:.4f})")
            print(f"  Holdout Accuracy:  {m['accuracy']:.4f}")
            print(f"  Holdout F1:        {m['f1_weighted']:.4f}")
            print(f"  Ordinal MAE:       {m['ordinal_mae']:.4f}")
            print(f"  Adjacent Accuracy: {m['adjacent_accuracy']:.4f}")

        print("\nSaving models and results...")
        predictor.save_models()

        best_name, best_model, best_mae = predictor.get_best_model()
        print(f"\n{'='*70}")
        print(f"Best {hazard_display} Model: {best_name}")
        print(f"Ordinal MAE: {best_mae:.4f}")
        print(f"{'='*70}")

        print(f"\nModels saved to: {predictor.output_dir}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
