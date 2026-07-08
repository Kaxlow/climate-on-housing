"""
CLI command to train climate risk prediction models.

Usage:
    train-climate-risk-model [--tune] [--model MODEL_NAME] [--hazard HAZARD_TYPE]
    train-climate-risk-model --all-hazards [--tune]
"""

import argparse
import sys
from pathlib import Path

from housing_climate_risk.modeling.climate_risk_prediction import (
    ClimateRiskPredictor,
    HAZARD_TYPES,
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
        default=2021,
        help='Minimum year for data (default: 2021)'
    )

    parser.add_argument(
        '--max-year',
        type=int,
        default=2023,
        help='Maximum year for data (default: 2023)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set fraction (default: 0.2)'
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
            print(f"  Data years: {args.min_year} - {args.max_year}")
            print(f"  Test size: {args.test_size}")
            print(f"  Hyperparameter tuning: {args.tune}")

            all_results = train_all_hazards(
                tune_hyperparams=args.tune,
                min_year=args.min_year,
                max_year=args.max_year
            )

            # Summary
            print("\n" + "=" * 70)
            print("Training Complete - Summary")
            print("=" * 70)

            for hazard_key, (predictor, results) in all_results.items():
                hazard_name = HAZARD_TYPES.get(hazard_key, 'Overall')
                best_name, _, best_f1 = predictor.get_best_model()
                print(f"{hazard_name:20} -> Best: {best_name:20} (F1: {best_f1:.4f})")

            return 0

        # Single hazard training
        hazard_type = None if args.hazard == 'overall' else args.hazard
        hazard_display = HAZARD_TYPES.get(hazard_type, 'Overall') if hazard_type else 'Overall'

        print(f"\nConfiguration:")
        print(f"  Hazard type: {hazard_display}")
        print(f"  Model(s): {args.model}")
        print(f"  Data years: {args.min_year} - {args.max_year}")
        print(f"  Test size: {args.test_size}")
        print(f"  Hyperparameter tuning: {args.tune}")

        predictor = ClimateRiskPredictor(hazard_type=hazard_type)

        # Load data
        print("\nLoading data...")
        df = predictor.load_data(min_year=args.min_year, max_year=args.max_year)

        if args.model == 'all':
            # Train all models
            print("\nTraining all models...")
            results = predictor.train_all_models(
                df,
                tune_hyperparams=args.tune,
                test_size=args.test_size
            )
        else:
            # Train single model
            print(f"\nTraining {args.model}...")
            df_engineered = predictor.engineer_features(df)
            X, y, feature_names = predictor.prepare_features(df_engineered)

            from sklearn.model_selection import train_test_split
            y_encoded = predictor.label_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=args.test_size, random_state=42, stratify=y_encoded
            )

            X_train_scaled = predictor.scaler.fit_transform(X_train)
            X_test_scaled = predictor.scaler.transform(X_test)

            model, train_metrics = predictor.train_model(
                args.model,
                X_train_scaled,
                y_train,
                tune_hyperparams=args.tune
            )

            eval_metrics = predictor.evaluate_model(
                model, X_test_scaled, y_test, args.model
            )

            predictor.models[args.model] = model
            predictor.results = {
                'feature_names': feature_names,
                'label_classes': predictor.label_encoder.classes_.tolist(),
                'models': {args.model: {**train_metrics, **eval_metrics}}
            }

            print(f"\n{args.model} Results:")
            print(f"  CV F1 Score: {train_metrics['cv_mean']:.4f} (+/- {train_metrics['cv_std']:.4f})")
            print(f"  Test Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Test F1 Score: {eval_metrics['f1_weighted']:.4f}")

        # Save models
        print("\nSaving models and results...")
        predictor.save_models()

        # Report best model
        best_name, best_model, best_f1 = predictor.get_best_model()
        print(f"\n{'='*70}")
        print(f"Best {hazard_display} Model: {best_name}")
        print(f"F1 Score: {best_f1:.4f}")
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
