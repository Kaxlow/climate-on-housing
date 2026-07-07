"""
Example: Using Trained Climate Risk Models for Prediction

This script demonstrates how to load a trained climate risk prediction model
and use it to make predictions on new county data.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from housing_climate_risk.modeling.climate_risk_prediction import ClimateRiskPredictor
from housing_climate_risk.modeling.preprocessing import FeatureEngineer
from housing_climate_risk.paths import OUTPUT_DIR


def load_latest_model(model_dir: Path, model_name: str = 'random_forest'):
    """
    Load the most recently trained model and its artifacts.

    Args:
        model_dir: Directory containing saved models
        model_name: Name of the model to load

    Returns:
        Tuple of (model, scaler, label_encoder, feature_names)
    """
    # Find latest model files
    model_files = sorted(model_dir.glob(f'{model_name}_*.joblib'), reverse=True)
    if not model_files:
        raise FileNotFoundError(f"No {model_name} model found in {model_dir}")

    model_file = model_files[0]
    timestamp = model_file.stem.split('_', 2)[-1]

    # Load all artifacts
    model = joblib.load(model_file)
    scaler = joblib.load(model_dir / f'scaler_{timestamp}.joblib')
    label_encoder = joblib.load(model_dir / f'label_encoder_{timestamp}.joblib')

    # Load feature names
    import json
    with open(model_dir / f'feature_names_{timestamp}.json') as f:
        feature_info = json.load(f)
    feature_names = feature_info['feature_names']

    print(f"Loaded model: {model_file.name}")
    print(f"Features: {len(feature_names)}")
    print(f"Risk classes: {label_encoder.classes_}")

    return model, scaler, label_encoder, feature_names


def predict_climate_risk(
    df: pd.DataFrame,
    model,
    scaler,
    label_encoder,
    feature_names: list
) -> pd.DataFrame:
    """
    Predict climate risk ratings for counties.

    Args:
        df: DataFrame with raw county features
        model: Trained model
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        feature_names: List of feature names

    Returns:
        DataFrame with predictions and probabilities
    """
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.fit_transform(df)

    # Extract feature matrix
    X = df_features[feature_names].values

    # Scale features
    X_scaled = scaler.transform(X)

    # Make predictions
    y_pred_encoded = model.predict(X_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_scaled)
        proba_df = pd.DataFrame(
            y_proba,
            columns=[f'prob_{cls}' for cls in label_encoder.classes_]
        )
    else:
        proba_df = pd.DataFrame()

    # Combine results
    result_df = df[['fips']].copy()
    result_df['predicted_risk'] = y_pred
    result_df['predicted_risk_encoded'] = y_pred_encoded

    if not proba_df.empty:
        result_df = pd.concat([result_df, proba_df], axis=1)

    return result_df


def example_batch_prediction():
    """Example: Predict risk for all counties in database."""
    print("="*70)
    print("EXAMPLE: Batch Prediction on All Counties")
    print("="*70)

    # Load data
    predictor = ClimateRiskPredictor()
    df = predictor.load_data(min_year=2021, max_year=2023)

    print(f"\nLoaded {len(df)} counties")

    # Load latest random forest model
    model_dir = OUTPUT_DIR / 'models' / 'climate_risk_prediction'
    model, scaler, label_encoder, feature_names = load_latest_model(
        model_dir, model_name='random_forest'
    )

    # Make predictions
    print("\nMaking predictions...")
    predictions = predict_climate_risk(
        df, model, scaler, label_encoder, feature_names
    )

    # Show results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)

    print("\nPredicted risk distribution:")
    print(predictions['predicted_risk'].value_counts().sort_index())

    print("\nSample predictions:")
    sample = predictions.sample(n=min(10, len(predictions)), random_state=42)
    print(sample.to_string(index=False))

    # Compare with actual if available
    if 'risk_rating' in df.columns:
        comparison = df[['fips', 'risk_rating']].merge(
            predictions[['fips', 'predicted_risk']],
            on='fips'
        )
        comparison['correct'] = (
            comparison['risk_rating'] == comparison['predicted_risk']
        )

        accuracy = comparison['correct'].mean()
        print(f"\nAccuracy on full dataset: {accuracy:.4f}")

    # Save predictions
    output_path = model_dir / 'predictions_all_counties.csv'
    predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")


def example_single_county_prediction():
    """Example: Predict risk for a single county with custom features."""
    print("\n" + "="*70)
    print("EXAMPLE: Single County Prediction")
    print("="*70)

    # Create example county data
    county_data = {
        'fips': ['06075'],  # San Francisco County
        'median_household_income': [119000.0],
        'housing_burden_30pct': [45.0],
        'insurance_premium': [2500.0],
        'property_taxes_utilities': [2800.0],
        'net_migration_rate': [2.5],
        'homes_sold_yoy': [-5.0],
        'median_dom_yoy': [15.0]
    }
    df = pd.DataFrame(county_data)

    print("\nCounty features:")
    for col, val in county_data.items():
        if col != 'fips':
            print(f"  {col}: {val[0]}")

    # Load model
    model_dir = OUTPUT_DIR / 'models' / 'climate_risk_prediction'
    model, scaler, label_encoder, feature_names = load_latest_model(
        model_dir, model_name='random_forest'
    )

    # Make prediction
    predictions = predict_climate_risk(
        df, model, scaler, label_encoder, feature_names
    )

    print("\n" + "="*70)
    print("PREDICTION")
    print("="*70)
    print(f"\nPredicted Risk: {predictions['predicted_risk'].iloc[0]}")

    if 'prob_Very Low' in predictions.columns:
        print("\nRisk probabilities:")
        for cls in label_encoder.classes_:
            prob = predictions[f'prob_{cls}'].iloc[0]
            print(f"  {cls}: {prob:.4f} ({prob*100:.1f}%)")


def example_feature_importance():
    """Example: Show feature importance from trained model."""
    print("\n" + "="*70)
    print("EXAMPLE: Feature Importance Analysis")
    print("="*70)

    # Load model
    model_dir = OUTPUT_DIR / 'models' / 'climate_risk_prediction'
    model, scaler, label_encoder, feature_names = load_latest_model(
        model_dir, model_name='random_forest'
    )

    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nTop 10 Most Important Features:")
        print("-" * 70)
        for i, idx in enumerate(indices[:10], 1):
            print(f"{i:2d}. {feature_names[idx]:30s} {importances[idx]:.6f}")
    else:
        print("\nModel does not provide feature importance.")


def main():
    """Run all examples."""
    try:
        # Check if models exist
        model_dir = OUTPUT_DIR / 'models' / 'climate_risk_prediction'
        if not model_dir.exists() or not list(model_dir.glob('*.joblib')):
            print("No trained models found!")
            print(f"Please run: train-climate-risk-model")
            print(f"This will train models and save them to: {model_dir}")
            return 1

        # Run examples
        example_batch_prediction()
        example_single_county_prediction()
        example_feature_importance()

        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
