"""
Test script for climate risk prediction pipeline.

This script tests the data loading, feature engineering, and model training
to ensure the pipeline works end-to-end.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from housing_climate_risk.modeling.climate_risk_prediction import ClimateRiskPredictor
from housing_climate_risk.modeling.preprocessing import FeatureEngineer


def test_data_loading():
    """Test data loading from database."""
    print("\n" + "="*60)
    print("TEST: Data Loading")
    print("="*60)

    predictor = ClimateRiskPredictor()
    df = predictor.load_data(min_year=2021, max_year=2023)

    assert len(df) > 0, "No data loaded"
    assert 'risk_rating' in df.columns, "Missing target column"
    assert 'median_household_income' in df.columns, "Missing income column"

    print(f"[OK] Loaded {len(df)} counties")
    print(f"[OK] Features: {df.columns.tolist()}")
    print(f"[OK] Risk rating distribution:\n{df['risk_rating'].value_counts()}")


def test_feature_engineering():
    """Test feature engineering."""
    print("\n" + "="*60)
    print("TEST: Feature Engineering")
    print("="*60)

    predictor = ClimateRiskPredictor()
    df = predictor.load_data(min_year=2021, max_year=2023)

    engineer = FeatureEngineer()
    df_engineered = engineer.fit_transform(df)

    expected_features = [
        'insurance_burden',
        'housing_cost_burden',
        'market_cooling'
    ]

    for feat in expected_features:
        assert feat in df_engineered.columns, f"Missing feature: {feat}"

    print(f"[OK] Created {len(df_engineered.columns)} features")
    print(f"[OK] Sample engineered features:")
    print(df_engineered[expected_features].head())


def test_model_training():
    """Test model training (single model, no tuning)."""
    print("\n" + "="*60)
    print("TEST: Model Training")
    print("="*60)

    predictor = ClimateRiskPredictor()
    df = predictor.load_data(min_year=2021, max_year=2023)

    # Use more data to ensure enough samples per class
    df_sample = df.sample(n=min(1000, len(df)), random_state=42)

    df_engineered = predictor.engineer_features(df_sample)
    X, y, feature_names = predictor.prepare_features(df_engineered)

    print(f"[OK] Prepared features: {X.shape}")
    print(f"[OK] Feature names: {feature_names[:5]}...")

    from sklearn.model_selection import train_test_split
    y_encoded = predictor.label_encoder.fit_transform(y)

    # Check class distribution
    from collections import Counter
    class_counts = Counter(y_encoded)
    print(f"[OK] Class distribution: {sorted(class_counts.items())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)

    # Train just logistic regression for speed
    model, train_metrics = predictor.train_model(
        'logistic_regression',
        X_train_scaled,
        y_train,
        tune_hyperparams=False
    )

    print(f"[OK] Trained model: {train_metrics['model_name']}")
    print(f"[OK] CV F1 Score: {train_metrics['cv_mean']:.4f} (+/- {train_metrics['cv_std']:.4f})")

    eval_metrics = predictor.evaluate_model(
        model, X_test_scaled, y_test, 'logistic_regression'
    )

    print(f"[OK] Test Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"[OK] Test F1 Score: {eval_metrics['f1_weighted']:.4f}")


def test_model_persistence():
    """Test saving and loading models."""
    print("\n" + "="*60)
    print("TEST: Model Persistence")
    print("="*60)

    predictor = ClimateRiskPredictor()
    df = predictor.load_data(min_year=2021, max_year=2023)

    # Subsample with enough for each class
    df_sample = df.sample(n=min(800, len(df)), random_state=42)

    df_engineered = predictor.engineer_features(df_sample)
    X, y, feature_names = predictor.prepare_features(df_engineered)

    from sklearn.model_selection import train_test_split
    y_encoded = predictor.label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    X_train_scaled = predictor.scaler.fit_transform(X_train)

    model, train_metrics = predictor.train_model(
        'logistic_regression',
        X_train_scaled,
        y_train
    )

    predictor.models['logistic_regression'] = model
    predictor.results = {
        'feature_names': feature_names,
        'label_classes': predictor.label_encoder.classes_.tolist(),
        'models': {'logistic_regression': train_metrics}
    }

    predictor.save_models()

    print(f"[OK] Models saved to: {predictor.output_dir}")
    print(f"[OK] Files created:")
    for file in predictor.output_dir.glob("*"):
        print(f"  - {file.name}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CLIMATE RISK PREDICTION PIPELINE TESTS")
    print("="*60)

    try:
        test_data_loading()
        test_feature_engineering()
        test_model_training()
        test_model_persistence()

        print("\n" + "="*60)
        print("[OK] ALL TESTS PASSED")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
