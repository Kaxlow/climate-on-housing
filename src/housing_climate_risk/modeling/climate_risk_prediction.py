"""
Climate Risk Level Prediction Model

This module provides a machine learning pipeline to predict county-level climate risk
ratings based on economic, housing, and demographic features.

Target Variables:
- Overall FEMA NRI Risk Rating (Very Low, Relatively Low, Relatively Moderate,
  Relatively High, Very High)
- Hazard-specific risk ratings for 5 common hazards (aligned with stormhouse-2.html):
  * ERQK (Earthquake)
  * IFLD (Riverine Flooding)
  * WFIR (Wildfire)
  * TRND (Tornado)
  * HAIL (Hail)

Features:
1. Income - Median household income
2. Housing Burden - Share with housing costs >= 30% of income
3. Insurance - Mean homeowner insurance premium
4. Property Taxes & Utilities - Median owner costs with mortgage
5. Net Migration - County net migration rate
6. Homes Sold YOY - Year-over-year change in homes sold
7. Median Days on Market YOY - Year-over-year change in days on market
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score
)

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from housing_climate_risk.paths import DATA_DIR, OUTPUT_DIR


# Top 5 hazards used in stormhouse-2.html "Are climate risks priced into housing markets?" section
HAZARD_TYPES = {
    'ERQK': 'Earthquake',
    'IFLD': 'Riverine Flooding',
    'WFIR': 'Wildfire',
    'TRND': 'Tornado',
    'HAIL': 'Hail'
}


class ClimateRiskPredictor:
    """
    Machine learning pipeline for predicting climate risk levels.

    Supports multiple model types with configurable hyperparameters,
    standardized preprocessing, and model persistence.
    """

    MODEL_CONFIGS = {
        'logistic_regression': {
            'class': LogisticRegression,
            'default_params': {
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga']
            }
        },
        'random_forest': {
            'class': RandomForestClassifier,
            'default_params': {
                'n_estimators': 100,
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            },
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'class': GradientBoostingClassifier,
            'default_params': {
                'n_estimators': 100,
                'random_state': 42,
                'learning_rate': 0.1
            },
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'neural_network': {
            'class': MLPClassifier,
            'default_params': {
                'hidden_layer_sizes': (100, 50),
                'random_state': 42,
                'max_iter': 500,
                'early_stopping': True
            },
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
    }

    def __init__(self, db_path: Optional[Path] = None, output_dir: Optional[Path] = None,
                 hazard_type: Optional[str] = None):
        """
        Initialize the climate risk predictor.

        Args:
            db_path: Path to DuckDB database (defaults to data/quoll.duckdb)
            output_dir: Path to output directory for models and results
            hazard_type: Specific hazard type code (e.g., 'LNDS', 'LTNG') or None for overall risk
        """
        self.db_path = db_path or DATA_DIR / 'quoll.duckdb'
        self.hazard_type = hazard_type

        # Set output directory based on hazard type
        if hazard_type:
            if hazard_type not in HAZARD_TYPES:
                raise ValueError(f"Unknown hazard type: {hazard_type}. Must be one of {list(HAZARD_TYPES.keys())}")
            self.output_dir = output_dir or OUTPUT_DIR / 'models' / 'climate_risk_prediction' / hazard_type.lower()
        else:
            self.output_dir = output_dir or OUTPUT_DIR / 'models' / 'climate_risk_prediction' / 'overall'

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.models = {}
        self.results = {}

    def load_data(self, min_year: int = 2021, max_year: int = 2023) -> pd.DataFrame:
        """
        Load and join data from database tables.

        Args:
            min_year: Minimum year for ACS and insurance data
            max_year: Maximum year for ACS and insurance data

        Returns:
            DataFrame with features and target variable
        """
        conn = duckdb.connect(str(self.db_path), read_only=True)

        # Determine which risk column to use
        if self.hazard_type:
            risk_col = f"{self.hazard_type}_RISKR"
            risk_score_col = f"{self.hazard_type}_RISKS"
            target_name = f"{HAZARD_TYPES[self.hazard_type]} Risk"
        else:
            risk_col = "risk_rating"
            risk_score_col = "risk_score"
            target_name = "Overall Risk"

        query = f"""
        WITH recent_housing AS (
            SELECT
                fips,
                AVG(CAST(HOMES_SOLD_YOY AS DOUBLE)) as homes_sold_yoy,
                AVG(CAST(MEDIAN_DOM_YOY AS DOUBLE)) as median_dom_yoy
            FROM mart.redfin_county_monthly
            WHERE period_begin >= '{min_year}-01-01'
                AND period_begin <= '{max_year}-12-31'
                AND property_type = 'All Residential'
            GROUP BY fips
        ),
        recent_acs_affordability AS (
            SELECT
                fips,
                AVG(CAST(owner_mortgage_cost_burden_30pct_plus AS DOUBLE)) as housing_burden_30pct,
                AVG(CAST(median_owner_costs_mortgage AS DOUBLE)) as property_taxes_utilities
            FROM mart.acs_county_affordability_annual
            WHERE year BETWEEN {min_year} AND {max_year}
            GROUP BY fips
        ),
        recent_acs_economic AS (
            SELECT
                fips,
                AVG(CAST(dp03_income_and_benefits_total_households_median_household_income_est AS DOUBLE)) as median_household_income
            FROM mart.acs_county_economic_annual
            WHERE year BETWEEN {min_year} AND {max_year}
            GROUP BY fips
        ),
        recent_insurance AS (
            SELECT
                fips,
                AVG(mean_premium) as insurance_premium
            FROM mart.insurance_premiums_annual
            WHERE year BETWEEN {min_year} AND {max_year}
            GROUP BY fips
        ),
        net_migration AS (
            SELECT
                fips,
                AVG(CAST(domestic_in_migration_rate AS DOUBLE)) as net_migration_rate
            FROM mart.acs_county_demographic_annual
            WHERE year BETWEEN {min_year} AND {max_year}
            GROUP BY fips
        )
        SELECT
            n.fips,
            n.{risk_col} as risk_rating,
            CAST(n.{risk_score_col} AS DOUBLE) as risk_score,
            e.median_household_income,
            a.housing_burden_30pct,
            i.insurance_premium,
            a.property_taxes_utilities,
            m.net_migration_rate,
            h.homes_sold_yoy,
            h.median_dom_yoy
        FROM mart.nri_county_risk n
        LEFT JOIN recent_acs_affordability a ON n.fips = a.fips
        LEFT JOIN recent_acs_economic e ON n.fips = e.fips
        LEFT JOIN recent_insurance i ON n.fips = i.fips
        LEFT JOIN net_migration m ON n.fips = m.fips
        LEFT JOIN recent_housing h ON n.fips = h.fips
        WHERE n.{risk_col} IS NOT NULL
            AND n.{risk_col} != ''
            AND n.{risk_col} NOT IN ('Insufficient Data', 'Not Applicable', 'No Rating')
        """

        df = conn.execute(query).fetchdf()

        conn.close()

        print(f"Loaded {len(df)} counties with complete data")
        print(f"\n{target_name} rating distribution:\n{df['risk_rating'].value_counts()}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features and handle missing values.

        Args:
            df: Raw feature DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Create income brackets
        df['income_bracket'] = pd.cut(
            df['median_household_income'],
            bins=[0, 40000, 60000, 80000, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )

        # Create burden severity levels
        df['burden_severity'] = pd.cut(
            df['housing_burden_30pct'],
            bins=[0, 20, 35, 50, 100],
            labels=['minimal', 'moderate', 'high', 'severe']
        )

        # Insurance premium per $1000 income
        df['insurance_burden'] = (
            df['insurance_premium'] * 12 / df['median_household_income'] * 100
        )

        # Housing costs per $1000 income
        df['housing_cost_burden'] = (
            df['property_taxes_utilities'] * 12 / df['median_household_income'] * 100
        )

        # Market cooling indicator (high DOM YoY + low sales YoY)
        df['market_cooling'] = (
            df['median_dom_yoy'].fillna(0) - df['homes_sold_yoy'].fillna(0)
        )

        # Fill missing numeric values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled {col} missing values with median: {median_val:.2f}")

        # One-hot encode categorical features
        if 'income_bracket' in df.columns:
            df = pd.get_dummies(df, columns=['income_bracket'], prefix='income', drop_first=True)
        if 'burden_severity' in df.columns:
            df = pd.get_dummies(df, columns=['burden_severity'], prefix='burden', drop_first=True)

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Select and scale features for modeling.

        Args:
            df: DataFrame with engineered features

        Returns:
            Tuple of (X features, y target, feature_names)
        """
        # Define feature columns
        base_features = [
            'median_household_income',
            'housing_burden_30pct',
            'insurance_premium',
            'property_taxes_utilities',
            'net_migration_rate',
            'homes_sold_yoy',
            'median_dom_yoy',
            'insurance_burden',
            'housing_cost_burden',
            'market_cooling'
        ]

        # Add engineered categorical features
        engineered_features = [col for col in df.columns
                              if col.startswith(('income_', 'burden_'))]

        feature_cols = base_features + engineered_features
        feature_cols = [col for col in feature_cols if col in df.columns]

        X = df[feature_cols].values
        y = df['risk_rating'].values

        self.feature_names = feature_cols

        return X, y, feature_cols

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        custom_params: Optional[Dict] = None,
        tune_hyperparams: bool = False
    ) -> Tuple[object, Dict]:
        """
        Train a single model with optional hyperparameter tuning.

        Args:
            model_name: Name of model type from MODEL_CONFIGS
            X_train: Training features
            y_train: Training labels
            custom_params: Optional custom parameters (override defaults)
            tune_hyperparams: Whether to run grid search

        Returns:
            Tuple of (trained model, metrics dict)
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        config = self.MODEL_CONFIGS[model_name]
        params = config['default_params'].copy()
        if custom_params:
            params.update(custom_params)

        model = config['class'](**params)

        if tune_hyperparams:
            print(f"\nTuning hyperparameters for {model_name}...")
            grid_search = GridSearchCV(
                model,
                config['param_grid'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
        else:
            model.fit(X_train, y_train)
            best_params = params

        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='f1_weighted'
        )

        metrics = {
            'model_name': model_name,
            'params': best_params,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'trained_at': datetime.now().isoformat()
        }

        return model, metrics

    def evaluate_model(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Model identifier

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)

        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        # Classification report
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            metrics['top_features'] = [
                {
                    'feature': self.feature_names[i],
                    'importance': float(importances[i])
                }
                for i in indices
            ]

        return metrics

    def train_all_models(
        self,
        df: pd.DataFrame,
        tune_hyperparams: bool = False,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train and evaluate all configured models.

        Args:
            df: Input DataFrame with features
            tune_hyperparams: Whether to tune hyperparameters
            test_size: Fraction of data for testing

        Returns:
            Dictionary of results for all models
        """
        print("Engineering features...")
        df_engineered = self.engineer_features(df)

        print("\nPreparing features...")
        X, y, feature_names = self.prepare_features(df_engineered)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        results = {
            'feature_names': feature_names,
            'label_classes': self.label_encoder.classes_.tolist(),
            'models': {}
        }

        # Train each model
        for model_name in self.MODEL_CONFIGS.keys():
            print(f"\n{'='*60}")
            print(f"Training {model_name}...")
            print(f"{'='*60}")

            model, train_metrics = self.train_model(
                model_name,
                X_train_scaled,
                y_train,
                tune_hyperparams=tune_hyperparams
            )

            eval_metrics = self.evaluate_model(
                model, X_test_scaled, y_test, model_name
            )

            # Combine metrics
            combined_metrics = {**train_metrics, **eval_metrics}

            # Store model and results
            self.models[model_name] = model
            results['models'][model_name] = combined_metrics

            print(f"\n{model_name} Results:")
            print(f"  CV F1 Score: {train_metrics['cv_mean']:.4f} (+/- {train_metrics['cv_std']:.4f})")
            print(f"  Test Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"  Test F1 Score: {eval_metrics['f1_weighted']:.4f}")

        self.results = results
        return results

    def save_models(self):
        """Save all trained models and artifacts."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Determine prefix based on hazard type
        prefix = f"{self.hazard_type.lower()}_" if self.hazard_type else "overall_"

        # Save models
        for model_name, model in self.models.items():
            model_path = self.output_dir / f'{prefix}{model_name}_{timestamp}.joblib'
            joblib.dump(model, model_path)
            print(f"Saved model: {model_path}")

        # Save scaler and label encoder
        scaler_path = self.output_dir / f'{prefix}scaler_{timestamp}.joblib'
        encoder_path = self.output_dir / f'{prefix}label_encoder_{timestamp}.joblib'
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)

        # Save results with metadata
        results_with_meta = {
            'hazard_type': self.hazard_type,
            'hazard_name': HAZARD_TYPES.get(self.hazard_type) if self.hazard_type else 'Overall',
            'timestamp': timestamp,
            **self.results
        }

        results_path = self.output_dir / f'{prefix}results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results_with_meta, f, indent=2, default=str)
        print(f"Saved results: {results_path}")

        # Save feature names
        features_path = self.output_dir / f'{prefix}feature_names_{timestamp}.json'
        with open(features_path, 'w') as f:
            json.dump({
                'hazard_type': self.hazard_type,
                'hazard_name': HAZARD_TYPES.get(self.hazard_type) if self.hazard_type else 'Overall',
                'feature_names': self.feature_names,
                'label_classes': self.label_encoder.classes_.tolist()
            }, f, indent=2)
        print(f"Saved feature info: {features_path}")

    def get_best_model(self) -> Tuple[str, object, float]:
        """
        Identify the best performing model based on test F1 score.

        Returns:
            Tuple of (model_name, model_object, f1_score)
        """
        best_model_name = None
        best_f1 = -1

        for model_name, metrics in self.results['models'].items():
            f1 = metrics['f1_weighted']
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name

        return best_model_name, self.models[best_model_name], best_f1


def train_all_hazards(tune_hyperparams: bool = False, min_year: int = 2021, max_year: int = 2023):
    """
    Train models for overall risk and all hazard types.

    Args:
        tune_hyperparams: Whether to tune hyperparameters
        min_year: Minimum year for data
        max_year: Maximum year for data

    Returns:
        Dictionary of {hazard_type: (predictor, results)} for all hazards plus overall
    """
    all_results = {}

    # Train overall risk model
    print("\n" + "=" * 70)
    print("Training Overall Climate Risk Model")
    print("=" * 70)

    overall_predictor = ClimateRiskPredictor(hazard_type=None)
    df_overall = overall_predictor.load_data(min_year=min_year, max_year=max_year)
    results_overall = overall_predictor.train_all_models(df_overall, tune_hyperparams=tune_hyperparams)
    overall_predictor.save_models()

    best_name, _, best_f1 = overall_predictor.get_best_model()
    print(f"\nBest Overall Model: {best_name} (F1: {best_f1:.4f})")

    all_results['overall'] = (overall_predictor, results_overall)

    # Train hazard-specific models
    for hazard_code, hazard_name in HAZARD_TYPES.items():
        print("\n" + "=" * 70)
        print(f"Training {hazard_name} ({hazard_code}) Risk Model")
        print("=" * 70)

        try:
            predictor = ClimateRiskPredictor(hazard_type=hazard_code)
            df = predictor.load_data(min_year=min_year, max_year=max_year)

            if len(df) < 100:
                print(f"Warning: Only {len(df)} counties with data. Skipping...")
                continue

            results = predictor.train_all_models(df, tune_hyperparams=tune_hyperparams)
            predictor.save_models()

            best_name, _, best_f1 = predictor.get_best_model()
            print(f"\nBest {hazard_name} Model: {best_name} (F1: {best_f1:.4f})")

            all_results[hazard_code] = (predictor, results)

        except Exception as e:
            print(f"Error training {hazard_name} model: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_results


def main():
    """Run the full modeling pipeline."""
    print("Climate Risk Prediction Model Training")
    print("=" * 70)

    predictor = ClimateRiskPredictor()

    # Load data
    print("\nLoading data...")
    df = predictor.load_data(min_year=2021, max_year=2023)

    # Train all models
    print("\nTraining models...")
    results = predictor.train_all_models(df, tune_hyperparams=False)

    # Save everything
    print("\nSaving models and results...")
    predictor.save_models()

    # Report best model
    best_name, best_model, best_f1 = predictor.get_best_model()
    print(f"\n{'='*60}")
    print(f"Best Model: {best_name}")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"{'='*60}")

    return predictor, results


if __name__ == '__main__':
    predictor, results = main()
