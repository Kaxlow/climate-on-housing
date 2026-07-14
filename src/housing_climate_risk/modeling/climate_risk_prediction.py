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

Features (12):
1.  Median Household Income of Homeowners (ACS S2503, owner-occupied units)
2.  Net Resident Earnings Per Capita (BEA net earnings by place of residence)
3.  Dividends, Interest, and Rent Per Capita (BEA)
4.  Transfer Receipts Per Capita (BEA)
5.  Utilities as % of Income (derived: no-mortgage monthly costs minus taxes and insurance)
6.  Insurance as % of Income (mean annual premium / homeowner income)
7.  Property Taxes as % of Income (ACS S2507 median annual taxes / homeowner income)
8.  Net Migration Rate (total net migration per 1,000 residents, StatsAmerica)
9.  Unemployment Rate (ACS DP03 civilian labor force unemployment rate)
10. New Listings YOY (Redfin ratio)
11. Homes Sold YOY (Redfin ratio)
12. Median Days on Market YOY (Redfin absolute day delta)
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib
import json
from datetime import datetime

from sklearn.model_selection import GroupKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, mean_absolute_error
)
from sklearn.inspection import permutation_importance

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from housing_climate_risk.paths import DATA_DIR, OUTPUT_DIR


# Ordinal encoding of risk classes — order must be preserved for MAE to be meaningful
RISK_ORDER = {
    'Very Low': 0,
    'Relatively Low': 1,
    'Relatively Moderate': 2,
    'Relatively High': 3,
    'Very High': 4,
}
RISK_ORDER_INVERSE = {v: k for k, v in RISK_ORDER.items()}

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
        self.feature_names = []
        self.models = {}
        self.results = {}
        self.n_spatial_splits = 5

    def load_data(self, min_year: Optional[int] = None, max_year: Optional[int] = None) -> pd.DataFrame:
        """
        Load and join data from database tables.

        Args:
            min_year: Minimum year for time-varying data. Defaults to max_year - 9 (last 10 years).
            max_year: Maximum year for time-varying data. Defaults to the most recent year
                      available in the BEA personal income table, which is typically the most
                      restrictive recent boundary.

        Returns:
            DataFrame with features and target variable
        """
        conn = duckdb.connect(str(self.db_path), read_only=True)

        if max_year is None:
            max_year = datetime.now().year - 1
        if min_year is None:
            min_year = max_year - 9

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
                AVG(CAST(NEW_LISTINGS_YOY AS DOUBLE))  AS new_listings_yoy,
                AVG(CAST(HOMES_SOLD_YOY AS DOUBLE))    AS homes_sold_yoy,
                AVG(CAST(MEDIAN_DOM_YOY AS DOUBLE))    AS median_dom_yoy
            FROM mart.redfin_county_monthly
            WHERE period_begin >= '{min_year}-01-01'
                AND period_begin <= '{max_year}-12-31'
                AND property_type = 'All Residential'
            GROUP BY fips
        ),
        recent_acs_affordability AS (
            SELECT
                fips,
                -- Median household income for owner-occupied units (S2503)
                AVG(TRY_CAST(s2503_owner_occupied_units_occupied_housing_units_household_income_past_12_months_median_household_income_est AS DOUBLE)) AS median_homeowner_income,
                -- Median annual real estate taxes, non-mortgaged owners (S2507) for property tax % of income
                AVG(TRY_CAST(s2507_owner_occupied_units_no_mortgage_real_estate_taxes_median_est AS DOUBLE)) AS median_property_taxes_annual,
                -- Median monthly owner costs (no mortgage) — includes taxes, insurance, utilities, etc.
                AVG(TRY_CAST(dp04_selected_monthly_owner_costs_housing_units_no_mortgage_median_est AS DOUBLE)) AS median_monthly_owner_costs_no_mortgage
            FROM mart.acs_county_affordability_annual
            WHERE year BETWEEN {min_year} AND {max_year}
            GROUP BY fips
        ),
        recent_insurance AS (
            SELECT
                fips,
                AVG(mean_premium) AS insurance_premium_annual
            FROM mart.insurance_premiums_annual
            WHERE year BETWEEN {min_year} AND {max_year}
            GROUP BY fips
        ),
        recent_bea AS (
            SELECT
                fips,
                AVG(NULLIF(population, 0)) AS population,
                AVG(net_earnings_by_place_of_residence_thousands) AS net_earnings_thousands,
                AVG(dividends_interest_rent_thousands)            AS dividends_interest_rent_thousands,
                AVG(transfer_receipts_thousands)                  AS transfer_receipts_thousands
            FROM mart.statsamerica_bea_personal_income_annual
            WHERE year BETWEEN {min_year} AND {max_year}
            GROUP BY fips
        ),
        recent_migration AS (
            SELECT
                fips,
                AVG(CAST(total_net_migration AS DOUBLE)) AS total_net_migration
            FROM mart.statsamerica_population_components_annual
            WHERE year BETWEEN {min_year} AND {max_year}
            GROUP BY fips
        ),
        recent_unemployment AS (
            SELECT
                fips,
                AVG(TRY_CAST(dp03_civilian_labor_force_unemployment_rate_pct AS DOUBLE)) AS unemployment_rate
            FROM mart.acs_county_economic_annual
            WHERE year BETWEEN {min_year} AND {max_year}
            GROUP BY fips
        )
        SELECT
            n.fips,
            LEFT(n.fips, 2)                     AS state_fips,
            n.{risk_col}                        AS risk_rating,
            CAST(n.{risk_score_col} AS DOUBLE)  AS risk_score,
            -- Income (homeowners)
            a.median_homeowner_income,
            -- BEA income components per capita
            b.net_earnings_thousands * 1000.0 / b.population           AS net_earnings_per_capita,
            b.dividends_interest_rent_thousands * 1000.0 / b.population AS dividends_interest_rent_per_capita,
            b.transfer_receipts_thousands * 1000.0 / b.population       AS transfer_receipts_per_capita,
            -- Raw inputs for % of income features (computed in engineer_features)
            a.median_property_taxes_annual,
            a.median_monthly_owner_costs_no_mortgage,
            i.insurance_premium_annual,
            -- Migration (rate computed in engineer_features using BEA population)
            m.total_net_migration,
            b.population                                                AS bea_population,
            -- Labor market
            u.unemployment_rate,
            -- Housing market dynamics
            h.new_listings_yoy,
            h.homes_sold_yoy,
            h.median_dom_yoy
        FROM mart.nri_county_risk n
        LEFT JOIN recent_acs_affordability a  ON n.fips = a.fips
        LEFT JOIN recent_insurance i          ON n.fips = i.fips
        LEFT JOIN recent_bea b                ON n.fips = b.fips
        LEFT JOIN recent_migration m          ON n.fips = m.fips
        LEFT JOIN recent_unemployment u       ON n.fips = u.fips
        LEFT JOIN recent_housing h            ON n.fips = h.fips
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
        Derive % of income features and net migration rate, then impute missing values.

        Args:
            df: Raw feature DataFrame

        Returns:
            DataFrame with all model features ready for scaling
        """
        df = df.copy()

        # Net migration rate per 1,000 residents
        df['net_migration_rate'] = np.where(
            df['bea_population'] > 0,
            df['total_net_migration'] / df['bea_population'] * 1000,
            np.nan
        )

        # % of income features — use median_homeowner_income as denominator
        # Guard against zero/missing income
        income = df['median_homeowner_income'].replace(0, np.nan)

        df['insurance_pct_income'] = df['insurance_premium_annual'] / income * 100

        df['property_taxes_pct_income'] = df['median_property_taxes_annual'] / income * 100

        # Utilities approximation: monthly no-mortgage owner costs include taxes,
        # insurance, and utilities; subtract taxes (monthly) and insurance (monthly)
        # to isolate the utilities + HOA + other component.
        monthly_costs = df['median_monthly_owner_costs_no_mortgage']
        monthly_taxes = df['median_property_taxes_annual'] / 12
        monthly_insurance = df['insurance_premium_annual'] / 12
        residual_monthly = (monthly_costs - monthly_taxes - monthly_insurance).clip(lower=0)
        df['utilities_pct_income'] = residual_monthly * 12 / income * 100

        # Drop raw intermediate columns not used as model features
        df.drop(columns=[
            'total_net_migration', 'bea_population',
            'median_property_taxes_annual',
            'median_monthly_owner_costs_no_mortgage',
            'insurance_premium_annual',
        ], inplace=True)

        # Impute remaining missing values with column medians
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled {col} missing values with median: {median_val:.2f}")

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
        feature_cols = [
            'median_homeowner_income',
            'net_earnings_per_capita',
            'dividends_interest_rent_per_capita',
            'transfer_receipts_per_capita',
            'utilities_pct_income',
            'insurance_pct_income',
            'property_taxes_pct_income',
            'net_migration_rate',
            'unemployment_rate',
            'new_listings_yoy',
            'homes_sold_yoy',
            'median_dom_yoy',
        ]

        # Only keep features that exist in the dataframe
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
        groups_train: np.ndarray,
        custom_params: Optional[Dict] = None,
        tune_hyperparams: bool = False
    ) -> Tuple[object, Dict]:
        """
        Train a single model with spatial cross-validation.

        Args:
            model_name: Name of model type from MODEL_CONFIGS
            X_train: Training features
            y_train: Ordinal-encoded labels
            groups_train: State FIPS array for GroupKFold
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

        spatial_cv = GroupKFold(n_splits=self.n_spatial_splits)

        if tune_hyperparams:
            print(f"\nTuning hyperparameters for {model_name}...")
            grid_search = GridSearchCV(
                model,
                config['param_grid'],
                cv=spatial_cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train, groups=groups_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
        else:
            model.fit(X_train, y_train)
            best_params = params

        # Spatial cross-validation scores on the training set
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=spatial_cv, groups=groups_train,
            scoring='f1_weighted'
        )

        metrics = {
            'model_name': model_name,
            'params': best_params,
            'spatial_cv_f1_mean': cv_scores.mean(),
            'spatial_cv_f1_std': cv_scores.std(),
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
        Evaluate model on the spatial holdout set.

        Ordinal MAE and adjacent accuracy are the primary metrics — they reward
        predictions that are close on the risk scale even when not exactly right.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Ordinal-encoded labels (0–4)
            model_name: Model identifier

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)

        # Standard classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        # Ordinal metrics — penalize by distance on the risk scale
        ordinal_mae = mean_absolute_error(y_test, y_pred)

        # Adjacent accuracy: fraction of predictions within ±1 ordinal step
        adjacent_accuracy = float(np.mean(np.abs(y_pred - y_test) <= 1))

        # Classification report using human-readable class labels
        class_labels = [RISK_ORDER_INVERSE[i] for i in sorted(RISK_ORDER_INVERSE)]
        present_classes = sorted(set(y_test) | set(y_pred))
        report = classification_report(
            y_test, y_pred,
            labels=present_classes,
            target_names=[RISK_ORDER_INVERSE[i] for i in present_classes],
            output_dict=True,
            zero_division=0
        )

        # Confusion matrix ordered by risk level
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(RISK_ORDER))))

        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'ordinal_mae': ordinal_mae,
            'adjacent_accuracy': adjacent_accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        # Permutation importance on the holdout set, scored by ordinal MAE degradation.
        # Negative values mean shuffling that feature *improved* MAE (noise feature).
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        # Higher mean = larger MAE increase when shuffled = more important.
        # Negate because scoring returns negative MAE.
        perm_mean = -perm.importances_mean
        perm_std = perm.importances_std
        indices = np.argsort(perm_mean)[::-1]
        metrics['top_features'] = [
            {
                'feature': self.feature_names[i],
                'importance': float(perm_mean[i]),
                'importance_std': float(perm_std[i])
            }
            for i in indices
        ]

        return metrics

    def train_all_models(
        self,
        df: pd.DataFrame,
        tune_hyperparams: bool = False,
    ) -> Dict:
        """
        Train and evaluate all configured models with ordinal encoding and spatial CV.

        Spatial split: the last GroupKFold fold (one held-out state group) is used
        as the test set; remaining folds form the training set. Spatial CV on the
        training set gives an honest estimate of out-of-region generalisation.

        Args:
            df: Input DataFrame with features (must include 'state_fips' column)
            tune_hyperparams: Whether to run spatial grid search

        Returns:
            Dictionary of results for all models
        """
        print("Engineering features...")
        df_engineered = self.engineer_features(df)

        print("\nPreparing features...")
        X, y_labels, feature_names = self.prepare_features(df_engineered)

        # Ordinal encoding: map risk labels to integers 0-4
        unknown = set(y_labels) - set(RISK_ORDER)
        if unknown:
            raise ValueError(f"Unknown risk labels in data: {unknown}")
        y = np.array([RISK_ORDER[label] for label in y_labels])

        # State FIPS groups for spatial splitting — aligned to df_engineered index
        groups = df_engineered['state_fips'].values

        # Spatial train/test split: reserve the last GroupKFold fold as the holdout.
        # This ensures no county from the test states leaks into training.
        gkf = GroupKFold(n_splits=self.n_spatial_splits)
        splits = list(gkf.split(X, y, groups))
        train_idx, test_idx = splits[-1]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # Scale on train only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        test_states = sorted(set(groups[test_idx]))
        print(f"\nTraining set: {X_train.shape} counties")
        print(f"Spatial holdout: {X_test.shape} counties from states {test_states}")

        results = {
            'feature_names': feature_names,
            'risk_order': RISK_ORDER,
            'spatial_cv_n_splits': self.n_spatial_splits,
            'holdout_states': test_states,
            'models': {}
        }

        for model_name in self.MODEL_CONFIGS.keys():
            print(f"\n{'='*60}")
            print(f"Training {model_name}...")
            print(f"{'='*60}")

            model, train_metrics = self.train_model(
                model_name,
                X_train_scaled,
                y_train,
                groups_train,
                tune_hyperparams=tune_hyperparams
            )

            eval_metrics = self.evaluate_model(
                model, X_test_scaled, y_test, model_name
            )

            combined_metrics = {**train_metrics, **eval_metrics}
            self.models[model_name] = model
            results['models'][model_name] = combined_metrics

            print(f"\n{model_name} Results:")
            print(f"  Spatial CV F1:     {train_metrics['spatial_cv_f1_mean']:.4f} "
                  f"(+/- {train_metrics['spatial_cv_f1_std']:.4f})")
            print(f"  Holdout Accuracy:  {eval_metrics['accuracy']:.4f}")
            print(f"  Holdout F1:        {eval_metrics['f1_weighted']:.4f}")
            print(f"  Ordinal MAE:       {eval_metrics['ordinal_mae']:.4f} "
                  f"(0 = perfect, 4 = worst)")
            print(f"  Adjacent Accuracy: {eval_metrics['adjacent_accuracy']:.4f} "
                  f"(within ±1 risk level)")

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

        # Save scaler
        scaler_path = self.output_dir / f'{prefix}scaler_{timestamp}.joblib'
        joblib.dump(self.scaler, scaler_path)

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
                'risk_order': RISK_ORDER,
            }, f, indent=2)
        print(f"Saved feature info: {features_path}")

    def get_best_model(self) -> Tuple[str, object, float]:
        """
        Identify the best performing model by ordinal MAE (lower = better).

        Returns:
            Tuple of (model_name, model_object, ordinal_mae)
        """
        best_model_name = None
        best_mae = float('inf')

        for model_name, metrics in self.results['models'].items():
            mae = metrics['ordinal_mae']
            if mae < best_mae:
                best_mae = mae
                best_model_name = model_name

        return best_model_name, self.models[best_model_name], best_mae


def train_all_hazards(tune_hyperparams: bool = False, min_year: Optional[int] = None, max_year: Optional[int] = None):
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

    best_name, _, best_mae = overall_predictor.get_best_model()
    print(f"\nBest Overall Model: {best_name} (Ordinal MAE: {best_mae:.4f})")

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

            best_name, _, best_mae = predictor.get_best_model()
            print(f"\nBest {hazard_name} Model: {best_name} (Ordinal MAE: {best_mae:.4f})")

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
    df = predictor.load_data()

    # Train all models
    print("\nTraining models...")
    results = predictor.train_all_models(df, tune_hyperparams=False)

    # Save everything
    print("\nSaving models and results...")
    predictor.save_models()

    # Report best model
    best_name, best_model, best_mae = predictor.get_best_model()
    print(f"\n{'='*60}")
    print(f"Best Model: {best_name}")
    print(f"Ordinal MAE: {best_mae:.4f}")
    print(f"{'='*60}")

    return predictor, results


if __name__ == '__main__':
    predictor, results = main()
