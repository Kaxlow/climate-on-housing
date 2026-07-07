"""
Feature Engineering and Preprocessing for Climate Risk Prediction

This module provides reusable preprocessing and feature engineering functions
that can be applied consistently across different modeling experiments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngineer:
    """
    Handles feature engineering for climate risk prediction.

    This class encapsulates all feature transformation logic, making it easy to
    apply consistent preprocessing across training and inference.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with optional configuration.

        Args:
            config: Dictionary with feature engineering parameters
        """
        self.config = config or self._default_config()
        self.fitted = False

    @staticmethod
    def _default_config() -> Dict:
        """Return default feature engineering configuration."""
        return {
            'income_brackets': {
                'bins': [0, 40000, 60000, 80000, float('inf')],
                'labels': ['low', 'medium', 'high', 'very_high']
            },
            'burden_brackets': {
                'bins': [0, 20, 35, 50, 100],
                'labels': ['minimal', 'moderate', 'high', 'severe']
            },
            'fill_strategy': 'median'
        }

    def create_income_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create income-based features.

        Args:
            df: DataFrame with median_household_income column

        Returns:
            DataFrame with additional income features
        """
        df = df.copy()

        # Income bracket
        if 'median_household_income' in df.columns:
            df['income_bracket'] = pd.cut(
                df['median_household_income'],
                bins=self.config['income_brackets']['bins'],
                labels=self.config['income_brackets']['labels']
            )

        return df

    def create_burden_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create housing burden features.

        Args:
            df: DataFrame with housing burden columns

        Returns:
            DataFrame with additional burden features
        """
        df = df.copy()

        # Burden severity
        if 'housing_burden_30pct' in df.columns:
            df['burden_severity'] = pd.cut(
                df['housing_burden_30pct'],
                bins=self.config['burden_brackets']['bins'],
                labels=self.config['burden_brackets']['labels']
            )

        return df

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio-based features (costs per income).

        Args:
            df: DataFrame with cost and income columns

        Returns:
            DataFrame with ratio features
        """
        df = df.copy()

        # Insurance burden (annual premium as % of income)
        if 'insurance_premium' in df.columns and 'median_household_income' in df.columns:
            df['insurance_burden'] = (
                df['insurance_premium'] * 12 / df['median_household_income'] * 100
            )

        # Housing cost burden (annual costs as % of income)
        if 'property_taxes_utilities' in df.columns and 'median_household_income' in df.columns:
            df['housing_cost_burden'] = (
                df['property_taxes_utilities'] * 12 / df['median_household_income'] * 100
            )

        return df

    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create housing market trend features.

        Args:
            df: DataFrame with housing market columns

        Returns:
            DataFrame with market features
        """
        df = df.copy()

        # Market cooling indicator
        if 'median_dom_yoy' in df.columns and 'homes_sold_yoy' in df.columns:
            df['market_cooling'] = (
                df['median_dom_yoy'].fillna(0) - df['homes_sold_yoy'].fillna(0)
            )

        # Market stress (high burden + cooling market)
        if 'housing_burden_30pct' in df.columns and 'market_cooling' in df.columns:
            df['market_stress'] = (
                df['housing_burden_30pct'].fillna(0) * 0.5 +
                df['market_cooling'].fillna(0) * 0.5
            )

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()

        # High burden + low migration (people trapped)
        if 'housing_burden_30pct' in df.columns and 'net_migration_rate' in df.columns:
            df['burden_migration_interaction'] = (
                df['housing_burden_30pct'].fillna(0) *
                (1 - df['net_migration_rate'].fillna(0) / 100)
            )

        # Insurance cost + climate risk awareness proxy
        if 'insurance_premium' in df.columns and 'median_dom_yoy' in df.columns:
            df['insurance_market_interaction'] = (
                df['insurance_premium'].fillna(0) *
                (1 + df['median_dom_yoy'].fillna(0) / 100)
            )

        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in numeric columns.

        Args:
            df: DataFrame with potential missing values
            strategy: Strategy to use ('median', 'mean', or 'zero')

        Returns:
            DataFrame with filled missing values
        """
        df = df.copy()
        strategy = strategy or self.config.get('fill_strategy', 'median')

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isna().any():
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                else:  # zero
                    fill_value = 0

                df[col] = df[col].fillna(fill_value)

        return df

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        drop_first: bool = True
    ) -> pd.DataFrame:
        """
        One-hot encode categorical features.

        Args:
            df: DataFrame with categorical columns
            drop_first: Whether to drop first dummy category

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()

        categorical_cols = []
        if 'income_bracket' in df.columns:
            categorical_cols.append('income_bracket')
        if 'burden_severity' in df.columns:
            categorical_cols.append('burden_severity')

        if categorical_cols:
            df = pd.get_dummies(
                df,
                columns=categorical_cols,
                drop_first=drop_first,
                dtype=float
            )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: Raw feature DataFrame

        Returns:
            DataFrame with engineered features
        """
        # Apply all transformations
        df = self.create_income_features(df)
        df = self.create_burden_features(df)
        df = self.create_ratio_features(df)
        df = self.create_market_features(df)
        df = self.create_interaction_features(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df)

        self.fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted parameters.

        Args:
            df: Raw feature DataFrame

        Returns:
            DataFrame with engineered features
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        return self.fit_transform(df)


def select_modeling_features(df: pd.DataFrame) -> List[str]:
    """
    Select final feature columns for modeling.

    Args:
        df: DataFrame with engineered features

    Returns:
        List of feature column names
    """
    base_features = [
        'median_household_income',
        'housing_burden_30pct',
        'insurance_premium',
        'property_taxes_utilities',
        'net_migration_rate',
        'homes_sold_yoy',
        'median_dom_yoy'
    ]

    derived_features = [
        'insurance_burden',
        'housing_cost_burden',
        'market_cooling',
        'market_stress',
        'burden_migration_interaction',
        'insurance_market_interaction'
    ]

    # Add categorical encoded features
    categorical_features = [
        col for col in df.columns
        if col.startswith(('income_', 'burden_'))
    ]

    all_features = base_features + derived_features + categorical_features
    available_features = [col for col in all_features if col in df.columns]

    return available_features


def prepare_model_data(
    df: pd.DataFrame,
    target_col: str = 'risk_rating',
    feature_engineer: Optional[FeatureEngineer] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and target for modeling.

    Args:
        df: Raw DataFrame
        target_col: Name of target column
        feature_engineer: Optional fitted feature engineer

    Returns:
        Tuple of (X features, y target, feature_names)
    """
    if feature_engineer is None:
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.fit_transform(df)
    else:
        df_features = feature_engineer.transform(df)

    feature_cols = select_modeling_features(df_features)

    X = df_features[feature_cols].values
    y = df_features[target_col].values

    return X, y, feature_cols


def get_feature_metadata(feature_names: List[str]) -> Dict[str, Dict]:
    """
    Get metadata about features for interpretability.

    Args:
        feature_names: List of feature names

    Returns:
        Dictionary mapping feature names to metadata
    """
    metadata = {
        'median_household_income': {
            'description': 'Median household income',
            'unit': 'USD',
            'source': 'ACS'
        },
        'housing_burden_30pct': {
            'description': 'Share with housing costs >= 30% of income',
            'unit': 'percentage',
            'source': 'ACS'
        },
        'insurance_premium': {
            'description': 'Mean homeowner insurance premium',
            'unit': 'USD',
            'source': 'Insurance data'
        },
        'property_taxes_utilities': {
            'description': 'Median owner costs with mortgage',
            'unit': 'USD',
            'source': 'ACS'
        },
        'net_migration_rate': {
            'description': 'Net migration rate',
            'unit': 'percentage',
            'source': 'ACS'
        },
        'homes_sold_yoy': {
            'description': 'Year-over-year change in homes sold',
            'unit': 'percentage',
            'source': 'Redfin'
        },
        'median_dom_yoy': {
            'description': 'Year-over-year change in days on market',
            'unit': 'percentage',
            'source': 'Redfin'
        },
        'insurance_burden': {
            'description': 'Annual insurance premium as % of income',
            'unit': 'percentage',
            'derived': True
        },
        'housing_cost_burden': {
            'description': 'Annual housing costs as % of income',
            'unit': 'percentage',
            'derived': True
        },
        'market_cooling': {
            'description': 'Market cooling indicator (DOM YoY - Sales YoY)',
            'unit': 'composite',
            'derived': True
        },
        'market_stress': {
            'description': 'Market stress (burden + cooling)',
            'unit': 'composite',
            'derived': True
        },
        'burden_migration_interaction': {
            'description': 'High burden + low migration interaction',
            'unit': 'composite',
            'derived': True
        },
        'insurance_market_interaction': {
            'description': 'Insurance cost + market timing interaction',
            'unit': 'composite',
            'derived': True
        }
    }

    return {name: metadata.get(name, {'description': name})
            for name in feature_names}
