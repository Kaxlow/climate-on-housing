from __future__ import annotations

import pandas as pd


def prepare_natural_disasters_df() -> pd.DataFrame:
    from housing_climate_risk.page_data.climate_housing_utils import prepare_natural_disasters_df as _prepare_natural_disasters_df

    return _prepare_natural_disasters_df()


def prepare_housing_df(*, include_profiles: bool = True) -> pd.DataFrame:
    from housing_climate_risk.page_data.climate_housing_utils import prepare_housing_df as _prepare_housing_df

    return _prepare_housing_df(include_profiles=include_profiles)


def build_county_profiles() -> dict[str, object]:
    from housing_climate_risk.page_data.climate_housing_utils import build_county_profiles as _build_county_profiles

    return _build_county_profiles()
