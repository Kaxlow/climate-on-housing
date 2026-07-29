from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import DatasetBuildResult, RISK_ORDER


MODEL_NAMES = ["elastic_net", "gradient_boosted_trees"]
TARGET_COLUMN = "relative_median_ppsf_yoy"
SAMPLE_INDICATOR = "is_very_high"
MODEL_GROUP_MEMBERS = {
    "Very Low": ["Very Low"],
    "Low": ["Low"],
    "Medium": ["Medium"],
    "High + Very High": ["High", "Very High"],
}
MODEL_GROUP_ORDER = list(MODEL_GROUP_MEMBERS)


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    random_state: int = 20260723
    outer_repeats: int = 3
    max_outer_splits: int = 5
    inner_splits: int = 3
    gradient_search_iterations: int = 12
    minimum_feature_coverage: float = 0.20
    maximum_absolute_correlation: float = 0.85
    n_jobs: int = -1


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _risk_slug(risk_group: str) -> str:
    return risk_group.lower().replace(" + ", "_").replace(" ", "_")


def _outer_split_count(county_count: int, maximum: int) -> int:
    if county_count < 6:
        raise ValueError(f"At least 6 counties are required; received {county_count}")
    return min(maximum, max(3, county_count // 5))


def _inner_split_count(training_count: int, maximum: int) -> int:
    return min(maximum, max(2, training_count // 4))


def _available_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    feature_meta: dict[str, tuple[str, str]],
    minimum_coverage: float,
) -> tuple[list[str], list[dict[str, object]]]:
    minimum_non_null = max(3, math.ceil(len(frame) * minimum_coverage))
    selected: list[str] = []
    coverage_rows: list[dict[str, object]] = []
    for feature in feature_columns:
        values = pd.to_numeric(frame[feature], errors="coerce")
        non_null = int(values.notna().sum())
        unique = int(values.dropna().nunique())
        reason = None
        if non_null < minimum_non_null:
            reason = "insufficient_non_null_values"
        elif unique < 2:
            reason = "constant_or_single_value"
        else:
            selected.append(feature)
        coverage_rows.append(
            {
                "feature": feature,
                "feature_group": feature_meta[feature][0],
                "feature_label": feature_meta[feature][1],
                "non_null_count": non_null,
                "non_null_fraction": non_null / len(frame),
                "unique_non_null_values": unique,
                "included": reason is None,
                "exclusion_reason": reason,
            }
        )
    return selected, coverage_rows


def _prune_high_correlations(
    frame: pd.DataFrame,
    features: list[str],
    *,
    threshold: float,
) -> tuple[list[str], dict[str, str]]:
    """Keep the first catalog feature from each highly correlated pair."""
    selected: list[str] = []
    exclusions: dict[str, str] = {}
    correlations = frame[features].corr(method="spearman", min_periods=3)
    for feature in features:
        conflict = next(
            (
                kept
                for kept in selected
                if pd.notna(correlations.loc[feature, kept])
                and abs(correlations.loc[feature, kept]) >= threshold
            ),
            None,
        )
        if conflict is None:
            selected.append(feature)
        else:
            exclusions[feature] = (
                f"high_absolute_correlation_with:{conflict}:"
                f"{correlations.loc[feature, conflict]:.6f}"
            )
    return selected, exclusions


def _expand_group_interactions(values: np.ndarray) -> np.ndarray:
    """Append predictor-by-sample-indicator interactions after imputation."""
    predictors = values[:, :-1]
    indicator = values[:, -1:]
    return np.column_stack([predictors, indicator, predictors * indicator])


def _elastic_pipeline(*, add_group_interactions: bool = False) -> Pipeline:
    regressor_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if add_group_interactions:
        regressor_steps.append(
            (
                "group_interactions",
                FunctionTransformer(_expand_group_interactions, validate=False),
            )
        )
    regressor_steps.extend(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    max_iter=50_000,
                    selection="cyclic",
                    tol=1e-5,
                    random_state=0,
                ),
            ),
        ]
    )
    regressor = Pipeline(regressor_steps)
    return Pipeline(
        [
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=regressor,
                    transformer=StandardScaler(),
                ),
            )
        ]
    )


def _gradient_pipeline(
    random_state: int,
    *,
    add_group_interactions: bool = False,
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if add_group_interactions:
        steps.append(
            (
                "group_interactions",
                FunctionTransformer(_expand_group_interactions, validate=False),
            )
        )
    steps.append(
        ("model", GradientBoostingRegressor(random_state=random_state, loss="huber"))
    )
    return Pipeline(steps)


def _make_search(
    model_name: str,
    *,
    inner_cv: KFold,
    random_state: int,
    n_jobs: int,
    gradient_search_iterations: int,
    add_group_interactions: bool = False,
) -> GridSearchCV | RandomizedSearchCV:
    if model_name == "elastic_net":
        estimator = _elastic_pipeline(
            add_group_interactions=add_group_interactions
        )
        parameters = {
            "regressor__regressor__model__alpha": np.logspace(-4, 0, 5),
            "regressor__regressor__model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
        }
        return GridSearchCV(
            estimator,
            parameters,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
        )
    if model_name == "gradient_boosted_trees":
        estimator = _gradient_pipeline(
            random_state,
            add_group_interactions=add_group_interactions,
        )
        parameters = {
            "model__n_estimators": [75, 125, 200, 300],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [1, 2, 3],
            "model__min_samples_leaf": [3, 5, 10, 20],
            "model__subsample": [0.7, 0.9, 1.0],
        }
        return RandomizedSearchCV(
            estimator,
            parameters,
            n_iter=gradient_search_iterations,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            random_state=random_state,
            n_jobs=n_jobs,
            refit=True,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) > 1 and np.ptp(y_true) > 1e-12 and np.ptp(y_pred) > 1e-12:
        spearman = float(spearmanr(y_true, y_pred).statistic)
    else:
        spearman = np.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "spearman": spearman,
    }


def _evaluate_group(
    group_frame: pd.DataFrame,
    features: list[str],
    risk_group: str,
    config: TrainingConfig,
    *,
    add_group_interactions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = group_frame.reset_index(drop=True)
    x = frame[features]
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    outer_splits = _outer_split_count(len(frame), config.max_outer_splits)

    for repeat in range(config.outer_repeats):
        outer_cv = KFold(
            n_splits=outer_splits,
            shuffle=True,
            random_state=config.random_state + repeat,
        )
        for fold, (train_index, test_index) in enumerate(outer_cv.split(x), start=1):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            inner_splits = _inner_split_count(len(train_index), config.inner_splits)
            baseline_prediction = np.repeat(np.median(y_train), len(test_index))
            baseline_metrics = _metrics(y_test, baseline_prediction)
            fold_rows.append(
                {
                    "risk_group": risk_group,
                    "model": "median_baseline",
                    "repeat": repeat + 1,
                    "fold": fold,
                    "county_count": len(frame),
                    "train_count": len(train_index),
                    "test_count": len(test_index),
                    "inner_splits": 0,
                    "best_parameters": "{}",
                    **baseline_metrics,
                }
            )
            for row_index, prediction in zip(test_index, baseline_prediction):
                prediction_rows.append(
                    {
                        "risk_group": risk_group,
                        "model": "median_baseline",
                        "repeat": repeat + 1,
                        "fold": fold,
                        "row_index": int(row_index),
                        "fips": frame.loc[row_index, "fips"],
                        "prediction": float(prediction),
                    }
                )

            inner_cv = KFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=config.random_state + repeat * 100 + fold,
            )
            for model_offset, model_name in enumerate(MODEL_NAMES):
                search = _make_search(
                    model_name,
                    inner_cv=inner_cv,
                    random_state=config.random_state + repeat * 1000 + fold * 10 + model_offset,
                    n_jobs=config.n_jobs,
                    gradient_search_iterations=config.gradient_search_iterations,
                    add_group_interactions=add_group_interactions,
                )
                search.fit(x_train, y_train)
                prediction = search.predict(x_test)
                fold_rows.append(
                    {
                        "risk_group": risk_group,
                        "model": model_name,
                        "repeat": repeat + 1,
                        "fold": fold,
                        "county_count": len(frame),
                        "train_count": len(train_index),
                        "test_count": len(test_index),
                        "inner_splits": inner_splits,
                        "best_parameters": json.dumps(_json_safe(search.best_params_), sort_keys=True),
                        **_metrics(y_test, prediction),
                    }
                )
                for row_index, value in zip(test_index, prediction):
                    prediction_rows.append(
                        {
                            "risk_group": risk_group,
                            "model": model_name,
                            "repeat": repeat + 1,
                            "fold": fold,
                            "row_index": int(row_index),
                            "fips": frame.loc[row_index, "fips"],
                            "prediction": float(value),
                        }
                    )

    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def _summarize_evaluation(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_metrics.groupby(["risk_group", "model"], as_index=False)
        .agg(
            folds=("fold", "size"),
            county_count=("county_count", "first"),
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2=("r2", "mean"),
            r2_std=("r2", "std"),
            spearman=("spearman", "mean"),
            spearman_std=("spearman", "std"),
        )
    )
    for metric in ["mae", "mae_std", "rmse", "rmse_std"]:
        summary[f"{metric}_percentage_points"] = summary[metric] * 100
    summary["selected"] = False
    for risk_group in MODEL_GROUP_ORDER:
        candidate_rows = summary.loc[
            (summary["risk_group"] == risk_group) & summary["model"].isin(MODEL_NAMES)
        ]
        if candidate_rows.empty:
            continue
        winner_index = candidate_rows.sort_values(["mae", "rmse", "model"]).index[0]
        summary.loc[winner_index, "selected"] = True
    return summary


def _fit_final_model(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    config: TrainingConfig,
    *,
    add_group_interactions: bool = False,
) -> GridSearchCV | RandomizedSearchCV:
    inner_splits = _inner_split_count(len(frame), config.inner_splits)
    inner_cv = KFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    search = _make_search(
        model_name,
        inner_cv=inner_cv,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        gradient_search_iterations=config.gradient_search_iterations,
        add_group_interactions=add_group_interactions,
    )
    search.fit(frame[features], frame[TARGET_COLUMN])
    return search


def _feature_importance(
    estimator: Pipeline,
    features: list[str],
    risk_group: str,
    model_name: str,
    feature_meta: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    if model_name == "elastic_net":
        transformed = estimator.named_steps["regressor"]
        coefficients = transformed.regressor_.named_steps["model"].coef_
        values = np.asarray(coefficients, dtype=float)
        importance_type = "standardized_coefficient"
    else:
        values = np.asarray(estimator.named_steps["model"].feature_importances_, dtype=float)
        importance_type = "impurity_importance"
    return pd.DataFrame(
        {
            "risk_group": risk_group,
            "model": model_name,
            "feature": features,
            "feature_group": [feature_meta[feature][0] for feature in features],
            "feature_label": [feature_meta[feature][1] for feature in features],
            "importance_type": importance_type,
            "importance": values,
            "absolute_importance": np.abs(values),
        }
    ).sort_values("absolute_importance", ascending=False)


def _subgroup_permutation_importance(
    estimator: Pipeline,
    frame: pd.DataFrame,
    input_features: list[str],
    base_features: list[str],
    risk_group: str,
    model_name: str,
    feature_meta: dict[str, tuple[str, str]],
    *,
    random_state: int,
    repeats: int = 20,
) -> pd.DataFrame:
    """Measure feature reliance within one original risk group using MAE."""
    subgroup = frame.loc[frame["risk_group"].eq(risk_group)].copy()
    x = subgroup[input_features]
    y = subgroup[TARGET_COLUMN].to_numpy(dtype=float)
    baseline_mae = mean_absolute_error(y, estimator.predict(x))
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for feature in base_features:
        deltas = []
        values = x[feature].to_numpy(copy=True)
        for _ in range(repeats):
            permuted = x.copy()
            permuted[feature] = rng.permutation(values)
            permuted_mae = mean_absolute_error(y, estimator.predict(permuted))
            deltas.append(permuted_mae - baseline_mae)
        raw_importance = float(np.mean(deltas))
        rows.append(
            {
                "risk_group": risk_group,
                "model_group": "High + Very High",
                "model": model_name,
                "feature": feature,
                "feature_group": feature_meta[feature][0],
                "feature_label": feature_meta[feature][1],
                "importance_type": "subgroup_permutation_mae",
                "importance": raw_importance,
                "absolute_importance": max(0.0, raw_importance),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["absolute_importance", "feature"],
        ascending=[False, True],
    )


def _save_plots(evaluation: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    candidates = evaluation.loc[evaluation["model"].isin(MODEL_NAMES)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(MODEL_GROUP_ORDER))
    width = 0.36
    for offset, model_name in enumerate(MODEL_NAMES):
        rows = (
            candidates.loc[candidates["model"] == model_name]
            .set_index("risk_group")
            .reindex(MODEL_GROUP_ORDER)
        )
        ax.bar(
            x_positions + (offset - 0.5) * width,
            rows["mae_percentage_points"],
            width,
            yerr=rows["mae_std_percentage_points"],
            label=model_name.replace("_", " ").title(),
            capsize=3,
        )
    ax.set_xticks(x_positions, MODEL_GROUP_ORDER)
    ax.set_ylabel("Repeated nested-CV MAE (percentage points)")
    ax.set_title("County Relative Median PPSF YoY Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True, sharey=True)
    for axis, risk_group in zip(axes, RISK_ORDER):
        rows = predictions.loc[predictions["risk_group"] == risk_group]
        axis.scatter(
            rows["actual_relative_median_ppsf_yoy"] * 100,
            rows["oof_prediction"] * 100,
            s=13,
            alpha=0.55,
        )
        bounds = np.array(
            [
                rows["actual_relative_median_ppsf_yoy"].min(),
                rows["actual_relative_median_ppsf_yoy"].max(),
            ]
        ) * 100
        axis.plot(bounds, bounds, color="#555", linestyle="--", linewidth=1)
        axis.set_title(risk_group)
        axis.set_xlabel("Observed")
    axes[0].set_ylabel("OOF prediction")
    fig.suptitle("Observed vs Out-of-Fold Relative Median PPSF YoY (percentage points)")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_vs_oof.png", dpi=160)
    plt.close(fig)


def train_all_risk_groups(
    dataset: DatasetBuildResult,
    config: TrainingConfig,
) -> dict[str, object]:
    """Compare model families and persist one model per training group."""
    output_dir = config.output_dir
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset.frame.to_parquet(output_dir / "county_modeling_dataset.parquet", index=False)
    all_fold_metrics: list[pd.DataFrame] = []
    all_raw_predictions: list[pd.DataFrame] = []
    all_coverage: list[dict[str, object]] = []
    group_features: dict[str, list[str]] = {}
    group_base_features: dict[str, list[str]] = {}
    feature_columns = list(dataset.metadata["feature_columns"])
    feature_meta = {
        key: tuple(value)
        for key, value in dataset.metadata["feature_meta"].items()
    }

    for model_group, risk_groups in MODEL_GROUP_MEMBERS.items():
        group_frame = dataset.frame.loc[
            dataset.frame["risk_group"].isin(risk_groups)
        ].copy()
        base_features, coverage = _available_features(
            group_frame,
            feature_columns,
            feature_meta,
            config.minimum_feature_coverage,
        )
        base_features, correlation_exclusions = _prune_high_correlations(
            group_frame,
            base_features,
            threshold=config.maximum_absolute_correlation,
        )
        for row in coverage:
            if row["feature"] in correlation_exclusions:
                row["included"] = False
                row["exclusion_reason"] = correlation_exclusions[row["feature"]]
        if not base_features:
            raise ValueError(f"No usable features for {model_group}")
        add_group_interactions = len(risk_groups) > 1
        if add_group_interactions:
            group_frame[SAMPLE_INDICATOR] = (
                group_frame["risk_group"].eq("Very High").astype(float)
            )
        features = base_features + (
            [SAMPLE_INDICATOR] if add_group_interactions else []
        )
        group_features[model_group] = features
        group_base_features[model_group] = base_features
        all_coverage.extend(
            {
                "risk_group": model_group,
                "model_group": model_group,
                "risk_groups": " | ".join(risk_groups),
                **row,
            }
            for row in coverage
        )
        fold_metrics, raw_predictions = _evaluate_group(
            group_frame,
            features,
            model_group,
            config,
            add_group_interactions=add_group_interactions,
        )
        all_fold_metrics.append(fold_metrics)
        all_raw_predictions.append(raw_predictions)

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    raw_predictions = pd.concat(all_raw_predictions, ignore_index=True)
    evaluation = _summarize_evaluation(fold_metrics)
    selected_models = {
        row.risk_group: row.model
        for row in evaluation.loc[evaluation["selected"]].itertuples(index=False)
    }

    prediction_outputs: list[pd.DataFrame] = []
    importance_outputs: list[pd.DataFrame] = []
    model_manifest: dict[str, object] = {}
    written_model_paths: set[Path] = set()
    for model_group, risk_groups in MODEL_GROUP_MEMBERS.items():
        group_frame = dataset.frame.loc[
            dataset.frame["risk_group"].isin(risk_groups)
        ].reset_index(drop=True)
        add_group_interactions = len(risk_groups) > 1
        if add_group_interactions:
            group_frame[SAMPLE_INDICATOR] = (
                group_frame["risk_group"].eq("Very High").astype(float)
            )
        features = group_features[model_group]
        base_features = group_base_features[model_group]
        model_name = selected_models[model_group]
        final_search = _fit_final_model(
            group_frame,
            features,
            model_name,
            config,
            add_group_interactions=add_group_interactions,
        )
        final_estimator = final_search.best_estimator_
        fitted_prediction = final_estimator.predict(group_frame[features])
        selected_oof = raw_predictions.loc[
            (raw_predictions["risk_group"] == model_group)
            & (raw_predictions["model"] == model_name)
        ]
        oof = (
            selected_oof.groupby("row_index")["prediction"]
            .agg(oof_prediction="mean", oof_prediction_std="std")
            .reindex(group_frame.index)
        )
        output = group_frame[
            [
                "fips",
                "county",
                "state",
                "risk_group",
                "risk_score",
                "county_median_ppsf_yoy",
                "group_median_ppsf_yoy",
                TARGET_COLUMN,
            ]
        ].copy()
        output["model_group"] = model_group
        output["selected_model"] = model_name
        output["oof_prediction"] = oof["oof_prediction"].to_numpy()
        output["oof_prediction_std"] = oof["oof_prediction_std"].fillna(0).to_numpy()
        output["oof_residual"] = output[TARGET_COLUMN] - output["oof_prediction"]
        output["fitted_prediction"] = fitted_prediction
        for column in [
            "county_median_ppsf_yoy",
            "group_median_ppsf_yoy",
            TARGET_COLUMN,
            "oof_prediction",
            "oof_prediction_std",
            "oof_residual",
            "fitted_prediction",
        ]:
            output[f"{column}_percentage_points"] = output[column] * 100
        output = output.rename(columns={TARGET_COLUMN: "actual_relative_median_ppsf_yoy"})
        prediction_outputs.append(output)
        if add_group_interactions:
            for risk_offset, risk_group in enumerate(risk_groups):
                importance_outputs.append(
                    _subgroup_permutation_importance(
                        final_estimator,
                        group_frame,
                        features,
                        base_features,
                        risk_group,
                        model_name,
                        feature_meta,
                        random_state=config.random_state + risk_offset,
                    )
                )
        else:
            model_importance = _feature_importance(
                final_estimator,
                features,
                model_group,
                model_name,
                feature_meta,
            ).rename(columns={"risk_group": "model_group"})
            for risk_group in risk_groups:
                risk_importance = model_importance.copy()
                risk_importance.insert(0, "risk_group", risk_group)
                importance_outputs.append(risk_importance)

        artifact = {
            "pipeline": final_estimator,
            "model_group": model_group,
            "risk_groups": risk_groups,
            "model_name": model_name,
            "features": features,
            "base_features": base_features,
            "sample_indicator": SAMPLE_INDICATOR if add_group_interactions else None,
            "interaction_features": (
                [f"{feature} * {SAMPLE_INDICATOR}" for feature in base_features]
                if add_group_interactions
                else []
            ),
            "target": TARGET_COLUMN,
            "target_definition": dataset.metadata["target_definition"],
            "best_parameters": _json_safe(final_search.best_params_),
            "maximum_absolute_correlation": config.maximum_absolute_correlation,
        }
        model_path = models_dir / f"{_risk_slug(model_group)}.joblib"
        joblib.dump(artifact, model_path)
        written_model_paths.add(model_path.resolve())
        model_manifest[model_group] = {
            "model_name": model_name,
            "model_path": model_path.relative_to(output_dir).as_posix(),
            "county_count": len(group_frame),
            "risk_groups": risk_groups,
            "feature_count": len(features),
            "features": features,
            "base_feature_count": len(base_features),
            "base_features": base_features,
            "sample_indicator": SAMPLE_INDICATOR if add_group_interactions else None,
            "interaction_feature_count": len(base_features) if add_group_interactions else 0,
            "engineered_feature_count": (
                len(base_features) * 2 + 1
                if add_group_interactions
                else len(base_features)
            ),
            "best_parameters": _json_safe(final_search.best_params_),
        }

    for existing_model_path in models_dir.glob("*.joblib"):
        if existing_model_path.resolve() not in written_model_paths:
            existing_model_path.unlink()

    predictions = pd.concat(prediction_outputs, ignore_index=True)
    feature_importance = pd.concat(importance_outputs, ignore_index=True)
    feature_coverage = pd.DataFrame(all_coverage)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    evaluation.to_csv(output_dir / "evaluation_summary.csv", index=False)
    predictions.to_csv(output_dir / "county_predictions.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    feature_coverage.to_csv(output_dir / "feature_coverage.csv", index=False)
    _save_plots(evaluation, predictions, output_dir)

    manifest = {
        "component": "county_relative_ppsf",
        "dataset": _json_safe(dataset.metadata),
        "evaluation": {
            "design": (
                "Repeated nested K-fold cross-validation by model group; "
                "High and Very High counties are pooled"
            ),
            "selection_metric": "Mean outer-fold MAE",
            "outer_repeats": config.outer_repeats,
            "maximum_outer_splits": config.max_outer_splits,
            "inner_splits": config.inner_splits,
            "baseline": "Training-fold median relative Median PPSF YoY",
            "models_compared": MODEL_NAMES,
        },
        "models": model_manifest,
        "config": _json_safe(asdict(config)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
