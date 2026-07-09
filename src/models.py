from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from src.features import PCAFeatureResult


# Assignment probability threshold:
# Long if model probability > 0.60
# Flat if model probability <= 0.60
PROBABILITY_THRESHOLD = 0.60

ML_POSITION_COL = "ml_position"
ML_TRADE_SIGNAL_COL = "ml_trade_signal"
ML_BUY_SIGNAL_COL = "ml_buy_signal"
ML_SELL_SIGNAL_COL = "ml_sell_signal"


@dataclass(frozen=True)
class MLSignalResult:
    """Container for the trained classifier and its signal DataFrame."""

    signal_df: pd.DataFrame
    pca_result: PCAFeatureResult
    model: BaseEstimator
    component_columns: list[str]
    probability_threshold: float
    trade_on_test_only: bool


def train_classifier(
    pca_result: PCAFeatureResult,
    classifier: BaseEstimator | None = None,
) -> BaseEstimator:
    """
    Train the assignment classifier on PCA components.

    This module starts after feature engineering/PCA. It does not build
    features, scale raw inputs, or fit PCA.
    """

    if pca_result.y_train.nunique() < 2:
        raise ValueError(
            "Training target has only one class. Use more data or a different date range/ticker."
        )

    model = classifier or LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    model.fit(pca_result.X_train_pca, pca_result.y_train.astype(int))
    return model


def _positive_class_probability(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """
    Return the predicted probability of class 1.

    Class 1 means:
        next-day return > 0
    """

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The classifier must support predict_proba because the assignment uses "
            "a 0.60 probability threshold."
        )

    probabilities = model.predict_proba(X)

    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError("The trained classifier does not contain class 1 in model.classes_.")

    positive_class_position = classes.index(1)
    return probabilities[:, positive_class_position]


def _build_signal_frame(
    pca_result: PCAFeatureResult,
    probability_threshold: float,
    model: BaseEstimator,
    trade_on_test_only: bool,
) -> pd.DataFrame:
    latest_unlabeled_index = pca_result.data.index[pca_result.data["target"].isna()]
    if trade_on_test_only:
        eligible_index = pca_result.test_index.union(latest_unlabeled_index)
    else:
        eligible_index = pca_result.data.index

    result = score_pca_features(
        pca_frame=pca_result.data,
        component_columns=pca_result.pca_columns,
        model=model,
        probability_threshold=probability_threshold,
        eligible_index=eligible_index,
    )
    result["ml_sample_type"] = "not_ready"
    result.loc[pca_result.train_index, "ml_sample_type"] = "train"
    result.loc[pca_result.test_index, "ml_sample_type"] = "test"
    result.loc[latest_unlabeled_index, "ml_sample_type"] = "latest_unlabeled"

    result.attrs["ml_feature_columns"] = pca_result.feature_columns
    result.attrs["ml_pca_columns"] = pca_result.pca_columns
    result.attrs["ml_pca_explained_variance_ratio"] = (
        pca_result.explained_variance_ratio.tolist()
    )
    result.attrs["ml_pca_cumulative_explained_variance"] = (
        pca_result.cumulative_explained_variance.tolist()
    )
    result.attrs["ml_model"] = model.__class__.__name__
    result.attrs["ml_probability_threshold"] = probability_threshold
    result.attrs["ml_trade_on_test_only"] = trade_on_test_only

    return result


def score_pca_features(
    pca_frame: pd.DataFrame,
    component_columns: list[str],
    model: BaseEstimator,
    probability_threshold: float = PROBABILITY_THRESHOLD,
    eligible_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Score any PCA-ready frame and append the standard ML signal columns."""

    if not 0 < probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0 and 1.")

    missing_components = [
        column for column in component_columns if column not in pca_frame.columns
    ]
    if missing_components:
        raise ValueError(f"Missing PCA component columns: {missing_components}")

    result = pca_frame.copy()
    probabilities = _positive_class_probability(model, result[component_columns])
    raw_signal = (probabilities > probability_threshold).astype(int)

    if eligible_index is None:
        eligible_index = result.index
    else:
        eligible_index = result.index.intersection(eligible_index)

    result["ml_probability"] = probabilities
    result["ml_predicted_target"] = (probabilities >= 0.50).astype(int)
    result["ml_raw_signal"] = raw_signal
    result["ml_signal"] = np.where(raw_signal == 1, "Long", "Flat")
    result[ML_POSITION_COL] = 0
    result.loc[eligible_index, ML_POSITION_COL] = (
        result.loc[eligible_index, "ml_raw_signal"].astype(int)
    )
    result[ML_POSITION_COL] = result[ML_POSITION_COL].fillna(0).astype(int)
    result[ML_TRADE_SIGNAL_COL] = (
        result[ML_POSITION_COL]
        .diff()
        .fillna(result[ML_POSITION_COL])
        .astype(int)
    )
    result[ML_BUY_SIGNAL_COL] = result[ML_TRADE_SIGNAL_COL] == 1
    result[ML_SELL_SIGNAL_COL] = result[ML_TRADE_SIGNAL_COL] == -1

    result["position"] = result[ML_POSITION_COL]
    result["trade_signal"] = result[ML_TRADE_SIGNAL_COL]
    result["buy_signal"] = result[ML_BUY_SIGNAL_COL]
    result["sell_signal"] = result[ML_SELL_SIGNAL_COL]

    return result


def run_ml_signal_pipeline(
    pca_result: PCAFeatureResult,
    probability_threshold: float = PROBABILITY_THRESHOLD,
    classifier: BaseEstimator | None = None,
    trade_on_test_only: bool = True,
) -> MLSignalResult:
    """
    Train Logistic Regression on PCA components and generate long/flat signals.

    Input must come from src.features.build_feature_pca_pipeline() or another
    object with the same PCAFeatureResult contract.
    """

    if not isinstance(pca_result, PCAFeatureResult):
        raise TypeError(
            "run_ml_signal_pipeline expects a PCAFeatureResult. Build features/PCA in "
            "src.features before calling src.models."
        )
    if not 0 < probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0 and 1.")

    model = train_classifier(pca_result, classifier=classifier)
    signal_df = _build_signal_frame(
        pca_result=pca_result,
        probability_threshold=probability_threshold,
        model=model,
        trade_on_test_only=trade_on_test_only,
    )

    return MLSignalResult(
        signal_df=signal_df,
        pca_result=pca_result,
        model=model,
        component_columns=pca_result.pca_columns.copy(),
        probability_threshold=probability_threshold,
        trade_on_test_only=trade_on_test_only,
    )


def generate_ml_signals(
    pca_result: PCAFeatureResult,
    probability_threshold: float = PROBABILITY_THRESHOLD,
    classifier: BaseEstimator | None = None,
    trade_on_test_only: bool = True,
) -> pd.DataFrame:
    """Compatibility wrapper that returns only the generated signal DataFrame."""

    return run_ml_signal_pipeline(
        pca_result=pca_result,
        probability_threshold=probability_threshold,
        classifier=classifier,
        trade_on_test_only=trade_on_test_only,
    ).signal_df


def predict_latest_signal(
    model_result: MLSignalResult,
    pca_frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Return the latest model-generated long/flat signal as a small dict."""

    signal_df = model_result.signal_df
    if pca_frame is not None:
        signal_df = score_pca_features(
            pca_frame=pca_frame,
            component_columns=model_result.component_columns,
            model=model_result.model,
            probability_threshold=model_result.probability_threshold,
        )

    ready = signal_df.dropna(subset=["ml_probability", ML_POSITION_COL])
    if ready.empty:
        raise ValueError("ML signal result does not contain a usable latest signal row.")

    latest = ready.iloc[-1]
    timestamp = latest["timestamp"] if "timestamp" in ready.columns else None
    trade_signal = (
        int(latest[ML_TRADE_SIGNAL_COL])
        if pd.notna(latest.get(ML_TRADE_SIGNAL_COL))
        else None
    )

    return {
        "timestamp": timestamp,
        "close": float(latest["close"]) if pd.notna(latest["close"]) else None,
        "probability": float(latest["ml_probability"]),
        "position": int(latest[ML_POSITION_COL]),
        "trade_signal": trade_signal,
        "label": "LONG" if int(latest[ML_POSITION_COL]) == 1 else "FLAT",
    }
