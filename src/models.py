from __future__ import annotations

import pandas as pd


# Signal rule: long if P(next-day up) > PROBABILITY_THRESHOLD, else flat.
PROBABILITY_THRESHOLD = 0.6
PCA_VARIANCE_THRESHOLD = 0.80


def generate_ml_signals(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Return a copy of df with the ML signal columns added.

    Input df is the output of features.add_ml_features(). The column names
    below are load-bearing: the backtester spec and the paper-trade executor
    read them exactly as written.

    Required output columns:
        ml_probability   float  model probability of a next-day up move
        ml_position      int    1 = long, 0 = flat (keep 0 on warmup/training rows)
        ml_trade_signal  int    1 on entry bar, -1 on exit bar, else 0
        ml_buy_signal    bool   ml_trade_signal == 1
        ml_sell_signal   bool   ml_trade_signal == -1

    Required attrs on the returned frame (used by the PCA chart):
        signals.attrs["pca_explained_variance_ratio"]  full per-component array
        signals.attrs["pca_n_components"]              components kept at threshold

    TODO:
        1. standardize FEATURE_COLUMNS with StandardScaler
        2. fit PCA; keep components until cumulative explained variance
           >= PCA_VARIANCE_THRESHOLD
        3. train one classifier (RF / LogReg / GB / SVM / MLP) on the binary
           target next-day return > 0, using the PCA components as inputs
        4. derive the columns above from the predicted probabilities
    """
    raise NotImplementedError("generate_ml_signals is not implemented yet.")
