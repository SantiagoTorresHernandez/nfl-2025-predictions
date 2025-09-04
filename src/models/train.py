from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

FEATURE_COLS = ["spread_line","home_favorite","home_prev_win_pct","home_prev_pdpg","away_prev_win_pct","away_prev_pdpg"]

def build_pipeline(random_state: int = 42) -> Pipeline:
    num_cols = FEATURE_COLS
    prep = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)],
        remainder="drop"
    )
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    pipe = Pipeline(steps=[("prep", prep), ("clf", clf)])
    return pipe

def split_train_val(df: pd.DataFrame, val_years: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: reserve the last `val_years` seasons as validation.
    """
    last_seasons = sorted(df["season"].unique())[-val_years:]
    train_df = df[~df["season"].isin(last_seasons)].copy()
    val_df   = df[df["season"].isin(last_seasons)].copy()
    return train_df, val_df
