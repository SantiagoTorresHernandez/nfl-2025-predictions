from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    # Vegas
    "spread_line",
    "home_favorite",
    # Previous season
    "home_prev_win_pct",
    "home_prev_pdpg",
    "away_prev_win_pct",
    "away_prev_pdpg",
    # Recent form
    "home_recent_win_pct_3",
    "home_recent_pdpg_3",
    "away_recent_win_pct_3",
    "away_recent_pdpg_3",
    # Rest and differential
    "home_days_rest",
    "away_days_rest",
    "rest_diff",
    # Rivalry / conference
    "divisional_game",
    "conference_game",
    # Travel
    "away_travel_km",
]

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
