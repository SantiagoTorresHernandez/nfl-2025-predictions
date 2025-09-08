from __future__ import annotations
import pandas as pd
import numpy as np
import nfl_data_py as nfl  # pip install nfl-data-py

def load_schedules(years: list[int]) -> pd.DataFrame:
    """
    Returns schedule data for given seasons (REG + POST + PRE).
    """
    df = nfl.import_schedules(years)  # schedule info incl. teams, scores, week, game_id
    return df

def load_scoring_lines(years: list[int]) -> pd.DataFrame:
    """
    Returns scoring lines (spreads/totals). Often joined on game_id.
    """
    lines = nfl.import_sc_lines(years)
    return lines

def compute_prev_season_team_stats(years: list[int]) -> pd.DataFrame:
    """
    Build previous-season aggregates per team:
      - prev_win_pct
      - prev_point_diff_per_game
    For season S rows, we compute stats from season S-1.
    """
    # We need the previous season too:
    all_years = sorted(set(years + [min(years) - 1]))
    sched = load_schedules(all_years)
    sched = sched[sched["game_type"] == "REG"].copy()

    home = sched[["season","home_team","home_score","away_score"]].copy()
    home["team"] = home["home_team"]
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]
    home["win"] = (home["home_score"] > home["away_score"]).astype(int)
    home = home[["season","team","points_for","points_against","win"]]

    away = sched[["season","away_team","home_score","away_score"]].copy()
    away["team"] = away["away_team"]
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]
    away["win"] = (away["away_score"] > away["home_score"]).astype(int)
    away = away[["season","team","points_for","points_against","win"]]

    team_season = pd.concat([home, away], ignore_index=True)
    agg = (team_season
           .groupby(["season","team"], as_index=False)
           .agg(games=("win","size"),
                wins=("win","sum"),
                pf=("points_for","sum"),
                pa=("points_against","sum")))

    agg["win_pct"] = np.where(agg["games"] > 0, agg["wins"] / agg["games"], np.nan)
    agg["pdpg"] = np.where(agg["games"] > 0, (agg["pf"] - agg["pa"]) / agg["games"], np.nan)

    # Shift to represent "previous season" stats for next season S
    agg["season"] = agg["season"] + 1
    agg = agg.rename(columns={"win_pct": "prev_win_pct", "pdpg": "prev_pdpg"})
    return agg[["season","team","prev_win_pct","prev_pdpg"]]


def make_training_frame(
    schedules: pd.DataFrame,
    prev_team_stats: pd.DataFrame,
    scoring_lines: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the supervised training frame.

    Expected inputs:
      - schedules: output of nfl.import_schedules(years)
      - prev_team_stats: output of compute_prev_season_team_stats(years)
      - scoring_lines: output of nfl.import_sc_lines(years)

    Returns a DataFrame with label `home_win` and required feature columns:
      spread_line, home_favorite, home_prev_win_pct, home_prev_pdpg,
      away_prev_win_pct, away_prev_pdpg
    """
    df = schedules.copy()
    # Keep regular season games with final scores
    if "game_type" in df.columns:
        df = df[df["game_type"] == "REG"].copy()

    # Label
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # Join previous-season team features (shifted already in prev_team_stats)
    home_prev = prev_team_stats.rename(columns={
        "team": "home_team",
        "prev_win_pct": "home_prev_win_pct",
        "prev_pdpg": "home_prev_pdpg",
    })
    away_prev = prev_team_stats.rename(columns={
        "team": "away_team",
        "prev_win_pct": "away_prev_win_pct",
        "prev_pdpg": "away_prev_pdpg",
    })

    df = df.merge(
        home_prev[["season","home_team","home_prev_win_pct","home_prev_pdpg"]],
        on=["season","home_team"],
        how="left",
    )
    df = df.merge(
        away_prev[["season","away_team","away_prev_win_pct","away_prev_pdpg"]],
        on=["season","away_team"],
        how="left",
    )

    # Join scoring lines on game_id and handle multiple possible spread column names
    spread_candidates = [
        "spread_line",
        "spread",
        "spread_close",
        "spread_open",
        "spread_line_close",
    ]
    chosen_spread_col = next((c for c in spread_candidates if c in scoring_lines.columns), None) if hasattr(scoring_lines, "columns") else None

    if "game_id" in df.columns and hasattr(scoring_lines, "columns") and "game_id" in scoring_lines.columns:
        # Normalize key types before merge
        try:
            df["game_id"] = df["game_id"].astype(str)
        except Exception:
            pass
        try:
            scoring_lines = scoring_lines.copy()
            scoring_lines["game_id"] = scoring_lines["game_id"].astype(str)
        except Exception:
            pass
        merge_cols = ["game_id"]
        if chosen_spread_col is not None:
            merge_cols.append(chosen_spread_col)
        if "team_favorite_id" in scoring_lines.columns:
            merge_cols.append("team_favorite_id")
        lines = scoring_lines[merge_cols].copy()
        if chosen_spread_col is not None and chosen_spread_col != "spread_line":
            lines = lines.rename(columns={chosen_spread_col: "spread_line"})
        df = df.merge(lines, on="game_id", how="left")
    else:
        df = df.assign(spread_line=pd.NA)

    # Derive whether home team was favorite
    if "team_favorite_id" in df.columns:
        df["home_favorite"] = (df["team_favorite_id"] == df["home_team"]).astype(int)
    else:
        df["home_favorite"] = (pd.to_numeric(df["spread_line"], errors="coerce") < 0).astype(int)
    if "team_favorite_id" in df.columns:
        df = df.drop(columns=["team_favorite_id"])  # not needed downstream

    # Select final columns and ensure no NaNs in features
    feature_cols = [
        "spread_line",
        "home_favorite",
        "home_prev_win_pct",
        "home_prev_pdpg",
        "away_prev_win_pct",
        "away_prev_pdpg",
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df[feature_cols] = df[feature_cols].astype(float).fillna(0.0)

    cols_to_return = ["season","week","game_id","home_team","away_team","home_win"] + feature_cols
    existing_cols = [c for c in cols_to_return if c in df.columns]
    return df[existing_cols].copy()


def make_inference_frame(
    week_schedule: pd.DataFrame,
    prev_team_stats: pd.DataFrame,
    scoring_lines: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the feature frame for a given week's schedule. Same feature schema as training.
    """
    df = week_schedule.copy()
    # Reuse the same logic as training, but do not compute labels
    home_prev = prev_team_stats.rename(columns={
        "team": "home_team",
        "prev_win_pct": "home_prev_win_pct",
        "prev_pdpg": "home_prev_pdpg",
    })
    away_prev = prev_team_stats.rename(columns={
        "team": "away_team",
        "prev_win_pct": "away_prev_win_pct",
        "prev_pdpg": "away_prev_pdpg",
    })

    df = df.merge(
        home_prev[["season","home_team","home_prev_win_pct","home_prev_pdpg"]],
        on=["season","home_team"],
        how="left",
    )
    df = df.merge(
        away_prev[["season","away_team","away_prev_win_pct","away_prev_pdpg"]],
        on=["season","away_team"],
        how="left",
    )

    # Spread merge (normalize keys and column names)
    spread_candidates = ["spread_line","spread","spread_close","spread_open","spread_line_close"]
    chosen_spread_col = next((c for c in spread_candidates if c in scoring_lines.columns), None) if hasattr(scoring_lines, "columns") else None

    if "game_id" in df.columns and hasattr(scoring_lines, "columns") and "game_id" in scoring_lines.columns:
        try:
            df["game_id"] = df["game_id"].astype(str)
            scoring_lines = scoring_lines.copy()
            scoring_lines["game_id"] = scoring_lines["game_id"].astype(str)
        except Exception:
            pass
        merge_cols = ["game_id"]
        if chosen_spread_col is not None:
            merge_cols.append(chosen_spread_col)
        if "team_favorite_id" in scoring_lines.columns:
            merge_cols.append("team_favorite_id")
        lines = scoring_lines[merge_cols].copy()
        if chosen_spread_col is not None and chosen_spread_col != "spread_line":
            lines = lines.rename(columns={chosen_spread_col: "spread_line"})
        df = df.merge(lines, on="game_id", how="left")
    else:
        df = df.assign(spread_line=pd.NA)

    if "team_favorite_id" in df.columns:
        df["home_favorite"] = (df["team_favorite_id"] == df["home_team"]).astype(int)
        df = df.drop(columns=["team_favorite_id"])
    else:
        df["home_favorite"] = (pd.to_numeric(df["spread_line"], errors="coerce") < 0).astype(int)

    feature_cols = [
        "spread_line",
        "home_favorite",
        "home_prev_win_pct",
        "home_prev_pdpg",
        "away_prev_win_pct",
        "away_prev_pdpg",
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feature_cols] = df[feature_cols].astype(float).fillna(0.0)

    cols_to_return = ["season","week","game_id","home_team","away_team"] + feature_cols
    existing_cols = [c for c in cols_to_return if c in df.columns]
    return df[existing_cols].copy()