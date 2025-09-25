from __future__ import annotations
import pandas as pd
import numpy as np
import nfl_data_py as nfl  # pip install nfl-data-py
from typing import Optional
from src.data.loaders import load_team_info

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

    Returns a DataFrame with label `home_win` and a richer set of features
    including team form, rest, divisional flags, and travel distance in addition
    to Vegas lines if available.
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

    # Enrich with robust non-Vegas features
    df = _augment_features_non_vegas(df)

    # Select final columns and ensure no NaNs in features
    feature_cols = _feature_cols()

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df[feature_cols] = df[feature_cols].astype(float).fillna(0.0)

    cols_to_return = ["season","week","game_id","home_team","away_team","home_win"] + feature_cols
    existing_cols = [c for c in cols_to_return if c in df.columns]
    return df[existing_cols].copy()


def make_inference_frame(
    season_schedules: pd.DataFrame,
    week_schedule: pd.DataFrame,
    prev_team_stats: pd.DataFrame,
    scoring_lines: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the feature frame for a given week's schedule. Uses the full season
    schedules to compute rolling and rest features up to the given week.
    """
    df = week_schedule.copy()
    # Reuse the same logic as training for previous-season features
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

    # Compute non-Vegas features using full season context up to this week
    season_aug = _augment_features_non_vegas(season_schedules)
    season_aug["game_id"] = season_aug["game_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    non_vegas_cols = [c for c in season_aug.columns if c not in ["home_win"]]
    df = df.merge(season_aug[["game_id"] + [c for c in non_vegas_cols if c not in df.columns]], on="game_id", how="left")

    feature_cols = _feature_cols()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[feature_cols] = df[feature_cols].astype(float).fillna(0.0)

    cols_to_return = ["season","week","game_id","home_team","away_team"] + feature_cols
    existing_cols = [c for c in cols_to_return if c in df.columns]
    return df[existing_cols].copy()


# -------------------------
# Feature engineering utils
# -------------------------

def _feature_cols() -> list[str]:
    return [
        # Vegas (optional)
        "spread_line",
        "home_favorite",
        # Previous season context
        "home_prev_win_pct",
        "home_prev_pdpg",
        "away_prev_win_pct",
        "away_prev_pdpg",
        # Recent form (this season)
        "home_recent_win_pct_3",
        "home_recent_pdpg_3",
        "away_recent_win_pct_3",
        "away_recent_pdpg_3",
        # Rest
        "home_days_rest",
        "away_days_rest",
        "rest_diff",
        # Rivalry / conference
        "divisional_game",
        "conference_game",
        # Travel burden
        "away_travel_km",
    ]


def _augment_features_non_vegas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add non-Vegas features to the schedules frame:
      - recent form (rolling last 3): win pct, PD per game
      - rest days for each team and rest differential
      - divisional and conference indicators
      - away team travel distance to home team (km)
    Does not assume label availability and will not leak future info (uses shift).
    """
    out = df.copy()

    # Choose and standardize a game datetime column
    date_col = _pick_date_col(out)
    out["_game_dt"] = _ensure_datetime(out[date_col]) if date_col is not None else pd.to_datetime(out.get("gameday", pd.NaT), errors="coerce")
    # fallback ordering by season/week if no dates
    if out["_game_dt"].isna().all():
        out["_game_dt"] = pd.to_datetime(out["season"].astype(str) + "-01-01", errors="coerce") + pd.to_timedelta(out["week"].fillna(0).astype(float) * 7, unit="D")

    # Build team game log for rolling stats and rest
    team_log = _build_team_game_log(out)

    # Rolling last 3 games (prior only)
    team_log = team_log.sort_values(["team","_game_dt"]).copy()
    team_log["prior_win_cnt_3"] = (team_log.groupby("team")["win"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).sum())).astype(float)
    team_log["prior_games_3"] = (team_log.groupby("team")["win"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).count())).astype(float)
    team_log["prior_win_pct_3"] = np.where(team_log["prior_games_3"] > 0, team_log["prior_win_cnt_3"] / team_log["prior_games_3"], 0.0)
    team_log["prior_pdpg_3"] = team_log.groupby("team")["margin"].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).astype(float).fillna(0.0)

    # Rest days: time since previous game
    team_log["days_since_last"] = team_log.groupby("team")["_game_dt"].apply(lambda s: s.diff().dt.days).fillna(14).clip(lower=0)

    # Map back to home/away
    home_map = team_log.rename(columns={
        "team": "home_team",
        "prior_win_pct_3": "home_recent_win_pct_3",
        "prior_pdpg_3": "home_recent_pdpg_3",
        "days_since_last": "home_days_rest",
    })[["game_id","home_team","home_recent_win_pct_3","home_recent_pdpg_3","home_days_rest"]]

    away_map = team_log.rename(columns={
        "team": "away_team",
        "prior_win_pct_3": "away_recent_win_pct_3",
        "prior_pdpg_3": "away_recent_pdpg_3",
        "days_since_last": "away_days_rest",
    })[["game_id","away_team","away_recent_win_pct_3","away_recent_pdpg_3","away_days_rest"]]

    out = out.merge(home_map, on=["game_id","home_team"], how="left")
    out = out.merge(away_map, on=["game_id","away_team"], how="left")

    out["rest_diff"] = (out["home_days_rest"].fillna(0).astype(float) - out["away_days_rest"].fillna(0).astype(float))

    # Divisional and conference indicators + travel
    teams = load_team_info()
    if len(teams) > 0 and "team" in teams.columns:
        t_home = teams.rename(columns={"team": "home_team", "conference": "home_conf", "division": "home_div", "latitude": "home_lat", "longitude": "home_lon"})
        t_away = teams.rename(columns={"team": "away_team", "conference": "away_conf", "division": "away_div", "latitude": "away_lat", "longitude": "away_lon"})
        out = out.merge(t_home[[c for c in ["home_team","home_conf","home_div","home_lat","home_lon"] if c in t_home.columns]], on="home_team", how="left")
        out = out.merge(t_away[[c for c in ["away_team","away_conf","away_div","away_lat","away_lon"] if c in t_away.columns]], on="away_team", how="left")
        out["divisional_game"] = ((out.get("home_div") == out.get("away_div")) & out.get("home_div").notna()).astype(int)
        out["conference_game"] = ((out.get("home_conf") == out.get("away_conf")) & out.get("home_conf").notna()).astype(int)
        out["away_travel_km"] = _haversine_km_series(out.get("away_lat"), out.get("away_lon"), out.get("home_lat"), out.get("home_lon")).fillna(0.0)
    else:
        out["divisional_game"] = 0
        out["conference_game"] = 0
        out["away_travel_km"] = 0.0

    # Ensure numeric types
    for col in [
        "home_recent_win_pct_3","home_recent_pdpg_3",
        "away_recent_win_pct_3","away_recent_pdpg_3",
        "home_days_rest","away_days_rest","rest_diff",
        "divisional_game","conference_game","away_travel_km",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def _pick_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "game_date", "gameday", "game_datetime", "start_time", "start_time_utc", "game_time"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.NaT)


def _build_team_game_log(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["season","week","game_id","_game_dt"] if c in df.columns]
    home = df[[*cols, "home_team","home_score","away_team","away_score"]].copy()
    home["team"] = home["home_team"]
    home["opp"] = home["away_team"]
    home["points_for"] = pd.to_numeric(home["home_score"], errors="coerce")
    home["points_against"] = pd.to_numeric(home["away_score"], errors="coerce")
    home["win"] = (home["points_for"] > home["points_against"]).astype(int)
    home = home[[*cols, "team","opp","points_for","points_against","win"]]

    away = df[[*cols, "home_team","home_score","away_team","away_score"]].copy()
    away["team"] = away["away_team"]
    away["opp"] = away["home_team"]
    away["points_for"] = pd.to_numeric(away["away_score"], errors="coerce")
    away["points_against"] = pd.to_numeric(away["home_score"], errors="coerce")
    away["win"] = (away["points_for"] > away["points_against"]).astype(int)
    away = away[[*cols, "team","opp","points_for","points_against","win"]]

    team_log = pd.concat([home, away], ignore_index=True)
    team_log["margin"] = (team_log["points_for"].fillna(0) - team_log["points_against"].fillna(0)).astype(float)
    return team_log


def _haversine_km_series(lat1, lon1, lat2, lon2) -> pd.Series:
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return pd.Series(0.0, index=lat1.index if hasattr(lat1, "index") else [])
    lat1 = pd.to_numeric(lat1, errors="coerce")
    lon1 = pd.to_numeric(lon1, errors="coerce")
    lat2 = pd.to_numeric(lat2, errors="coerce")
    lon2 = pd.to_numeric(lon2, errors="coerce")
    rlat1 = np.radians(lat1)
    rlon1 = np.radians(lon1)
    rlat2 = np.radians(lat2)
    rlon2 = np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat/2.0)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R_km = 6371.0
    return R_km * c