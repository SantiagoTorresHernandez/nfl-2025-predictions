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
