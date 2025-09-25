from __future__ import annotations
import argparse, json, os
import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime
from src.data.loaders import load_schedules, load_scoring_lines, compute_prev_season_team_stats
from src.features.build_features import make_inference_frame
from src.models.train import FEATURE_COLS

def load_accuracy() -> float | None:
    try:
        with open("artifacts/metrics.json","r") as f:
            return float(json.load(f)["accuracy"])
    except Exception:
        return None

def main(season: int, week: int):
    os.makedirs("outputs", exist_ok=True)

    # Load model & metrics
    model = load("artifacts/model_weekly_logreg.pkl")
    model_accuracy = load_accuracy()  # <-- you can print/report this

    # Build inference frame for target week
    schedules = load_schedules([season])
    week_df = schedules[(schedules["season"] == season) &
                        (schedules["game_type"] == "REG") &
                        (schedules["week"] == week)].copy()

    lines = load_scoring_lines([season])
    prev  = compute_prev_season_team_stats([season])

    # Pass the full season schedules to compute rolling features up to the target week
    season_sched = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")].copy()
    infer_df = make_inference_frame(season_sched, week_df, prev, lines)
    X = infer_df[FEATURE_COLS]

    # Predict home win probabilities
    p_home = model.predict_proba(X)[:, 1]
    infer_df["p_home"] = p_home
    infer_df["pred_winner"] = np.where(infer_df["p_home"] >= 0.5, infer_df["home_team"], infer_df["away_team"])
    infer_df["win_prob"]    = np.where(infer_df["p_home"] >= 0.5, infer_df["p_home"], 1.0 - infer_df["p_home"])

    # Pretty output
    out = (infer_df[["season","week","game_id","away_team","home_team","pred_winner","win_prob"]]
           .sort_values(["season","week","game_id"]))
    out["win_prob"] = (out["win_prob"] * 100).round(1)  # percent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"outputs/week_{season}_{week}_predictions_{timestamp}.csv"
    out.to_csv(csv_path, index=False)

    print(f"\n=== Week {week} {season} Predictions ===")
    for _, r in out.iterrows():
        print(f"{r['away_team']} @ {r['home_team']}  ->  {r['pred_winner']} ({r['win_prob']}%)")
    if model_accuracy is not None:
        print(f"\nMODEL_ACCURACY (held-out seasons): {model_accuracy:.3f}")
    else:
        print("\nMODEL_ACCURACY not available yet (train first).")
    print(f"\nSaved predictions to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)  # adjust as needed
    parser.add_argument("--week",   type=int, default=1)
    args = parser.parse_args()
    main(args.season, args.week)
