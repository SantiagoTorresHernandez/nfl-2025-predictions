from __future__ import annotations
import argparse, json, os
import pandas as pd
from joblib import dump
from src.config import Config
from src.data.loaders import load_schedules, load_scoring_lines, compute_prev_season_team_stats
from src.features.build_features import make_training_frame
from src.models.train import build_pipeline, split_train_val, FEATURE_COLS
from src.evaluate.metrics import evaluate_binary

def main(start: int, end: int, val_years: int):
    os.makedirs("artifacts", exist_ok=True)

    # Load training data
    schedules = load_schedules(list(range(start, end + 1)))
    lines     = load_scoring_lines(list(range(start, end + 1)))
    prev      = compute_prev_season_team_stats(list(range(start, end + 1)))

    df = make_training_frame(schedules, prev, lines)

    # Split
    train_df, val_df = split_train_val(df, val_years=val_years)

    X_train, y_train = train_df[FEATURE_COLS], train_df["home_win"]
    X_val,   y_val   = val_df[FEATURE_COLS],   val_df["home_win"]

    # Train
    pipe = build_pipeline(random_state=Config.RANDOM_STATE)
    pipe.fit(X_train, y_train)

    # Validate
    metrics = evaluate_binary(pipe, X_val, y_val)

    # Persist
    dump(pipe, "artifacts/model_weekly_logreg.pkl")
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to artifacts/model_weekly_logreg.pkl")
    print("Validation metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=Config.TRAIN_START_SEASON)
    parser.add_argument("--end",   type=int, default=Config.TRAIN_END_SEASON)
    parser.add_argument("--val_years", type=int, default=Config.VAL_YEARS)
    args = parser.parse_args()
    main(args.start, args.end, args.val_years)
