What this project does (high level)
Predicts NFL game winners each week using a simple supervised model (logistic regression).
Trains on historical schedules, betting lines, and prior-season team strength summaries.
Saves a trained model and validation metrics, then produces weekly predictions as a CSV.

Repository layout and each file’s role
requirements.txt
Lists dependencies: nfl-data-py, pandas, numpy, scikit-learn, pyarrow. Install to reproduce the environment.
artifacts/
model_weekly_logreg.pkl: Serialized scikit-learn pipeline saved with joblib after training.
metrics.json: Validation metrics from the last training (accuracy, log loss).
outputs/
week_2025_1_predictions.csv: Example prediction output for season 2025, week 1. Columns: season,week,game_id,away_team,home_team,pred_winner,win_prob.
src/config.py

Defines Config constants for runs:
RANDOM_STATE: seed for model reproducibility.
TRAIN_START_SEASON, TRAIN_END_SEASON: range of historical seasons to load.
VAL_YEARS: number of most recent seasons reserved for validation.
src/data/loaders.py

Data acquisition and simple aggregates using nfl_data_py:
load_schedules(years): pulls schedules (teams, scores, week, game_id, etc.) for given years.
load_scoring_lines(years): pulls betting line info (spreads/totals) for given years.
compute_prev_season_team_stats(years): computes prior-season team stats per team and shifts them to the following season:
prev_win_pct: previous-season win percentage.
prev_pdpg: previous-season point differential per game.

Output keys: season, team, prev_win_pct, prev_pdpg where season is shifted to “next season.”
src/features/build_features.py
Builds model-ready DataFrames.
make_training_frame(schedules, prev_team_stats, scoring_lines):

Filters to regular season (REG) games with final scores.
Label: home_win = 1 if home_score > away_score, else 0.
Joins previous-season team stats onto both home and away sides to form:
home_prev_win_pct, home_prev_pdpg, away_prev_win_pct, away_prev_pdpg.
Joins scoring lines on game_id. It flexibly picks a spread column (tries spread_line, spread, spread_close, etc.) and normalizes to spread_line. If absent, fills with NA/0.
Derives home_favorite:
Uses team_favorite_id if present; else falls back to spread_line sign (<0 means home is favored).
Ensures numeric feature columns and fills missing with 0.0.
Returns columns:
Keys: season, week, game_id, home_team, away_team
Label: home_win
Features: spread_line, home_favorite, home_prev_win_pct, home_prev_pdpg, away_prev_win_pct, away_prev_pdpg
make_inference_frame(week_schedule, prev_team_stats, scoring_lines):
Same feature-building logic as training, but without labels.
Returns the same feature columns plus identifiers.
Note: This file also duplicates the loader functions at the top; the scripts primarily import loaders from src.data.loaders.
src/models/train.py
FEATURE_COLS: canonical list of model features (must match frames built above).
build_pipeline(random_state):
Scikit-learn Pipeline with:
StandardScaler on numeric features.
LogisticRegression(max_iter=1000, random_state=...).
split_train_val(df, val_years):
Time-based split: reserves the last val_years seasons as validation.
src/evaluate/metrics.py
evaluate_binary(model, X, y):
Computes accuracy and log_loss using predicted probabilities and a 0.5 threshold.
scripts/train_model.py
CLI to train and evaluate the model end-to-end:
Loads schedules, scoring lines, and previous season team stats for the requested range.
Builds the training frame with make_training_frame(...).
Splits train/validation by the most recent VAL_YEARS seasons.
Builds pipeline via build_pipeline(...), fits on training data.
Evaluates on validation set with evaluate_binary(...).
Saves:
artifacts/model_weekly_logreg.pkl (trained pipeline).
artifacts/metrics.json (accuracy, log loss).
Prints a short summary.
scripts/predict_week.py
CLI to generate predictions for a specific season and week:
Loads the saved model from artifacts/model_weekly_logreg.pkl.
Optionally loads metrics.json and prints the held-out accuracy if available.
Pulls the season’s schedules, filters to the requested week (REG), and builds the inference frame via make_inference_frame(...).
Computes home win probabilities (predict_proba), derives:
p_home = probability the home team wins.
pred_winner = home_team if p_home >= 0.5, else away_team.
win_prob = the probability of the predicted winner (max of p_home and 1 - p_home).
Exports a sorted CSV to outputs/week_{season}_{week}_predictions.csv and prints a readable summary.
__pycache__/ folders
Python’s bytecode caches; auto-generated and safe to ignore.
Data and features used
Label
home_win (binary), from the final game scores.
Market-based input
spread_line (or closest matching column); negative values imply home favorite.
home_favorite (1 if home is favored).
Team-strength priors (previous season)
home_prev_win_pct, home_prev_pdpg
away_prev_win_pct, away_prev_pdpg
Missing or absent columns are safely handled and filled to keep the schema consistent.
How the whole project flows
Training flow
Load historical schedules and scoring lines for seasons TRAIN_START_SEASON…TRAIN_END_SEASON.
Compute previous-season team summaries and shift them to align with season S.
Build the supervised training frame with label and the six features.
Split by season: last VAL_YEARS seasons become validation; the earlier years are training.
Fit a scaling + logistic regression pipeline.
Evaluate on validation; save model and metrics to artifacts/.
Inference (weekly predictions)
Load the saved model (and metrics if available).
Pull the schedule for the chosen season, filter to a single regular-season week.
Build the inference feature frame with the same six features.
Score with predict_proba, choose winner by a 0.5 threshold, compute win_prob.
Save outputs/week_{season}_{week}_predictions.csv and print readable lines.
How to run
Install dependencies (PowerShell):
python -m pip install -r requirements.txt
Train a model (uses defaults from Config):
python scripts/train_model.py
Optional args:
--start 2010 --end 2024 --val_years 2
Predict a specific week (requires a trained model in artifacts/):
python scripts/predict_week.py --season 2025 --week 1
Notes and assumptions
Weekly predictions depend on nfl-data-py APIs; ensure internet access and that the selected spread column exists for your seasons.
If you haven’t trained yet, predict_week.py will still run if the model file exists, but it will show “accuracy not available” if metrics.json is missing.
I finished mapping the files and data flow. If you want, I can add a quick architecture diagram or extend features (e.g., include Elo, injuries, QB stats, rolling in-season metrics).
Clear roles for each file in src/ and scripts/, with artifacts/ for saved model/metrics and outputs/ for CSV predictions.
End-to-end flow: historical data → feature frame → time-based split → scaled logistic regression → saved model → weekly inference CSV.
