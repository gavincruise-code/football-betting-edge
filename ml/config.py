"""
ML Pipeline Configuration Constants
====================================
Central configuration for the ML betting pipeline.
All tuneable parameters are defined here.
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------

# Understat league identifiers
UNDERSTAT_LEAGUES = {
    "EPL": "EPL",
    "La_Liga": "La_Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie_A",
    "Ligue_1": "Ligue_1",
    "USA (MLS)": "USA",
    "Argentina": "ARG",
    "Brazil": "BRA",
    "Mexico": "MEX",
    "Japan": "JPN",
    "China": "CHN",
    "Sweden": "SWE",
    "Norway": "NOR",
    "Denmark": "DNK",
    "Finland": "FIN",
    "Poland": "POL",
    "Romania": "ROU",
    "Switzerland": "SWZ",
    "Austria": "AUT",
    "Calcutta Premier Division": "ind.2",
    "Indian Super League": "ind.1",
    "India": "ind.1",
    "India (Calcutta Premier Division)": "ind.2",
    "India (Super League)": "ind.1",
}

# Mapping from league keys to football-data.co.uk league codes
UNDERSTAT_TO_FD_LEAGUE = {
    "EPL": "E0",
    "La_Liga": "SP1",
    "La_liga": "SP1",
    "Bundesliga": "D1",
    "Serie_A": "I1",
    "Ligue_1": "F1",
    "USA (MLS)": "USA",
    "Argentina": "ARG",
    "Brazil": "BRA",
    "Mexico": "MEX",
    "Japan": "JPN",
    "China": "CHN",
    "Sweden": "SWE",
    "Norway": "NOR",
    "Denmark": "DNK",
    "Finland": "FIN",
    "Poland": "POL",
    "Romania": "ROU",
    "Switzerland": "SWZ",
    "Austria": "AUT",
}

# Seasons to use (last 5 seasons as per user decision)
TRAINING_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
TRAINING_SEASON_CODES = ["1920", "2021", "2122", "2223", "2324"]

# Understat uses calendar years for season IDs
UNDERSTAT_SEASON_IDS = [2019, 2020, 2021, 2022, 2023]

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

# Rolling window sizes
ROLLING_WINDOWS = [10, 20, 38]

# Minimum matches required before making predictions
MIN_HISTORY_MATCHES = 10

# Congestion threshold (days)
CONGESTION_THRESHOLD_DAYS = 4

# League positions for contextual flags
RELEGATION_ZONE_SIZE = 5
TOP_POSITIONS = 6

# ---------------------------------------------------------------------------
# Dixon-Coles Model
# ---------------------------------------------------------------------------

# Time-decay parameter (ξ)
DIXON_COLES_XI = 0.005

# Maximum goals in score matrix
MAX_GOALS_MATRIX = 8

# Refit frequency
DIXON_COLES_REFIT_FREQUENCY = "monthly"  # 'weekly', 'monthly', 'per_matchweek'

# ---------------------------------------------------------------------------
# XGBoost Model
# ---------------------------------------------------------------------------

# Optuna hyperparameter search
OPTUNA_N_TRIALS = 50

# Hyperparameter search space
XGBOOST_PARAM_SPACE = {
    "max_depth": (3, 8),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (100, 1000),
    "min_child_weight": (1, 10),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
}

# Probability bounds for betting
MIN_MODEL_PROBABILITY = 0.15
MAX_MODEL_PROBABILITY = 0.85

# Odds bounds for betting
MIN_ODDS = 1.40
MAX_ODDS = 4.00

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

# Walk-forward settings
MIN_TRAIN_SEASONS = 3
PURGE_DAYS = 14

# Default edge margin
DEFAULT_EDGE_MARGIN = 0.05

# Default stake (GBP)
DEFAULT_STAKE = 10.0

# Kelly Criterion
KELLY_FRACTION = 0.25  # Quarter-Kelly
KELLY_MAX_PCT = 0.05   # Cap at 5% of bankroll

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

CACHE_DIR = "cache"
MODELS_DIR = "models"
