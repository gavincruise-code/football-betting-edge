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

# Understat league identifiers (used for dropdown + fixture feed matching)
# ─────────────────────────────────────────────────────────────────────────────
# Source A  football-data.co.uk (free CSVs, no API key required)
# Source B  API-Football Pro    (api-football.com, key in .env)
# ─────────────────────────────────────────────────────────────────────────────
UNDERSTAT_LEAGUES = {
    # ── England ───────────────────────────────────────────────── Source A ──
    "EPL":                  "EPL",
    "Premier League":       "EPL",
    "Championship":         "E1",
    "League 1":             "E2",
    "League 2":             "E3",
    "Conference":           "EC",
    # ── Scotland ─────────────────────────────────────────────── Source A ──
    "Scottish Premiership": "SC0",
    "Scottish Championship":"SC1",
    "Scottish League 1":    "SC2",
    "Scottish League 2":    "SC3",
    # ── Spain ────────────────────────────────────────────────── Source A ──
    "La_Liga":              "La_Liga",
    "La Liga":              "La_Liga",
    "Segunda Division":     "SP2",
    # ── Germany ──────────────────────────────────────────────── Source A ──
    "Bundesliga":           "Bundesliga",
    "Bundesliga 2":         "D2",
    # ── Italy ────────────────────────────────────────────────── Source A ──
    "Serie_A":              "Serie_A",
    "Serie A":              "Serie_A",
    "Serie B":              "I2",
    # ── France ───────────────────────────────────────────────── Source A ──
    "Ligue_1":              "Ligue_1",
    "Ligue 1":              "Ligue_1",
    "Ligue 2":              "F2",
    # ── Other Europe (football-data.co.uk) ───────────────────── Source A ──
    "Eredivisie":           "N1",
    "Netherlands":          "N1",
    "Belgium":              "B1",
    "Jupiler Pro League":   "B1",
    "Liga Portugal":        "P1",
    "Portugal":             "P1",
    "Super Lig":            "T1",
    "Turkey":               "T1",
    "Super League Greece":  "G1",
    "Greece":               "G1",
    # ── Americas ─────────────────────────────────────────────── Source B ──
    "USA (MLS)":            "USA",
    "Brazil":               "BRA",
    "Argentina":            "ARG",
    "Mexico":               "MEX",
    "Colombia":             "COL",
    "Chile":                "CHL",
    "Uruguay":              "URU",
    "Ecuador":              "ECU",
    "Peru":                 "PER",
    # ── Asia ─────────────────────────────────────────────────── Source B ──
    "Japan":                "JPN",
    "China":                "CHN",
    "South Korea":          "KOR",
    "India":                "ind.1",
    "Indian Super League":  "ind.1",
    "Saudi Arabia":         "SAU",
    "Thailand":             "THA",
    # ── Africa ───────────────────────────────────────────────── Source B ──
    "Egypt":                "EGY",
    "Morocco":              "MAR",
    "South Africa":         "RSA",
    # ── Other Europe (API-Football) ───────────────────────────── Source B ──
    "Sweden":               "SWE",
    "Norway":               "NOR",
    "Denmark":              "DNK",
    "Finland":              "FIN",
    "Poland":               "POL",
    "Romania":              "ROU",
    "Switzerland":          "SWZ",
    "Austria":              "AUT",
    "Croatia":              "HNL",
    "Russia":               "RUS",
    "Serbia":               "SRB",
    "Czech Republic":       "CZE",
    "Ukraine":              "UKR",
    "Slovakia":             "SVK",
    "Hungary":              "HUN",
    "Israel":               "ISR",
    # ── Oceania ──────────────────────────────────────────────── Source B ──
    "Australia":            "AUS",
}

# Mapping from league keys to football-data.co.uk league codes
# (only leagues actually available on football-data.co.uk)
UNDERSTAT_TO_FD_LEAGUE = {
    "EPL":              "E0",
    "Premier League":   "E0",
    "Championship":     "E1",
    "League 1":         "E2",
    "League 2":         "E3",
    "Conference":       "EC",
    "Scottish Premiership":  "SC0",
    "Scottish Championship": "SC1",
    "Scottish League 1":     "SC2",
    "Scottish League 2":     "SC3",
    "La_Liga":          "SP1",
    "La Liga":          "SP1",
    "Segunda Division": "SP2",
    "Bundesliga":       "D1",
    "Bundesliga 2":     "D2",
    "Serie_A":          "I1",
    "Serie A":          "I1",
    "Serie B":          "I2",
    "Ligue_1":          "F1",
    "Ligue 1":          "F1",
    "Ligue 2":          "F2",
    "Eredivisie":       "N1",
    "Netherlands":      "N1",
    "Belgium":          "B1",
    "Jupiler Pro League": "B1",
    "Liga Portugal":    "P1",
    "Portugal":         "P1",
    "Super Lig":        "T1",
    "Turkey":           "T1",
    "Super League Greece": "G1",
    "Greece":           "G1",
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
