"""Configuration settings for baseball-stats."""

import os

# Seasons to use for training data
SEASONS = [2023, 2024, 2025]

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# XGBoost hyperparameters per model
XGBOOST_PARAMS = {
    "total_runs": {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "game_winner": {
        "objective": "binary:logistic",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "home_runs": {
        "objective": "count:poisson",
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "strikeouts": {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "hits": {
        "objective": "count:poisson",
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
}

# Park factors – runs (1.0 = neutral)
PARK_FACTORS = {
    "ARI": {"runs": 1.010, "hr": 1.020, "hits": 1.008},
    "ATL": {"runs": 1.013, "hr": 1.015, "hits": 1.010},
    "BAL": {"runs": 1.005, "hr": 1.010, "hits": 1.003},
    "BOS": {"runs": 1.018, "hr": 1.012, "hits": 1.020},
    "CHC": {"runs": 1.002, "hr": 1.005, "hits": 1.000},
    "CHW": {"runs": 0.998, "hr": 1.008, "hits": 0.997},
    "CIN": {"runs": 1.017, "hr": 1.025, "hits": 1.012},
    "CLE": {"runs": 1.008, "hr": 1.000, "hits": 1.005},
    "COL": {"runs": 1.031, "hr": 1.040, "hits": 1.030},
    "DET": {"runs": 1.001, "hr": 0.995, "hits": 1.002},
    "HOU": {"runs": 1.015, "hr": 1.010, "hits": 1.012},
    "KCR": {"runs": 1.011, "hr": 1.005, "hits": 1.008},
    "LAA": {"runs": 1.000, "hr": 1.002, "hits": 0.999},
    "LAD": {"runs": 1.010, "hr": 1.008, "hits": 1.007},
    "MIA": {"runs": 1.006, "hr": 0.998, "hits": 1.003},
    "MIL": {"runs": 1.012, "hr": 1.015, "hits": 1.008},
    "MIN": {"runs": 1.009, "hr": 1.012, "hits": 1.005},
    "NYM": {"runs": 1.014, "hr": 1.010, "hits": 1.012},
    "NYY": {"runs": 1.020, "hr": 1.030, "hits": 1.015},
    "OAK": {"runs": 1.005, "hr": 0.998, "hits": 1.002},
    "PHI": {"runs": 1.020, "hr": 1.025, "hits": 1.015},
    "PIT": {"runs": 1.007, "hr": 1.000, "hits": 1.005},
    "SDP": {"runs": 1.003, "hr": 0.998, "hits": 1.001},
    "SEA": {"runs": 1.016, "hr": 1.005, "hits": 1.010},
    "SFG": {"runs": 1.019, "hr": 0.988, "hits": 1.015},
    "STL": {"runs": 1.018, "hr": 1.010, "hits": 1.015},
    "TBR": {"runs": 1.010, "hr": 1.005, "hits": 1.008},
    "TEX": {"runs": 1.024, "hr": 1.030, "hits": 1.018},
    "TOR": {"runs": 1.022, "hr": 1.018, "hits": 1.015},
    "WSN": {"runs": 1.017, "hr": 1.020, "hits": 1.010},
}

# MLB team abbreviation map (statsapi name -> pybaseball abbrev)
TEAM_ABBREV_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
    "Athletics": "OAK",
}