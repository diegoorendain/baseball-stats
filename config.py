"""Configuration settings for baseball-stats."""

import os

# Seasons to use for training data
SEASONS = [2023, 2024, 2025]

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Weather API (optional — graceful fallback if not set)
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

# Stadium coordinates for weather lookups (lat, lon)
STADIUM_COORDINATES: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667), "ATL": (33.8907, -84.4677),
    "BAL": (39.2839, -76.6217), "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553), "CHW": (41.8299, -87.6338),
    "CIN": (39.0975, -84.5069), "CLE": (41.4959, -81.6853),
    "COL": (39.7559, -104.9942), "DET": (42.3390, -83.0485),
    "HOU": (29.7572, -95.3555), "KCR": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197), "MIL": (43.0280, -87.9712),
    "MIN": (44.9818, -93.2775), "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262), "OAK": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665), "PIT": (40.4468, -80.0057),
    "SDP": (32.7076, -117.1570), "SEA": (47.5914, -122.3325),
    "SFG": (37.7786, -122.3893), "STL": (38.6226, -90.1928),
    "TBR": (27.7682, -82.6534), "TEX": (32.7512, -97.0832),
    "TOR": (43.6414, -79.3894), "WSN": (38.8730, -77.0074),
}

# Stadiums with a roof (retractable or fixed dome) — weather matters less
INDOOR_STADIUMS: set[str] = {"ARI", "HOU", "MIA", "MIL", "MIN", "SEA", "TBR", "TOR"}

# Value betting thresholds
MIN_EDGE_PCT: float = 0.03          # 3% minimum edge to recommend a bet
KELLY_FRACTION: float = 0.25        # Quarter-Kelly for bankroll management
HIGH_CONFIDENCE_EDGE: float = 0.07  # 7%+ edge → 🔥 HIGH
MED_CONFIDENCE_EDGE: float = 0.05   # 5%+ edge → ✅ MEDIUM

# Ensemble model weights
ENSEMBLE_WEIGHTS: dict[str, float] = {
    "xgboost": 0.50,
    "lightgbm": 0.30,
    "logistic": 0.20,
}

# Umpire K-rate adjustments (home plate umpire name → K rate delta)
# Positive = more Ks than average, negative = fewer
UMPIRE_K_ADJUSTMENTS: dict[str, float] = {
    "Angel Hernandez": -0.032,
    "CB Bucknor": -0.028,
    "Joe West": -0.015,
    "Pat Hoberg": 0.022,
    "Stu Scheurwater": 0.018,
}

# LightGBM hyperparameters per model
LIGHTGBM_PARAMS: dict[str, dict] = {
    "total_runs": {
        "objective": "regression",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    "game_winner": {
        "objective": "binary",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    "home_runs": {
        "objective": "poisson",
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    "strikeouts": {
        "objective": "regression",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    "hits": {
        "objective": "poisson",
        "max_depth": 4,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
}

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