# Configuration Settings for Baseball Stats

## Seasons
SEASONS = ['2021', '2022', '2023', '2024', '2025', '2026']

## Directories
CACHE_DIR = './cache/'
MODELS_DIR = './models/'
OUTPUT_DIR = './output/'

## XGBoost Parameters for Models
XGBOOST_PARAMS = {
    'total_runs': {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100
    },
    'game_winner': {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100
    },
    'home_runs': {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100
    },
    'strikeouts': {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100
    },
    'hits': {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100
    }
}

## Park Factors for MLB Teams
PARK_FACTORS = {
    'AAR': 1.005,
    'ATL': 1.013,
    'BOS': 1.018,
    'CHC': 1.002,
    'CIN': 1.017,
    'CLE': 1.008,
    'COL': 1.031,
    'DET': 1.001,
    'HOU': 1.015,
    'KCR': 1.011,
    'LAD': 1.010,
    'MIA': 1.006,
    'MIL': 1.012,
    'MIN': 1.009,
    'NYY': 1.020,
    'NYM': 1.014,
    'OAK': 1.005,
    'PHI': 1.020,
    'PIT': 1.007,
    'SDP': 1.003,
    'SFG': 1.019,
    'SEA': 1.016,
    'STL': 1.018,
    'TBR': 1.010,
    'TEX': 1.024,
    'TOR': 1.022,
    'WSN': 1.017
}