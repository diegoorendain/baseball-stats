"""Main script: train models (if needed) and generate MLB predictions.

Usage:
    python predict_today.py                    # predict today's games
    python predict_today.py --retrain          # force re-train before predicting
    python predict_today.py --date 2026-03-27  # predict a specific date
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, SEASONS, TEAM_ABBREV_MAP
from data_loader import (
    load_all_batting_stats,
    load_all_pitching_stats,
    load_game_results,
    load_schedule_today,
    load_team_game_logs,
)
from features import (
    build_batter_features,
    build_game_features,
    build_pitcher_features,
    build_today_batter_features,
    build_today_game_features,
    build_today_pitcher_features,
)
from models import (
    GameWinnerModel,
    HitsModel,
    HomeRunsModel,
    StrikeoutsModel,
    TotalRunsModel,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

_GAME_FEATURE_COLS = [
    "runs_per_game_l10",
    "runs_per_game_l30",
    "ra_per_game_l10",
    "win_pct_l10",
    "win_pct_season",
    "run_diff_l10",
    "park_factor_runs",
    "sp_era_l10",
    "sp_whip_l5",
]

_BATTER_FEATURE_COLS = [
    "hr_per_game",
    "iso",
    "barrel_rate",
    "avg_exit_velo",
    "avg_launch_angle",
    "hard_hit_rate",
    "ops",
    "hits_per_game",
    "avg",
    "k_rate",
    "bb_rate",
]

_PITCHER_FEATURE_COLS = [
    "k_per_game",
    "k_per_9",
    "k_rate",
    "ip_per_start",
    "era",
    "whip",
    "whiff_rate",
    "chase_rate",
    "avg_fastball_velo",
    "pitcher_hand",
]


def _select_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return only the columns that exist in *df* (fill missing with 0)."""
    present = [c for c in cols if c in df.columns]
    result = df[present].copy()
    for c in cols:
        if c not in result.columns:
            result[c] = 0.0
    return result[cols].fillna(0.0)


def train_game_models(
    game_logs: dict[str, pd.DataFrame],
    pitching_stats: pd.DataFrame,
    total_runs_model: TotalRunsModel,
    game_winner_model: GameWinnerModel,
) -> None:
    """Train Total Runs and Game Winner models from historical game logs."""
    print("[TRAIN] Building game-level features …")
    features_df = build_game_features(game_logs, pitching_stats)
    if features_df.empty or len(features_df) < 50:
        print("[WARN] Not enough game data for training, using synthetic fallback.")
        features_df = _synthetic_game_features()

    X = _select_cols(features_df, _GAME_FEATURE_COLS)
    y_runs = features_df["total_runs"].fillna(8.0)
    if "runs_scored" in features_df.columns and "ra_per_game_l10" in features_df.columns:
        y_winner = (features_df["runs_scored"] > features_df["ra_per_game_l10"]).astype(int)
    else:
        y_winner = pd.Series([1] * len(features_df))
    print(f"[TRAIN] Training TotalRunsModel on {len(X)} samples …")
    total_runs_model.train(X, y_runs)
    total_runs_model.save()

    print(f"[TRAIN] Training GameWinnerModel on {len(X)} samples …")
    game_winner_model.train(X, y_winner)
    game_winner_model.save()


def train_batter_models(
    batting_stats: pd.DataFrame,
    hr_model: HomeRunsModel,
    hits_model: HitsModel,
) -> None:
    """Train Home Runs and Hits models from batting stats."""
    print("[TRAIN] Building batter features …")
    features_df = build_batter_features(batting_stats)
    if features_df.empty or len(features_df) < 20:
        print("[WARN] Not enough batter data, using synthetic fallback.")
        features_df = _synthetic_batter_features()

    X = _select_cols(features_df, _BATTER_FEATURE_COLS)
    y_hr = features_df.get("hr_per_game", pd.Series(0.04, index=features_df.index)).fillna(0.04)
    y_hits = features_df.get("hits_per_game", pd.Series(0.85, index=features_df.index)).fillna(0.85)

    print(f"[TRAIN] Training HomeRunsModel on {len(X)} samples …")
    hr_model.train(X, y_hr)
    hr_model.save()

    print(f"[TRAIN] Training HitsModel on {len(X)} samples …")
    hits_model.train(X, y_hits)
    hits_model.save()


def train_pitcher_models(
    pitching_stats: pd.DataFrame,
    strikeouts_model: StrikeoutsModel,
) -> None:
    """Train the Strikeouts model from pitching stats."""
    print("[TRAIN] Building pitcher features …")
    features_df = build_pitcher_features(pitching_stats)
    if features_df.empty or len(features_df) < 20:
        print("[WARN] Not enough pitcher data, using synthetic fallback.")
        features_df = _synthetic_pitcher_features()

    feats = _PITCHER_FEATURE_COLS[:]
    X_cols = [c for c in feats if c != "pitcher_hand"]
    X_cols.append("pitcher_hand")
    # Ensure pitcher_hand is numeric
    if "hand" in features_df.columns and "pitcher_hand" not in features_df.columns:
        features_df["pitcher_hand"] = (features_df["hand"] == "R").astype(float)

    X = _select_cols(features_df, _PITCHER_FEATURE_COLS)
    y_k = features_df.get("k_per_game", pd.Series(6.0, index=features_df.index)).fillna(6.0)

    print(f"[TRAIN] Training StrikeoutsModel on {len(X)} samples …")
    strikeouts_model.train(X, y_k)
    strikeouts_model.save()


# ---------------------------------------------------------------------------
# Synthetic fallback data (so training never hard-fails)
# ---------------------------------------------------------------------------

def _synthetic_game_features(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "runs_per_game_l10": rng.normal(4.5, 0.8, n),
            "runs_per_game_l30": rng.normal(4.5, 0.6, n),
            "ra_per_game_l10": rng.normal(4.5, 0.8, n),
            "win_pct_l10": rng.uniform(0.3, 0.7, n),
            "win_pct_season": rng.uniform(0.35, 0.65, n),
            "run_diff_l10": rng.normal(0, 1.2, n),
            "park_factor_runs": rng.uniform(0.98, 1.04, n),
            "sp_era_l10": rng.normal(4.0, 0.8, n),
            "sp_whip_l5": rng.normal(1.25, 0.2, n),
            "total_runs": rng.normal(9.0, 2.0, n).clip(2, 20),
            "runs_scored": rng.normal(4.5, 1.5, n).clip(0, 15),
        }
    )


def _synthetic_batter_features(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "hr_per_game": rng.exponential(0.04, n),
            "iso": rng.normal(0.15, 0.05, n).clip(0, 0.35),
            "barrel_rate": rng.beta(2, 20, n),
            "avg_exit_velo": rng.normal(88, 3, n),
            "avg_launch_angle": rng.normal(12, 8, n),
            "hard_hit_rate": rng.beta(5, 10, n),
            "ops": rng.normal(0.72, 0.10, n).clip(0.4, 1.1),
            "hits_per_game": rng.normal(0.85, 0.20, n).clip(0.2, 1.8),
            "avg": rng.normal(0.250, 0.030, n).clip(0.15, 0.35),
            "k_rate": rng.beta(5, 18, n),
            "bb_rate": rng.beta(3, 30, n),
        }
    )


def _synthetic_pitcher_features(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "k_per_game": rng.normal(6.0, 1.5, n).clip(1, 14),
            "k_per_9": rng.normal(8.5, 1.5, n).clip(3, 16),
            "k_rate": rng.beta(5, 18, n),
            "ip_per_start": rng.normal(5.5, 0.8, n).clip(3, 8),
            "era": rng.normal(4.0, 0.8, n).clip(1.5, 7),
            "whip": rng.normal(1.25, 0.2, n).clip(0.8, 1.8),
            "whiff_rate": rng.beta(5, 15, n),
            "chase_rate": rng.beta(6, 14, n),
            "avg_fastball_velo": rng.normal(93, 2, n).clip(85, 100),
            "pitcher_hand": rng.choice([0.0, 1.0], n),
        }
    )


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _get_team_batting_summary(
    batting_stats: pd.DataFrame, team_abbrev: str
) -> dict[str, Any]:
    """Compute aggregate batting stats for a team."""
    if batting_stats.empty:
        return {"k_rate": 0.22, "ops": 0.720}
    bs = batting_stats.copy()
    bs.columns = [c.lower().replace(" ", "_").replace("%", "_pct") for c in bs.columns]
    team_col = next((c for c in bs.columns if c in ("team", "tm")), None)
    if team_col is None:
        return {"k_rate": 0.22, "ops": 0.720}
    team_df = bs[bs[team_col].str.upper() == team_abbrev.upper()]
    if team_df.empty:
        return {"k_rate": 0.22, "ops": 0.720}
    pa = pd.to_numeric(team_df.get("pa", team_df.get("ab", pd.Series())), errors="coerce").fillna(1)
    ab = pd.to_numeric(team_df.get("ab", pa), errors="coerce").fillna(pa)
    hits = pd.to_numeric(team_df.get("h", pd.Series(dtype=float)), errors="coerce").fillna(0)
    bb = pd.to_numeric(team_df.get("bb", pd.Series(dtype=float)), errors="coerce").fillna(0)
    so = pd.to_numeric(team_df.get("so", team_df.get("k", pd.Series(dtype=float))), errors="coerce").fillna(0)
    doubles = pd.to_numeric(team_df.get("2b", pd.Series(dtype=float)), errors="coerce").fillna(0)
    triples = pd.to_numeric(team_df.get("3b", pd.Series(dtype=float)), errors="coerce").fillna(0)
    hr = pd.to_numeric(team_df.get("hr", pd.Series(dtype=float)), errors="coerce").fillna(0)

    total_pa = pa.sum()
    total_ab = ab.sum()
    slg = (hits.sum() + doubles.sum() + 2 * triples.sum() + 3 * hr.sum()) / max(total_ab, 1)
    obp = (hits.sum() + bb.sum()) / max(total_pa, 1)
    return {
        "k_rate": so.sum() / max(total_pa, 1),
        "ops": obp + slg,
    }


def _build_mock_lineup(
    team_name: str,
    batting_stats: pd.DataFrame,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Return top-N batters for a team as lineup dicts."""
    if batting_stats.empty:
        return []
    bs = batting_stats.copy()
    bs.columns = [c.lower().replace(" ", "_").replace("%", "_pct") for c in bs.columns]
    team_col = next((c for c in bs.columns if c in ("team", "tm")), None)
    name_col = next((c for c in bs.columns if c in ("name", "player_name")), None)
    if team_col is None or name_col is None:
        return []

    abbrev = _team_abbrev_from_name(team_name)
    team_df = bs[bs[team_col].str.upper() == abbrev.upper()]
    if team_df.empty:
        return []
    # Sort by OPS proxy
    pa = pd.to_numeric(team_df.get("pa", team_df.get("ab", pd.Series(dtype=float))), errors="coerce").fillna(1)
    team_df = team_df[pa >= 50]
    if team_df.empty:
        return []
    hr_col = team_df.get("hr", pd.Series(0, index=team_df.index))
    team_df = team_df.assign(_sort_hr=pd.to_numeric(hr_col, errors="coerce").fillna(0))
    team_df = team_df.sort_values("_sort_hr", ascending=False).head(n)

    lineup = []
    for _, row in team_df.iterrows():
        hand = str(row.get("bat", row.get("b", row.get("bats", "R"))) or "R")
        if hand not in ("L", "R", "S"):
            hand = "R"
        lineup.append(
            {
                "player_name": str(row[name_col]),
                "player_id": int(row.get("playerid", row.get("player_id", 0)) or 0),
                "hand": hand,
            }
        )
    return lineup


def _team_abbrev_from_name(name: str) -> str:
    return TEAM_ABBREV_MAP.get(name, name[:3].upper())


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_header(date_str: str) -> None:
    print()
    print("=" * 45)
    print(f"  MLB PREDICTIONS - {date_str}")
    print("=" * 45)


def _print_game(
    game: dict[str, Any],
    winner_pred: dict[str, Any],
    total_runs_pred: dict[str, Any],
    hr_preds: list[dict[str, Any]],
    k_preds: list[dict[str, Any]],
    hits_preds: list[dict[str, Any]],
) -> None:
    away = game.get("away_team", "Away")
    home = game.get("home_team", "Home")
    venue = game.get("venue", "")
    away_sp = game.get("away_sp", "TBD")
    home_sp = game.get("home_sp", "TBD")

    away_win_prob = winner_pred.get("away_win_prob", 0.5)
    home_win_prob = winner_pred.get("home_win_prob", 0.5)
    winner = home if home_win_prob >= away_win_prob else away
    win_prob = max(home_win_prob, away_win_prob) * 100

    total_runs = total_runs_pred.get("predicted_runs", 9.0)
    margin = total_runs_pred.get("margin", 2.0)

    print()
    print(f"🏟️  {away} vs {home}" + (f" ({venue})" if venue else ""))
    print(f"   SP: {away_sp} vs {home_sp}")
    print("   " + "─" * 28)
    print(f"   Winner:      {winner} ({win_prob:.1f}% confidence)")
    print(f"   Total Runs:  {total_runs:.1f} (±{margin:.1f})")
    print("   " + "─" * 28)

    if hr_preds:
        print("   HR Props:")
        for p in hr_preds:
            name = p.get("player_name", "Unknown")
            exp = p.get("expected_hr", 0.0)
            prob = p.get("prob_hr_ge_1", 0.0) * 100
            print(f"     {name[:18]:<18}  {exp:.1f} HR expected (P(HR≥1) = {prob:.1f}%)")

    print("   " + "─" * 28)

    if k_preds:
        print("   Strikeouts:")
        for p in k_preds:
            name = p.get("player_name", "Unknown")
            exp = p.get("expected_k", 0.0)
            print(f"     {name[:18]:<18}  {exp:.1f} K expected")

    print("   " + "─" * 28)

    if hits_preds:
        print("   Hits Props:")
        for p in hits_preds:
            name = p.get("player_name", "Unknown")
            exp = p.get("expected_hits", 0.0)
            prob = p.get("prob_hit_ge_1", 0.0) * 100
            print(f"     {name[:18]:<18}  {exp:.1f} hits expected (P(H≥1) = {prob:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB game predictions")
    parser.add_argument(
        "--retrain", action="store_true", help="Force re-training of all models"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to predict (YYYY-MM-DD). Defaults to today.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_date: datetime.date
    if args.date:
        target_date = datetime.date.fromisoformat(args.date)
    else:
        target_date = datetime.date.today()

    date_str = target_date.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # 1. Load / download historical data
    # ------------------------------------------------------------------
    print("[INFO] Loading historical batting stats …")
    batting_stats = load_all_batting_stats()

    print("[INFO] Loading historical pitching stats …")
    pitching_stats = load_all_pitching_stats()

    print("[INFO] Loading historical game logs for all teams …")
    game_logs: dict[str, pd.DataFrame] = {}
    unique_teams = list(dict.fromkeys(TEAM_ABBREV_MAP.values()))
    for team in unique_teams:
        for season in SEASONS:
            df = load_team_game_logs(team, season)
            if not df.empty:
                if team in game_logs:
                    game_logs[team] = pd.concat([game_logs[team], df], ignore_index=True)
                else:
                    game_logs[team] = df

    # ------------------------------------------------------------------
    # 2. Instantiate models
    # ------------------------------------------------------------------
    total_runs_model = TotalRunsModel()
    game_winner_model = GameWinnerModel()
    hr_model = HomeRunsModel()
    strikeouts_model = StrikeoutsModel()
    hits_model = HitsModel()

    models_all = [total_runs_model, game_winner_model, hr_model, strikeouts_model, hits_model]

    # ------------------------------------------------------------------
    # 3. Train or load models
    # ------------------------------------------------------------------
    if args.retrain or not all(m.is_saved() for m in models_all):
        print("[INFO] Training models …")
        train_game_models(game_logs, pitching_stats, total_runs_model, game_winner_model)
        train_batter_models(batting_stats, hr_model, hits_model)
        train_pitcher_models(pitching_stats, strikeouts_model)
    else:
        print("[INFO] Loading saved models …")
        for m in models_all:
            m.load()

    # ------------------------------------------------------------------
    # 4. Get today's schedule
    # ------------------------------------------------------------------
    print(f"[INFO] Fetching schedule for {date_str} …")
    today_games_raw = load_schedule_today(target_date)

    if not today_games_raw:
        print(f"[INFO] No games found for {date_str}.")
        return

    # ------------------------------------------------------------------
    # 5. Build today's game features
    # ------------------------------------------------------------------
    today_game_features = build_today_game_features(today_games_raw, game_logs, pitching_stats)

    # ------------------------------------------------------------------
    # 6. Generate predictions + collect output rows
    # ------------------------------------------------------------------
    _print_header(date_str)

    output_rows: list[dict[str, Any]] = []

    for i, game in enumerate(today_games_raw):
        away_name = game.get("away_team", "Away")
        home_name = game.get("home_team", "Home")
        away_abbrev = _team_abbrev_from_name(away_name)
        home_abbrev = _team_abbrev_from_name(home_name)

        away_sp_name = game.get("away_probable_pitcher", "TBD")
        home_sp_name = game.get("home_probable_pitcher", "TBD")

        park_hr = float(today_game_features["park_factor_hr"].iloc[i]) if i < len(today_game_features) else 1.0
        park_hits = float(today_game_features["park_factor_hits"].iloc[i]) if i < len(today_game_features) else 1.0

        # --- Game-level prediction ---
        if i < len(today_game_features):
            game_feat_row = today_game_features.iloc[[i]]
            game_X_cols = [
                c
                for c in _GAME_FEATURE_COLS
                if c in game_feat_row.columns
            ]
            # Use only model columns present; fill missing with defaults
            game_X = pd.DataFrame(
                {c: game_feat_row[c].values if c in game_feat_row.columns else [0.0] for c in _GAME_FEATURE_COLS}
            ).fillna(0.0)

            total_runs_preds = total_runs_model.predict_with_confidence(game_X)
            total_runs_pred = total_runs_preds[0] if total_runs_preds else {"predicted_runs": 9.0, "margin": 2.0}

            home_win_prob_arr = game_winner_model.predict_proba(game_X)
            home_win_prob = float(np.atleast_1d(home_win_prob_arr)[0])
            away_win_prob = 1.0 - home_win_prob
        else:
            total_runs_pred = {"predicted_runs": 9.0, "margin": 2.0}
            home_win_prob = 0.5
            away_win_prob = 0.5

        winner_pred = {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
        }

        # --- Batter props (home + away lineups combined) ---
        away_lineup = _build_mock_lineup(away_name, batting_stats)
        home_lineup = _build_mock_lineup(home_name, batting_stats)

        def _get_pitcher_summary(pitcher_name: str) -> dict[str, Any]:
            if pitching_stats.empty or pitcher_name in ("TBD", ""):
                return {"era": 4.50, "whip": 1.30, "k_per_9": 8.0, "hand": "R"}
            ps = pitching_stats.copy()
            ps.columns = [c.lower().replace(" ", "_").replace("%", "_pct") for c in ps.columns]
            name_col = next((c for c in ps.columns if c in ("name", "player_name")), None)
            if name_col is None:
                return {"era": 4.50, "whip": 1.30, "k_per_9": 8.0, "hand": "R"}
            last = pitcher_name.split()[-1] if pitcher_name else ""
            match = ps[ps[name_col].str.contains(last, case=False, na=False)]
            if match.empty:
                return {"era": 4.50, "whip": 1.30, "k_per_9": 8.0, "hand": "R"}
            p = match.iloc[0]
            ip = float(p.get("ip", 1) or 1)
            so = float(p.get("so", p.get("k", 0)) or 0)
            bb = float(p.get("bb", 0) or 0)
            h = float(p.get("h", 0) or 0)
            return {
                "era": float(p.get("era", 4.50) or 4.50),
                "whip": (h + bb) / max(ip, 1),
                "k_per_9": so / max(ip, 1) * 9,
                "hand": str(p.get("throws", p.get("hand", p.get("p", "R"))) or "R"),
            }

        home_pitcher_summary = _get_pitcher_summary(home_sp_name)
        away_pitcher_summary = _get_pitcher_summary(away_sp_name)

        # Away batters face home pitcher
        away_batter_feat = build_today_batter_features(
            away_lineup, batting_stats,
            opp_pitcher_stats=home_pitcher_summary,
            park_factor_hr=park_hr,
            park_factor_hits=park_hits,
        )
        # Home batters face away pitcher
        home_batter_feat = build_today_batter_features(
            home_lineup, batting_stats,
            opp_pitcher_stats=away_pitcher_summary,
            park_factor_hr=park_hr,
            park_factor_hits=park_hits,
        )
        combined_batter_feat = pd.concat([away_batter_feat, home_batter_feat], ignore_index=True)

        hr_preds_list: list[dict[str, Any]] = []
        hits_preds_list: list[dict[str, Any]] = []

        if not combined_batter_feat.empty:
            batter_X = _select_cols(combined_batter_feat, _BATTER_FEATURE_COLS)
            hr_results = hr_model.predict_with_confidence(batter_X)
            hits_results = hits_model.predict_with_confidence(batter_X)

            for j, row_feat in combined_batter_feat.iterrows():
                hr_r = hr_results[j] if j < len(hr_results) else {"expected_hr": 0.0, "prob_hr_ge_1": 0.0}
                hits_r = hits_results[j] if j < len(hits_results) else {"expected_hits": 0.0, "prob_hit_ge_1": 0.0}
                hr_preds_list.append({
                    "player_name": row_feat.get("player_name", ""),
                    **hr_r,
                })
                hits_preds_list.append({
                    "player_name": row_feat.get("player_name", ""),
                    **hits_r,
                })

        # Sort by expected HR descending
        hr_preds_list.sort(key=lambda x: x.get("expected_hr", 0), reverse=True)
        hits_preds_list.sort(key=lambda x: x.get("expected_hits", 0), reverse=True)

        # --- Pitcher strikeouts ---
        away_pitcher_dict = [{"player_name": away_sp_name, "hand": away_pitcher_summary.get("hand", "R")}] if away_sp_name != "TBD" else []
        home_pitcher_dict = [{"player_name": home_sp_name, "hand": home_pitcher_summary.get("hand", "R")}] if home_sp_name != "TBD" else []

        away_team_bat = _get_team_batting_summary(batting_stats, away_abbrev)
        home_team_bat = _get_team_batting_summary(batting_stats, home_abbrev)

        away_k_feat = build_today_pitcher_features(
            away_pitcher_dict, pitching_stats,
            opp_team_batting=home_team_bat,
        )
        home_k_feat = build_today_pitcher_features(
            home_pitcher_dict, pitching_stats,
            opp_team_batting=away_team_bat,
        )
        combined_pitcher_feat = pd.concat([away_k_feat, home_k_feat], ignore_index=True)

        k_preds_list: list[dict[str, Any]] = []
        if not combined_pitcher_feat.empty:
            pitcher_X = _select_cols(combined_pitcher_feat, _PITCHER_FEATURE_COLS)
            k_results = strikeouts_model.predict_expected(pitcher_X)
            for j, row_feat in combined_pitcher_feat.iterrows():
                k_preds_list.append({
                    "player_name": row_feat.get("player_name", ""),
                    "expected_k": k_results[j] if j < len(k_results) else 6.0,
                })

        # --- Print game block ---
        _print_game(
            {**game, "away_sp": away_sp_name, "home_sp": home_sp_name},
            winner_pred,
            total_runs_pred,
            hr_preds_list[:5],
            k_preds_list,
            hits_preds_list[:5],
        )

        # --- Accumulate output CSV rows ---
        winner_name = home_name if home_win_prob >= away_win_prob else away_name
        output_rows.append(
            {
                "date": date_str,
                "away_team": away_name,
                "home_team": home_name,
                "away_sp": away_sp_name,
                "home_sp": home_sp_name,
                "venue": game.get("venue", ""),
                "predicted_winner": winner_name,
                "winner_confidence_pct": round(max(home_win_prob, away_win_prob) * 100, 1),
                "predicted_total_runs": round(total_runs_pred["predicted_runs"], 1),
                "total_runs_margin": round(total_runs_pred["margin"], 1),
            }
        )
        for p in hr_preds_list:
            output_rows[-1][f'hr_{p["player_name"][:10]}_expected'] = round(p.get("expected_hr", 0.0), 2)
            output_rows[-1][f'hr_{p["player_name"][:10]}_prob'] = round(p.get("prob_hr_ge_1", 0.0) * 100, 1)

    # ------------------------------------------------------------------
    # 7. Save output CSV
    # ------------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, f"predicciones_{date_str}.csv")
    if output_rows:
        keys = sorted(set(k for row in output_rows for k in row.keys()))
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in output_rows:
                writer.writerow(row)
        print()
        print(f"Saved to: {out_path}")
    else:
        print("[INFO] No prediction rows to save.")


if __name__ == "__main__":
    main()
