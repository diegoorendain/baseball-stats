"""Main script: train models (if needed) and generate MLB predictions.

Usage:
    python predict_today.py                    # predict today's games
    python predict_today.py --retrain          # force re-train before predicting
    python predict_today.py --date 2026-03-27  # predict a specific date
    python predict_today.py --backtest         # run historical backtester
    python predict_today.py --odds odds_today.json  # include value betting
    python predict_today.py --no-weather       # skip weather API calls
    python predict_today.py --no-ensemble     # use single XGBoost models
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

from config import OUTPUT_DIR, SEASONS, TEAM_ABBREV_MAP, UMPIRE_K_ADJUSTMENTS
from data_loader import (
    load_all_batting_stats,
    load_all_pitching_stats,
    load_game_results,
    load_schedule_today,
    load_team_game_logs,
    load_umpire_data,
    load_weather_data,
    load_team_schedule_context,
    load_bullpen_usage,
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
    EnsembleGameWinnerModel,
    EnsembleTotalRunsModel,
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
    print(f"  MLB SHARP PREDICTIONS - {date_str}")
    print("=" * 45)


def _print_game(
    game: dict[str, Any],
    winner_pred: dict[str, Any],
    total_runs_pred: dict[str, Any],
    hr_preds: list[dict[str, Any]],
    k_preds: list[dict[str, Any]],
    hits_preds: list[dict[str, Any]],
    umpire_name: str = "",
    weather: dict[str, Any] | None = None,
    bullpen_home: float = 0.5,
    bullpen_away: float = 0.5,
    value_bets: list[dict[str, Any]] | None = None,
    market_odds: dict[str, Any] | None = None,
) -> None:
    away = game.get("away_team", "Away")
    home = game.get("home_team", "Home")
    venue = game.get("venue", "")
    away_sp = game.get("away_sp", "TBD")
    home_sp = game.get("home_sp", "TBD")

    away_win_prob = winner_pred.get("away_win_prob", 0.5)
    home_win_prob = winner_pred.get("home_win_prob", 0.5)
    winner = home if home_win_prob >= away_win_prob else away
    winner_prob = max(home_win_prob, away_win_prob) * 100

    total_runs = total_runs_pred.get("predicted_runs", 9.0)
    margin = total_runs_pred.get("margin", 2.0)

    print()
    print(f"🏟️  {away} vs {home}" + (f" ({venue})" if venue else ""))
    print(f"   SP: {away_sp} vs {home_sp}")

    # Umpire line
    if umpire_name:
        k_adj = UMPIRE_K_ADJUSTMENTS.get(umpire_name, 0.0)
        adj_str = f"(K adj: {k_adj:+.1%})" if k_adj != 0.0 else ""
        print(f"   Umpire: {umpire_name} {adj_str}".rstrip())

    # Weather line
    if weather:
        temp = weather.get("temp_f", "?")
        wind = weather.get("wind_speed_mph", "?")
        desc = weather.get("description", "")
        is_out = weather.get("is_outdoor", 1)
        roof_tag = "⛺ DOME" if not is_out else "☀️"
        print(f"   Weather: {temp:.0f}°F, Wind {wind:.0f}mph {roof_tag}  {desc}")

    # Bullpen fatigue
    home_abbrev = game.get("home_abbrev", home[:3].upper())
    away_abbrev = game.get("away_abbrev", away[:3].upper())
    home_bull_emoji = "😴 fatigued" if bullpen_home >= 0.65 else "✅ fresh"
    away_bull_emoji = "😴 fatigued" if bullpen_away >= 0.65 else "✅ fresh"
    print(f"   Bullpen: {home_abbrev} {home_bull_emoji} ({bullpen_home:.2f}) | "
          f"{away_abbrev} {away_bull_emoji} ({bullpen_away:.2f})")

    print("   " + "─" * 36)

    # Winner / total lines with optional market comparison
    winner_line = f"   Winner:      {winner} ({winner_prob:.1f}%)"
    if market_odds:
        game_key = f"{away_abbrev}_vs_{home_abbrev}"
        gm = market_odds.get(game_key, {})
        if home_win_prob >= 0.5:
            ml_odds = gm.get("moneyline_home")
        else:
            ml_odds = gm.get("moneyline_away")
        if ml_odds is not None:
            from value_betting import american_to_implied
            market_impl = american_to_implied(int(ml_odds)) * 100
            edge = winner_prob - market_impl
            edge_tag = f"✅ VALUE +{edge:.1f}%" if edge >= 3 else f"Edge {edge:+.1f}%"
            winner_line += f" | Market: {ml_odds:+d} ({market_impl:.1f}%) | {edge_tag}"
    print(winner_line)

    runs_line = f"   Total Runs:  {total_runs:.1f} (±{margin:.1f})"
    if market_odds:
        game_key = f"{away_abbrev}_vs_{home_abbrev}"
        gm = market_odds.get(game_key, {})
        total_line_val = gm.get("total_over")
        if total_line_val is not None:
            direction = "LEAN OVER" if total_runs > float(total_line_val) else "LEAN UNDER"
            tag = "✅" if abs(total_runs - float(total_line_val)) >= 0.5 else "📊"
            runs_line += f" | Market O/U: {total_line_val}  | {tag} {direction}"
    print(runs_line)

    print("   " + "─" * 36)

    # Value bets block
    if value_bets:
        from value_betting import print_value_bets
        print_value_bets(value_bets)
        print("   " + "─" * 36)

    if hr_preds:
        print("   HR Props:")
        for p in hr_preds:
            name = p.get("player_name", "Unknown")
            exp = p.get("expected_hr", 0.0)
            prob = p.get("prob_hr_ge_1", 0.0) * 100
            xwoba = p.get("xwoba", 0.0)
            streak = p.get("hot_cold_streak", 1.0)
            streak_tag = " 🔥 HOT" if streak > 1.1 else (" 🧊 COLD" if streak < 0.9 else "")
            xwoba_str = f" | xwOBA: .{int(xwoba * 1000):03d}" if xwoba else ""
            print(f"     {name[:18]:<18}  {exp:.1f} HR (P≥1: {prob:.1f}%){xwoba_str}{streak_tag}")

    print("   " + "─" * 36)

    if k_preds:
        print("   Strikeouts:")
        for p in k_preds:
            name_k = p.get("player_name", "Unknown")
            exp_k = p.get("expected_k", 0.0)
            whiff = p.get("whiff_rate", 0.0)
            chase = p.get("chase_rate", 0.0)
            print(f"     {name_k[:18]:<18}  {exp_k:.1f} K | whiff: {whiff:.0%} | chase: {chase:.0%}")

    print("   " + "─" * 36)

    if hits_preds:
        print("   Hits Props:")
        for p in hits_preds:
            name_h = p.get("player_name", "Unknown")
            exp_h = p.get("expected_hits", 0.0)
            prob_h = p.get("prob_hit_ge_1", 0.0) * 100
            xba = p.get("xba", 0.0)
            xba_str = f" | xBA: .{int(xba * 1000):03d}" if xba else ""
            print(f"     {name_h[:18]:<18}  {exp_h:.1f} hits (P≥1: {prob_h:.1f}%){xba_str}")


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
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run historical backtester instead of daily predictions.",
    )
    parser.add_argument(
        "--odds",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to JSON file with market odds for value betting analysis.",
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        dest="no_weather",
        help="Skip weather API calls (faster).",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        dest="no_ensemble",
        help="Use single XGBoost models instead of the ensemble. Default: ensemble is on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Backtest mode
    # ------------------------------------------------------------------
    if args.backtest:
        from backtester import run_backtest
        run_backtest()
        return

    target_date: datetime.date
    if args.date:
        target_date = datetime.date.fromisoformat(args.date)
    else:
        target_date = datetime.date.today()

    date_str = target_date.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Load market odds (optional)
    # ------------------------------------------------------------------
    market_odds: dict[str, Any] | None = None
    if args.odds:
        from value_betting import load_odds_file
        market_odds = load_odds_file(args.odds)
        if market_odds:
            print(f"[INFO] Loaded market odds from {args.odds} ({len(market_odds)} games).")
        else:
            print(f"[WARN] Could not load odds from {args.odds}.")
    else:
        # Auto-detect default odds file
        from value_betting import load_odds_file, ODDS_FILE_DEFAULT
        market_odds = load_odds_file()
        if market_odds:
            print(f"[INFO] Auto-loaded odds from odds_today.json ({len(market_odds)} games).")

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
    # 2. Get today's schedule  (do this early so we can load context data)
    # ------------------------------------------------------------------
    print(f"[INFO] Fetching schedule for {date_str} …")
    today_games_raw = load_schedule_today(target_date)

    if not today_games_raw:
        print(f"[INFO] No games found for {date_str}.")
        return

    # ------------------------------------------------------------------
    # 3. Load umpire data for today's games
    # ------------------------------------------------------------------
    umpire_by_game: dict[str, Any] = {}
    for game in today_games_raw:
        game_id = game.get("game_id")
        if game_id:
            try:
                ud = load_umpire_data(int(game_id))
                umpire_by_game[str(game_id)] = ud
            except Exception as exc:
                print(f"[WARN] Could not load umpire data for game {game_id}: {exc}")

    # ------------------------------------------------------------------
    # 4. Load weather data (unless --no-weather)
    # ------------------------------------------------------------------
    weather_by_team: dict[str, dict[str, Any]] = {}
    if not args.no_weather:
        teams_today = set()
        for game in today_games_raw:
            ht = TEAM_ABBREV_MAP.get(game.get("home_team", ""), "")
            if ht:
                teams_today.add(ht)
        for team in teams_today:
            try:
                weather_by_team[team] = load_weather_data(team, date_str)
            except Exception as exc:
                print(f"[WARN] Could not load weather for {team}: {exc}")

    # ------------------------------------------------------------------
    # 5. Load schedule context (rest days, travel, bullpen)
    # ------------------------------------------------------------------
    schedule_ctx: dict[str, dict[str, Any]] = {}
    bullpen_ctx: dict[str, dict[str, Any]] = {}
    for team, logs_df in game_logs.items():
        try:
            schedule_ctx[team] = load_team_schedule_context(team, date_str, logs_df)
        except Exception as exc:
            print(f"[WARN] Could not load schedule context for {team}: {exc}")
    for team in unique_teams:
        for season in SEASONS:
            try:
                bu = load_bullpen_usage(team, date_str, season)
                if team not in bullpen_ctx or bu.get("bullpen_pitches_last_3d", 0) > 0:
                    bullpen_ctx[team] = bu
            except Exception as exc:
                print(f"[WARN] Could not load bullpen usage for {team}: {exc}")

    # ------------------------------------------------------------------
    # 6. Instantiate models
    # ------------------------------------------------------------------
    use_ensemble = not args.no_ensemble

    if use_ensemble:
        total_runs_model_ensemble = EnsembleTotalRunsModel()
        game_winner_model_ensemble = EnsembleGameWinnerModel()

    total_runs_model = TotalRunsModel()
    game_winner_model = GameWinnerModel()
    hr_model = HomeRunsModel()
    strikeouts_model = StrikeoutsModel()
    hits_model = HitsModel()

    base_models = [total_runs_model, game_winner_model, hr_model, strikeouts_model, hits_model]

    # ------------------------------------------------------------------
    # 7. Train or load models
    # ------------------------------------------------------------------
    if args.retrain or not all(m.is_saved() for m in base_models):
        print("[INFO] Training models …")
        train_game_models(game_logs, pitching_stats, total_runs_model, game_winner_model)
        train_batter_models(batting_stats, hr_model, hits_model)
        train_pitcher_models(pitching_stats, strikeouts_model)
        if use_ensemble:
            print("[INFO] Training ensemble models …")
            from features import build_game_features
            feats_df = build_game_features(game_logs, pitching_stats)
            if not feats_df.empty and len(feats_df) >= 50:
                X_ens = _select_cols(feats_df, _GAME_FEATURE_COLS)
                y_runs_ens = feats_df["total_runs"].fillna(8.0)
                if "runs_scored" in feats_df.columns:
                    y_win_ens = (feats_df["runs_scored"] > feats_df["ra_per_game_l10"]).astype(int)
                else:
                    y_win_ens = pd.Series([1] * len(feats_df))
                try:
                    total_runs_model_ensemble.train(X_ens, y_runs_ens)
                    game_winner_model_ensemble.train(X_ens, y_win_ens)
                except Exception as exc:
                    print(f"[WARN] Ensemble training failed: {exc}")
                    use_ensemble = False
            else:
                use_ensemble = False
    else:
        print("[INFO] Loading saved models …")
        for m in base_models:
            m.load()
        if use_ensemble:
            # Ensemble models are not persisted; re-train on available data
            try:
                from features import build_game_features
                feats_df = build_game_features(game_logs, pitching_stats)
                if not feats_df.empty and len(feats_df) >= 50:
                    X_ens = _select_cols(feats_df, _GAME_FEATURE_COLS)
                    y_runs_ens = feats_df["total_runs"].fillna(8.0)
                    if "runs_scored" in feats_df.columns:
                        y_win_ens = (feats_df["runs_scored"] > feats_df["ra_per_game_l10"]).astype(int)
                    else:
                        y_win_ens = pd.Series([1] * len(feats_df))
                    total_runs_model_ensemble.train(X_ens, y_runs_ens)
                    game_winner_model_ensemble.train(X_ens, y_win_ens)
                else:
                    use_ensemble = False
            except Exception as exc:
                print(f"[WARN] Ensemble init failed: {exc}")
                use_ensemble = False

    # ------------------------------------------------------------------
    # 8. Build today's game features (with new context)
    # ------------------------------------------------------------------
    today_game_features = build_today_game_features(
        today_games_raw,
        game_logs,
        pitching_stats,
        umpire_data=umpire_by_game,
        weather_data=weather_by_team,
        schedule_context=schedule_ctx,
        bullpen_data=bullpen_ctx,
    )

    # ------------------------------------------------------------------
    # 9. Generate predictions + collect output rows
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

        # Umpire info for this game
        game_id = game.get("game_id")
        umpire_name = ""
        umpire_k_adj_val = 0.0
        if game_id:
            ump_info = umpire_by_game.get(str(game_id), {})
            umpire_name = ump_info.get("home_plate_umpire", "") if isinstance(ump_info, dict) else ""
            umpire_k_adj_val = UMPIRE_K_ADJUSTMENTS.get(umpire_name, 0.0)

        weather_info = weather_by_team.get(home_abbrev)
        bullpen_home = bullpen_ctx.get(home_abbrev, {}).get("bullpen_fatigue_score", 0.5)
        bullpen_away = bullpen_ctx.get(away_abbrev, {}).get("bullpen_fatigue_score", 0.5)

        # --- Game-level prediction ---
        if i < len(today_game_features):
            game_feat_row = today_game_features.iloc[[i]]
            game_X = pd.DataFrame(
                {c: game_feat_row[c].values if c in game_feat_row.columns else [0.0]
                 for c in _GAME_FEATURE_COLS}
            ).fillna(0.0)

            if use_ensemble and total_runs_model_ensemble.is_trained():
                total_runs_preds = total_runs_model_ensemble.predict_with_confidence(game_X)
            else:
                total_runs_preds = total_runs_model.predict_with_confidence(game_X)
            total_runs_pred = total_runs_preds[0] if total_runs_preds else {"predicted_runs": 9.0, "margin": 2.0}

            if use_ensemble and game_winner_model_ensemble.is_trained():
                home_win_prob_arr = game_winner_model_ensemble.predict_calibrated_proba(game_X)
            else:
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

        # --- Value bets ---
        game_value_bets: list[dict[str, Any]] = []
        if market_odds is not None:
            from value_betting import find_value_bets
            game_key = f"{away_abbrev}_vs_{home_abbrev}"
            preds_for_vb = [
                {
                    "bet_type": "moneyline_home",
                    "description": f"{home_abbrev} ML",
                    "model_prob": home_win_prob,
                    "game_key": game_key,
                },
                {
                    "bet_type": "moneyline_away",
                    "description": f"{away_abbrev} ML",
                    "model_prob": away_win_prob,
                    "game_key": game_key,
                },
            ]
            # Total runs over/under
            gm_odds = market_odds.get(game_key, {})
            if "total_over" in gm_odds:
                over_line = float(gm_odds["total_over"])
                pred_total = total_runs_pred.get("predicted_runs", 9.0)
                # Model over prob: if prediction > line assume ~60-70% over
                # Simple linear mapping: each 0.5-run difference from the line
                # translates to ~4% probability shift (0.08 per run difference),
                # capped at ±25% from neutral (max 75% / min 25%)
                over_prob = 0.5 + min(0.25, max(-0.25, (pred_total - over_line) * 0.08))
                preds_for_vb.append({
                    "bet_type": "total_over",
                    "description": f"Over {over_line} {gm_odds.get('total_over_odds', -110):+d}",
                    "model_prob": over_prob,
                    "game_key": game_key,
                })
            game_value_bets = find_value_bets(preds_for_vb, market_odds)

        # --- Batter props ---
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

        away_batter_feat = build_today_batter_features(
            away_lineup, batting_stats,
            opp_pitcher_stats=home_pitcher_summary,
            park_factor_hr=park_hr,
            park_factor_hits=park_hits,
            umpire_k_adj=umpire_k_adj_val,
        )
        home_batter_feat = build_today_batter_features(
            home_lineup, batting_stats,
            opp_pitcher_stats=away_pitcher_summary,
            park_factor_hr=park_hr,
            park_factor_hits=park_hits,
            umpire_k_adj=umpire_k_adj_val,
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
                    "xwoba": float(row_feat.get("xwoba", 0.0)),
                    "hot_cold_streak": float(row_feat.get("hot_cold_streak", 1.0)),
                    **hr_r,
                })
                hits_preds_list.append({
                    "player_name": row_feat.get("player_name", ""),
                    "xba": float(row_feat.get("xba", 0.0)),
                    **hits_r,
                })

        hr_preds_list.sort(key=lambda x: x.get("expected_hr", 0), reverse=True)
        hits_preds_list.sort(key=lambda x: x.get("expected_hits", 0), reverse=True)

        # --- Pitcher strikeouts ---
        away_pitcher_dict = [{"player_name": away_sp_name, "hand": away_pitcher_summary.get("hand", "R")}] if away_sp_name != "TBD" else []
        home_pitcher_dict = [{"player_name": home_sp_name, "hand": home_pitcher_summary.get("hand", "R")}] if home_sp_name != "TBD" else []

        away_team_bat = _get_team_batting_summary(batting_stats, away_abbrev)
        home_team_bat = _get_team_batting_summary(batting_stats, home_abbrev)

        away_rest = schedule_ctx.get(away_abbrev, {}).get("rest_days", 5)
        home_rest = schedule_ctx.get(home_abbrev, {}).get("rest_days", 5)

        away_k_feat = build_today_pitcher_features(
            away_pitcher_dict, pitching_stats,
            opp_team_batting=home_team_bat,
            rest_days=away_rest,
            umpire_k_adj=umpire_k_adj_val,
        )
        home_k_feat = build_today_pitcher_features(
            home_pitcher_dict, pitching_stats,
            opp_team_batting=away_team_bat,
            rest_days=home_rest,
            umpire_k_adj=umpire_k_adj_val,
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
                    "whiff_rate": float(row_feat.get("whiff_rate", 0.25)),
                    "chase_rate": float(row_feat.get("chase_rate", 0.28)),
                })

        # --- Print game block ---
        _print_game(
            {**game, "away_sp": away_sp_name, "home_sp": home_sp_name,
             "away_abbrev": away_abbrev, "home_abbrev": home_abbrev},
            winner_pred,
            total_runs_pred,
            hr_preds_list[:5],
            k_preds_list,
            hits_preds_list[:5],
            umpire_name=umpire_name,
            weather=weather_info,
            bullpen_home=bullpen_home,
            bullpen_away=bullpen_away,
            value_bets=game_value_bets,
            market_odds=market_odds,
        )

        # --- Accumulate output CSV rows ---
        winner_name = home_name if home_win_prob >= away_win_prob else away_name
        csv_row: dict[str, Any] = {
            "date": date_str,
            "away_team": away_name,
            "home_team": home_name,
            "away_sp": away_sp_name,
            "home_sp": home_sp_name,
            "venue": game.get("venue", ""),
            "umpire": umpire_name,
            "weather_temp_f": weather_info.get("temp_f", "") if weather_info else "",
            "weather_wind_mph": weather_info.get("wind_speed_mph", "") if weather_info else "",
            "home_bullpen_fatigue": round(bullpen_home, 3),
            "away_bullpen_fatigue": round(bullpen_away, 3),
            "predicted_winner": winner_name,
            "winner_confidence_pct": round(max(home_win_prob, away_win_prob) * 100, 1),
            "predicted_total_runs": round(total_runs_pred["predicted_runs"], 1),
            "total_runs_margin": round(total_runs_pred["margin"], 1),
        }
        # Value bets summary
        if game_value_bets:
            csv_row["value_bets_count"] = len(game_value_bets)
            csv_row["top_edge_pct"] = round(game_value_bets[0]["edge"] * 100, 2)
        output_rows.append(csv_row)

        for p in hr_preds_list:
            output_rows[-1][f'hr_{p["player_name"][:10]}_expected'] = round(p.get("expected_hr", 0.0), 2)
            output_rows[-1][f'hr_{p["player_name"][:10]}_prob'] = round(p.get("prob_hr_ge_1", 0.0) * 100, 1)

    # ------------------------------------------------------------------
    # 10. Save output CSV
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
