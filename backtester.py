"""Historical backtesting engine for MLB prediction model validation.

Usage::

    python backtester.py                    # backtest last two seasons
    python backtester.py --season 2024      # backtest a single season
    python backtester.py --season 2023 2024 # backtest multiple seasons
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import pandas as pd

from config import MIN_EDGE_PCT


# ---------------------------------------------------------------------------
# Core backtest function
# ---------------------------------------------------------------------------

def backtest_season(
    season: int,
    models: dict[str, Any],
    data_loader_module: Any,
) -> dict[str, Any]:
    """Run predictions on historical games and compare to actual results.

    Parameters
    ----------
    season:
        MLB season year to backtest.
    models:
        Dict mapping model names to trained model instances.
        Expected keys: ``"total_runs"``, ``"game_winner"``.
    data_loader_module:
        The ``data_loader`` module (or any object exposing ``load_game_results``
        and ``load_team_game_logs``).

    Returns
    -------
    Dict with keys:
    - ``total_games``
    - ``correct_winners``
    - ``winner_accuracy``
    - ``total_runs_mae``
    - ``brier_score``
    - ``roi_simulation``
    - ``calibration_buckets``
    """
    print(f"[BACKTEST] Loading game results for {season}...")
    try:
        game_results = data_loader_module.load_game_results(season)
    except Exception as exc:
        print(f"[WARN] Could not load game results for {season}: {exc}")
        return _empty_results(season)

    if game_results.empty:
        print(f"[WARN] No game results found for {season}.")
        return _empty_results(season)

    print(f"[BACKTEST] Loading game logs for {season}...")
    from config import TEAM_ABBREV_MAP
    from features import build_today_game_features

    unique_teams = list(dict.fromkeys(TEAM_ABBREV_MAP.values()))
    game_logs: dict[str, pd.DataFrame] = {}
    for team in unique_teams:
        try:
            df = data_loader_module.load_team_game_logs(team, season)
            if not df.empty:
                game_logs[team] = df
        except Exception as exc:
            print(f"[WARN] Could not load game logs for {team} {season}: {exc}")

    pitching_stats = pd.DataFrame()
    try:
        pitching_stats = data_loader_module.load_pitching_stats(season)
    except Exception as exc:
        print(f"[WARN] Could not load pitching stats for {season}: {exc}")

    total_runs_model = models.get("total_runs")
    game_winner_model = models.get("game_winner")

    _GAME_FEATURE_COLS = [
        "runs_per_game_l10", "runs_per_game_l30", "ra_per_game_l10",
        "win_pct_l10", "win_pct_season", "run_diff_l10",
        "park_factor_runs", "sp_era_l10", "sp_whip_l5",
    ]

    total_games = 0
    correct_winners = 0
    runs_errors: list[float] = []
    brier_scores: list[float] = []
    roi_bets: list[dict[str, float]] = []

    # Calibration buckets: prob range → [predicted_wins, total]
    cal_buckets: dict[str, list[int]] = {
        "50-55": [0, 0], "55-60": [0, 0], "60-65": [0, 0],
        "65-70": [0, 0], "70+": [0, 0],
    }

    # Process each completed game
    for _, game_row in game_results.iterrows():
        try:
            away_team = str(game_row.get("away_team", ""))
            home_team = str(game_row.get("home_team", ""))
            away_score = float(game_row.get("away_score", 0) or 0)
            home_score = float(game_row.get("home_score", 0) or 0)
            actual_total = away_score + home_score
            actual_winner_is_home = int(home_score > away_score)

            # Build a mock "today_games" entry for this historical game
            mock_game: dict[str, Any] = {
                "game_id": game_row.get("game_id", ""),
                "date": game_row.get("date", ""),
                "away_team": away_team,
                "home_team": home_team,
                "away_probable_pitcher": "TBD",
                "home_probable_pitcher": "TBD",
                "venue": "",
                "game_time": "",
                "status": "Final",
            }

            feat_df = build_today_game_features([mock_game], game_logs, pitching_stats)
            if feat_df.empty:
                continue

            game_X = pd.DataFrame(
                {c: feat_df[c].values if c in feat_df.columns else [0.0]
                 for c in _GAME_FEATURE_COLS}
            ).fillna(0.0)

            # Total runs prediction
            pred_total: float = 9.0
            if total_runs_model is not None and total_runs_model.is_trained():
                try:
                    preds = total_runs_model.predict(game_X)
                    pred_total = float(np.atleast_1d(preds)[0])
                except Exception:
                    pass

            # Winner probability
            home_win_prob: float = 0.5
            if game_winner_model is not None and game_winner_model.is_trained():
                try:
                    probs = game_winner_model.predict_proba(game_X)
                    home_win_prob = float(np.atleast_1d(probs)[0])
                except Exception:
                    pass

            pred_winner_is_home = int(home_win_prob >= 0.5)

            total_games += 1
            if pred_winner_is_home == actual_winner_is_home:
                correct_winners += 1

            runs_errors.append(abs(pred_total - actual_total))

            # Brier score: (prob - outcome)^2
            brier_scores.append((home_win_prob - actual_winner_is_home) ** 2)

            # Calibration
            conf = max(home_win_prob, 1 - home_win_prob)
            conf_pct = conf * 100
            if conf_pct < 55:
                bucket = "50-55"
            elif conf_pct < 60:
                bucket = "55-60"
            elif conf_pct < 65:
                bucket = "60-65"
            elif conf_pct < 70:
                bucket = "65-70"
            else:
                bucket = "70+"
            cal_buckets[bucket][1] += 1
            # Count as correct if model's favoured side matches actual
            if pred_winner_is_home == actual_winner_is_home:
                cal_buckets[bucket][0] += 1

            # ROI simulation: bet on model's favoured side at closing line −110
            # −110 implied probability ≈ 52.38% (100/190)
            edge = conf - 0.5238
            if edge >= MIN_EDGE_PCT:
                stake = 1.0
                # Approximate +EV: win stake * (100/110) if correct, else lose stake
                payoff = stake * (100 / 110) if pred_winner_is_home == actual_winner_is_home else -stake
                roi_bets.append({"stake": stake, "payoff": payoff})

        except Exception as exc:
            print(f"[WARN] Error processing game row for backtest: {exc}")

    winner_accuracy = correct_winners / max(total_games, 1)
    total_runs_mae = float(np.mean(runs_errors)) if runs_errors else 0.0
    brier_score = float(np.mean(brier_scores)) if brier_scores else 0.0

    # ROI
    if roi_bets:
        total_staked = sum(b["stake"] for b in roi_bets)
        total_payoff = sum(b["payoff"] for b in roi_bets)
        roi_simulation = total_payoff / max(total_staked, 1)
    else:
        roi_simulation = 0.0

    # Calibration buckets → win rates
    calibration_buckets: dict[str, float] = {}
    for bkt, (wins, total) in cal_buckets.items():
        calibration_buckets[bkt] = round(wins / max(total, 1), 3)

    return {
        "season": season,
        "total_games": total_games,
        "correct_winners": correct_winners,
        "winner_accuracy": round(winner_accuracy, 4),
        "total_runs_mae": round(total_runs_mae, 3),
        "brier_score": round(brier_score, 4),
        "roi_simulation": round(roi_simulation, 4),
        "calibration_buckets": calibration_buckets,
    }


def _empty_results(season: int) -> dict[str, Any]:
    return {
        "season": season,
        "total_games": 0,
        "correct_winners": 0,
        "winner_accuracy": 0.0,
        "total_runs_mae": 0.0,
        "brier_score": 0.0,
        "roi_simulation": 0.0,
        "calibration_buckets": {},
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_backtest_report(results: dict[str, Any]) -> None:
    """Pretty-print backtest results to the console."""
    season = results.get("season", "?")
    print()
    print("=" * 50)
    print(f"  BACKTEST REPORT — {season} Season")
    print("=" * 50)
    print(f"  Total games evaluated : {results.get('total_games', 0)}")
    acc = results.get("winner_accuracy", 0.0) * 100
    print(f"  Winner accuracy       : {results.get('correct_winners', 0)} / "
          f"{results.get('total_games', 0)}  ({acc:.1f}%)")
    print(f"  Total Runs MAE        : {results.get('total_runs_mae', 0.0):.2f} runs")
    print(f"  Brier Score           : {results.get('brier_score', 0.0):.4f}  "
          f"(lower is better, 0.25 = coin flip)")
    roi_pct = results.get("roi_simulation", 0.0) * 100
    roi_sign = "+" if roi_pct >= 0 else ""
    print(f"  Simulated ROI (≥3% edge bets): {roi_sign}{roi_pct:.1f}%")
    print()
    print("  Calibration (model confidence → actual win rate):")
    for bucket, rate in results.get("calibration_buckets", {}).items():
        bar_filled = int(rate * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"    {bucket:>5}%  [{bar}]  {rate * 100:.1f}%")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_backtest(seasons: list[int] | None = None) -> None:
    """Main entry point: train models on all available data then backtest.

    Parameters
    ----------
    seasons:
        List of season years to backtest. Defaults to the two most recent
        seasons in config.SEASONS.
    """
    from config import SEASONS as _SEASONS
    import data_loader as dl

    if seasons is None:
        seasons = _SEASONS[-2:] if len(_SEASONS) >= 2 else _SEASONS

    print(f"[BACKTEST] Loading training data for seasons: {_SEASONS}")

    # Load training data
    from data_loader import (
        load_all_batting_stats,
        load_all_pitching_stats,
        load_team_game_logs,
    )
    from features import build_game_features, build_batter_features, build_pitcher_features
    from models import TotalRunsModel, GameWinnerModel, HomeRunsModel, HitsModel, StrikeoutsModel
    import numpy as np

    batting_stats = load_all_batting_stats()
    pitching_stats = load_all_pitching_stats()

    from config import TEAM_ABBREV_MAP
    unique_teams = list(dict.fromkeys(TEAM_ABBREV_MAP.values()))
    game_logs: dict[str, pd.DataFrame] = {}
    for team in unique_teams:
        for s in _SEASONS:
            df_log = load_team_game_logs(team, s)
            if not df_log.empty:
                if team in game_logs:
                    game_logs[team] = pd.concat([game_logs[team], df_log], ignore_index=True)
                else:
                    game_logs[team] = df_log

    # Train / load models
    total_runs_model = TotalRunsModel()
    game_winner_model = GameWinnerModel()
    hr_model = HomeRunsModel()
    strikeouts_model = StrikeoutsModel()
    hits_model = HitsModel()

    _GAME_FEATURE_COLS = [
        "runs_per_game_l10", "runs_per_game_l30", "ra_per_game_l10",
        "win_pct_l10", "win_pct_season", "run_diff_l10",
        "park_factor_runs", "sp_era_l10", "sp_whip_l5",
    ]

    def _select_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        result = df.reindex(columns=cols, fill_value=0.0).fillna(0.0)
        return result

    all_models = [total_runs_model, game_winner_model, hr_model, strikeouts_model, hits_model]
    if not all(m.is_saved() for m in all_models):
        print("[BACKTEST] Training models (no saved models found)...")
        features_df = build_game_features(game_logs, pitching_stats)
        if not features_df.empty and len(features_df) >= 50:
            X = _select_cols(features_df, _GAME_FEATURE_COLS)
            y_runs = features_df["total_runs"].fillna(8.0)
            if "runs_scored" in features_df.columns:
                y_winner = (features_df["runs_scored"] > features_df["ra_per_game_l10"]).astype(int)
            else:
                y_winner = pd.Series([1] * len(features_df))
            total_runs_model.train(X, y_runs)
            total_runs_model.save()
            game_winner_model.train(X, y_winner)
            game_winner_model.save()
        else:
            print("[WARN] Insufficient game data for training; backtest accuracy will be limited.")
    else:
        print("[BACKTEST] Loading saved models...")
        for m in all_models:
            m.load()

    models_dict = {
        "total_runs": total_runs_model,
        "game_winner": game_winner_model,
    }

    # Run backtest for each season
    for season in seasons:
        results = backtest_season(season, models_dict, dl)
        print_backtest_report(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB prediction model backtester")
    parser.add_argument(
        "--season",
        type=int,
        nargs="+",
        default=None,
        help="Season year(s) to backtest (e.g. --season 2023 2024). Defaults to last 2 seasons.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(seasons=args.season)
