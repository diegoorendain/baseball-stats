"""Feature engineering for all five prediction models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from config import PARK_FACTORS, TEAM_ABBREV_MAP


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rolling_mean(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).mean()


def _safe_get(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = d.get(key, default)
    return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else default


def _park_factor(team: str, stat: str = "runs") -> float:
    factors = PARK_FACTORS.get(team, {})
    if isinstance(factors, dict):
        return factors.get(stat, 1.0)
    # Fallback – old scalar format
    return float(factors)


def _team_abbrev(name: str) -> str:
    return TEAM_ABBREV_MAP.get(name, name[:3].upper())


# ---------------------------------------------------------------------------
# 1. Game-level features (Total Runs + Game Winner models)
# ---------------------------------------------------------------------------

def build_game_features(
    game_logs: dict[str, pd.DataFrame],
    pitching_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per historical game with rolling team/pitcher features.

    Parameters
    ----------
    game_logs:
        Mapping from team abbreviation to its per-game log DataFrame
        (output of ``load_team_game_logs``).
    pitching_stats:
        Season pitching stats DataFrame (output of ``load_pitching_stats``).

    Returns
    -------
    DataFrame with features for Total Runs and Game Winner models.
    """
    rows: list[dict[str, Any]] = []

    for team, logs in game_logs.items():
        if logs.empty:
            continue

        logs = logs.copy().reset_index(drop=True)

        # Normalise column names to lowercase
        logs.columns = [c.lower().replace(" ", "_") for c in logs.columns]

        # Identify run columns (vary by pybaseball version)
        run_col = next(
            (c for c in logs.columns if c in ("r", "runs", "rs", "r_scored")),
            None,
        )
        ra_col = next(
            (c for c in logs.columns if c in ("ra", "runs_allowed", "ra_allowed")),
            None,
        )
        win_col = next(
            (c for c in logs.columns if c in ("w/l", "w_l", "result", "win")),
            None,
        )

        if run_col is None or ra_col is None:
            continue

        logs[run_col] = pd.to_numeric(logs[run_col], errors="coerce").fillna(0)
        logs[ra_col] = pd.to_numeric(logs[ra_col], errors="coerce").fillna(0)

        for idx in range(len(logs)):
            if idx < 5:  # need at least 5 games of history
                continue

            past = logs.iloc[:idx]
            current = logs.iloc[idx]

            run_l10 = _rolling_mean(past[run_col], 10).iloc[-1]
            ra_l10 = _rolling_mean(past[ra_col], 10).iloc[-1]
            run_l30 = _rolling_mean(past[run_col], 30).iloc[-1]

            total_runs = float(current[run_col]) + float(current[ra_col])

            # Win percentage
            if win_col and win_col in logs.columns:
                wins = past[win_col].str.startswith("W").sum() if hasattr(past[win_col], "str") else 0
                win_pct_l10_val = (
                    past.tail(10)[win_col].str.startswith("W").sum() / 10.0
                    if hasattr(past.tail(10)[win_col], "str")
                    else 0.5
                )
                win_pct_season_val = wins / max(len(past), 1)
            else:
                win_pct_l10_val = 0.5
                win_pct_season_val = 0.5

            row: dict[str, Any] = {
                "team": team,
                "game_idx": idx,
                "total_runs": total_runs,
                "runs_scored": float(current[run_col]),
                # Rolling batting features (home team perspective)
                "runs_per_game_l10": run_l10,
                "runs_per_game_l30": run_l30,
                "ra_per_game_l10": ra_l10,
                "win_pct_l10": win_pct_l10_val,
                "win_pct_season": win_pct_season_val,
                "run_diff_l10": run_l10 - ra_l10,
                # Park factor
                "park_factor_runs": _park_factor(team, "runs"),
                # Placeholder pitcher features (filled below if available)
                "sp_era_l10": 4.50,
                "sp_whip_l5": 1.30,
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.dropna(subset=["total_runs"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Batter-level features (Home Runs + Hits models)
# ---------------------------------------------------------------------------

def build_batter_features(
    batting_stats: pd.DataFrame,
    statcast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-player batter features.

    Parameters
    ----------
    batting_stats:
        Season batting stats from pybaseball.
    statcast:
        Optional statcast DataFrame for the player.

    Returns
    -------
    DataFrame with one row per player with aggregated features.
    """
    if batting_stats.empty:
        return pd.DataFrame()

    bs = batting_stats.copy()
    bs.columns = [c.lower().replace(" ", "_").replace("%", "_pct") for c in bs.columns]

    rows: list[dict[str, Any]] = []

    for _, player in bs.iterrows():
        pa = float(player.get("pa", player.get("ab", 1)) or 1)
        ab = float(player.get("ab", pa) or pa)
        hits = float(player.get("h", 0) or 0)
        hr = float(player.get("hr", 0) or 0)
        bb = float(player.get("bb", 0) or 0)
        so = float(player.get("so", player.get("k", 0)) or 0)
        doubles = float(player.get("2b", 0) or 0)
        triples = float(player.get("3b", 0) or 0)
        avg = hits / max(ab, 1)
        slg = (hits + doubles + 2 * triples + 3 * hr) / max(ab, 1)
        obp = (hits + bb) / max(pa, 1)
        ops = obp + slg
        iso = slg - avg
        k_rate = so / max(pa, 1)
        bb_rate = bb / max(pa, 1)
        g = float(player.get("g", 1) or 1)

        # Statcast metrics (defaults if not available)
        barrel_rate = 0.07
        avg_exit_velo = 88.0
        avg_launch_angle = 12.0
        hard_hit_rate = 0.35

        # Use pre-computed statcast metrics from the stats DataFrame when available
        # (populated by load_batting_stats when data comes from Statcast aggregation)
        if "barrel_rate" in player.index and pd.notna(player.get("barrel_rate")):
            barrel_rate = float(player["barrel_rate"])
        if "avg_exit_velo" in player.index and pd.notna(player.get("avg_exit_velo")):
            avg_exit_velo = float(player["avg_exit_velo"])
        if "avg_launch_angle" in player.index and pd.notna(player.get("avg_launch_angle")):
            avg_launch_angle = float(player["avg_launch_angle"])
        if "hard_hit_rate" in player.index and pd.notna(player.get("hard_hit_rate")):
            hard_hit_rate = float(player["hard_hit_rate"])

        if statcast is not None and not statcast.empty:
            sc = statcast.copy()
            sc.columns = [c.lower() for c in sc.columns]
            if "launch_speed" in sc.columns:
                avg_exit_velo = float(sc["launch_speed"].dropna().mean() or 88.0)
            if "launch_angle" in sc.columns:
                avg_launch_angle = float(sc["launch_angle"].dropna().mean() or 12.0)
            if "barrel" in sc.columns:
                barrel_rate = float((sc["barrel"] == 1).mean() or 0.07)
            if "launch_speed" in sc.columns:
                hard_hit_rate = float((sc["launch_speed"] >= 95).mean() or 0.35)

        row: dict[str, Any] = {
            "player_name": str(player.get("name", player.get("player_name", "Unknown"))),
            "player_id": int(player.get("playerid", player.get("player_id", 0)) or 0),
            "season": int(player.get("season", 0) or 0),
            "team": str(player.get("team", player.get("tm", "")) or ""),
            # Batting averages / rates
            "avg": avg,
            "ops": ops,
            "obp": obp,
            "slg": slg,
            "iso": iso,
            "k_rate": k_rate,
            "bb_rate": bb_rate,
            # Per-game rates (use season totals as proxy for rolling)
            "hr_per_game": hr / max(g, 1),
            "hits_per_game": hits / max(g, 1),
            # Statcast
            "barrel_rate": barrel_rate,
            "avg_exit_velo": avg_exit_velo,
            "avg_launch_angle": avg_launch_angle,
            "hard_hit_rate": hard_hit_rate,
            # Raw counts
            "hr_season": hr,
            "hits_season": hits,
            "pa_season": pa,
            "games": g,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Pitcher-level features (Strikeouts model)
# ---------------------------------------------------------------------------

def build_pitcher_features(
    pitching_stats: pd.DataFrame,
    statcast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-pitcher features.

    Parameters
    ----------
    pitching_stats:
        Season pitching stats from pybaseball.
    statcast:
        Optional statcast DataFrame for the pitcher.

    Returns
    -------
    DataFrame with one row per pitcher with aggregated features.
    """
    if pitching_stats.empty:
        return pd.DataFrame()

    ps = pitching_stats.copy()
    ps.columns = [c.lower().replace(" ", "_").replace("%", "_pct") for c in ps.columns]

    rows: list[dict[str, Any]] = []

    for _, p in ps.iterrows():
        ip = float(p.get("ip", 1) or 1)
        gs = float(p.get("gs", 1) or 1)
        so = float(p.get("so", p.get("k", 0)) or 0)
        bb = float(p.get("bb", 0) or 0)
        h = float(p.get("h", 0) or 0)
        er = float(p.get("er", 0) or 0)
        era = float(p.get("era", 4.50) or 4.50)
        whip = (h + bb) / max(ip, 1)
        k_per_9 = so / max(ip, 1) * 9
        k_rate = so / max(so + bb + h, 1)
        ip_per_start = ip / max(gs, 1)

        # Statcast pitcher metrics
        whiff_rate = 0.25
        chase_rate = 0.28
        avg_fastball_velo = 93.0

        # Use pre-computed statcast metrics from the stats DataFrame when available
        # (populated by load_pitching_stats when data comes from Statcast aggregation)
        if "whiff_rate" in p.index and pd.notna(p.get("whiff_rate")):
            whiff_rate = float(p["whiff_rate"])
        if "chase_rate" in p.index and pd.notna(p.get("chase_rate")):
            chase_rate = float(p["chase_rate"])
        if "avg_fastball_velo" in p.index and pd.notna(p.get("avg_fastball_velo")):
            avg_fastball_velo = float(p["avg_fastball_velo"])

        if statcast is not None and not statcast.empty:
            sc = statcast.copy()
            sc.columns = [c.lower() for c in sc.columns]
            if "release_speed" in sc.columns:
                fastballs = sc[sc.get("pitch_type", pd.Series()).isin(["FF", "FA", "SI", "FT"])]
                if not fastballs.empty:
                    avg_fastball_velo = float(fastballs["release_speed"].dropna().mean() or 93.0)
            if "description" in sc.columns:
                swings = sc[sc["description"].str.contains("swing|foul|hit", case=False, na=False)]
                whiffs = sc[sc["description"].str.contains("swinging_strike", case=False, na=False)]
                if len(swings) > 0:
                    whiff_rate = len(whiffs) / len(swings)

        row: dict[str, Any] = {
            "player_name": str(p.get("name", p.get("player_name", "Unknown"))),
            "player_id": int(p.get("playerid", p.get("player_id", 0)) or 0),
            "season": int(p.get("season", 0) or 0),
            "team": str(p.get("team", p.get("tm", "")) or ""),
            "era": era,
            "whip": whip,
            "k_per_9": k_per_9,
            "k_rate": k_rate,
            "k_per_game": so / max(gs, 1),
            "ip_per_start": ip_per_start,
            "whiff_rate": whiff_rate,
            "chase_rate": chase_rate,
            "avg_fastball_velo": avg_fastball_velo,
            "bb_per_9": bb / max(ip, 1) * 9,
            "hand": str(p.get("throws", p.get("p_throws", p.get("hand", p.get("p", "R")))) or "R"),
            "gs": gs,
            "so_season": so,
            "ip_season": ip,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Today's game-level features
# ---------------------------------------------------------------------------

def build_today_game_features(
    today_games: list[dict[str, Any]],
    game_logs: dict[str, pd.DataFrame],
    pitching_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Build features for today's games for Total Runs + Game Winner models.

    Parameters
    ----------
    today_games:
        List of game dicts from ``load_schedule_today()``.
    game_logs:
        Historical game logs keyed by team abbreviation.
    pitching_stats:
        Recent pitching stats DataFrame.

    Returns
    -------
    DataFrame with one row per game.
    """
    if not today_games:
        return pd.DataFrame()

    ps = pitching_stats.copy() if not pitching_stats.empty else pd.DataFrame()
    if not ps.empty:
        ps.columns = [c.lower().replace(" ", "_").replace("%", "_pct") for c in ps.columns]

    rows: list[dict[str, Any]] = []

    for game in today_games:
        away_name = game.get("away_team", "")
        home_name = game.get("home_team", "")
        away_abbrev = _team_abbrev(away_name)
        home_abbrev = _team_abbrev(home_name)

        def _team_features(abbrev: str, prefix: str) -> dict[str, float]:
            logs = game_logs.get(abbrev, pd.DataFrame())
            feat: dict[str, float] = {}
            if logs.empty:
                feat.update(
                    {
                        f"{prefix}_runs_l10": 4.5,
                        f"{prefix}_runs_l30": 4.5,
                        f"{prefix}_ra_l10": 4.5,
                        f"{prefix}_win_pct_l10": 0.5,
                        f"{prefix}_win_pct_season": 0.5,
                        f"{prefix}_run_diff_l10": 0.0,
                    }
                )
                return feat

            logs = logs.copy()
            logs.columns = [c.lower().replace(" ", "_") for c in logs.columns]
            run_col = next((c for c in logs.columns if c in ("r", "runs", "rs")), None)
            ra_col = next((c for c in logs.columns if c in ("ra", "runs_allowed")), None)
            win_col = next((c for c in logs.columns if c in ("w/l", "w_l", "result")), None)

            if run_col:
                logs[run_col] = pd.to_numeric(logs[run_col], errors="coerce").fillna(0)
                feat[f"{prefix}_runs_l10"] = float(_rolling_mean(logs[run_col], 10).iloc[-1])
                feat[f"{prefix}_runs_l30"] = float(_rolling_mean(logs[run_col], 30).iloc[-1])
            else:
                feat[f"{prefix}_runs_l10"] = 4.5
                feat[f"{prefix}_runs_l30"] = 4.5

            if ra_col:
                logs[ra_col] = pd.to_numeric(logs[ra_col], errors="coerce").fillna(0)
                feat[f"{prefix}_ra_l10"] = float(_rolling_mean(logs[ra_col], 10).iloc[-1])
            else:
                feat[f"{prefix}_ra_l10"] = 4.5

            if win_col and win_col in logs.columns and hasattr(logs[win_col], "str"):
                win_series = logs[win_col].str.startswith("W").astype(int)
                feat[f"{prefix}_win_pct_l10"] = float(win_series.tail(10).mean())
                feat[f"{prefix}_win_pct_season"] = float(win_series.mean())
            else:
                feat[f"{prefix}_win_pct_l10"] = 0.5
                feat[f"{prefix}_win_pct_season"] = 0.5

            feat[f"{prefix}_run_diff_l10"] = (
                feat[f"{prefix}_runs_l10"] - feat[f"{prefix}_ra_l10"]
            )
            return feat

        def _pitcher_features(pitcher_name: str, prefix: str) -> dict[str, float]:
            feat = {
                f"{prefix}_era": 4.50,
                f"{prefix}_whip": 1.30,
                f"{prefix}_k_per_9": 8.0,
                f"{prefix}_hand": 1.0,
            }
            if ps.empty or not pitcher_name or pitcher_name == "TBD":
                return feat
            name_col = next(
                (c for c in ps.columns if c in ("name", "player_name", "name_display_first_last")),
                None,
            )
            if name_col is None:
                return feat
            match = ps[ps[name_col].str.contains(pitcher_name.split()[-1], case=False, na=False)]
            if match.empty:
                return feat
            row = match.iloc[0]
            ip = float(row.get("ip", 1) or 1)
            so = float(row.get("so", row.get("k", 0)) or 0)
            bb = float(row.get("bb", 0) or 0)
            h = float(row.get("h", 0) or 0)
            feat[f"{prefix}_era"] = float(row.get("era", 4.50) or 4.50)
            feat[f"{prefix}_whip"] = (h + bb) / max(ip, 1)
            feat[f"{prefix}_k_per_9"] = so / max(ip, 1) * 9
            hand = str(row.get("throws", row.get("hand", row.get("p", "R"))) or "R")
            feat[f"{prefix}_hand"] = 1.0 if hand == "R" else 0.0
            return feat

        away_feat = _team_features(away_abbrev, "away")
        home_feat = _team_features(home_abbrev, "home")
        away_sp_feat = _pitcher_features(game.get("away_probable_pitcher", ""), "away_sp")
        home_sp_feat = _pitcher_features(game.get("home_probable_pitcher", ""), "home_sp")

        row: dict[str, Any] = {
            "game_id": game.get("game_id"),
            "date": game.get("date"),
            "away_team": away_name,
            "home_team": home_name,
            "away_abbrev": away_abbrev,
            "home_abbrev": home_abbrev,
            "away_sp": game.get("away_probable_pitcher", "TBD"),
            "home_sp": game.get("home_probable_pitcher", "TBD"),
            "venue": game.get("venue", ""),
            "park_factor_runs": _park_factor(home_abbrev, "runs"),
            "park_factor_hr": _park_factor(home_abbrev, "hr"),
            "park_factor_hits": _park_factor(home_abbrev, "hits"),
        }
        row.update(away_feat)
        row.update(home_feat)
        row.update(away_sp_feat)
        row.update(home_sp_feat)
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Today's batter features (Home Runs + Hits)
# ---------------------------------------------------------------------------

def build_today_batter_features(
    lineup: list[dict[str, Any]],
    batting_stats: pd.DataFrame,
    statcast: pd.DataFrame | None = None,
    opp_pitcher_stats: dict[str, Any] | None = None,
    park_factor_hr: float = 1.0,
    park_factor_hits: float = 1.0,
) -> pd.DataFrame:
    """Build features for today's batters.

    Parameters
    ----------
    lineup:
        List of dicts with keys ``player_name``, ``player_id``, ``hand``.
    batting_stats:
        Season batting stats DataFrame.
    statcast:
        Optional combined statcast DataFrame for recent batter data.
    opp_pitcher_stats:
        Dict of opposing pitcher stats (era, whip, k_per_9, hand).
    park_factor_hr:
        Home run park factor for today's venue.
    park_factor_hits:
        Hits park factor for today's venue.

    Returns
    -------
    DataFrame with one row per batter.
    """
    if not lineup:
        return pd.DataFrame()

    opp = opp_pitcher_stats or {}
    bs = batting_stats.copy() if not batting_stats.empty else pd.DataFrame()
    if not bs.empty:
        bs.columns = [c.lower().replace(" ", "_").replace("%", "_pct") for c in bs.columns]

    rows: list[dict[str, Any]] = []

    for batter in lineup:
        name = batter.get("player_name", "")
        player_id = batter.get("player_id", 0)
        hand = batter.get("hand", "R")

        # Defaults
        avg = 0.250
        ops = 0.720
        iso = 0.150
        k_rate = 0.220
        bb_rate = 0.085
        hr_per_game = 0.040
        hits_per_game = 0.850
        barrel_rate = 0.07
        avg_exit_velo = 88.0
        avg_launch_angle = 12.0
        hard_hit_rate = 0.35

        if not bs.empty:
            name_col = next(
                (c for c in bs.columns if c in ("name", "player_name", "name_display_first_last")),
                None,
            )
            if name_col:
                match = bs[bs[name_col].str.contains(name.split()[-1] if name else "", case=False, na=False)]
                if not match.empty:
                    p = match.iloc[0]
                    pa = float(p.get("pa", p.get("ab", 1)) or 1)
                    ab = float(p.get("ab", pa) or pa)
                    hits = float(p.get("h", 0) or 0)
                    hr = float(p.get("hr", 0) or 0)
                    bb = float(p.get("bb", 0) or 0)
                    so = float(p.get("so", p.get("k", 0)) or 0)
                    doubles = float(p.get("2b", 0) or 0)
                    triples = float(p.get("3b", 0) or 0)
                    g = float(p.get("g", 1) or 1)
                    avg = hits / max(ab, 1)
                    slg = (hits + doubles + 2 * triples + 3 * hr) / max(ab, 1)
                    obp = (hits + bb) / max(pa, 1)
                    ops = obp + slg
                    iso = slg - avg
                    k_rate = so / max(pa, 1)
                    bb_rate = bb / max(pa, 1)
                    hr_per_game = hr / max(g, 1)
                    hits_per_game = hits / max(g, 1)

        if statcast is not None and not statcast.empty:
            sc = statcast.copy()
            sc.columns = [c.lower() for c in sc.columns]
            if "launch_speed" in sc.columns:
                avg_exit_velo = float(sc["launch_speed"].dropna().mean() or 88.0)
                hard_hit_rate = float((sc["launch_speed"] >= 95).mean() or 0.35)
            if "launch_angle" in sc.columns:
                avg_launch_angle = float(sc["launch_angle"].dropna().mean() or 12.0)
            if "barrel" in sc.columns:
                barrel_rate = float((sc["barrel"] == 1).mean() or 0.07)

        # Platoon advantage: LHB vs RHP → advantage
        opp_hand = str(opp.get("hand", "R"))
        platoon = 1.0 if (hand == "L" and opp_hand == "R") or (hand == "R" and opp_hand == "L") else 0.0

        row: dict[str, Any] = {
            "player_name": name,
            "player_id": player_id,
            "hand": hand,
            # HR model features
            "hr_per_game": hr_per_game,
            "iso": iso,
            "barrel_rate": barrel_rate,
            "avg_exit_velo": avg_exit_velo,
            "avg_launch_angle": avg_launch_angle,
            "hard_hit_rate": hard_hit_rate,
            "ops": ops,
            # Hits model features
            "hits_per_game": hits_per_game,
            "avg": avg,
            "k_rate": k_rate,
            "bb_rate": bb_rate,
            # Matchup
            "opp_pitcher_era": float(opp.get("era", 4.50)),
            "opp_pitcher_whip": float(opp.get("whip", 1.30)),
            "opp_pitcher_k_per_9": float(opp.get("k_per_9", 8.0)),
            "platoon": platoon,
            # Context
            "park_factor_hr": park_factor_hr,
            "park_factor_hits": park_factor_hits,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Today's pitcher features (Strikeouts)
# ---------------------------------------------------------------------------

def build_today_pitcher_features(
    pitchers: list[dict[str, Any]],
    pitching_stats: pd.DataFrame,
    statcast: pd.DataFrame | None = None,
    opp_team_batting: dict[str, Any] | None = None,
    rest_days: int = 5,
) -> pd.DataFrame:
    """Build features for today's starting pitchers.

    Parameters
    ----------
    pitchers:
        List of dicts with keys ``player_name``, ``player_id``, ``hand``.
    pitching_stats:
        Season pitching stats DataFrame.
    statcast:
        Optional statcast DataFrame for recent pitcher data.
    opp_team_batting:
        Dict of opposing team batting stats (k_rate, ops).
    rest_days:
        Days since last start.

    Returns
    -------
    DataFrame with one row per pitcher.
    """
    if not pitchers:
        return pd.DataFrame()

    opp = opp_team_batting or {}
    ps = pitching_stats.copy() if not pitching_stats.empty else pd.DataFrame()
    if not ps.empty:
        ps.columns = [c.lower().replace(" ", "_").replace("%", "_pct") for c in ps.columns]

    rows: list[dict[str, Any]] = []

    for pitcher in pitchers:
        name = pitcher.get("player_name", "")
        player_id = pitcher.get("player_id", 0)
        hand = pitcher.get("hand", "R")

        # Defaults
        k_per_game = 6.0
        k_per_9 = 8.0
        k_rate = 0.24
        ip_per_start = 5.5
        era = 4.50
        whip = 1.30
        whiff_rate = 0.25
        chase_rate = 0.28
        avg_fastball_velo = 93.0

        if not ps.empty:
            name_col = next(
                (c for c in ps.columns if c in ("name", "player_name", "name_display_first_last")),
                None,
            )
            if name_col:
                match = ps[ps[name_col].str.contains(name.split()[-1] if name else "", case=False, na=False)]
                if not match.empty:
                    p = match.iloc[0]
                    ip = float(p.get("ip", 1) or 1)
                    gs = float(p.get("gs", 1) or 1)
                    so = float(p.get("so", p.get("k", 0)) or 0)
                    bb = float(p.get("bb", 0) or 0)
                    h = float(p.get("h", 0) or 0)
                    era = float(p.get("era", 4.50) or 4.50)
                    whip = (h + bb) / max(ip, 1)
                    k_per_9 = so / max(ip, 1) * 9
                    k_rate = so / max(so + bb + h, 1)
                    k_per_game = so / max(gs, 1)
                    ip_per_start = ip / max(gs, 1)

        if statcast is not None and not statcast.empty:
            sc = statcast.copy()
            sc.columns = [c.lower() for c in sc.columns]
            if "release_speed" in sc.columns:
                avg_fastball_velo = float(sc["release_speed"].dropna().mean() or 93.0)
            if "description" in sc.columns:
                swings = sc[sc["description"].str.contains("swing|foul|hit", case=False, na=False)]
                whiffs_sc = sc[sc["description"].str.contains("swinging_strike", case=False, na=False)]
                if len(swings) > 0:
                    whiff_rate = len(whiffs_sc) / len(swings)

        row: dict[str, Any] = {
            "player_name": name,
            "player_id": player_id,
            "hand": hand,
            "era": era,
            "whip": whip,
            "k_per_game": k_per_game,
            "k_per_9": k_per_9,
            "k_rate": k_rate,
            "ip_per_start": ip_per_start,
            "whiff_rate": whiff_rate,
            "chase_rate": chase_rate,
            "avg_fastball_velo": avg_fastball_velo,
            "opp_team_k_rate": float(opp.get("k_rate", 0.22)),
            "opp_team_ops": float(opp.get("ops", 0.720)),
            "rest_days": float(rest_days),
            "pitcher_hand": 1.0 if hand == "R" else 0.0,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).reset_index(drop=True)
