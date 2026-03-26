"""Data loading functions with pybaseball + statsapi and CSV cache."""

import os
import datetime
from typing import Any

import pandas as pd

import pybaseball
import statsapi

# Fix FanGraphs 403 error: identify requests as a real browser
pybaseball.cache.DEFAULT_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

from config import CACHE_DIR, SEASONS

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.csv")


def _is_fresh_today(path: str) -> bool:
    """Return True if the file exists and was last modified today."""
    if not os.path.exists(path):
        return False
    mtime = datetime.date.fromtimestamp(os.path.getmtime(path))
    return mtime == datetime.date.today()


def load_team_game_logs(team: str, season: int) -> pd.DataFrame:
    """Download (or load from cache) per-game logs for *team* in *season*."""
    import pybaseball

    cache_file = _cache_path(f"team_game_logs_{team}_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    try:
        df = pybaseball.team_game_logs(season, team)
        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[WARN] Could not load team_game_logs({team}, {season}): {exc}")
        return pd.DataFrame()


def load_batting_stats(season: int) -> pd.DataFrame:
    """Download (or load from cache) season batting stats."""
    import pybaseball

    cache_file = _cache_path(f"batting_stats_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    try:
        df = pybaseball.batting_stats(season, qual=50)
        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[WARN] Could not load batting_stats({season}): {exc}")
        return pd.DataFrame()


def load_pitching_stats(season: int) -> pd.DataFrame:
    """Download (or load from cache) season pitching stats."""
    import pybaseball

    cache_file = _cache_path(f"pitching_stats_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    try:
        df = pybaseball.pitching_stats(season, qual=20)
        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[WARN] Could not load pitching_stats({season}): {exc}")
        return pd.DataFrame()


def load_statcast_batter(player_id: int, season: int) -> pd.DataFrame:
    """Download (or load from cache) statcast data for a single batter."""
    import pybaseball

    cache_file = _cache_path(f"statcast_batter_{player_id}_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    start = f"{season}-03-01"
    end = f"{season}-11-30"
    try:
        df = pybaseball.statcast_batter(start, end, player_id=player_id)
        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[WARN] Could not load statcast_batter({player_id}, {season}): {exc}")
        return pd.DataFrame()


def load_statcast_pitcher(player_id: int, season: int) -> pd.DataFrame:
    """Download (or load from cache) statcast data for a single pitcher."""
    import pybaseball

    cache_file = _cache_path(f"statcast_pitcher_{player_id}_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    start = f"{season}-03-01"
    end = f"{season}-11-30"
    try:
        df = pybaseball.statcast_pitcher(start, end, player_id=player_id)
        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)
        return df if df is not None else pd.DataFrame()
    except Exception as exc:
        print(f"[WARN] Could not load statcast_pitcher({player_id}, {season}): {exc}")
        return pd.DataFrame()


def load_schedule_today(date: datetime.date | None = None) -> list[dict[str, Any]]:
    """Return today's (or *date*'s) MLB schedule via statsapi."""
    target = date or datetime.date.today()
    date_str = target.strftime("%Y-%m-%d")
    cache_file = _cache_path(f"schedule_{date_str}")

    # Use fresh cache only for today
    if target == datetime.date.today() and _is_fresh_today(cache_file):
        df = pd.read_csv(cache_file)
        return df.to_dict("records")

    if os.path.exists(cache_file) and target != datetime.date.today():
        df = pd.read_csv(cache_file)
        return df.to_dict("records")

    try:
        games_raw = statsapi.schedule(date=date_str)
        if not games_raw:
            print(f"[INFO] No games found for {date_str}")
            return []
        games: list[dict[str, Any]] = []
        for g in games_raw:
            games.append(
                {
                    "game_id": g.get("game_id"),
                    "date": date_str,
                    "away_team": g.get("away_name", ""),
                    "home_team": g.get("home_name", ""),
                    "away_probable_pitcher": g.get("away_probable_pitcher", "TBD"),
                    "home_probable_pitcher": g.get("home_probable_pitcher", "TBD"),
                    "venue": g.get("venue_name", ""),
                    "game_time": g.get("game_datetime", ""),
                    "status": g.get("status", ""),
                }
            )
        df = pd.DataFrame(games)
        df.to_csv(cache_file, index=False)
        return games
    except Exception as exc:
        print(f"[WARN] Could not load schedule for {date_str}: {exc}")
        return []


def load_game_results(season: int) -> pd.DataFrame:
    """Load historical game results (win/loss, runs) for all teams in *season*."""
    import pybaseball
    from config import TEAM_ABBREV_MAP

    cache_file = _cache_path(f"game_results_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    all_records: list[pd.DataFrame] = []
    teams = list(TEAM_ABBREV_MAP.values())
    seen: set[str] = set()
    unique_teams = [t for t in teams if not (t in seen or seen.add(t))]

    for team in unique_teams:
        try:
            df = pybaseball.schedule_and_record(season, team)
            if df is not None and not df.empty:
                df["team"] = team
                all_records.append(df)
        except Exception as exc:
            print(f"[WARN] schedule_and_record({season}, {team}): {exc}")

    if not all_records:
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    combined.to_csv(cache_file, index=False)
    return combined


def load_all_batting_stats(seasons: list[int] | None = None) -> pd.DataFrame:
    """Concatenate batting stats across multiple seasons."""
    seasons = seasons or SEASONS
    frames = [load_batting_stats(s) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "Season" not in combined.columns and "season" not in combined.columns:
        pass
    return combined


def load_all_pitching_stats(seasons: list[int] | None = None) -> pd.DataFrame:
    """Concatenate pitching stats across multiple seasons."""
    seasons = seasons or SEASONS
    frames = [load_pitching_stats(s) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)