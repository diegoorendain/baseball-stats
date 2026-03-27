"""Data loading functions with pybaseball (Statcast) + statsapi and CSV cache."""

import os
import datetime
from typing import Any

import numpy as np
import pandas as pd

import statsapi

from config import CACHE_DIR, SEASONS, TEAM_ABBREV_MAP, OPENWEATHER_API_KEY, STADIUM_COORDINATES

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# MLB Stats API team ID map (abbreviation → statsapi integer team ID)
# ---------------------------------------------------------------------------
_STATSAPI_TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CHW": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KCR": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SDP": 135, "SEA": 136, "SFG": 137,
    "STL": 138, "TBR": 139, "TEX": 140, "TOR": 141, "WSN": 120,
}

# Reverse map: statsapi full team name → abbreviation
# Forward map: full team name → abbreviation + identity entries for abbreviations
_FULL_NAME_TO_ABBREV: dict[str, str] = dict(TEAM_ABBREV_MAP)
_FULL_NAME_TO_ABBREV.update({v: v for v in TEAM_ABBREV_MAP.values()})


def _name_to_abbrev(name: str) -> str:
    """Convert a full team name or abbreviation to a 2-3 letter abbreviation."""
    return _FULL_NAME_TO_ABBREV.get(name, TEAM_ABBREV_MAP.get(name, name[:3].upper() if len(name) >= 3 else name.upper()))


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.csv")


def _is_fresh_today(path: str) -> bool:
    """Return True if the file exists and was last modified today."""
    if not os.path.exists(path):
        return False
    mtime = datetime.date.fromtimestamp(os.path.getmtime(path))
    return mtime == datetime.date.today()


# ---------------------------------------------------------------------------
# Statcast raw data downloader (season-level, with chunking)
# ---------------------------------------------------------------------------

def _get_season_statcast(season: int) -> pd.DataFrame:
    """Download (or load from cache) full-season Statcast data in ~2-month chunks.

    The raw data is cached as ``statcast_raw_{season}.csv`` so subsequent calls
    for batting/pitching stats in the same season reuse the same download.
    """
    import pybaseball

    cache_file = _cache_path(f"statcast_raw_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, low_memory=False)

    # ~2-month date windows covering the MLB regular season
    chunks = [
        (f"{season}-03-20", f"{season}-04-30"),
        (f"{season}-05-01", f"{season}-06-30"),
        (f"{season}-07-01", f"{season}-08-31"),
        (f"{season}-09-01", f"{season}-10-15"),
    ]

    frames: list[pd.DataFrame] = []
    # Pre-compute labels to avoid repeated datetime parsing
    _CHUNK_LABELS = [("Mar", "Apr"), ("May", "Jun"), ("Jul", "Aug"), ("Sep", "Oct")]
    for (start_dt, end_dt), (start_lbl, end_lbl) in zip(chunks, _CHUNK_LABELS):
        print(f"[INFO] Downloading Statcast data for {season} ({start_lbl}-{end_lbl})...")
        try:
            chunk = pybaseball.statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
            if chunk is not None and not chunk.empty:
                frames.append(chunk)
        except Exception as exc:
            print(f"[WARN] Could not download Statcast {season} ({start_dt} to {end_dt}): {exc}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(cache_file, index=False)
    return df


# ---------------------------------------------------------------------------
# Batting stats — aggregated from Statcast
# ---------------------------------------------------------------------------

def load_batting_stats(season: int) -> pd.DataFrame:
    """Download (or load from cache) season batting stats aggregated from Statcast.

    Uses ``pybaseball.statcast()`` (Baseball Savant) instead of FanGraphs.
    Filters to batters with at least 50 plate appearances.
    """
    import pybaseball

    cache_file = _cache_path(f"batting_stats_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    print(f"[INFO] Aggregating batting stats from Statcast for {season}...")
    sc = _get_season_statcast(season)
    if sc.empty:
        return pd.DataFrame()

    # Keep only rows that represent a completed plate appearance
    sc_pa = sc[sc["events"].notna()].copy()
    if sc_pa.empty:
        return pd.DataFrame()

    # Event classification helpers
    _HIT_EVENTS = {"single", "double", "triple", "home_run"}
    _NON_AB_EVENTS = {
        "walk", "intentional_walk", "hit_by_pitch",
        "sac_fly", "sac_bunt", "catcher_interf",
        "sac_fly_double_play",
    }

    sc_pa["is_ab"] = ~sc_pa["events"].isin(_NON_AB_EVENTS)
    sc_pa["is_hit"] = sc_pa["events"].isin(_HIT_EVENTS)
    sc_pa["is_hr"] = sc_pa["events"] == "home_run"
    sc_pa["is_bb"] = sc_pa["events"].isin({"walk", "intentional_walk"})
    sc_pa["is_hbp"] = sc_pa["events"] == "hit_by_pitch"
    sc_pa["is_so"] = sc_pa["events"] == "strikeout"
    sc_pa["is_double"] = sc_pa["events"] == "double"
    sc_pa["is_triple"] = sc_pa["events"] == "triple"

    # Determine batter's team from inning half (Top = away batting, Bot = home batting)
    if "inning_topbot" in sc_pa.columns and "home_team" in sc_pa.columns:
        sc_pa["batter_team"] = np.where(
            sc_pa["inning_topbot"] == "Bot",
            sc_pa["home_team"],
            sc_pa["away_team"],
        )
    else:
        sc_pa["batter_team"] = ""

    # Aggregate by batter MLBAM ID
    agg = sc_pa.groupby("batter", as_index=False).agg(
        pa=("events", "count"),
        ab=("is_ab", "sum"),
        h=("is_hit", "sum"),
        hr=("is_hr", "sum"),
        bb=("is_bb", "sum"),
        hbp=("is_hbp", "sum"),
        so=("is_so", "sum"),
        doubles=("is_double", "sum"),
        triples=("is_triple", "sum"),
        g=("game_pk", "nunique"),
        batter_team=("batter_team", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
    )
    agg.rename(columns={"doubles": "2b", "triples": "3b"}, inplace=True)

    # Convert to float for safe arithmetic
    for col in ("pa", "ab", "h", "hr", "bb", "hbp", "so", "2b", "3b"):
        agg[col] = agg[col].astype(float)

    # Filter by minimum 50 plate appearances
    agg = agg[agg["pa"] >= 50].copy()
    if agg.empty:
        return pd.DataFrame()

    # Derived batting stats
    agg["avg"] = agg["h"] / agg["ab"].clip(lower=1)
    agg["obp"] = (agg["h"] + agg["bb"] + agg["hbp"]) / agg["pa"].clip(lower=1)
    singles = agg["h"] - agg["hr"] - agg["2b"] - agg["3b"]
    agg["slg"] = (singles + 2 * agg["2b"] + 3 * agg["3b"] + 4 * agg["hr"]) / agg["ab"].clip(lower=1)
    agg["ops"] = agg["obp"] + agg["slg"]
    agg["iso"] = agg["slg"] - agg["avg"]
    # Simplified wOBA (2024 linear weights)
    agg["woba"] = (
        0.690 * agg["bb"]
        + 0.720 * agg["hbp"]
        + 0.890 * singles
        + 1.270 * agg["2b"]
        + 1.620 * agg["3b"]
        + 2.100 * agg["hr"]
    ) / agg["pa"].clip(lower=1)

    # Statcast batted-ball metrics
    batted = sc[sc["launch_speed"].notna() & sc["batter"].notna()].copy()
    if not batted.empty:
        batted_agg = batted.groupby("batter", as_index=False).agg(
            avg_exit_velo=("launch_speed", "mean"),
            avg_launch_angle=("launch_angle", "mean"),
            batted_balls=("launch_speed", "count"),
            hard_hit_count=("launch_speed", lambda x: (x >= 95).sum()),
        )
        if "barrel" in batted.columns:
            barrel_agg = batted.groupby("batter", as_index=False).agg(
                barrel_sum=("barrel", lambda x: (x == 1).sum()),
                barrel_denom=("barrel", "count"),
            )
            batted_agg = batted_agg.merge(barrel_agg, on="batter", how="left")
            batted_agg["barrel_rate"] = (
                batted_agg["barrel_sum"] / batted_agg["barrel_denom"].clip(lower=1)
            )
        else:
            batted_agg["barrel_rate"] = 0.07
        batted_agg["hard_hit_rate"] = (
            batted_agg["hard_hit_count"] / batted_agg["batted_balls"].clip(lower=1)
        )
        agg = agg.merge(
            batted_agg[["batter", "avg_exit_velo", "avg_launch_angle", "barrel_rate", "hard_hit_rate"]],
            on="batter",
            how="left",
        )
    else:
        agg["avg_exit_velo"] = 88.0
        agg["avg_launch_angle"] = 12.0
        agg["barrel_rate"] = 0.07
        agg["hard_hit_rate"] = 0.35

    # Fill any remaining NaN statcast values with league-average defaults
    agg["avg_exit_velo"] = agg["avg_exit_velo"].fillna(88.0)
    agg["avg_launch_angle"] = agg["avg_launch_angle"].fillna(12.0)
    agg["barrel_rate"] = agg["barrel_rate"].fillna(0.07)
    agg["hard_hit_rate"] = agg["hard_hit_rate"].fillna(0.35)

    # Look up player names via pybaseball reverse lookup (batch to avoid timeouts)
    try:
        batter_ids = agg["batter"].dropna().astype(int).tolist()
        _BATCH_SIZE = 200
        lookup_frames: list[pd.DataFrame] = []
        for i in range(0, len(batter_ids), _BATCH_SIZE):
            batch = batter_ids[i : i + _BATCH_SIZE]
            batch_result = pybaseball.playerid_reverse_lookup(batch, key_type="mlbam")
            if batch_result is not None and not batch_result.empty:
                lookup_frames.append(batch_result)
        if lookup_frames:
            lookup = pd.concat(lookup_frames, ignore_index=True)
            lookup = lookup[["key_mlbam", "name_first", "name_last"]].copy()
            lookup["player_name"] = lookup["name_first"] + " " + lookup["name_last"]
            lookup.rename(columns={"key_mlbam": "batter"}, inplace=True)
            agg = agg.merge(lookup[["batter", "player_name"]], on="batter", how="left")
        else:
            agg["player_name"] = agg["batter"].astype(str)
    except Exception:
        agg["player_name"] = agg["batter"].astype(str)

    agg["player_name"] = agg["player_name"].fillna(agg["batter"].astype(str))

    # Metadata columns
    agg["season"] = season
    agg["player_id"] = agg["batter"].astype(int)
    agg["team"] = agg["batter_team"]

    agg.to_csv(cache_file, index=False)
    return agg


# ---------------------------------------------------------------------------
# Pitching stats — aggregated from Statcast
# ---------------------------------------------------------------------------

def load_pitching_stats(season: int) -> pd.DataFrame:
    """Download (or load from cache) season pitching stats aggregated from Statcast.

    Uses ``pybaseball.statcast()`` (Baseball Savant) instead of FanGraphs.
    Filters to pitchers with at least 20 innings pitched.
    """
    cache_file = _cache_path(f"pitching_stats_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    print(f"[INFO] Aggregating pitching stats from Statcast for {season}...")
    sc = _get_season_statcast(season)
    if sc.empty:
        return pd.DataFrame()

    # Keep only rows that represent a completed plate appearance
    sc_pa = sc[sc["events"].notna()].copy()
    if sc_pa.empty:
        return pd.DataFrame()

    # Determine pitcher's team (opposite side from batter)
    if "inning_topbot" in sc_pa.columns and "home_team" in sc_pa.columns:
        sc_pa["pitcher_team"] = np.where(
            sc_pa["inning_topbot"] == "Bot",
            sc_pa["away_team"],
            sc_pa["home_team"],
        )
    else:
        sc_pa["pitcher_team"] = ""

    # Pitcher handedness (p_throws column in statcast)
    if "p_throws" in sc_pa.columns:
        sc_pa["throws"] = sc_pa["p_throws"].fillna("R")
    else:
        sc_pa["throws"] = "R"

    # Event classification
    _HIT_EVENTS = {"single", "double", "triple", "home_run"}
    sc_pa["is_hit"] = sc_pa["events"].isin(_HIT_EVENTS)
    sc_pa["is_k"] = sc_pa["events"] == "strikeout"
    sc_pa["is_bb"] = sc_pa["events"].isin({"walk", "intentional_walk"})
    sc_pa["is_hr"] = sc_pa["events"] == "home_run"
    sc_pa["is_hbp"] = sc_pa["events"] == "hit_by_pitch"

    # Outs recorded per plate appearance (GDPs = 2 outs, triple plays = 3)
    _OUTS_MAP = {
        "strikeout": 1, "field_out": 1, "force_out": 1, "sac_fly": 1,
        "sac_bunt": 1, "fielders_choice_out": 1, "other_out": 1,
        "grounded_into_double_play": 2, "double_play": 2,
        "strikeout_double_play": 2, "sac_fly_double_play": 2,
        "triple_play": 3,
    }
    sc_pa["outs_recorded"] = sc_pa["events"].map(_OUTS_MAP).fillna(0)

    # Aggregate by pitcher ID + name (player_name in statcast is the pitcher's name)
    name_col = "player_name" if "player_name" in sc_pa.columns else "pitcher"
    agg = sc_pa.groupby(["pitcher", name_col], as_index=False).agg(
        bf=("events", "count"),
        h=("is_hit", "sum"),
        hr=("is_hr", "sum"),
        bb=("is_bb", "sum"),
        hbp=("is_hbp", "sum"),
        so=("is_k", "sum"),
        outs=("outs_recorded", "sum"),
        games=("game_pk", "nunique"),
        pitcher_team=("pitcher_team", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
        throws=("throws", lambda x: x.mode().iloc[0] if not x.mode().empty else "R"),
    )
    agg.rename(columns={name_col: "player_name"}, inplace=True)

    # Game starts: appeared in inning 1
    if "inning" in sc_pa.columns:
        starts = (
            sc_pa[sc_pa["inning"] == 1]
            .groupby("pitcher", as_index=False)
            .agg(gs=("game_pk", "nunique"))
        )
        agg = agg.merge(starts, on="pitcher", how="left")
    else:
        agg["gs"] = 1

    agg["gs"] = agg["gs"].fillna(0)

    # Compute innings pitched and filter by minimum 20 IP
    agg["ip"] = agg["outs"] / 3.0
    agg = agg[agg["ip"] >= 20.0].copy()
    if agg.empty:
        return pd.DataFrame()

    # Derived pitching stats
    agg["whip"] = (agg["h"] + agg["bb"]) / agg["ip"].clip(lower=0.333)
    agg["k_per_9"] = agg["so"] / agg["ip"].clip(lower=0.333) * 9
    agg["bb_per_9"] = agg["bb"] / agg["ip"].clip(lower=0.333) * 9
    agg["k_rate"] = agg["so"] / agg["bf"].clip(lower=1)
    agg["bb_rate"] = agg["bb"] / agg["bf"].clip(lower=1)
    agg["ip_per_start"] = agg["ip"] / agg["gs"].clip(lower=1)

    # FIP-based ERA approximation (FIP constant ≈ 3.10 for recent seasons)
    _FIP_CONSTANT = 3.10
    agg["era"] = (
        (13 * agg["hr"] + 3 * (agg["bb"] + agg["hbp"]) - 2 * agg["so"])
        / agg["ip"].clip(lower=0.333)
        + _FIP_CONSTANT
    ).clip(lower=0.0, upper=10.0)

    # Whiff rate from pitch-level data
    _SWING_DESCS = {
        "swinging_strike", "foul", "foul_tip", "hit_into_play",
        "hit_into_play_no_out", "hit_into_play_score", "foul_bunt",
        "bunt_foul_tip", "swinging_strike_blocked", "missed_bunt",
        "swinging_pitchout",
    }
    _WHIFF_DESCS = {
        "swinging_strike", "swinging_strike_blocked", "swinging_pitchout",
    }
    if "description" in sc.columns:
        sc_pitches = sc[sc["pitcher"].notna()].copy()
        sc_pitches["is_swing"] = sc_pitches["description"].isin(_SWING_DESCS)
        sc_pitches["is_whiff"] = sc_pitches["description"].isin(_WHIFF_DESCS)
        whiff_agg = sc_pitches.groupby("pitcher", as_index=False).agg(
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
        )
        whiff_agg["whiff_rate"] = whiff_agg["whiffs"] / whiff_agg["swings"].clip(lower=1)
        agg = agg.merge(whiff_agg[["pitcher", "whiff_rate"]], on="pitcher", how="left")
    else:
        agg["whiff_rate"] = 0.25

    # Chase rate (swings on out-of-zone pitches / out-of-zone pitches)
    if "zone" in sc.columns and "description" in sc.columns:
        sc_zone = sc[sc["pitcher"].notna() & sc["zone"].notna()].copy()
        sc_zone["is_out_of_zone"] = sc_zone["zone"] > 9
        sc_zone["is_swing"] = sc_zone["description"].isin(_SWING_DESCS)
        sc_ooze = sc_zone[sc_zone["is_out_of_zone"]].copy()
        chase_agg = sc_ooze.groupby("pitcher", as_index=False).agg(
            ooze_pitches=("zone", "count"),
            ooze_swings=("is_swing", "sum"),
        )
        chase_agg["chase_rate"] = chase_agg["ooze_swings"] / chase_agg["ooze_pitches"].clip(lower=1)
        agg = agg.merge(chase_agg[["pitcher", "chase_rate"]], on="pitcher", how="left")
    else:
        agg["chase_rate"] = 0.28

    # Average fastball velocity
    if "release_speed" in sc.columns and "pitch_type" in sc.columns:
        _FB_TYPES = {"FF", "FA", "SI", "FT", "FC"}
        sc_fb = sc[
            sc["pitch_type"].isin(_FB_TYPES)
            & sc["pitcher"].notna()
            & sc["release_speed"].notna()
        ].copy()
        if not sc_fb.empty:
            fb_agg = sc_fb.groupby("pitcher", as_index=False).agg(
                avg_fastball_velo=("release_speed", "mean")
            )
            agg = agg.merge(fb_agg, on="pitcher", how="left")
        else:
            agg["avg_fastball_velo"] = 93.0
    else:
        agg["avg_fastball_velo"] = 93.0

    # Fill missing statcast metrics with league-average defaults
    agg["whiff_rate"] = agg["whiff_rate"].fillna(0.25)
    agg["chase_rate"] = agg["chase_rate"].fillna(0.28)
    agg["avg_fastball_velo"] = agg["avg_fastball_velo"].fillna(93.0)

    # Metadata
    agg["season"] = season
    agg["player_id"] = agg["pitcher"].astype(int)
    agg["team"] = agg["pitcher_team"]

    agg.to_csv(cache_file, index=False)
    return agg


# ---------------------------------------------------------------------------
# Team game logs — via MLB Stats API
# ---------------------------------------------------------------------------

def load_team_game_logs(team: str, season: int) -> pd.DataFrame:
    """Download (or load from cache) per-game logs for *team* in *season*.

    Uses ``statsapi.schedule()`` (MLB Stats API) instead of FanGraphs.
    Returns a DataFrame with columns: date, r, ra, result, opp, game_id.
    """
    cache_file = _cache_path(f"team_game_logs_{team}_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    team_id = _STATSAPI_TEAM_IDS.get(team)
    if team_id is None:
        print(f"[WARN] No statsapi team ID found for {team}")
        return pd.DataFrame()

    start_date = f"{season}-03-20"
    end_date = f"{season}-10-05"

    try:
        games_raw = statsapi.schedule(start_date=start_date, end_date=end_date, team=team_id)
        if not games_raw:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for g in games_raw:
            if g.get("status") != "Final":
                continue
            away_score = g.get("away_score", 0) or 0
            home_score = g.get("home_score", 0) or 0
            home_abbrev = _name_to_abbrev(g.get("home_name", ""))
            away_abbrev = _name_to_abbrev(g.get("away_name", ""))
            if home_abbrev == team:
                runs, ra, opp = home_score, away_score, away_abbrev
            else:
                runs, ra, opp = away_score, home_score, home_abbrev
            rows.append(
                {
                    "date": g.get("game_date", ""),
                    "r": runs,
                    "ra": ra,
                    "result": "W" if runs > ra else "L",
                    "opp": opp,
                    "game_id": g.get("game_id", ""),
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        return df
    except Exception as exc:
        print(f"[WARN] Could not load team_game_logs({team}, {season}): {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Game results — via MLB Stats API
# ---------------------------------------------------------------------------

def load_game_results(season: int) -> pd.DataFrame:
    """Load all completed game results for *season* using MLB Stats API.

    Uses ``statsapi.schedule()`` instead of FanGraphs.
    Returns a DataFrame with columns: game_id, date, away_team, home_team,
    away_score, home_score, winning_team, losing_team.
    """
    cache_file = _cache_path(f"game_results_{season}")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    start_date = f"{season}-03-20"
    end_date = f"{season}-10-05"

    try:
        games_raw = statsapi.schedule(start_date=start_date, end_date=end_date)
        if not games_raw:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for g in games_raw:
            if g.get("status") != "Final":
                continue
            rows.append(
                {
                    "game_id": g.get("game_id", ""),
                    "date": g.get("game_date", ""),
                    "away_team": _name_to_abbrev(g.get("away_name", "")),
                    "home_team": _name_to_abbrev(g.get("home_name", "")),
                    "away_score": g.get("away_score", 0) or 0,
                    "home_score": g.get("home_score", 0) or 0,
                    "winning_team": _name_to_abbrev(g.get("winning_team", "")),
                    "losing_team": _name_to_abbrev(g.get("losing_team", "")),
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        return df
    except Exception as exc:
        print(f"[WARN] Could not load game_results({season}): {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Individual player Statcast data (unchanged — already uses Baseball Savant)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Umpire data — via MLB Stats API boxscore
# ---------------------------------------------------------------------------

_UMPIRE_CACHE: dict[int, dict[str, Any]] = {}


def load_umpire_data(game_id: int) -> dict[str, Any]:
    """Get umpire assignment for *game_id* from statsapi boxscore.

    Returns a dict with ``home_plate_umpire`` (str) and ``game_id`` (int).
    Results are cached in memory.
    """
    if game_id in _UMPIRE_CACHE:
        return _UMPIRE_CACHE[game_id]

    result: dict[str, Any] = {"game_id": game_id, "home_plate_umpire": ""}
    try:
        boxscore = statsapi.boxscore_data(game_id)
        officials = boxscore.get("officials", []) if isinstance(boxscore, dict) else []
        for official in officials:
            if isinstance(official, dict):
                title = official.get("officialType", "")
                if "Home Plate" in title or "home plate" in title.lower():
                    name_info = official.get("official", {})
                    result["home_plate_umpire"] = name_info.get("fullName", "")
                    break
    except Exception as exc:
        print(f"[WARN] Could not load umpire data for game {game_id}: {exc}")

    _UMPIRE_CACHE[game_id] = result
    return result


# ---------------------------------------------------------------------------
# Weather data — via OpenWeatherMap free API
# ---------------------------------------------------------------------------

_WEATHER_CACHE: dict[str, dict[str, Any]] = {}


def load_weather_data(team: str, date: str) -> dict[str, Any]:
    """Get weather forecast/current conditions for *team*'s stadium on *date*.

    Uses OpenWeatherMap free current-weather endpoint.  Requires
    ``OPENWEATHER_API_KEY`` to be set in the environment (or config.py).

    Parameters
    ----------
    team:
        Team abbreviation (e.g. ``"NYY"``).
    date:
        Date string ``YYYY-MM-DD`` (used as cache key; current weather is
        fetched regardless of date since free tier has no forecast beyond 3h).

    Returns
    -------
    Dict with keys: temp_f, wind_speed_mph, wind_direction, humidity,
    description, is_outdoor (1/0).
    """
    from config import INDOOR_STADIUMS

    cache_key = f"{team}_{date}"
    if cache_key in _WEATHER_CACHE:
        return _WEATHER_CACHE[cache_key]

    is_outdoor = 0 if team in INDOOR_STADIUMS else 1
    default: dict[str, Any] = {
        "temp_f": 70.0,
        "wind_speed_mph": 5.0,
        "wind_direction": 0,
        "humidity": 50,
        "description": "N/A",
        "is_outdoor": is_outdoor,
    }

    if not OPENWEATHER_API_KEY:
        _WEATHER_CACHE[cache_key] = default
        return default

    coords = STADIUM_COORDINATES.get(team)
    if coords is None:
        _WEATHER_CACHE[cache_key] = default
        return default

    lat, lon = coords
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=imperial"
    )
    try:
        import requests
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            weather = data.get("weather", [{}])[0]
            main = data.get("main", {})
            wind = data.get("wind", {})
            result: dict[str, Any] = {
                "temp_f": float(main.get("temp", 70.0)),
                "wind_speed_mph": float(wind.get("speed", 5.0)),
                "wind_direction": int(wind.get("deg", 0)),
                "humidity": int(main.get("humidity", 50)),
                "description": weather.get("description", ""),
                "is_outdoor": is_outdoor,
            }
            _WEATHER_CACHE[cache_key] = result
            return result
        else:
            print(f"[WARN] OpenWeatherMap returned status {resp.status_code} for {team}.")
    except Exception as exc:
        print(f"[WARN] Could not load weather data for {team}: {exc}")

    _WEATHER_CACHE[cache_key] = default
    return default


# ---------------------------------------------------------------------------
# Team schedule context (rest days, travel, bullpen usage)
# ---------------------------------------------------------------------------

def load_team_schedule_context(
    team: str,
    date: str,
    game_logs: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate schedule-based context features for *team* on *date*.

    Parameters
    ----------
    team:
        Team abbreviation.
    date:
        Today's date string ``YYYY-MM-DD``.
    game_logs:
        DataFrame of this team's recent game log (output of
        ``load_team_game_logs``).

    Returns
    -------
    Dict with: rest_days, games_last_7, travel_flag, is_day_game.
    """
    default: dict[str, Any] = {
        "rest_days": 1,
        "games_last_7": 5,
        "travel_flag": 0,
        "is_day_game": 0,
    }

    try:
        if game_logs.empty:
            return default

        logs = game_logs.copy()
        logs.columns = [c.lower() for c in logs.columns]
        if "date" not in logs.columns:
            return default

        logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
        logs = logs.dropna(subset=["date"]).sort_values("date")

        target_dt = pd.to_datetime(date, errors="coerce")
        if pd.isna(target_dt):
            return default

        past = logs[logs["date"] < target_dt]
        if past.empty:
            return default

        last_game_date = past["date"].iloc[-1]
        rest_days = max(0, (target_dt - last_game_date).days)

        cutoff_7 = target_dt - pd.Timedelta(days=7)
        games_last_7 = int((past["date"] >= cutoff_7).sum())

        # Travel flag: opponent changed day-over-day (coarse approximation)
        travel_flag = 0
        if "opp" in past.columns and len(past) >= 2:
            last_opp = str(past["opp"].iloc[-1])
            prev_opp = str(past["opp"].iloc[-2])
            travel_flag = int(last_opp != prev_opp)

        return {
            "rest_days": int(rest_days),
            "games_last_7": int(games_last_7),
            "travel_flag": travel_flag,
            "is_day_game": 0,  # day/night requires live game feed
        }
    except Exception as exc:
        print(f"[WARN] Could not compute schedule context for {team}: {exc}")
        return default


# ---------------------------------------------------------------------------
# Bullpen usage / fatigue
# ---------------------------------------------------------------------------

def load_bullpen_usage(team: str, date: str, season: int) -> dict[str, Any]:
    """Estimate bullpen workload from Statcast data for *team* around *date*.

    Parameters
    ----------
    team:
        Team abbreviation.
    date:
        Today's date string ``YYYY-MM-DD``.
    season:
        MLB season year.

    Returns
    -------
    Dict with: bullpen_pitches_last_3d, bullpen_ip_last_3d,
    bullpen_fatigue_score (0-1, higher = more tired).
    """
    default: dict[str, Any] = {
        "bullpen_pitches_last_3d": 0,
        "bullpen_ip_last_3d": 0.0,
        "bullpen_fatigue_score": 0.5,
    }

    try:
        sc = _get_season_statcast(season)
        if sc.empty:
            return default

        sc = sc.copy()
        if "game_date" not in sc.columns:
            return default

        sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce")
        target_dt = pd.to_datetime(date, errors="coerce")
        if pd.isna(target_dt):
            return default

        cutoff = target_dt - pd.Timedelta(days=3)

        # Identify pitcher team
        if "inning_topbot" not in sc.columns or "home_team" not in sc.columns:
            return default

        sc["pitcher_team"] = np.where(
            sc["inning_topbot"] == "Bot",
            sc["away_team"],   # Bottom inning: away team pitches, home team bats
            sc["home_team"],   # Top inning: home team pitches, away team bats
        )

        # Filter: team, last 3 days, relief pitchers (not inning 1)
        mask = (
            (sc["pitcher_team"] == team)
            & (sc["game_date"] >= cutoff)
            & (sc["game_date"] < target_dt)
        )
        if "inning" in sc.columns:
            mask = mask & (sc["inning"] > 1)

        recent = sc[mask]
        if recent.empty:
            return {**default, "bullpen_fatigue_score": 0.2}

        # Estimate outs per pitch via events
        bullpen_pitches = len(recent)
        _OUTS_MAP = {
            "strikeout": 1, "field_out": 1, "force_out": 1, "sac_fly": 1,
            "sac_bunt": 1, "fielders_choice_out": 1, "other_out": 1,
            "grounded_into_double_play": 2, "double_play": 2,
            "strikeout_double_play": 2, "sac_fly_double_play": 2,
            "triple_play": 3,
        }
        if "events" in recent.columns:
            outs_recorded = recent["events"].map(_OUTS_MAP).fillna(0).sum()
        else:
            outs_recorded = bullpen_pitches * 0.33  # rough proxy

        bullpen_ip = float(outs_recorded) / 3.0

        # Normalize: ~100 pitches/day is average bullpen usage; scale 0-1
        # A team that threw 300 pitches in 3 days would score ~1.0
        fatigue = min(1.0, bullpen_pitches / 300.0)

        return {
            "bullpen_pitches_last_3d": int(bullpen_pitches),
            "bullpen_ip_last_3d": round(bullpen_ip, 1),
            "bullpen_fatigue_score": round(fatigue, 3),
        }
    except Exception as exc:
        print(f"[WARN] Could not compute bullpen usage for {team}: {exc}")
        return default