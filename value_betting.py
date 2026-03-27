"""Value betting engine: compare model probabilities vs market odds to find +EV bets."""

from __future__ import annotations

import json
import os
from typing import Any

from config import (
    BASE_DIR,
    HIGH_CONFIDENCE_EDGE,
    KELLY_FRACTION,
    MED_CONFIDENCE_EDGE,
    MIN_EDGE_PCT,
)

# Default path for manual odds file
ODDS_FILE_DEFAULT = os.path.join(BASE_DIR, "odds_today.json")


# ---------------------------------------------------------------------------
# Odds conversions
# ---------------------------------------------------------------------------

def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability (0-1).

    Parameters
    ----------
    odds:
        American odds integer (e.g. -150 or +130).

    Returns
    -------
    Implied probability as a float between 0 and 1.
    """
    try:
        odds = int(odds)
        if odds < 0:
            return (-odds) / (-odds + 100)
        return 100 / (odds + 100)
    except (ValueError, ZeroDivisionError) as exc:
        print(f"[WARN] american_to_implied: could not convert odds '{odds}': {exc}")
        return 0.5


def implied_to_american(prob: float) -> int:
    """Convert a probability (0-1) to American odds integer.

    Parameters
    ----------
    prob:
        Win probability between 0 and 1.

    Returns
    -------
    American odds integer.
    """
    try:
        prob = float(prob)
        prob = max(0.001, min(0.999, prob))
        if prob >= 0.5:
            return int(round(-(prob / (1 - prob)) * 100))
        return int(round((1 - prob) / prob * 100))
    except (ValueError, ZeroDivisionError):
        return 100


# ---------------------------------------------------------------------------
# Edge / Kelly
# ---------------------------------------------------------------------------

def calculate_edge(model_prob: float, market_implied: float) -> float:
    """Return the edge percentage (model_prob - market_implied).

    A positive value means the model thinks the true probability is higher
    than what the market implies — a potential value bet.
    """
    try:
        return float(model_prob) - float(market_implied)
    except (TypeError, ValueError):
        return 0.0


def kelly_criterion(edge: float, odds: int, fraction: float = KELLY_FRACTION) -> float:
    """Return recommended bet size in units using fractional Kelly.

    Parameters
    ----------
    edge:
        Edge as a decimal (e.g. 0.05 for 5% edge).
    odds:
        American odds for this bet.
    fraction:
        Kelly fraction (default: KELLY_FRACTION = 0.25 for quarter-Kelly).

    Returns
    -------
    Recommended bet size in units (0 if edge <= 0 or no value).
    """
    try:
        if edge <= 0:
            return 0.0
        # Decimal odds multiplier (net profit per unit wagered)
        odds_int = int(odds)
        if odds_int > 0:
            decimal_odds = odds_int / 100.0
        else:
            decimal_odds = 100.0 / (-odds_int)

        # Full Kelly = edge / decimal_odds
        full_kelly = edge / max(decimal_odds, 0.001)
        return round(max(0.0, full_kelly * fraction), 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


# ---------------------------------------------------------------------------
# Confidence tier
# ---------------------------------------------------------------------------

def _confidence_tier(edge: float) -> str:
    """Return a confidence tier emoji label based on edge size."""
    if edge >= HIGH_CONFIDENCE_EDGE:
        return "🔥 HIGH"
    if edge >= MED_CONFIDENCE_EDGE:
        return "✅ MEDIUM"
    return "📊 LOW"


# ---------------------------------------------------------------------------
# Main value-finding function
# ---------------------------------------------------------------------------

def find_value_bets(
    predictions: list[dict[str, Any]],
    market_odds: dict[str, Any] | None = None,
    min_edge: float = MIN_EDGE_PCT,
) -> list[dict[str, Any]]:
    """Filter predictions to only those with edge >= min_edge.

    Parameters
    ----------
    predictions:
        List of prediction dicts. Each dict should include keys like:
        ``bet_type``, ``description``, ``model_prob``, ``market_odds_american``.
    market_odds:
        Optional dict loaded from ``odds_today.json``.  If None, this function
        falls back to the ``market_odds_american`` already embedded in each
        prediction dict (if present).
    min_edge:
        Minimum edge threshold (default 3%).

    Returns
    -------
    List of dicts with: bet_type, description, model_prob, market_implied,
    edge, kelly_units, confidence_tier.
    """
    value_bets: list[dict[str, Any]] = []

    for pred in predictions:
        try:
            bet_type = pred.get("bet_type", "")
            description = pred.get("description", "")
            model_prob = float(pred.get("model_prob", 0.0))

            # Resolve market odds
            market_odds_american: int | None = pred.get("market_odds_american")

            # Try to look up from the odds dict by game key
            if market_odds and "game_key" in pred:
                game_key = pred["game_key"]
                game_market = market_odds.get(game_key, {})
                field_map: dict[str, str] = {
                    "moneyline_home": "moneyline_home",
                    "moneyline_away": "moneyline_away",
                    "total_over": "total_over_odds",
                    "total_under": "total_under_odds",
                }
                mapped_field = field_map.get(bet_type)
                if mapped_field and mapped_field in game_market:
                    market_odds_american = int(game_market[mapped_field])

            if market_odds_american is None:
                continue

            market_implied = american_to_implied(market_odds_american)
            edge = calculate_edge(model_prob, market_implied)

            if edge < min_edge:
                continue

            kelly = kelly_criterion(edge, market_odds_american)
            value_bets.append(
                {
                    "bet_type": bet_type,
                    "description": description,
                    "model_prob": round(model_prob, 4),
                    "market_implied": round(market_implied, 4),
                    "edge": round(edge, 4),
                    "kelly_units": kelly,
                    "confidence_tier": _confidence_tier(edge),
                }
            )
        except Exception as exc:
            print(f"[WARN] Error processing value bet prediction: {exc}")

    # Sort by edge descending
    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


# ---------------------------------------------------------------------------
# Odds file loader
# ---------------------------------------------------------------------------

def load_odds_file(path: str | None = None) -> dict[str, Any] | None:
    """Load market odds from a JSON file.

    Returns None if the file does not exist or cannot be parsed.

    Expected format::

        {
          "NYY_vs_BOS": {
            "moneyline_home": -150,
            "moneyline_away": 130,
            "total_over": 8.5,
            "total_over_odds": -110,
            "total_under_odds": -110
          }
        }
    """
    fpath = path or ODDS_FILE_DEFAULT
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            print(f"[WARN] Odds file {fpath} does not contain a JSON object.")
            return None
        return data
    except Exception as exc:
        print(f"[WARN] Could not load odds file {fpath}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_value_bets(value_bets: list[dict[str, Any]]) -> None:
    """Pretty-print a list of value bets."""
    if not value_bets:
        return
    print("   💰 Recommended Bets:")
    for bet in value_bets:
        desc = bet.get("description", "")
        kelly = bet.get("kelly_units", 0.0)
        edge_pct = bet.get("edge", 0.0) * 100
        tier = bet.get("confidence_tier", "")
        print(f"     • {desc:<22} → {kelly:.1f}u (Kelly)  [Edge: {edge_pct:.1f}%]  {tier}")
