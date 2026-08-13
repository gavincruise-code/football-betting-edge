"""
League Calibrator
=================
Runs walk-forward backtests across every scannable league and produces a
data-driven optimal strategy recommendation for each one.

Results are saved to ``league_strategy_cache.json`` and consumed by the
Live Opportunity Scanner so that the Auto-Optimal strategy selection is
always grounded in the most recent historical data rather than hard-coded
rules.

Strategy selection logic
------------------------
The calibrator considers two independent signals:

1. **Backtest ROI** – actual profit/loss of the Poisson/Dixon-Coles model
   on the last 2–3 seasons of completed matches.
2. **League goal profile** – average goals/match, home-win %, draw rate.
   High-scoring / low-draw leagues tend to benefit from ensemble blending,
   while low-scoring / tactical leagues fit Dixon-Coles more precisely.

Combined, these signals map to one of three scanner strategies:
  • ``Dixon-Coles Only``   – best for low-scoring, tactical leagues (≤2.3 avg)
  • ``Dual Ensemble``      – best for high-variance or mid-range leagues
  • ``XGBoost ML Only``    – rarely recommended; only when DC ROI is deeply
                             negative AND avg goals are high (≥2.9)

A recommended minimum edge % is also stored per league so the scanner can
auto-tighten the filter for volatile markets.
"""

import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# All leagues the scanner can discover, mapped to their football-data.co.uk
# download codes.  UEFA / ESPN-only leagues are excluded because they have no
# football-data CSV to backtest against.
# ──────────────────────────────────────────────────────────────────────────────
CALIBRATION_LEAGUES = {
    # English
    "EPL":                  "E0",
    "Championship":         "E1",
    "League 1":             "E2",
    "League 2":             "E3",
    "National League":      "EC",
    # Scottish
    "Scottish Premiership": "SC0",
    "Scottish Championship":"SC1",
    # Spanish
    "La Liga":              "SP1",
    "Segunda Division":     "SP2",
    # German
    "Bundesliga":           "D1",
    "Bundesliga 2":         "D2",
    # Italian
    "Serie A":              "I1",
    "Serie B":              "I2",
    # French
    "Ligue 1":              "F1",
    "Ligue 2":              "F2",
    # Dutch / Belgian / Portuguese / Turkish / Greek
    "Eredivisie":           "N1",
    "Pro League":           "B1",
    "Liga Portugal":        "P1",
    "Super Lig":            "T1",
    "Super League":         "G1",
    # Americas
    "USA (MLS)":            "USA",
    "Argentina":            "ARG",
    "Brazil":               "BRA",
    "Mexico":               "MEX",
    # Asia
    "Japan":                "JPN",
    "China":                "CHN",
    "Calcutta Premier Division": "ind.2",
    "Indian Super League":  "ind.1",
    # Nordic / Eastern Europe
    "Sweden":               "SWE",
    "Norway":               "NOR",
    "Denmark":              "DNK",
    "Finland":              "FIN",
    "Poland":               "POL",
    "Romania":              "ROU",
    "Switzerland":          "SWZ",
    "Austria":              "AUT",
}

CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "league_strategy_cache.json",
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pick_strategy(avg_goals: float, draw_rate: float, roi: float) -> str:
    """Return the recommended scanner strategy given league characteristics."""
    # Deeply negative ROI AND very high-scoring → try XGBoost
    if roi < -15 and avg_goals >= 2.9:
        return "XGBoost ML Only"

    # Low-scoring / tactical leagues → Dixon-Coles fits tightly
    if avg_goals <= 2.30 or draw_rate >= 0.28:
        return "Dixon-Coles Only"

    # High-scoring or volatile → ensemble blending adds value
    if avg_goals >= 2.70 or draw_rate <= 0.22:
        return "Dual Ensemble"

    # Mid-range → lean on ensemble as the safer default
    return "Dual Ensemble"


def _recommended_edge(roi: float) -> float:
    """Tighten the minimum edge filter for weak or negative-ROI leagues."""
    if roi < -15:
        return 0.10   # 10 % — only very strong edges
    if roi < -8:
        return 0.08   # 8 %
    if roi < -3:
        return 0.07   # 7 %
    return 0.05       # 5 % — baseline


# ──────────────────────────────────────────────────────────────────────────────
# Core calibration routine
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_all_leagues(
    n_matches: int = 300,
    edge_margin: float = 0.05,
    stake: float = 10.0,
    progress_cb=None,
) -> dict:
    """
    Backtest all scannable leagues and determine the optimal scanner strategy
    for each one.

    Parameters
    ----------
    n_matches : int
        Number of most-recent completed matches to backtest per league.
    edge_margin : float
        Minimum model edge used in the backtest.
    stake : float
        Flat stake per bet in the backtest.
    progress_cb : callable | None
        Optional ``progress_cb(league_name, current, total)`` for UI feedback.

    Returns
    -------
    dict
        Full calibration result (also written to ``league_strategy_cache.json``).
    """
    from data_utils import download_league_data
    from backtester import run_backtest

    leagues = list(CALIBRATION_LEAGUES.items())
    total = len(leagues)
    results = {}

    for i, (league_name, fd_code) in enumerate(leagues):
        if progress_cb:
            progress_cb(league_name, i, total)

        try:
            df = download_league_data(fd_code)
            if df.empty or "FTHG" not in df.columns:
                logger.warning("No data for %s (%s)", league_name, fd_code)
                continue

            completed = df.dropna(subset=["FTHG", "FTAG"]).copy()
            completed["FTHG"] = pd.to_numeric(completed["FTHG"], errors="coerce")
            completed["FTAG"] = pd.to_numeric(completed["FTAG"], errors="coerce")
            completed = completed.dropna(subset=["FTHG", "FTAG"])

            if len(completed) < 30:
                logger.warning("Too few completed matches for %s", league_name)
                continue

            # League-level statistics
            completed["total_goals"] = completed["FTHG"] + completed["FTAG"]
            avg_goals  = float(completed["total_goals"].mean())
            home_win   = float((completed.get("FTR", pd.Series()) == "H").mean()) if "FTR" in completed.columns else 0.45
            draw_rate  = float((completed.get("FTR", pd.Series()) == "D").mean()) if "FTR" in completed.columns else 0.25

            # Backtest on the N most-recent matches
            sample = completed.tail(n_matches)
            bt = run_backtest(sample, edge_margin=edge_margin, stake=stake)

            strategy    = _pick_strategy(avg_goals, draw_rate, bt.roi)
            min_edge    = _recommended_edge(bt.roi)

            results[league_name] = {
                "fd_code":           fd_code,
                "strategy":          strategy,
                "backtest_roi":      round(bt.roi, 2),
                "backtest_pl":       round(bt.total_profit_loss, 2),
                "backtest_bets":     bt.total_bets,
                "backtest_win_rate": round(bt.win_rate, 3),
                "avg_goals":         round(avg_goals, 3),
                "home_win_pct":      round(home_win * 100, 1),
                "draw_rate_pct":     round(draw_rate * 100, 1),
                "recommended_min_edge": min_edge,
                "sample_matches":    len(sample),
            }
            logger.info(
                "Calibrated %-22s → %-22s  ROI %+.1f%%  avg_goals %.2f",
                league_name, strategy, bt.roi, avg_goals,
            )

        except Exception as exc:
            logger.warning("Failed to calibrate %s: %s", league_name, exc)

    payload = {
        "calibrated_at": datetime.now().isoformat(),
        "n_matches":     n_matches,
        "edge_margin":   edge_margin,
        "leagues":       results,
    }

    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Calibration cache written to %s", CACHE_FILE)
    except Exception as exc:
        logger.warning("Could not write calibration cache: %s", exc)

    return payload


# ──────────────────────────────────────────────────────────────────────────────
# Cache reader
# ──────────────────────────────────────────────────────────────────────────────

def load_calibration_cache() -> dict:
    """
    Load the most recent calibration cache from disk.

    Returns an empty dict if the cache file does not exist yet.
    """
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not read calibration cache: %s", exc)
        return {}


def get_strategy_for_league(league_name: str, cache: dict | None = None) -> str:
    """
    Return the cached optimal strategy for ``league_name``.

    Falls back to the rule-based heuristic if the cache is unavailable or the
    league is not in it.
    """
    if cache is None:
        cache = load_calibration_cache()

    league_data = cache.get("leagues", {}).get(league_name)
    if league_data:
        return league_data.get("strategy", "Dual Ensemble")

    # Legacy fallback (same heuristic as before)
    if any(w in league_name for w in ["La Liga", "EPL", "Premier", "La_Liga"]):
        return "Dixon-Coles Only"
    return "Dual Ensemble"


def cache_age_hours(cache: dict | None = None) -> float | None:
    """Return the age of the cache in hours, or None if not calibrated yet."""
    if cache is None:
        cache = load_calibration_cache()
    ts = cache.get("calibrated_at")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return (datetime.now() - dt).total_seconds() / 3600
    except Exception:
        return None
