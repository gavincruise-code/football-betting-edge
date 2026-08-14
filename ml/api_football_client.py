# -*- coding: utf-8 -*-
"""
ml/api_football_client.py
=========================
Fetches historical match results from api-football.com for leagues
not covered by football-data.co.uk. Returns DataFrames in the same
column format so the rest of the scanner pipeline works unchanged.

Columns returned: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_API_KEY = os.getenv("API_FOOTBALL_KEY", "")
_BASE_URL = "https://v3.football.api-sports.io"
_HEADERS = {"x-apisports-key": _API_KEY}
_CACHE_DIR = Path(__file__).parent.parent / "cache" / "api_football"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_MAX_AGE_DAYS = 3

# ---------------------------------------------------------------------------
# League ID map  (league_name_as_seen_in_scanner -> api-football league id)
# ---------------------------------------------------------------------------
LEAGUE_ID_MAP = {
    # Asia
    "Japan": 98, "J1 League": 98,
    "China": 169, "Chinese Super League": 169,
    "South Korea": 292, "K League 1": 292,
    "India": 323, "Indian Super League": 323,
    "Saudi Arabia": 307, "Saudi Pro League": 307,
    "Thailand": 296, "Thai League 1": 296,
    # Americas
    "USA (MLS)": 253, "MLS": 253,
    "Brazil": 71, "Brazil Serie A": 71,
    "Argentina": 128, "Argentina Primera": 128,
    "Mexico": 262, "Liga MX": 262,
    "Colombia": 239, "Categoria Primera A": 239,
    "Chile": 265, "Primera Division Chile": 265,
    "Uruguay": 268, "Primera Division Uruguay": 268,
    "Ecuador": 240, "LigaPro": 240,
    "Peru": 281, "Liga 1 Peru": 281,
    # Africa
    "Egypt": 233, "Egyptian Premier League": 233,
    "Morocco": 200, "Botola Pro": 200,
    "South Africa": 288, "Premier Soccer League": 288,
    # Europe extras (football-data.co.uk does not cover these)
    "Sweden": 113, "Norway": 103, "Denmark": 119,
    "Finland": 244, "Poland": 106, "Romania": 283,
    "Switzerland": 207, "Austria": 218, "Ireland": 357,
    "Russia": 235, "Croatia": 210, "Czech Republic": 345,
    "Ukraine": 333, "Serbia": 286, "Slovakia": 332,
    "Hungary": 271, "Israel": 384,
    # Oceania
    "Australia": 188, "A-League": 188,
}

_SEASONS = [2024, 2023, 2022, 2021, 2020]


def _cache_path(league_id, season):
    return _CACHE_DIR / f"league_{league_id}_season_{season}.json"


def _cache_fresh(path):
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=_CACHE_MAX_AGE_DAYS)


def _fetch_season(league_id, season):
    """Fetch all finished fixtures for one league/season."""
    path = _cache_path(league_id, season)
    if _cache_fresh(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    try:
        resp = requests.get(
            f"{_BASE_URL}/fixtures",
            headers={"x-apisports-key": os.getenv("API_FOOTBALL_KEY", "")},
            params={"league": league_id, "season": season, "status": "FT"},
            timeout=15,
        )
        resp.raise_for_status()
        fixtures = resp.json().get("response", [])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fixtures, f)
        remaining = int(resp.headers.get("x-ratelimit-requests-remaining", 100))
        if remaining < 5:
            time.sleep(2)
        return fixtures
    except Exception as exc:
        log.warning("API-Football fetch failed league=%s season=%s: %s", league_id, season, exc)
        return []


def _fixtures_to_df(fixtures):
    """Convert raw API-Football fixture list to football-data.co.uk format."""
    rows = []
    for fix in fixtures:
        try:
            goals = fix.get("goals") or fix.get("score", {}).get("fulltime", {})
            hg = goals.get("home")
            ag = goals.get("away")
            if hg is None or ag is None:
                continue
            hg, ag = int(hg), int(ag)
            rows.append({
                "Date":     pd.to_datetime(fix["fixture"]["date"][:10]),
                "HomeTeam": fix["teams"]["home"]["name"],
                "AwayTeam": fix["teams"]["away"]["name"],
                "FTHG":     hg,
                "FTAG":     ag,
                "FTR":      "H" if hg > ag else ("A" if ag > hg else "D"),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def get_league_id(league_name):
    return LEAGUE_ID_MAP.get(league_name)


def fetch_league_history(league_name, seasons=None, min_matches=50):
    """
    Fetch historical results for a league from api-football.com.
    Returns DataFrame with columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
    Returns empty DataFrame if league is unsupported or API fails.
    """
    api_key = os.getenv("API_FOOTBALL_KEY", "")
    if not api_key:
        log.warning("API_FOOTBALL_KEY not set in .env")
        return pd.DataFrame()
    league_id = get_league_id(league_name)
    if league_id is None:
        return pd.DataFrame()
    seasons_to_try = seasons or _SEASONS
    all_dfs = []
    for season in seasons_to_try:
        raw = _fetch_season(league_id, season)
        df = _fixtures_to_df(raw)
        if not df.empty:
            all_dfs.append(df)
        if sum(len(d) for d in all_dfs) >= min_matches:
            break
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True).sort_values("Date").reset_index(drop=True)


def supported_leagues():
    return sorted(LEAGUE_ID_MAP.keys())


def check_api_key():
    """Validate the API key. Returns (True, status_string) or (False, error)."""
    api_key = os.getenv("API_FOOTBALL_KEY", "")
    if not api_key:
        return False, "API_FOOTBALL_KEY not set in .env"
    try:
        resp = requests.get(
            f"{_BASE_URL}/status",
            headers={"x-apisports-key": api_key},
            timeout=10,
        )
        data = resp.json().get("response", {})
        plan = data.get("subscription", {}).get("plan", "Unknown")
        used = data.get("requests", {}).get("current", "?")
        limit = data.get("requests", {}).get("limit_day", "?")
        return True, f"Plan: {plan} | Requests today: {used}/{limit}"
    except Exception as exc:
        return False, str(exc)
