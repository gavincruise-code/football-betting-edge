"""
Bet Tracker Module
===================
Handles logging placed bets from the Live Opportunity Scanner, updating bet outcomes,
calculating cumulative Profit & Loss (£), auto-settling based on data feeds, and rendering P/L analytics.
"""

import os
import unicodedata
import pandas as pd
import numpy as np
from datetime import datetime
import logging

LOG_FILE = "placed_bets_log.csv"
logger = logging.getLogger(__name__)


def _norm(name: str) -> str:
    """Normalise team name for fuzzy matching."""
    if not name:
        return ""
    s = str(name).lower()
    for src, tgt in [('ø','o'),('æ','ae'),('å','a'),('ß','ss'),('ü','u'),
                     ('ö','o'),('ä','a'),('é','e'),('è','e'),('à','a'),('ç','c'),('ñ','n')]:
        s = s.replace(src, tgt)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    for prefix in ['f.c. ', 'fc ', 'fk ', 'ifk ', 'sk ', 'ac ', 'cd ', 'sc ', 'as ', 'ss ', 'sv ', 'bk ', 'if ', 'rb ', 'hsc ']:
        if s.startswith(prefix):
            s = s[len(prefix):]
    aliases = {
        'kobenhavn': 'copenhagen',
        'copenhague': 'copenhagen',
        'wien': 'vienna',
        'munchen': 'munich',
        'lisbon': 'sporting',
        'crvena zvezda': 'red star',
    }
    for k, v in aliases.items():
        if k in s:
            s = s.replace(k, v)
    return s.strip()



def _teams_match(a: str, b: str, cutoff: float = 0.5) -> bool:
    """Return True if team names are similar enough."""
    import difflib
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    if na[:4] == nb[:4] and len(na) >= 4:
        return True
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return ratio >= cutoff


def get_placed_bets() -> pd.DataFrame:
    """Load existing logged bets from CSV file and compute P/L statistics."""
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty:
                return df

            # Ensure required columns exist (backwards-compatible with older CSV rows)
            if 'Result' not in df.columns:
                df['Result'] = 'PENDING'
            if 'Profit_Loss_£' not in df.columns:
                df['Profit_Loss_£'] = 0.0
            if 'Cumulative_PL_£' not in df.columns:
                df['Cumulative_PL_£'] = 0.0
            # FIX M5: CLV columns — blank for pre-existing rows, filled in later
            if 'Closing_Odds' not in df.columns:
                df['Closing_Odds'] = ''
            if 'CLV_%' not in df.columns:
                df['CLV_%'] = ''

            # Recalculate P/L and cumulative P/L from scratch
            pl_vals = []
            for _, r in df.iterrows():
                res = str(r.get('Result', 'PENDING')).upper()
                odds = float(r.get('Odds', 2.0)) if pd.notna(r.get('Odds')) else 2.0
                stake = float(r.get('Recommended_Stake_£', 10.0)) if pd.notna(r.get('Recommended_Stake_£')) else 10.0
                if res == 'WIN':
                    pl = stake * (odds - 1.0)
                elif res == 'LOSS':
                    pl = -stake
                else:
                    pl = 0.0
                pl_vals.append(round(pl, 2))

            df['Profit_Loss_£'] = pl_vals
            df['Cumulative_PL_£'] = [round(c, 2) for c in np.cumsum(pl_vals)]
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()



def _git_sync_bet_log():
    """Attempt background sync of placed_bets_log.csv to GitHub (via REST API or git)."""
    import threading, requests, base64, subprocess
    def _worker():
        gh_token = os.getenv("GITHUB_TOKEN", "")
        repo = os.getenv("GITHUB_REPO", "gavincruise-code/football-betting-edge")

        if gh_token:
            try:
                if not os.path.exists(LOG_FILE):
                    return
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    content_str = f.read()

                b64_content = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
                url = f"https://api.github.com/repos/{repo}/contents/{LOG_FILE}"
                headers = {
                    "Authorization": f"Bearer {gh_token}",
                    "Accept": "application/vnd.github+json"
                }
                r_get = requests.get(url, headers=headers, timeout=10)
                sha = r_get.json().get("sha") if r_get.status_code == 200 else None

                payload = {
                    "message": "Auto-sync bet log from app [skip ci]",
                    "content": b64_content,
                }
                if sha:
                    payload["sha"] = sha

                requests.put(url, headers=headers, json=payload, timeout=10)
                logger.info("Synced placed_bets_log.csv to GitHub via REST API")
                return
            except Exception as e:
                logger.warning(f"GitHub API sync error: {e}")

        # Fallback to local git subprocess
        try:
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            subprocess.run(["git", "add", "placed_bets_log.csv"], cwd=repo_dir, capture_output=True, timeout=10)
            subprocess.run(["git", "commit", "-m", "Auto-sync bet log"], cwd=repo_dir, capture_output=True, timeout=10)
            subprocess.run(["git", "push", "origin", "main"], cwd=repo_dir, capture_output=True, timeout=15)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()



def record_bet(
    date: str,
    league: str,
    home_team: str,
    away_team: str,
    market: str,
    odds: float,
    strategy: str,
    model_prob: float,
    implied_prob: float,
    edge_pct: float,
    recommended_stake: float,
    notes: str = "",
    closing_odds: float = None,       # FIX M5: BSP / Betfair closing price
) -> bool:
    """Append a newly placed bet record to placed_bets_log.csv.

    CLV (Closing Line Value) measures whether the odds taken beat the
    closing market price — the most reliable short-term indicator of genuine
    model edge, independent of result noise.

    Record closing_odds (BSP) after each market closes to populate CLV_%.
    CLV_% > 0 means you consistently beat the closing line — a strong signal
    of real edge even at small sample sizes (useful from ~30 bets onwards).
    """
    # Compute CLV if closing odds provided
    if closing_odds and closing_odds > 1.0 and model_prob and model_prob > 0:
        clv_pct = round((model_prob * closing_odds - 1.0) * 100, 2)
    else:
        clv_pct = None

    record = {
        'Timestamp_Recorded': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Match_Date': str(date),
        'League': str(league),
        'Home_Team': str(home_team),
        'Away_Team': str(away_team),
        'Market': str(market),
        'Odds': float(odds) if pd.notna(odds) else 0.0,
        'Strategy': str(strategy),
        'Model_Prob_%': round(float(model_prob) * 100, 1),
        'Implied_Prob_%': round(float(implied_prob) * 100, 1),
        'Edge_%': round(float(edge_pct) * 100, 1),
        'Recommended_Stake_£': round(float(recommended_stake), 2),
        'Closing_Odds': round(float(closing_odds), 3) if closing_odds else '',
        'CLV_%': clv_pct if clv_pct is not None else '',
        'Result': 'PENDING',
        'Profit_Loss_£': 0.0,
        'Cumulative_PL_£': 0.0,
        'Notes': str(notes)
    }
    df = get_placed_bets()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
    _git_sync_bet_log()
    return True


def update_bet_result(row_idx: int, result: str) -> bool:
    """Update result ('WIN' or 'LOSS') for a specific logged bet by index."""
    df = get_placed_bets()
    if df.empty or row_idx >= len(df):
        return False
    df.loc[row_idx, 'Result'] = str(result).upper()
    df.to_csv(LOG_FILE, index=False)
    _git_sync_bet_log()
    return True



def is_bet_recorded(home_team: str, away_team: str, market: str) -> bool:
    """Check if a fixture/market combination has already been logged."""
    df = get_placed_bets()
    if df.empty:
        return False
    match = df[
        (df['Home_Team'].str.lower() == home_team.lower()) &
        (df['Away_Team'].str.lower() == away_team.lower()) &
        (df['Market'].str.lower() == market.lower())
    ]
    return not match.empty


# ─────────────────────────────────────────────────────────────────────────────
_LEAGUE_CODE_MAP = {
    'EPL': 'E0', 'Premier League': 'E0', 'Championship': 'E1',
    'League 1': 'E2', 'League 2': 'E3', 'Conference': 'EC',
    'Scottish Premiership': 'SC0', 'Scottish Championship': 'SC1',
    'Scottish League 1': 'SC2', 'Scottish League 2': 'SC3',
    'La_Liga': 'SP1', 'La Liga': 'SP1', 'Segunda Division': 'SP2',
    'Bundesliga': 'D1', 'Bundesliga 2': 'D2',
    'Serie_A': 'I1', 'Serie A': 'I1', 'Serie B': 'I2',
    'Ligue_1': 'F1', 'Ligue 1': 'F1', 'Ligue 2': 'F2',
    'Eredivisie': 'N1', 'Netherlands': 'N1',
    'Pro League': 'B1', 'Belgium': 'B1', 'Jupiler Pro League': 'B1',
    'Liga Portugal': 'P1', 'Portugal': 'P1',
    'Super Lig': 'T1', 'Turkey': 'T1', '1. Lig': 'T2', 'Turkish 1. Lig': 'T2',
    'Super League': 'G1', 'Greece': 'G1',
    'USA (MLS)': 'USA', 'USA': 'USA',
    'Argentina': 'ARG', 'Brazil': 'BRA', 'Mexico': 'MEX',
    'Japan': 'JPN', 'China': 'CHN', 'Sweden': 'SWE', 'Norway': 'NOR',
    'Denmark': 'DNK', 'Finland': 'FIN', 'Poland': 'POL', 'Romania': 'ROU',
    'Switzerland': 'SWZ', 'Austria': 'AUT', 'Ireland': 'IRL', 'Russia': 'RUS',
    'India': 'ind.1', 'Calcutta Premier Division': 'ind.2', 'Indian Super League': 'ind.1',
    'India (Calcutta Premier Division)': 'ind.2', 'India (Super League)': 'ind.1',
}

_results_cache: dict = {}


def _fetch_results_for_league(league: str) -> pd.DataFrame:
    """Download completed match results for a league (cached per session)."""
    if league in _results_cache:
        return _results_cache[league]
    try:
        from data_utils import download_league_data
        code = _LEAGUE_CODE_MAP.get(league, league)
        df = download_league_data(code)
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            df = df.dropna(subset=['FTHG', 'FTAG'])
            df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
            df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
            df = df.dropna(subset=['FTHG', 'FTAG'])
        _results_cache[league] = df
        return df
    except Exception as e:
        logger.warning(f"Could not fetch results for {league}: {e}")
        return pd.DataFrame()


_ESPN_SLUGS = {
    'EPL': 'eng.1', 'Premier League': 'eng.1', 'Championship': 'eng.2',
    'League 1': 'eng.3', 'League 2': 'eng.4', 'National League': 'eng.5', 'Conference': 'eng.5',
    'Scottish Premiership': 'sco.1', 'Scottish Championship': 'sco.2',
    'La Liga': 'esp.1', 'La_Liga': 'esp.1', 'Segunda Division': 'esp.2',
    'Bundesliga': 'ger.1', 'Bundesliga 2': 'ger.2',
    'Serie A': 'ita.1', 'Serie_A': 'ita.1', 'Serie B': 'ita.2',
    'Ligue 1': 'fra.1', 'Ligue_1': 'fra.1', 'Ligue 2': 'fra.2',
    'Eredivisie': 'ned.1', 'Netherlands': 'ned.1',
    'Pro League': 'bel.1', 'Belgium': 'bel.1', 'Jupiler Pro League': 'bel.1',
    'Liga Portugal': 'por.1', 'Portugal': 'por.1',
    'Super Lig': 'tur.1', 'Turkey': 'tur.1',
    'Super League': 'gre.1', 'Greece': 'gre.1',
    'USA (MLS)': 'usa.1', 'USA': 'usa.1',
    'Argentina': 'arg.1', 'Brazil': 'bra.1', 'Mexico': 'mex.1',
    'Japan': 'jpn.1', 'China': 'chn.1', 'Sweden': 'swe.1', 'Norway': 'nor.1',
    'Denmark': 'dnk.1', 'Finland': 'fin.1', 'Poland': 'pol.1', 'Romania': 'rou.1',
    'Switzerland': 'sui.1', 'Austria': 'aut.1',
    'Indian Super League': 'ind.1', 'Calcutta Premier Division': 'ind.2', 'India': 'ind.1',
    'UEFA Champions League': 'uefa.champions', 'UEFA CL Qualifying': 'uefa.champions_qual',
    'UEFA Europa League': 'uefa.europa', 'UEFA EL Qualifying': 'uefa.europa_qual',
    'UEFA Conference League': 'uefa.europa.conf', 'UEFA ECL Qualifying': 'uefa.europa.conf_qual',
}

_espn_cache: dict = {}


def _fetch_espn_total_goals(league: str, home: str, away: str, m_date) -> float:
    """Fetch completed match total goals from ESPN Scoreboard API."""
    import requests
    slug = _ESPN_SLUGS.get(league)
    if not slug or pd.isnull(m_date):
        return None

    # Check match_date and adjacent days
    dates_to_check = [
        m_date.strftime('%Y%m%d'),
        (m_date - pd.Timedelta(days=1)).strftime('%Y%m%d'),
        (m_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
    ]

    for d_str in dates_to_check:
        cache_key = f"{slug}_{d_str}"
        if cache_key in _espn_cache:
            events = _espn_cache[cache_key]
        else:
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={d_str}"
                r = requests.get(url, timeout=4)
                events = r.json().get('events', []) if r.status_code == 200 else []
                _espn_cache[cache_key] = events
            except Exception:
                events = []

        for ev in events:
            state = ev.get('status', {}).get('type', {}).get('state', '')
            if state != 'post':
                continue
            comps = ev.get('competitions', [{}])[0].get('competitors', [])
            if len(comps) < 2:
                continue
            h_comp = next((c for c in comps if c.get('homeAway') == 'home'), comps[0])
            a_comp = next((c for c in comps if c.get('homeAway') == 'away'), comps[1])

            h_name = h_comp.get('team', {}).get('displayName', '')
            a_name = a_comp.get('team', {}).get('displayName', '')

            if _teams_match(home, h_name) and _teams_match(away, a_name):
                h_score = h_comp.get('score')
                a_score = a_comp.get('score')
                if h_score is not None and a_score is not None:
                    try:
                        return float(h_score) + float(a_score)
                    except ValueError:
                        pass
    return None


def auto_settle_bets() -> dict:
    """
    Automatically settle all PENDING bets by looking up actual match results
    from football-data.co.uk CSV feeds, falling back to ESPN Scoreboard API.

    Returns a dict with keys:
        settled   – number of bets newly settled
        not_found – number of bets whose result could not be found yet
        already   – number already settled
    """
    df = get_placed_bets()
    if df.empty:
        return {'settled': 0, 'not_found': 0, 'already': 0}

    settled = 0
    not_found = 0
    already = 0
    changed = False

    for idx, row in df.iterrows():
        result = str(row.get('Result', 'PENDING')).upper()
        if result in ('WIN', 'LOSS'):
            already += 1
            continue

        league  = str(row.get('League', ''))
        home    = str(row.get('Home_Team', ''))
        away    = str(row.get('Away_Team', ''))
        market  = str(row.get('Market', '')).lower()
        m_date  = pd.to_datetime(row.get('Match_Date'), errors='coerce')

        # Skip bets on future dates
        if pd.notnull(m_date) and m_date.normalize() > pd.Timestamp.now().normalize():
            not_found += 1
            continue

        total_goals = None

        # Source 1: football-data.co.uk CSV
        results_df = _fetch_results_for_league(league)
        if not results_df.empty and 'HomeTeam' in results_df.columns:
            if 'Date' in results_df.columns and pd.notnull(m_date):
                date_mask = (
                    (results_df['Date'] >= m_date - pd.Timedelta(days=1)) &
                    (results_df['Date'] <= m_date + pd.Timedelta(days=1))
                )
                candidates = results_df[date_mask]
            else:
                candidates = results_df

            matched = None
            for _, res_row in candidates.iterrows():
                if (_teams_match(home, str(res_row.get('HomeTeam', ''))) and
                        _teams_match(away, str(res_row.get('AwayTeam', '')))):
                    matched = res_row
                    break

            if matched is not None:
                fthg = matched.get('FTHG')
                ftag = matched.get('FTAG')
                if pd.notna(fthg) and pd.notna(ftag):
                    total_goals = float(fthg) + float(ftag)

        # Source 2: ESPN Scoreboard API Fallback (if not found in CSV)
        if total_goals is None:
            total_goals = _fetch_espn_total_goals(league, home, away, m_date)

        if total_goals is None:
            not_found += 1
            continue

        over25 = total_goals > 2.5

        if 'over' in market:
            outcome = 'WIN' if over25 else 'LOSS'
        elif 'under' in market:
            outcome = 'WIN' if not over25 else 'LOSS'
        else:
            not_found += 1
            continue

        df.loc[idx, 'Result'] = outcome
        settled += 1
        changed = True
        logger.info(f"Auto-settled {home} v {away} ({league}) as {outcome} "
                    f"(goals: {total_goals:.0f}, market: {market})")

    if changed:
        df.to_csv(LOG_FILE, index=False)

    return {'settled': settled, 'not_found': not_found, 'already': already}

