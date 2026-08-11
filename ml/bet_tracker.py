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
    for prefix in ['fc ', 'sk ', 'ac ', 'cd ', 'sc ', 'as ', 'ss ', 'sv ', 'bk ', 'if ']:
        if s.startswith(prefix):
            s = s[len(prefix):]
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

            # Ensure required columns exist
            if 'Result' not in df.columns:
                df['Result'] = 'PENDING'
            if 'Profit_Loss_£' not in df.columns:
                df['Profit_Loss_£'] = 0.0
            if 'Cumulative_PL_£' not in df.columns:
                df['Cumulative_PL_£'] = 0.0

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
    notes: str = ""
) -> bool:
    """Append a newly placed bet record to placed_bets_log.csv."""
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
        'Result': 'PENDING',
        'Profit_Loss_£': 0.0,
        'Cumulative_PL_£': 0.0,
        'Notes': str(notes)
    }
    df = get_placed_bets()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
    return True


def update_bet_result(row_idx: int, result: str) -> bool:
    """Update result ('WIN' or 'LOSS') for a specific logged bet by index."""
    df = get_placed_bets()
    if df.empty or row_idx >= len(df):
        return False
    df.loc[row_idx, 'Result'] = str(result).upper()
    df.to_csv(LOG_FILE, index=False)
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
# AUTO-SETTLEMENT
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
        # Only keep rows with completed scores
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


def auto_settle_bets() -> dict:
    """
    Automatically settle all PENDING bets by looking up actual match results
    from the football-data.co.uk feed.

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

        results_df = _fetch_results_for_league(league)
        if results_df.empty or 'HomeTeam' not in results_df.columns:
            not_found += 1
            continue

        # Filter to matches on or near the recorded date (±1 day tolerance)
        if 'Date' in results_df.columns and pd.notnull(m_date):
            date_mask = (
                (results_df['Date'] >= m_date - pd.Timedelta(days=1)) &
                (results_df['Date'] <= m_date + pd.Timedelta(days=1))
            )
            candidates = results_df[date_mask]
        else:
            candidates = results_df

        # Fuzzy-match team names
        matched = None
        for _, res_row in candidates.iterrows():
            if (_teams_match(home, str(res_row.get('HomeTeam', ''))) and
                    _teams_match(away, str(res_row.get('AwayTeam', '')))):
                matched = res_row
                break

        if matched is None:
            not_found += 1
            continue

        fthg = matched.get('FTHG')
        ftag = matched.get('FTAG')
        if pd.isna(fthg) or pd.isna(ftag):
            not_found += 1
            continue

        total_goals = float(fthg) + float(ftag)
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
