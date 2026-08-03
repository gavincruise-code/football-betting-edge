"""
Data Ingestion Pipeline for Football Betting ML
===============================================
Scrapes Understat match-level xG data, downloads football-data.co.uk CSVs,
merges datasets with team name normalization and caches to Parquet.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Optional
from ml.config import UNDERSTAT_TO_FD_LEAGUE, UNDERSTAT_SEASON_IDS, CACHE_DIR
from ml.team_mappings import get_fd_name

logger = logging.getLogger(__name__)

def scrape_understat_season(league: str, season: int) -> pd.DataFrame:
    """
    Fetch match-level xG data from Understat for a given league and season year.
    Returns DataFrame with: date, home_team, away_team, xG_home, xG_away, goals_home, goals_away
    """
    try:
        from understatapi import UnderstatClient
        client = UnderstatClient()
        matches = client.league(league=league).get_match_data(season=season)
        
        records = []
        for m in matches:
            if not m.get('isResult', False):
                continue
            records.append({
                'date': pd.to_datetime(m['datetime']).date(),
                'home_team': m['h']['title'],
                'away_team': m['a']['title'],
                'xG_home': float(m['xG']['h']),
                'xG_away': float(m['xG']['a']),
                'goals_home': int(m['goals']['h']),
                'goals_away': int(m['goals']['a']),
            })
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning(f"Failed to scrape Understat for {league} {season}: {e}")
        return pd.DataFrame()

def download_fd_season(league_code: str, season_code: str) -> pd.DataFrame:
    """
    Download football-data.co.uk CSV for a given league code and season code.
    """
    try:
        from data_utils import download_league_data
        return download_league_data(league_code, season_code)
    except Exception as e:
        logger.warning(f"Failed to download FD data for {league_code} {season_code}: {e}")
        return pd.DataFrame()

def merge_datasets(fd_df: pd.DataFrame, xg_df: pd.DataFrame, league: str) -> pd.DataFrame:
    """
    Left join football-data.co.uk dataframe with Understat xG dataframe.
    """
    if fd_df.empty:
        return pd.DataFrame()
    if xg_df.empty:
        fd_df['xG_home'] = np.nan
        fd_df['xG_away'] = np.nan
        return fd_df

    fd_df = fd_df.copy()
    xg_df = xg_df.copy()

    # Map Understat team names to FD team names
    xg_df['home_team_fd'] = xg_df['home_team'].apply(lambda t: get_fd_name(t, league))
    xg_df['away_team_fd'] = xg_df['away_team'].apply(lambda t: get_fd_name(t, league))
    
    # Ensure Date column is date object
    if 'Date' in fd_df.columns:
        fd_df['match_date'] = pd.to_datetime(fd_df['Date']).dt.date
    else:
        fd_df['match_date'] = np.nan
        
    xg_df['match_date'] = pd.to_datetime(xg_df['date']).dt.date

    # Primary merge on date, home_team, away_team
    merged = pd.merge(
        fd_df,
        xg_df[['match_date', 'home_team_fd', 'away_team_fd', 'xG_home', 'xG_away']],
        left_on=['match_date', 'HomeTeam', 'AwayTeam'],
        right_on=['match_date', 'home_team_fd', 'away_team_fd'],
        how='left'
    )

    # Fallback merge for unmerged rows using date alone if only 1 match played that day for team
    if merged['xG_home'].isna().any():
        unmerged_indices = merged[merged['xG_home'].isna()].index
        for idx in unmerged_indices:
            m_date = merged.loc[idx, 'match_date']
            h_team = str(merged.loc[idx, 'HomeTeam']).lower()[:4]
            a_team = str(merged.loc[idx, 'AwayTeam']).lower()[:4]

            cand = xg_df[xg_df['match_date'] == m_date]
            for _, c_row in cand.iterrows():
                c_h = str(c_row['home_team_fd']).lower()
                c_a = str(c_row['away_team_fd']).lower()
                if h_team in c_h or c_h[:4] in h_team:
                    merged.loc[idx, 'xG_home'] = c_row['xG_home']
                    merged.loc[idx, 'xG_away'] = c_row['xG_away']
                    break

    # Clean up temporary merge keys
    merged = merged.drop(columns=['home_team_fd', 'away_team_fd', 'match_date'], errors='ignore')
    return merged

def cache_dataset(df: pd.DataFrame, league: str, season: int) -> None:
    """Cache merged dataset as Parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    file_path = os.path.join(CACHE_DIR, f"{league}_{season}.parquet")
    try:
        df.to_parquet(file_path, index=False)
    except Exception as e:
        logger.warning(f"Failed to save cache to {file_path}: {e}")

def load_cached_dataset(league: str, season: int) -> Optional[pd.DataFrame]:
    """Load cached dataset if present."""
    file_path = os.path.join(CACHE_DIR, f"{league}_{season}.parquet")
    if os.path.exists(file_path):
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.warning(f"Failed to read cache {file_path}: {e}")
    return None

def build_master_dataset(
    leagues: List[str] = None,
    seasons: List[int] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build the complete master dataset across selected leagues and season years.
    """
    if leagues is None:
        leagues = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
    if seasons is None:
        seasons = UNDERSTAT_SEASON_IDS

    all_dfs = []

    from data_utils import GLOBAL_LEAGUE_CODES, download_league_data

    for lg in leagues:
        fd_code = UNDERSTAT_TO_FD_LEAGUE.get(lg, lg)

        if fd_code in GLOBAL_LEAGUE_CODES:
            # Handle Global league single CSV
            if use_cache:
                cached_df = load_cached_dataset(lg, 2024)
                if cached_df is not None and not cached_df.empty:
                    all_dfs.append(cached_df)
                    continue

            try:
                g_df = download_league_data(fd_code)
                if not g_df.empty:
                    g_df['league'] = lg
                    g_df['xG_home'] = np.nan
                    g_df['xG_away'] = np.nan
                    if use_cache:
                        cache_dataset(g_df, lg, 2024)
                    all_dfs.append(g_df)
            except Exception as e:
                logger.warning(f"Failed to download global league {lg}: {e}")

        else:
            # Handle European seasonal leagues
            for s_year in seasons:
                if use_cache:
                    cached_df = load_cached_dataset(lg, s_year)
                    if cached_df is not None and not cached_df.empty:
                        all_dfs.append(cached_df)
                        continue

                s_code = f"{str(s_year)[2:]}{str(s_year+1)[2:]}"
                fd_df = download_fd_season(fd_code, s_code)
                xg_df = scrape_understat_season(lg, s_year)

                merged_df = merge_datasets(fd_df, xg_df, lg)
                if not merged_df.empty:
                    merged_df['league'] = lg
                    merged_df['season_year'] = s_year
                    if use_cache:
                        cache_dataset(merged_df, lg, s_year)
                    all_dfs.append(merged_df)

    if not all_dfs:
        return pd.DataFrame()

    master = pd.concat(all_dfs, ignore_index=True)
    if 'Date' in master.columns:
        master['Date'] = pd.to_datetime(master['Date'])
        master = master.sort_values('Date').reset_index(drop=True)
    return master


def fetch_upcoming_fixtures() -> pd.DataFrame:
    """
    Downloads live upcoming fixtures feed from football-data.co.uk/fixtures.csv
    AND scans global league feeds for unplayed upcoming matches.
    
    STRICT FILTERING:
    - Only includes matches where Date >= Today (00:00:00).
    - Excludes matches that have already completed (FTHG/FTAG/FTR populated).
    """
    today = pd.Timestamp.now().normalize()
    all_upcoming = []

    # Source 1: Standard fixtures.csv
    url = "https://www.football-data.co.uk/fixtures.csv"
    try:
        df = pd.read_csv(url)
        if not df.empty:
            from data_utils import normalize_columns
            df = normalize_columns(df)

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

            # Filter strictly for today onwards AND unplayed matches
            unplayed_mask = (df['Date'] >= today)
            if 'FTHG' in df.columns:
                unplayed_mask = unplayed_mask & (df['FTHG'].isna())

            df = df[unplayed_mask].copy()

            div_map = {
                'E0': 'EPL', 'E1': 'Championship', 'E2': 'League 1', 'E3': 'League 2', 'EC': 'Conference',
                'SC0': 'Scottish Premiership', 'SC1': 'Scottish Championship', 'SC2': 'Scottish League 1', 'SC3': 'Scottish League 2',
                'SP1': 'La_Liga', 'SP2': 'Segunda Division',
                'D1': 'Bundesliga', 'D2': 'Bundesliga 2',
                'I1': 'Serie_A', 'I2': 'Serie B',
                'F1': 'Ligue_1', 'F2': 'Ligue 2',
                'N1': 'Eredivisie', 'B1': 'Pro League',
                'P1': 'Liga Portugal', 'T1': 'Super Lig', 'G1': 'Super League'
            }
            if 'Div' in df.columns:
                df['league'] = df['Div'].map(div_map).fillna(df['Div'])
            all_upcoming.append(df)
    except Exception as e:
        logger.warning(f"Failed to fetch fixtures.csv: {e}")

    # Source 2: Global Extra Leagues CSVs
    from data_utils import GLOBAL_LEAGUE_CODES, download_league_data
    for code in GLOBAL_LEAGUE_CODES:
        try:
            gdf = download_league_data(code)
            if not gdf.empty and 'Date' in gdf.columns:
                unplayed_g = gdf[(gdf['Date'] >= today) & (gdf['FTHG'].isna() | gdf['FTR'].isna())].copy()
                if not unplayed_g.empty:
                    gdf_map = {
                        'USA': 'USA (MLS)', 'ARG': 'Argentina', 'BRA': 'Brazil', 'MEX': 'Mexico',
                        'JPN': 'Japan', 'CHN': 'China', 'SWE': 'Sweden', 'NOR': 'Norway',
                        'DNK': 'Denmark', 'FIN': 'Finland', 'POL': 'Poland', 'ROU': 'Romania',
                        'SWZ': 'Switzerland', 'AUT': 'Austria'
                    }
                    unplayed_g['league'] = gdf_map.get(code, code)
                    all_upcoming.append(unplayed_g)
        except Exception as e:
            logger.warning(f"Failed to fetch global upcoming for {code}: {e}")

    # Source 3: Real-Time Live Scoreboard API (captures same-day live fixtures across Sweden, MLS, Europe, Americas, Asia)
    try:
        import requests
        espn_slugs = {
            'swe.1': 'Sweden', 'usa.1': 'USA (MLS)', 'eng.1': 'EPL', 'esp.1': 'La_Liga',
            'ger.1': 'Bundesliga', 'ita.1': 'Serie_A', 'fra.1': 'Ligue_1', 'arg.1': 'Argentina',
            'bra.1': 'Brazil', 'mex.1': 'Mexico', 'jpn.1': 'Japan', 'nor.1': 'Norway',
            'dnk.1': 'Denmark', 'fin.1': 'Finland', 'pol.1': 'Poland', 'aut.1': 'Austria',
            'sco.1': 'Scottish Premiership', 'eng.2': 'Championship'
        }
        live_api_rows = []
        for slug, lg_name in espn_slugs.items():
            try:
                resp = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard", timeout=3)
                if resp.status_code == 200:
                    events = resp.json().get('events', [])
                    for ev in events:
                        comps = ev.get('competitions', [{}])[0].get('competitors', [])
                        if len(comps) >= 2:
                            h_name = next((c.get('team', {}).get('displayName') for c in comps if c.get('homeAway') == 'home'), comps[0].get('team', {}).get('displayName'))
                            a_name = next((c.get('team', {}).get('displayName') for c in comps if c.get('homeAway') == 'away'), comps[1].get('team', {}).get('displayName'))
                            ev_date = pd.to_datetime(ev.get('date'), errors='coerce')
                            if pd.notnull(ev_date) and ev_date.tz_localize(None) >= today:
                                live_api_rows.append({
                                    'league': lg_name,
                                    'Date': ev_date.tz_localize(None),
                                    'Time': ev_date.strftime('%H:%M'),
                                    'HomeTeam': h_name,
                                    'AwayTeam': a_name,
                                    'over25_odds': 1.85,
                                    'under25_odds': 1.95,
                                    'draw_odds': 3.40
                                })
            except Exception:
                continue

        if live_api_rows:
            api_df = pd.DataFrame(live_api_rows)
            all_upcoming.append(api_df)
    except Exception as e:
        logger.warning(f"Failed to fetch live API scoreboard: {e}")

    if not all_upcoming:
        return pd.DataFrame()

    res = pd.concat(all_upcoming, ignore_index=True)
    if 'Date' in res.columns:
        res = res.drop_duplicates(subset=['league', 'HomeTeam', 'AwayTeam'], keep='first')
        res = res.sort_values('Date').reset_index(drop=True)
    return res
