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

def impute_missing_ou_odds(df: pd.DataFrame, default_margin: float = 0.05) -> pd.DataFrame:
    """
    Impute missing Over/Under 2.5 odds for extra/global leagues (ARG, CHN, USA, BRA, MEX, JPN, etc.)
    using a zero-lookahead rolling goal Poisson model with standard 5% bookmaker overround.
    """
    if df.empty:
        return df

    df = df.copy()
    if 'over25_odds' not in df.columns:
        df['over25_odds'] = np.nan
    if 'under25_odds' not in df.columns:
        df['under25_odds'] = np.nan

    missing = df['over25_odds'].isna() | (df['over25_odds'] <= 1.0)
    if not missing.any():
        return df

    try:
        from poisson_engine import prob_over_n_goals
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            tot_g = (df['FTHG'].fillna(1.2) + df['FTAG'].fillna(1.0)).values
        else:
            tot_g = np.full(len(df), 2.4)

        shifted_g = pd.Series(tot_g).shift(1)
        rolling_g = shifted_g.rolling(50, min_periods=5).mean().fillna(2.4).values

        o25_p = np.array([prob_over_n_goals(lam, 2) for lam in rolling_g])
        u25_p = np.maximum(0.01, 1.0 - o25_p)

        est_o = np.round(1.0 / (o25_p * (1.0 + default_margin / 2.0)), 2)
        est_u = np.round(1.0 / (u25_p * (1.0 + default_margin / 2.0)), 2)

        df.loc[missing, 'over25_odds'] = np.maximum(1.10, est_o[missing])
        df.loc[missing, 'under25_odds'] = np.maximum(1.10, est_u[missing])
    except Exception as e:
        logger.warning(f"Error imputing missing O/U odds: {e}")

    return df

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
                    cached_df = impute_missing_ou_odds(cached_df)
                    all_dfs.append(cached_df)
                    continue

            try:
                g_df = download_league_data(fd_code)
                if not g_df.empty:
                    g_df['league'] = lg
                    g_df['xG_home'] = np.nan
                    g_df['xG_away'] = np.nan
                    g_df = impute_missing_ou_odds(g_df)
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
                        cached_df = impute_missing_ou_odds(cached_df)
                        all_dfs.append(cached_df)
                        continue

                s_code = f"{str(s_year)[2:]}{str(s_year+1)[2:]}"
                fd_df = download_fd_season(fd_code, s_code)
                xg_df = scrape_understat_season(lg, s_year)

                merged_df = merge_datasets(fd_df, xg_df, lg)
                if not merged_df.empty:
                    merged_df['league'] = lg
                    merged_df['season_year'] = s_year
                    merged_df = impute_missing_ou_odds(merged_df)
                    if use_cache:
                        cache_dataset(merged_df, lg, s_year)
                    all_dfs.append(merged_df)

    if not all_dfs:
        return pd.DataFrame()

    master = pd.concat(all_dfs, ignore_index=True)
    if 'Date' in master.columns:
        master['Date'] = pd.to_datetime(master['Date'])
        master = master.sort_values('Date').reset_index(drop=True)
        if 'season_year' not in master.columns:
            master['season_year'] = master['Date'].dt.year
    master = impute_missing_ou_odds(master)
    return master


def fetch_upcoming_fixtures() -> pd.DataFrame:
    """
    Downloads live upcoming fixtures from:
      Source 1 – football-data.co.uk/fixtures.csv  (all European leagues)
      Source 2 – Global league CSVs                (MLS, Brazil, Japan, etc.)
      Source 3 – ESPN scoreboard API               (same-day / next-day, incl. UEFA)

    Strict filtering:
      - Date >= today (no past matches)
      - Date <= today + 7 days (no fixtures far in future)
      - Excludes rows where FTHG/FTR is already filled (match finished)
    """
    today    = pd.Timestamp.now().normalize()
    max_date = today + pd.Timedelta(days=7)
    all_upcoming = []

    # ── Source 1: football-data.co.uk/fixtures.csv ────────────────────────────
    try:
        import requests
        url = "https://www.football-data.co.uk/fixtures.csv"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            from io import StringIO
            from data_utils import normalize_columns
            df = pd.read_csv(StringIO(resp.text))
            df = normalize_columns(df)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            # Keep only upcoming unplayed matches within the next 7 days
            mask = (df['Date'] >= today) & (df['Date'] <= max_date)
            if 'FTHG' in df.columns:
                mask = mask & df['FTHG'].isna()
            df = df[mask].copy()
            div_map = {
                'E0': 'EPL',          'E1': 'Championship',     'E2': 'League 1',
                'E3': 'League 2',     'EC': 'Conference',
                'SC0': 'Scottish Premiership', 'SC1': 'Scottish Championship',
                'SC2': 'Scottish League 1',    'SC3': 'Scottish League 2',
                'SP1': 'La_Liga',     'SP2': 'Segunda Division',
                'D1': 'Bundesliga',   'D2': 'Bundesliga 2',
                'I1': 'Serie_A',      'I2': 'Serie B',
                'F1': 'Ligue_1',      'F2': 'Ligue 2',
                'N1': 'Eredivisie',   'B1': 'Pro League',
                'P1': 'Liga Portugal','T1': 'Super Lig',         'G1': 'Super League',
            }
            if 'Div' in df.columns:
                df['league'] = df['Div'].map(div_map).fillna(df['Div'])
            # Drop rows where league mapped to NaN (blank Div rows in fixtures.csv)
            if 'league' in df.columns:
                df = df[df['league'].notna() & (df['league'].astype(str).str.strip() != '')]
            if not df.empty:
                all_upcoming.append(df)
    except Exception as e:
        logger.warning(f"Failed to fetch fixtures.csv: {e}")

    # ── Source 2: Global Extra League CSVs ────────────────────────────────────
    from data_utils import GLOBAL_LEAGUE_CODES, download_league_data
    gdf_map = {
        'USA': 'USA (MLS)', 'ARG': 'Argentina', 'BRA': 'Brazil', 'MEX': 'Mexico',
        'JPN': 'Japan',     'CHN': 'China',     'SWE': 'Sweden', 'NOR': 'Norway',
        'DNK': 'Denmark',   'FIN': 'Finland',   'POL': 'Poland', 'ROU': 'Romania',
        'SWZ': 'Switzerland','AUT': 'Austria',  'IRL': 'Ireland','RUS': 'Russia',
    }
    for code in GLOBAL_LEAGUE_CODES:
        try:
            gdf = download_league_data(code)
            if gdf.empty or 'Date' not in gdf.columns:
                continue
            fthg_col = 'FTHG' if 'FTHG' in gdf.columns else None
            ftr_col  = 'FTR'  if 'FTR'  in gdf.columns else None
            mask = (gdf['Date'] >= today) & (gdf['Date'] <= max_date)
            if fthg_col:
                mask = mask & gdf[fthg_col].isna()
            elif ftr_col:
                mask = mask & gdf[ftr_col].isna()
            unplayed = gdf[mask].copy()
            if not unplayed.empty:
                unplayed['league'] = gdf_map.get(code, code)
                all_upcoming.append(unplayed)
        except Exception as e:
            logger.warning(f"Failed to fetch global upcoming for {code}: {e}")

    # ── Source 3: ESPN Scoreboard API (today's & tomorrow's fixtures) ─────────
    try:
        espn_slugs = {
            # ── England ────────────────────────────────────────────────────
            'eng.1':  'EPL',              'eng.2':  'Championship',
            'eng.3':  'League 1',         'eng.4':  'League 2',
            'eng.5':  'Conference',
            # ── Scotland ───────────────────────────────────────────────────
            'sco.1':  'Scottish Premiership', 'sco.2': 'Scottish Championship',
            # ── Spain ──────────────────────────────────────────────────────
            'esp.1':  'La_Liga',          'esp.2':  'Segunda Division',
            # ── Germany ────────────────────────────────────────────────────
            'ger.1':  'Bundesliga',       'ger.2':  'Bundesliga 2',
            # ── Italy ──────────────────────────────────────────────────────
            'ita.1':  'Serie_A',          'ita.2':  'Serie B',
            # ── France ─────────────────────────────────────────────────────
            'fra.1':  'Ligue_1',          'fra.2':  'Ligue 2',
            # ── Other Europe (football-data.co.uk) ─────────────────────────
            'ned.1':  'Eredivisie',       'bel.1':  'Belgium',
            'por.1':  'Liga Portugal',    'tur.1':  'Super Lig',
            'gre.1':  'Super League Greece',
            # ── Other Europe (API-Football) ─────────────────────────────────
            'swe.1':  'Sweden',           'nor.1':  'Norway',
            'dnk.1':  'Denmark',          'fin.1':  'Finland',
            'pol.1':  'Poland',           'aut.1':  'Austria',
            'sui.1':  'Switzerland',      'rou.1':  'Romania',
            'cro.1':  'Croatia',          'rus.1':  'Russia',
            'srb.1':  'Serbia',           'cze.1':  'Czech Republic',
            'ukr.1':  'Ukraine',          'svk.1':  'Slovakia',
            'hun.1':  'Hungary',          'isr.1':  'Israel',
            # ── Asia ───────────────────────────────────────────────────────
            'chn.1':  'China',            'jpn.1':  'Japan',
            'kor.1':  'South Korea',      'sau.1':  'Saudi Arabia',
            'ind.1':  'Indian Super League', 'tha.1': 'Thailand',
            # ── Americas ───────────────────────────────────────────────────
            'usa.1':  'USA (MLS)',         'arg.1':  'Argentina',
            'bra.1':  'Brazil',            'mex.1':  'Mexico',
            'col.1':  'Colombia',          'chl.1':  'Chile',
            'uru.1':  'Uruguay',           'ecu.1':  'Ecuador',
            'per.1':  'Peru',
            # ── Oceania ────────────────────────────────────────────────────
            'aus.1':  'Australia',
            # ── Africa ─────────────────────────────────────────────────────
            'egy.1':  'Egypt',            'mar.1':  'Morocco',
            'rsa.1':  'South Africa',
            # ── UEFA ───────────────────────────────────────────────────────
            'uefa.champions':      'UEFA Champions League',
            'uefa.champions_qual': 'UEFA CL Qualifying',
            'uefa.europa':         'UEFA Europa League',
            'uefa.europa_qual':    'UEFA EL Qualifying',
            'uefa.europa.conf':    'UEFA Conference League',
            'uefa.europa.conf_qual': 'UEFA ECL Qualifying',
        }
        espn_rows = []
        for slug, lg_name in espn_slugs.items():
            try:
                r = requests.get(
                    f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard",
                    timeout=4
                )
                if r.status_code != 200:
                    continue
                for ev in r.json().get('events', []):
                    # Only include events not yet completed
                    status_type = ev.get('status', {}).get('type', {}).get('state', '')
                    if status_type == 'post':   # already finished
                        continue
                    comps = ev.get('competitions', [{}])[0].get('competitors', [])
                    if len(comps) < 2:
                        continue
                    h_name = next(
                        (c.get('team', {}).get('displayName') for c in comps if c.get('homeAway') == 'home'),
                        comps[0].get('team', {}).get('displayName')
                    )
                    a_name = next(
                        (c.get('team', {}).get('displayName') for c in comps if c.get('homeAway') == 'away'),
                        comps[1].get('team', {}).get('displayName')
                    )
                    raw_dt = ev.get('date', '')
                    ev_date = pd.to_datetime(raw_dt, utc=True, errors='coerce')
                    if pd.isnull(ev_date):
                        continue
                    ev_date = ev_date.tz_localize(None)
                    # Strict window: today to +7 days
                    if not (today <= ev_date.normalize() <= max_date):
                        continue
                    espn_rows.append({
                        'league':        lg_name,
                        'Date':          ev_date.normalize(),
                        'Time':          ev_date.strftime('%H:%M'),
                        'HomeTeam':      h_name,
                        'AwayTeam':      a_name,
                        'over25_odds':   np.nan,   # No real odds from ESPN — will be filled by Betfair or suppressed
                        'under25_odds':  np.nan,
                        'draw_odds':     np.nan,
                    })
            except Exception:
                continue
        if espn_rows:
            all_upcoming.append(pd.DataFrame(espn_rows))
    except Exception as e:
        logger.warning(f"Failed to fetch ESPN scoreboard: {e}")

    # ── Combine, deduplicate, sort ─────────────────────────────────────────────
    if not all_upcoming:
        return pd.DataFrame()

    res = pd.concat(all_upcoming, ignore_index=True)

    # Deduplicate by fuzzy 5-char team key
    if 'HomeTeam' in res.columns and 'AwayTeam' in res.columns:
        res['h_key'] = res['HomeTeam'].apply(
            lambda x: ''.join(c for c in str(x) if c.isalnum()).lower()[:5] if pd.notnull(x) else ''
        )
        res['a_key'] = res['AwayTeam'].apply(
            lambda x: ''.join(c for c in str(x) if c.isalnum()).lower()[:5] if pd.notnull(x) else ''
        )
        res = res.drop_duplicates(subset=['h_key', 'a_key'], keep='first').drop(columns=['h_key', 'a_key'])

    if 'Date' in res.columns:
        res = res.sort_values('Date').reset_index(drop=True)

    return res
