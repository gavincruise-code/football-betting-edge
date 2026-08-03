import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

# League definitions
LEAGUES: Dict[str, Dict[str, str]] = {
    "England": {"Premier League": "E0", "Championship": "E1", "League 1": "E2", "League 2": "E3", "Conference": "EC"},
    "Scotland": {"Premiership": "SC0", "Championship": "SC1", "League 1": "SC2", "League 2": "SC3"},
    "Germany": {"Bundesliga": "D1", "Bundesliga 2": "D2"},
    "Italy": {"Serie A": "I1", "Serie B": "I2"},
    "Spain": {"La Liga": "SP1", "Segunda Division": "SP2"},
    "France": {"Ligue 1": "F1", "Ligue 2": "F2"},
    "Netherlands": {"Eredivisie": "N1"},
    "Belgium": {"Pro League": "B1"},
    "Portugal": {"Liga Portugal": "P1"},
    "Turkey": {"Super Lig": "T1"},
    "Greece": {"Super League": "G1"},
    "USA": {"Major League Soccer (MLS)": "USA"},
    "Argentina": {"Primera Division": "ARG"},
    "Brazil": {"Serie A": "BRA"},
    "Mexico": {"Liga MX": "MEX"},
    "Japan": {"J-League": "JPN"},
    "China": {"Super League": "CHN"},
    "Sweden": {"Allsvenskan": "SWE"},
    "Norway": {"Eliteserien": "NOR"},
    "Denmark": {"Superligaen": "DNK"},
    "Finland": {"Veikkausliiga": "FIN"},
    "Poland": {"Ekstraklasa": "POL"},
    "Romania": {"Liga 1": "ROU"},
    "Switzerland": {"Super League": "SWZ"},
    "Austria": {"Bundesliga": "AUT"},
}

GLOBAL_LEAGUE_CODES = ["USA", "ARG", "BRA", "MEX", "JPN", "CHN", "SWE", "NOR", "DNK", "FIN", "POL", "ROU", "SWZ", "AUT"]

def get_available_seasons(start_year: int = 2005, end_year: int = 2025) -> List[str]:
    """Returns list of season strings like '2024-25', '2023-24', etc."""
    seasons = []
    for yr in range(end_year - 1, start_year - 1, -1):
        ny = str(yr + 1)[2:]
        seasons.append(f"{yr}-{ny}")
    return seasons

def season_to_code(season: str) -> str:
    """Convert '2024-25' to '2425'."""
    parts = season.split('-')
    if len(parts) == 2:
        return parts[0][2:] + parts[1]
    return season

def download_league_data(league_code: str, season_code: str = "") -> pd.DataFrame:
    """Download CSV from football-data.co.uk. Handles both European seasonal and Global extra league CSVs."""
    if league_code in GLOBAL_LEAGUE_CODES:
        url = f"https://www.football-data.co.uk/new/{league_code}.csv"
    else:
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    
    try:
        df = pd.read_csv(url)
        return normalize_columns(df)
    except Exception as e:
        raise ValueError(f"Failed to download or parse {url}: {e}")

def load_csv(file_or_path) -> pd.DataFrame:
    """Parse uploaded CSV with flexible date parsing and column normalization."""
    df = pd.read_csv(file_or_path)
    return normalize_columns(df)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map variant column names (European & Global formats) to standard names."""
    df = df.copy()

    # Column name mappings for Global leagues CSV format
    rename_dict = {
        'Home': 'HomeTeam',
        'Away': 'AwayTeam',
        'HG': 'FTHG',
        'AG': 'FTAG',
        'Res': 'FTR'
    }
    df = df.rename(columns=rename_dict)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed', errors='coerce')
    
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
        df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
        df['total_goals'] = df['FTHG'] + df['FTAG']
        df['over25'] = df['total_goals'] > 2.5
    
    # over25_odds
    o25_cols = ['B365>2.5', 'BbAv>2.5', 'BbMx>2.5', 'P>2.5', 'Max>2.5', 'Avg>2.5', 'B365C>2.5', 'AvgC>2.5', 'MaxC>2.5', 'PSC>2.5']
    avail_o25 = [c for c in o25_cols if c in df.columns]
    if avail_o25:
        df['over25_odds'] = df[avail_o25].apply(pd.to_numeric, errors='coerce').max(axis=1)
    else:
        df['over25_odds'] = np.nan
        
    # under25_odds
    u25_cols = ['B365<2.5', 'BbAv<2.5', 'BbMx<2.5', 'P<2.5', 'Max<2.5', 'Avg<2.5', 'B365C<2.5', 'AvgC<2.5', 'MaxC<2.5', 'PSC<2.5']
    avail_u25 = [c for c in u25_cols if c in df.columns]
    if avail_u25:
        df['under25_odds'] = df[avail_u25].apply(pd.to_numeric, errors='coerce').max(axis=1)
    else:
        df['under25_odds'] = np.nan
        
    # draw_odds
    draw_cols = ['B365D', 'AvgCD', 'MaxCD', 'B365CD', 'PSCD', 'BFECD']
    avail_draw = [c for c in draw_cols if c in df.columns]
    if avail_draw:
        df['draw_odds'] = df[avail_draw].apply(pd.to_numeric, errors='coerce').max(axis=1)
    else:
        df['draw_odds'] = np.nan
        
    # home_odds, away_odds
    home_cols = ['B365H', 'AvgCH', 'MaxCH', 'B365CH', 'PSCH', 'BFECH']
    avail_home = [c for c in home_cols if c in df.columns]
    df['home_odds'] = df[avail_home].apply(pd.to_numeric, errors='coerce').max(axis=1) if avail_home else np.nan

    away_cols = ['B365A', 'AvgCA', 'MaxCA', 'B365CA', 'PSCA', 'BFECA']
    avail_away = [c for c in away_cols if c in df.columns]
    df['away_odds'] = df[avail_away].apply(pd.to_numeric, errors='coerce').max(axis=1) if avail_away else np.nan

    # Sort by date
    if 'Date' in df.columns:
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
    return df

def get_team_history(df: pd.DataFrame, team: str, before_date, n: int = 5) -> pd.DataFrame:
    """Last N matches (home or away) for team before given date."""
    hist = df[(df['Date'] < before_date) & ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))]
    return hist.tail(n)

def get_home_history(df: pd.DataFrame, team: str, before_date, n: int = 5) -> pd.DataFrame:
    """Last N HOME matches for team before given date."""
    hist = df[(df['Date'] < before_date) & (df['HomeTeam'] == team)]
    return hist.tail(n)

def get_away_history(df: pd.DataFrame, team: str, before_date, n: int = 5) -> pd.DataFrame:
    """Last N AWAY matches for team before given date."""
    hist = df[(df['Date'] < before_date) & (df['AwayTeam'] == team)]
    return hist.tail(n)

def get_h2h(df: pd.DataFrame, team_a: str, team_b: str, before_date, n: int = 5) -> pd.DataFrame:
    """Last N head-to-head matches between two teams (either direction) before date."""
    hist = df[(df['Date'] < before_date) & (
        ((df['HomeTeam'] == team_a) & (df['AwayTeam'] == team_b)) | 
        ((df['HomeTeam'] == team_b) & (df['AwayTeam'] == team_a))
    )]
    return hist.tail(n)

def compute_team_avg_goals(matches: pd.DataFrame, team: str) -> float:
    """Average total goals (scored + conceded) for team across given matches."""
    if matches.empty:
        return 0.0
    return matches['total_goals'].mean()

def get_all_flat_leagues() -> List[Tuple[str, str, str]]:
    """Returns flat list of (country, league_name, league_code) tuples."""
    flat = []
    for country, lgs in LEAGUES.items():
        for name, code in lgs.items():
            flat.append((country, name, code))
    return flat
