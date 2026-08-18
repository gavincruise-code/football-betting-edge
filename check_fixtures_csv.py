import requests, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
from io import StringIO
from data_utils import normalize_columns

resp = requests.get("https://www.football-data.co.uk/fixtures.csv", timeout=10)
df = pd.read_csv(StringIO(resp.text))
df = normalize_columns(df)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

today = pd.Timestamp.now().normalize()
mask = df['Date'] >= today
if 'FTHG' in df.columns:
    mask = mask & df['FTHG'].isna()
upcoming = df[mask].copy()

div_map = {
    'E0': 'EPL', 'E1': 'Championship', 'E2': 'League 1', 'E3': 'League 2', 'EC': 'Conference',
    'SC0': 'Scottish Premiership', 'SC1': 'Scottish Championship',
    'SP1': 'La_Liga', 'SP2': 'Segunda Division', 'D1': 'Bundesliga', 'D2': 'Bundesliga 2',
    'I1': 'Serie_A', 'I2': 'Serie B', 'F1': 'Ligue_1', 'F2': 'Ligue 2',
    'N1': 'Eredivisie', 'B1': 'Pro League', 'P1': 'Liga Portugal', 'T1': 'Super Lig', 'G1': 'Super League',
}
if 'Div' in upcoming.columns:
    upcoming['league'] = upcoming['Div'].map(div_map).fillna(upcoming['Div'])

# Does it have a Time column?
print(f"Columns: {list(upcoming.columns)}")
print(f"'Time' column present: {'Time' in upcoming.columns}")
print()

# Show Championship and Eredivisie rows
for lg in ['Championship', 'Eredivisie']:
    rows = upcoming[upcoming['league'] == lg] if 'league' in upcoming.columns else pd.DataFrame()
    print(f"{lg}: {len(rows)} rows in fixtures.csv")
    if not rows.empty:
        print(rows[['Date', 'HomeTeam', 'AwayTeam', 'league'] + (['Time'] if 'Time' in rows.columns else [])].to_string())
    print()

# Show deduplication keys for Wolves and Telstar
print("Dedup keys for Wolves/Blackburn:")
for t in ['Wolverhampton', 'Blackburn', 'Telstar', 'Sparta']:
    key = ''.join(c for c in t if c.isalnum()).lower()[:5]
    print(f"  {t} -> '{key}'")
