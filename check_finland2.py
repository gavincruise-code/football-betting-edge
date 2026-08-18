import requests
import pandas as pd

# Also try the fixtures.csv from football-data.co.uk - does it cover Finland?
from data_utils import GLOBAL_LEAGUE_CODES, download_league_data

print(f"GLOBAL_LEAGUE_CODES: {GLOBAL_LEAGUE_CODES}")
print()

# Try downloading Finland data
try:
    df = download_league_data('FIN')
    print(f"FIN via download_league_data: {len(df)} rows")
    if not df.empty and 'Date' in df.columns:
        today = pd.Timestamp.now().normalize()
        upcoming = df[df['Date'] >= today]
        print(f"Upcoming FIN fixtures: {len(upcoming)}")
        print(upcoming[['Date','HomeTeam','AwayTeam']].head(10).to_string())
except Exception as e:
    print(f"FIN download failed: {e}")

print()

# Check football-data.co.uk fixtures.csv for Finland
try:
    resp = requests.get("https://www.football-data.co.uk/fixtures.csv", timeout=10)
    from io import StringIO
    raw_df = pd.read_csv(StringIO(resp.text))
    print(f"fixtures.csv Div values: {sorted(raw_df['Div'].dropna().unique().tolist())}")
    # Finland won't be in here - it's only European leagues they cover
except Exception as e:
    print(f"fixtures.csv check failed: {e}")

print()

# Try ESPN with alternative slug patterns
for test_slug in ['fin.1', 'fin.veikkausliiga', 'soccer.fin.1']:
    try:
        r = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/{test_slug}/scoreboard",
            timeout=5
        )
        events = r.json().get('events', [])
        print(f"ESPN slug '{test_slug}': {r.status_code}, {len(events)} events")
    except Exception as e:
        print(f"ESPN slug '{test_slug}': ERROR {e}")
