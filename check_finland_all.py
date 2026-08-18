import os, requests, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_FOOTBALL_KEY", "")
today = pd.Timestamp.now().normalize()

# Try without status filter to see all Finland fixtures today
resp = requests.get(
    "https://v3.football.api-sports.io/fixtures",
    headers={"x-apisports-key": api_key},
    params={
        "league": 244,
        "date": today.strftime('%Y-%m-%d'),
        "timezone": "Europe/London",
    },
    timeout=8,
)
data = resp.json()
fixtures = data.get("response", [])
print(f"Finland today (all statuses): {len(fixtures)} fixtures")
for fix in fixtures:
    ft = fix.get("fixture", {})
    teams = fix.get("teams", {})
    status = ft.get("status", {})
    raw_dt = ft.get("date", "")
    ev = pd.to_datetime(raw_dt, errors='coerce')
    # Check tz_localize vs replace
    if pd.notna(ev) and ev.tzinfo is not None:
        ev_stripped = ev.replace(tzinfo=None)
    else:
        ev_stripped = ev
    h = teams.get('home', {}).get('name', '?')
    a = teams.get('away', {}).get('name', '?')
    print(f"  {h} vs {a}")
    print(f"    Date: {raw_dt} -> stripped: {ev_stripped}")
    print(f"    Status: {status.get('short')} ({status.get('long')})")

# Also check if "NS" is the right code by listing fixture statuses
print()
print("Rate limit remaining:", resp.headers.get('x-ratelimit-requests-remaining', '?'))
