import os, requests, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_FOOTBALL_KEY", "")
headers = {"x-apisports-key": api_key}

# Check next 7 days for Veikkausliiga to see when rounds are scheduled
today = pd.Timestamp.now().normalize()
max_date = today + pd.Timedelta(days=7)

r = requests.get(
    "https://v3.football.api-sports.io/fixtures",
    headers=headers,
    params={
        "league": 244,
        "from": today.strftime('%Y-%m-%d'),
        "to": max_date.strftime('%Y-%m-%d'),
        "timezone": "Europe/London",
    },
    timeout=8,
)
fixtures = r.json().get("response", [])
print(f"Veikkausliiga fixtures next 7 days: {len(fixtures)}")
for fix in fixtures:
    ft = fix.get("fixture", {})
    teams = fix.get("teams", {})
    status = ft.get("status", {})
    raw_dt = ft.get("date", "")
    ev = pd.to_datetime(raw_dt, errors='coerce')
    ev_str = ev.replace(tzinfo=None).strftime('%a %d %b %H:%M BST') if pd.notna(ev) and ev.tzinfo else str(ev)
    h = teams.get('home', {}).get('name', '?')
    a = teams.get('away', {}).get('name', '?')
    print(f"  {ev_str}  |  {h} vs {a}  |  {status.get('short')}")

print(f"\nRequests remaining: {r.headers.get('x-ratelimit-requests-remaining', '?')}")
