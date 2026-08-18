import os, requests, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_FOOTBALL_KEY", "")
headers = {"x-apisports-key": api_key}
today = pd.Timestamp.now().normalize()

# Try with explicit season=2026
for season in [2026, 2025, None]:
    params = {
        "league": 244,
        "date": today.strftime('%Y-%m-%d'),
        "timezone": "Europe/London",
    }
    if season:
        params["season"] = season

    r = requests.get("https://v3.football.api-sports.io/fixtures", headers=headers, params=params, timeout=8)
    data = r.json()
    fixtures = data.get("response", [])
    errors = data.get("errors", {})
    print(f"season={season}: {len(fixtures)} fixtures | errors={errors}")
    for fix in fixtures:
        ft = fix.get("fixture", {})
        teams = fix.get("teams", {})
        status = ft.get("status", {})
        raw_dt = ft.get("date", "")
        ev = pd.to_datetime(raw_dt, errors='coerce')
        ev_str = ev.replace(tzinfo=None).strftime('%H:%M BST') if pd.notna(ev) and ev.tzinfo else str(ev)
        h = teams.get('home', {}).get('name', '?')
        a = teams.get('away', {}).get('name', '?')
        print(f"  {h} vs {a}  |  {ev_str}  |  {status.get('short')}")

print()

# Also search by team name: VPS
r2 = requests.get("https://v3.football.api-sports.io/teams", headers=headers,
                  params={"name": "VPS", "country": "Finland"}, timeout=8)
teams_found = r2.json().get("response", [])
print(f"Teams named 'VPS' in Finland: {len(teams_found)}")
for t in teams_found:
    print(f"  ID={t['team']['id']}  {t['team']['name']}")

r3 = requests.get("https://v3.football.api-sports.io/teams", headers=headers,
                  params={"name": "TPS", "country": "Finland"}, timeout=8)
for t in r3.json().get("response", []):
    print(f"  ID={t['team']['id']}  {t['team']['name']}")

print(f"\nRequests remaining: {r.headers.get('x-ratelimit-requests-remaining', '?')}")
