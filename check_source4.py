import os, requests, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_FOOTBALL_KEY", "")
print(f"API key set: {bool(api_key)}")

if not api_key:
    print("No API key - Source 4 won't run")
else:
    today = pd.Timestamp.now().normalize()
    resp = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers={"x-apisports-key": api_key},
        params={
            "league": 244,          # Finland
            "from": today.strftime('%Y-%m-%d'),
            "to": (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            "status": "NS",
            "timezone": "Europe/London",
        },
        timeout=8,
    )
    print(f"Status: {resp.status_code}")
    data = resp.json()
    fixtures = data.get("response", [])
    print(f"Fixtures returned: {len(fixtures)}")
    for fix in fixtures:
        ft = fix.get("fixture", {})
        teams = fix.get("teams", {})
        raw_dt = ft.get("date", "")
        print(f"  Raw date: {raw_dt}")
        ev = pd.to_datetime(raw_dt, errors='coerce')
        print(f"  pd.to_datetime result: {ev}  tzinfo: {ev.tzinfo}")
        # Test the bug
        try:
            result = ev.tz_localize(None)
            print(f"  tz_localize(None) -> {result}  [WORKED]")
        except Exception as e:
            print(f"  tz_localize(None) -> ERROR: {e}  [BUG CONFIRMED]")
            result = ev.replace(tzinfo=None)
            print(f"  replace(tzinfo=None) -> {result}  [FIXED]")
        print(f"  {teams.get('home',{}).get('name')} vs {teams.get('away',{}).get('name')}")
