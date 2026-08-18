import os, requests, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_FOOTBALL_KEY", "")
headers = {"x-apisports-key": api_key}
today = pd.Timestamp.now().normalize()
tomorrow = today + pd.Timedelta(days=1)

# Check ALL Finnish leagues for today's fixtures (any status)
finnish_league_ids = [244, 245, 246, 640, 1087]  # Veikkausliiga + others

print(f"Checking Finnish leagues for {today.date()}...\n")
all_found = []
for lg_id in finnish_league_ids:
    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=headers,
        params={"league": lg_id, "date": today.strftime('%Y-%m-%d'), "timezone": "Europe/London"},
        timeout=8,
    )
    fixtures = r.json().get("response", [])
    if fixtures:
        print(f"League {lg_id}: {len(fixtures)} fixtures")
        for fix in fixtures:
            ft = fix.get("fixture", {})
            teams = fix.get("teams", {})
            status = ft.get("status", {})
            raw_dt = ft.get("date", "")
            ev = pd.to_datetime(raw_dt, errors='coerce')
            ev_str = ev.replace(tzinfo=None).strftime('%H:%M BST') if pd.notna(ev) and ev.tzinfo else str(ev)
            h = teams.get('home', {}).get('name', '?')
            a = teams.get('away', {}).get('name', '?')
            print(f"  {h} vs {a}  |  {ev_str}  |  {status.get('short')} - {status.get('long')}")
            all_found.append((lg_id, h, a))
    else:
        print(f"League {lg_id}: 0 fixtures today")

print(f"\nTotal Finnish fixtures found today: {len(all_found)}")
print(f"Requests remaining: {r.headers.get('x-ratelimit-requests-remaining', '?')}")
