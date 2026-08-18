import requests
import pandas as pd

# Check what ESPN returns for Finland today
slug = 'fin.1'
url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
r = requests.get(url, timeout=10)
print(f"Status: {r.status_code}")

data = r.json()
events = data.get('events', [])
print(f"Total events returned: {len(events)}")

now_utc = pd.Timestamp.now('UTC')
now_uk  = pd.Timestamp.now()  # local machine time (BST)
print(f"Now UTC: {now_utc}")
print(f"Now UK (machine local): {now_uk}")
print()

for ev in events:
    status_type = ev.get('status', {}).get('type', {}).get('state', '')
    comps = ev.get('competitions', [{}])[0].get('competitors', [])
    h_name = next((c.get('team', {}).get('displayName') for c in comps if c.get('homeAway') == 'home'), '?')
    a_name = next((c.get('team', {}).get('displayName') for c in comps if c.get('homeAway') == 'away'), '?')
    raw_dt = ev.get('date', '')

    ev_utc = pd.to_datetime(raw_dt, utc=True, errors='coerce')
    try:
        ev_bst = ev_utc.tz_convert('Europe/London').tz_localize(None)
    except Exception:
        ev_bst = ev_utc.tz_localize(None)

    kicked_off = ev_bst < now_uk if pd.notna(ev_bst) else False

    print(f"  {h_name} vs {a_name}")
    print(f"    Raw UTC: {raw_dt}")
    print(f"    BST:     {ev_bst}  ({'KICKED OFF' if kicked_off else 'UPCOMING'})")
    print(f"    Status:  {status_type}")
    print()
