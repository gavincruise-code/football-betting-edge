import requests
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

now_uk = pd.Timestamp.now()
today  = now_uk.normalize()
max_date = today + pd.Timedelta(days=7)
print(f"Now UK: {now_uk.strftime('%H:%M:%S BST')}\n")

for slug, league in [('eng.2', 'Championship'), ('ned.1', 'Eredivisie')]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
    r = requests.get(url, timeout=8)
    events = r.json().get('events', [])
    print(f"=== {league} (ESPN: {slug}) --- {len(events)} events ===")
    for ev in events:
        state = ev.get('status', {}).get('type', {}).get('state', '?')
        status_detail = ev.get('status', {}).get('type', {}).get('description', '?')
        comps = ev.get('competitions', [{}])[0].get('competitors', [])
        h = next((c.get('team', {}).get('displayName') for c in comps if c.get('homeAway') == 'home'), '?')
        a = next((c.get('team', {}).get('displayName') for c in comps if c.get('homeAway') == 'away'), '?')
        raw_dt = ev.get('date', '')
        ev_utc = pd.to_datetime(raw_dt, utc=True, errors='coerce')
        try:
            ev_bst = ev_utc.tz_convert('Europe/London').tz_localize(None)
        except Exception:
            ev_bst = ev_utc.tz_localize(None)

        in_window = today <= ev_bst.normalize() <= max_date if pd.notna(ev_bst) else False
        kicked_off = ev_bst < now_uk if pd.notna(ev_bst) else False
        post_filter = (state == 'post')

        print(f"  {h} vs {a}")
        print(f"    UTC: {raw_dt}  ->  BST: {ev_bst.strftime('%Y-%m-%d %H:%M') if pd.notna(ev_bst) else 'NaT'}")
        print(f"    State: {state} ({status_detail})")
        print(f"    Filtered by 'post' check: {post_filter}")
        print(f"    In 7-day window: {in_window}  |  Kicked off (BST): {kicked_off}")
    if not events:
        print("  (no events returned)")
    print()
