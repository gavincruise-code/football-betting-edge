import os, requests, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_FOOTBALL_KEY", "")
headers = {"x-apisports-key": api_key}
today = pd.Timestamp.now().normalize()

# 1. Check what league 244 actually is
print("=== League 244 info ===")
r = requests.get("https://v3.football.api-sports.io/leagues", headers=headers,
                 params={"id": 244}, timeout=8)
for lg in r.json().get("response", []):
    print(f"  {lg['league']['name']} ({lg['country']['name']}) — ID {lg['league']['id']}")

# 2. Search for Finland leagues
print("\n=== All Finland leagues ===")
r2 = requests.get("https://v3.football.api-sports.io/leagues", headers=headers,
                  params={"country": "Finland", "season": 2024}, timeout=8)
for lg in r2.json().get("response", []):
    info = lg['league']
    cov  = lg.get('seasons', [{}])[-1].get('coverage', {}).get('fixtures', {})
    print(f"  ID={info['id']:5}  {info['name']:<40}  Type={info['type']}")

print()
print("Requests remaining:", r2.headers.get('x-ratelimit-requests-remaining', '?'))
