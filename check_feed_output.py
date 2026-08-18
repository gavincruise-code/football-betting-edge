import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
import ml.data_ingestion as di

print("Fetching upcoming fixtures...")
df = di.fetch_upcoming_fixtures()
print(f"Total returned: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

today = pd.Timestamp.now().normalize()
mask = df['Date'] >= today
today_df = df[mask].copy() if 'Date' in df.columns else df

# Show Championship and Eredivisie
for lg in ['Championship', 'Eredivisie']:
    rows = today_df[today_df.get('league', pd.Series(dtype=str)) == lg] if 'league' in today_df.columns else pd.DataFrame()
    print(f"=== {lg}: {len(rows)} fixtures ===")
    if not rows.empty:
        cols = ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'league']
        cols = [c for c in cols if c in rows.columns]
        print(rows[cols].to_string())
    print()

# Also show any Wolves or Telstar entries
print("=== Any 'Wolv' or 'Telst' in feed ===")
if 'HomeTeam' in df.columns:
    mask2 = df['HomeTeam'].str.contains('Wolv|Telst', case=False, na=False) | \
            df['AwayTeam'].str.contains('Wolv|Telst', case=False, na=False)
    print(df[mask2][['Date','Time','HomeTeam','AwayTeam','league']].to_string() if mask2.any() else "None found")
