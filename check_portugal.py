from data_utils import download_league_data
import warnings; warnings.filterwarnings('ignore')

df = download_league_data('P1')
print(f'Total P1 rows: {len(df)}')

if 'Date' in df.columns:
    years = sorted(df['Date'].dt.year.dropna().unique().tolist())
    print(f'Seasons present: {years}')

all_teams = list(df['HomeTeam'].dropna().unique()) + list(df['AwayTeam'].dropna().unique())
unique_teams = sorted(set(all_teams))
print(f'\nAll teams in dataset ({len(unique_teams)}):')
for t in unique_teams:
    matches = df[(df['HomeTeam'] == t) | (df['AwayTeam'] == t)]
    print(f'  {t}: {len(matches)} matches')
