from data_utils import download_league_data
import warnings; warnings.filterwarnings('ignore')

df = download_league_data('T1')
print(f'Total T1 rows: {len(df)}')

if 'Date' in df.columns:
    years = sorted(df['Date'].dt.year.dropna().unique().tolist())
    print(f'Seasons present: {years}')

# Check for Corum FK
all_teams = list(df['HomeTeam'].dropna().unique()) + list(df['AwayTeam'].dropna().unique())
corum_names = [t for t in set(all_teams) if 'orum' in str(t).lower()]
print(f'Corum FK name variants found: {corum_names}')

for name in corum_names:
    matches = df[(df['HomeTeam'] == name) | (df['AwayTeam'] == name)]
    print(f'  {name}: {len(matches)} matches in dataset')

# Also check how many matches Galatasaray have (for comparison)
gala = [t for t in set(all_teams) if 'alata' in str(t).lower()]
print(f'\nGalatasaray variants: {gala}')
for name in gala:
    matches = df[(df['HomeTeam'] == name) | (df['AwayTeam'] == name)]
    print(f'  {name}: {len(matches)} matches in dataset')
