from ml.api_football_client import fetch_league_history, get_league_id
import warnings; warnings.filterwarnings('ignore')

# Check what league name the scanner would use for Austria
league_name = 'Austria'
lg_id = get_league_id(league_name)
print(f"League ID for '{league_name}': {lg_id}")

if lg_id:
    df = fetch_league_history(league_name, min_matches=10)
    print(f"Rows returned: {len(df)}")
    if not df.empty:
        all_teams = list(df['HomeTeam'].dropna().unique()) + list(df['AwayTeam'].dropna().unique())
        unique_teams = sorted(set(all_teams))
        print(f"\nTeams in dataset ({len(unique_teams)}):")
        for t in unique_teams:
            matches = df[(df['HomeTeam'] == t) | (df['AwayTeam'] == t)]
            lask = 'LASK' in str(t) or 'lask' in str(t).lower()
            ried = 'Ried' in str(t) or 'ried' in str(t).lower()
            if lask or ried:
                print(f"  *** {t}: {len(matches)} matches ***")
            else:
                print(f"  {t}: {len(matches)} matches")
else:
    print("No league ID found - checking football-data.co.uk...")
    from data_utils import download_league_data
    # Austria uses 'AUT' code? Check GLOBAL_LEAGUE_CODES
    from data_utils import GLOBAL_LEAGUE_CODES
    print(f"GLOBAL_LEAGUE_CODES: {GLOBAL_LEAGUE_CODES}")
