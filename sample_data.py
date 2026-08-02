import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_csv(output_path: str = 'sample_epl_2324.csv', n_teams: int = 20, seed: int = 42):
    """
    Generate realistic sample football data in football-data.co.uk format.
    - 20 teams, 38 matchdays, 380 matches
    - Poisson-distributed goals (home λ ≈ 1.55, away λ ≈ 1.15)
    - Realistic Bet365 odds columns: B365H, B365D, B365A, B365>2.5, B365<2.5
    - Proper date sequencing (Aug 2023 - May 2024)
    - Include team names that look realistic (Team A, Team B, etc. is fine)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    teams = [f"Team {chr(65+i)}" for i in range(n_teams)]
    
    start_date = datetime(2023, 8, 12)
    dates = [start_date + timedelta(days=d) for d in range(0, 280, 7)] # rough matchdays
    
    # Generate schedule (round-robin)
    matches = []
    for h in teams:
        for a in teams:
            if h != a:
                matches.append((h, a))
                
    np.random.shuffle(matches)
    
    data = []
    for i, (h, a) in enumerate(matches):
        match_date = dates[i // (n_teams // 2)]
        
        # Base lambda
        lam_h = np.random.normal(1.55, 0.3)
        lam_a = np.random.normal(1.15, 0.3)
        
        lam_h = max(0.1, lam_h)
        lam_a = max(0.1, lam_a)
        
        fthg = np.random.poisson(lam_h)
        ftag = np.random.poisson(lam_a)
        
        ftr = 'H' if fthg > ftag else ('A' if ftag > fthg else 'D')
        
        # Odds roughly around actual probabilities with overround
        sum_g = lam_h + lam_a
        p_h = lam_h / (sum_g + 0.1)
        p_a = lam_a / (sum_g + 0.1)
        p_d = 1.0 - p_h - p_a
        
        p_h = max(0.1, min(p_h, 0.8))
        p_a = max(0.1, min(p_a, 0.8))
        p_d = max(0.1, min(1 - p_h - p_a, 0.8))
        
        b365h = 1.0 / p_h * 0.95
        b365d = 1.0 / p_d * 0.95
        b365a = 1.0 / p_a * 0.95
        
        # Over/Under 2.5 odds
        prob_o25 = 1 - (np.exp(-sum_g) * (1 + sum_g + (sum_g**2)/2))
        prob_u25 = 1.0 - prob_o25
        prob_o25 = max(0.05, min(0.95, prob_o25))
        prob_u25 = max(0.05, min(0.95, prob_u25))
        
        b365_o25 = 1.0 / prob_o25 * 0.95
        b365_u25 = 1.0 / prob_u25 * 0.95
        
        data.append({
            'Date': match_date.strftime('%d/%m/%Y'),
            'HomeTeam': h,
            'AwayTeam': a,
            'FTHG': fthg,
            'FTAG': ftag,
            'FTR': ftr,
            'B365H': round(b365h, 2),
            'B365D': round(b365d, 2),
            'B365A': round(b365a, 2),
            'B365>2.5': round(b365_o25, 2),
            'B365<2.5': round(b365_u25, 2)
        })
        
    df = pd.DataFrame(data)
    # Sort by date
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date_dt').drop('Date_dt', axis=1)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    generate_sample_csv()
    print('Sample CSV generated.')
