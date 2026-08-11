"""
Bet Tracker Module
===================
Handles logging placed bets from the Live Opportunity Scanner, updating bet outcomes,
calculating cumulative Profit & Loss (£), and rendering P/L analytics.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

LOG_FILE = "placed_bets_log.csv"

def get_placed_bets() -> pd.DataFrame:
    """Load existing logged bets from CSV file and compute P/L statistics."""
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            if df.empty:
                return df
            
            # Ensure P/L columns exist
            if 'Result' not in df.columns:
                df['Result'] = 'PENDING'
            if 'Profit_Loss_£' not in df.columns:
                df['Profit_Loss_£'] = 0.0
            if 'Cumulative_PL_£' not in df.columns:
                df['Cumulative_PL_£'] = 0.0

            # Recalculate P/L and cumulative P/L
            pl_vals = []
            cum = 0.0
            for _, r in df.iterrows():
                res = str(r.get('Result', 'PENDING')).upper()
                odds = float(r.get('Odds', 2.0)) if pd.notna(r.get('Odds')) else 2.0
                stake = float(r.get('Recommended_Stake_£', 10.0)) if pd.notna(r.get('Recommended_Stake_£')) else 10.0
                
                if res == 'WIN':
                    pl = stake * (odds - 1.0)
                elif res == 'LOSS':
                    pl = -stake
                else:
                    pl = 0.0
                
                pl_vals.append(round(pl, 2))
                cum += pl
            
            df['Profit_Loss_£'] = pl_vals
            df['Cumulative_PL_£'] = [round(c, 2) for c in np.cumsum(pl_vals)]
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def record_bet(
    date: str,
    league: str,
    home_team: str,
    away_team: str,
    market: str,
    odds: float,
    strategy: str,
    model_prob: float,
    implied_prob: float,
    edge_pct: float,
    recommended_stake: float,
    notes: str = ""
) -> bool:
    """Append a newly placed bet record to placed_bets_log.csv."""
    record = {
        'Timestamp_Recorded': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Match_Date': str(date),
        'League': str(league),
        'Home_Team': str(home_team),
        'Away_Team': str(away_team),
        'Market': str(market),
        'Odds': float(odds) if pd.notna(odds) else 0.0,
        'Strategy': str(strategy),
        'Model_Prob_%': round(float(model_prob) * 100, 1),
        'Implied_Prob_%': round(float(implied_prob) * 100, 1),
        'Edge_%': round(float(edge_pct) * 100, 1),
        'Recommended_Stake_£': round(float(recommended_stake), 2),
        'Result': 'PENDING',
        'Profit_Loss_£': 0.0,
        'Cumulative_PL_£': 0.0,
        'Notes': str(notes)
    }

    df = get_placed_bets()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)
    return True

def update_bet_result(row_idx: int, result: str) -> bool:
    """Update result ('WIN' or 'LOSS') for a specific logged bet by index."""
    df = get_placed_bets()
    if df.empty or row_idx >= len(df):
        return False
    
    df.loc[row_idx, 'Result'] = str(result).upper()
    df.to_csv(LOG_FILE, index=False)
    return True

def is_bet_recorded(home_team: str, away_team: str, market: str) -> bool:
    """Check if a fixture/market combination has already been logged."""
    df = get_placed_bets()
    if df.empty:
        return False
    
    match = df[(df['Home_Team'].str.lower() == home_team.lower()) & 
               (df['Away_Team'].str.lower() == away_team.lower()) & 
               (df['Market'].str.lower() == market.lower())]
    return not match.empty
