"""
Bet Tracker Module
===================
Handles logging placed bets to a persistent CSV file and CSV export.
"""

import os
import pandas as pd
from datetime import datetime

LOG_FILE = "placed_bets_log.csv"

def get_placed_bets() -> pd.DataFrame:
    """Load existing logged bets from CSV file."""
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
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
        'Notes': str(notes)
    }

    df = get_placed_bets()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
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
