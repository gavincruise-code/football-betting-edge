import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from ev_calculator import assess_match

@dataclass
class BetRecord:
    date: str
    home_team: str
    away_team: str
    market: str  # 'Over 2.5' or 'Draw'
    model_prob: float
    implied_prob: float
    edge: float
    signal_strength: int
    odds: float
    stake: float
    result: str  # 'WIN' or 'LOSS'
    profit_loss: float
    cumulative_pl: float
    actual_score: str  # e.g. '2-1'

@dataclass
class BacktestResult:
    total_matches: int
    matches_analyzed: int  # matches with enough history
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    total_profit_loss: float
    roi: float
    peak_bankroll: float
    max_drawdown: float
    bet_log: List[BetRecord]
    bankroll_curve: List[float]  # cumulative P/L at each bet
    monthly_pl: dict  # {month_str: pl_amount}

def run_backtest(
    df: pd.DataFrame,
    edge_margin: float = 0.05,
    min_signal_strength: int = 2,
    stake: float = 10.0,
    markets: list = None,  # ['over25', 'draw'] or subset
    min_history: int = 5,
) -> BacktestResult:
    """
    CRITICAL: Zero lookahead bias implementation.
    
    Iterate through matches chronologically.
    For each match at index i, ONLY use data from matches[0:i] for calculations.
    The current match's result is only available after prediction.
    """
    if markets is None:
        markets = ['over25', 'draw']
        
    # Ensure dataframe is sorted by Date
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
        
    bet_log = []
    bankroll_curve = []
    monthly_pl = {}
    
    cumulative_pl = 0.0
    peak_bankroll = 0.0
    max_drawdown = 0.0
    
    total_matches = len(df)
    matches_analyzed = 0
    wins = 0
    losses = 0
    
    for i in range(len(df)):
        current_match = df.iloc[i]
        date = current_match['Date']
        home_team = current_match['HomeTeam']
        away_team = current_match['AwayTeam']
        
        hist_df = df.iloc[:i]
        
        home_hist = hist_df[(hist_df['HomeTeam'] == home_team) | (hist_df['AwayTeam'] == home_team)]
        away_hist = hist_df[(hist_df['HomeTeam'] == away_team) | (hist_df['AwayTeam'] == away_team)]
        
        if len(home_hist) < min_history or len(away_hist) < min_history:
            continue
            
        matches_analyzed += 1
        
        o25_odds = current_match.get('over25_odds')
        u25_odds = current_match.get('under25_odds')
        draw_odds = current_match.get('draw_odds')
        
        # Check odds
        if pd.isna(o25_odds): o25_odds = None
        if pd.isna(u25_odds): u25_odds = None
        if pd.isna(draw_odds): draw_odds = None
        
        assessment = assess_match(
            hist_df, home_team, away_team, date,
            odds_over25=o25_odds, odds_under25=u25_odds, odds_draw=draw_odds, margin=edge_margin
        )
        
        actual_h = current_match.get('FTHG')
        actual_a = current_match.get('FTAG')
        if pd.isna(actual_h) or pd.isna(actual_a):
            continue
        actual_score = f"{int(actual_h)}-{int(actual_a)}"
        total_goals = float(actual_h) + float(actual_a)
        is_draw = actual_h == actual_a
        is_o25 = total_goals > 2.5
        
        month_str = date.strftime('%Y-%m') if pd.notnull(date) else 'Unknown'
        
        # Over 2.5 Market
        if 'over25' in markets and assessment.has_edge_over25 and assessment.signal_strength >= min_signal_strength and o25_odds:
            result_str = 'WIN' if is_o25 else 'LOSS'
            pl = (stake * o25_odds - stake) if is_o25 else -stake
            cumulative_pl += pl
            bankroll_curve.append(cumulative_pl)
            
            if cumulative_pl > peak_bankroll:
                peak_bankroll = cumulative_pl
            
            drawdown = peak_bankroll - cumulative_pl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
            monthly_pl[month_str] = monthly_pl.get(month_str, 0.0) + pl
            
            if is_o25: wins += 1
            else: losses += 1
            
            bet_log.append(BetRecord(
                date=str(date.date()) if pd.notnull(date) else '',
                home_team=home_team,
                away_team=away_team,
                market='Over 2.5',
                model_prob=assessment.model_prob_over25,
                implied_prob=assessment.implied_prob_over25,
                edge=assessment.edge_over25,
                signal_strength=assessment.signal_strength,
                odds=o25_odds,
                stake=stake,
                result=result_str,
                profit_loss=pl,
                cumulative_pl=cumulative_pl,
                actual_score=actual_score
            ))
            
        # Draw Market
        if 'draw' in markets and assessment.has_edge_draw and assessment.signal_strength >= min_signal_strength and draw_odds:
            result_str = 'WIN' if is_draw else 'LOSS'
            pl = (stake * draw_odds - stake) if is_draw else -stake
            cumulative_pl += pl
            bankroll_curve.append(cumulative_pl)
            
            if cumulative_pl > peak_bankroll:
                peak_bankroll = cumulative_pl
            
            drawdown = peak_bankroll - cumulative_pl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
            monthly_pl[month_str] = monthly_pl.get(month_str, 0.0) + pl
            
            if is_draw: wins += 1
            else: losses += 1
            
            bet_log.append(BetRecord(
                date=str(date.date()) if pd.notnull(date) else '',
                home_team=home_team,
                away_team=away_team,
                market='Draw',
                model_prob=assessment.model_prob_draw,
                implied_prob=assessment.implied_prob_draw,
                edge=assessment.edge_draw,
                signal_strength=assessment.signal_strength,
                odds=draw_odds,
                stake=stake,
                result=result_str,
                profit_loss=pl,
                cumulative_pl=cumulative_pl,
                actual_score=actual_score
            ))
            
    total_bets = len(bet_log)
    win_rate = (wins / total_bets) if total_bets > 0 else 0.0
    roi = (cumulative_pl / (total_bets * stake)) if total_bets > 0 else 0.0
    
    return BacktestResult(
        total_matches=total_matches,
        matches_analyzed=matches_analyzed,
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_profit_loss=cumulative_pl,
        roi=roi,
        peak_bankroll=peak_bankroll,
        max_drawdown=max_drawdown,
        bet_log=bet_log,
        bankroll_curve=bankroll_curve,
        monthly_pl=monthly_pl
    )
