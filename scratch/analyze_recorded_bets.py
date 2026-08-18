import pandas as pd
import numpy as np

df = pd.read_csv('placed_bets_log.csv')
settled = df[df['Result'].isin(['WIN', 'LOSS'])].copy()

total_bets = len(settled)
wins = (settled['Result'] == 'WIN').sum()
losses = (settled['Result'] == 'LOSS').sum()
win_rate = wins / total_bets if total_bets > 0 else 0

total_staked = settled['Recommended_Stake_£'].sum()
total_profit = settled['Profit_Loss_£'].sum()
roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0

avg_odds = settled['Odds'].mean()
avg_model_prob = settled['Model_Prob_%'].mean()
avg_implied_prob = settled['Implied_Prob_%'].mean()
avg_edge = settled['Edge_%'].mean()

# Performance by Market
by_market = settled.groupby('Market').agg(
    Bets=('Result', 'count'),
    Wins=('Result', lambda x: (x == 'WIN').sum()),
    WinRate=('Result', lambda x: (x == 'WIN').mean() * 100),
    Staked=('Recommended_Stake_£', 'sum'),
    Profit=('Profit_Loss_£', 'sum'),
    ROI=('Profit_Loss_£', lambda x: (x.sum() / settled.loc[x.index, 'Recommended_Stake_£'].sum()) * 100),
    AvgOdds=('Odds', 'mean')
)

# Performance by League
by_league = settled.groupby('League').agg(
    Bets=('Result', 'count'),
    Wins=('Result', lambda x: (x == 'WIN').sum()),
    WinRate=('Result', lambda x: (x == 'WIN').mean() * 100),
    Staked=('Recommended_Stake_£', 'sum'),
    Profit=('Profit_Loss_£', 'sum'),
    ROI=('Profit_Loss_£', lambda x: (x.sum() / settled.loc[x.index, 'Recommended_Stake_£'].sum()) * 100)
).sort_values('Profit', ascending=False)

# Drawdown calculation
settled['Peak_PL'] = settled['Cumulative_PL_£'].cummax()
settled['Drawdown'] = settled['Cumulative_PL_£'] - settled['Peak_PL']
max_drawdown = settled['Drawdown'].min()
peak_profit = settled['Cumulative_PL_£'].max()

# Statistical Significance (t-test on per-bet profit / stake percentage return)
returns = settled['Profit_Loss_£'] / settled['Recommended_Stake_£']
mean_return = returns.mean()
std_return = returns.std()
t_stat = mean_return / (std_return / np.sqrt(total_bets)) if std_return > 0 else 0

print("=== OVERALL METRICS ===")
print(f"Total Settled Bets: {total_bets}")
print(f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate*100:.2f}%")
print(f"Total Staked: £{total_staked:,.2f}")
print(f"Total Profit: £{total_profit:,.2f}")
print(f"Overall ROI: {roi:.2f}%")
print(f"Average Odds: {avg_odds:.2f}")
print(f"Average Model Prob: {avg_model_prob:.1f}% vs Implied: {avg_implied_prob:.1f}% (Avg Edge: {avg_edge:.1f}%)")
print(f"Peak Profit: £{peak_profit:,.2f} | Max Drawdown: £{max_drawdown:,.2f}")
print(f"T-Statistic: {t_stat:.2f}")

print("\n=== BREAKDOWN BY MARKET ===")
print(by_market.round(2).to_string())

print("\n=== BREAKDOWN BY LEAGUE ===")
print(by_league.round(2).to_string())
