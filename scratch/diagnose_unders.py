import pandas as pd
import numpy as np

df = pd.read_csv('placed_bets_log.csv')
settled = df[df['Result'].isin(['WIN', 'LOSS'])].copy()
unders = settled[settled['Market'] == 'Under 2.5'].copy()
overs = settled[settled['Market'] == 'Over 2.5'].copy()

print("=== ODDS BUCKET ANALYSIS FOR UNDERS ===")
bins = [1.0, 1.80, 2.20, 2.70, 5.00]
labels = ['Low (<=1.80)', 'Mid-Low (1.81-2.20)', 'Mid-High (2.21-2.70)', 'High (>2.70)']
unders['Odds_Bucket'] = pd.cut(unders['Odds'], bins=bins, labels=labels)

bucket_summary = unders.groupby('Odds_Bucket', observed=False).agg(
    Bets=('Result', 'count'),
    Wins=('Result', lambda x: (x == 'WIN').sum()),
    WinRate=('Result', lambda x: (x == 'WIN').mean() * 100),
    AvgOdds=('Odds', 'mean'),
    AvgModelProb=('Model_Prob_%', 'mean'),
    AvgImplied=('Implied_Prob_%', 'mean'),
    Staked=('Recommended_Stake_£', 'sum'),
    Profit=('Profit_Loss_£', 'sum'),
    ROI=('Profit_Loss_£', lambda x: (x.sum() / unders.loc[x.index, 'Recommended_Stake_£'].sum()) * 100)
)
print(bucket_summary.round(2).to_string())

print("\n=== UNDERS BY COMPETITION TYPE ===")
unders['Is_UEFA'] = unders['League'].str.contains('UEFA', case=False, na=False)
uefa_summary = unders.groupby('Is_UEFA').agg(
    Bets=('Result', 'count'),
    Wins=('Result', lambda x: (x == 'WIN').sum()),
    WinRate=('Result', lambda x: (x == 'WIN').mean() * 100),
    Profit=('Profit_Loss_£', 'sum'),
    ROI=('Profit_Loss_£', lambda x: (x.sum() / unders.loc[x.index, 'Recommended_Stake_£'].sum()) * 100),
    AvgOdds=('Odds', 'mean')
)
print(uefa_summary.round(2).to_string())

print("\n=== CALIBRATION / PROBABILITY CHECK FOR UNDERS ===")
print(f"Under 2.5: Avg Model Prob = {unders['Model_Prob_%'].mean():.1f}%, Actual Win Rate = {(unders['Result']=='WIN').mean()*100:.1f}% (Over-prediction bias: {unders['Model_Prob_%'].mean() - (unders['Result']=='WIN').mean()*100:.1f}%)")
print(f"Over 2.5:  Avg Model Prob = {overs['Model_Prob_%'].mean():.1f}%, Actual Win Rate = {(overs['Result']=='WIN').mean()*100:.1f}%")

print("\n=== UNDERS EXCLUDING UEFA QUALIFIERS ===")
dom_unders = unders[~unders['Is_UEFA']]
print(f"Domestic Unders Bets: {len(dom_unders)}, Wins: {(dom_unders['Result']=='WIN').sum()}, WinRate: {(dom_unders['Result']=='WIN').mean()*100:.1f}%, Profit: £{dom_unders['Profit_Loss_£'].sum():.2f}, ROI: {(dom_unders['Profit_Loss_£'].sum()/dom_unders['Recommended_Stake_£'].sum())*100:.2f}%")
