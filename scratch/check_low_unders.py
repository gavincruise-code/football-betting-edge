import pandas as pd

df = pd.read_csv('placed_bets_log.csv')
settled = df[df['Result'].isin(['WIN', 'LOSS'])].copy()
unders = settled[settled['Market'] == 'Under 2.5'].copy()

low_odds_loss = unders[(unders['Odds'] <= 1.80) & (unders['Result'] == 'LOSS')]
print("=== SHORT ODDS UNDERS LOSSES (Odds <= 1.80) ===")
print(low_odds_loss[['Match_Date', 'League', 'Home_Team', 'Away_Team', 'Odds', 'Model_Prob_%', 'Recommended_Stake_£', 'Profit_Loss_£']].to_string())

print("\n=== AUGUST EARLY SEASON IMPACT ===")
print(f"Total August bets: {len(settled)}")
print("League counts in settled bets:")
print(settled['League'].value_counts().to_string())
