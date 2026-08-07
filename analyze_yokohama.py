import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from data_utils import download_league_data, get_team_history, get_home_history, get_away_history, get_h2h
from poisson_engine import score_matrix, prob_over_n_goals, prob_draw, expected_value, has_edge
from ml.dixon_coles import fit_dixon_coles, predict_match_outcome
from ml.calibration import kelly_stake

print("=== Downloading Japan J-League Data ===")
df = download_league_data('JPN')
print(f"Total matches in JPN dataset: {len(df)}")
if not df.empty:
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Search team names
    all_teams = sorted(list(set(df['HomeTeam'].dropna().unique()).union(set(df['AwayTeam'].dropna().unique()))))
    print(f"All Teams ({len(all_teams)}): {all_teams}")

    h_candidates = [t for t in all_teams if 'yokohama' in t.lower() or 'marinos' in t.lower()]
    a_candidates = [t for t in all_teams if 'kashima' in t.lower() or 'antlers' in t.lower()]
    print(f"Yokohama Candidates: {h_candidates}")
    print(f"Kashima Candidates: {a_candidates}")

    home_team = h_candidates[0] if h_candidates else 'Yokohama M.'
    away_team = a_candidates[0] if a_candidates else 'Kashima'

    print(f"\nAnalyzing: {home_team} (Home) vs {away_team} (Away)")

    # History
    h_all = get_team_history(df, home_team, pd.Timestamp.now(), n=10)
    a_all = get_team_history(df, away_team, pd.Timestamp.now(), n=10)
    h_home = get_home_history(df, home_team, pd.Timestamp.now(), n=10)
    a_away = get_away_history(df, away_team, pd.Timestamp.now(), n=10)
    h2h = get_h2h(df, home_team, away_team, pd.Timestamp.now(), n=10)

    print(f"\n--- {home_team} Recent Home Form ---")
    print(h_home[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'total_goals']].tail(5))

    print(f"\n--- {away_team} Recent Away Form ---")
    print(a_away[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'total_goals']].tail(5))

    print(f"\n--- Head-to-Head Meetings ---")
    print(h2h[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'total_goals']].tail(5))

    # Averages
    h_gf_home = h_home['FTHG'].mean() if not h_home.empty else 1.6
    h_ga_home = h_home['FTAG'].mean() if not h_home.empty else 1.1
    a_gf_away = a_away['FTAG'].mean() if not a_away.empty else 1.3
    a_ga_away = a_away['FTHG'].mean() if not a_away.empty else 1.2

    lam_h = (h_gf_home + a_ga_away) / 2.0
    lam_a = (a_gf_away + h_ga_home) / 2.0

    print(f"\nCalculated Poisson Parameters:")
    print(f"Home Expected Goals (lambda_home): {lam_h:.2f}")
    print(f"Away Expected Goals (lambda_away): {lam_a:.2f}")

    sm = score_matrix(lam_h, lam_a)
    p_o25 = sum(sm[i][j] for i in range(7) for j in range(7) if i + j >= 3)
    p_u25 = 1.0 - p_o25
    p_draw = sum(sm[i][i] for i in range(7))

    print(f"\n--- Poisson Model Probabilities ---")
    print(f"P(Over 2.5):  {p_o25*100:.1f}%")
    print(f"P(Under 2.5): {p_u25*100:.1f}%")
    print(f"P(Draw):      {p_draw*100:.1f}%")

    # Dixon Coles
    dc_params = fit_dixon_coles(df.tail(500), use_xg=False)
    dc_preds = predict_match_outcome(dc_params, home_team, away_team)
    print(f"\n--- Dixon-Coles Model Predictions ---")
    print(f"P(Over 2.5):  {dc_preds['over25']*100:.1f}%")
    print(f"P(Under 2.5): {dc_preds['under25']*100:.1f}%")
    print(f"P(Draw):      {dc_preds['draw']*100:.1f}%")

    # Top scorelines
    score_probs = []
    for h_g in range(5):
        for a_g in range(5):
            score_probs.append((f"{h_g}-{a_g}", sm[h_g][a_g]))
    score_probs.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 Most Likely Scorelines:")
    for score, prob in score_probs[:5]:
        print(f"  {score}: {prob*100:.1f}%")
