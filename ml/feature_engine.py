"""
Feature Engineering Module for Football Betting ML
===================================================
Calculates 120+ features across 3 rolling windows (10, 20, 38 matches)
with strict venue splits (Home-only, Away-only) and zero lookahead bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ml.config import ROLLING_WINDOWS, MIN_HISTORY_MATCHES, CONGESTION_THRESHOLD_DAYS, RELEGATION_ZONE_SIZE, TOP_POSITIONS

def compute_rolling_goal_features(
    history: pd.DataFrame,
    team: str,
    prefix: str,
    window: int
) -> Dict[str, float]:
    """
    Category A: Goal-based rolling features.
    """
    res = {}
    subset = history.tail(window)
    if subset.empty:
        for feat in ['goals_scored_avg', 'goals_conceded_avg', 'total_goals_avg', 'clean_sheet_rate', 'btts_rate', 'over25_rate']:
            res[f"{feat}_{prefix}_{window}"] = np.nan
        return res

    scored = []
    conceded = []
    for _, row in subset.iterrows():
        if row['HomeTeam'] == team:
            scored.append(row['FTHG'])
            conceded.append(row['FTAG'])
        else:
            scored.append(row['FTAG'])
            conceded.append(row['FTHG'])

    scored = np.array(scored, dtype=float)
    conceded = np.array(conceded, dtype=float)
    total = scored + conceded

    res[f"goals_scored_avg_{prefix}_{window}"] = float(np.mean(scored))
    res[f"goals_conceded_avg_{prefix}_{window}"] = float(np.mean(conceded))
    res[f"total_goals_avg_{prefix}_{window}"] = float(np.mean(total))
    res[f"clean_sheet_rate_{prefix}_{window}"] = float(np.mean(conceded == 0))
    res[f"btts_rate_{prefix}_{window}"] = float(np.mean((scored > 0) & (conceded > 0)))
    res[f"over25_rate_{prefix}_{window}"] = float(np.mean(total > 2.5))

    return res

def compute_rolling_xg_features(
    history: pd.DataFrame,
    team: str,
    prefix: str,
    window: int
) -> Dict[str, float]:
    """
    Category B: xG-based rolling features.
    """
    res = {}
    feat_names = ['xG_avg', 'xGA_avg', 'xG_delta', 'xGA_delta', 'xG_per_shot', 'xG_overperformance', 'xG_volatility']
    
    subset = history.tail(window)
    if subset.empty or 'xG_home' not in subset.columns or subset['xG_home'].isna().all():
        for feat in feat_names:
            res[f"{feat}_{prefix}_{window}"] = np.nan
        return res

    xg_list = []
    xga_list = []
    goals_list = []
    conceded_list = []
    shots_list = []

    for _, row in subset.iterrows():
        if row['HomeTeam'] == team:
            xg_list.append(row.get('xG_home', np.nan))
            xga_list.append(row.get('xG_away', np.nan))
            goals_list.append(row['FTHG'])
            conceded_list.append(row['FTAG'])
            shots_list.append(row.get('HS', np.nan))
        else:
            xg_list.append(row.get('xG_away', np.nan))
            xga_list.append(row.get('xG_home', np.nan))
            goals_list.append(row['FTAG'])
            conceded_list.append(row['FTHG'])
            shots_list.append(row.get('AS', np.nan))

    xg_arr = np.array(xg_list, dtype=float)
    xga_arr = np.array(xga_list, dtype=float)
    g_arr = np.array(goals_list, dtype=float)
    gc_arr = np.array(conceded_list, dtype=float)
    s_arr = np.array(shots_list, dtype=float)

    if np.isnan(xg_arr).all():
        for feat in feat_names:
            res[f"{feat}_{prefix}_{window}"] = np.nan
        return res

    res[f"xG_avg_{prefix}_{window}"] = float(np.nanmean(xg_arr))
    res[f"xGA_avg_{prefix}_{window}"] = float(np.nanmean(xga_arr))
    res[f"xG_delta_{prefix}_{window}"] = float(np.nanmean(g_arr - xg_arr))
    res[f"xGA_delta_{prefix}_{window}"] = float(np.nanmean(gc_arr - xga_arr))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        per_shot = np.where(s_arr > 0, xg_arr / s_arr, np.nan)
    res[f"xG_per_shot_{prefix}_{window}"] = float(np.nanmean(per_shot))
    res[f"xG_overperformance_{prefix}_{window}"] = float(np.nansum(g_arr - xg_arr))
    res[f"xG_volatility_{prefix}_{window}"] = float(np.nanstd(xg_arr)) if len(xg_arr) > 1 else 0.0

    return res

def compute_rolling_shot_features(
    history: pd.DataFrame,
    team: str,
    prefix: str,
    window: int
) -> Dict[str, float]:
    """
    Category C: Shot profile rolling features.
    """
    res = {}
    feat_names = ['shots_avg', 'shots_on_target_avg', 'shot_accuracy', 'shot_conversion']
    subset = history.tail(window)
    if subset.empty or 'HS' not in subset.columns:
        for feat in feat_names:
            res[f"{feat}_{prefix}_{window}"] = np.nan
        return res

    shots_list = []
    sot_list = []
    goals_list = []

    for _, row in subset.iterrows():
        if row['HomeTeam'] == team:
            shots_list.append(row.get('HS', np.nan))
            sot_list.append(row.get('HST', np.nan))
            goals_list.append(row['FTHG'])
        else:
            shots_list.append(row.get('AS', np.nan))
            sot_list.append(row.get('AST', np.nan))
            goals_list.append(row['FTAG'])

    shots = np.array(shots_list, dtype=float)
    sot = np.array(sot_list, dtype=float)
    goals = np.array(goals_list, dtype=float)

    res[f"shots_avg_{prefix}_{window}"] = float(np.nanmean(shots))
    res[f"shots_on_target_avg_{prefix}_{window}"] = float(np.nanmean(sot))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(shots > 0, sot / shots, np.nan)
        conv = np.where(sot > 0, goals / sot, np.nan)

    res[f"shot_accuracy_{prefix}_{window}"] = float(np.nanmean(acc))
    res[f"shot_conversion_{prefix}_{window}"] = float(np.nanmean(conv))

    return res

def compute_contextual_features(
    full_df: pd.DataFrame,
    match_idx: int,
    match_date,
    home_team: str,
    away_team: str,
) -> Dict[str, float]:
    """
    Category D: Contextual features per match.
    """
    res = {}
    hist = full_df.iloc[:match_idx]

    # Days since last match
    home_prev = hist[(hist['HomeTeam'] == home_team) | (hist['AwayTeam'] == home_team)]
    away_prev = hist[(hist['HomeTeam'] == away_team) | (hist['AwayTeam'] == away_team)]

    if not home_prev.empty:
        last_date = home_prev.iloc[-1]['Date']
        days_home = (match_date - last_date).days
    else:
        days_home = 7

    if not away_prev.empty:
        last_date = away_prev.iloc[-1]['Date']
        days_away = (match_date - last_date).days
    else:
        days_away = 7

    res['days_since_last_home'] = float(days_home)
    res['days_since_last_away'] = float(days_away)
    res['is_congested_home'] = float(days_home < CONGESTION_THRESHOLD_DAYS)
    res['is_congested_away'] = float(days_away < CONGESTION_THRESHOLD_DAYS)

    # Month
    res['month'] = float(match_date.month)

    return res

def compute_h2h_features(
    full_df: pd.DataFrame,
    match_idx: int,
    home_team: str,
    away_team: str,
    n: int = 5
) -> Dict[str, float]:
    """
    Category E: Head-to-head features.
    """
    hist = full_df.iloc[:match_idx]
    h2h = hist[
        ((hist['HomeTeam'] == home_team) & (hist['AwayTeam'] == away_team)) |
        ((hist['HomeTeam'] == away_team) & (hist['AwayTeam'] == home_team))
    ].tail(n)

    res = {}
    if h2h.empty:
        res['h2h_over25_rate_5'] = np.nan
        res['h2h_avg_goals_5'] = np.nan
        res['h2h_matches_available'] = 0.0
    else:
        tot_goals = h2h['FTHG'] + h2h['FTAG']
        res['h2h_over25_rate_5'] = float(np.mean(tot_goals > 2.5))
        res['h2h_avg_goals_5'] = float(np.mean(tot_goals))
        res['h2h_matches_available'] = float(len(h2h))

    return res

def compute_market_features(match_row: pd.Series) -> Dict[str, float]:
    """
    Category F: Market odds derived features.
    """
    res = {}
    o25 = match_row.get('over25_odds', np.nan)
    u25 = match_row.get('under25_odds', np.nan)

    if pd.notna(o25) and o25 > 1.0:
        imp_o25 = 1.0 / o25
    else:
        imp_o25 = np.nan

    if pd.notna(u25) and u25 > 1.0:
        imp_u25 = 1.0 / u25
    else:
        imp_u25 = np.nan

    res['implied_prob_over25'] = imp_o25
    res['implied_prob_under25'] = imp_u25

    if pd.notna(imp_o25) and pd.notna(imp_u25):
        res['overround'] = imp_o25 + imp_u25 - 1.0
        res['odds_ratio_over25'] = o25 / u25 if u25 > 0 else np.nan
    else:
        res['overround'] = np.nan
        res['odds_ratio_over25'] = np.nan

    return res

def compute_all_features_for_match(
    full_df: pd.DataFrame,
    match_idx: int,
    home_h_hist: pd.DataFrame = None,
    home_all_hist: pd.DataFrame = None,
    away_a_hist: pd.DataFrame = None,
    away_all_hist: pd.DataFrame = None,
) -> Dict[str, float]:
    """
    Computes all ~120+ features for match at index match_idx using ONLY data prior to match_idx.
    """
    match_row = full_df.iloc[match_idx]
    match_date = match_row['Date']
    home_team = match_row['HomeTeam']
    away_team = match_row['AwayTeam']

    if home_h_hist is None:
        hist = full_df.iloc[:match_idx]
        home_h_hist = hist[hist['HomeTeam'] == home_team]
        home_all_hist = hist[(hist['HomeTeam'] == home_team) | (hist['AwayTeam'] == home_team)]
        away_a_hist = hist[hist['AwayTeam'] == away_team]
        away_all_hist = hist[(hist['HomeTeam'] == away_team) | (hist['AwayTeam'] == away_team)]

    feats = {}

    for w in ROLLING_WINDOWS:
        # Home team - Home venue
        feats.update(compute_rolling_goal_features(home_h_hist, home_team, 'H_H', w))
        feats.update(compute_rolling_xg_features(home_h_hist, home_team, 'H_H', w))
        feats.update(compute_rolling_shot_features(home_h_hist, home_team, 'H_H', w))

        # Home team - Overall
        feats.update(compute_rolling_goal_features(home_all_hist, home_team, 'H_All', w))
        feats.update(compute_rolling_xg_features(home_all_hist, home_team, 'H_All', w))
        feats.update(compute_rolling_shot_features(home_all_hist, home_team, 'H_All', w))

        # Away team - Away venue
        feats.update(compute_rolling_goal_features(away_a_hist, away_team, 'A_A', w))
        feats.update(compute_rolling_xg_features(away_a_hist, away_team, 'A_A', w))
        feats.update(compute_rolling_shot_features(away_a_hist, away_team, 'A_A', w))

        # Away team - Overall
        feats.update(compute_rolling_goal_features(away_all_hist, away_team, 'A_All', w))
        feats.update(compute_rolling_xg_features(away_all_hist, away_team, 'A_All', w))
        feats.update(compute_rolling_shot_features(away_all_hist, away_team, 'A_All', w))

    # Trends (short-term vs long-term)
    for team_prefix in ['H_H', 'H_All', 'A_A', 'A_All']:
        g10 = feats.get(f"goals_scored_avg_{team_prefix}_10", np.nan)
        g38 = feats.get(f"goals_scored_avg_{team_prefix}_38", np.nan)
        feats[f"goals_trend_{team_prefix}"] = g10 - g38 if pd.notna(g10) and pd.notna(g38) else np.nan

        xg10 = feats.get(f"xG_avg_{team_prefix}_10", np.nan)
        xg38 = feats.get(f"xG_avg_{team_prefix}_38", np.nan)
        feats[f"xG_trend_{team_prefix}"] = xg10 - xg38 if pd.notna(xg10) and pd.notna(xg38) else np.nan

    # Contextual, H2H, Market
    feats.update(compute_contextual_features(full_df, match_idx, match_date, home_team, away_team))
    feats.update(compute_h2h_features(full_df, match_idx, home_team, away_team))
    feats.update(compute_market_features(match_row))

    return feats

def compute_all_features(df: pd.DataFrame, min_history: int = MIN_HISTORY_MATCHES) -> pd.DataFrame:
    """
    Computes feature matrix for entire dataset chronologically.
    Optimized with fast team index tracking.
    """
    df = df.sort_values('Date').reset_index(drop=True)
    all_feature_rows = []

    # Map team -> list of prior match indices
    team_all_idx = {}
    team_home_idx = {}
    team_away_idx = {}

    for i in range(len(df)):
        home_team = df.iloc[i]['HomeTeam']
        away_team = df.iloc[i]['AwayTeam']

        h_all = team_all_idx.get(home_team, [])
        a_all = team_all_idx.get(away_team, [])

        if len(h_all) >= min_history and len(a_all) >= min_history:
            h_h = team_home_idx.get(home_team, [])
            a_a = team_away_idx.get(away_team, [])

            row_feats = compute_all_features_for_match(
                df, i,
                home_h_hist=df.iloc[h_h],
                home_all_hist=df.iloc[h_all],
                away_a_hist=df.iloc[a_a],
                away_all_hist=df.iloc[a_all]
            )
            
            # Meta columns
            row_feats['Date'] = df.iloc[i]['Date']
            row_feats['HomeTeam'] = home_team
            row_feats['AwayTeam'] = away_team
            row_feats['FTHG'] = df.iloc[i]['FTHG']
            row_feats['FTAG'] = df.iloc[i]['FTAG']
            row_feats['over25'] = bool((df.iloc[i]['FTHG'] + df.iloc[i]['FTAG']) > 2.5)
            row_feats['over25_odds'] = df.iloc[i].get('over25_odds', np.nan)
            row_feats['under25_odds'] = df.iloc[i].get('under25_odds', np.nan)
            row_feats['draw_odds'] = df.iloc[i].get('draw_odds', np.nan)
            if 'league' in df.columns:
                row_feats['league'] = df.iloc[i]['league']

            all_feature_rows.append(row_feats)

        # Update team history indices after checking
        if home_team not in team_all_idx: team_all_idx[home_team] = []
        if away_team not in team_all_idx: team_all_idx[away_team] = []
        if home_team not in team_home_idx: team_home_idx[home_team] = []
        if away_team not in team_away_idx: team_away_idx[away_team] = []

        team_all_idx[home_team].append(i)
        team_all_idx[away_team].append(i)
        team_home_idx[home_team].append(i)
        team_away_idx[away_team].append(i)

    if not all_feature_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_feature_rows)
