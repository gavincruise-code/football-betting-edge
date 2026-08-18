"""
Dixon-Coles Statistical Model for Over 2.5 Goals
================================================
Bivariate Poisson model with low-score dependency correction tau(x,y)
and exponential time-decay weighting. Fits team attack & defense parameters.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import math
import logging
from ml.config import DIXON_COLES_XI, MAX_GOALS_MATRIX

logger = logging.getLogger(__name__)

def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon-Coles adjustment factor for low scores.
    """
    if x == 0 and y == 0:
        return max(1e-6, 1.0 - lam * mu * rho)
    elif x == 1 and y == 0:
        return max(1e-6, 1.0 + mu * rho)
    elif x == 0 and y == 1:
        return max(1e-6, 1.0 + lam * rho)
    elif x == 1 and y == 1:
        return max(1e-6, 1.0 - rho)
    else:
        return 1.0

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function with numeric safeguards."""
    if k < 0 or lam <= 0:
        return 0.0
    lam = min(lam, 20.0)
    try:
        return (math.pow(lam, k) * math.exp(-lam)) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0

def fit_dixon_coles(
    matches_df: pd.DataFrame,
    xi: float = DIXON_COLES_XI,
    use_xg: bool = True,
) -> Dict:
    """
    Fit Dixon-Coles parameters on a historical matches DataFrame.
    
    matches_df requires: Date, HomeTeam, AwayTeam, FTHG, FTAG, (and xG_home, xG_away if use_xg=True).
    """
    if matches_df.empty:
        return {'converged': False}

    teams = sorted(list(set(matches_df['HomeTeam']).union(set(matches_df['AwayTeam']))))
    n_teams = len(teams)
    if n_teams < 2:
        return {'converged': False}

    team_map = {t: i for i, t in enumerate(teams)}
    
    # Calculate days ago from the latest date in matches_df
    max_date = matches_df['Date'].max()
    dates = (max_date - matches_df['Date']).dt.days.values
    weights = np.exp(-xi * dates)

    # Goals/xG targets
    has_xg = use_xg and 'xG_home' in matches_df.columns and not matches_df['xG_home'].isna().all()

    home_indices = np.array([team_map[t] for t in matches_df['HomeTeam']])
    away_indices = np.array([team_map[t] for t in matches_df['AwayTeam']])

    if has_xg:
        # Use rounded xG as pseudo-goals for Poisson likelihood
        home_goals = np.round(matches_df['xG_home'].fillna(matches_df['FTHG'])).astype(int).values
        away_goals = np.round(matches_df['xG_away'].fillna(matches_df['FTAG'])).astype(int).values
    else:
        home_goals = matches_df['FTHG'].values
        away_goals = matches_df['FTAG'].values

    # Initial parameter vector: [home_adv, rho, att_0..n-1, def_0..n-1]
    init_params = np.zeros(2 + 2 * n_teams)
    init_params[0] = 0.25   # home advantage
    init_params[1] = -0.05  # rho

    def neg_log_likelihood(params):
        home_adv = params[0]
        rho = params[1]
        att = params[2:2 + n_teams]
        deff = params[2 + n_teams:]

        # Clip log-rates to avoid overflow during optimization
        log_lam = np.clip(home_adv + att[home_indices] + deff[away_indices], -5.0, 3.0)
        log_mu = np.clip(att[away_indices] + deff[home_indices], -5.0, 3.0)

        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        log_l = 0.0
        for i in range(len(home_goals)):
            x = int(home_goals[i])
            y = int(away_goals[i])
            l_val = float(lam[i])
            m_val = float(mu[i])
            w = float(weights[i])

            p_x = poisson_pmf(x, l_val)
            p_y = poisson_pmf(y, m_val)
            t_val = tau(x, y, l_val, m_val, rho)

            prob = t_val * p_x * p_y
            if prob > 1e-12:
                log_l += w * math.log(prob)
            else:
                log_l += w * -27.6  # penalty

        # Identifiability constraint penalty: sum(att) == 0
        penalty = 100.0 * (np.sum(att) ** 2)
        return -log_l + penalty

    res = minimize(neg_log_likelihood, init_params, method='SLSQP', options={'maxiter': 200})

    # FIX N2: Stricter convergence check.
    # Previous guard: `not res.success and not res.fun` almost never triggered —
    # a partially converged result could slip through with a near-zero likelihood.
    # Now: treat any non-convergent run as failed and return {'converged': False}
    # so callers fall back to Poisson-only rather than using potentially corrupted params.
    if not res.success:
        logger.warning(
            "Dixon-Coles optimiser failed to converge for %d teams (%d matches). "
            "Message: %s. Falling back to Poisson-only for this fixture.",
            n_teams, len(matches_df), res.message
        )
        return {'converged': False}

    params = res.x
    home_adv = params[0]
    rho = params[1]
    att = params[2:2 + n_teams]
    deff = params[2 + n_teams:]

    return {
        'converged': True,
        'home_adv': home_adv,
        'rho': rho,
        'attack': {team: att[i] for team, i in team_map.items()},
        'defense': {team: deff[i] for team, i in team_map.items()},
        'team_map': team_map,
    }

def predict_score_probs(
    params: Dict,
    home_team: str,
    away_team: str,
    max_goals: int = MAX_GOALS_MATRIX,
) -> np.ndarray:
    """
    Generate score probability matrix (max_goals+1 x max_goals+1).

    Returns None if params are missing or unconverged — callers must handle
    this case explicitly. Previously returned a uniform matrix (every scoreline
    equally likely) which silently corrupted Over/Under predictions.
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    if not params or not params.get('converged', False):
        # FIX N2: Return None instead of uniform fallback.
        # The uniform matrix (1/81 per scoreline) assigned the same probability
        # to 0-0 and 6-6, completely corrupting the Over 2.5 estimate.
        return None

    home_adv = params['home_adv']
    rho = params['rho']
    att = params['attack']
    deff = params['defense']

    # Default to average attack/defense if team not in model
    att_h = att.get(home_team, 0.0)
    def_h = deff.get(home_team, 0.0)
    att_a = att.get(away_team, 0.0)
    def_a = deff.get(away_team, 0.0)

    lam = math.exp(home_adv + att_h + def_a)
    mu = math.exp(att_a + def_h)

    for h in range(max_goals + 1):
        p_h = poisson_pmf(h, lam)
        for a in range(max_goals + 1):
            p_a = poisson_pmf(a, mu)
            t_val = tau(h, a, lam, mu, rho)
            matrix[h, a] = t_val * p_h * p_a

    # Normalize matrix to sum to 1
    total = np.sum(matrix)
    if total > 0:
        matrix /= total

    return matrix

def predict_over25(params: Dict, home_team: str, away_team: str) -> float:
    """Calculates P(Over 2.5 goals) from Dixon-Coles parameters in O(1) time."""
    if not params or not params.get('converged', False):
        return 0.50

    home_adv = params['home_adv']
    rho = params['rho']
    att = params['attack']
    deff = params['defense']

    att_h = att.get(home_team, 0.0)
    def_h = deff.get(home_team, 0.0)
    att_a = att.get(away_team, 0.0)
    def_a = deff.get(away_team, 0.0)

    lam = math.exp(home_adv + att_h + def_a)
    mu = math.exp(att_a + def_h)

    e_neg_tot = math.exp(-(lam + mu))

    # Low score probabilities with Dixon-Coles tau correction
    p00 = tau(0, 0, lam, mu, rho) * e_neg_tot
    p10 = tau(1, 0, lam, mu, rho) * lam * e_neg_tot
    p01 = tau(0, 1, lam, mu, rho) * mu * e_neg_tot
    p11 = tau(1, 1, lam, mu, rho) * lam * mu * e_neg_tot
    p20 = (lam ** 2 / 2.0) * e_neg_tot
    p02 = (mu ** 2 / 2.0) * e_neg_tot

    p_under25 = p00 + p10 + p01 + p11 + p20 + p02
    return max(0.05, min(0.95, 1.0 - p_under25))

def predict_draw(params: Dict, home_team: str, away_team: str) -> float:
    """Calculates P(Draw) from Dixon-Coles matrix. Returns NaN if model unconverged."""
    matrix = predict_score_probs(params, home_team, away_team)
    if matrix is None:
        return float('nan')
    prob = sum(matrix[k, k] for k in range(matrix.shape[0]))
    return float(prob)

def predict_match_outcome(params: Dict, home_team: str, away_team: str) -> Dict[str, float]:
    """
    Returns full match prediction probabilities.
    Returns dict of NaN values if Dixon-Coles failed to converge.
    """
    matrix = predict_score_probs(params, home_team, away_team)
    if matrix is None:
        return {'home_win': float('nan'), 'draw': float('nan'),
                'away_win': float('nan'), 'over25': float('nan'), 'under25': float('nan')}
    home_win = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, 1)))
    over25 = float(predict_over25(params, home_team, away_team))

    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win,
        'over25': over25,
        'under25': 1.0 - over25,
    }
