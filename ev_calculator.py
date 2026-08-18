from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from poisson_engine import prob_over25_matrix, prob_draw, implied_probability, has_edge, expected_value


@dataclass
class FactorResult:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str

@dataclass
class MatchAssessment:
    home_team: str
    away_team: str
    factors: list  # List[FactorResult]
    signal_strength: int  # 0-4 count of passing factors
    model_prob_over25: float
    model_prob_under25: float
    model_prob_draw: float
    implied_prob_over25: Optional[float]
    implied_prob_draw: Optional[float]
    edge_over25: Optional[float]
    edge_draw: Optional[float]
    has_edge_over25: bool
    has_edge_draw: bool
    ev_over25: Optional[float]
    ev_draw: Optional[float]
    lambda_home: float
    lambda_away: float

def calculate_double_high(history_df: pd.DataFrame, home_team: str, away_team: str, before_date, n: int = 5) -> FactorResult:
    """Factor 1: Both teams must average >2.5 total goals (scored+conceded) over last 5 matches."""
    from data_utils import get_team_history, compute_team_avg_goals
    
    home_history = get_team_history(history_df, home_team, before_date, n)
    away_history = get_team_history(history_df, away_team, before_date, n)
    
    home_avg = compute_team_avg_goals(home_history, home_team)
    away_avg = compute_team_avg_goals(away_history, away_team)
    
    val = (home_avg + away_avg) / 2.0
    passed = (home_avg > 2.5) and (away_avg > 2.5)
    
    return FactorResult(
        name="Double High Goals",
        passed=passed,
        value=val,
        threshold=2.5,
        detail=f"Home Avg: {home_avg:.2f}, Away Avg: {away_avg:.2f}"
    )

def calculate_home_away_splits(history_df: pd.DataFrame, home_team: str, away_team: str, before_date, n: int = 5) -> FactorResult:
    """Factor 2: Home team avg goals scored at home + Away team avg goals conceded away > 3.0."""
    from data_utils import get_home_history, get_away_history
    
    home_history = get_home_history(history_df, home_team, before_date, n)
    away_history = get_away_history(history_df, away_team, before_date, n)
    
    home_scored   = home_history['FTHG'].mean() if not home_history.empty else 0.0
    # In away_history rows, the team is the AWAY side:
    #   FTHG = goals scored by the HOME opponent = goals CONCEDED by our team ✓
    #   FTAG = goals SCORED by our team (away side)
    # So FTHG is correct here — original code was right, the audit finding was wrong.
    away_conceded = away_history['FTHG'].mean() if not away_history.empty else 0.0
    
    val = home_scored + away_conceded
    passed = val > 3.0
    
    return FactorResult(
        name="Home/Away Splits",
        passed=passed,
        value=val,
        threshold=3.0,
        detail=f"Home Scored: {home_scored:.2f}, Away Conceded: {away_conceded:.2f}"
    )

def calculate_h2h(history_df: pd.DataFrame, home_team: str, away_team: str, before_date, n: int = 5) -> FactorResult:
    """Factor 3: At least 3 of last 5 H2H meetings ended Over 2.5."""
    from data_utils import get_h2h
    
    h2h_df = get_h2h(history_df, home_team, away_team, before_date, n)
    
    if h2h_df.empty:
        return FactorResult(name="H2H Over 2.5", passed=False, value=0.0, threshold=3.0, detail="No H2H history")
        
    over25_count = h2h_df['over25'].sum()
    passed = over25_count >= 3
    
    return FactorResult(
        name="H2H Over 2.5",
        passed=passed,
        value=float(over25_count),
        threshold=3.0,
        detail=f"{over25_count} of {len(h2h_df)} H2H matches over 2.5"
    )

def _decay_weighted_mean(values: pd.Series, dates: pd.Series, ref_date, decay_rate: float = 0.003) -> float:
    """
    FIX M2: Exponential time-decay weighted mean.

    Weight for each match: w_i = exp(-decay_rate * days_ago)
    At decay_rate=0.003: half-life ≈ 231 days (~one full season).
    A match from 2 seasons ago (730 days) gets weight = exp(-2.19) ≈ 0.11.
    Last week's match gets weight ≈ 0.98.

    This prevents stale form (old managers/squads) distorting lambda estimates.
    """
    if values.empty:
        return float('nan')
    try:
        ref = pd.Timestamp(ref_date)
        days_ago = (ref - pd.to_datetime(dates)).dt.days.clip(lower=0).astype(float)
        weights = np.exp(-decay_rate * days_ago)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            return float(values.mean())
        return float((values * weights).sum() / weights_sum)
    except Exception:
        return float(values.mean())

def calculate_poisson_edge(
    history_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_date,
    odds_over25: float = None,
    odds_draw: float = None,
    margin: float = 0.05,
    n_matches: int = 10
) -> tuple:
    """
    Factor 4: Poisson model edge.
    Returns (FactorResult, model_prob_over25, model_prob_draw, lam_h, lam_a).

    Improvements applied:
    - FIX M2: Time-decay weighting — recent matches weighted more than old ones
    - FIX M1: League-average fallback — replaces global 1.0 with actual league goal rates
    """
    from data_utils import get_home_history, get_away_history

    home_history = get_home_history(history_df, home_team, before_date, n_matches)
    away_history = get_away_history(history_df, away_team, before_date, n_matches)

    # ── FIX M2: Time-decay weighted lambda estimates ───────────────────────────
    if not home_history.empty and 'Date' in home_history.columns:
        lam_h = _decay_weighted_mean(home_history['FTHG'], home_history['Date'], before_date)
    else:
        lam_h = float('nan')

    if not away_history.empty and 'Date' in away_history.columns:
        lam_a = _decay_weighted_mean(away_history['FTAG'], away_history['Date'], before_date)
    else:
        lam_a = float('nan')

    # ── FIX M1: League-average fallback (replaces global 1.0) ─────────────────
    # Using a universal 1.0 created phantom edges (Poisson(2.0) gives P(O25)=32.3%,
    # which appears to beat high-odds markets on missing data teams).
    # Now uses actual league scoring rates from the available dataset.
    if pd.isna(lam_h) or pd.isna(lam_a):
        try:
            _all_before = history_df[pd.to_datetime(history_df['Date']) < pd.Timestamp(before_date)]
            _league_h_avg = float(_all_before['FTHG'].mean()) if not _all_before.empty else 1.35
            _league_a_avg = float(_all_before['FTAG'].mean()) if not _all_before.empty else 1.10
        except Exception:
            _league_h_avg, _league_a_avg = 1.35, 1.10
        if pd.isna(lam_h):
            lam_h = _league_h_avg
        if pd.isna(lam_a):
            lam_a = _league_a_avg

    # Sanity clamp — prevents extreme lambdas from corrupting score matrix
    lam_h = float(np.clip(lam_h, 0.3, 5.0))
    lam_a = float(np.clip(lam_a, 0.3, 5.0))

    model_prob_over25 = prob_over25_matrix(lam_h, lam_a)
    model_prob_draw = prob_draw(lam_h, lam_a)

    passed = False
    detail = "No edge"

    if odds_over25 and odds_over25 > 1.0:
        imp_over25 = implied_probability(odds_over25)
        passed = has_edge(model_prob_over25, imp_over25, margin)
        detail = f"Model: {model_prob_over25:.2f}, Implied: {imp_over25:.2f}"

    res = FactorResult(
        name="Poisson Edge",
        passed=passed,
        value=model_prob_over25,
        threshold=margin,
        detail=detail
    )

    return res, model_prob_over25, model_prob_draw, lam_h, lam_a

def assess_match(history_df: pd.DataFrame, home_team: str, away_team: str, before_date, odds_over25: float = None, odds_under25: float = None, odds_draw: float = None, margin: float = 0.05) -> MatchAssessment:
    """Run all 4 factors and return complete assessment."""
    f1 = calculate_double_high(history_df, home_team, away_team, before_date)
    f2 = calculate_home_away_splits(history_df, home_team, away_team, before_date)
    f3 = calculate_h2h(history_df, home_team, away_team, before_date)
    f4, prob_o25, prob_dr, lam_h, lam_a = calculate_poisson_edge(history_df, home_team, away_team, before_date, odds_over25, odds_draw, margin)
    
    factors = [f1, f2, f3, f4]
    signal_strength = sum(1 for f in factors if f.passed)
    
    imp_o25 = implied_probability(odds_over25) if odds_over25 and odds_over25 > 1.0 else None
    imp_dr = implied_probability(odds_draw) if odds_draw and odds_draw > 1.0 else None
    
    edge_o25 = (prob_o25 - imp_o25) if imp_o25 else None
    edge_dr = (prob_dr - imp_dr) if imp_dr else None
    
    has_edge_o25 = f4.passed
    has_edge_dr = has_edge(prob_dr, imp_dr, margin) if imp_dr else False
    
    ev_o25 = expected_value(prob_o25, odds_over25) if odds_over25 and odds_over25 > 1.0 else None
    ev_dr = expected_value(prob_dr, odds_draw) if odds_draw and odds_draw > 1.0 else None
    
    return MatchAssessment(
        home_team=home_team,
        away_team=away_team,
        factors=factors,
        signal_strength=signal_strength,
        model_prob_over25=prob_o25,
        model_prob_under25=1.0 - prob_o25,
        model_prob_draw=prob_dr,
        implied_prob_over25=imp_o25,
        implied_prob_draw=imp_dr,
        edge_over25=edge_o25,
        edge_draw=edge_dr,
        has_edge_over25=has_edge_o25,
        has_edge_draw=has_edge_dr,
        ev_over25=ev_o25,
        ev_draw=ev_dr,
        lambda_home=lam_h,
        lambda_away=lam_a
    )
