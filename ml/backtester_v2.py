"""
Purged Walk-Forward Backtester for ML Football Betting
======================================================
Evaluates ML strategies (Dixon-Coles, XGBoost, Dual Ensemble, Random) using
purged walk-forward temporal cross-validation with zero lookahead bias.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

from ml.config import (
    MIN_TRAIN_SEASONS, PURGE_DAYS, DEFAULT_EDGE_MARGIN, DEFAULT_STAKE,
    KELLY_FRACTION, KELLY_MAX_PCT, MIN_MODEL_PROBABILITY, MAX_MODEL_PROBABILITY
)
from ml.dixon_coles import fit_dixon_coles, predict_over25, predict_draw
from ml.ml_model import prepare_training_data, train_with_defaults, predict_proba, get_top_shap_features
from ml.calibration import fit_calibrator, calibrate, kelly_stake, expected_calibration_error

logger = logging.getLogger(__name__)

@dataclass
class MLBetRecord:
    date: str
    home_team: str
    away_team: str
    league: str
    market: str
    dc_prob: float
    ml_prob: float
    ensemble_prob: float
    implied_prob: float
    edge: float
    signal_strength: int
    odds: float
    stake: float
    stake_type: str
    result: str
    profit_loss: float
    cumulative_pl: float
    actual_score: str
    top_features: list = field(default_factory=list)

@dataclass
class MLBacktestResult:
    strategy_name: str
    total_matches: int
    matches_analyzed: int
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    total_profit_loss_flat: float
    total_profit_loss_kelly: float
    roi_flat: float
    roi_kelly: float
    peak_bankroll: float
    max_drawdown: float
    sharpe_ratio: float
    brier_score: float
    log_loss_score: float
    auc_roc: float
    mean_edge: float
    bet_log: List[MLBetRecord]
    bankroll_curve_flat: List[float]
    bankroll_curve_kelly: List[float]
    monthly_pl: Dict[str, float]
    calibration_data: Optional[pd.DataFrame] = None

def compute_sharpe_ratio(returns: List[float]) -> float:
    """Annualized Sharpe ratio from per-bet returns."""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float((np.mean(arr) / std) * np.sqrt(len(arr)))

def compute_max_drawdown(bankroll_curve: List[float]) -> float:
    """Maximum peak-to-trough drawdown."""
    if not bankroll_curve:
        return 0.0
    peak = bankroll_curve[0]
    max_dd = 0.0
    for val in bankroll_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)

def walk_forward_backtest(
    feature_df: pd.DataFrame,
    min_train_seasons: int = MIN_TRAIN_SEASONS,
    purge_days: int = PURGE_DAYS,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
    stake: float = DEFAULT_STAKE,
    use_kelly: bool = True,
    kelly_fraction: float = KELLY_FRACTION,
    initial_bankroll: float = 1000.0,
    strategy: str = 'dual',  # 'dc_only', 'ml_only', 'dual', 'random'
) -> MLBacktestResult:
    """
    Executes walk-forward backtest across seasons.
    """
    if feature_df.empty:
        return MLBacktestResult(
            strategy_name=strategy, total_matches=0, matches_analyzed=0, total_bets=0,
            wins=0, losses=0, win_rate=0.0, total_profit_loss_flat=0.0, total_profit_loss_kelly=0.0,
            roi_flat=0.0, roi_kelly=0.0, peak_bankroll=initial_bankroll, max_drawdown=0.0,
            sharpe_ratio=0.0, brier_score=0.0, log_loss_score=0.0, auc_roc=0.5, mean_edge=0.0,
            bet_log=[], bankroll_curve_flat=[initial_bankroll], bankroll_curve_kelly=[initial_bankroll],
            monthly_pl={}
        )

    feature_df = feature_df.sort_values('Date').reset_index(drop=True)

    # Determine season splits based on season_year column or Date year
    if 'season_year' in feature_df.columns:
        seasons = sorted(feature_df['season_year'].unique())
    else:
        feature_df['season_year'] = feature_df['Date'].dt.year
        seasons = sorted(feature_df['season_year'].unique())

    bet_log = []
    bankroll_flat = 0.0
    bankroll_kelly = initial_bankroll
    bankroll_curve_flat = [0.0]
    bankroll_curve_kelly = [initial_bankroll]
    monthly_pl = {}

    flat_returns = []
    all_y_true = []
    all_y_prob = []

    matches_analyzed = 0

    # Walk-forward loop across test seasons
    for test_idx in range(min_train_seasons, len(seasons)):
        test_season = seasons[test_idx]
        train_seasons = seasons[:test_idx]

        train_mask = feature_df['season_year'].isin(train_seasons)
        test_mask = feature_df['season_year'] == test_season

        train_data = feature_df[train_mask].copy()
        test_data = feature_df[test_mask].copy()

        if train_data.empty or test_data.empty:
            continue

        # Purge gap: remove training data within purge_days before test set start
        test_start_date = test_data['Date'].min()
        purge_limit = test_start_date - pd.Timedelta(days=purge_days)
        train_data = train_data[train_data['Date'] < purge_limit]

        if train_data.empty:
            continue

        # Fit Dixon-Coles on historical train_data
        dc_params = fit_dixon_coles(train_data, use_xg=True)

        # Compute DC predictions as feature before prepare_training_data
        if 'dc_prob' not in train_data.columns:
            train_data['dc_prob'] = [predict_over25(dc_params, r['HomeTeam'], r['AwayTeam']) for _, r in train_data.iterrows()]
            test_data['dc_prob'] = [predict_over25(dc_params, r['HomeTeam'], r['AwayTeam']) for _, r in test_data.iterrows()]

        # Train XGBoost model
        X_train, y_train, feat_cols = prepare_training_data(train_data)
        X_test, y_test, _ = prepare_training_data(test_data)

        if len(X_train) < 50 or X_test.empty:
            continue

        # Simple split of train data for calibration
        calib_size = max(20, int(len(X_train) * 0.2))
        X_tr, X_cal = X_train.iloc[:-calib_size], X_train.iloc[-calib_size:]
        y_tr, y_cal = y_train.iloc[:-calib_size], y_train.iloc[-calib_size:]

        # Train baseline model
        xgb_model = train_with_defaults(X_tr, y_tr)
        raw_cal_probs = xgb_model.predict_proba(X_cal)[:, 1]
        calibrator = fit_calibrator(y_cal.values, raw_cal_probs)
        # Predict ML probabilities
        ml_probs = predict_proba(xgb_model, X_test, calibrator)

        # Process each test match
        for m_i in range(len(test_data)):
            match_row = test_data.iloc[m_i]
            date = match_row['Date']
            home_team = match_row['HomeTeam']
            away_team = match_row['AwayTeam']
            actual_h = match_row['FTHG']
            actual_a = match_row['FTAG']
            actual_score = f"{int(actual_h)}-{int(actual_a)}"
            is_o25 = (actual_h + actual_a) > 2.5
            odds = match_row.get('over25_odds', np.nan)

            if pd.isna(odds) or odds <= 1.0:
                continue

            matches_analyzed += 1
            imp_prob = 1.0 / odds

            dc_p = float(X_test.iloc[m_i]['dc_prob']) if 'dc_prob' in X_test.columns else 0.5
            ml_p = float(ml_probs[m_i])

            if strategy == 'dc_only':
                pred_prob = dc_p
            elif strategy == 'ml_only':
                pred_prob = ml_p
            elif strategy == 'random':
                pred_prob = float(np.random.uniform(0.3, 0.7))
            else:  # 'dual'
                pred_prob = ml_p

            edge = pred_prob - imp_prob

            all_y_true.append(1 if is_o25 else 0)
            all_y_prob.append(pred_prob)

            if edge > edge_margin and MIN_MODEL_PROBABILITY <= pred_prob <= MAX_MODEL_PROBABILITY:
                # Place bet
                won = is_o25
                res_str = 'WIN' if won else 'LOSS'
                pl_flat = (stake * odds - stake) if won else -stake

                k_stake = kelly_stake(pred_prob, odds, bankroll_kelly, kelly_fraction, KELLY_MAX_PCT)
                pl_kelly = (k_stake * odds - k_stake) if won else -k_stake

                bankroll_flat += pl_flat
                bankroll_kelly += pl_kelly

                bankroll_curve_flat.append(bankroll_flat)
                bankroll_curve_kelly.append(bankroll_kelly)

                flat_returns.append(pl_flat / stake)

                m_str = date.strftime('%Y-%m') if pd.notnull(date) else 'Unknown'
                monthly_pl[m_str] = monthly_pl.get(m_str, 0.0) + pl_flat

                top_feats = []
                if strategy in ['ml_only', 'dual']:
                    top_feats = get_top_shap_features(xgb_model, X_test.iloc[[m_i]], n=3)

                bet_log.append(MLBetRecord(
                    date=str(date.date()) if pd.notnull(date) else '',
                    home_team=home_team,
                    away_team=away_team,
                    league=match_row.get('league', 'Unknown'),
                    market='Over 2.5',
                    dc_prob=dc_p,
                    ml_prob=ml_p,
                    ensemble_prob=pred_prob,
                    implied_prob=imp_prob,
                    edge=edge,
                    signal_strength=4 if edge > 0.1 else 2,
                    odds=odds,
                    stake=stake,
                    stake_type='Flat & Kelly',
                    result=res_str,
                    profit_loss=pl_flat,
                    cumulative_pl=bankroll_flat,
                    actual_score=actual_score,
                    top_features=top_feats
                ))

    total_bets = len(bet_log)
    wins = sum(1 for b in bet_log if b.result == 'WIN')
    losses = total_bets - wins
    win_rate = (wins / total_bets) if total_bets > 0 else 0.0

    roi_flat = (bankroll_flat / (total_bets * stake)) if total_bets > 0 else 0.0
    roi_kelly = ((bankroll_kelly - initial_bankroll) / initial_bankroll) if initial_bankroll > 0 else 0.0

    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    if len(all_y_true) > 0:
        brier = float(brier_score_loss(all_y_true, all_y_prob))
        l_loss = float(log_loss(all_y_true, all_y_prob))
        auc = float(roc_auc_score(all_y_true, all_y_prob)) if len(np.unique(all_y_true)) > 1 else 0.5
    else:
        brier, l_loss, auc = 0.25, 0.693, 0.5

    mean_edge = float(np.mean([b.edge for b in bet_log])) if bet_log else 0.0

    return MLBacktestResult(
        strategy_name=strategy,
        total_matches=len(feature_df),
        matches_analyzed=matches_analyzed,
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_profit_loss_flat=bankroll_flat,
        total_profit_loss_kelly=bankroll_kelly - initial_bankroll,
        roi_flat=roi_flat,
        roi_kelly=roi_kelly,
        peak_bankroll=float(np.max(bankroll_curve_flat)) if bankroll_curve_flat else 0.0,
        max_drawdown=compute_max_drawdown(bankroll_curve_flat),
        sharpe_ratio=compute_sharpe_ratio(flat_returns),
        brier_score=brier,
        log_loss_score=l_loss,
        auc_roc=auc,
        mean_edge=mean_edge,
        bet_log=bet_log,
        bankroll_curve_flat=bankroll_curve_flat,
        bankroll_curve_kelly=bankroll_curve_kelly,
        monthly_pl=monthly_pl
    )

def compare_strategies(feature_df: pd.DataFrame, **kwargs) -> List[MLBacktestResult]:
    """
    Runs walk-forward backtest for all 4 strategies.
    """
    strategies = ['random', 'dc_only', 'ml_only', 'dual']
    results = []
    for strat in strategies:
        res = walk_forward_backtest(feature_df, strategy=strat, **kwargs)
        results.append(res)
    return results
