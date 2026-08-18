"""
Probability Calibration and Staking Module
============================================
Platt Scaling (logistic regression) calibration for probability reliability,
ECE error calculation, and Quarter-Kelly Criterion bankroll management.

FIX H2: Replaced IsotonicRegression with Platt Scaling.
Isotonic regression fits staircase step functions that overfit on small
per-league validation sets (N ≈ 60–80 matches). Platt scaling fits a
smooth 2-parameter logistic function that is far more robust on small
sample sizes and generalises better out-of-sample.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from typing import Tuple, Optional
import logging

from ml.config import KELLY_FRACTION, KELLY_MAX_PCT

logger = logging.getLogger(__name__)

def fit_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    """
    Fits a Platt Scaling calibrator (logistic regression on raw model outputs).

    Maps uncalibrated model probabilities → empirical posterior probabilities.
    Robust on small datasets (N ≥ 20) unlike isotonic which overfits for N < 200.

    Returns a fitted LogisticRegression that responds to .predict_proba().
    Use calibrate() to apply it.
    """
    if len(y_true) < 5:
        logger.warning("Too few samples (%d) to fit calibrator — returning None", len(y_true))
        return None

    X = y_prob.reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    try:
        lr.fit(X, y_true)
        return lr
    except Exception as e:
        logger.warning("Platt calibration fitting failed: %s", e)
        return None

def calibrate(
    calibrator,
    y_prob: np.ndarray,
    clip_min: float = 0.01,
    clip_max: float = 0.99
) -> np.ndarray:
    """
    Applies fitted Platt calibrator to raw model probabilities.
    Falls back to clipped raw probabilities if no calibrator is fitted.
    """
    if calibrator is None:
        return np.clip(y_prob, clip_min, clip_max)
    try:
        X = np.array(y_prob).reshape(-1, 1)
        calibrated = calibrator.predict_proba(X)[:, 1]
        return np.clip(calibrated, clip_min, clip_max)
    except Exception:
        return np.clip(y_prob, clip_min, clip_max)

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    if len(y_true) == 0:
        return 0.0

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # Calculate bin weights
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1

    ece = 0.0
    n_samples = len(y_prob)

    for i in range(len(prob_true)):
        bin_mask = (bin_assignments == i)
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = prob_true[i]
            bin_conf = prob_pred[i]
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)

def calibration_report(y_true: np.ndarray, y_prob_raw: np.ndarray, y_prob_calibrated: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Generates report dataframe comparing raw vs calibrated predicted probabilities.
    """
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=n_bins)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=n_bins)

    records = []
    min_len = min(len(prob_pred_raw), len(prob_pred_cal))

    for i in range(min_len):
        records.append({
            'bin_pred_raw': prob_pred_raw[i],
            'actual_freq_raw': prob_true_raw[i],
            'bin_pred_cal': prob_pred_cal[i],
            'actual_freq_cal': prob_true_cal[i],
        })

    return pd.DataFrame(records)

def kelly_stake(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float = KELLY_FRACTION,
    max_pct: float = KELLY_MAX_PCT,
    max_abs_stake: float = 500.0,   # FIX N6: hard ceiling in £ regardless of bankroll
) -> float:
    """
    Calculates Quarter-Kelly Criterion stake in GBP.

    Full Kelly fraction f* = (b*p - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p

    max_abs_stake caps the recommendation at £500 by default.
    This prevents extreme recommendations if a large bankroll is entered
    (e.g. £100,000 bankroll → 5% cap = £5,000 stake, above Betfair liquidity).
    """
    if decimal_odds <= 1.0 or model_prob <= 0.0 or bankroll <= 0.0:
        return 0.0

    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p

    f_star = (b * p - q) / b
    if f_star <= 0.0:
        return 0.0

    f_adj = f_star * fraction
    f_adj = min(f_adj, max_pct)

    raw_stake = bankroll * f_adj
    # FIX N6: Apply hard absolute ceiling
    return round(min(raw_stake, max_abs_stake), 2)
