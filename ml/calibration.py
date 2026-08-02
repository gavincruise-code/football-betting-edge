"""
Probability Calibration and Staking Module
============================================
Isotonic regression calibration for probability reliability, ECE error calculation,
and Quarter-Kelly Criterion bankroll management.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from typing import Tuple, Optional
import logging

from ml.config import KELLY_FRACTION, KELLY_MAX_PCT

logger = logging.getLogger(__name__)

def fit_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    """
    Fits isotonic regression calibrator mapping uncalibrated probabilities to empirical frequencies.
    """
    ir = IsotonicRegression(out_of_bounds='clip', y_min=0.01, y_max=0.99)
    ir.fit(y_prob, y_true)
    return ir

def calibrate(
    calibrator: IsotonicRegression,
    y_prob: np.ndarray,
    clip_min: float = 0.01,
    clip_max: float = 0.99
) -> np.ndarray:
    """
    Applies fitted calibrator to probabilities.
    """
    if calibrator is None:
        return np.clip(y_prob, clip_min, clip_max)
    calibrated = calibrator.predict(y_prob)
    return np.clip(calibrated, clip_min, clip_max)

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
    max_pct: float = KELLY_MAX_PCT
) -> float:
    """
    Calculates Quarter-Kelly Criterion stake in GBP.
    
    Full Kelly fraction f* = (b*p - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p
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

    return round(bankroll * f_adj, 2)
