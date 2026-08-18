import math

def poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) = (λ^k * e^-λ) / k!"""
    if k < 0: return 0.0
    return (math.pow(lam, k) * math.exp(-lam)) / math.factorial(k)

def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) = sum of PMF from 0 to k"""
    if k < 0: return 0.0
    return sum(poisson_pmf(i, lam) for i in range(k + 1))

def prob_over_n_goals(lam: float, n: int = 2) -> float:
    """P(total goals > n) = 1 - CDF(n, λ). For Over 2.5, n=2."""
    return 1.0 - poisson_cdf(n, lam)

def prob_under_n_goals(lam: float, n: int = 2) -> float:
    """P(total goals <= n) = CDF(n, λ). For Under 2.5, n=2."""
    return poisson_cdf(n, lam)

def prob_exact_score(lam_home: float, lam_away: float, h: int, a: int) -> float:
    """Joint Poisson probability of exact scoreline h-a."""
    return poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)

def prob_draw(lam_home: float, lam_away: float, max_goals: int = 8) -> float:
    """Sum of P(k,k) for k=0..max_goals."""
    return sum(prob_exact_score(lam_home, lam_away, k, k) for k in range(max_goals + 1))

def prob_over25_matrix(lam_home: float, lam_away: float, max_goals: int = 8) -> float:
    """Score matrix method: sum all (h,a) where h+a >= 3."""
    prob = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a >= 3:
                prob += prob_exact_score(lam_home, lam_away, h, a)
    return prob

def prob_under15(lam: float) -> float:
    """P(X <= 1) for Under 1.5 goals."""
    return poisson_cdf(1, lam)

def implied_probability(decimal_odds: float) -> float:
    """
    Returns fair implied probability = 1 / decimal_odds.

    Returns float('nan') for invalid odds (≤ 1.0 or non-finite).
    NaN propagates correctly through edge calculations so corrupted
    data rows are treated as 'no valid odds' rather than masking as
    a 100% implied probability (the previous behaviour).
    """
    import math
    if decimal_odds is None or not math.isfinite(float(decimal_odds)):
        return float('nan')
    if decimal_odds <= 1.0:
        return float('nan')
    return 1.0 / decimal_odds

def has_edge(model_prob: float, implied_prob: float, margin: float = 0.05) -> bool:
    """True if model_prob > implied_prob + margin."""
    return model_prob > (implied_prob + margin)

def expected_value(model_prob: float, decimal_odds: float, stake: float = 10.0) -> float:
    """EV = (prob * odds * stake) - stake."""
    return (model_prob * decimal_odds * stake) - stake

def score_matrix(lam_home: float, lam_away: float, max_goals: int = 6) -> list:
    """Returns a 2D list of probabilities for each scoreline 0-0 through max_goals-max_goals."""
    return [[prob_exact_score(lam_home, lam_away, h, a) for a in range(max_goals + 1)] for h in range(max_goals + 1)]
