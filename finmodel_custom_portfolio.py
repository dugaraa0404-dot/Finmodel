"""Custom long-only portfolio allocation analytics for Finmodel.

Use alongside existing automatic minimum-volatility and maximum-Sharpe optimizers.
All return inputs must be decimal annual rates; covariance is annualized.
"""
from __future__ import annotations
import numpy as np


def analyze_custom_allocation(
    symbols: list[str],
    weights_percent: list[float],
    expected_returns: list[float],
    covariance_matrix: list[list[float]],
    risk_free_rate: float = 0.0,
    asset_betas: list[float] | None = None,
) -> dict:
    """Evaluate user-specified weights without optimizing or changing them."""
    n = len(symbols)
    if n < 1 or len(set(symbols)) != n:
        raise ValueError("Select at least one unique asset.")
    w = np.asarray(weights_percent, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance_matrix, dtype=float)
    if w.shape != (n,) or mu.shape != (n,) or cov.shape != (n, n):
        raise ValueError("Weights, returns, and covariance dimensions must match symbols.")
    if not (np.isfinite(w).all() and np.isfinite(mu).all() and np.isfinite(cov).all()):
        raise ValueError("All numerical inputs must be finite.")
    if np.any(w < 0) or np.any(w > 100) or not np.isclose(w.sum(), 100, atol=0.01):
        raise ValueError("Long-only weights must be 0–100% and sum to 100%.")
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("Covariance matrix must be symmetric.")
    if np.linalg.eigvalsh(cov).min() < -1e-8:
        raise ValueError("Covariance matrix must be positive semidefinite.")
    w = w / 100.0
    annual_return = float(w @ mu)
    annual_variance = max(0.0, float(w @ cov @ w))
    annual_volatility = float(np.sqrt(annual_variance))
    sharpe = ((annual_return - risk_free_rate) / annual_volatility
              if annual_volatility > 0 else None)
    marginal = cov @ w
    variance_contribution = w * marginal
    volatility_contribution = (variance_contribution / annual_volatility
                               if annual_volatility > 0 else np.zeros(n))
    result = {
        "method": "custom",
        "symbols": symbols,
        "weights_percent": dict(zip(symbols, (w * 100).tolist())),
        "expected_return": annual_return,
        "variance": annual_variance,
        "volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "risk_contributions": dict(zip(symbols, volatility_contribution.tolist())),
        "risk_contribution_percent": dict(zip(symbols, (variance_contribution / annual_variance * 100).tolist()
                                              if annual_variance > 0 else [0.0] * n)),
    }
    if asset_betas is not None:
        betas = np.asarray(asset_betas, dtype=float)
        if betas.shape != (n,) or not np.isfinite(betas).all():
            raise ValueError("Asset beta dimensions must match symbols.")
        result["beta"] = float(w @ betas)
    return result
