"""
This module contains functions for calculating different acquisition functions
used in Bayesian Optimization, including Expected Improvement (EI) and Upper
Confidence Bound (UCB) with and without Hessian.
"""

from typing import Tuple
import numpy as np
from scipy.stats import norm

def ei(explore: float = 0, best_so_far: float = None) -> np.ndarray:

    _best_so_far = best_so_far

    def _compute(stats : Tuple) -> np.ndarray:
        """
        Calculate the Expected Improvement (EI) for Bayesian Optimization.

        Parameters:
        mu (np.ndarray): Mean predictions from the GP model.
        sigma (np.ndarray): Standard deviation predictions from the GP model.
        x (np.ndarray): Input array of shape (D,) where D is the number of features.

        Returns:
        np.ndarray: The expected improvement value.
        """
        mu, sigma = stats

        # Calculate z (standardized improvement)
        z = (mu - _best_so_far - explore) / sigma

        # Exploitation (mean improvement)
        exploit_val = (mu - _best_so_far - explore) * norm.cdf(z)

        # Exploration (uncertainty)
        explore_val = sigma * norm.pdf(z)

        return (exploit_val + explore_val).reshape(-1)

    return _compute

def pi(explore: float = 0.01, best_so_far: float = None) -> np.ndarray:
    """
    Calculate the Probability of Improvement (PI) acquisition function for Bayesian Optimization.

    Parameters:
    explore (float): Controls exploration. Higher values encourage exploring uncertain regions.
    exploit (float): Multiplier for the mean term (optional, included for structural consistency).
    best_so_far (float): Current best observed function value.

    Returns:
    Callable: A function that computes PI given (mu, sigma).
    """
    _best_so_far = best_so_far

    def _compute(stats: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        mu, sigma = stats
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero

        if _best_so_far is None:
            raise ValueError("best_so_far must be provided to compute PI.")

        adjusted_mu = mu  # included for flexibility/consistency

        z = (adjusted_mu - _best_so_far - explore) / sigma
        pi_value = norm.cdf(z)
        return pi_value.reshape(-1)

    return _compute

def ucb(explore: float = 0.5, exploit: float = 0.5) -> np.ndarray:
    """
    Calculate the Upper Confidence Bound (UCB) for Bayesian Optimization.

    Parameters:
    mu (np.ndarray): Mean predictions from the GP model.
    sigma (np.ndarray): Standard deviation predictions from the GP model.
    x (np.ndarray): Input array.
    explore (float): Exploration parameter.
    exploit (float): Exploitation parameter.

    Returns:
    np.ndarray: The UCB value.
    """
    _explore = explore
    _exploit = exploit

    def _compute(stats: Tuple):
        mu, sigma = stats
        result = (_exploit * mu + _explore * sigma).reshape(-1)
        return result

    return _compute
