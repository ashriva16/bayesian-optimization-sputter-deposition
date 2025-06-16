from typing import Callable, Tuple
import numpy as np
import scipy as sp

def optimize_acquisition_function(gp,
                    acq_func: Callable[[np.ndarray], np.ndarray],
                    bounds: np.ndarray, n_restarts: int = 10,
                    seed=42) -> Tuple[np.ndarray, float]:
    """
    Perform greedy sampling to select the next point to query.

    Parameters:
    acq_func (Callable): The acquisition function to optimize.
    bounds (np.ndarray): Bounds for the search space.
    n_restarts (int): Number of restarts for optimization.

    Returns:
    Tuple[np.ndarray, float]: The next point to query and its acquisition value.
    """
    rng = np.random.default_rng(seed)

    def objective(x):
        x = np.array(x).reshape(1, -1)
        _pred_stats = gp.predict(x)
        return -acq_func(_pred_stats)  # Minimize the negative acquisition function

    dimension = bounds.shape[0]
    init_array = rng.uniform(low=bounds[:, 0], high=bounds[:, 1],
                                   size=(n_restarts, dimension))

    best_score = np.inf
    best_x = None
    for x_init in init_array:
        res = sp.optimize.minimize(objective, x_init, bounds=bounds,
                                   options={'eps': 1e-6, 'gtol': 1e-10, 'ftol': 1e-8},
                                   method='L-BFGS-B')
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x

    if best_x is not None:
        best_x = best_x.reshape(1, dimension)
    return best_x, -best_score


def probabilistic_sampling(gp, acq_func: Callable[[np.ndarray], float],
                           bounds: np.ndarray,
                           batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rejection sampling approach for probabilistic sampling.

    Parameters:
    sigma_func (Callable): Function to compute the standard deviation for a given input.
    bounds (np.ndarray): Bounds for the search space.
    batch_size (int): Number of samples to generate.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Sampled points and their scores.
    """

    def truncated_multivariate_normal(mean: np.ndarray,
                                    cov: np.ndarray,
                                    lower: np.ndarray,
                                    upper: np.ndarray, size: int) -> np.ndarray:
        """
        Sample from a multivariate truncated normal distribution using rejection sampling.

        Parameters:
        - mean: array-like, shape (N,)
        - cov: array-like, shape (N, N)
        - lower: array-like, shape (D,)
        - upper: array-like, shape (D,)
        - size: int, number of samples

        Returns:
        - samples: array, shape (size, N)
        """
        samples = []
        mvn = sp.stats.multivariate_normal(mean, cov)

        while len(samples) < size:
            # Sample from the multivariate normal distribution
            sample = mvn.rvs(size=size)

            # Handle the case where size=1 and sample shape becomes (D,)
            if size == 1:
                # Reshape to (1, D) to make the masking consistent
                sample = sample.reshape(1, -1)

            # Check if each sample is within the bounds (lower and upper)
            mask = np.all((sample >= lower) & (sample <= upper), axis=1)

            # Append valid samples
            valid_samples = sample[mask]
            samples.extend(valid_samples)

            # If enough valid samples are collected, stop
            if len(samples) >= size:
                break
        return np.array(samples[:size])

    def objective(x):
        x = np.array(x).reshape(1, -1)
        _pred_stats = gp.predict(x)
        return -acq_func(_pred_stats)  # Minimize the negative acquisition function

    dimension = bounds.shape[0]

    res = sp.optimize.minimize(objective, x0=np.zeros(dimension), bounds=bounds, method='L-BFGS-B')
    max_score = -res.fun

    xs, scores = [], []
    mean = np.zeros([dimension])
    cov = np.identity(dimension)

    while len(xs) < batch_size:
        u = np.random.uniform(0, 1)
        x = truncated_multivariate_normal(mean, cov, bounds[:, 0], bounds[:, 1], size=1)
        pred_stats = gp.predict(x)
        score = acq_func(pred_stats)
        if u < score / max_score:
            xs.extend(x)
            scores.append(score)

    return np.array(xs), np.array(scores).flatten()
