# pylint: disable=C0114, C0115, C0116, C0103, C3001

import copy
import jax
import jax.numpy as jnp
# from scipy.linalg import cho_solve, cholesky
from jax.scipy.linalg import cho_solve
from sklearn.model_selection import KFold

def compute_log_likelihood(
                        kernel,
                        x_input: jnp.ndarray,
                        y_input: jnp.ndarray,
                        theta = None,
                        ext_noise = None,
                        stability_param: float = 1e-5):
    """
    Compute negative log marginal likelihood of a GP using a kernel, with input/output scaling.

    Parameters:
    - kernel: Function that computes the kernel matrix.
    - x_input (jnp.ndarray): Training inputs, shape (N, D).
    - y_input (jnp.ndarray): Training targets, shape (N,) or (N, 1).
    - feature_space (jnp.ndarray): 2 x D array defining [min, max] bounds for each feature.
    - noise (float or jnp.ndarray): Observation noise (scalar or per-point vector).
    - prior_mean (jnp.ndarray): Optional prior mean (same shape as y_input).
    - scale_inputs (bool): Whether to normalize inputs based on feature_space.
    - stability_param (float): Jitter for numerical stability.

    Returns:
    - neg_log_likelihood (float): Negative log marginal likelihood.
    - (t1, t2, t3): Tuple of individual likelihood terms.
    """

    num_samples = x_input.shape[0]

    if y_input.ndim == 1:
        y_input = y_input[:, None]

    # Kernel matrix
    k_mat = kernel(x_input, x_input, theta=theta)
    if ext_noise is None:
        k_mat += stability_param**2 * jnp.eye(num_samples)
    elif jnp.isscalar(ext_noise) or len(ext_noise) == 1:
        k_mat += (ext_noise**2) * jnp.eye(num_samples)
    elif len(ext_noise) == num_samples:
        k_mat += jnp.diag(ext_noise**2)
    else:
        raise ValueError(
            f"ext_noise must be None, a scalar, or have length {num_samples}, \
                got shape {jnp.shape(ext_noise)}"
        )

    # Cholesky decomposition
    lower_traingle = jax.scipy.linalg.cholesky(k_mat, lower=True)

    # Compute alpha = (K + σ²I)^-1 * (y - μ)
    alpha = cho_solve((lower_traingle, True), y_input)

    # Log likelihood terms
    t1 = 0.5 * jnp.sum(jnp.einsum("ik,ik->k", y_input, alpha))
    t2 = jnp.sum(jnp.log(jnp.clip(jnp.diag(lower_traingle), a_min=1e-10)))
    t3 = 0.5 * num_samples * jnp.log(2 * jnp.pi)

    neg_log_likelihood = t1 + t2 + t3

    return neg_log_likelihood, (t1, t2, t3)

def compute_rmse(gp_model, x_input: jnp.ndarray, y_input: jnp.ndarray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) of the Gaussian Process model.

    Parameters:
    - gp_model: The Gaussian Process model.
    - x_input (jnp.ndarray): Input data.
    - y_input (jnp.ndarray): Target values.

    Returns:
    - rmse (float): The RMSE of the model.
    """
    mu, _ = gp_model.predict(x_input)
    rmse = jnp.linalg.norm(y_input - mu.flatten()) / jnp.sqrt(len(y_input))

    return rmse

def compute_crossvalidation_error(base_model, x_train: jnp.ndarray,
                                  y_train: jnp.ndarray, k: int = 5) -> float:
    """
    Perform k-fold cross-validation on a Gaussian Process surrogate model.

    Parameters:
    - base_model: The base Gaussian Process model.
    - x_train (jnp.ndarray): Training input data.
    - y_train (jnp.ndarray): Training target values.
    - k (int): Number of folds.

    Returns:
    - avg_validation_error (float): Mean validation error across folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    validation_errors = []

    for train_index, val_index in kf.split(x_train):
        # Split data into training and validation sets
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train the surrogate model
        surrogate_model = copy.deepcopy(base_model)
        surrogate_model.noise = 0.001
        surrogate_model.add_observation(x_train_fold, y_train_fold, scale=True)

        # Evaluate on validation set (Negative Log Marginal Likelihood or RMSE)
        y_pred, _ = surrogate_model.predict(x_val_fold)
        error = jnp.mean((y_pred - y_val_fold) ** 2)  # Mean Squared Error (MSE)
        validation_errors.append(error)

    avg_validation_error = jnp.mean(validation_errors)
    return avg_validation_error

def compute_bootstrap_error(base_model, x_train: jnp.ndarray,
                            y_train: jnp.ndarray,
                            n_bootstraps: int = 30) -> float:
    """
    Perform bootstrap resampling cross-validation on a Gaussian Process surrogate model.

    Parameters:
    - base_model: The base Gaussian Process model.
    - x_train (jnp.ndarray): Training input data.
    - y_train (jnp.ndarray): Training target values.
    - n_bootstraps (int): Number of bootstrap resamples.

    Returns:
    - avg_validation_error (float): Mean validation error across bootstrap samples.
    """
    n_samples = len(x_train)
    validation_errors = []

    key = jax.random.PRNGKey(0)
    for _ in range(n_bootstraps):
        # Bootstrap sample: randomly sample indices with replacement
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(subkey, n_samples, shape=(n_samples,), replace=True)
        x_train_bootstrap, y_train_bootstrap = x_train[indices], y_train[indices]

        # Select out-of-bag (OOB) data points (not included in bootstrap sample)
        oob_mask = jnp.ones(n_samples, dtype=bool)
        oob_mask[indices] = False  # Mark sampled points
        x_val_oob, y_val_oob = x_train[oob_mask], y_train[oob_mask]

        # Train surrogate model on bootstrap sample
        surrogate_model = copy.deepcopy(base_model)
        surrogate_model.noise = 0.001
        surrogate_model.add_observation(
            x_train_bootstrap, y_train_bootstrap, scale=True)

        # Evaluate on OOB samples if available
        if len(x_val_oob) > 0:
            y_pred, _ = surrogate_model.predict(x_val_oob)
            # Mean Squared Error (MSE)
            error = jnp.mean((y_pred - y_val_oob) ** 2)
            validation_errors.append(error)

    avg_validation_error = jnp.mean(
        validation_errors) if validation_errors else jnp.nan
    return avg_validation_error
