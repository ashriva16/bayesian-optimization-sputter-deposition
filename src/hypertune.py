# pylint: disable=C0114, C0115, C0116, C0103, C3001

from typing import Callable, Dict, Union
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scinav.utils.logger import setup_systems_logger
from .metrics import compute_log_likelihood

log = setup_systems_logger()

def tune_params(x_input: jnp.ndarray,
                y_input: jnp.ndarray,
                kernel: Callable[[jnp.ndarray], Callable],
                func: Callable = compute_log_likelihood,
                ext_noise = None,
                stability_param: float = 1e-5) -> Dict[str, Union[float, jnp.ndarray]]:
    """
    Optimize kernel (and optionally noise) hyperparameters for GP using log-marginal likelihood.

    Parameters:
    - x_input: Training inputs (N, D)
    - y_input: Training targets (N,)
    - kernel_class: Class/function returning kernel given theta (e.g., RBF)
    - kernel_bounds: Bounds for kernel hyperparameters [(min, max), ...]
    - feature_space: 2 x D array with [min, max] for each feature
    - func: Log likelihood function (default: compute_log_likelihood)
    - noise: Scalar noise. If None, will be optimized as an extra parameter.

    Returns:
    - Dict with best score and best parameters (theta [+ noise])
    """
    log.info("Tuning kernel parameters...")

    kernel_bounds = kernel.bounds

    def objective(theta: jnp.ndarray) -> float:
        nll,_ = func(
            kernel=kernel,
            x_input=x_input,
            y_input=y_input,
            theta=theta,
            ext_noise=ext_noise,
            stability_param=stability_param
        )
        return nll

    objective_and_grad = jax.jit(jax.value_and_grad(objective))

    # Grid initialization
    num_init = 5
    grid_ranges = [jnp.linspace(b[0], b[1], num_init) for b in kernel_bounds]
    grid_mesh = jnp.meshgrid(*grid_ranges)
    initial_guesses = jnp.vstack([g.flatten() for g in grid_mesh]).T

    use_jac = True
    best_score = jnp.inf
    best_params = None

    def fun_wrapper(th):
        return float(objective_and_grad(th)[0])

    def jac_wrapper(th):
        grad = objective_and_grad(th)[1]
        if jnp.isnan(grad).any():
            # Detected nan in gradient: disable jac next time
            nonlocal use_jac
            use_jac = False
            return None  # signal minimize to ignore jac
        return jnp.array(grad)

    for init in initial_guesses:
        res = minimize(
            fun=fun_wrapper,
            x0=init,
            jac=jac_wrapper if use_jac else None,
            bounds=kernel_bounds,
            method='L-BFGS-B',
            options={'disp': True}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x

    # log.info(f"Best score: {best_score}, Best params: {best_params}")
    return {
        "score": best_score,
        "theta": best_params,
    }


# def optimize_kernel_param(model,
#                     hyperpram_bound: List[Tuple[float, float]],
#                  func: Callable = compute_log_likelihood) -> Dict[str, Union[float, jnp.ndarray]]:
#     """
#     Optimize the kernel parameters of a given model.

#     Parameters:
#     model: The model whose kernel parameters are to be optimized.
#     hyperpram_bound (List[Tuple[float, float]]): Bounds for the hyperparameters.
#     func (Callable): The function to compute the log likelihood.
#                     Defaults to compute_log_likelihood.

#     Returns:
#     Dict[str, Union[float, jnp.ndarray]]: A dictionary containing the best score
#                                         and the corresponding parameters.
#     """
#     best_score = jnp.inf

#     def objective(theta: jnp.ndarray) -> float:
#         """
#         Objective function to be minimized.

#         Parameters:
#         theta (jnp.ndarray): The hyperparameters to be optimized.

#         Returns:
#         float: The negative log likelihood.
#         """
#         if len(theta) == len(model.kernel.theta):
#             model.kernel.theta = theta
#         elif len(theta) == (len(model.kernel.theta) + 1):
#             model.kernel.theta = theta[1:]
#             model.noise = theta[0]
#         else:
#             raise ValueError(
#                 'Incorrect number of hyperparamters to be optimized')

#         return -func(model, model.X_obs, model.Y_obs)

#     bounds = list(map(tuple, hyperpram_bound))
#     num_points = 5
#     grid_points = [jnp.linspace(b[0], b[1], num_points) for b in bounds]
#     mesh = jnp.meshgrid(*grid_points)
#     initializations = jnp.vstack([m.flatten() for m in mesh]).T

#     best_score = jnp.inf
#     best_x = None
#     for _, hyperparam_init in enumerate(initializations):
#         # Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
#         # We should actually run the minimization several times with different
#         # initializations to avoid local minima but this is skipped here for
#         # simplicity.
#         res = minimize(objective, hyperparam_init,
#                        bounds=bounds,
#                        method='L-BFGS-B')

#         # Keep x if it's the best so far
#         if res.fun < best_score:
#             best_score = res.fun
#             best_x = res.x

#     return {"score": best_score, "param": best_x}

# def tune_hyperparam_parallel(model, hyperpram_bound):

#     import time
#     from timeit import default_timer as timer
#     from multiprocessing import Pool, cpu_count
#     # Store the optimization results in global variables so that we can
#     # compare it later with the results from other implementations.

#     noise_space = [1, .8, .6, .4, .2, .1, .05, .01, .005, .001]
#     l_n = len(noise_space)
#     tl_x, _ = GP_model.kernel.theta_space.shape
#     hyperparam_space = jnp.append(jnp.array(noise_space*tl_x).reshape(-1,1),
#                             jnp.repeat(GP_model.kernel.theta_space, l_n, axis=0),
#                             axis=1)
#     arr = jnp.array([jnp.min(hyperparam_space, axis=0),
#                     jnp.max(hyperparam_space, axis=0)]).T
#     hyperpram_search_bound = tuple(map(tuple, arr))

#     no_procs = 50 # cpu_count()
#     values = [(copy.deepcopy(GP_model), subspace, hyperpram_search_bound)
#               for subspace in jnp.split(hyperparam_space, no_procs)]
#     with Pool() as pool:
#         res = pool.starmap(optimize, values)
#         best_param = min(res, key=lambda x:x['score'])

#     log.info("optimum:\t ", best_param['score'], best_param['param'])
#     return best_param['param']
