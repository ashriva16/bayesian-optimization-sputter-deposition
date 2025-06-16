# pylint: disable=C0114, C0115, C0116, C0103, C3001

from typing import Tuple, Union, Callable
import copy
from jax.scipy.linalg import cholesky, cho_solve, solve_triangular
import jax.numpy as jnp
import jax
from scinav.utils.logger import setup_systems_logger
from .hypertune import tune_params
from .metrics import compute_log_likelihood

log = setup_systems_logger()

GPR_CHOLESKY_LOWER = True
jax.config.update("jax_enable_x64", True)

def _zero_mean_function(x: jnp.ndarray) -> jnp.ndarray:
    """Returns a zero vector of shape (x.shape[0], 1)"""
    result = jnp.zeros((x.shape[0], ))
    return jnp.asarray(result, dtype=jnp.float64)

def validate_y_shape(y):
    """
    validate shape of output
    """

    if y.ndim == 1:
        return y
    elif y.ndim == 2 and y.shape[1] == 1:
        return y
    else:
        raise ValueError("y must be of shape (n,) or (n,1)")

class GaussainProcess:
    """
    A Gaussian Process (GP) model for regression tasks.

    This class provides methods to initialize a GP model,
    add observations, update noise and kernel functions,
    switch scaling, and calculate the posterior distribution,
    gradient, and Hessian for new samples.

    TODO Test assumtions for kernel is stationary and isotropic,

    Attributes:
        prior_mean_func (callable): The prior mean function.
        _kernel (callable): The kernel function.
        bounds (jnp.ndarray): The feature bounds for scaling.
        noise (float): The noise level in the GP model.
        _scale (bool): Whether to scale the feature and output data.
        x_obs (jnp.ndarray): The observed feature data.
        y_obs (jnp.ndarray): The observed output data.
        x_obs_std (jnp.ndarray): The scaled observed feature data.
        y_obs_std (jnp.ndarray): The scaled observed output data.
        center_y (float): The mean of the observed output data for scaling.
        scale_y (float): The standard deviation of the observed output data for scaling.
        prior_cov (jnp.ndarray): The prior covariance matrix.
        mu_prior (jnp.ndarray): The prior mean of the observed output data.

    Methods:
        __init__(self, prior_mean=None, kernel=None, feature_bound=None, scale=False):
            Initializes the Gaussian Process model.
        get_scale_x(self, x: jnp.ndarray) -> jnp.ndarray:
            Scales the feature data.
        invscale_x(self, x_std: jnp.ndarray) -> jnp.ndarray:
            Inversely scales the feature data.
        get_scale_y(self, y: jnp.ndarray) -> jnp.ndarray:
            Scales the output data.
        invscale_y(self, y_std: jnp.ndarray) -> jnp.ndarray:
            Inversely scales the output data.
        _scale_observations(self) -> None:
            Scales the observed feature and output data.
        _update_prior(self, nx: int) -> None:
            Updates the prior covariance matrix with new observations.
        fit(self, x: Union[jnp.ndarray, float], y: Union[jnp.ndarray, float]) -> None:
            Adds new observations to the Gaussian Process model.
        update_noise(self, new_noise: float) -> None:
            Updates the noise level in the Gaussian Process model.
        update_kernel(self, new_kernel: Optional[callable] = None,
            new_theta: Optional[jnp.ndarray] = None) -> None:
            Updates the kernel function or its parameters.
        switch_scaling(self, scale: bool) -> None:
            Switches the scaling of the feature and output data.
        get_posterior(self, x: Union[jnp.ndarray, float]) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Calculates the posterior distribution, serving as predictions for new samples.
        get_posterior_grad(self, x: Union[jnp.ndarray, float]) -> Tuple[jnp.ndarray, jnp.ndarray]:
            Computes posterior mean and covariance for the gradient of the function.
        get_posterior_hessian(self, x: Union[jnp.ndarray, float]) -> Tuple[jnp.ndarray,
                                                                            jnp.ndarray]:
            Computes posterior mean and covariance for the Hessian of the function.

    """

    def __init__(self, kernel, feature_bound, prior_mean=None, scale=True):

        """

        Initializes the Gaussian Process model.

        Args:
            prior_mean (callable, optional): The prior mean function. Defaults to zero function.
            kernel (callable, optional): The kernel function. Defaults to None.
            feature_bound (jnp.ndarray, optional): The feature bounds for scaling. Defaults to None.
            scale (bool, optional): Whether to scale the feature and output data. Defaults to False.

        """

        assert feature_bound is not None, "Feature bounds must be provided."
        self.feature_space = jnp.asarray(feature_bound.T, dtype=jnp.float64)
        self.feature_dim = self.feature_space.shape[1]

        assert kernel is not None, "Kernel function must be provided."
        self.kernel = copy.copy(kernel)

        self.stability_param = 1e-5  # Default non-zero Noise for stability
        self._scale = scale  # Always scale the observed data
        self.optimparams = kernel.theta

        self.x_train = None
        self.y_train = None
        self.noise_log = None
        self.sigma = None

        self._x_std = None
        self._y_std = None

        # Default scaling params
        self.center_y = 0
        self.center_x = 0
        self.scale_y = 1
        self.scale_x = 1

        # prior mean is provided at original scale
        self.prior_cov = None
        self.prior_mean = None

        # always on original data
        if prior_mean is None:
            self.mu_prior_func = _zero_mean_function  # Default zero mean function
        else:
            self.mu_prior_func = prior_mean  # Assign user-provided function

        # internal variables
        self.low_triang_k = None
        self.alpha = None
        self.acq_order = 0

    def get_scale_x(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Scales the feature data.

        Args:
            x (jnp.ndarray): The feature data to be scaled.

        Returns:
            jnp.ndarray: The scaled feature data.
        """

        assert x.shape[1] == self.feature_space.shape[1]

        x_std = (x - self.center_x)/self.scale_x
        return jnp.asarray(x_std, dtype=jnp.float64)

    def invscale_x(self, x_std: jnp.ndarray) -> jnp.ndarray:
        """
        Inversely scales the feature data.

        Args:
            x_std (jnp.ndarray): The scaled feature data.

        Returns:
            jnp.ndarray: The original feature data.
        """

        assert x_std.shape[1] == self.feature_space.shape[1]

        if self._scale:
            min_val = self.feature_space[0]
            x = x_std * self.scale_x + min_val
        else:
            x = x_std

        return jnp.asarray(x, dtype=jnp.float64)

    def get_scale_y(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Scales the output data.

        Args:
            y (jnp.ndarray): The output data to be scaled.

        Returns:
            jnp.ndarray: The scaled output data.
        """

        y_std = (y - self.center_y) / jnp.where(self.scale_y != 0, self.scale_y, 1)

        return y_std

    def invscale_y(self, y_std: jnp.ndarray) -> jnp.ndarray:
        """
        Inversely scales the output data.

        Args:
            y_std (jnp.ndarray): The scaled output data.

        Returns:
            jnp.ndarray: The original output data.
        """

        y = y_std * jnp.where(self.scale_y != 0, self.scale_y, 1) + self.center_y

        return jnp.asarray(y, dtype=jnp.float64)

    def _scale_observations(self) -> None:
        """
        Scales the observed feature and output data.
        In practice, such normalization is often used to
        (a) enhance the numerical stability and
        (b) scale the training objective to make it “easier” to optimize numerically.
        nothing to do with assumption of y ~ N(0, k(x,x))
        """
        # fit scales
        if self._scale:
            self.scale_x = self.feature_space[1]-self.feature_space[0]
            self.center_x = self.feature_space[0]
            self.center_y = self.y_train.mean(axis=0, keepdims=True).T
            self.scale_y = self.y_train.std(axis=0, keepdims=True).T

        # GP is modeled on this data with 0 mean and 1 variance
        self._x_std = self.get_scale_x(self.x_train)
        self._y_std = self.get_scale_y(self.y_train)

    def _update_prior(self, nx: int = None) -> None:
        """
        Updates the prior covariance matrix with new observations.
        However, the lenght scale for kernels is fixed here.

        Args:
            nx (int): The number of new observations.
        """
        log.info("Updating prior covariance with %s new observations.", nx)
        log.info("input data: \n%s", self.x_train)
        log.info("scaled input data: \n%s", self._x_std)
        log.info("response data: \n%s", self.y_train)
        log.info("standardzed response data: \n%s", self._y_std)
        log.info("ext noise data: \n%s", self.sigma)

        # always on original data
        prior_mean_std = self.get_scale_y(self.mu_prior_func(self.x_train))

        if self.prior_cov is None or nx is None:
            # Build prior covariance matrix from scratch
            self.prior_cov = self.kernel(self._x_std, self._x_std)
            self.prior_cov += jnp.diag(self.sigma**2)

        else:
            # Update the prior covariance matrix with new observations
            new_x_obs = self._x_std[-nx:]  # The newly added observations

            # off-diagonal block of the new observations
            off_diagonal_block_2 = self.kernel(new_x_obs, self._x_std[:-nx])
            off_diagonal_block_1 = off_diagonal_block_2.T

            # diagonal block of the new observations
            diagonal_block = self.kernel(new_x_obs, new_x_obs)
            diagonal_block += jnp.diag(self.sigma[-nx:]**2)

            self.prior_cov = jnp.block([
                [self.prior_cov, off_diagonal_block_1],
                [off_diagonal_block_2, diagonal_block]
            ])

        # K = L L^T cholesky decomposition
        self.low_triang_k = cholesky(self.prior_cov, lower=GPR_CHOLESKY_LOWER, check_finite=False)

        # alpha = (K(X,X)+sigma^2I)^−1 (Y-mu(X)) = L^T \ (L \ y)
        self.alpha = cho_solve((self.low_triang_k, GPR_CHOLESKY_LOWER),
                                self._y_std - prior_mean_std,
                                check_finite=False,)

    def _add_observations(self, x: jnp.ndarray, y: jnp.ndarray,
                        obs_noise: jnp.ndarray = None) -> None:
        """
        Args:
            x (jnp.ndarray): _description_
            y (jnp.ndarray): _description_
            noise (jnp.ndarray, optional): _description_. Defaults to None.
        """
        # Ensure features are numpy arrays
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        num_obs = x.shape[0]

        # Ensure `x` is 2D: shape (n_samples, n_features)
        assert x.ndim == 2, f"Expected features to be 2D \
        (n_samples, n_dimension), but got shape {x.shape}"

        # Ensure `y` is either of shape (n,) or (n,1)
        y = validate_y_shape(y)

        # Ensure `x` and `y` have the same number of samples
        assert x.shape[0] == y.shape[0], \
            f"Mismatch: features has {x.shape[0]} samples, " \
            f"but target has {y.shape[0]} samples"

        assert x.shape[1] == self.feature_dim, (
            f"Feature mismatch: Expected features to have {self.feature_dim} dimensions, "
            f"but got {x.shape[1]}"
        )

        # Update data
        if self.x_train is None:
            self.x_train = x.copy()
            self.y_train = y.copy()
        else:
            self.x_train = jnp.append(self.x_train, x, axis=0)
            self.y_train = jnp.append(self.y_train, y)

        # scale data
        self._scale_observations()

        # Determine noise array
        if obs_noise is None:
            scale_noise = False
            noise_arr = jnp.full((num_obs,), self.stability_param, dtype=jnp.float64)
        else:
            scale_noise = True
            obs_noise = jnp.asarray(obs_noise, dtype=jnp.float64)
            if obs_noise.ndim == 0 or obs_noise.shape[0] == 1:
                noise_arr = jnp.full((num_obs,), obs_noise, dtype=jnp.float64)
            elif obs_noise.ndim == 1 and obs_noise.shape[0] == num_obs:
                noise_arr = obs_noise
            else:
                raise ValueError(f"Expected obs_noise to be of length 1 or {num_obs}, "
                                f"but got shape {obs_noise.shape}")

        self.noise_log = noise_arr if self.noise_log is None else jnp.concatenate([self.noise_log,
                                                                                noise_arr])

        # Apply scaling only if obs_noise was provided
        self.sigma = self.noise_log / self.scale_y if scale_noise else self.noise_log

    def fit(self,
            x: Union[jnp.ndarray, float],
            y: Union[jnp.ndarray, float],
            obs_noise: jnp.ndarray = None,
            auto_tune: bool = True,
            func: Callable = compute_log_likelihood) -> None:

        self._add_observations(x, y, obs_noise)

        if auto_tune:
            # Tune parameters
            result = tune_params(
                                x_input=self._x_std,
                                y_input=self._y_std,
                                kernel = self.kernel,
                                func=func,
                                ext_noise=self.sigma,
                                stability_param=self.stability_param
                                )
            self.optimparams = result['theta']
            log.info("Optimized kernel parameters:\t%s", result['theta'])
            self.kernel.theta = jnp.asarray(result['theta'], dtype=jnp.float64)
            self._update_prior()
        else:
            num_new_obs = len(x)
            self._update_prior(num_new_obs)

    def _get_posterior(self, x_star: Union[jnp.ndarray, float],
                    return_cov = False) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # Computing posterior mean and variance ------------------------------------
        # mu_pos(x*) = mu(x*) + K(X*,X) (K(X,X)+sigma^2I)^−1 (Y-mu(X))
        # Sigma_pos(x*) = (K(x*,x*)) − K(x*,X) (K(X,X)+sigma^2I)^−1 K(X,x*)

        x = self.invscale_x(x_star)
        posterior_mu = self.get_scale_y(self.mu_prior_func(x))
        posterior_cov = self.kernel(x_star, x_star)

        if self.x_train is not None and self.y_train is not None:

            k_star_x = self.kernel(x_star, self._x_std)
            k_x_star = k_star_x.T

            # computing K(X*,X) (K(X,X)+sigma^2I)^−1 (Y_train_std)
            # posterior_mu =  k_star_x @ self.alpha
            posterior_mu += jnp.einsum('ij,j->i', k_star_x, self.alpha)

            # computing K(x*,X) (K(X,X)+sigma^2I)^−1 K(X,x*)
            # Solve L V = K(X, X^*) for V (forward substitution)
            v = solve_triangular(self.low_triang_k, k_x_star, lower=True, check_finite=False)

            # Solve L^T Vt = V for Vt (backward substitution)
            vt = solve_triangular(self.low_triang_k.T, v, lower=False, check_finite=False)

            # posterior_cov -= (k_star_x @ vt)
            posterior_cov -= jnp.einsum('ij,jk->ik', k_star_x, vt)

        # undo normalisation
        posterior_mu = self.invscale_y(posterior_mu)
        posterior_cov = posterior_cov * self.scale_y**2

        if return_cov:
            return posterior_mu, posterior_cov

        posterior_sd = jnp.sqrt(jnp.diag(posterior_cov)).astype(jnp.float64)
        return posterior_mu, posterior_sd

    def _get_posterior_grad(self, x_star: Union[jnp.ndarray, float],
                            return_cov = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes posterior mean and covariance for the gradient of the function.

        Args:
            x (Union[jnp.ndarray, float]): feature data points for which gradient predictions are
            made.
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Predictive mean and variance
            of the posterior distribution for the gradient at each feature data point.
        """

        # Computing posterior mean and variance ------------------------------------
        # mu_pos(x*) = mu(x*) + K(X*,X) (K(X,X)+sigma^2I)^−1 (Y-mu(X))
        # Sigma_pos(x*) = (K(x*,x*)) − K(x*,X) (K(X,X)+sigma^2I)^−1 K(X,x*)

        mu_std_deriv = jnp.zeros((x_star.shape))
        posterior_cov = self.kernel(x_star, x_star, deriv_type='d2kdadb')

        if self.x_train is not None and self.y_train is not None:

            dk_star_x = self.kernel(x_star, self._x_std, deriv_type='dkda')
            dk_x_star = self.kernel(self._x_std, x_star, deriv_type='dkdb')

            # computing K(X*,X) (K(X,X)+sigma^2I)^−1 (Y_train_std)
            # mu_std_deriv +=  dk_star_x @ self.alpha
            mu_std_deriv +=  jnp.einsum("nmd,m->nd", dk_star_x, self.alpha)

            # computing K(x*,X) (K(X,X)+sigma^2I)^−1 K(X,x*)
            # Solve L V = K(X, X^*) for V (forward substitution)
            v = solve_triangular(self.low_triang_k, dk_x_star, lower=True, check_finite=False)

            # Solve L^T Vt = V for Vt (backward substitution)
            vt = solve_triangular(self.low_triang_k.T, v, lower=False, check_finite=False)

            # posterior_cov -= (dk_star_x @ vt)
            posterior_cov -= jnp.einsum('amb,mcd->acbd', dk_star_x, vt)

        # undo normalisation
        posterior_mu = mu_std_deriv * self.scale_y / self.scale_x
        posterior_cov = posterior_cov * self.scale_y**2 / self.scale_x ** 2

        if return_cov:
            return posterior_mu, posterior_cov

        dim = x_star.shape[1]
        num = x_star.shape[0]
        diagonal_elements = jnp.array([posterior_cov[i, i, j, j] \
            for i in range(num) for j in range(dim)]).reshape(num, dim)
        posterior_sd = jnp.sqrt(diagonal_elements).astype(jnp.float64)

        return posterior_mu, posterior_sd

    def _get_posterior_hessian(self, x_star: Union[jnp.ndarray, float],
                            return_cov = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes posterior mean and covariance for the Hessian of the function.

        Args:
            x (Union[jnp.ndarray, float]): feature data points
            for which Hessian predictions are made.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Predictive mean and variance
            of the posterior distribution for the Hessian at each feature data point.
        """

        # Computing posterior mean and variance ------------------------------------
        # mu_pos(x*) = mu(x*) + K(X*,X) (K(X,X)+sigma^2I)^−1 (Y-mu(X))
        # Sigma_pos(x*) = (K(x*,x*)) − K(x*,X) (K(X,X)+sigma^2I)^−1 K(X,x*)

        mu_std_hess = jnp.zeros((x_star.shape[0], x_star.shape[1], x_star.shape[1])) # for now
        posterior_cov = self.kernel(x_star, x_star, deriv_type='d4kda2db2')
        if self.x_train is not None and self.y_train is not None:

            hk_star_x = self.kernel(x_star, self._x_std, deriv_type='d2kda2')
            hk_x_star = self.kernel(self._x_std, x_star, deriv_type='d2kdb2')

            # computing K(X*,X) (K(X,X)+sigma^2I)^−1 (Y_train_std)
            # mu_std +=  hk_star_x @ self.alpha
            mu_std_hess +=  jnp.einsum("nmab,m->nab", hk_star_x, self.alpha)

            # computing K(x*,X) (K(X,X)+sigma^2I)^−1 K(X,x*)
            # Solve L V = K(X, X^*) for V (forward substitution)
            v = solve_triangular(self.low_triang_k, hk_x_star, lower=True, check_finite=False)

            # Solve L^T Vt = V for Vt (backward substitution)
            vt = solve_triangular(self.low_triang_k.T, v, lower=False, check_finite=False)

            posterior_cov -= jnp.einsum('amij,mbkl->abijkl', hk_star_x, vt)

        # undo normalisation
        posterior_mu = mu_std_hess * self.scale_y / self.scale_x**2
        posterior_cov = posterior_cov * self.scale_y**2 / self.scale_x ** 4

        if return_cov:
            return posterior_mu, posterior_cov

        num = x_star.shape[0]
        dim = x_star.shape[1]

        # only interested in  gp of d2f/dx2_i and not d2f/dx_i dx_j
        posterior_mu_diag = jnp.array([posterior_mu[i, j, j] \
            for i in range(num) for j in range(dim)]).reshape(num, dim)

        diagonal_elements = jnp.array([posterior_cov[i, i, j, j, j, j] \
            for i in range(num) for j in range(dim)]).reshape(num, dim)
        posterior_sd = jnp.sqrt(diagonal_elements).astype(jnp.float64)

        return posterior_mu_diag, posterior_sd

    def predict(self, x: Union[jnp.ndarray, float], order = 0,
                return_cov = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates the posterior distribution, serving as predictions for new samples.

        Args:
            x (Union[jnp.ndarray, float]): feature data points for which predictions are made.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Predictive mean and variance
            of the posterior distribution for each feature data point.
        """

        # make test feature data type consistent
        x = jnp.asarray(x)

        assert x.ndim == 2, f"Expected x to be 2D (n_samples, n_dimensions), \
            but got shape {x.shape}"
        assert x.shape[1] == self.feature_dim, (
            f"Feature mismatch: Expected features to have {self.feature_dim} dimensions, "
            f"but got {x.shape[1]}"
        )
        # scale the test feature
        x_star = self.get_scale_x(x)

        if order == 0:
            return self._get_posterior(x_star, return_cov = return_cov)
        elif order == 1:
            return self._get_posterior_grad(x_star, return_cov = return_cov)
        elif order == 2:
            return self._get_posterior_hessian(x_star, return_cov = return_cov)
