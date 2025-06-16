# pylint: disable=C0114, C0115, C0116, C0103, C3001

from typing import Optional, Sequence, Tuple, Union, final
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import jit, vmap, grad, jacrev
from scinav.utils.logger import setup_systems_logger

log = setup_systems_logger()

class BaseKernel(ABC):
    def __init__(self):
        self._jit_cache = {}

    @staticmethod
    @abstractmethod
    def kernel_fn(x1, x2, theta=None):
        # needs to be static for computation using jit, vmap, grad etc.
        pass  # This should be overridden by subclasses

    @abstractmethod
    def _validate_theta(self, x1, x2, theta):
        pass

    @final
    def _evaluate(self, kernel_fn, x1, x2, deriv_type="0", theta=None):
        x1 = jnp.asarray(x1, dtype=jnp.float64)
        x2 = jnp.asarray(x2, dtype=jnp.float64)

        theta = self._validate_theta(x1, x2, theta)

        if deriv_type == "0":
            inner = lambda a: vmap(lambda b: kernel_fn(a, b, theta))(x2)
            return vmap(inner)(x1)

        elif deriv_type == "dkda":
            grad_k = grad(kernel_fn, argnums=0)
            inner = lambda a: vmap(lambda b: grad_k(a, b, theta))(x2)
            return vmap(inner)(x1)

        elif deriv_type == "dkdb":
            grad_k = grad(kernel_fn, argnums=1)
            inner = lambda a: vmap(lambda b: grad_k(a, b, theta))(x2)
            return vmap(inner)(x1)

        elif deriv_type == "d2kda2":
            hess = jacrev(grad(kernel_fn, argnums=0), argnums=0)
            inner = lambda a: vmap(lambda b: hess(a, b, theta))(x2)
            return vmap(inner)(x1)

        elif deriv_type == "d2kdb2":
            hess = jacrev(grad(kernel_fn, argnums=1), argnums=1)
            inner = lambda a: vmap(lambda b: hess(a, b, theta))(x2)
            return vmap(inner)(x1)

        elif deriv_type == "d2kdadb":
            mix_hess = jacrev(grad(kernel_fn, argnums=0), argnums=1)
            inner = lambda a: vmap(lambda b: mix_hess(a, b, theta))(x2)
            return vmap(inner)(x1)

        elif deriv_type == "d4kda2db2":
            d4 = jacrev(jacrev(jacrev(jacrev(kernel_fn, 0), 0), 1), 1)
            inner = lambda a: vmap(lambda b: d4(a, b, theta))(x2)
            return vmap(inner)(x1)

        elif deriv_type == "dktheta":
            key = (id(self.kernel_fn), deriv_type, x1.shape, x2.shape)

            if key not in self._jit_cache:
                log.info("Compiling JIT for %s, %s, %s", key, (x1, x2), theta)
                grad_theta = grad(kernel_fn, argnums=2)

                # Create a lambda that takes x1, x2 as constants, returns a function of theta
                fn = lambda theta: vmap(lambda a: vmap(lambda b: grad_theta(a, b, theta))(x2))(x1)

                # Compile for fixed shapes (use x1, x2 passed here to close over them)
                self._jit_cache[key] = jit(fn)

            return self._jit_cache[key](theta)

        else:
            raise ValueError(f"Unknown deriv_type: {deriv_type}")

    @final
    def __add__(self, other):
        if not isinstance(other, BaseKernel):
            raise ValueError("Can only add with another BaseKernel")
        return SumKernel(self, other)

    def __call__(self, x1, x2, deriv_type="0", theta=None):
        return self._evaluate(self.kernel_fn, x1, x2, deriv_type, theta)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Linear(BaseKernel):
    def __init__(self, theta: Union[float, Sequence[float]]=1.0,
                bounds: Optional[Sequence[Tuple[float, float]]] = None):
        super().__init__()
        self.theta = theta

        if bounds is None:
            self.bounds = [(0.1, 10.0)]
        else:
            self.bounds = bounds

        self.num_params = 0

    def _validate_theta(self, x1, x2, theta):
        dim = x1.shape[-1]

        if theta is None:
            return jnp.full((dim,), self.theta)

        theta = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float64))

        if theta.ndim == 0:
            return jnp.full((dim,), theta)
        elif theta.shape != (dim,):
            raise ValueError(f"theta must be scalar or have shape ({dim},), got {theta.shape}")

        return theta

    @staticmethod
    def kernel_fn(x1, x2, theta=None):
        return 1 + jnp.dot(x1, x2)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class ScalarParamKernel(BaseKernel):
    def __init__(
        self,
        theta: float = 1.0,
        bounds: Optional[Sequence[Tuple[float, float]]] = None
    ):
        super().__init__()
        self.theta = jnp.array([theta], dtype=jnp.float64)  # Always shape (1,)
        self.n_dim = 0 # for scalar param
        self.num_params = 1

        if bounds is None:
            self.bounds = [(0.01, 10.0)]
        else:
            if len(bounds) != self.num_params:
                raise ValueError(f"Expected bounds of length 1, got {len(bounds)}")
            self.bounds = bounds

    def _validate_theta(self, x1, x2, theta):
        if theta is None:
            theta = self.theta
        else:
            theta = jnp.array(theta, dtype=jnp.float64)
            if theta.ndim == 0:
                theta = theta[None]  # Convert scalar to shape (1,)
            elif theta.shape != (1,):
                raise ValueError(f"Expected scalar theta (shape (1,)), got {theta.shape}")
        return theta

class ConstantKernel(ScalarParamKernel):

    def __init__(
        self,
        constant_value: float = 1.0,
        bounds: Optional[Sequence[Tuple[float, float]]] = None
    ):
        if bounds is None:
            bounds = [(1e-6, 5)]
        super().__init__(theta=constant_value, bounds=bounds)

    @staticmethod
    def kernel_fn(x1, x2, theta=None):
        # theta can be used as the constant value if provided
        return jnp.squeeze(theta)

class WhiteKernel(ScalarParamKernel):

    def __init__(
        self,
        noise: float = 1.0,
        bounds: Optional[Sequence[Tuple[float, float]]] = None
    ):
        if bounds is None:
            bounds = [(1e-6, 1)]
        super().__init__(theta=noise, bounds=bounds)

    @staticmethod
    def kernel_fn(x1, x2, theta=None):
        is_close = jnp.allclose(x1, x2, rtol=1e-5, atol=1e-8)
        return jnp.where(is_close, jnp.squeeze(theta)**2, 0.0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class SingleParamKernel(BaseKernel):

    def __init__(
        self,
        theta: Union[float, Sequence[float]],
        bounds: Optional[Sequence[Tuple[float, float]]] = None
        ):
        super().__init__()

        # Convert to array
        theta = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float64))
        self.theta = theta
        self.num_params = theta.size
        self.n_dim = self.num_params  # One param per dimension

        if bounds is None:
            self.bounds = [(0.01, 1.0)] * self.n_dim
        else:
            if len(bounds) != self.n_dim:
                raise ValueError(f"Expected bounds of length {self.n_dim}, got {len(bounds)}")
            self.bounds = bounds

    def _validate_theta(self, x1, x2, theta):
        n_dims = x1.shape[-1]

        if n_dims != self.n_dim:
            raise ValueError(f"Input dimension {n_dims} \
                does not match kernel dimension {self.n_dim}")

        if theta is None:
            theta = self.theta
        else:
            theta = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float64))

        if theta.ndim != 1 or theta.size != n_dims:
            raise ValueError(
                f"Expected theta to be a flat array of length {n_dims}, got shape {theta.shape}"
            )

        return theta

class RBF(SingleParamKernel):
    def __init__(self, length_scale=1.0, bounds=None):
        super().__init__(theta=length_scale, bounds=bounds)

    @staticmethod
    def kernel_fn(x1, x2, theta=None):
        # theta must always be provided here
        lengthscales = theta
        return jnp.exp(-jnp.sum((x1 - x2)**2 / (2 * lengthscales**2)))

class Exponential(SingleParamKernel):
    def __init__(self, length_scale=1.0, bounds=None):
        super().__init__(theta=length_scale, bounds=bounds)

    @staticmethod
    def kernel_fn(x1, x2, theta=None):
        lenghtscales = theta
        r = jnp.sqrt(jnp.sum(jnp.abs(x1 - x2) / lenghtscales))
        return jnp.exp(-r)

class Matern32(SingleParamKernel):
    def __init__(self, length_scale=1.0, bounds=None):
        super().__init__(theta=length_scale, bounds=bounds)

    @staticmethod
    def kernel_fn(x1, x2, theta=None):
        lengthscales = theta
        r = jnp.sqrt(jnp.sum(((x1 - x2) / lengthscales) ** 2))
        sqrt3_r = jnp.sqrt(3) * r
        return (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class Periodic(BaseKernel):
    def __init__(
        self,
        theta: Sequence[Tuple[float, float]],  # Required to infer dimensionality
        bounds: Optional[Sequence[Tuple[float, float]]] = None
    ):
        super().__init__()

        if not isinstance(theta, Sequence) or not all(isinstance(p, Tuple) and len(p) == 2 for p in theta):
            raise TypeError("theta must be a sequence of (lengthscale, period) tuples.")

        self.n_dim = len(theta)
        self.theta = jnp.array([v for pair in theta for v in pair], dtype=jnp.float64)
        self.num_params = self.theta.size

        if bounds is None:
            self.bounds = [(1e-5, 1e2), (1e-2, 1e2)] * self.n_dim
        else:
            if not isinstance(bounds, Sequence) or len(bounds) != 2 * self.n_dim:
                raise ValueError(f"bounds must be of length {2 * self.n_dim}, got {len(bounds)}")
            self.bounds = bounds

        self.default_set = True

    def _validate_theta(self, x1, x2, theta):
        n_dims = x1.shape[-1]

        if n_dims != self.n_dim:
            raise ValueError(f"Input dimension {n_dims} \
                does not match kernel dimension {self.n_dim}")

        if theta is None:
            theta = self.theta
        else:
            theta = jnp.asarray(theta, dtype=jnp.float64)

        if theta.ndim != 1 or theta.size != 2 * n_dims:
            raise ValueError(
                f"Expected theta to be a flat array of length {2 * n_dims}, got shape {theta.shape}"
            )

        return theta

    @staticmethod
    def kernel_fn(x1, x2, theta=None):

        n_dims = x1.shape[-1]
        lengthscales = theta[:n_dims]
        periods = theta[n_dims:]

        # Assumes x1, x2 are 1D arrays of the same shape
        diff = x1 - x2  # (d,)
        sin_term = jnp.sin(jnp.pi * diff / periods)
        scaled = -2 * jnp.sum((sin_term ** 2) / (lengthscales ** 2))
        return jnp.exp(scaled)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class KernelOperations(BaseKernel):
    def __init__(self, k1: BaseKernel, k2: BaseKernel):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.theta = jnp.concatenate([k1.theta, k2.theta])
        self.num_params = self.k1.num_params + self.k2.num_params
        self.bounds = self.k1.bounds + self.k2.bounds

    @staticmethod
    def kernel_fn(x1, x2, theta=None):
        # Not directly usable; subclasses must implement __call__
        raise NotImplementedError("kernel_fn is not used in KernelOperations. \
                                Use __call__ in derived classes.")

    def _validate_theta(self, x1, x2, theta: Optional[jnp.ndarray]) -> Tuple[Optional[jnp.ndarray],
                                                                            Optional[jnp.ndarray]]:
        if theta is None:
            theta1 = self.theta[:self.k1.num_params]
            theta2 = self.theta[self.k1.num_params:]
        else:
            theta = jnp.asarray(theta, dtype=jnp.float64)
            if theta.size != self.num_params:
                raise ValueError(
                    f"Expected theta of length {self.num_params}, got {theta.size}"
                )

            theta1 = theta[:self.k1.num_params]
            theta2 = theta[self.k1.num_params:]
        return theta1, theta2

class SumKernel(KernelOperations):

    def _compute_dkdtheta(self, x1, x2, theta1, theta2):
        dkdtheta1 = self.k1(x1, x2, deriv_type="dktheta", theta=theta1)
        dkdtheta2 = self.k2(x1, x2, deriv_type="dktheta", theta=theta2)

        return jnp.concatenate([dkdtheta1, dkdtheta2], axis=-1)

    def __call__(self, x1, x2, deriv_type="0", theta=None):
        theta1, theta2 = self._validate_theta(x1, x2, theta)
        if deriv_type == "dktheta":
            return self._compute_dkdtheta(x1, x2, theta1, theta2)
        return self.k1(x1, x2, deriv_type, theta1) + self.k2(x1, x2, deriv_type, theta2)

# END
