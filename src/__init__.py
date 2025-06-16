from typing import Any
from ..base import Agent
from .models import GaussainProcess
from .kernels import RBF, WhiteKernel, Exponential, Matern32, Periodic, ConstantKernel, Linear
from .scores import ucb, ei, pi
from .metrics import compute_bootstrap_error, compute_crossvalidation_error, compute_rmse, compute_log_likelihood
from .samplers import optimize_acquisition_function

# Registries for dynamic selection
KERNELS = {
    "rbf": lambda cfg: RBF(cfg.get("length_scale", 0.5)) + WhiteKernel(cfg.get("noise", 1e-6)),
    "matern32": lambda cfg: Matern32(cfg.get("length_scale", 0.5)) + WhiteKernel(cfg.get("noise", 1e-6)),
    "exponential": lambda cfg: Exponential(cfg.get("length_scale", 0.5)) + WhiteKernel(cfg.get("noise", 1e-6)),
    "periodic": lambda cfg: Periodic([cfg.get("length_scale", 0.5), cfg.get("period", 1.0)]) + WhiteKernel(cfg.get("noise", 1e-6)),
    "constant": lambda cfg: ConstantKernel(cfg.get("constant_value", 1.0)) + WhiteKernel(cfg.get("noise", 1e-6)),
    "linear": lambda cfg: Linear(cfg.get("constant_value", 1.0)) + WhiteKernel(cfg.get("noise", 1e-6)),
    "white": lambda cfg: WhiteKernel(cfg.get("noise", 1e-6)),
}

SCORES = {
    "ucb": lambda cfg: ucb(explore=cfg.get("explore", 2.0), exploit=cfg.get("exploit", 1.0)),
    "ei": lambda cfg, y_opt: ei(best_so_far=y_opt),
    "pi": lambda cfg, y_opt: pi(best_so_far=y_opt),
}

MODELS = {
    "classic": GaussainProcess,
    # Add other model types here if needed
}

SAMPLERS = {
    "default": optimize_acquisition_function,
    # Add other samplers here if needed
}

METRICS = {
    "rmse": compute_rmse,
    "log_likelihood": compute_log_likelihood,
    "crossval_error": compute_crossvalidation_error,
    "bootstrap_error": compute_bootstrap_error
    # Add other metrics here if needed
}


def build_component(config: dict, registry: dict, **extra_kwargs):
    config = config.copy()
    if "type" not in config:
        raise ValueError("Missing 'type' key in config.")
    cls_name = config.pop("type")
    if cls_name not in registry:
        raise ValueError(f"Unknown component type: {cls_name}")

    cls_or_lambda = registry[cls_name]

    # If it's a lambda (as in KERNELS), call with config and extra_kwargs
    if callable(cls_or_lambda) and getattr(cls_or_lambda, "__name__", "") == "<lambda>":
        return cls_or_lambda({**config, **extra_kwargs})

    # If it's a class, instantiate with kwargs
    return cls_or_lambda(**config, **extra_kwargs)

class GPExplorer(Agent):
    def __init__(self, config, search_space):
        self.config = config
        self.search_space = search_space
        self.observations = []  # [(x, y)]
        self.new_obs = None

        # Kernel and model
        kernel = build_component(config["kernel"], KERNELS)
        self.model = build_component(config["model"], MODELS, kernel=kernel, feature_bound=search_space)

        # Acquisition function (score)
        self.acquisition_fn = None  # Will be built in learn() when y_sample is available

        # Selector (sampler)
        self.selector = build_component(config["selector"], SAMPLERS)

    def select_action(self, _state=None) -> Any:
        # Use the latest acquisition function
        x_next, _ = self.selector(self.model, self.acquisition_fn, self.search_space)
        return x_next

    def remember(self, _state, _action, _reward, _next_state, _done):
        self.new_obs = len(_action)
        self.observations.append((_action, _next_state))

    def learn(self):
        # Gather all x and y
        x_all = [a for a, _ in self.observations]
        y_all = [s for _, s in self.observations]

        # Fit model
        self.model.fit(x_all, y_all)

        # Build acquisition function with current y_all if needed
        self.acquisition_fn = build_component(self.config["score"], SCORES, y_sample=y_all)