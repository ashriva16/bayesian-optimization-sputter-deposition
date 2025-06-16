"""
This module contains the BayesoptTracker class,
which is used to track and log the progress of Bayesian Optimization.

The BayesoptTracker class provides methods to log training setup,
track log likelihood and RMSE, log experiment details,
save checkpoints, and plot metrics.

Dependencies:
    - logging
    - pathlib.Path
    - pickle
    - datetime.datetime
    - numpy
    - scipy.linalg.cho_solve, scipy.linalg.cholesky
    - main.utils.scplot

Usage:
    tracker = BayesoptTracker(model)
    tracker.log_train_setup(config_dict, exp_name)
    tracker.track_indicators()
    tracker.log_experiment(x_next, acq_score, y_next)
    tracker.save_checkpoint()
    tracker.plot_metrics(save_path)
"""
# pylint: disable=C0114, C0115, C0116, C0103, C3001

import re
import logging
from pathlib import Path
import pickle
from datetime import datetime

import numpy as np
import scinav.utils.scplot as splt

splt.style.use(splt.PUBLICATION_STYLE)
# save_checkpoint(surrogate_model, config['output_dir'])

class BayesoptTracker:
    """
    A class to track and log the progress of Bayesian Optimization.

    Attributes:
        model: The surrogate model used in Bayesian Optimization.
        metrics: A dictionary to store various metrics.
        metrics_titles: A dictionary to store titles for the metrics.
        output_dir: Path to the log directory.
        file: Logger object for logging information.
        formatter: Formatter object for formatting log messages.
    """

    def __init__(self, config_dict: dict,
                exp_name: str = None,
                display: bool = True,
                log_to_file: bool = False):
        """
        Initializes the BayesoptTracker with a state and optional display setting.

        Args:
            state: The surrogate model used in Bayesian Optimization.
            display (bool): Whether to display logs on the console. Defaults to True.
        """

        # Initialize dictionaries to store the metrics
        self.metrics = {
            "rmse": [],
            "nllh": [],
            "acq_score": [],
            "x_next": [],
            "y_next": [],
            "observed_x_opt": [],
            "observed_y_opt": [],
            # "K_condition_number": []
        }

        self.metrics_titles = {
            "acq_score": "Acquisition Score",
            "x_next": "Next Sample Input",
            "y_next": "Next Sample Output",
            "observed_x_opt": "Observed Optimal Input",
            "observed_y_opt": "Observed Optimal Output"
        }

        self.indicators = {}

        self.model = None
        self.output_dir = None
        self.log_to_file = log_to_file

        # logger setup
        self.file = logging.getLogger(f"tracker.{exp_name or 'default'}")
        self.file.setLevel(logging.INFO)
        self.formatter = logging.Formatter("%(message)s")
        # self.formatter = logging.Formatter(
        #     "%(asctime)-15s %(levelname)-8s %(name)s:%(lineno)d \n%(message)s\n")

        self.file.handlers = []  # Clear previous handlers to avoid duplicate logs
        if display:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(self.formatter)
            self.file.addHandler(console_handler)

        self.config_dict = config_dict
        if 'output_dir' in self.config_dict:
            directory = self.config_dict['output_dir']
        else:
            # If result_dir is not provided, set a default directory
            directory = Path.cwd() / "results"

        directory = Path(directory) / f"{exp_name}"
        self.config_dict['output_dir'] = directory

    def log_train_setup(self, model, name_suffix_list: list = None):
        """
        Sets up the logging directory and initial configuration for training.

        Args:
            config_dict (dict): Configuration dictionary containing training settings.
            exp_name (str): Name of the experiment.
            name_suffix_list (list, optional): List of suffixes to append
                                to the experiment name. Defaults to None.
            log_to_file (bool): Whether to log to a file. Defaults to True.
        """
        self.model = model
        relog_flag = True # by default, log the initial setup

        # Result directory setup --------------------------------
        if name_suffix_list is not None:
            name_suffix_str = "_".join(map(str, name_suffix_list))
            self.config_dict['name_suffix_str'] = name_suffix_str
            output_dir = Path(self.config_dict['output_dir']) / f"_{name_suffix_str}"
        else:
            output_dir = Path(self.config_dict['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True) # output dir with name suffix

        if 'run_ver' not in self.config_dict or self.config_dict['run_ver'] is None:
            # creating new run folder
            existing_runs = [entry.name for entry in output_dir.iterdir() if entry.is_dir()]
            run_nums = [int(re.search(r'run(\d+)', name).group(1)) \
                for name in existing_runs if re.match(r'run\d+', name)]
            next_run = f'run{max(run_nums)+1}' if run_nums else 'run0'
            self.config_dict['run_ver'] = next_run
            output_dir = Path(output_dir) / self.config_dict['run_ver']
            self.config_dict['output_dir'] = output_dir
        else:
            relog_flag = False  # if run_ver is already set, do not log again

        self.output_dir = self.config_dict['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True) # outdir with run_ver

        # Add logfile handler -----------------------------------
        if self.log_to_file:
            logfilename = Path(self.config_dict['output_dir']) / 'bayesopt.log'
            file_handler = logging.FileHandler(logfilename)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(self.formatter)
            self.file.addHandler(file_handler)

        if relog_flag:
            # Log initial info --------------------------------------
            self.file.info("\nSettings")
            self.file.info("%s", '==' * 28)
            self.file.info("".join([f'{key}:\t{value}\n' for key, value in self.config_dict.items()]))
            self.file.info("%s", '==' * 28)

    def track_indicator(self, name, metric) -> float:

        self.indicators.setdefault(name, []).append(metric)
        self.file.info("\t%s:\t %s", name, self.indicators[name][-1])

    def log_experiment(self, x_next: np.ndarray, acq_score: float, y_next: float = None):
        """
        Tracks and logs the next sampled point, acquisition score, and best observed point.

        Args:
            x_next (np.ndarray): The next sampled point in the search space.
            acq_score (float): The acquisition function score for `x_next`.
            y_next (float, optional): The observed function value at `x_next`. Defaults to None.
        """
        self.metrics["x_next"].append(x_next)
        vals = ", ".join(f"{x:.4f}" for x in x_next.flatten().tolist())
        self.file.info(
            "\t%s:\t %s", self.metrics_titles['x_next'], vals)

        self.metrics["acq_score"].append(acq_score)
        self.file.info(
            "\t%s:\t %s", self.metrics_titles['acq_score'], self.metrics['acq_score'][-1])

        if y_next is not None:
            self.metrics["y_next"].append(y_next.flatten())
            vals = ", ".join(f"{x:.4f}" for x in y_next.flatten().tolist())
            self.file.info("\t%s:\t %s", self.metrics_titles['y_next'], vals)

        # Update the best observed optimum if the new point is better
        optm_indx = np.argmax(self.model.y_train)

        self.metrics["observed_y_opt"].append(self.model.y_train[optm_indx])
        self.file.info("\t%s:\t %s", self.metrics_titles['observed_y_opt'],
                    self.metrics['observed_y_opt'][-1].item())

        self.metrics["observed_x_opt"].append(self.model.x_train[optm_indx])
        vals = ", ".join(f"{x:.4f}" for x in self.model.x_train[optm_indx].flatten().tolist())
        self.file.info("\t%s:\t %s", self.metrics_titles['observed_x_opt'], vals)

    def log_additional_info(self, step: int, additional_info: str = None):
        """
        Logs the most recent tracked metrics to console and file.

        Args:
            step (int): The current step or iteration.
            additional_info (dict, optional): Any additional information
            to log (e.g., model parameters). Defaults to None.
        """
        # Log any additional info (if provided)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.info("\n\nStep: %d | Time: %s", step, timestamp)

        # Log any additional info (if provided)
        if additional_info is not None:
            if not isinstance(additional_info, str):
                raise TypeError(f"`additional_info` must be a string, got \
                    {type(additional_info).__name__}")
            self.file.info("\tAdditional Info: %s", additional_info)

    def save_checkpoint(self, addtional_suffix_list: list = None):
        """
        Saves the model at a checkpoint using pickle.

        Args:
            addtional_suffix_list (list, optional): List of suffixes to
            append to the checkpoint filename. Defaults to None.
        """
        if addtional_suffix_list is not None:
            name_suffix_str = "_".join(map(str, addtional_suffix_list))
            filename = name_suffix_str
            file_path = Path(self.output_dir) / f"model_{filename}.pkl"
        else:
            file_path = Path(self.output_dir) / "model.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

        self.file.info("✅ model saved at: %s", file_path)

    def plot_metrics(self, save_path: str = None):
        """
        Plots all metrics and indicators in the respective dictionaries and saves the plot.

        Args:
            save_path (str): Path to save the plot.
        """
        # Merge metrics and indicators for plotting
        combined_data = {}

        # Reorganize special cases in metrics
        def reorganize():
            if "x_next" in self.metrics and self.metrics["x_next"]:
                self.metrics["x_next"] = np.vstack(self.metrics["x_next"])
            if "observed_x_opt" in self.metrics and self.metrics["observed_x_opt"]:
                self.metrics["observed_x_opt"] = np.vstack(self.metrics["observed_x_opt"])

        reorganize()

        # Add non-empty metrics
        combined_data.update({k: v for k, v in self.metrics.items() if len(v)})
        # Add non-empty indicators
        combined_data.update({k: v for k, v in self.indicators.items() if len(v)})

        num_metrics = len(combined_data)
        num_cols = 2
        num_rows = (num_metrics + 1) // num_cols

        fig, axs = splt.subplots(num_rows, num_cols, figsize=(12, num_rows * 4))
        axs = axs.flatten()

        for idx, (name, values) in enumerate(combined_data.items()):
            try:
                axs[idx].plot(values, label=name)
                axs[idx].set_title(name)
                axs[idx].legend()
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not plot {name} due to error: {e}")

        # Remove unused axes
        for i in range(num_metrics, len(axs)):
            fig.delaxes(axs[i])

        splt.tight_layout()

        if save_path is None:
            fig.savefig(Path(self.config_dict['output_dir']) / "track_bayesopt.png")
        splt.close(fig)