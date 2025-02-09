import os
import sys

sys.path.append(os.path.abspath(os.path.join("..")))

from methods.models.abstract_model import SpatialFairnessModel
import numpy as np
from methods.heuristics.heuristics_func import (
    get_simple_heu_points,
    get_iterative_points,
    get_exhaustive_points,
    get_regions_flip_info,
    get_df_points,
)

from utils.scores import compute_statistic
from utils.stats_utils import get_signif_threshold
import pandas as pd
import pickle


class SpatialFlipFairnessModel(SpatialFairnessModel):
    """
    A class that implements a spatial fairness model based on heuristic methods.
    """

    def __init__(self, method):
        """
        Initializes the SpatialFlipFairnessModel with a specified heuristic method.

        Args:
            method (str): The heuristic method to use. Must be one of the predefined methods.
        """

        super().__init__(method)

        assert (
            method in self.spatial_flip_methods
        ), f"Invalid Method, should be one of: {self.spatial_flip_methods}"
        self.heu_method_func_dict = {
            "nregions": get_simple_heu_points,
            "nflips": get_simple_heu_points,
            "stat": get_simple_heu_points,
            "iter": get_iterative_points,
            "exhaust": get_exhaustive_points,
        }

    def _load(self, file_path):
        """
        Loads the model from a serialized pickle (.pkl) file.

        Args:
            file_path (str): The path to the file containing the serialized model.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

        with open(file_path, "rb") as f:
            model_data = pickle.load(f)

        super().__init__(model_data["method"])
        self.budget_to_solution_info = model_data["budget_to_solution_info"]
        self.method = model_data["method"]
        self.overlap = model_data["overlap"]
        self.fitted = 2
        self.heu_method_func_dict = {
            "nregions": get_simple_heu_points,
            "nflips": get_simple_heu_points,
            "stat": get_simple_heu_points,
            "iter": get_iterative_points,
            "exhaust": get_exhaustive_points,
        }

    def get_model_fit_info_as_df(self):
        """
        Returns the model's fitting information as a Pandas DataFrame.

        Raises:
            ValueError: If the model has not been fitted.

        Returns:
            pd.DataFrame: A DataFrame containing information about budgets, changes and fit times.
        """

        if self.fitted != 2:
            raise ValueError("Model must be fitted before getting info.")

        data = []
        for n_flips, info in self.budget_to_solution_info.items():
            data.append(
                {
                    "budget": n_flips,
                    "pts_to_change": info["pts_to_change"],
                    "pts_to_change_sol": info["pts_to_change_sol"],
                    "time": info["time"],
                }
            )

        return pd.DataFrame(data)

    @classmethod
    def from_pretrained(cls, file_path):
        """
        Loads a pretrained model from a file and returns a new instance.

        Args:
            file_path (str): Path to the serialized model file.

        Returns:
            SpatialFlipFairnessModel: A new instance of the model loaded from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

        # Create an instance with a dummy method (will be overwritten)
        model = cls.__new__(cls)
        model._load(file_path)
        return model

    def multi_fit(
        self,
        points_per_region,
        n_flips_start,
        step,
        n_flips,
        y_pred,
        overlap=False,
        verbose=0,
    ):
        """
        Fits the fairness model using heuristic-based flipping strategies to adjust predictions for fairness.

        This function iteratively computes the required flips in predicted labels (`y_pred`)
        to mitigate spatial bias sutisgying budget constraint (number of flips allowed). It evaluates multiple
        levels of flips, starting from `n_flips_start` and increasing up to `n_flips`, with a step size of `step`
        and for each save the labels indices that should be changed and the respective flips 0/-1/1 for
        no action / positive to negative flip / negative to positive flip.
        The function supports **statistical parity** fairness notion.

        Parameters:
            points_per_region (list of lists):
                A list where each sublist contains indices of data points belonging to a particular spatial region.

            n_flips_start (int):
                The starting number of label flips in the heuristic process.

            step (int):
                The increment size for the number of flips in each iteration.

            n_flips (int):
                The maximum number of flips allowed in the heuristic optimization.

            y_pred (numpy array):
                The predicted binary labels (0 or 1) for each data point.

            overlap (bool, optional, default=False):
                Whether to consider overlapping regions when computing fairness constraints.

            verbose (int, optional, default=0):
                Controls the level of output during processing:
                    - `0`: No output.
                    - `1`: Displays progress information.

        """

        self.overlap = overlap
        y_pred = np.array(y_pred)
        non_empty_pts_per_region, _ = self._get_non_empty_regions_info(
            points_per_region
        )

        N = len(y_pred)
        P = np.sum(y_pred)

        regions = [{"points": pts} for pts in non_empty_pts_per_region]
        signif_thresh_ = get_signif_threshold(
            0.005, 400, regions, N, P, seed=36, verbose=False
        )

        regions_df = pd.DataFrame(regions)

        regions_df["signif"] = regions_df["points"].apply(
            lambda pts: compute_statistic(len(pts), np.sum(y_pred[pts]), N, P)
            > signif_thresh_
        )
        df_regions_with_info = get_regions_flip_info(regions_df, N, P, y_pred)
        df_points = get_df_points(df_regions_with_info, N)

        self.budget_to_solution_info = self._get_heuristics_solutions(
            n_flips_start=n_flips_start,
            step=step,
            n_flips_=n_flips,
            df_signif=df_regions_with_info,
            df_points=df_points,
            N=N,
            P=P,
            y_pred=y_pred,
            method=self.method,
            signif_thresh_=signif_thresh_,
            report_progress=verbose > 0,
        )

        self.fitted = 2

    def multi_predict(self, y_pred, budgets=[]):
        """
        Generates spatial fairness predictions for different budget levels by applying the computed flips.

        Args:
            y_pred (np.ndarray): Predictions.
            budgets (list, optional): List of budget levels. Defaults to an empty list.

        Returns:
            dict: A dictionary mapping budgets to predicted labels.

        Raises:
            ValueError: If the model has not been multi-fitted before calling this method.
        """

        if self.fitted != 2:
            raise ValueError("Model must be multi fitted before multi prediction.")

        budget_to_predictions = {}
        budgets = budgets if budgets != [] else self.budget_to_solution_info.keys()

        for budget in budgets:
            solution_info = self.budget_to_solution_info[budget]
            y_pred_new = y_pred.copy()
            y_pred_new[solution_info["pts_to_change"]] = (
                y_pred_new[solution_info["pts_to_change"]]
                + solution_info["pts_to_change_sol"]
            )
            budget_to_predictions[budget] = y_pred_new

        return budget_to_predictions

    def _get_heuristics_solutions(
        self,
        n_flips_start,
        step,
        n_flips_,
        df_signif,
        df_points,
        N,
        P,
        y_pred,
        method,
        signif_thresh_=None,
        report_progress=True,
        pts_sol_idx_map=None,
    ):
        """
        Computes heuristic-based solutions for flipping classification labels to achieve spatial fairness.

        This function applies various heuristic methods to iteratively adjust predictions (`y_pred`) within
        specified fairness constraints. The goal is to determine an optimal set of label changes that minimize
        fairness disparity while preserving accuracy.

        Parameters:
            n_flips_start (int):
                The starting number of label flips in the heuristic process.

            step (int):
                The step size to increment the number of flips in each iteration.

            n_flips_ (int):
                The total number of flips allowed for the optimization.

            df_signif (pd.DataFrame):
                A DataFrame containing information about spatial regions, their points, and associated significance.

            df_points (pd.DataFrame):
                A DataFrame containing individual points and their related information.

            N (int):
                The total number of data points.

            P (int):
                The total number of positively classified points in `y_pred`.

            y_pred (numpy array):
                The predicted binary labels (0 or 1) for each data point.

            method (str):
                The heuristic method to use. Supported methods include:
                    - `"nregions"`
                    - `"nflips"`
                    - `"stat"`
                    - `"iter"`
                    - `"exhaust"`

            signif_thresh_ (float, optional, default=None):
                The significance threshold used to determine significant regions for flipping.

            report_progress (bool, optional, default=True):
                Whether to display progress updates during computation.

            pts_sol_idx_map (dict, optional, default=None):
                A mapping from solution point indices to their original indices.

        Returns:
            dict:
                A dictionary mapping each budget (number of flips) to a solution containing:
                - `"pts_to_change"`: The points selected for flipping.
                - `"pts_to_change_sol"`: The corresponding flip directions.
                - `"time"`: Execution time for each flip count.

        """

        pts_per_region = df_signif["points"].tolist()

        methods_to_func_args = {
            "nregions": {
                "func": get_simple_heu_points,
                "args": (
                    n_flips_,
                    df_points,
                    "nregions",
                    y_pred,
                    pts_per_region,
                    signif_thresh_,
                ),
            },
            "nflips": {
                "func": get_simple_heu_points,
                "args": (
                    n_flips_,
                    df_points,
                    "rank_flips",
                    y_pred,
                    pts_per_region,
                    signif_thresh_,
                ),
            },
            "stat": {
                "func": get_simple_heu_points,
                "args": (
                    n_flips_,
                    df_points,
                    "rank_stat",
                    y_pred,
                    pts_per_region,
                    signif_thresh_,
                ),
            },
            "iter": {
                "func": get_iterative_points,
                "args": (
                    n_flips_,
                    df_points,
                    df_signif,
                    y_pred,
                    P,
                    N,
                    signif_thresh_,
                    report_progress,
                ),
            },
            "exhaust": {
                "func": get_exhaustive_points,
                "args": (
                    n_flips_,
                    df_points,
                    df_signif,
                    y_pred,
                    P,
                    N,
                    signif_thresh_,
                    report_progress,
                ),
            },
        }

        iter_range = list(range(n_flips_start, n_flips_, step))
        if iter_range[-1] != n_flips_:
            iter_range.append(n_flips_)

        func = methods_to_func_args[method]["func"]
        args = methods_to_func_args[method]["args"]

        pts_to_change, pts_to_change_sol, exec_times = func(*args)
        pts_to_change_list = []
        pts_to_change_sol_list = []
        exec_times_list = []

        exec_times = np.array(exec_times)

        pts_to_change_mapped = (
            [pts_sol_idx_map[x] for x in pts_to_change]
            if pts_sol_idx_map is not None
            else pts_to_change
        )
        budget_to_solution_info = {}
        for i in iter_range:
            cur_sol = np.zeros(len(y_pred))
            cur_pts_to_change = pts_to_change[:i]
            cur_pts_to_change_mapped = pts_to_change_mapped[:i]
            cur_pts_to_change_sol = pts_to_change_sol[:i]
            cur_sol[cur_pts_to_change] = cur_pts_to_change_sol
            exec_times_list.append(exec_times[i - 1])
            pts_to_change_list.append(cur_pts_to_change_mapped)
            pts_to_change_sol_list.append(cur_pts_to_change_sol)

            budget_to_solution_info[i] = {
                "pts_to_change": cur_pts_to_change_mapped,
                "pts_to_change_sol": cur_pts_to_change_sol,
                "time": exec_times[i - 1],
            }

        return budget_to_solution_info

    def save_model(self, file_path):
        """
        Saves the model to a file using pickle serialization.

        Args:
            file_path (str): The path where the model should be saved.

        Raises:
            ValueError: If the model has not been fitted.
        """

        if self.fitted != 2:
            raise ValueError("Model must be fitted before saving.")

        dir_name = os.path.dirname(file_path)

        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        model_data = {
            "budget_to_solution_info": self.budget_to_solution_info,
            "method": self.method,
            "overlap": self.overlap,
            "optimization": self.optimization,
        }

        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)
