import os
import sys

sys.path.append(os.path.abspath(os.path.join("..")))

from methods.models.abstract_model import SpatialFairnessModel
import numpy as np
from utils.thresholds_utils import (
    adjusted_thresholds,
    convert_indiv_sol_2_regions_sol,
)
from methods.optimization.promis_opt import (
    minimize_promis_opt_obj,
    minimize_promis_opt_obj_overlap,
)
from methods.optimization.promis_app import (
    minimize_promis_app_obj,
    minimize_promis_app_obj_overlap,
)
from time import time
import pandas as pd
from tqdm import tqdm
import pickle
import random


class SpatialOptimFairnessModel(SpatialFairnessModel):
    """
    This class implements an optimization-based approach to spatial fairness,
    extending the `SpatialFairnessModel` class
    """

    def __init__(self, method):
        super().__init__(method)

        assert (
            method in self.optim_methods
        ), f"Invalid Method, should be one of: {self.optim_methods}"
        self.optim_method_func_dict = {
            ("promis_opt", True): minimize_promis_opt_obj_overlap,
            ("promis_opt", False): minimize_promis_opt_obj,
            (
                "promis_app",
                True,
            ): minimize_promis_app_obj_overlap,
            ("promis_app", False): minimize_promis_app_obj,
        }
        self.status = None
        self.obj_val = None
        self.regions_weights = None

    def _load(self, file_path):
        """
        Load a pretrained model from a pickle (.pkl) file.

        Args:
            file_path (str): Path to the model file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

        with open(file_path, "rb") as f:
            model_data = pickle.load(f)

        self.budget_to_solution_info = model_data["budget_to_solution_info"]
        self.init_threshold = model_data["init_threshold"]
        self.method = model_data["method"]
        self.st_par = model_data["st_par"]
        self.overlap = model_data["overlap"]
        self.val_n_s = model_data["val_n_s"]
        self.regs_id_2_reg_indices = model_data["regs_id_2_reg_indices"]
        self.reg_indices_2_regs_id = model_data["reg_indices_2_regs_id"]
        self.reg_id_2_n = model_data["reg_id_2_n"]
        self.regions_weights = model_data["regions_weights"]
        self.fitted = 2
        self.optimization = model_data["optimization"]

        self.optim_method_func_dict = {
            ("promis_opt", True): minimize_promis_opt_obj_overlap,
            ("promis_opt", False): minimize_promis_opt_obj,
            (
                "promis_app",
                True,
            ): minimize_promis_app_obj_overlap,
            ("promis_app", False): minimize_promis_app_obj,
        }

    @classmethod
    def from_pretrained(cls, file_path):
        """
        Load a pretrained model and return an instance of `SpatialOptimFairnessModel`.

        Args:
            file_path (str): Path to the saved model file.

        Returns:
            SpatialOptimFairnessModel: An instance of the class with loaded parameters.
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
        y_pred_probs=None,
        y_true=None,
        wlimit=None,
        fair_notion="statistical_parity",
        overlap=False,
        init_threshold=None,
        no_of_threads=0,
        verbose=0,
        append_to_sol=False,
        non_linear=True,
        max_pr_shift=None,
    ):
        """
        Perform multiple iterations of the optimization process, storing results for different numbers of flips.

        Args:
            points_per_region (list): List of point groupings defining regions.
            n_flips_start (int): The starting number of label flips.
            step (int): Step size for label flips.
            n_flips (int): Maximum number of label flips.
            y_pred (np.ndarray): Predicted labels.
            y_pred_probs (np.ndarray, optional): Probability predictions. Probabilities are required for Equal Opportunity.
            y_true (np.ndarray, optional): True labels, required for Equal Opportunity.
            wlimit (float, optional): Work limit for optimization.
            fair_notion (str): Either "statistical_parity" or "equal_opportunity".
            overlap (bool): Whether partitioning is overlapping.
            init_threshold (float, optional): Initial threshold for prediction.
            no_of_threads (int, optional): Number of parallel threads to use.
            verbose (int, optional): Verbosity level.
            append_to_sol (bool, optional): Whether to append results to existing solutions.
            non_linear (bool, optional): Whether to allow non-linear optimization.
            max_pr_shift (float, optional): Maximum initial ratio shift: for Equal Opportunity refers to True Positive Ration, for statistical parity refers to Positive Ratio.
        """

        if append_to_sol and self.fitted != 2:
            raise ValueError(
                "The model should have been multi fitted for append_to_sol=True"
            )

        if fair_notion == "equal_opportunity" and (
            y_true is None or y_pred_probs is None
        ):
            raise ValueError(
                "For Equal Opportunity the <y_true> and <y_pred_probs> should be provided"
            )

        if not append_to_sol:
            self.budget_to_solution_info = {}
        n_flips_range = list(range(n_flips_start, n_flips, step))
        if len(n_flips_range) == 0 or n_flips_range[-1] != n_flips:
            n_flips_range.append(n_flips)

        n_flips_range = (
            tqdm(
                n_flips_range,
                desc=f"Flipping with {self.method} Optimization Method",
                leave=False,
            )
            if verbose in [1, 2]
            else n_flips_range
        )
        print(n_flips_range)
        for n_flips in n_flips_range:
            s_time = time()
            self.fit(
                points_per_region=points_per_region,
                n_flips=n_flips,
                y_pred=y_pred,
                y_pred_probs=y_pred_probs,
                y_true=y_true,
                wlimit=wlimit,
                fair_notion=fair_notion,
                overlap=overlap,
                init_threshold=init_threshold,
                no_of_threads=no_of_threads,
                verbose=verbose,
                non_linear=non_linear,
                max_pr_shift=max_pr_shift,
            )
            fit_time = time() - s_time
            self.budget_to_solution_info[n_flips] = {
                "pts_to_change": self.pts_to_change,
                "pts_to_change_sol": self.pts_to_change_sol,
                "new_regions_thresh": self.thresholds,
                "eq_to_thresh_flip_probs": self.eq_to_thresh_flip_probs,
                "regions_dp": self.regions_dp,
                "status": self.status,
                "obj_val": self.obj_val,
                "time": fit_time,
                "wlimit": wlimit,
            }
        self.fitted = 2

    def fit(
        self,
        points_per_region,
        n_flips,
        y_pred,
        y_pred_probs=None,
        y_true=None,
        wlimit=None,
        fair_notion="statistical_parity",
        overlap=False,
        init_threshold=None,
        non_linear=True,
        no_of_threads=0,
        verbose=0,
        max_pr_shift=None,
    ):
        """
        Fit the model based on optimization techniques.

        Args:
            points_per_region (list): List of points per region.
            n_flips (int): Number of flips allowed.
            y_pred (np.array): Predicted labels.
            y_pred_probs (np.ndarray, optional): Probability predictions. Probabilities are required for Equal Opportunity.
            y_true (np.array, optional): True labels, required for equal opportunity fairness notion.
            wlimit (int, optional): Work limit for the optimization solver.
            fair_notion (str, optional): Type of fairness constraint ('statistical_parity' or 'equal_opportunity').
            overlap (bool, optional): Whether partitioning is overlapping.
            init_threshold (float, optional): Initial threshold for decision-making.
            non_linear (bool, optional): Whether to use a non-linear optimization approach (useful only for "promis_opt" method).
            no_of_threads (int, optional): Number of threads for parallel computation.
            verbose (int, optional): Verbosity level.
            max_pr_shift (float, optional): Maximum initial ratio shift: for Equal Opportunity refers to True Positive Ration, for statistical parity refers to Positive Ratio.

        Raises:
            ValueError: If y_true is not provided for 'equal_opportunity' fairness.
        """

        assert fair_notion in ["statistical_parity", "equal_opportunity"]

        y_pred = np.array(y_pred)
        if y_pred_probs is not None:
            y_pred_probs = np.array(y_pred_probs)

        if fair_notion == "equal_opportunity" and (
            y_true is None or y_pred_probs is None
        ):
            raise ValueError(
                "For Equal Opportunity the <y_true> and <y_pred_probs> should be provided"
            )

        n_regions = len(points_per_region)
        self.init_threshold = 0.5 if init_threshold is None else init_threshold
        self.thresholds = np.zeros(n_regions) + self.init_threshold
        self.eq_to_thresh_flip_probs = np.zeros(n_regions) - 1
        self.regions_dp = np.zeros(n_regions)

        y_pred_reset_pos_idx_2_y_pred_idx = None
        self.st_par = True
        if fair_notion == "equal_opportunity":
            self.st_par = False

            pos_y_true_indices, points_per_region = self._get_pos_info_regions(
                y_true, points_per_region
            )

            y_pred = y_pred[pos_y_true_indices]
            y_pred_probs = y_pred_probs[pos_y_true_indices]
            y_pred_reset_pos_idx_2_y_pred_idx = {
                i: pos_y_true_indices[i] for i in range(len(y_pred))
            }

        self.overlap = False
        if overlap:
            self.overlap = True
            self.val_n_s = [len(pts) for pts in points_per_region]

        non_empty_pts_per_region, non_empty_reg_idx_to_orig_idx = (
            self._get_non_empty_regions_info(points_per_region)
        )

        min_pr = None
        max_pr = None

        if max_pr_shift is not None:
            init_pr = sum(y_pred) / len(y_pred)
            min_pr = max(0.0, (1 - max_pr_shift) * init_pr)
            max_pr = min(1.0, (1 + max_pr_shift) * init_pr)

        self.regions_weights = self._get_weights(y_pred, non_empty_pts_per_region)
        (
            status,
            obj_val,
            pts_to_change,
            pts_to_change_sol,
            new_thresholds,
            eq_to_thresh_flip_probs,
            regions_dp,
        ) = self._get_optimization_solutions(
            optim_method_func=self.optim_method_func_dict[(self.method, overlap)],
            optim_method_label=self.method,
            n_flips=n_flips,
            points_per_region=non_empty_pts_per_region,
            y_pred=y_pred,
            y_pred_probs=y_pred_probs,
            init_threshold=init_threshold,
            signif_pts_idxs=[],
            wlimit=wlimit,
            no_of_threads=no_of_threads,
            show_msg=(verbose == 2),
            non_convex_param=0 if self.method != "promis_opt" else -1,
            weights=self.regions_weights,
            overlap=overlap,
            min_pr=min_pr,
            max_pr=max_pr,
            non_linear=non_linear,
            cont_sol=True,
            pts_sol_idx_map=y_pred_reset_pos_idx_2_y_pred_idx,
        )

        if status in [1, 3]:
            self.fitted = 1
            self.obj_val = obj_val
            self.status = status
            self.pts_to_change = pts_to_change
            self.pts_to_change_sol = pts_to_change_sol
            for reg_idx in range(len(non_empty_pts_per_region)):
                orig_idx = non_empty_reg_idx_to_orig_idx[reg_idx]
                reg_dp = regions_dp[reg_idx]
                self.regions_dp[orig_idx] = reg_dp

                if y_pred_probs is not None:
                    thresh = new_thresholds[reg_idx]
                    prob = eq_to_thresh_flip_probs[reg_idx]
                    self.thresholds[orig_idx] = thresh
                    self.eq_to_thresh_flip_probs[orig_idx] = prob

        else:
            raise ValueError(f"Fit failed with error code {status}")

    def multi_predict(
        self, points_per_region, y_pred, budgets=[], apply_fit_flips=False
    ):
        """
        Generates spatial fairness predictions for different budget levels.

        Args:
            points_per_region (list): List of points per region.
            y_pred (np.ndarray): Predictions probabilities if apply_fit_flips==False, predictions otherwise.
            budgets (list, optional): List of budget levels. Defaults to an empty list.
            apply_fit_flips (bool, optional): Whether to apply flips from the fit or use new thresholds for inference. Defaults to False.

        Returns:
            dict: A dictionary mapping budgets to predicted labels.

        Raises:
            ValueError: If the model has not been multi-fitted before calling this method.
        """

        if self.fitted != 2:
            raise ValueError("Model must be multi fitted before multi prediction.")

        budgets = budgets if budgets else self.budget_to_solution_info.keys()
        budget_to_predictions = {}

        if apply_fit_flips:
            for budget in budgets:
                solution_info = self.budget_to_solution_info[budget]
                y_pred_new = y_pred.copy()
                y_pred_new[solution_info["pts_to_change"]] = (
                    y_pred_new[solution_info["pts_to_change"]]
                    + solution_info["pts_to_change_sol"]
                )
                budget_to_predictions[budget] = y_pred_new

        else:
            pts_2_region_indices = self._get_pts_region_indices(points_per_region)

            if not self.overlap:
                for budget in budgets:
                    solution_info = self.budget_to_solution_info[budget]
                    budget_to_predictions[budget] = np.array(
                        self._get_non_over_predictions(
                            y_pred,
                            pts_2_region_indices,
                            solution_info["new_regions_thresh"],
                            solution_info["eq_to_thresh_flip_probs"],
                        )
                    )
            else:
                comb_n_s = self._combined_regions(points_per_region)

                for budget in budgets:
                    solution_info = self.budget_to_solution_info[budget]
                    budget_to_predictions[budget] = np.array(
                        self._get_over_predictions(
                            y_pred,
                            solution_info["new_regions_thresh"],
                            solution_info["eq_to_thresh_flip_probs"],
                            pts_2_region_indices,
                            comb_n_s,
                        )
                    )

        return budget_to_predictions

    def _get_over_predictions(
        self, probs, thresholds, eq_to_thresh_flip_probs, pts_2_region_indices, comb_n_s
    ):
        """
        Computes predictions for overlapping regions using weighted averaging.

        Args:
            probs (np.ndarray): Probability predictions for each point.
            thresholds (np.ndarray): Threshold values for classification.
            eq_to_thresh_flip_probs (np.ndarray): Probabilities of flipping labels at specific thresholds.
            pts_2_region_indices (dict): Mapping from points to their respective region indices.
            comb_n_s (list): Combined number of samples for each region.

        Returns:
            list: List of final predictions for each point.
        """

        cover_ratio_weighted_applied_preds = []

        for point, pred_prob in enumerate(probs):
            point_reg_indices = pts_2_region_indices[point]
            regs_thesh = thresholds[point_reg_indices]
            regs_flip_probs = eq_to_thresh_flip_probs[point_reg_indices]

            regs_preds = [
                self._get_prob_thresh_pred(thresh, pred_prob, flip_prob)
                for thresh, flip_prob in zip(regs_thesh, regs_flip_probs)
            ]
            if len(point_reg_indices) == 1:
                cover_ratio_weighted_applied_preds.append(regs_preds[0])
            else:
                reg_ratios = [1 / comb_n_s[reg_idx] for reg_idx in point_reg_indices]
                cover_ratio_weighted_applied_preds.append(
                    self._multi_predictions_to_single(regs_preds, reg_ratios)
                )

        return cover_ratio_weighted_applied_preds

    def _combined_regions(self, points_per_region):
        """
        Combines the populations used in fit and the given for inference for each region.

        Args:
            points_per_region (list): List of points per region.

        Returns:
            list: Combined number of samples for each region.
        """
        comb_n_s = [
            self.val_n_s[i] + len(points_per_region[i])
            for i in range(len(points_per_region))
        ]

        return comb_n_s

    def _get_prob_thresh_pred(self, thresh, pred_prob, eq_to_thresh_flip_prob):
        """
        Determines the predicted label based on probability and threshold.

        Args:
            thresh (float): The decision threshold.
            pred_prob (float): The probability of the positive class.
            eq_to_thresh_flip_prob (float): Probability of flipping labels if the prediction is equal to the threshold.

        Returns:
            int: The predicted label (0 or 1).
        """

        if eq_to_thresh_flip_prob != -1 and pred_prob == thresh:
            return int(random.random() < eq_to_thresh_flip_prob)
        return int(pred_prob > thresh)

    def _multi_predictions_to_single(self, predictions, weights):
        """
        Aggregates multiple predictions into a single prediction using weighted sum.

        Args:
            predictions (list): List of binary predictions (0 or 1).
            weights (list): List of corresponding weights for each prediction.

        Returns:
            bool: The final aggregated prediction.
        """

        return (
            sum(
                weight if pred else -weight
                for pred, weight in zip(predictions, weights)
            )
            > 0
        )

    def _get_non_over_predictions(
        self, probs, pts_2_region_indices, thresholds, eq_to_thresh_flip_probs
    ):
        """
        Computes predictions for non-overlapping regions.

        Args:
            probs (np.ndarray): Probability predictions for each point.
            pts_2_region_indices (dict): Mapping from points to their region indices.
            thresholds (np.ndarray): Threshold values for classification.
            eq_to_thresh_flip_probs (np.ndarray): Probabilities of flipping labels at specific thresholds.

        Returns:
            list: List of final predictions for each point.
        """

        return [
            self._get_prob_thresh_pred(
                thresholds[pts_2_region_indices[point]],
                pred_prob,
                eq_to_thresh_flip_probs[pts_2_region_indices[point]],
            )
            for point, pred_prob in enumerate(probs)
        ]

    def _get_weights(self, y_pred, points_per_region):
        """
        Compute region weights.

        Args:
            y_pred (numpy array): The predicted labels for the dataset.
            points_per_region (list of lists): A list where each element is a list of point indices belonging to a region.

        Returns:
            list: A list of computed weights for each region, or None if the method does not require weighting.
        """

        N = len(y_pred)
        n_s = [len(reg_pts) for reg_pts in points_per_region]

        if self.method == "promis_app":
            return [
                np.sqrt((n_s[i] * (N - n_s[i])) / N)
                for i in range(len(points_per_region))
            ]
        else:
            return None

    def _convert_region_sol_2_int_indiv_sol(self, dp_s, points_per_region, y_pred):
        """
        Converts a region-based solution into an individual point-based solution.

        Args:
            dp_s (list or numpy array): The solution values for each region (number of label flips).
            points_per_region (list of lists): A list where each element is a list of point indices belonging to a region.
            y_pred (numpy array): The predicted labels for the dataset.

        Returns:
            numpy array: A solution array where each index corresponds to a point and its value represents
                        the flip direction (-1 for label 1 to 0, 1 for label 0 to 1, 0 for no change).

        This method ensures that the specified number of flips is applied only to points that currently
        have the original label.
        """

        sol = np.zeros(len(y_pred))

        for idx, dp in enumerate(dp_s):
            int_dp = int(round(dp))

            # if the region has no flips then continue
            if int_dp != 0:
                reg_pts = np.array(points_per_region[idx])

                if int_dp < 0:
                    flip_dir = -1
                    old_label_to_flip = 1
                else:
                    flip_dir = 1
                    old_label_to_flip = 0

                # find the points in the region that have the old label
                all_pts_to_change_reg_pts_indices = np.where(
                    y_pred[reg_pts] == old_label_to_flip
                )
                all_pts_to_change_indices = all_pts_to_change_reg_pts_indices[0]

                # get the points to change indices
                pts_to_change_indices = all_pts_to_change_indices[: np.abs(int_dp)]

                # get the points to change
                pts_to_change = reg_pts[pts_to_change_indices]

                assert len(pts_to_change) == np.abs(
                    int(int_dp)
                ), "Points to change does not match the number of flips"

                # change the labels of the points to the new label
                sol[pts_to_change] = flip_dir

        return sol

    def _get_optimization_solutions(
        self,
        optim_method_func,
        optim_method_label,
        n_flips,
        points_per_region,
        y_pred,
        y_pred_probs=None,
        init_threshold=None,
        signif_pts_idxs=[],
        wlimit=None,
        no_of_threads=0,
        show_msg=False,
        non_convex_param=0,
        weights=None,
        overlap=False,
        min_pr=None,
        max_pr=None,
        non_linear=True,
        cont_sol=True,
        pts_sol_idx_map=None,
    ):
        """
        Runs an optimization method to compute the best solution for flipping predictions in a spatial fairness context.

        Args:
            optim_method_func (function): The optimization function to use.
            optim_method_label (str): The label identifying the optimization method.
            n_flips (int): The number of label flips allowed.
            points_per_region (list of lists): A list where each sublist contains indices of points in a specific region.
            y_pred (numpy array): The predicted labels for the dataset.
            y_pred_probs (numpy array): The prediction probabilities for the dataset.
            init_threshold (float, optional): The initial threshold for region-based decision-making. Defaults to None.
            signif_pts_idxs (list, optional): Indices of significant points to consider for fairness constraints. Defaults to [].
            wlimit (int, optional): Work limit for the optimization solver. Defaults to None.
            no_of_threads (int, optional): Number of threads for optimization. Defaults to 0.
            show_msg (bool, optional): Whether to display solver messages. Defaults to False.
            non_convex_param (int, optional): Parameter controlling non-convex optimization settings. Defaults to 0.
            weights (list, optional): Weights assigned to each region in the optimization. Defaults to None.
            overlap (bool, optional): Whether regions overlap. Defaults to False.
            min_pr (float, optional): Minimum ratio constraint. Defaults to None.
            max_pr (float, optional): Maximum ratio constraint. Defaults to None.
            non_linear (bool, optional): Whether to use a non-linear optimization method. Defaults to True.
            cont_sol (bool, optional): Whether to allow continuous solutions. Defaults to True.
            pts_sol_idx_map (dict, optional): A mapping of points to new indices for fairness evaluation. Defaults to None.

        Returns:
            tuple:
                - status (int): Status of the optimization solver.
                - obj_val (float or None): Objective value of the optimization solution, if available.
                - pts_to_change (list or None): Indices of points where label flips should be applied.
                - pts_to_change_sol (list or None): Flip directions for the corresponding points in `pts_to_change`.
                - new_thresholds (list or None): Updated decision thresholds for each region.
                - eq_to_thresh_flip_probs (list or None): Probability thresholds for label flips.
                - regions_dp (list or None): The optimized decision policies for regions.

        This function applies an optimization method to determine the best way to adjust the predictions in a spatial fairness
        setting, given a constraint on the number of label flips. The method constructs the necessary arguments based on whether
        the optimization is applied to non-overlapping or overlapping regions and then executes the specified optimization function.

        The function also converts region-based solutions to individual point-based solutions where necessary and adjusts
        decision thresholds accordingly.
        """

        init_thresholds = (
            [0.5] * len(points_per_region)
            if init_threshold is None
            else [init_threshold] * len(points_per_region)
        )

        if overlap:
            args = {
                "labels": y_pred,
                "points_per_region": points_per_region,
                "signif_pts_idxs": signif_pts_idxs,
                "show_msg": show_msg,
                "wlimit": wlimit,
                "no_of_threads": no_of_threads,
                "non_convex_param": non_convex_param,
                "min_pr": min_pr,
                "max_pr": max_pr,
                "cont_sol": cont_sol,
            }
        else:
            n_s = [len(reg_pts) for reg_pts in points_per_region]
            p_s = [np.sum(y_pred[reg_pts]) for reg_pts in points_per_region]

            args = {
                "n_s": n_s,
                "p_s": p_s,
                "show_msg": show_msg,
                "wlimit": wlimit,
                "no_of_threads": no_of_threads,
                "non_convex_param": non_convex_param,
                "min_pr": min_pr,
                "max_pr": max_pr,
                "cont_sol": cont_sol,
            }

        if weights is not None:
            args["weights"] = weights

        if optim_method_label.startswith("promis_opt"):
            args["non_linear"] = non_linear

        sol, status, obj_val = optim_method_func(**args, C=n_flips)
        new_thresholds = None
        eq_to_thresh_flip_probs = None
        # if optimal sol (1) or solver reached limit (3)
        if status in [1, 3]:

            if not overlap:
                regions_dp = sol.tolist()
                if y_pred_probs is not None:
                    new_thresholds, eq_to_thresh_flip_probs = adjusted_thresholds(
                        sol, init_thresholds, points_per_region, y_pred_probs
                    )
                # convert region solution to individual points solution
                sol = self._convert_region_sol_2_int_indiv_sol(
                    sol, points_per_region, y_pred
                )
            else:
                regions_dp = convert_indiv_sol_2_regions_sol(sol, points_per_region)
                if y_pred_probs is not None:
                    new_thresholds, eq_to_thresh_flip_probs = adjusted_thresholds(
                        regions_dp, init_thresholds, points_per_region, y_pred_probs
                    )
                regions_dp = regions_dp.tolist()
                sol = np.round(sol).astype(int)

            pts_to_change = np.where(sol != 0)[0]
            pts_to_change_sol = sol[pts_to_change]

            pts_to_change = pts_to_change.tolist()
            if pts_sol_idx_map is not None:
                pts_to_change = [pts_sol_idx_map[x] for x in pts_to_change]

            return (
                status,
                obj_val,
                pts_to_change,
                pts_to_change_sol,
                new_thresholds,
                eq_to_thresh_flip_probs,
                regions_dp,
            )

        return status, None, None, None, None, None, None

    def save_model(self, file_path):
        """
        Save the model's multi-fit information to a binary file using pickle.

        Args:
            file_path (str): The path where the model data should be saved.

        Raises:
            ValueError: If the model has not been fitted before attempting to save.

        This function serializes and saves the optimization results and model parameters,
        allowing future reuse or analysis of the fitted model.
        """
        if self.fitted != 2:
            raise ValueError("Model must be fitted before saving.")

        dir_name = os.path.dirname(file_path)

        # Only create directory if one is specified
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        model_data = {
            "budget_to_solution_info": self.budget_to_solution_info,
            "init_threshold": self.init_threshold,
            "method": self.method,
            "st_par": self.st_par,
            "overlap": self.overlap,
            "val_n_s": self.val_n_s,
            "regs_id_2_reg_indices": self.regs_id_2_reg_indices,
            "reg_indices_2_regs_id": self.reg_indices_2_regs_id,
            "reg_id_2_n": self.reg_id_2_n,
            "optimization": self.optimization,
            "regions_weights": self.regions_weights,
        }

        with open(file_path, "wb") as f:
            pickle.dump(model_data, f)

    def get_model_fit_info_as_df(self):
        """
        Return the model's multi-fit information as a Pandas DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame containing optimization results, including:
                - budget (int): The number of label flips allowed.
                - pts_to_change (list): Indices of points where label changes are applied.
                - pts_to_change_sol (list): The new labels assigned to the changed points.
                - new_regions_thresh (list): Updated decision thresholds for each region.
                - eq_to_thresh_flip_probs (list): Flip probability thresholds per region.
                - regions_dp (list): Region-based decision policies after optimization.
                - status (int): The status code returned by the optimizer.
                - obj_val (float): The objective value from the optimization process.
                - time (float): The execution time taken for the optimization.

        Raises:
            ValueError: If the model has not been fitted before attempting to retrieve fit information.

        This method compiles stored optimization results into a structured DataFrame
        for analysis, visualization, or comparison across different budgets.
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
                    "new_regions_thresh": info["new_regions_thresh"],
                    "eq_to_thresh_flip_probs": info["eq_to_thresh_flip_probs"],
                    "regions_dp": info["regions_dp"],
                    "status": info["status"],
                    "obj_val": info["obj_val"],
                    "time": info["time"],
                }
            )

        return pd.DataFrame(data)
