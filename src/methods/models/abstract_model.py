from abc import ABC, abstractmethod
import numpy as np


class SpatialFairnessModel(ABC):
    """
    Abstract base class for spatial fairness models.

    This class provides a framework for implementing models that ensure spatial fairness
    in classification tasks by modifying predictions based on fairness constraints.

    Attributes:
        method (str): Name of the fairness method.
        fitted (int): Indicator for whether the model has been fitted (0: not fitted, 1: single-fitted, 2: multi-fitted).
        init_threshold (float): Initial threshold value for decision-making.
        budget_to_solution_info (dict): Mapping of budget levels to solution details.
        pts_to_change (list): Indices of points that need label changes (for single-fitted).
        pts_to_change_sol (list): Solution values for label changes (for single-fitted).
        thresholds (list): Threshold values for decision-making (for single-fitted).
        eq_to_thresh_flip_probs (dict): Probabilities for flipping labels at specific thresholds (for single-fitted).
        regions_dp (dict): Data structure containing region details (for single-fitted).
        st_par (bool): True: Statistical parity, False: Equal Opportunity
        overlap (bool): Whether the regions are overlapping.
        val_n_s (list): List of validation set sizes per region.
        regs_id_2_reg_indices (dict): Mapping of region group IDs to region indices.
        reg_indices_2_regs_id (dict): Mapping of region indices to region group IDs.
        reg_id_2_n (dict): Mapping of region IDs to number of points.

    Methods:
        multi_fit: Abstract method for fitting the model based on input data.
        multi_predict: Generates spatial fairness predictions for different budget levels.
        save_model: Abstract method for saving the fitted model.
        _get_pts_region_indices: Maps points to their respective region indices.
        _get_non_empty_regions_info: Filters out empty regions from a given set of regions.
        _get_pos_info_regions: Extracts positive-label points from given regions.
    """

    def __init__(self, method):
        """
        Initializes the spatial fairness model with a given fairness method.

        Args:
            method (str): The method to be used for correcting the spatial bias.
        """

        self.method = method
        self.fitted = 0
        self.init_threshold = None
        self.budget_to_solution_info = None
        self.pts_to_change = None
        self.pts_to_change_sol = None
        self.thresholds = None
        self.eq_to_thresh_flip_probs = None
        self.regions_dp = None
        self.st_par = None
        self.overlap = None
        self.val_n_s = None
        self.regs_id_2_reg_indices = None
        self.reg_indices_2_regs_id = None
        self.reg_id_2_n = None

        self.optim_methods = [
            "promis_opt",
            "promis_app",
        ]

        self.spatial_flip_methods = [
            "nregions",
            "nflips",
            "stat",
            "iter",
            "exhaust",
        ]

        if method in self.optim_methods:
            self.optimization = True
        else:
            self.optimization = False

    @abstractmethod
    def save_model(self, path):
        """
        Saves the fitted model information to a specified file path.

        Args:
            path (str): The file path to save model information.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        pass

    @abstractmethod
    def multi_fit(
        self,
        points_per_region,
        n_flips_start,
        step,
        n_flips,
        y_pred,
        overlap=False,
        verbose=0,
        **kwargs,
    ):
        """
        Fits the model based on the input data.

        Args:
            points_per_region (list): A list of lists containing indices of points per region.
            n_flips_start (int): Starting number of label flips.
            step (int): Step size for increasing flips.
            n_flips (int): Total number of flips to consider.
            y_pred (np.ndarray): Predicted labels.
            y_pred_probs (np.ndarray): Probability predictions.
            y_true (np.ndarray, optional): Ground truth labels. Defaults to None.
            fair_notion (str, optional): Fairness notion to be applied ("statistical_parity"/"equal_opportunity"). Defaults to "statistical_parity".
            overlap (bool, optional): Whether regions have overlapping points. Defaults to False.
            init_threshold (float, optional): Initial threshold for classification. Defaults to None.
            verbose (int, optional): Verbosity level for logging. Defaults to 0.
            **kwargs: Additional parameters.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        pass

    @abstractmethod
    def multi_predict(self, budgets=[], **kwargs):
        """
        Generates spatial fairness predictions for different budget levels.

        Args:
            budgets (list, optional): List of budget levels. Defaults to an empty list.
            **kwargs: Additional parameters.

        Returns:
            dict: A dictionary mapping budgets to predicted labels.

        Raises:
            ValueError: If the model has not been multi-fitted before calling this method.
        """

    def _get_pts_region_indices(self, points_per_region):
        """
        Creates a mapping from points to their corresponding region indices.

        Args:
            points_per_region (list): A list of lists, where each sublist contains indices of points in a region.

        Returns:
            dict: Mapping from points to region indices.
        """

        pts_2_region_indices = {}
        for region_idx, pts in enumerate(points_per_region):
            for point in pts:
                if point not in pts_2_region_indices:
                    pts_2_region_indices[point] = []
                pts_2_region_indices[point].append(region_idx)

        return pts_2_region_indices

    def _get_non_empty_regions_info(self, points_per_region):
        """
        Extracts non-empty regions from a given set of regions.

        Args:
            points_per_region (list): A list of lists, where each sublist contains indices of points in a region.

        Returns:
            tuple: A tuple containing:
                - non_empty_pts_per_region (list): List of non-empty regions.
                - non_empty_idx_to_orig_idx (dict): Mapping of non-empty region indices to original indices.
        """

        non_empty_pts_per_region = []
        non_empty_idx_to_orig_idx = {}
        for i, pts in enumerate(points_per_region):
            if len(pts) > 0:
                non_empty_pts_per_region.append(pts)
                non_empty_idx_to_orig_idx[len(non_empty_pts_per_region) - 1] = i

        return non_empty_pts_per_region, non_empty_idx_to_orig_idx

    def _get_pos_info_regions(self, y_true, points_per_region):
        """
        Extracts positive-label points from given regions.

        Args:
            y_true (np.ndarray): Ground truth labels.
            points_per_region (list): List of lists, where each sublist contains indices of points in a region.

        Returns:
            tuple: A tuple containing:
                - pos_y_true_indices (np.ndarray): Indices of points with positive labels.
                - pos_points_per_region (list): Lists of positive-label points per region.
        """

        pos_y_true_indices = np.where(np.array(y_true) == 1)[0]
        point_2_new_pos_set_idx = {
            point: i for i, point in enumerate(pos_y_true_indices)
        }

        pos_points_per_region = [[] for _ in range(len(points_per_region))]
        for region_idx, pts in enumerate(points_per_region):
            for point in pts:
                if point in pos_y_true_indices:
                    pos_points_per_region[region_idx].append(
                        point_2_new_pos_set_idx[point]
                    )

        return pos_y_true_indices, pos_points_per_region
