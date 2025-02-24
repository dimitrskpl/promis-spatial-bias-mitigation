# this script is used to run the experiments on crime dataset with equal opportunity fairness notion
# You can also opt to run on specific audit regions by commenting/uncommenting the
# desired lines in the partioning_type_names list.
# You can also opt to run on specific PROMIS method by commenting/uncommenting the
# desired lines in the promis_methods list.
"""
For each experiment the directory xgb_eq_opp_exp/regions_\<partitioning_name>|pred_xgb_crime is created.

Inside the directory the following files are created:
    * spatial_optim/equal_opportunity_<promis_method_name>.pkl (for promis_opt method the wlimit_\<work_limit> is appended)
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join("..")))

from methods.models.optimization_model import SpatialOptimFairnessModel
from utils.data_utils import read_scanned_regs, get_y
from utils.results_names_utils import combine_world_info, get_train_val_test_paths
import pandas as pd


base_path = "../../data/"
results_base_path = "../../results/xgb_eq_opp_exp/"
n_flips_start = 500
n_flips = 3500
step = 500
wlimit = 300
iter_flips_range = list(range(n_flips_start, n_flips + step, step))
no_of_threads = 0
clf_names = ["xgb"]
dataset_name = "crime"
partioning_type_names = [
    ("overlap_k_10_radii_4", True),  # Audit Regions = Scan Regions
    ("non_overlap_k_8", False),  # Audit Regions = Clusters
    ("5_x_5", True),  # Audit Regions = Grids
]  # [(partioning_name, is overlapping partitioning?)]
fairness_notion = "equal_opportunity"

promis_methods = [
    "promis_app",
    "promis_opt",
]

max_pr_shift = 0.1
results = {}
for clf_name in clf_names:
    for partioning_type_name, overlap in partioning_type_names:
        res_desc_label, partioning_name, prediction_name = combine_world_info(
            dataset_name, partioning_type_name, clf_name
        )
        _, val_path_info, test_path_info = get_train_val_test_paths(
            base_path, partioning_name, prediction_name, dataset_name
        )

        val_regions_df = read_scanned_regs(val_path_info["regions"])
        val_pred_df = pd.read_csv(val_path_info["predictions"])
        val_labels_df = pd.read_csv(val_path_info["labels"])
        y_pred_val = get_y(val_pred_df, "pred")
        y_pred_probs_val = get_y(val_pred_df, "prob")
        y_true_val = get_y(val_labels_df, "label")

        test_regions_df = read_scanned_regs(test_path_info["regions"])
        test_pred_df = pd.read_csv(test_path_info["predictions"])
        test_labels_df = pd.read_csv(test_path_info["labels"])
        y_pred_test = get_y(test_pred_df, "pred")
        y_pred_probs_test = get_y(test_pred_df, "prob")
        y_true_test = get_y(test_labels_df, "label")

        val_points_per_region = val_regions_df["points"].tolist()
        test_points_per_region = test_regions_df["points"].tolist()

        results_path = f"{results_base_path}{res_desc_label}/"

        print(f"{clf_name}, {partioning_type_name}, {fairness_notion}")

        for optim_method in promis_methods:
            fair_model = SpatialOptimFairnessModel(optim_method)
            fair_model.multi_fit(
                points_per_region=val_points_per_region,
                n_flips_start=n_flips_start,
                step=step,
                n_flips=n_flips,
                y_pred=y_pred_val,
                y_true=y_true_val,
                y_pred_probs=y_pred_probs_val,
                wlimit=wlimit,
                fair_notion=fairness_notion,
                overlap=overlap,
                init_threshold=None,
                no_of_threads=no_of_threads,
                verbose=1,
                max_pr_shift=max_pr_shift,
            )

            if optim_method == "promis_opt":
                model_save_file = f"{results_path}spatial_optim_models/{fairness_notion}/{optim_method}_wlimit_{wlimit}.pkl"
            else:
                model_save_file = f"{results_path}spatial_optim_models/{fairness_notion}/{optim_method}.pkl"

            fair_model.save_model(model_save_file)
