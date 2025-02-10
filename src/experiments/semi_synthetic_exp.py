# this script is used to run the experiments on the semi-synthetic crime dataset
# You may opt to run on specific audit regions by commenting/uncommenting the
# the desired lines in the partioning_type_names list.
# You can also opt to run on specific PROMIS method by commenting/uncommenting the
# the desired lines in the promis_methods list.
"""
regions_non_overlap_k_8_crime|pred_semi_synthetic_regions_non_overlap_k_8_crime
For each experiment the directory crime_semi_synth_exp/regions_\<partitioning_name>|pred_semi_synthetic_regions_<partitioning_name>_crime is created.

Inside the directory the following files are created:

* spatial_optim/statistical_parity_<promis_method_name>.pkl (for promis_opt method the wlimit_\<work_limit> is appended)
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join("..")))

from methods.models.optimization_model import SpatialOptimFairnessModel
from methods.models.spatial_flip_model import SpatialFlipFairnessModel
from utils.data_utils import read_scanned_regs, get_y
from utils.results_names_utils import combine_world_info, get_train_val_test_paths
import pandas as pd


base_path = "../../data/"
results_base_path = "../../results/crime_semi_synth_exp/"
n_flips_start = 7000
step = 7000
no_of_threads = 0
wlimit = 300  # working limit
clf_name = "semi_synthetic"
dataset_name = "crime"
partioning_type_names = [
    ("non_overlap_k_8", False),
    ("5_x_5", True),
    ("overlap_k_10_radii_4", True),
]  # [(partioning_name, is overlapping partitioning?)]
fairness_notion = "statistical_parity"

promis_methods = [
    "promis_app",
    "promis_opt",
]

max_pr_shift = 0.1
results = {}
for partioning_type_name, overlap in partioning_type_names:
    res_desc_label, partioning_name, prediction_name = combine_world_info(
        dataset_name, partioning_type_name, f"{clf_name}_regions_{partioning_type_name}"
    )
    _, _, test_path_info = get_train_val_test_paths(
        base_path, partioning_name, prediction_name, dataset_name
    )

    test_regions_df = read_scanned_regs(test_path_info["regions"])
    test_pred_df = pd.read_csv(test_path_info["predictions"])
    y_pred_test = get_y(test_pred_df, "pred")

    test_points_per_region = test_regions_df["points"].tolist()

    with open(
        f"{base_path}predictions/test_pred_{clf_name}_{partioning_name}_budget.txt",
        "r",
    ) as file:
        n_flips = int(file.read())

    results_path = f"{results_base_path}{res_desc_label}/"

    print(f"{clf_name}, {partioning_type_name}, {fairness_notion}")

    for method in promis_methods:
        fair_model = SpatialOptimFairnessModel(method)
        fair_model.multi_fit(
            points_per_region=test_points_per_region,
            n_flips_start=n_flips_start,
            step=step,
            n_flips=n_flips,
            y_pred=y_pred_test,
            wlimit=wlimit,
            fair_notion=fairness_notion,
            overlap=overlap,
            no_of_threads=no_of_threads,
            verbose=1,
            max_pr_shift=max_pr_shift,
        )

        if method == "promis_opt":
            model_save_file = f"{results_path}spatial_optim_models/{fairness_notion}/{method}_wlimit_{wlimit}.pkl"
        else:
            model_save_file = (
                f"{results_path}spatial_optim_models/{fairness_notion}/{method}.pkl"
            )

        fair_model.save_model(model_save_file)
