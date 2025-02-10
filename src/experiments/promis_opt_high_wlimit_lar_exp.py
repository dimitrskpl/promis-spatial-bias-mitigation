# this script is used to run the experiments on the LAR dataset only
# using the PROMIS Opt optimization method with a high working limit of 1800
# You can also opt to run on specific audit regions by commenting/uncommenting the
# the desired lines in the partioning_type_names list.
# You can also opt to run on specific PROMIS method by commenting/uncommenting the
# the desired lines in the promis_methods list.

"""
For this experiment the directory lar_exp/regions_\<partitioning_name>|pred__crime is created.

Inside the directory the following files are created:
* spatial_optim/statistical_parity/promis_opt_wlimit_<work_limit>.pkl
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
results_base_path = "../../results/lar_exp/"
n_flips_start = 3000
n_flips = 15000
step = 3000
iter_flips_range = list(range(n_flips_start, n_flips + step, step))
no_of_threads = 0
wlimit = 1800  # working limit
clf_name = ""
dataset_name = "lar"
partioning_type_names = [
    ("non_overlap_k_100", False),
    # ("overlap_k_100_radii_30", True),
    # ("5_x_5", True),
]  # [(partioning_name, is overlapping partitioning?)]
fairness_notion = "statistical_parity"


max_pr_shift = 0.1
results = {}
for partioning_type_name, overlap in partioning_type_names:
    res_desc_label, partioning_name, prediction_name = combine_world_info(
        dataset_name, partioning_type_name, clf_name
    )
    train_path_info, _, _ = get_train_val_test_paths(
        base_path, partioning_name, prediction_name, dataset_name
    )

    regions_df = read_scanned_regs(train_path_info["regions"])
    df = pd.read_csv(f"{base_path}preprocess/lar.csv")
    y = get_y(df, "label")

    points_per_region = regions_df["points"].tolist()

    results_path = f"{results_base_path}{res_desc_label}/"

    print(f"{clf_name}, {partioning_type_name}, {fairness_notion}")

    fair_model = SpatialOptimFairnessModel("promis_opt")
    fair_model.multi_fit(
        points_per_region=points_per_region,
        n_flips_start=n_flips_start,
        step=step,
        n_flips=n_flips,
        y_pred=y,
        wlimit=wlimit,
        fair_notion=fairness_notion,
        overlap=overlap,
        init_threshold=None,
        no_of_threads=no_of_threads,
        verbose=1,
        max_pr_shift=max_pr_shift,
    )

    model_save_file = f"{results_path}spatial_optim_models/{fairness_notion}/promis_opt_wlimit_{wlimit}.pkl"

    fair_model.save_model(model_save_file)
