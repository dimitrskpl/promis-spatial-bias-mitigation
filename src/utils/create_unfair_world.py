import pandas as pd
import numpy as np
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join("..")))
from utils.stats_utils import get_stats, get_signif_threshold
from utils.stats_utils import get_stats
from utils.scores import get_sbi
from utils.data_utils import read_scanned_regs, get_y


def create_unfair_world(
    rho,
    test_path_info,
    predictions_path,
    partioning_name,
    seed=36,
    signif_level=0.0005,
    n_alt_worlds=200,
):
    """
    Creates a semi-synthetic “unfair world” scenario by flipping labels between
    non-overlapping pairs of regions while preserving the overall number of positive labels.

    This function first reads region data, original predictions, and labels from the paths provided
    in `test_path_info`. It then generates a “fair” set of labels (y_pred) by drawing from a binomial
    distribution with probability `rho`. Next, it identifies pairs of geographically regions
    and flips labels between them to introduce localized biases (“unfairness”). The flipping is constrained
    so that the total count of positive labels remains unchanged. Finally, new predictions,
    and a flip “budget” are saved to the specified output paths.

    Parameters
    ----------
    rho : float
        Probability in [0, 1] used to generate initial positive labels from a binomial distribution.
    test_path_info : dict
        Dictionary containing the file paths/keys required to load the test dataset:
            - "regions": path to the CSV file with region information (must have columns such as
              'center_lat', 'center_lon', and 'points').
            - "predictions": path to the CSV file with original predictions.
            - "labels": path to the CSV file with original labels.
    predictions_path : str
        Directory path where the newly generated predictions CSV and the budget text file will be saved.
    labels_path : str
        Directory path where the newly generated labels CSV will be saved.
    partioning_name : str
        A name (suffix) used to distinguish output files for different partitions/scenarios.
    seed : int, optional (default=36)
        Random seed for reproducibility of label generation and region flipping.
    signif_level : float, optional (default=0.0005)
        Significance level passed to the function that computes region-level significance thresholds.
    n_alt_worlds : int, optional (default=200)
        Number of alternative worlds used when computing significance thresholds.

    """

    def gen_y_pred(n, rho, seed=None):

        if seed:
            np.random.seed(seed)
            random.seed(seed)

        types = np.random.binomial(size=n, n=1, p=rho)

        # guarantee n * rho positives
        n_pos = int(n * rho)

        while np.sum(types) != n_pos:
            idx = np.random.randint(0, n)
            if np.sum(types) > n_pos and types[idx] == 1:
                types[idx] = 0
            elif np.sum(types) < n_pos and types[idx] == 0:
                types[idx] = 1

        return types

    test_regions_df = read_scanned_regs(test_path_info["regions"])
    test_pred_df = pd.read_csv(test_path_info["predictions"])
    test_labels_df = pd.read_csv(test_path_info["labels"])
    test_points_per_region = test_regions_df["points"].tolist()
    test_regions = [{"points": pts} for pts in test_points_per_region]

    # --------------------------------------
    #  Generate "fair" distribution of labels
    # --------------------------------------

    df_fair = test_pred_df.copy()
    fair_y_pred = gen_y_pred(df_fair.shape[0], rho, seed=seed)
    df_fair["pred"] = fair_y_pred
    df_fair.to_csv(
        f"{predictions_path}test_fair_pred_semi_synthetic_{partioning_name}.csv",
        index=False,
    )

    N_fair, P_fair = get_stats(df_fair, "pred")
    print(
        f"Stats with generation of true types with binomial of rho {rho} making the world unfair:"
    )
    print(f"N={N_fair} points")
    print(f"P={P_fair} positives")

    fair_sbi, fair_stats = get_sbi(fair_y_pred, test_points_per_region, with_stats=True)
    fair_signif_thresh = get_signif_threshold(
        signif_level, n_alt_worlds, test_regions, N_fair, P_fair
    )
    fair_total_signif = sum([1 for s in fair_stats if s > fair_signif_thresh])

    print(f"Original SBI: {fair_sbi:.3f}")
    print(f"Significance threshold: {fair_signif_thresh:.3f}")
    print(f"Total significant regions: {fair_total_signif}")

    indices = test_regions_df.index.tolist()

    # --------------------------------------
    #  Identify non-overlapping pairs of regions for flipping
    # --------------------------------------
    candidate_pairs_indices = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            candidate_pairs_indices.append((indices[i], indices[j]))

    valid_pairs_info = []  # keeps valid (reg1_idx, reg2_idx, max_reg1_pts_to_flip)
    for i in range(len(candidate_pairs_indices)):
        reg1_idx = candidate_pairs_indices[i][0]
        reg2_idx = candidate_pairs_indices[i][1]
        reg1_pts = test_regions_df[test_regions_df.index == reg1_idx]["points"].values[
            0
        ]
        reg2_pts = test_regions_df[test_regions_df.index == reg2_idx]["points"].values[
            0
        ]

        reg1_pts_set = set(reg1_pts)
        reg2_pts_set = set(reg2_pts)

        # for simplicity we choose pairs of
        # regions whose members dont overlap
        reg_1_2_common_pts = reg1_pts_set & reg2_pts_set

        if len(reg_1_2_common_pts) == 0:
            # no common points

            n1 = len(reg1_pts)
            pos1 = np.sum(fair_y_pred[reg1_pts])
            neg1 = n1 - pos1

            n2 = len(reg2_pts)
            pos2 = np.sum(fair_y_pred[reg2_pts])
            neg2 = pos2 - n2

            # Determine direction of flipping
            # max(min(neg1, pos2), min(pos1, neg2))
            if min(neg1, pos2) > min(pos1, neg2):
                if neg1 < pos2:
                    max_reg1_pts_to_flip = neg1
                else:
                    max_reg1_pts_to_flip = pos2
            else:
                if neg2 < pos1:
                    max_reg1_pts_to_flip = -neg2
                else:
                    max_reg1_pts_to_flip = -pos1

            valid_pairs_info.append((reg1_idx, reg2_idx, max_reg1_pts_to_flip))

    # --------------------------------------
    #  Sort valid pairs by maximum flippable points
    #  and select non-overlapping regions
    # --------------------------------------
    sorted_by_most_pts_valid_pairs_info = sorted(
        valid_pairs_info, key=lambda x: x[2], reverse=True
    )
    selected_reg_idxs = set()
    unique_valid_pairs = []  # a region should appear only once in all pairs
    for reg_1_idx, reg_2_idx, no_of_pts in sorted_by_most_pts_valid_pairs_info:
        if reg_1_idx not in selected_reg_idxs and reg_2_idx not in selected_reg_idxs:
            unique_valid_pairs.append((reg_1_idx, reg_2_idx, no_of_pts))
            selected_reg_idxs.add(reg_1_idx)
            selected_reg_idxs.add(reg_2_idx)

    total_pts_to_flip = 0
    for _, _, no_of_pts in unique_valid_pairs:
        total_pts_to_flip += (
            2 * no_of_pts
        )  # each no_of_pts corresponds to no of pts to flips to each region in each pair

    valid_pairs_info = unique_valid_pairs

    # --------------------------------------
    #  Perform label flipping
    # --------------------------------------
    np.random.seed(seed)
    pos_label = 1
    neg_label = 0
    total_pairs_valid_pairs_info = valid_pairs_info

    unfair_df = df_fair.copy()
    unfair_df.drop("prob", axis=1, inplace=True)

    total_pts_flipped = 0
    for reg1_idx, reg2_idx, max_reg1_pts_to_flip in total_pairs_valid_pairs_info:
        reg1_pts = test_regions_df.loc[test_regions_df.index == reg1_idx][
            "points"
        ].values[0]
        reg2_pts = test_regions_df.loc[test_regions_df.index == reg2_idx][
            "points"
        ].values[0]

        if max_reg1_pts_to_flip > 0:
            # Flip direction: reg1: neg->pos, reg2: pos->neg
            search_pred_reg_1 = 0
            search_pred_reg_1 = 0
            search_pred_reg_2 = 1
            reg1_new_label = pos_label
            reg2_new_label = neg_label
        else:
            # Flip direction: reg1: pos->neg, reg2: neg->pos
            search_pred_reg_1 = 1
            search_pred_reg_2 = 0
            reg1_new_label = neg_label
            reg2_new_label = pos_label

        total_flips = np.abs(max_reg1_pts_to_flip)

        # Identify points to flip
        reg1_pts_to_flip_idxs = unfair_df[
            (unfair_df.index.isin(reg1_pts)) & (unfair_df["pred"] == search_pred_reg_1)
        ].index.tolist()[:total_flips]
        reg2_pts_to_flip_idxs = unfair_df[
            (unfair_df.index.isin(reg2_pts)) & (unfair_df["pred"] == search_pred_reg_2)
        ].index.tolist()[:total_flips]

        # Adjust if there’s overlap or fewer points than expected
        pts_to_keep = min(len(reg1_pts_to_flip_idxs), len(reg2_pts_to_flip_idxs))
        total_pts_flipped += pts_to_keep

        # adjust the selected point indices
        reg1_pts_to_flip_idxs = reg1_pts_to_flip_idxs[:pts_to_keep]
        reg2_pts_to_flip_idxs = reg2_pts_to_flip_idxs[:pts_to_keep]

        common_points = set(reg1_pts_to_flip_idxs) & set(reg2_pts_to_flip_idxs)
        assert common_points == set()  # test if pair of regions overlap

        # Apply flips
        unfair_df.loc[reg1_pts_to_flip_idxs, "pred"] = reg1_new_label
        unfair_df.loc[reg2_pts_to_flip_idxs, "pred"] = reg2_new_label

        N, P = get_stats(unfair_df, "pred")
        print()
        # Ensure total positives remain the same
        assert P == P_fair, "Number of positive labels changed unexpectedly."

    # --------------------------------------
    #  Verify flips and compute final stats
    # --------------------------------------
    unfair_y_pred = get_y(unfair_df, "pred")

    total_diff_fair_unfair_preds = np.sum(np.abs(unfair_y_pred - fair_y_pred))
    print(f"actual flips: {total_diff_fair_unfair_preds}")

    diff_fair_unfair_ttypes = unfair_y_pred - fair_y_pred
    pos_to_neg_idxs = np.where(diff_fair_unfair_ttypes == 1)[0]
    neg_to_pos_idxs = np.where(diff_fair_unfair_ttypes == -1)[0]
    print(
        f"pos2neg flips: {len(pos_to_neg_idxs)}, neg2pos flips {len(neg_to_pos_idxs)}, total: {len(pos_to_neg_idxs)+ len(neg_to_pos_idxs)}"
    )
    n_flips = total_diff_fair_unfair_preds

    unfair_y_pred = get_y(unfair_df, "pred")
    N, P = get_stats(unfair_df, "pred")

    unfair_sbi, unfair_stats = get_sbi(
        unfair_y_pred, test_points_per_region, with_stats=True
    )
    unfair_signif_thresh = get_signif_threshold(
        signif_level, n_alt_worlds, test_regions, N, P
    )
    unfair_total_signif = sum([1 for s in unfair_stats if s > unfair_signif_thresh])

    print(f"Unfair World SBI: {unfair_sbi:.3f}")
    print(f"Significance threshold: {unfair_signif_thresh:.3f}")
    print(f"Total significant regions: {unfair_total_signif}")

    unfair_df.to_csv(
        f"{predictions_path}test_pred_semi_synthetic_{partioning_name}.csv", index=False
    )
    with open(
        f"{predictions_path}test_pred_semi_synthetic_{partioning_name}_budget.txt", "w"
    ) as file:
        file.write(str(n_flips))
