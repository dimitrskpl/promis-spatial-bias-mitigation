import numpy as np
from tqdm import tqdm
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import math

from utils.stats_utils import get_simple_stats, are_all_regions_fair
from utils.scores import compute_statistic


def get_direction(row, y_pred, N, P):
    """
    Determines the direction (-1 or 1) in which a point's label should change to decrease a given statistic.

    Args:
        row (pd.Series): A row from a DataFrame containing region information.
        y_pred (np.ndarray): Array of predicted labels.
        N (int): Total number of samples.
        P (int): Total number of positive labels.

    Returns:
        int: The direction of label change (-1 for decrease, 1 for increase).
    """

    stat_ = row["statistic"].to_list()[0]
    points_ = row["points"].to_list()[0]
    n_ = len(points_)
    p_ = y_pred[points_].sum()
    sign = 0

    if p_ + 1 > n_:
        return -1

    if compute_statistic(n_, p_ + 1, N, P) < stat_:
        sign = 1
    else:
        sign = -1
    return sign


def get_df_points(df_signif, N):
    """
    Constructs a DataFrame summarizing point-level statistics based on significance regions.

    Args:
        df_signif (pd.DataFrame): DataFrame containing significance information for regions.
        N (int): Total number of samples.

    Returns:
        pd.DataFrame: A DataFrame containing point-level significance statistics.
    """

    points_info = {}
    for point in range(N):
        points_info[point] = [[], []]

    for indice in df_signif.index:
        p_direction = df_signif.at[indice, "p_direction"]
        points = df_signif.at[indice, "points"]
        for point in points:
            points_info[point][0].append(p_direction)
            points_info[point][1].append(indice)

    for point in range(N):
        points_info[point].append(len(points_info[point][0]))
        points_info[point].append(np.sum(points_info[point][0]))

    df_points = pd.DataFrame.from_dict(
        points_info, orient="index", columns=["_", "idx_reg", "count", "sum_pdir"]
    )
    df_points.drop("_", inplace=True, axis=1)
    df_points = df_points.reset_index().rename(columns={"index": "points"})
    df_points["nregions"] = df_points[
        "sum_pdir"
    ].abs()  # the regions directions (-1/1) sum diff
    df_points["sug_pdir"] = df_points["sum_pdir"].apply(
        lambda x: -1 if x < 0 else 1
    )  # indicates the desired direction based on the regions directions (-1/1) sum

    df_points["sum_weights_flips"] = 0.0
    for index in df_signif.index:
        weight_to_add = df_signif.at[
            index, "weight_flips"
        ]  # reverse normalized number of flips required to make region fair
        points_involved = df_signif.at[index, "points"]
        df_points.loc[
            df_points["points"].isin(points_involved), "sum_weights_flips"
        ] += weight_to_add
    df_points["rank_flips"] = df_points["nregions"] * df_points["sum_weights_flips"]

    df_points["sum_weights_stat"] = 0.0
    for index in df_signif.index:
        weight_to_add = df_signif.at[index, "weight_stat"]
        points_involved = df_signif.at[index, "points"]
        df_points.loc[
            df_points["points"].isin(points_involved), "sum_weights_stat"
        ] += weight_to_add
    df_points["rank_stat"] = df_points["nregions"] * df_points["sum_weights_stat"]

    df_points.drop(
        ["count", "sum_pdir", "sum_weights_flips", "sum_weights_stat"],
        inplace=True,
        axis=1,
    )

    return df_points


def get_statistic_df(df, N, P, y_pred):
    """
    Computes the statistic for each region based on the updated y_pred array.

    Args:
        df (pd.DataFrame): DataFrame where each row represents a region.
        N (int): Total number of samples.
        P (int): Total number of positive labels.
        y_pred (np.ndarray): Array of predicted labels.

    Returns:
        list: A list containing the computed statistics for each region.
    """
    list_stats = []

    all_pts = set()
    for pt in df["points"].to_list():
        all_pts.update(pt)
    for index in df.index:
        points = df.at[index, "points"]
        n = len(points)
        p = y_pred[points].sum()
        list_stats.append(compute_statistic(n, p, N, P))
    return list_stats


def n_flips(Rho_dataset, y_pred, region_points):
    """
    Computes the number of flips needed to make the positive-negative ratio of a region match the dataset-wide ratio.

    Args:
        Rho_dataset (float): Overall positive-to-total ratio of the dataset.
        y_pred (np.ndarray): Array of predicted labels.
        region_points (list): List of points in a region.

    Returns:
        int: The estimated number of label flips needed.
    """

    n, p, _ = get_simple_stats(region_points, y_pred)

    n_flips = int(math.ceil(np.abs(p - (n * Rho_dataset))))

    return n_flips


def get_actual_flip_pts(
    df_points, label, y_pred, n_flips_, pts_per_region=None, signif_thresh=None
):
    """
    Selects the actual points to flip based on ranking in the DataFrame.

    Args:
        df_points (pd.DataFrame): DataFrame containing point-level rankings.
        label (str): Column name used for ranking selection.
        y_pred (np.ndarray): Array of predicted labels.
        n_flips_ (int): Number of flips to perform.
        pts_per_region (list, optional): List of regions containing points. Defaults to None.
        signif_thresh (float, optional): Significance threshold for fair regions. Defaults to None.

    Returns:
        tuple: (list of flipped points, list of flip directions)
    """

    y_pred_copy = y_pred.copy()
    df_points_sorted = df_points.sort_values(
        by=label, ascending=False, ignore_index=True
    )
    suggested_pts = df_points_sorted["points"].tolist()
    suggested_pdirs = df_points_sorted["sug_pdir"].tolist()
    actual_flips = 0
    actual_flip_pts = []
    actual_flip_pts_sols = []
    N = len(y_pred)
    P_mod = np.sum(y_pred)
    if (
        signif_thresh
        and pts_per_region
        and are_all_regions_fair(pts_per_region, y_pred_copy, signif_thresh, P=P_mod)
    ):
        return [], []

    for i in range(len(suggested_pts)):
        suggested_pdir = suggested_pdirs[i]
        new_label = 1
        if suggested_pdir == -1:
            new_label = 0

        if new_label != y_pred[suggested_pts[i]]:
            actual_flips += 1
            actual_flip_pts.append(suggested_pts[i])
            actual_flip_pts_sols.append(suggested_pdirs[i])

            y_pred_copy[suggested_pts[i]] = new_label
            P_mod += suggested_pdirs[i]

            if (
                signif_thresh is not None
                and pts_per_region is not None
                and are_all_regions_fair(
                    pts_per_region, y_pred_copy, signif_thresh, P=P_mod
                )
            ):
                break

        if actual_flips == n_flips_:
            break

    return actual_flip_pts, actual_flip_pts_sols


def get_iterative_points(
    n_flips,
    df_points,
    df_signif,
    y_pred,
    P,
    N,
    signif_thresh,
    report_progress=False,
):
    """
    Iteratively selects points to flip using an ensemble heuristic method, aiming to optimize fairness.

    Args:
        n_flips (int): Number of flips to perform.
        df_points (pd.DataFrame): DataFrame containing point-level statistics.
        df_signif (pd.DataFrame): DataFrame containing regional significance statistics.
        y_pred (np.ndarray): Current predicted labels.
        P (int): Total number of positive labels.
        N (int): Total number of samples.
        signif_thresh (float): Significance threshold for fair regions.
        report_progress (bool, optional): Whether to display progress using tqdm. Defaults to False.

    Returns:
        tuple: (list of flipped points, list of flip directions, list of execution times per iteration)
    """

    s_exec_time = time.time()
    exec_times = []

    df_points_alt = df_points.copy()
    df_signif_alt = df_signif.copy()
    cur_y_pred = y_pred.copy()

    selected_points = []
    selected_points_set = set()
    selected_points_sols = []
    P_mod = P

    pts_per_region = df_signif["points"].tolist()

    point_regions, point_regions_sol = get_actual_flip_pts(
        df_points_alt, "nregions", y_pred, n_flips, pts_per_region, signif_thresh
    )
    point_regions_flips, point_regions_flips_sol = get_actual_flip_pts(
        df_points_alt, "rank_flips", y_pred, n_flips, pts_per_region, signif_thresh
    )
    point_stat, point_stat_sol = get_actual_flip_pts(
        df_points_alt, "rank_stat", y_pred, n_flips, pts_per_region, signif_thresh
    )

    nregions_idx, nflips_idx, stat_idx = 0, 0, 0

    points_to_pdirs = {}
    for pt, sol in zip(
        point_regions + point_regions_flips + point_stat,
        point_regions_sol + point_regions_flips_sol + point_stat_sol,
    ):
        points_to_pdirs[pt] = sol

    iter_range = range(n_flips)
    if report_progress:
        iterator = tqdm(
            iter_range, desc="Flipping with Ensemble Heuristic Method", leave=False
        )
    else:
        iterator = iter_range

    for i in iterator:

        while point_regions[nregions_idx] in selected_points_set:
            nregions_idx += 1

        while point_regions_flips[nflips_idx] in selected_points_set:
            nflips_idx += 1

        while point_stat[stat_idx] in selected_points_set:
            stat_idx += 1

        candidate_pts = set(
            [
                point_regions[nregions_idx],
                point_regions_flips[nflips_idx],
                point_stat[stat_idx],
            ]
        )
        min_stat = np.inf
        min_point = None
        min_pdir = None
        meanStatisticValues = {}

        for point in candidate_pts:
            y_pred_copy = cur_y_pred.copy()
            suggested_pdir = points_to_pdirs[point]

            if suggested_pdir == -1:
                y_pred_copy[point] = 0
            if suggested_pdir == 1:
                y_pred_copy[point] = 1

            stat = np.mean(
                get_statistic_df(df_signif_alt, N, P_mod + suggested_pdir, y_pred_copy)
            )
            meanStatisticValues[point] = stat

            if stat < min_stat:
                min_stat = stat
                min_point = point
                min_pdir = suggested_pdir

        selected_points.append(min_point)

        selected_points_set.add(min_point)
        selected_points_sols.append(min_pdir)

        if points_to_pdirs[min_point] == -1:
            cur_y_pred[min_point] = 0
            P_mod -= 1

        if points_to_pdirs[min_point] == 1:
            cur_y_pred[min_point] = 1
            P_mod += 1

        df_signif_alt["statistic"] = get_statistic_df(
            df_signif_alt, N, P_mod, cur_y_pred
        )

        exec_times.append(time.time() - s_exec_time)

        if (
            signif_thresh is not None
            and df_signif_alt[df_signif_alt["statistic"] > signif_thresh].shape[0] == 0
        ):
            break

    return selected_points, selected_points_sols, exec_times


def get_exhaustive_points(
    n_flips_,
    df_points_,
    df_signif_,
    y_pred_,
    P_,
    N,
    signif_thresh_=None,
    report_progress=False,
):
    """
    Exhaustively selects the best points to flip based on statistical significance.

    Args:
        n_flips_ (int): Number of flips to perform.
        df_points_ (pd.DataFrame): DataFrame containing point-level rankings.
        df_signif_ (pd.DataFrame): DataFrame containing regional significance information.
        y_pred_ (np.ndarray): Current predicted labels.
        P_ (int): Total number of positive labels.
        N (int): Total number of samples.
        signif_thresh_ (float, optional): Significance threshold for fair regions. Defaults to None.
        report_progress (bool, optional): Whether to display progress using tqdm. Defaults to False.

    Returns:
        tuple: (list of flipped points, list of flip directions, list of execution times per iteration)
    """

    exec_times = []
    s_exec_time = time.time()

    P_mod = P_
    y_pred_copy = y_pred_.copy()
    df_points_alt = df_points_.copy()
    df_signif_alt = df_signif_.copy()
    selected_points = []
    selected_points_sols = []

    pts_per_region = df_signif_alt["points"].tolist()
    points_sug_pdir_dict = df_points_alt.set_index("points")["sug_pdir"].to_dict()

    nregions_strat, _ = get_actual_flip_pts(
        df_points_,
        "nregions",
        y_pred_copy,
        n_flips_,
        pts_per_region,
        signif_thresh_,
    )
    nflips_strat, _ = get_actual_flip_pts(
        df_points_,
        "rank_flips",
        y_pred_copy,
        n_flips_,
        pts_per_region,
        signif_thresh_,
    )
    stat_strat, _ = get_actual_flip_pts(
        df_points_,
        "rank_stat",
        y_pred_copy,
        n_flips_,
        pts_per_region,
        signif_thresh_,
    )

    remaining_points = set(nregions_strat + nflips_strat + stat_strat)

    if not remaining_points:
        return [], [], []

    iter_range = range(n_flips_)
    if report_progress:
        iterator = tqdm(
            iter_range, desc="Flipping with EnsembleLA Heuristic Method", leave=False
        )
    else:
        iterator = iter_range

    for _ in iterator:

        min_point = None
        min_p_dir = None
        min_stat = np.inf
        for point in remaining_points:
            y_pred_aux = y_pred_copy.copy()
            suggested_pdir = points_sug_pdir_dict[point]

            if suggested_pdir == -1:
                y_pred_aux[point] = 0
            if suggested_pdir == 1:
                y_pred_aux[point] = 1

            stat_ = np.mean(
                get_statistic_df(df_signif_alt, N, P_mod + suggested_pdir, y_pred_aux)
            )
            if stat_ < min_stat:
                min_stat = stat_
                min_point = point
                min_p_dir = suggested_pdir

        selected_points.append(min_point)
        selected_points_sols.append(min_p_dir)

        if min_p_dir == -1:
            y_pred_copy[min_point] = 0
            P_mod -= 1

        if min_p_dir == 1:
            y_pred_copy[min_point] = 1
            P_mod += 1

        df_signif_alt["statistic"] = get_statistic_df(
            df_signif_alt, N, P_mod, y_pred_copy
        )

        # we remove the already selected point.
        remaining_points.remove(min_point)

        exec_times.append(time.time() - s_exec_time)

        if (
            signif_thresh_ is not None
            and df_signif_alt[df_signif_alt["statistic"] > signif_thresh_].shape[0] == 0
        ):
            break

    return selected_points, selected_points_sols, exec_times


def get_regions_flip_info(df, N, P, y_pred):
    """
    Computes and updates region-wise statistics, including the number of flips needed to balance fairness.

    Args:
        df (pd.DataFrame): DataFrame where each row represents a region and its associated points.
        N (int): Total number of samples in the dataset.
        P (int): Total number of positive labels in the dataset.
        y_pred (np.ndarray): Array of predicted labels (binary classification).

    Returns:
        pd.DataFrame: Updated DataFrame containing region-level statistics such as:
            - `statistic`: Computed fairness-related statistic for each region.
            - `p_direction`: Direction of label change (-1 or 1) needed for fairness.
            - `n_flips`: Number of label flips needed to match the overall dataset's positive ratio.
            - `n`: Number of points in each region.
            - `p`: Number of positive labels in each region.
            - `weight_flips`: Normalized inverse of `n_flips` (higher weight for regions needing fewer flips).
            - `weight_stat`: Normalized statistic values.
    """

    df = df.copy()
    Rho = P / N
    df["statistic"] = get_statistic_df(df, N, P, y_pred)
    df["p_direction"] = [
        get_direction(df.query("index == @x"), y_pred, N, P) for x in df.index
    ]
    df["n_flips"] = [
        n_flips(Rho, y_pred, region_points) for region_points in df["points"]
    ]

    df["n"] = df["points"].apply(len)
    df["p"] = df["points"].apply(lambda pts: np.count_nonzero(y_pred[pts]))

    df["weight_flips"] = MinMaxScaler().fit_transform(df[["n_flips"]])
    df["weight_flips"] = 1 - df[["weight_flips"]]
    df["weight_stat"] = MinMaxScaler().fit_transform(df[["statistic"]])

    return df


def get_simple_heu_points(
    n_flips_, df_points, label, y_pred, pts_per_region, signif_thresh
):
    """
    Selects points to flip using a simple heuristic method based on ranking.

    Args:
        n_flips_ (int): Number of flips to perform.
        df_points (pd.DataFrame): DataFrame containing point-level statistics and rankings.
        label (str): Column name in `df_points` used for ranking the importance of points.
        y_pred (np.ndarray): Array of predicted labels (binary classification).
        pts_per_region (list): List of regions, each containing a set of points.
        signif_thresh (float): Significance threshold for determining fair regions.

    Returns:
        tuple:
            - list: Selected points to flip.
            - list: Corresponding flip directions (-1 or 1).
            - list: Execution times for each flip iteration (all identical since sorting dominates execution time).
    """

    s_exec_time = time.time()

    actual_points_to_change, actual_points_to_change_sols = get_actual_flip_pts(
        df_points, label, y_pred, n_flips_, pts_per_region, signif_thresh
    )

    exec_time = time.time() - s_exec_time
    # either for 1 or for more flips the exec time is the same
    # since the costly part is to sort the points by the label
    exec_times = len(actual_points_to_change) * [exec_time]

    return actual_points_to_change, actual_points_to_change_sols, exec_times
