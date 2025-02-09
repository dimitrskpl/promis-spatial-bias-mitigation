import numpy as np
import pandas as pd
from utils.scores import compute_statistic


def get_random_types(N, P, seed=None):
    """
    Generates a random binary array representing types based on a binomial distribution.

    Args:
        N (int): Total number of elements.
        P (int): Total number of positive elements.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: A binary array of size N with approximately P positive values.
    """

    if seed is not None:
        np.random.seed(seed)
    return np.random.binomial(size=N, n=1, p=P / N)


def scan_regions(regions, types, N, P, verbose=False):
    """
    Computes the statistic for each region and identifies the region with the highest likelihood.

    Args:
        regions (list): List of region dictionaries, each containing "points".
        types (np.ndarray): Binary array indicating type assignment.
        N (int): Total number of elements.
        P (int): Total number of positive elements.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        tuple: The best region dictionary, the maximum likelihood value, and a list of statistics for all regions.
    """

    statistics = []

    for region in regions:
        n, p, _ = get_simple_stats(region["points"], types)
        statistics.append(compute_statistic(n, p, N, P))

    idx = np.argmax(statistics)

    max_likelihood = statistics[idx]

    if verbose:
        print("range", np.amin(statistics), np.amax(statistics))
        print("max likelihood", max_likelihood)
        n, p, _ = get_simple_stats(regions[idx]["points"], types)
        compute_statistic(n, p, N, P, verbose=verbose)

    return regions[idx], max_likelihood, statistics


def get_signif_threshold(
    signif_level, n_alt_worlds, regions, N, P, seed=None, verbose=False
):
    """
    Computes the significance threshold based on alternative worlds.

    Args:
        signif_level (float): Significance level (e.g., 0.05 for 5% significance).
        n_alt_worlds (int): Number of alternative worlds to generate.
        regions (list): List of regions.
        N (int): Total number of elements.
        P (int): Total number of positive elements.
        seed (int, optional): Seed for reproducibility. Defaults to None.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        float: The computed significance threshold.
    """

    alt_worlds, _ = scan_alt_worlds(n_alt_worlds, regions, N, P, seed, verbose)

    k = int(signif_level * n_alt_worlds)

    signif_thresh = alt_worlds[k][2]  ## get the max likelihood at position k

    return signif_thresh


def scan_alt_worlds(n_alt_worlds, regions, N, P, seed=None, verbose=False):
    """
    Scans multiple alternative worlds and ranks them by maximum likelihood.

    Args:
        n_alt_worlds (int): Number of alternative worlds to generate.
        regions (list): List of regions.
        N (int): Total number of elements.
        P (int): Total number of positive elements.
        seed (int, optional): Seed for reproducibility. Defaults to None.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        tuple: A list of alternative worlds sorted by likelihood and the highest likelihood value.
    """

    alt_worlds = []
    current_seed = seed

    for _ in range(n_alt_worlds):
        alt_types = get_random_types(N, P, current_seed)
        cur_P = np.sum(alt_types)
        alt_best_region, alt_max_likeli, _ = scan_regions(
            regions, alt_types, N, cur_P, verbose=verbose
        )
        alt_worlds.append((alt_types, alt_best_region, alt_max_likeli))

        if current_seed is not None:
            current_seed += 1

    alt_worlds.sort(key=lambda x: -x[2])

    return alt_worlds, alt_worlds[0][2]


def get_stats(df, label):
    """
    Computes basic statistics for a given dataset and label.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        label (str): The column name used to count occurrences.

    Returns:
        tuple: Total number of samples (N) and the count of positive samples (P).
    """

    N = len(df)
    P = df.loc[df[label] == 1, label].count()

    return N, P


def get_simple_stats(points, types):
    """
    Computes the number of points, number of positive cases, and their ratio.

    Args:
        points (list): List of point indices.
        types (np.ndarray): Binary array indicating type assignment.

    Returns:
        tuple: Number of points (n), number of positives (p), and ratio (rho).
    """

    n = len(points)
    p = types[points].sum()
    if n > 0:
        rho = p / n
    else:
        rho = np.nan

    return (n, p, rho)


def id2loc(df, point_id):
    """
    Retrieves latitude and longitude for a given point ID.

    Args:
        df (pd.DataFrame): Dataframe containing geographical data.
        point_id (int): Index of the point.

    Returns:
        tuple: Latitude and longitude of the specified point.
    """

    lat = df.loc[[point_id]]["lat"].values[0]
    lon = df.loc[[point_id]]["lon"].values[0]

    return (lat, lon)


def get_total_signif_regs(labels, points_per_region, sign_thres):
    """
    Counts the number of significant regions based on a given threshold.

    Args:
        labels (np.ndarray): Binary labels indicating positive cases.
        points_per_region (list): List of regions, each containing point indices.
        sign_thres (float): Significance threshold.

    Returns:
        int: Number of significant regions.
    """

    P = np.sum(labels)
    N = len(labels)
    cnt = 0

    for i in range(len(points_per_region)):
        n = len(points_per_region[i])
        p = np.sum(labels[points_per_region[i]])
        stat_i = compute_statistic(n, p, N, P)
        if stat_i > sign_thres:
            cnt += 1

    return cnt


def are_all_regions_fair(points_per_region, y_pred, signif_thresh, P=None):
    """
    Checks if all regions satisfy the fairness constraint.

    Args:
        points_per_region (list): List of regions, each containing point indices.
        y_pred (np.ndarray): Predicted labels.
        signif_thresh (float): Significance threshold.
        P (int, optional): Total number of positive predictions. If None, it is computed.

    Returns:
        bool: True if all regions are fair, False otherwise.
    """

    N = len(y_pred)

    if P == None:
        P = np.sum(y_pred)

    for pts in points_per_region:
        n = len(pts)
        p = np.sum(y_pred[pts])
        stat = compute_statistic(n, p, N, P)
        if stat >= signif_thresh:
            return False

    return True


def get_points_not_covered(df, regions):
    """
    Identifies points that are not covered by any region.

    Args:
        df (pd.DataFrame): Dataframe containing all points.
        regions (list): List of region dictionaries, each containing "points".

    Returns:
        list: List of point indices that are not covered by any region.
    """

    covered_points = set()

    for region in regions:
        for point in region["points"]:
            covered_points.add(point)

    all_points = set(df.index)
    missing_points = all_points - covered_points

    return list(missing_points)
