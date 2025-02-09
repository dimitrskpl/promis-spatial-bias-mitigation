import numpy as np
import math


def compute_max_likeli(n, p, N, P, verbose=False):
    """
    Computes the maximum likelihood (l1max) for a given region, comparing it to the global likelihood (l0max).

    Args:
        n (int): Number of points in the region.
        p (int): Number of positive labels in the region.
        N (int): Total number of points.
        P (int): Total number of positive labels.
        verbose (bool, optional): If True, prints intermediate steps. Defaults to False.

    Returns:
        float: The maximum likelihood value for the region.
    """

    ## handle extreme cases
    rho = P / N
    if rho == 1 or rho == 0:
        l0max = 0
    else:
        l0max = P * math.log(rho) + (N - P) * math.log(1 - rho)

    if n == 0 or n == N:  ## rho_in == 0/0 or rho_out == 0/0
        l1max = l0max
        if verbose:
            print("n == 0 or n == N")
        return l1max

    rho_in = p / n
    rho_out = (P - p) / (N - n)

    ##########################################################################
    if rho_in == rho_out:
        if verbose:
            print("p/n == (P-p)/(N-n)")
        return l0max
    ##########################################################################
    # both bin are 0
    elif (p == n or p == 0) and (p == P or N - n == P - p):
        if verbose:
            print("p == n and p == P")

        l1max = 0
    elif p == 0:  ## rho_in == 0
        if verbose:
            print("rho_in == 0")

        l1max = P * math.log(rho_out) + (N - n - P) * math.log(1 - rho_out)
    elif p == n:  ## rho_in == 1
        if verbose:
            print("p == n")

        l1max = (P - p) * math.log(rho_out) + (N - P) * math.log(1 - rho_out)
    elif p == P:  ## rho_out == 0
        if verbose:
            print("p == P")

        l1max = p * math.log(rho_in) + (n - p) * math.log(1 - rho_in)
    elif (P - p) / (N - n) == 1:  # rho_out == 1
        if verbose:
            print("P-p == N-n")

        l1max = p * math.log(rho_in) + (n - p) * math.log(1 - rho_in)
    else:
        if verbose:
            print(
                f"{rho_in=}, {rho_out=}, 1-rho_in: {1-rho_in}, 1-rho_out: {1-rho_out}"
            )

        l1max = (
            p * math.log(rho_in)
            + (n - p) * math.log(1 - rho_in)
            + (P - p) * math.log(rho_out)
            + (N - n - (P - p)) * math.log(1 - rho_out)
        )

    return l1max


def compute_statistic(n, p, N, P, verbose=False):
    """
    Computes the test statistic (l1max - l0max) for evaluating fairness in a region.

    Args:
        n (int): Number of points in the region.
        p (int): Number of positive labels in the region.
        N (int): Total number of points.
        P (int): Total number of positive labels.
        verbose (bool, optional): If True, prints intermediate steps. Defaults to False.

    Returns:
        float: The test statistic value.
    """

    ## l1max - l0max

    if verbose:
        print(f"{n=}, {p=}, {N=}, {P=}")

    if n == 0 or n == N:  ## rho_in == 0/0 or rho_out == 0/0
        if verbose:
            print("n == 0 or n == N")
        return 0

    rho = P / N
    rho_in = p / n
    rho_out = (P - p) / (N - n)

    if verbose:
        print(f"{rho=}, {rho_in=}, {rho_out=}")

    if rho == 1 or rho == 0:
        if verbose:
            print("rho == 1 or rho == 0 -> l0max = 0")
        l0max = 0
    else:
        l0max = P * math.log(rho) + (N - P) * math.log(1 - rho)

    if verbose:
        print(f"{l0max=}")

    l1max = compute_max_likeli(n, p, N, P, verbose)
    if verbose:
        print(f"{l1max=}")

    statistic = l1max - l0max

    # l1max = p*math.log(rho_in) + (n-p)*math.log(1-rho_in) + (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out)
    # l0max = P*math.log(rho) + (N-P)*math.log(1-rho)

    if verbose:
        print(f"{l0max=}, {l1max=}, {statistic=}")

    return statistic


def compute_statistic_l0_l1(n, p, N, P, verbose=False):
    """
    Computes the test statistic along with l0max and l1max values.

    Args:
        n (int): Number of points in the region.
        p (int): Number of positive labels in the region.
        N (int): Total number of points.
        P (int): Total number of positive labels.
        verbose (bool, optional): If True, prints intermediate steps. Defaults to False.

    Returns:
        tuple: (statistic, l0max, l1max) where `statistic` is l1max - l0max.
    """

    ## l1max - l0max
    if verbose:
        print(f"{n=}, {p=}, {N=}, {P=}")

    if n == 0 or n == N:  ## rho_in == 0/0 or rho_out == 0/0
        print("n == 0 or n == N")
        return 0, 0, 0

    rho = P / N
    rho_in = p / n
    rho_out = (P - p) / (N - n)

    if verbose:
        print(f"{rho=}, {rho_in=}, {rho_out=}")

    if rho == 1 or rho == 0:
        l0max = 0
    else:
        l0max = P * math.log(rho) + (N - P) * math.log(1 - rho)

    l1max = compute_max_likeli(n, p, N, P)
    statistic = l1max - l0max

    # l1max = p*math.log(rho_in) + (n-p)*math.log(1-rho_in) + (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out)
    # l0max = P*math.log(rho) + (N-P)*math.log(1-rho)

    if verbose:
        print(f"{l0max=}, {l1max=}, {statistic=}")

    return statistic, l0max, l1max


def compute_statistic_with_info(n, p, N, P, verbose=False):
    """
    Computes the test statistic and returns it along with in-region and out-region proportions.

    Args:
        n (int): Number of points in the region.
        p (int): Number of positive labels in the region.
        N (int): Total number of points.
        P (int): Total number of positive labels.
        verbose (bool, optional): If True, prints intermediate steps. Defaults to False.

    Returns:
        tuple: (statistic, rho_in, rho_out) where `statistic` is l1max - l0max, `rho_in` is the in-region proportion, and `rho_out` is the out-region proportion.
    """

    ## l1max - l0max

    if verbose:
        print(f"{n=}, {p=}, {N=}, {P=}")

    rho = P / N
    rho_in = p / n
    rho_out = (P - p) / (N - n)

    if n == 0 or n == N:  ## rho_in == 0/0 or rho_out == 0/0
        if verbose:
            print("n == 0 or n == N")
        return 0, rho_in, rho_out

    if verbose:
        print(f"{rho=}, {rho_in=}, {rho_out=}")

    if rho == 1 or rho == 0:
        if verbose:
            print("rho == 1 or rho == 0 -> l0max = 0")
        l0max = 0
    else:
        l0max = P * math.log(rho) + (N - P) * math.log(1 - rho)

    if verbose:
        print(f"{l0max=}")

    l1max = compute_max_likeli(n, p, N, P)
    if verbose:
        print(f"{l1max=}")

    statistic = l1max - l0max

    # l1max = p*math.log(rho_in) + (n-p)*math.log(1-rho_in) + (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out)
    # l0max = P*math.log(rho) + (N-P)*math.log(1-rho)

    if verbose:
        print(f"{l0max=}, {l1max=}, {statistic=}")

    return statistic, rho_in, rho_out


def fair_mlr(labels, points_per_region, sign_thres):
    """
    Computes the mean likelihood ratio (MLR) for regions, setting values below the significance threshold to zero.

    Args:
        labels (np.ndarray): Array of binary labels.
        points_per_region (list): List of regions, each containing indices of points.
        sign_thres (float): Significance threshold for filtering.

    Returns:
        float: Mean likelihood ratio (MLR) across the regions.
    """

    P = np.sum(labels)
    N = len(labels)
    list_stats = []
    for i in range(len(points_per_region)):
        n = len(points_per_region[i])
        p = np.sum(labels[points_per_region[i]])
        stat_i = compute_statistic(n, p, N, P)
        if stat_i < sign_thres:
            stat_i = 0

        list_stats.append(stat_i)

    mlr = np.mean(list_stats)
    return mlr


def get_fair_mlr_ratio(labels, points_per_region, sign_thres):
    """
    Computes the mean likelihood ratio (MLR) ratio for regions relative to a significance threshold.

    Args:
        labels (np.ndarray): Array of binary labels.
        points_per_region (list): List of regions, each containing indices of points.
        sign_thres (float): Significance threshold for filtering.

    Returns:
        float: Mean likelihood ratio (MLR) ratio across the regions.
    """

    P = np.sum(labels)
    N = len(labels)
    list_stat_ratio = []

    for i in range(len(points_per_region)):
        n = len(points_per_region[i])
        p = np.sum(labels[points_per_region[i]])
        stat_i = compute_statistic(n, p, N, P)
        if stat_i < sign_thres:
            stat_i = sign_thres

        stat_i_ratio = sign_thres / stat_i
        list_stat_ratio.append(stat_i_ratio)

    mlr_ratio = np.mean(list_stat_ratio)
    return mlr_ratio


def stats_to_fair_mlr(stats, sign_thres):
    """
    Adjusts a list of statistics by setting values below a threshold to zero and computes the mean.

    Args:
        stats (list): List of computed statistics.
        sign_thres (float): Significance threshold for filtering.

    Returns:
        float: Mean of adjusted statistics.
    """
    new_stats = [0 if stats[i] < sign_thres else stats[i] for i in range(len(stats))]
    return np.mean(new_stats)


def stats_to_fair_mlr_ratio(stats, sign_thres):
    """
    Adjusts a list of statistics by computing the ratio to the threshold for values above it and returns the mean.

    Args:
        stats (list): List of computed statistics.
        sign_thres (float): Significance threshold for filtering.

    Returns:
        float: Mean ratio of the statistics relative to the significance threshold.
    """

    stats_ratios = [
        1 if stats[i] < sign_thres else sign_thres / stats[i] for i in range(len(stats))
    ]
    return np.mean(stats_ratios)


def get_mlr(labels, points_per_region, with_stats=False):
    """
    Computes the Mean Likelihood Ratio (MLR) for a set of regions based on the label distribution.

    Args:
        labels (np.ndarray): Array of binary labels for all points.
        points_per_region (list): List of regions, where each region contains indices of points.

    Returns:
        float: The Mean Likelihood Ratio (MLR) across all regions.
    """

    P = np.sum(labels)
    N = len(labels)
    list_stats = []

    for i in range(len(points_per_region)):
        n = len(points_per_region[i])
        p = np.sum(labels[points_per_region[i]])
        stat_i = compute_statistic(n, p, N, P)
        list_stats.append(stat_i)

    mlr = np.mean(list_stats)
    if with_stats:
        return mlr, list_stats

    return mlr


def get_fair_stat_ratios(stats, pr_s, PR, max_stat=None):
    """
    Computes normalized fairness statistic based on a given threshold
    and set sign indicating favor or unfavor region

    This function normalizes a list of statistical values by the maximum statistic
    and assigns a positive or negative sign depending on whether the probability
    ratio (`pr`) exceeds a given threshold (`PR`).

    Args:
        stats (list of float): A list of statistical values to be normalized.
        pr_s (list of float): A list of probability ratios corresponding to each statistic.
        PR (float): Threshold probability ratio to determine the sign of the fairness statistic.
        max_stat (float, optional): Maximum statistic value used for normalization.
            If None, it is set to the maximum of `stats`. Defaults to None.

    Returns:
        tuple:
            - fair_stat_ratios (list of float): List of normalized fairness statistic ratios.
            - max_stat (float): The maximum statistic value used for normalization.
    """

    max_stat = max(stats) if max_stat is None else max_stat
    fair_stat_ratios = []
    for stat, pr in zip(stats, pr_s):
        fair_stat_ratio = stat / max_stat if pr > PR else -stat / max_stat
        fair_stat_ratios.append(fair_stat_ratio)

    return fair_stat_ratios, max_stat
