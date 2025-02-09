import numpy as np


def convert_indiv_sol_2_regions_sol(sol, points_per_region):
    """
    Converts individual solution adjustments into regional solution adjustments.

    Args:
        sol (np.ndarray): An array representing the individual adjustments for each data point.
        points_per_region (list): A list where each element contains indices of points belonging to a specific region.

    Returns:
        np.ndarray: An array representing the total sum of adjustments per region.
    """
    return np.array([np.sum(sol[pts]) for pts in points_per_region])


def adjusted_thresholds(regions_dp, init_thresholds, points_per_region, probs):
    """
    Adjusts classification thresholds for each region based on decision points.

    Args:
        regions_dp (np.ndarray): An array containing the number of changes needed for each region.
        init_thresholds (np.ndarray): An array of initial classification thresholds per region.
        points_per_region (list): A list where each element contains indices of points belonging to a specific region.
        probs (np.ndarray): An array containing probability scores for each data point.

    Returns:
        tuple:
            - list: New adjusted thresholds per region.
            - list: Probabilistic flipping probabilities for cases where the decision is not deterministic.
    """

    new_thresholds = []
    pos_flip_probs = []

    for region_idx, dp in enumerate(regions_dp):
        prev_thresh = init_thresholds[region_idx]
        new_thresh = prev_thresh
        pos_flip_prob = -1
        if dp != 0:
            search_prob_idx = int(abs(dp)) - 1
            region_pts = np.array(points_per_region[region_idx])
            region_probs = probs[region_pts]
            decim_dp = dp - int(dp)

            region_pts_exam_probs_indices = (
                np.where(region_probs <= prev_thresh)[0]
                if dp > 0
                else np.where(region_probs > prev_thresh)[0]
            )

            region_pts_exam_probs = region_probs[region_pts_exam_probs_indices]
            sorted_probs = (
                np.sort(region_pts_exam_probs)[::-1]
                if dp > 0
                else np.sort(region_pts_exam_probs)
            )

            if search_prob_idx == -1:
                # if dp = +/- 0.xx new_thresh = old and nxt is the 0th prob
                if sorted_probs[0] != prev_thresh:
                    middle_prob = (prev_thresh + sorted_probs[0]) / 2
                    if decim_dp > 0:
                        new_thresh = prev_thresh - decim_dp * (
                            prev_thresh - middle_prob
                        )
                    elif decim_dp < 0:
                        new_thresh = prev_thresh + np.abs(decim_dp) * (
                            middle_prob - prev_thresh
                        )
                else:
                    new_thresh = prev_thresh
                    pos_flip_prob = np.abs(decim_dp) / len(
                        np.where(sorted_probs == prev_thresh)[0]
                    )
            elif (
                search_prob_idx < len(region_pts_exam_probs_indices) - 1
                and sorted_probs[search_prob_idx] == sorted_probs[search_prob_idx + 1]
            ):
                # if the number of probabilities that has the same value
                # equal to the new threshold exceeds the search_prob_idx
                # then more than the desirable number of points will be flipped
                # if we change the new threshold and less than the desirable number
                # of points will be flipped if we do not change the new threshold.
                # In this case we consider a probabilistic approach to decide
                # whether to include the indexed probability or not with a probability
                # equal to the decimal part of the dp

                new_thresh = sorted_probs[search_prob_idx]
                # determine the desirable no of equal probabilities to the new threshold
                included_eq_prob_to_thresh_cnt = 0
                for idx in range(search_prob_idx, -1, -1):
                    if sorted_probs[idx] != new_thresh:
                        break
                    included_eq_prob_to_thresh_cnt += 1

                # determine the undesirable no of equal probabilities to the new threshold
                extra_eq_prob_to_thresh_cnt = 0
                for idx in range(
                    search_prob_idx + 1, len(region_pts_exam_probs_indices)
                ):
                    if sorted_probs[idx] != new_thresh:
                        break
                    extra_eq_prob_to_thresh_cnt += 1

                pos_flip_prob = (included_eq_prob_to_thresh_cnt + np.abs(decim_dp)) / (
                    included_eq_prob_to_thresh_cnt + extra_eq_prob_to_thresh_cnt
                )
                if dp < 0:
                    pos_flip_prob = 1 - pos_flip_prob
            else:
                # normal case find threshold shifted by the decimal part of dp threh whihc include int(dp) points
                if search_prob_idx < len(region_pts_exam_probs_indices) - 1:
                    nxt_2_prob = sorted_probs[search_prob_idx + 1]
                else:
                    nxt_2_prob = 0 if dp > 0 else 1
                middle_prob = (sorted_probs[search_prob_idx] + nxt_2_prob) / 2
                if decim_dp > 0:
                    new_thresh = sorted_probs[search_prob_idx] - decim_dp * (
                        sorted_probs[search_prob_idx] - middle_prob
                    )
                elif decim_dp < 0:
                    new_thresh = sorted_probs[search_prob_idx] + np.abs(decim_dp) * (
                        middle_prob - sorted_probs[search_prob_idx]
                    )
                else:
                    new_thresh = middle_prob

        new_thresholds.append(new_thresh)
        pos_flip_probs.append(pos_flip_prob)
    return new_thresholds, pos_flip_probs
