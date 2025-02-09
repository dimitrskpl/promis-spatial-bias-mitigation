import pandas as pd
import numpy as np
from sklearn import metrics
from tqdm.notebook import tqdm
from utils.scores import get_mlr
from utils.data_utils import get_pos_info_regions
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def get_all_methods_modes_labels(labels):
    """
    Categorizes a list of method labels into different classification groups
    based on their prefixes and suffixes.

    Args:
        labels (list): A list of method labels (strings) to be categorized.

    Returns:
        dict: A dictionary containing categorized labels, with keys:
            - "opt_labels" (list): Methods classified as optimization-based.
            - "heu_labels" (list): Methods classified as heuristic-based.

    The function:
    - Iterates through the given list of method labels.
    - Classifies them as optimization-based (`opt_labels`) or heuristic-based (`heu_labels`).
    - Returns a dictionary containing all categorized labels.
    """

    opt_labels = []
    heu_labels = []

    for method in labels:
        if method.startswith("promis_opt") or method.startswith("promis_app"):
            opt_labels.append(method)
        else:
            heu_labels.append(method)

    splitted_labels = {
        "opt_labels": opt_labels,
        "heu_labels": heu_labels,
    }

    return splitted_labels


def get_sol_info(pts_to_change, pts_to_change_sol, y_pred):
    """
    Computes solution information for label modifications, including the solution vector,
    updated labels, and the number of actual label flips.

    Args:
        pts_to_change (np.ndarray): Indices of points whose labels need to be changed.
        pts_to_change_sol (np.ndarray): Solution values corresponding to the points to change.
        y_pred (np.ndarray): Original predicted labels.

    Returns:
        pd.Series: A series containing:
            - "sol" (np.ndarray): Solution vector indicating label modifications.
            - "new_labels" (np.ndarray): Updated labels after applying the solution.
            - "actual_flips" (int): Total number of label flips applied.
    """

    sol = np.zeros(len(y_pred))
    new_labels = y_pred.copy()
    if pts_to_change.size != 0:
        sol[pts_to_change] = pts_to_change_sol
        new_labels[pts_to_change] = y_pred[pts_to_change] + sol[pts_to_change]
    actual_flips = np.sum(np.abs(sol))

    result = pd.Series(
        [sol, new_labels, actual_flips],
        index=[
            "sol",
            "new_labels",
            "actual_flips",
        ],
    )

    return result


def get_labels_info(labels, points_per_region):
    """
    Computes the number of positive labels,
    their positive ratio, and MLR

    Args:
        labels (np.ndarray): Binary labels (0s and 1s) representing classifications.
        points_per_region (list): List of lists, where each sublist contains indices of points in a region.

    Returns:
        pd.Series: A series containing:
            - "P" (int): Total count of positive labels.
            - "RHO" (float): Ratio of positive labels to total labels.
            - "mlr" (float): Mean label ratio (MLR) across all regions.
    """

    P = np.sum(labels)
    N = len(labels)
    RHO = P / N
    mlr = get_mlr(labels, points_per_region)

    result = pd.Series(
        [P, RHO, mlr],
        index=[
            "P",
            "RHO",
            "mlr",
        ],
    )

    return result


def get_actual_flips(new_preds, old_preds):
    """
    Computes the number of actual label flips between two sets of predictions.

    Args:
        new_preds (np.ndarray): Updated predicted labels.
        old_preds (np.ndarray): Original predicted labels.

    Returns:
        int: Number of labels that changed between old and new predictions.
    """
    return len(np.where(np.array(new_preds) != np.array(old_preds))[0])


def compute_all_results_info(
    all_meths_2_pretrained_models,
    y_pred_test_orig,
    test_points_per_region=None,
    y_pred_test_probs=None,
    y_true_test=None,
    apply_fit_flips=False,
):
    """
    Computes various performance metrics and statistical information for multiple
    pre-trained models, evaluating their predictive performance across different budgets.

    Args:
        all_meths_2_pretrained_models (dict): A dictionary mapping method names to their pre-trained models.
        y_pred_test_orig (np.ndarray): Original test set predictions.
        test_points_per_region (list, optional): List of lists containing test region point indices. Defaults to None.
        y_pred_test_probs (np.ndarray, optional): Probability predictions for the test set. Defaults to None.
        y_true_test (np.ndarray, optional): True labels for the test set. Defaults to None.
        apply_fit_flips=False (bool, optional): Whether to apply the flips computed during the fit phase (for PROMIS methods). Defaults to False.

    Returns:
        tuple:
            - dict: A dictionary mapping method names to DataFrames containing computed results.
            - list: A list representing the range of budget values.

    The function performs the following:
    - Initializes a results dictionary for each method.
    - Iterates over all methods and retrieves their model-fit information.
    - Predicts test set labels for different budget levels.
    - Computes actual label flips between original and predicted values.
    - Extracts statistical metrics such as:
        - `P`: Total count of positive labels.
        - `RHO`: Proportion of positive labels.
        - `mlr_st_par`: Mean label ratio.
    - Computes fairness-related metrics MLR for equal opportunity and statistical parity.
    - Calculates accuracy and balanced accuracy scores for each budget if true labels are provided.
    - Returns all results in a dictionary.
    """

    all_methods_to_results_info = {
        method: pd.DataFrame() for method in all_meths_2_pretrained_models.keys()
    }

    for method, res_df in tqdm(
        all_methods_to_results_info.items(),
        desc=f"Computing solutions info for each method",
        leave=False,
        unit="method",
    ):
        print(method)
        model = all_meths_2_pretrained_models[method]
        res_df = model.get_model_fit_info_as_df()

        if model.optimization:
            if apply_fit_flips:
                budget_2_test_predictions = model.multi_predict(
                    test_points_per_region,
                    y_pred_test_orig,
                    apply_fit_flips=apply_fit_flips,
                )
            else:
                budget_2_test_predictions = model.multi_predict(
                    test_points_per_region,
                    y_pred_test_probs,
                    apply_fit_flips=apply_fit_flips,
                )
        else:
            budget_2_test_predictions = model.multi_predict(y_pred_test_orig)

        res_df["y_pred_test"] = res_df["budget"].apply(
            lambda budget: budget_2_test_predictions[budget]
        )

        res_df["actual_flips_test"] = res_df["y_pred_test"].apply(
            lambda preds: get_actual_flips(preds, y_pred_test_orig)
        )

        compute_info = [
            (
                [
                    "P_test",
                    "RHO_test",
                    "mlr_st_par_test",
                ],
                "y_pred_test",
                test_points_per_region,
            ),
        ]

        for (
            colnames,
            labels_name,
            pts_per_region,
        ) in compute_info:
            res_df[colnames] = res_df[labels_name].apply(
                lambda preds: get_labels_info(
                    np.array(preds),
                    pts_per_region,
                ),
            )

        compute_info = []

        if y_true_test is not None:

            test_pos_y_true_indices, test_pos_points_per_region = get_pos_info_regions(
                y_true_test, test_points_per_region
            )

            compute_info = [
                (
                    [
                        "TP_test",
                        "TPR_test",
                        "mlr_eq_opp_test",
                    ],
                    "y_pred_test",
                    test_pos_y_true_indices,
                    test_pos_points_per_region,
                ),
            ]

            for (
                col_names,
                labels_name,
                indices,
                pts_per_region,
            ) in compute_info:
                res_df[col_names] = res_df[labels_name].apply(
                    lambda labels: get_labels_info(
                        np.array(labels)[indices],
                        pts_per_region,
                    ),
                )

            compute_info = [
                (
                    "actual_flips_test_pos",
                    "y_pred_test",
                    test_pos_y_true_indices,
                    y_pred_test_orig,
                ),
            ]

            for new_col_name, labels_name, indices, exp_y_pred_orig in compute_info:
                res_df[new_col_name] = res_df[labels_name].apply(
                    lambda labels: get_actual_flips(
                        np.array(labels)[indices], exp_y_pred_orig[indices]
                    )
                )

            compute_info = [
                ("accuracy_test", "y_pred_test", y_true_test),
            ]

            for acc_label, labels_name, y_true in compute_info:
                res_df[acc_label] = res_df[labels_name].apply(
                    lambda labels: metrics.accuracy_score(y_true, labels)
                )

            compute_info = [
                ("f1_test", "y_pred_test", y_true_test),
            ]
            for f1_label, labels_name, y_true in compute_info:
                res_df[f1_label] = res_df[labels_name].apply(
                    lambda labels: metrics.f1_score(y_true, labels)
                )

        all_methods_to_results_info[method] = pd.concat(
            [all_methods_to_results_info[method], res_df], ignore_index=True
        )

    amethod = list(all_methods_to_results_info.keys())[0]
    budget_range = all_methods_to_results_info[amethod]["budget"].tolist()

    return all_methods_to_results_info, budget_range


def get_partionings_fairness_loss_all(
    y_pred,
    ids,
    all_partitioning_data_list,
    y_true=None,
    weighted=False,
    score_func=metrics.recall_score,
):
    """
    Computes the fairness loss (disparity) for different partitionings of the dataset.

    This function calculates how much the model's performance varies across different
    partitions of the dataset, using a specified scoring (FairWhere) function. It returns an array
    of fairness loss values for each partitioning.

    Args:
        y_pred (array-like): Predicted labels or scores from the model.
        ids (list of tuples): A list of (index1, index2) pairs, where each pair corresponds
            to a partitioning structure.
        all_partitioning_data_list (list of lists): A list containing partition indices,
            where each inner list corresponds to a partition.
        y_true (array-like, optional): True labels, required if `score_func` needs ground truth
            (e.g., `recall_score`). Defaults to None.
        weighted (bool, optional): If True, the fairness loss is weighted by partition sizes.
            Defaults to False.
        score_func (callable, optional): The scoring function used to evaluate performance,
            defaults to `metrics.recall_score`.

    Returns:
        np.ndarray: An array of fairness loss values for each partitioning.
    """

    if score_func == metrics.recall_score:
        model_mean_score = score_func(y_true, y_pred)
    else:
        model_mean_score = score_func(y_pred)

    fairness_loss_list = []

    for (index1, index2), partitioning in zip(ids, all_partitioning_data_list):
        score_list = []
        weights = []

        for i in range(index1 * index2):
            if len(partitioning[i]) == 0:
                continue

            y_pred_partition = y_pred[partitioning[i]]
            if score_func == metrics.recall_score:
                y_true_partition = y_true[partitioning[i]]
                score_list.append(
                    score_func(y_true_partition, y_pred_partition, zero_division=0)
                )
            else:
                score_list.append(score_func(y_pred_partition))
            weights.append(len(partitioning[i]) / len(y_true))

        if weighted:
            fairness_loss_list.append(
                np.sum(
                    np.abs(model_mean_score - np.array(score_list)) * np.array(weights)
                )
            )
        else:
            fairness_loss_list.append(
                np.mean(np.abs(model_mean_score - np.array(score_list)))
            )

    return np.array(fairness_loss_list)
