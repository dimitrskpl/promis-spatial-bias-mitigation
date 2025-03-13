import pandas as pd
import numpy as np
from sklearn import metrics
from tqdm.notebook import tqdm
from utils.scores import get_sbi
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
    their positive ratio, and SBI

    Args:
        labels (np.ndarray): Binary labels (0s and 1s) representing classifications.
        points_per_region (list): List of lists, where each sublist contains indices of points in a region.

    Returns:
        pd.Series: A series containing:
            - "P" (int): Total count of positive labels.
            - "RHO" (float): Ratio of positive labels to total labels.
            - "sbi" (float): Mean label ratio (SBI) across all regions.
    """

    P = np.sum(labels)
    N = len(labels)
    RHO = P / N
    sbi = get_sbi(labels, points_per_region)

    result = pd.Series(
        [P, RHO, sbi],
        index=[
            "P",
            "RHO",
            "sbi",
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
        - `sbi_st_par`: Mean label ratio.
    - Computes fairness-related metrics SBI for equal opportunity and statistical parity.
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
                    "sbi_st_par_test",
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
                        "sbi_eq_opp_test",
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


def compute_avg_disparity_where_metrics(
    all_methods_to_results_info,
    y_pred,
    y_pred_where,
    y_true,
    points_per_region,
    pos_points_per_region,
    ids,
    partitionings,
    fair_score_func,
    set_label="test",
):
    """
    Computes disparity for all given predictions plus SBI and performance metrics for the "FairWhere" method's predictions.

    Parameters:
    - all_methods_to_results_info (dict): Dictionary mapping method names to DataFrames containing results.
    - y_pred (array-like): Initial predictions for all samples.
    - y_pred_where (array-like): Predictions computed by the "FairWhere" method.
    - y_true (array-like): Ground truth labels.
    - points_per_region (array-like): Number of points in each region.
    - pos_points_per_region (array-like): Number of positive points in each region.
    - ids (array-like): Identifiers for the partitionings.
    - partitionings (array-like): Different partitioning schemes for fairness analysis.
    - fair_score_func (function): Function used to compute fairness scores (statistical parity/equal opportunity).
    - set_label (str, optional): Label for the dataset partition being evaluated (e.g., "test"). Defaults to "test".

    Returns:
    - tuple: Contains the following computed metrics:
        - all_methods_to_results_info (dict): Updated dictionary with fairness loss values.
        - P_where (int): Number of positive predictions in the subset.
        - RHO_where (float): Ratio of positive predictions in the subset.
        - TP_where (int): Number of true positive predictions in the subset.
        - TPR_where (float): True positive rate for the subset.
        - sbi_where_st_par (float): Mean likelihood ratio for statistical parity.
        - sbi_where_eq_opp (float): Mean likelihood ratio for equal opportunity.
        - acc_where (float): Accuracy of predictions in the subset.
        - f1_where (float): F1-score of predictions in the subset.
        - init_fairness_loss_list (list): Fairness loss for each partition without weighting.
        - init_fairness_loss_list_weighted (list): Fairness loss for each partition with weighting.
        - init_fairness_loss_sum (float): Summed fairness loss across partitions without weighting.
        - init_fairness_loss_sum_weighted (float): Summed fairness loss across partitions with weighting.
        - where_fairness_loss_list (list): Fairness loss for each partition in the evaluated subset (without weighting).
        - where_fairness_loss_list_weighted (list): Fairness loss for each partition in the evaluated subset (with weighting).
        - where_fairness_loss_sum (float): Summed fairness loss for the evaluated subset (without weighting).
        - where_fairness_loss_weighted_sum (float): Summed fairness loss for the evaluated subset (with weighting).
    """

    pos_y_true_indices = np.where(y_true == 1)[0]

    P_where = np.sum(y_pred_where)
    N_where = len(y_pred_where)
    RHO_where = P_where / N_where
    TP_where = np.sum(y_pred_where[pos_y_true_indices])
    TPR_where = TP_where / len(pos_y_true_indices)

    # SBI
    sbi_where_st_par = get_sbi(y_pred_where, points_per_region)
    sbi_where_eq_opp = get_sbi(y_pred_where[pos_y_true_indices], pos_points_per_region)

    # Accuracy
    acc_where = metrics.accuracy_score(y_true, y_pred_where)
    f1_where = metrics.f1_score(y_true, y_pred_where)

    where_fairness_loss_list = get_partionings_fairness_loss_all(
        y_pred_where,
        ids,
        partitionings,
        y_true,
        weighted=False,
        score_func=fair_score_func,
    )

    where_fairness_loss_sum = np.sum(where_fairness_loss_list)

    init_fairness_loss_list = get_partionings_fairness_loss_all(
        y_pred,
        ids,
        partitionings,
        y_true,
        weighted=False,
        score_func=fair_score_func,
    )

    init_fairness_loss_sum = np.sum(init_fairness_loss_list)

    # fair loss score weighed per partitioning
    where_fairness_loss_list_weighted = get_partionings_fairness_loss_all(
        y_pred_where,
        ids,
        partitionings,
        y_true,
        weighted=True,
        score_func=fair_score_func,
    )

    where_fairness_loss_weighted_sum = np.sum(where_fairness_loss_list_weighted)

    init_fairness_loss_list_weighted = get_partionings_fairness_loss_all(
        y_pred,
        ids,
        partitionings,
        y_true,
        weighted=True,
        score_func=fair_score_func,
    )

    init_fairness_loss_sum_weighted = np.sum(init_fairness_loss_list_weighted)

    for method, res_df in all_methods_to_results_info.items():
        res_df[f"fair_loss_list_{set_label}"] = res_df[f"y_pred_{set_label}"].apply(
            lambda x: get_partionings_fairness_loss_all(
                x,
                ids,
                partitionings,
                y_true,
                weighted=False,
                score_func=fair_score_func,
            )
        )

        res_df[f"fair_loss_sum_{set_label}"] = res_df[
            f"fair_loss_list_{set_label}"
        ].apply(lambda x: np.sum(x))

        res_df[f"fair_loss_list_weighted_{set_label}"] = res_df[
            f"y_pred_{set_label}"
        ].apply(
            lambda x: get_partionings_fairness_loss_all(
                x,
                ids,
                partitionings,
                y_true,
                weighted=True,
                score_func=fair_score_func,
            )
        )

        res_df[f"fair_loss_sum_weighted_{set_label}"] = res_df[
            f"fair_loss_list_weighted_{set_label}"
        ].apply(lambda x: np.sum(x))

        all_methods_to_results_info[method] = res_df

    return (
        all_methods_to_results_info,
        P_where,
        RHO_where,
        TP_where,
        TPR_where,
        sbi_where_st_par,
        sbi_where_eq_opp,
        acc_where,
        f1_where,
        init_fairness_loss_list,
        init_fairness_loss_list_weighted,
        init_fairness_loss_sum,
        init_fairness_loss_sum_weighted,
        where_fairness_loss_list,
        where_fairness_loss_list_weighted,
        where_fairness_loss_sum,
        where_fairness_loss_weighted_sum,
    )


def get_audit_regions_name(x):
    if "non_overlap" in x:
        return "Non-Overlapping KMeans"
    elif "overlap" in x:
        return "Overlapping KMeans"
    else:
        return "Overlapping Partitionings"


def get_clf_name(x):
    if x == "dnn":
        return "DNN"
    elif x == "xgb":
        return "XGB"
    elif x.startswith("semi_synthetic"):
        return "Unfair by Design"
    else:
        return "-"


def compute_max_budget_info(
    all_methods_to_results_info,
    budget_range,
    dataset_name,
    clf_name,
    partioning_type_name,
    fairness_notion,
    points_per_region,
    init_sbi_st_par,
    init_sbi_eq_opp,
    init_stats_st_par,
    init_stats_eq_opp,
    where_fit_time,
    y_pred_where,
    y_true,
    sbi_where_st_par,
    sbi_where_eq_opp,
    init_f1,
    f1_where_test,
    init_acc,
    init_fairness_loss_sum,
    where_fairness_loss_sum,
):
    """
    Compute and compile final budget-based performance, fairness, and timing results into a pandas DataFrame.

    This function aggregates results from different methods (passed as `all_methods_to_results_info`)
    for a maximum budget of label flips, and appends those results along with initial and optional
    "FairWhere" baseline outcomes to form a consolidated report. The report includes performance
    metrics (accuracy or F1), fairness metrics (SBI and related statistics), total time taken,
    and metadata describing the dataset, classifier, and fairness notion.

    Args:
        all_methods_to_results_info (dict):
            A dictionary mapping method names (str) to pandas DataFrames of experiment results.
            Each DataFrame must contain columns including at least:
                - "budget": The number of flips used.
                - "time": The time taken for the method when using a certain budget.
                - "sbi_st_par_test" and/or "sbi_eq_opp_test": Measured SBI values for the final test set,
                  depending on the fairness notion.
                - "y_pred_test": The predicted labels on the test set.
                - "fair_loss_sum_test": The sum of fairness losses on the test set (for DNN).
                - "<performance_label>_test": The performance metric (e.g., F1 or accuracy) on the test set.
        budget_range (list):
            A list of possible budget values for label flipping. The maximum of this range
            (i.e., `budget_range[-1]`) is used as the final budget in the report.
        dataset_name (str):
            Name of the dataset (e.g., "crime" or any other). Used to label the final DataFrame.
        clf_name (str):
            Name of the classifier (e.g., "dnn", "rf"). Determines which performance metric
            is used (F1 for DNN, accuracy otherwise).
        partioning_type_name (str):
            The type of partitioning or region definition for the dataset. Used to label the
            final DataFrame (via `get_audit_regions_name`).
        fairness_notion (str):
            The type of fairness metric used, either "statistical_parity" or "equal_opportunity".
        points_per_region (list or np.ndarray):
            A list or array indicating how many points fall into each region of interest
            (used for calculating SBI).
        init_sbi_st_par (float):
            The initial (pre-flipping) SBI under statistical parity.
        init_sbi_eq_opp (float):
            The initial (pre-flipping) SBI under equal opportunity.
        init_stats_st_par (float):
            The initial (pre-flipping) statistics under statistical parity.
        init_stats_eq_opp (float):
            The initial (pre-flipping) statistics under equal opportunity.
        where_fit_time (float):
            The runtime taken by the "FairWhere" method (applicable only if `clf_name == "dnn"`).
        y_pred_where (array-like):
            The predictions from "FairWhere" on the test set (applicable only if `clf_name == "dnn"`).
        y_true (array-like):
            Ground truth labels for the test set. Used for measuring final performance metrics
            if it is not None.
        sbi_where_st_par (float):
            The SBI achieved by "FairWhere" under statistical parity (if `clf_name == "dnn"`).
        sbi_where_eq_opp (float):
            The SBI achieved by "FairWhere" under equal opportunity (if `clf_name == "dnn"`).
        init_f1 (float):
            The initial F1 score prior to any flipping (applicable if `clf_name` starts with "dnn").
        f1_where_test (float):
            The F1 score achieved by "FairWhere" on the test set (if `clf_name == "dnn"`).
        init_acc (float):
            The initial accuracy score prior to any flipping (applicable if `clf_name` does not start with "dnn").
        init_fairness_loss_sum (float):
            The initial (pre-flipping) sum of fairness losses (used for DNN).
        where_fairness_loss_sum (float):
            The sum of fairness losses after applying "FairWhere" (used for DNN).

    Returns:
        pandas.DataFrame:
            A DataFrame containing the compiled results. Columns include:

            - "Budget": The budget values used (0 for the initial, and maximum budget for each method).
            - "Method": The names of the methods (plus an "init" row for the initial state, and
              optional "FairWhere" row if `clf_name` is "dnn").
            - "Time": The time taken for each method at the max budget (and None for the initial row).
            - "Accuracy": The accuracy scores (or None if classifier is DNN).
            - "F1": The F1 scores (or None if classifier is not DNN).
            - "SBI": The final measured SBI for each method at max budget (and the initial value).
            - "SBIr": Supporting statistics for the SBI calculation per method.
            - "MeanDev": The fairness loss sums (if classifier is DNN), else None.
            - "Dataset": The dataset name (e.g., "Crime" or "LAR").
            - "Classifier": Human-readable classifier name (derived from `clf_name` via `get_clf_name`).
            - "Audit Regions": A label describing the partitioning, derived from `partioning_type_name`.
            - "Fairness Notion": A label describing the fairness notion used."
    """

    pos_y_true_indices, pos_points_per_region = get_pos_info_regions(
        y_true, points_per_region
    )
    methods = ["init"]
    final_sbis_st_par_test = [init_sbi_st_par]
    final_sbis_eq_opp_test = [init_sbi_eq_opp]
    final_times = [None]
    n_flips_ = budget_range[-1]
    budget_list = [0] + [n_flips_] * len(all_methods_to_results_info)
    final_stats_st_par_test_list = [init_stats_st_par]
    final_stats_eq_opp_test_list = [init_stats_eq_opp]
    final_performance_label = "f1" if clf_name.startswith("dnn") else "accuracy"
    final_performance_test_list = (
        [init_f1] if clf_name.startswith("dnn") else [init_acc]
    )
    final_fair_score_test_list = [init_fairness_loss_sum]
    for method, exp_res_df in all_methods_to_results_info.items():
        if fairness_notion == "statistical_parity":
            sbi_st_par_test = exp_res_df[exp_res_df["budget"] == n_flips_][
                "sbi_st_par_test"
            ].tolist()[0]
            final_sbis_st_par_test.append(sbi_st_par_test)
            y_test_pred = exp_res_df[exp_res_df["budget"] == n_flips_][
                "y_pred_test"
            ].tolist()[0]
            _, final_stats_st_par_test = get_sbi(
                y_test_pred, points_per_region, with_stats=True
            )
            final_stats_st_par_test_list.append(final_stats_st_par_test)
            if clf_name.startswith("dnn"):
                final_fair_score_test_list.append(
                    exp_res_df[exp_res_df["budget"] == n_flips_][
                        "fair_loss_sum_test"
                    ].tolist()[0]
                )
        else:
            sbi_eq_opp_test = exp_res_df[exp_res_df["budget"] == n_flips_][
                "sbi_eq_opp_test"
            ].tolist()[0]
            y_test_pred = exp_res_df[exp_res_df["budget"] == n_flips_][
                "y_pred_test"
            ].tolist()[0]
            y_test_pred_pos = y_test_pred[pos_y_true_indices]
            _, final_stats_eq_opp_test = get_sbi(
                y_test_pred_pos, pos_points_per_region, with_stats=True
            )
            final_stats_eq_opp_test_list.append(final_stats_eq_opp_test)

            if clf_name.startswith("dnn"):
                final_fair_score_test_list.append(
                    exp_res_df[exp_res_df["budget"] == n_flips_][
                        "fair_loss_sum_test"
                    ].tolist()[0]
                )

        final_flip_time = exp_res_df[exp_res_df["budget"] == n_flips_]["time"].tolist()[
            0
        ]
        final_times.append(final_flip_time)
        methods.append(method)
        if y_true is not None:
            final_performance_test_list.append(
                exp_res_df[exp_res_df["budget"] == n_flips_][
                    f"{final_performance_label}_test"
                ].tolist()[0]
            )
        else:
            final_performance_test_list.append(None)

        if fairness_notion == "equal_opportunity":
            final_sbis_eq_opp_test.append(sbi_eq_opp_test)

    final_results = {
        "Budget": budget_list,
        "Method": methods,
        "Time": final_times,
    }
    if final_performance_label == "f1":
        final_results["Accuracy"] = [None] * len(final_performance_test_list)
        final_results["F1"] = final_performance_test_list
    else:
        final_results["Accuracy"] = final_performance_test_list
        final_results["F1"] = [None] * len(final_performance_test_list)

    if fairness_notion == "statistical_parity":
        final_results["SBI"] = final_sbis_st_par_test
        final_results["Statistics"] = final_stats_st_par_test_list
    else:
        final_results["SBI"] = final_sbis_eq_opp_test
        final_results["Statistics"] = final_stats_eq_opp_test_list

    if clf_name == "dnn":
        final_results["Method"].append("FairWhere")
        final_results["Budget"].append(n_flips_)
        final_results["Time"].append(where_fit_time)
        final_results["F1"].append(f1_where_test)
        final_results["Accuracy"].append(None)
        if fairness_notion == "statistical_parity":
            final_results["SBI"].append(sbi_where_st_par)
            _, final_stats_st_par_test = get_sbi(
                y_pred_where, points_per_region, with_stats=True
            )
            final_results["Statistics"].append(final_stats_st_par_test)
            final_fair_score_test_list.append(where_fairness_loss_sum)
            final_results["MeanDev"] = final_fair_score_test_list

        else:
            final_results["SBI"].append(sbi_where_eq_opp)
            _, final_stats_eq_opp_test = get_sbi(
                y_pred_where[pos_y_true_indices],
                pos_points_per_region,
                with_stats=True,
            )
            final_results["Statistics"].append(final_stats_eq_opp_test)
            final_fair_score_test_list.append(where_fairness_loss_sum)
            final_results["MeanDev"] = final_fair_score_test_list
    else:
        final_results["MeanDev"] = [None] * len(final_performance_test_list)

    final_results["Dataset"] = "Crime" if dataset_name == "crime" else "LAR"
    final_results["Classifier"] = get_clf_name(clf_name)
    final_results["Audit Regions"] = get_audit_regions_name(partioning_type_name)
    final_results["Fairness Notion"] = (
        "Statistical Parity"
        if fairness_notion == "statistical_parity"
        else "Equal Opportunity"
    )

    final_results_df = pd.DataFrame(final_results)

    return final_results_df
