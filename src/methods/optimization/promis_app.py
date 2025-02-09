import gurobipy as gp
import numpy as np


def minimize_promis_app_obj(
    n_s,
    p_s,
    weights,
    C=None,
    show_msg=False,
    wlimit=None,
    no_of_threads=0,
    non_convex_param=0,
    min_pr=None,
    max_pr=None,
    cont_sol=True,
):
    """
    Solves an optimization problem to minimize the in-out disparity in classification fairness.

    Args:
        n_s (list): List containing the number of samples in each region.
        p_s (list): List containing the number of positive labels in each region.
        weights (list): List of weights assigned to each region.
        C (int, optional): Constraint on the maximum number of label changes allowed. Defaults to None.
        show_msg (bool, optional): Whether to display solver messages. Defaults to False.
        wlimit (int, optional): Work limit for Gurobi optimization. Defaults to None.
        no_of_threads (int, optional): Number of threads for parallel computation. Defaults to 0.
        non_convex_param (int, optional): Parameter for handling non-convexity in optimization. Defaults to 0.
        min_pr (float, optional): Minimum proportion constraint for fairness. Defaults to None.
        max_pr (float, optional): Maximum proportion constraint for fairness. Defaults to None.
        cont_sol (bool, optional): Whether to use continuous (True) or integer (False) optimization. Defaults to True.

    Returns:
        tuple:
            - np.ndarray: Optimal solution representing the changes to be applied to labels.
            - int: Solver status code.
            - float: Objective value of the solution.

    The function:
    - Defines an optimization model using Gurobi.
    - Ensures fairness constraints by adjusting the proportion of positive labels in each region.
    - Balances in-region (`rho_in`) and out-region (`rho_out`) label proportions.
    - Returns the optimal solution, solver status, and objective value.
    """

    if min_pr is not None and min_pr == max_pr and C % 2 != 0:
        raise ValueError(
            "The constraint C should be an even number in order to maintain PR"
        )

    int_var_to_type = gp.GRB.CONTINUOUS if cont_sol else gp.GRB.INTEGER

    P = sum(p_s)
    N = sum(n_s)
    total_regs = len(n_s)
    if weights is None:
        weights = [1] * len(n_s)

    model = gp.Model("promis_app")

    delta_p = {}
    abs_delta_p = {}
    for i in range(total_regs):
        delta_p[i] = model.addVar(
            vtype=int_var_to_type, lb=-p_s[i], ub=n_s[i] - p_s[i], name=f"delta_p_{i}"
        )
        abs_delta_p[i] = model.addVar(vtype=int_var_to_type, name=f"abs_delta_p_{i}")

    new_p = {}
    new_p_out = {}
    for i in range(total_regs):
        new_p[i] = model.addVar(
            vtype=int_var_to_type, lb=0, ub=n_s[i], name=f"new_p_{i}"
        )
        new_p_out[i] = model.addVar(
            vtype=int_var_to_type, lb=0, ub=N - n_s[i], name=f"new_p_out_{i}"
        )

    rho = model.addVar(vtype=gp.GRB.CONTINUOUS, name="rho")

    model.addConstr(rho == (P + gp.quicksum(delta_p[i] for i in range(total_regs))) / N)

    if min_pr is not None and min_pr == max_pr:
        delta_p_sum = model.addVar(vtype=int_var_to_type, name="delta_p_sum")
        model.addConstr(
            delta_p_sum == gp.quicksum(delta_p[i] for i in range(total_regs))
        )
        model.addConstr(delta_p_sum == 0)
    else:
        if min_pr is not None:
            model.addConstr(rho >= min_pr)
        if max_pr is not None:
            model.addConstr(rho <= max_pr)

    rho_in = model.addVars(
        total_regs, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="rho_in"
    )
    rho_out = model.addVars(
        total_regs, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="rho_out"
    )
    newP = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=N, name="newP")

    in_out_diff = {}
    in_out_diff_abs = {}
    for i in range(total_regs):
        in_out_diff[i] = model.addVar(
            vtype=gp.GRB.CONTINUOUS,
            lb=-weights[i],
            ub=weights[i],
            name=f"in_out_diff_{i}",
        )
        in_out_diff_abs[i] = model.addVar(
            vtype=gp.GRB.CONTINUOUS,
            lb=0,
            ub=weights[i],
            name=f"in_out_diff_abs_{i}",
        )

    model.addConstr(newP == P + gp.quicksum(delta_p[i] for i in range(total_regs)))
    for i in range(total_regs):
        model.addConstr(new_p[i] == p_s[i] + delta_p[i])
        model.addConstr(new_p_out[i] == newP - new_p[i])
        model.addConstr(rho_in[i] == new_p[i] / n_s[i])
        model.addConstr(rho_out[i] == new_p_out[i] / (N - n_s[i]))
        model.addConstr(in_out_diff[i] == weights[i] * (rho_in[i] - rho_out[i]))
        model.addConstr(in_out_diff_abs[i] == gp.abs_(in_out_diff[i]))

    for i in range(total_regs):
        model.addConstr(abs_delta_p[i] == gp.abs_(delta_p[i]))

    if C is not None:
        model.addConstr(gp.quicksum(abs_delta_p[i] for i in range(total_regs)) <= C)

    model.setObjective(
        gp.quicksum(in_out_diff_abs[i] for i in range(total_regs)), gp.GRB.MINIMIZE
    )

    if show_msg:
        model.Params.OutputFlag = 1
    else:
        model.Params.OutputFlag = 0

    model.Params.NonConvex = non_convex_param

    if wlimit is not None:
        model.Params.WorkLimit = wlimit

    model.params.Threads = no_of_threads

    model.optimize()

    sol = None
    status = -1
    obj_val = None
    if model.status == gp.GRB.INFEASIBLE:
        model.computeIIS()
        model.write("model.ilp")
        print("Infeasible model. Written to model.ilp")
    elif model.status == gp.GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif model.status == gp.GRB.WORK_LIMIT:
        delta_p_optimal = [delta_p[i].X for i in range(total_regs)]
        sol = np.array(delta_p_optimal)
        status = 3
        obj_val = model.ObjVal
    elif model.status == gp.GRB.OPTIMAL:
        delta_p_optimal = [delta_p[i].X for i in range(total_regs)]
        sol = np.array(delta_p_optimal)
        status = 1
        obj_val = model.ObjVal
    elif model.status == gp.GRB.SUBOPTIMAL:
        delta_p_optimal = [delta_p[i].X for i in range(total_regs)]
        sol = np.array(delta_p_optimal)
        status = 2
        obj_val = model.ObjVal
    elif model.status == gp.GRB.USER_OBJ_LIMIT:
        delta_p_optimal = [delta_p[i].X for i in range(total_regs)]
        sol = np.array(delta_p_optimal)
        status = 4
        obj_val = model.ObjVal
    else:
        print(f"No optimal or suboptimal solution found. Status: {model.status}")
        status = 0

    return sol, status, obj_val


def minimize_promis_app_obj_overlap(
    labels,
    points_per_region,
    C=None,
    signif_pts_idxs=[],
    show_msg=False,
    wlimit=None,
    no_of_threads=0,
    non_convex_param=0,
    weights=None,
    min_pr=None,
    max_pr=None,
    cont_sol=True,
):
    """
    Solves an optimization problem to minimize the in-out disparity in classification fairness.

    Args:
        labels (np.ndarray): Array of binary classification labels (0s and 1s).
        points_per_region (list): A list of lists, where each sublist contains indices of points in a region.
        C (int, optional): Constraint on the maximum number of label changes allowed. Defaults to None.
        signif_pts_idxs (list, optional): Indices of significant points that can be changed. Defaults to [].
        show_msg (bool, optional): Whether to display solver messages. Defaults to False.
        wlimit (int, optional): Work limit for Gurobi optimization. Defaults to None.
        no_of_threads (int, optional): Number of threads for parallel computation. Defaults to 0.
        non_convex_param (int, optional): Parameter for handling non-convexity in optimization. Defaults to 0.
        weights (list, optional): List of weights for different regions. Defaults to None.
        min_pr (float, optional): Minimum proportion constraint for fairness. Defaults to None.
        max_pr (float, optional): Maximum proportion constraint for fairness. Defaults to None.
        cont_sol (bool, optional): Whether to use continuous (True) or integer (False) optimization. Defaults to True.

    Returns:
        tuple:
            - np.ndarray: Optimal solution representing the changes to be applied to labels.
            - int: Solver status code.
            - float: Objective value of the solution.

    The function:
    - Defines an optimization model in Gurobi.
    - Applies fairness constraints by adjusting the proportion of positive labels.
    - Ensures that in-region (`rho_in`) and out-region (`rho_out`) ratios are balanced.
    - Returns the optimal changes to labels, solver status, and objective value.
    """

    if min_pr is not None and min_pr == max_pr and C % 2 != 0:
        raise ValueError(
            "The constraint C should be an even number in order to maintain PR"
        )

    int_var_to_type = gp.GRB.CONTINUOUS if cont_sol else gp.GRB.INTEGER
    total_points = len(labels)
    N = len(labels)
    P = np.sum(labels)
    total_regs = len(points_per_region)
    n_s = [len(reg_pts) for reg_pts in points_per_region]
    p_s = [np.sum(labels[reg_pts_indexes]) for reg_pts_indexes in points_per_region]
    if weights is None:
        weights = [1] * len(n_s)

    model = gp.Model("promis_app_overlap")

    delta_p = {}
    if signif_pts_idxs:
        for i in range(total_points):
            if i in signif_pts_idxs:
                if labels[i] == 1:
                    delta_p[i] = model.addVar(
                        vtype=int_var_to_type, lb=-1, ub=0, name=f"delta_p_{i}"
                    )
                else:
                    delta_p[i] = model.addVar(
                        vtype=int_var_to_type, lb=0, ub=1, name=f"delta_p_{i}"
                    )
            else:
                delta_p[i] = model.addVar(
                    vtype=int_var_to_type, lb=0, ub=0, name=f"delta_p_{i}"
                )
    else:
        for i in range(total_points):
            if labels[i] == 1:
                delta_p[i] = model.addVar(
                    vtype=int_var_to_type, lb=-1, ub=0, name=f"delta_p_{i}"
                )
            else:
                delta_p[i] = model.addVar(
                    vtype=int_var_to_type, lb=0, ub=1, name=f"delta_p_{i}"
                )

    new_p = model.addVars(total_regs, vtype=int_var_to_type, name="new_p")
    new_p_out = model.addVars(total_regs, vtype=int_var_to_type, name="new_p_out")
    abs_delta_p = model.addVars(total_points, vtype=int_var_to_type, name="abs_delta_p")
    newP = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=N, name="newP")
    rho = model.addVar(vtype=gp.GRB.CONTINUOUS, name="rho")

    rho_in = model.addVars(total_regs, vtype=gp.GRB.CONTINUOUS, name="rho_in")
    rho_out = model.addVars(total_regs, vtype=gp.GRB.CONTINUOUS, name="rho_out")
    in_out_diff = model.addVars(
        total_regs, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="in_overall_diff"
    )
    in_out_diff_abs = model.addVars(
        total_regs, vtype=gp.GRB.CONTINUOUS, name="in_out_diff_abs"
    )

    model.addConstr(
        rho == (P + gp.quicksum(delta_p[i] for i in range(total_points))) / total_points
    )

    if min_pr is not None and min_pr == max_pr:
        delta_p_sum = model.addVar(vtype=int_var_to_type, name="delta_p_sum")
        model.addConstr(
            delta_p_sum == gp.quicksum(delta_p[i] for i in range(total_points))
        )
        model.addConstr(delta_p_sum == 0)
    else:
        if min_pr is not None:
            model.addConstr(rho >= min_pr)
        if max_pr is not None:
            model.addConstr(rho <= max_pr)

    model.addConstr(newP == P + gp.quicksum(delta_p[i] for i in range(total_regs)))
    for i in range(total_regs):
        region_indexes = points_per_region[i]
        model.addConstr(
            new_p[i] == p_s[i] + gp.quicksum(delta_p[j] for j in region_indexes)
        )
        model.addConstr(new_p_out[i] == newP - new_p[i])
        model.addConstr(rho_in[i] == new_p[i] / n_s[i])
        model.addConstr(rho_out[i] == new_p_out[i] / (N - n_s[i]))
        model.addConstr(in_out_diff[i] == weights[i] * (rho_in[i] - rho_out[i]))
        model.addConstr(in_out_diff_abs[i] == gp.abs_(in_out_diff[i]))

    for i in range(total_points):
        model.addConstr(abs_delta_p[i] == gp.abs_(delta_p[i]))

    if C is not None:
        model.addConstr(gp.quicksum(abs_delta_p[i] for i in range(total_points)) <= C)

    model.setObjective(
        gp.quicksum(in_out_diff_abs[i] for i in range(total_regs)), gp.GRB.MINIMIZE
    )

    # set model params

    if show_msg:
        model.Params.OutputFlag = 1
    else:
        model.Params.OutputFlag = 0

    model.Params.NonConvex = non_convex_param

    if wlimit is not None:
        model.Params.WorkLimit = wlimit

    model.params.Threads = no_of_threads

    model.optimize()

    # check solution
    sol = None
    status = -1
    obj_val = None
    if model.status == gp.GRB.INFEASIBLE:
        model.computeIIS()
        model.write("model.ilp")
        print("Infeasible model. Written to model.ilp")
    elif model.status == gp.GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif model.status == gp.GRB.WORK_LIMIT:
        status = 3
        delta_p_optimal = [delta_p[i].X for i in range(total_points)]
        sol = np.array(delta_p_optimal)
        obj_val = model.ObjVal
    elif model.status == gp.GRB.OPTIMAL:
        status = 1
        delta_p_optimal = [delta_p[i].X for i in range(total_points)]
        sol = np.array(delta_p_optimal)
        obj_val = model.ObjVal
    elif model.status == gp.GRB.SUBOPTIMAL:
        delta_p_optimal = [delta_p[i].X for i in range(total_points)]
        sol = np.array(delta_p_optimal)
        obj_val = model.ObjVal
        status = 2
    elif model.status == gp.GRB.USER_OBJ_LIMIT:
        delta_p_optimal = [delta_p[i].X for i in range(total_points)]
        sol = np.array(delta_p_optimal)
        obj_val = model.ObjVal
        status = 4
    else:
        print(f"No optimal or suboptimal solution found. Status: {model.status}")
        status = 0

    return sol, status, obj_val
