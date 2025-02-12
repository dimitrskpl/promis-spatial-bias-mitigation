import gurobipy as gp
import numpy as np


def minimize_promis_opt_obj(
    n_s,
    p_s,
    C=None,
    show_msg=False,
    non_linear=False,
    non_convex_param=-1,
    wlimit=None,
    no_of_threads=0,
    best_obj_stop=None,
    min_pr=None,
    max_pr=None,
    cont_sol=True,
):
    """
    Solve the minimum MLR optimization problem.

    This function uses Gurobi to minimize the MLR, ensuring fair distribution
    of positive labels within regions while adjusting probability ratios.

    Args:
        n_s (list): List of sizes of each region (number of points per region).
        p_s (list): List of counts of positive labels in each region.
        C (int, optional): The maximum number of label flips allowed.
        show_msg (bool, optional): If True, prints Gurobi solver messages.
        non_linear (bool, optional): If True, applies non-linear constraints for log transformations.
        non_convex_param (int, optional): Controls the NonConvex parameter of the Gurobi model.
        wlimit (float, optional): Work limit for the optimization solver.
        no_of_threads (int, optional): Number of threads to use for Gurobi optimization.
        best_obj_stop (float, optional): Stopping criteria based on objective function value.
        min_pr (float, optional): Minimum positive rate constraint.
        max_pr (float, optional): Maximum positive rate constraint.
        cont_sol (bool, optional): If True, uses continuous variables; otherwise, uses integer variables.

    Returns:
        tuple:
            - sol (numpy.ndarray or None): Optimal solution array indicating label changes, or None if infeasible.
            - status (int): Gurobi solver status code.
                - 1: Optimal solution found.
                - 2: Suboptimal solution found.
                - 3: Work limit reached.
                - 4: User objective limit reached.
                - 0: No feasible solution found.
            - obj_val (float or None): Objective function value, or None if no feasible solution is found.

    Raises:
        ValueError: If `min_pr` is equal to `max_pr` and `C` is not an even number.
    """

    if min_pr is not None and min_pr == max_pr and C % 2 != 0:
        raise ValueError(
            "The constraint C should be an even number in order to maintain PR"
        )

    int_var_to_type = gp.GRB.CONTINUOUS if cont_sol else gp.GRB.INTEGER

    n = len(n_s)
    P = sum(p_s)
    N = np.sum(n_s)

    model = gp.Model("minimize_promis_opt")

    delta_p = {}
    abs_delta_p = {}
    for i in range(n):
        delta_p[i] = model.addVar(
            vtype=int_var_to_type, lb=-p_s[i], ub=n_s[i] - p_s[i], name=f"delta_p_{i}"
        )
        abs_delta_p[i] = model.addVar(vtype=int_var_to_type, name=f"abs_delta_p_{i}")

    new_p = {}
    for i in range(n):
        new_p[i] = model.addVar(
            vtype=int_var_to_type, lb=0, ub=n_s[i], name=f"new_p_{i}"
        )

    rho = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="rho")
    rho_in = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="rho_in")
    rho_out = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="rho_out")
    rho_in_opp = model.addVars(
        n, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="rho_in_opp"
    )
    rho_out_opp = model.addVars(
        n, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="rho_out_opp"
    )
    rho_opp = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="rho_opp")
    new_P = model.addVar(vtype=int_var_to_type, lb=0, ub=N, name="new_P")
    stat = model.addVars(n, vtype=gp.GRB.CONTINUOUS, name="stat")
    l0max = model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="l0max")
    l1max = model.addVars(n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="l1max")

    log_rho_in = model.addVars(
        n, lb=-gp.GRB.INFINITY, ub=0, vtype=gp.GRB.CONTINUOUS, name="log_rho_in"
    )
    log_rho_out = model.addVars(
        n, lb=-gp.GRB.INFINITY, ub=0, vtype=gp.GRB.CONTINUOUS, name="log_rho_out"
    )
    log_rho = model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="log_rho")
    log_rho_in_opp = model.addVars(
        n, lb=-gp.GRB.INFINITY, ub=0, vtype=gp.GRB.CONTINUOUS, name="log_rho_in_opp"
    )
    log_rho_out_opp = model.addVars(
        n, lb=-gp.GRB.INFINITY, ub=0, vtype=gp.GRB.CONTINUOUS, name="log_rho_out_opp"
    )
    log_rho_opp = model.addVar(
        lb=-gp.GRB.INFINITY, ub=0, vtype=gp.GRB.CONTINUOUS, name="log_rho_opp"
    )

    min_rho = 1e-10
    max_rho = 1 - 1e-10

    model.addConstr(new_P == P + gp.quicksum(delta_p[j] for j in range(n)))
    model.addConstr(rho == new_P / N)

    if min_pr is not None and min_pr == max_pr:
        delta_p_sum = model.addVar(
            vtype=int_var_to_type, lb=-P, ub=N - P, name="delta_p_sum"
        )
        model.addConstr(delta_p_sum == gp.quicksum(delta_p[i] for i in range(n)))
        model.addConstr(delta_p_sum == 0)
    else:
        if min_pr is not None:
            model.addConstr(rho >= min_pr)
        if max_pr is not None:
            model.addConstr(rho <= max_pr)

    if min_pr is None:
        model.addConstr(rho >= min_rho)
    if max_pr is None:
        model.addConstr(rho <= max_rho)

    ###################################################
    gc_log_rho_in = {}
    gc_log_rho_out = {}
    gc_log_opp_rho_in = {}
    gc_log_opp_rho_out = {}

    for i in range(n):
        model.addConstr(abs_delta_p[i] == gp.abs_(delta_p[i]))

    for i in range(n):

        gc_log_rho_in[i] = model.addGenConstrLog(rho_in[i], log_rho_in[i])
        gc_log_rho_out[i] = model.addGenConstrLog(rho_out[i], log_rho_out[i])

        gc_log_opp_rho_in[i] = model.addGenConstrLog(rho_in_opp[i], log_rho_in_opp[i])
        gc_log_opp_rho_out[i] = model.addGenConstrLog(
            rho_out_opp[i], log_rho_out_opp[i]
        )

    gc_log_rho = model.addGenConstrLog(rho, log_rho)  # , name="log_rho")
    model.addConstr(rho_opp == 1 - rho)
    gc_log_opp_rho = model.addGenConstrLog(rho_opp, log_rho_opp)

    #####################################################
    if non_linear:
        model.update()  # Update the model here
        for i in range(n):
            gc_log_rho_in[i].FuncNonlinear = 1
            gc_log_rho_out[i].FuncNonlinear = 1
            gc_log_opp_rho_in[i].FuncNonlinear = 1
            gc_log_opp_rho_out[i].FuncNonlinear = 1
        gc_log_rho.FuncNonlinear = 1
        gc_log_opp_rho.FuncNonlinear = 1
    ####################################################

    model.addConstr(l0max == new_P * log_rho + (N - new_P) * log_rho_opp)
    for i in range(n):
        model.addConstr(new_p[i] == p_s[i] + delta_p[i])

        model.addConstr(rho_in[i] == new_p[i] / n_s[i])
        model.addConstr(rho_in[i] >= min_rho)
        model.addConstr(rho_in[i] <= max_rho)

        model.addConstr(rho_out[i] == (new_P - new_p[i]) / (N - n_s[i]))
        model.addConstr(rho_out[i] >= min_rho)
        model.addConstr(rho_out[i] <= max_rho)

        model.addConstr(rho_in_opp[i] == 1 - rho_in[i])
        model.addConstr(rho_out_opp[i] == 1 - rho_out[i])

        # l1max = p*math.log(rho_in) + (n-p)*math.log(1-rho_in) + (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out)
        model.addConstr(
            l1max[i]
            == new_p[i] * log_rho_in[i]
            + (n_s[i] - new_p[i]) * log_rho_in_opp[i]
            + (new_P - new_p[i]) * log_rho_out[i]
            + (N - n_s[i] - (new_P - new_p[i])) * log_rho_out_opp[i]
        )

        model.addConstr(stat[i] == l1max[i] - l0max)

    if C is not None:
        model.addConstr(gp.quicksum(abs_delta_p[j] for j in range(n)) <= C)

    model.setObjective(gp.quicksum(stat[i] for i in range(n)), gp.GRB.MINIMIZE)

    # set model params
    model.Params.NonConvex = non_convex_param

    if show_msg:
        model.Params.OutputFlag = 1
    else:
        model.Params.OutputFlag = 0

    if wlimit is not None:
        model.Params.WorkLimit = wlimit

    if best_obj_stop:
        model.Params.BestObjStop = best_obj_stop

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
        all_var_sol_exist = True
        delta_p_optimal = []
        for i in range(n):
            if hasattr(delta_p[i], "X"):  # Check if .X is available
                delta_p_optimal.append(delta_p[i].X)
            elif hasattr(delta_p[i], "Xn"):  # Check if .Xn is available
                delta_p_optimal.append(delta_p[i].Xn)
            else:
                all_var_sol_exist = False
                print(f"Not all var sol exist, stopped in idx: {i}")
                break
        if all_var_sol_exist:
            status = 3
            sol = np.array(delta_p_optimal)
            obj_val = model.ObjVal
        else:
            status = model.status + 10

    elif model.status == gp.GRB.OPTIMAL:
        delta_p_optimal = [delta_p[i].X for i in range(n)]
        sol = delta_p_optimal
        status = 1
        obj_val = model.ObjVal
    elif model.status == gp.GRB.SUBOPTIMAL:
        delta_p_optimal = [delta_p[i].X for i in range(n)]
        sol = delta_p_optimal
        status = 2
        obj_val = model.ObjVal
    elif model.status == gp.GRB.USER_OBJ_LIMIT:
        delta_p_optimal = [delta_p[i].X for i in range(n)]
        sol = delta_p_optimal
        status = 4
        obj_val = model.ObjVal
    else:
        print(f"No optimal or suboptimal solution found!!! Status: {model.status}")
        status = 0
    sol = np.array(sol)
    return sol, status, obj_val


def minimize_promis_opt_obj_overlap(
    labels,
    points_per_region,
    C=None,
    signif_pts_idxs=[],
    show_msg=False,
    non_linear=False,
    non_convex_param=-1,
    wlimit=None,
    no_of_threads=0,
    best_obj_stop=None,
    min_pr=None,
    max_pr=None,
    cont_sol=True,
):
    """
    Solve the minimum MLR optimization problem with interaction constraints.

    This function uses Gurobi to solve an optimization problem that minimizes
    the MLR.

    Args:
        labels (array-like): A binary array where 1 represents a positive label and 0 represents a negative label.
        points_per_region (list of lists): Each sublist contains indices of points belonging to a given region.
        C (int, optional): The maximum number of label flips allowed.
        signif_pts_idxs (list, optional): List of indices of significant points that should be optimized.
        show_msg (bool, optional): If True, prints Gurobi solver messages.
        non_linear (bool, optional): If True, enforces non-linear constraints on logarithmic transformations.
        non_convex_param (int, optional): Controls the NonConvex parameter of the Gurobi model.
        wlimit (float, optional): Work limit for the optimization solver.
        no_of_threads (int, optional): Number of threads to use for Gurobi optimization.
        best_obj_stop (float, optional): Stopping criteria based on objective function value.
        min_pr (float, optional): Minimum positive rate constraint.
        max_pr (float, optional): Maximum positive rate constraint.
        cont_sol (bool, optional): If True, uses continuous variables; otherwise, uses integer variables.

    Returns:
        tuple:
            - sol (numpy.ndarray or None): Optimal solution array indicating label changes, or None if infeasible.
            - status (int): Gurobi solver status code.
                - 1: Optimal solution found.
                - 2: Suboptimal solution found.
                - 3: Work limit reached.
                - 4: User objective limit reached.
                - 0: No feasible solution found.
            - obj_val (float or None): Objective function value, or None if no feasible solution is found.

    Raises:
        ValueError: If `min_pr` is equal to `max_pr` and `C` is not an even number.
    """

    if min_pr is not None and min_pr == max_pr and C % 2 != 0:
        raise ValueError(
            "The constraint C should be an even number in order to maintain PR"
        )

    int_var_to_type = gp.GRB.CONTINUOUS if cont_sol else gp.GRB.INTEGER

    N = len(labels)
    P = np.sum(labels)
    n = len(points_per_region)
    n_s = [len(reg_pts) for reg_pts in points_per_region]
    p_s = [np.sum(labels[reg_pts]) for reg_pts in points_per_region]

    model = gp.Model("minimize_promis_opt_overlap")

    delta_p = {}
    if signif_pts_idxs:
        for i in range(N):
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
        for i in range(N):
            if labels[i] == 1:
                delta_p[i] = model.addVar(
                    vtype=int_var_to_type, lb=-1, ub=0, name=f"delta_p_{i}"
                )
            else:
                delta_p[i] = model.addVar(
                    vtype=int_var_to_type, lb=0, ub=1, name=f"delta_p_{i}"
                )

    new_p = model.addVars(n, vtype=int_var_to_type, name="new_p")
    abs_delta_p = model.addVars(N, vtype=int_var_to_type, name="abs_delta_p")

    rho_in = model.addVars(n, vtype=gp.GRB.CONTINUOUS, name="rho_in")
    rho_out = model.addVars(n, vtype=gp.GRB.CONTINUOUS, name="rho_out")
    rho = model.addVar(vtype=gp.GRB.CONTINUOUS, name="rho")
    rho_in_opp = model.addVars(n, vtype=gp.GRB.CONTINUOUS, name="rho_in_opp")
    rho_out_opp = model.addVars(n, vtype=gp.GRB.CONTINUOUS, name="rho_out_opp")
    rho_opp = model.addVar(vtype=gp.GRB.CONTINUOUS, name="rho_opp")
    new_P = model.addVar(vtype=int_var_to_type, name="new_P")
    stat = model.addVars(n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="stat")
    l0max = model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="l0max")
    l1max = model.addVars(n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="l1max")

    log_rho_in = model.addVars(
        n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="log_rho_in"
    )
    log_rho_out = model.addVars(
        n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="log_rho_out"
    )
    log_rho = model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="log_rho")
    log_rho_in_opp = model.addVars(
        n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="log_rho_in_opp"
    )
    log_rho_out_opp = model.addVars(
        n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="log_rho_out_opp"
    )
    log_rho_opp = model.addVar(
        lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="log_rho_opp"
    )

    min_rho = 1e-10
    max_rho = 1 - 1e-10

    model.addConstr(new_P == P + gp.quicksum(delta_p[j] for j in range(N)))
    model.addConstr(rho == new_P / N)

    if min_pr is not None and min_pr == max_pr:
        delta_p_sum = model.addVar(vtype=int_var_to_type, name="delta_p_sum")
        model.addConstr(delta_p_sum == gp.quicksum(delta_p[i] for i in range(N)))
        model.addConstr(delta_p_sum == 0)
    else:
        if min_pr is not None:
            model.addConstr(rho >= min_pr)
        if max_pr is not None:
            model.addConstr(rho <= max_pr)

    if min_pr is None:
        model.addConstr(rho >= min_rho)
    if max_pr is None:
        model.addConstr(rho <= max_rho)

    ###################################################
    gc_log_rho_in = {}
    gc_log_rho_out = {}
    gc_log_opp_rho_in = {}
    gc_log_opp_rho_out = {}

    for i in range(N):
        model.addConstr(abs_delta_p[i] == gp.abs_(delta_p[i]))

    for i in range(n):

        gc_log_rho_in[i] = model.addGenConstrLog(rho_in[i], log_rho_in[i])
        gc_log_rho_out[i] = model.addGenConstrLog(rho_out[i], log_rho_out[i])

        gc_log_opp_rho_in[i] = model.addGenConstrLog(rho_in_opp[i], log_rho_in_opp[i])
        gc_log_opp_rho_out[i] = model.addGenConstrLog(
            rho_out_opp[i], log_rho_out_opp[i]
        )

    gc_log_rho = model.addGenConstrLog(rho, log_rho)
    model.addConstr(rho_opp == 1 - rho)
    gc_log_opp_rho = model.addGenConstrLog(rho_opp, log_rho_opp)

    #####################################################
    if non_linear:
        model.update()  # Update the model here
        for i in range(n):
            gc_log_rho_in[i].FuncNonlinear = 1
            gc_log_rho_out[i].FuncNonlinear = 1
            gc_log_opp_rho_in[i].FuncNonlinear = 1
            gc_log_opp_rho_out[i].FuncNonlinear = 1
        gc_log_rho.FuncNonlinear = 1
        gc_log_opp_rho.FuncNonlinear = 1
    ####################################################

    model.addConstr(l0max == new_P * log_rho + (N - new_P) * log_rho_opp)
    for i in range(n):
        model.addConstr(
            new_p[i] == p_s[i] + gp.quicksum(delta_p[j] for j in points_per_region[i])
        )

        model.addConstr(rho_in[i] == new_p[i] / n_s[i])
        model.addConstr(rho_in[i] >= min_rho)
        model.addConstr(rho_in[i] <= max_rho)

        model.addConstr(rho_out[i] == (new_P - new_p[i]) / (N - n_s[i]))
        model.addConstr(rho_out[i] >= min_rho)
        model.addConstr(rho_out[i] <= max_rho)

        model.addConstr(rho_in_opp[i] == 1 - rho_in[i])
        model.addConstr(rho_out_opp[i] == 1 - rho_out[i])

        # l1max = p*math.log(rho_in) + (n-p)*math.log(1-rho_in) + (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out)
        model.addConstr(
            l1max[i]
            == new_p[i] * log_rho_in[i]
            + (n_s[i] - new_p[i]) * log_rho_in_opp[i]
            + (new_P - new_p[i]) * log_rho_out[i]
            + (N - n_s[i] - (new_P - new_p[i])) * log_rho_out_opp[i]
        )

        model.addConstr(stat[i] == l1max[i] - l0max)

    if C is not None:
        model.addConstr(gp.quicksum(abs_delta_p[j] for j in range(N)) <= C)

    model.setObjective(gp.quicksum(stat[i] for i in range(n)), gp.GRB.MINIMIZE)

    # set model params
    model.Params.NonConvex = non_convex_param

    if show_msg:
        model.Params.OutputFlag = 1
    else:
        model.Params.OutputFlag = 0

    if wlimit is not None:
        model.Params.WorkLimit = wlimit

    if best_obj_stop:
        model.Params.BestObjStop = best_obj_stop

    model.Params.Threads = no_of_threads

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
        all_var_sol_exist = True
        delta_p_optimal = []
        for i in range(N):
            if hasattr(delta_p[i], "X"):  # Check if .X is available
                delta_p_optimal.append(delta_p[i].X)
            elif hasattr(delta_p[i], "Xn"):  # Check if .Xn is available
                delta_p_optimal.append(delta_p[i].Xn)
            else:
                all_var_sol_exist = False
                print(f"Not all var sol exist, stopped in idx: {i}")
                break
        if all_var_sol_exist:
            status = 3
            sol = np.array(delta_p_optimal)
            obj_val = model.ObjVal
        else:
            status = model.status + 10

    elif model.status == gp.GRB.OPTIMAL:
        delta_p_optimal = [delta_p[i].X for i in range(N)]
        sol = np.array(delta_p_optimal)
        status = 1
        obj_val = model.ObjVal
    elif model.status == gp.GRB.SUBOPTIMAL:
        delta_p_optimal = [delta_p[i].X for i in range(N)]
        sol = np.array(delta_p_optimal)
        status = 2
        obj_val = model.ObjVal
    elif model.status == gp.GRB.USER_OBJ_LIMIT:
        delta_p_optimal = [delta_p[i].X for i in range(N)]
        sol = np.array(delta_p_optimal)
        status = 4
        obj_val = model.ObjVal
    else:
        print(f"No optimal or suboptimal solution found!!! Status: {model.status}")
        status = 0

    return sol, status, obj_val
