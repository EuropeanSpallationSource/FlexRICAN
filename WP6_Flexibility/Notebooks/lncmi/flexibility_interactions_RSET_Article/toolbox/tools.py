import time as tm
import pulp
import pandas as pd


def get_lp_stats(model, solve_time=None):
    variables = model.variables()

    n_true_binary = 0     
    n_binary_like = 0     
    n_integer_other = 0   
    n_continuous = 0

    for v in variables:
        if v.cat == pulp.LpBinary:
            n_true_binary += 1
        elif v.cat == pulp.LpInteger:
            if v.lowBound == 0 and v.upBound == 1:
                n_binary_like += 1
            else:
                n_integer_other += 1
        else:  
            n_continuous += 1

    stats = {
        "n_variables_total": len(variables),
        "n_binary": n_true_binary + n_binary_like,
        "n_binary_declared": n_true_binary,
        "n_binary_by_bounds": n_binary_like,
        "n_integer_general": n_integer_other,
        "n_continuous": n_continuous,
        "n_constraints": len(model.constraints),
        "status": pulp.LpStatus[model.status],
        "solve_time_s": solve_time,
    }
    return stats