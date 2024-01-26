#  ___________________________________________________________________________
#
#  Variable Elimination: Research code for variable elimination in NLPs
#
#  Copyright (c) 2023. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001
#  for Los Alamos National Laboratory (LANL), which is operated by Triad
#  National Security, LLC for the U.S. Department of Energy/National Nuclear
#  Security Administration. All rights in the program are reserved by Triad
#  National Security, LLC, and the U.S. Department of Energy/National Nuclear
#  Security Administration. The Government is granted for itself and others
#  acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
#  in this material to reproduce, prepare derivative works, distribute copies
#  to the public, perform publicly and display publicly, and to permit others
#  to do so.
#
#  This software is distributed under the 3-clause BSD license.
#  ___________________________________________________________________________

from pyomo.core.expr import value as pyo_value
from pyomo.core.base.var import Var
# Note that this adds an IDAES dependency
from idaes.core.util.model_statistics import large_residuals_set


def validate_solution(model, tolerance=0.0):
    violated_cons_reduced = large_residuals_set(model, tol=tolerance)

    vars_violating_bounds = []
    for var in model.component_data_objects(Var):
        if var.value is None:
            continue
        if var.ub is not None:
            ub_diff = pyo_value(var.value - var.ub)
            if ub_diff > tolerance:
                vars_violating_bounds.append((var, var.ub, ub_diff))
        if var.lb is not None:
            lb_diff = pyo_value(var.value - var.lb)
            if lb_diff < - tolerance:
                vars_violating_bounds.append((var, var.lb, lb_diff))

    if violated_cons_reduced:
        print("WARNING: Constraints in the reduced-space model are violated")

    if vars_violating_bounds:
        print("WARNING: There are variables violating their bounds")

    violations = (violated_cons_reduced, vars_violating_bounds)
    valid = not any(violations)

    return valid, violations
