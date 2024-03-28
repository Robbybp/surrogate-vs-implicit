#  ___________________________________________________________________________
#
#  Surrogate vs. Implicit: Experiments comparing nonlinear optimization
#  formulations
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
#  __________________________________________________________________________

import pyomo.environ as pyo
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from svi.auto_thermal_reformer.fullspace_flowsheet import make_simulation_model
from svi.validate import validate_solution
import svi.auto_thermal_reformer.config as config
from svi.external import add_external_function_libraries_to_environment
import pandas as pd


"""Script for validating a single set of inputs from a CSV file of sweep
results

"""


def simulate_model(m, tee=True):
    # Use scipy.fsolve (Powell trust region) as it seems a little more
    # reliable than Ipopt, and is not slower for small systems, despite
    # SciPy's implementation not exploiting sparsity.
    #
    # We solve simulation problems for validation to a tighter tolerance to try
    # and minimize the variance in our analysis due to this part of the process
    solver = pyo.SolverFactory("scipy.fsolve")
    solver.options["xtol"] = 1e-12
    #solver.options["tol"] = 1e-10
    solve_strongly_connected_components(
        m,
        solver=solver,
        use_calc_var=False,
        # Hard-code tee=False for sub-solvers
        solve_kwds=dict(tee=False),
    )
    postsolver = pyo.SolverFactory("ipopt")
    # Can't figure out how to set options...
    #postsolver = pyo.SolverFactory("ipopt_v2")
    postsolver.options["tol"] = 1e-9
    postsolver.options["print_user_options"] = "yes"
    res = postsolver.solve(m, tee=True)
    return res


def validate_model_simulation(m, feastol=0.0):
    try:
        simulate_model(m)
    except (ValueError, RuntimeError) as err:
        # We sometimes get ValueErrors when Ipopt throws an error, even when the
        # solve is acceptable. In this case, the solution doesn't get loaded into
        # the model, so it is arguable whether we should continue...
        print("WARNING: Got an error:")
        print(err)
        print("WARNING: Continuing with validation despite error")
        pass
    valid, violations = validate_solution(m, tolerance=feastol)
    return valid, violations


def main():
    argparser = config.get_argparser()
    argparser.add_argument(
        "experiment_fpath",
        help="CSV file path containing parameter sweep results to validate",
    )
    argparser.add_argument(
        "--feastol",
        default=1e-5,
        help="Tolerance used to check feasibility of parameters (default=1e-5)",
    )
    argparser.add_argument(
        "--row",
        default=None,
        type=int,
        help="Row of the input data whose inputs are validated",
    )
    args = argparser.parse_args()
    df = pd.read_csv(args.experiment_fpath)

    if args.row is None:
        # For hard-coding a row to validate
        row = 55
    else:
        row = args.row

    conversion = df["X"][row]
    pressure = df["P"][row]
    steam = df["Steam"][row]
    bypass = df["Bypass Frac"][row]
    ch4_feed = df["CH4 Feed"][row]

    m = make_simulation_model(
        pressure,
        conversion=conversion,
        flow_H2O=steam,
        bypass_fraction=bypass,
        feed_flow_CH4=ch4_feed,
    )
    m._obj = pyo.Objective(expr=0.0)
    add_external_function_libraries_to_environment(m)
    validate_model_simulation(m, feastol=args.feastol)


if __name__ == "__main__":
    main()
