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
    calc_var_kwds = dict(eps=1e-7)
    solve_kwds = dict(tee=tee)
    solver = pyo.SolverFactory("cyipopt")
    solve_strongly_connected_components(
        m,
        solver=solver,
        calc_var_kwds=calc_var_kwds,
        solve_kwds=solve_kwds,
    )
    res = solver.solve(m, tee=tee)
    return res


def validate_model(m, feastol=0.0):
    try:
        simulate_model(m)
    except ValueError:
        # We sometimes get ValueErrors when Ipopt throws an error, even when the
        # solve is acceptable.
        pass
    valid, violations = validate_solution(m, tolerance=feastol)
    return valid


def main():
    argparser = config.get_argparser()
    argparser.add_argument(
        "experiment_fpath",
        help="CSV file path containing parameter sweep results to validate",
    )
    argparser.add_argument(
        "--feastol",
        default=1e-6,
        help="Tolerance used to check feasibility of parameters",
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
    validate_model(m, feastol=args.feastol)


if __name__ == "__main__":
    main()
