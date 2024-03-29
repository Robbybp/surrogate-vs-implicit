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

######## IMPORT PACKAGES ########
import os
import pandas as pd
import numpy as np
from svi.auto_thermal_reformer.config import get_argparser

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns


"""Script to analyze the results of a parameter sweep and, optionally, the
subsequent validation.

We compute:
    - The percent of instances that are successful (s.t. some validation feastol)
    - The average and max solve time of successful instances
    - The average solve time over some intersection of instances (?)

"""
# TODO: Should this entire script be done in the validation script?
# Probably not, because the validation script is quite time-consuming.
# How to handle statistics that should be computed for an intersection of instances?
# Maybe accept as an argument which instances to include?
# - Other than instances to use, everything I need to compute can be computed just
#   from the sweep data and the validation data, right?
# - For now, I can compute statistics over the instances where the full-space
#   was successful just by checking if we have an objective error


argparser = get_argparser()
# TODO: Add positional argument for file whose results to plot. For now, this
# is hard-coded below.
argparser.add_argument(
    "experiment_fpath",
    help="CSV file path containing parameter sweep results to validate",
)
argparser.add_argument(
    "--validation-fpath",
    default=None,
    help="CSV file path containing the results of a validation",
)
argparser.add_argument(
    "--feastol",
    default=None,
    type=float,
    help="Optional parameter to override 'Feasible' column with a relaxed tolerance",
)


def analyze_results(
    data,
    validation_df=None,
    feastol=None,
    subset=None,
):
    """Display average/max solve times and objective errors
    """
    n_total = len(data)
    if subset is None:
        # If no subset provided, consider all the rows.
        subset = np.array(list(range(n_total)))
    n_subset = len(subset)
    print(f"Analyzing {n_subset} instances")
    subset_set = set(subset)

    solver_optimal = (data["Termination"] == "optimal")
    if validation_df is not None:
        if feastol is None:
            model_feasible = (validation_df["Feasible"] == 1)
        else:
            # If feastol is set, we can relax the tolerance that was used to
            # determine the "Feasible" flag. NOTE that we can't tighten the
            # tolerance here, only relax it.
            model_feasible = (
                (validation_df["Feasible"] == 1)
                | (validation_df["Infeasibility"] <= feastol)
            )
    else:
        # If we don't have validation data, just assume that optimal => feasible
        model_feasible = np.ones(len(data))
    success = (solver_optimal & model_feasible) # potentially: & obj_within_margin)

    success_rows = np.where(success == 1)[0]
    n_success = len(success_rows)
    # Should we use len(subset) as the denominator? Kind of seems like yes
    percent_conv = round(n_success / n_subset * 100, 1)
    print(f"Converged {n_success} / {n_subset} ({percent_conv}%) instances")

    subset_success = [i for i in success_rows if i in subset_set]
    solve_times = [data["Time"][i] for i in subset_success]
    ave_solve_time = sum(solve_times) / len(solve_times)
    print(f"Of the specified instances, {len(subset_success)} were successful")
    print(f"Of these, average solve time is {ave_solve_time} s")
    print(f"Of these, max solve time is {max(solve_times)} s")
    # median solve time?

    if validation_df is not None:
        # Cast to float to get rid of potential None
        objerr = validation_df["objective-error"].astype(float)
        # baseline succeeded where objerr is not NaN
        baseline_success = (objerr == objerr) # checks for NaN
        baseline_success_rows = np.where(baseline_success == 1)[0]
        subset_baseline_success = [i for i in baseline_success_rows if i in subset_set]
        obj_errors = [validation_df["objective-error"][i] for i in subset_baseline_success]
        ave_obj_error = sum(obj_errors) / len(obj_errors)
        print(
            "Of the specified instances, the baseline was successful in"
            f" {len(subset_baseline_success)}"
        )
        print(
            "Of these, the average relative objective difference,"
            f" (experiment - baseline) / baseline, is {ave_obj_error}"
        )
        print(
            "Of these, the max relative objective difference,"
            f" (experiment - baseline) / baseline, is {max(obj_errors)}"
        )
        print(
            "Of these, the min relative objective difference,"
            f" (experiment - baseline) / baseline, is {min(obj_errors)}"
        )
        # median objective error?


if __name__ == "__main__":
    args = argparser.parse_args()

    experiment_df = pd.read_csv(args.experiment_fpath)
    if args.validation_fpath is not None:
        validation_df = pd.read_csv(args.validation_fpath)
    else:
        validation_df = None

    analyze_results(
        experiment_df,
        validation_df=validation_df,
        feastol=args.feastol,
    )
