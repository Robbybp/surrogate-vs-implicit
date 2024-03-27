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
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pyomo.environ import (
    Constraint,
    Var,
    ConcreteModel,
    Expression,
    Objective,
    TransformationFactory,
    value,
    units as pyunits,
)

from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from svi.auto_thermal_reformer.fullspace_flowsheet import make_simulation_model
import svi.auto_thermal_reformer.config as config
from svi.auto_thermal_reformer.validate_inputs import validate_model_simulation
from svi.validate import validate_solution
from svi.external import add_external_function_libraries_to_environment

import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns
import argparse


"""Script for validating the results of a parameter sweep by simulating
with the full-space model and, if a baseline parameter sweep file is given,
comparing errors between the simulation and this baseline.

"""


###### FUNCTION TO VALIDATE ALAMO AND NEURAL NETWORK RESULTS ######

def validate_results(df, feastol=0.0):
    """Validate the results of an optimization with a surrogate model by using
    the calculated inputs to simulate the original, "full space" model.

    The main output is the objective function (outlet concentration of H2)
    value obtained by simulating the full space model with the inputs from
    optimization with the surrogate.

    """
    # Parse the data needed for validation
    validation_inputs = df[['X', 'P', 'Steam', 'Bypass Frac', 'CH4 Feed']]

    # Create an empty dataframe to store the objective value from the validation process.
    # df_val_res stores the result of the full space simulation that takes surrogate DOF
    # as inputs.

    df_val_res = {'X':[], 'P':[], 'Feasible': [], 'Infeasibility': [], 'Objective':[]}

    for index, row in validation_inputs.iterrows():
        X = row['X']
        P = row['P']
        Steam = row['Steam']
        Bypass_Frac = row['Bypass Frac']
        CH4_Feed = row['CH4 Feed']

        m = make_simulation_model(
            P,
            conversion=X,
            flow_H2O=Steam,
            bypass_fraction=Bypass_Frac,
            feed_flow_CH4=CH4_Feed,
        )
        add_external_function_libraries_to_environment(m)
        valid, violations = validate_model_simulation(m, feastol=feastol)
        con_violations, bound_violations = violations
        default = (None, None, 0.0)

        con_infeas = abs(
            max(con_violations, key=lambda item: abs(item[2]), default=default)[2]
        )
        bound_infeas = abs(
            max(bound_violations, key=lambda item: abs(item[2]), default=default)[2]
        )
        infeas = max(con_infeas, bound_infeas)
        df_val_res['X'].append(X)
        df_val_res['P'].append(P)
        df_val_res['Infeasibility'].append(infeas)
        df_val_res['Feasible'].append(valid)
        df_val_res['Objective'].append(value(m.fs.product.mole_frac_comp[0,'H2']))

    df_val_res = pd.DataFrame(df_val_res)
    return df_val_res


INVALID = 999


def calculate_objective_errors(input_df, baseline_df, output_df=None):
    """Compute errors between objective values in the input DataFrame and those
    in the baseline DataFrame. If provided, the results are added to the output
    dataframe.

    """
    df_fullspace = baseline_df
    df_val_res = input_df
    if output_df is None:
        # TODO: in-place modification is probably not a good default
        output_df = input_df

    list_of_optimal_results = list()
    list_of_invalid_indices = list()

    for index, row in df_fullspace.iterrows():
        if row['Termination'] == "optimal":
            list_of_optimal_results.append(index)
    for index, row in df_val_res.iterrows():
        if row['Objective'] == INVALID:
            list_of_invalid_indices.append(index)

    baseline_lookup = {
        (row["X"], row["P"]): (row["Termination"], row["Objective"])
        for index, row in baseline_df.iterrows()
    }
    input_lookup = {
        # TODO: This input row should have a "feasible" field we can check
        (row["X"], row["P"]): row["Objective"]
        for index, row in input_df.iterrows()
    }

    # Initialize empty column of dataframe
    errors = []
    for i, row in output_df.iterrows():
        params = (row["X"], row["P"])
        if (
            params in baseline_lookup and baseline_lookup[params][0] == "optimal" and baseline_lookup[params][1] != INVALID
            and params in input_lookup and input_lookup[params] != INVALID
        ):
            # This is a signed fractional error, i.e. a negative value
            # indicates a lower objective than the baseline
            error = (
                (input_lookup[params] - baseline_lookup[params][1])
                / baseline_lookup[params][1]
            )
        else:
            error = None
        errors.append(error)
    output_df["objective-error"] = errors
    return output_df


def main():
    argparser = config.get_argparser()
    argparser.add_argument(
        "experiment_fpath",
        help="CSV file path containing parameter sweep results to validate",
    )
    argparser.add_argument(
        "--baseline-fpath",
        default=None,
        help="CSV file path containing parameter sweep results to compare against",
    )
    argparser.add_argument(
        "--validation-fname",
        default=None,
        help="Basename of data file that validation results are written to",
    )
    argparser.add_argument(
        "--feastol",
        default=1e-8,
        help="Tolerance used to check feasibility of parameters (default 1e-8)",
    )
    args = argparser.parse_args()

    if args.validation_fname is None:
        experiment_basename = os.path.basename(args.experiment_fpath)
        if "." in experiment_basename:
            experiment_extension = "." + experiment_basename.split(".")[-1]
            name = experiment_basename[:-len(experiment_extension)]
            validation_fname = name + "-validation" + experiment_extension
        else:
            validation_fname = experiment_basename + "-validation"
        print(f"No validation-fname provided. Default is: {validation_fname}")
    else:
        validation_fname = args.validation_fname
    validation_fpath = os.path.join(args.data_dir, validation_fname)

    input_df = pd.read_csv(args.experiment_fpath)
    validation_df = validate_results(input_df, feastol=args.feastol)

    if args.baseline_fpath is not None:
        baseline_df = pd.read_csv(args.baseline_fpath)
        calculate_objective_errors(validation_df, baseline_df)

        ierrors = [
            (i, err) for i, err in enumerate(validation_df["objective-error"])
            # This is a check for NaN:
            if err is not None and err == err
        ]
        neg_ierrors = [(i, err) for i, err in ierrors if err < 0]
        pos_ierrors = [(i, err) for i, err in ierrors if err > 0]
        pos_err_indices = [i for i, err in pos_ierrors]
        errors = [err for i, err in ierrors]
        average_error = sum(errors) / len(errors) if errors else None
        print(f"Number of instances where both converge: {len(errors)}")
        print(f"Average fractional error in objective: {average_error}")
        print(f"Number of instances with positive errors: {len(pos_ierrors)}")
        print(f"Indices with positive errors: {pos_err_indices}")

    print("Validated results")
    print("-----------------")
    print(validation_df)

    validation_df.to_csv(validation_fpath)


if __name__ == "__main__":
    main()
