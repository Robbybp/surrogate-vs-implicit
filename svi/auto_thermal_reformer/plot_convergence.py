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
from svi.auto_thermal_reformer.config import get_argparser

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns


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
    help="Optional parameter to override 'Feasible' column with a relaxed tolerance",
)


def plot_convergence_reliability(fname, validation_df=None, feastol=None):
    data = pd.read_csv(fname)

    condition = data["Termination"] == "optimal"

    # I would rather not modify data in-place here
    data.loc[condition, "Termination"] = 1
    data.loc[~condition, "Termination"] = 0

    # To determine whether to plot as a "success", I want to check:
    # - sweep["Termination"]
    # - validation["Feasible"]
    # - optinally, validation["Infeasibility"]

    data = data.drop("Unnamed: 0", axis=1)
    data["Termination"] = data["Termination"].astype(float)

    data["P"] = data["P"] / 1e6
    data["P"] = data["P"].round(2)

    df_for_plotting = data[["X", "P", "Termination"]]
    pivoting = np.round(
        pd.pivot_table(
            df_for_plotting,
            values="Termination",
            index="X",
            columns="P",
            aggfunc="first",
        ),
        2,
    )

    fig = plt.figure(figsize=(7, 7))
    ax = sns.heatmap(
        pivoting,
        linewidths=1,
        linecolor="darkgray",
        linewidth=1,
        cbar=False,
        cmap=ListedColormap(["black", "bisque"]),
        vmin=0,
        vmax=1,
    )

    # This function should plot on a pair of axes. The title and filename can be set
    # outside of this function.
    # plt.title(name+' Formulation', fontsize = 18.5)
    plt.xlabel("Pressure (MPa)", fontsize=18.5)
    plt.ylabel("Conversion", fontsize=18.5)

    original_labels_conversion = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
    new_labels_conversion = [0.91, 0.93, 0.95, 0.97]
    labels_conversion_plotting = [
        label if label in new_labels_conversion else ""
        for label in original_labels_conversion
    ]
    ax.set_yticklabels(labels_conversion_plotting, fontsize=16.5)

    original_labels_pressure = [1.45, 1.52, 1.59, 1.66, 1.73, 1.80, 1.87, 1.94]
    new_labels_pressure = [1.52, 1.66, 1.80, 1.94]
    labels_pressure_plotting = [
        label if label in new_labels_pressure else ""
        for label in original_labels_pressure
    ]
    ax.set_xticklabels(labels_pressure_plotting, fontsize=16.5)

    legend_handles = [
        Patch(color="bisque", label="Successful"),
        Patch(color="black", label="Unsuccessful"),
    ]

    plt.legend(
        handles=legend_handles,
        ncol=1,
        fontsize=16,
        handlelength=0.8,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
    )

    plt.gca().invert_yaxis()

    # fig.savefig(name + ' Plot', bbox_inches='tight')
    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = argparser.parse_args()

    if args.validation_fpath is not None:
        validation_df = pd.read_csv(args.validation_fpath)
    else:
        validation_df = None

    plot_convergence_reliability(
        fname=args.experiment_fpath,
        validation_df=validation_df,
        feastol=args.feastol,
    )
