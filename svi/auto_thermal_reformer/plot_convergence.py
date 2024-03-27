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
argparser.add_argument("--show", action="store_true", help="Flag to show the plot")
argparser.add_argument("--no-save", action="store_true", help="Flag to not save the plot")
argparser.add_argument("--plot-fname", default=None, help="Basename for plot file")
argparser.add_argument("--no-legend", action="store_true", help="Flag to exclude a legend")


def plot_convergence_reliability(
    data,
    validation_df=None,
    feastol=None,
    legend=True,
):
    # To determine whether to plot as a "success", I want to check:
    # - sweep["Termination"]
    # - validation["Feasible"]
    # - optionally, validation["Infeasibility"]

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

    # Extract columns that we need for plotting and combine with the success array
    df_for_plotting = pd.DataFrame(
        {
            "X": data["X"].values.round(2),
            "P": (data["P"].values / 1e6).round(2),
            "success": success,
        }
    )

    # Re-shape the "flattened" table of instances into a structured X-P grid
    pivoting = pd.pivot_table(
        df_for_plotting,
        values="success",
        index="X",
        columns="P",
        aggfunc="first",
    )

    fig = plt.figure(figsize=(10, 7))
    # Annoyingly, the default figure size seems to cut off the legend...
    #fig = plt.figure()

    # Plot the convergence grid
    ax = sns.heatmap(
        pivoting,
        linewidths=1,
        linecolor="darkgray",
        linewidth=1,
        cbar=False,
        cmap=ListedColormap(["black", "bisque"]),
        vmin=0,
        vmax=1,
        square=True,
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
    ax.set_yticklabels(labels_conversion_plotting, fontsize=16.5, rotation=0)

    original_labels_pressure = [1.45, 1.52, 1.59, 1.66, 1.73, 1.80, 1.87, 1.94]
    new_labels_pressure = [1.52, 1.66, 1.80, 1.94]
    labels_pressure_plotting = [
        label if label in new_labels_pressure else ""
        for label in original_labels_pressure
    ]
    ax.set_xticklabels(labels_pressure_plotting, fontsize=16.5)

    if legend:
        legend_handles = [
            Patch(color="bisque", label="Successful"),
            Patch(color="black", label="Unsuccessful"),
        ]
        ax.legend(
            handles=legend_handles,
            ncol=1,
            fontsize=16,
            handlelength=0.8,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
        )
    plt.gca().invert_yaxis()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    args = argparser.parse_args()

    experiment_df = pd.read_csv(args.experiment_fpath)
    if args.validation_fpath is not None:
        validation_df = pd.read_csv(args.validation_fpath)
    else:
        validation_df = None

    fig, ax = plot_convergence_reliability(
        experiment_df,
        validation_df=validation_df,
        feastol=args.feastol,
        legend=not args.no_legend,
    )

    if not args.no_save:
        if args.plot_fname is None:
            plot_fname = os.path.basename(args.experiment_fpath)
            data_ext = "." + plot_fname.split(".")[-1]
            ext_len = len(data_ext)

            validated = "-validated" if args.validation_fpath is not None else ""

            plot_fname = plot_fname[:-ext_len] + "-convergence" + validated + ".pdf"
        else:
            plot_fname = args.plot_fname
        plot_fpath = os.path.join(args.results_dir, plot_fname)
        print(f"Saving figure to {plot_fpath}")
        fig.savefig(plot_fpath, transparent=True)

    if args.show:
        plt.show()
