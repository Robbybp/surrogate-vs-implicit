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

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import svi.auto_thermal_reformer.config as config


def plot_surrogate_error(
    df,
    legend=True,
    show_training_bounds=False,
):
    # TODO: Make this (as well as font.family) a global option set
    # when we get the argparser? Or maybe just make this configurable from the
    # command line with a global default?
    # But the same font size might not be appropriate for all plots.
    plt.rcParams["font.size"] = 20
    fig = plt.figure()

    conversion_list = list(sorted(set(df["X"])))
    pressure_list = list(sorted(set(df["P"])))
    n_conversion = len(conversion_list)
    n_pressure = len(pressure_list)
    error_lookup = {
        (df["X"][i], df["P"][i]): df["surrogate-error"][i]
        for i in range(len(df))
    }
    error_array = [
        [error_lookup[conversion_list[j], pressure_list[i]] for i in range(n_pressure)]
        for j in range(n_conversion)
    ]

    cbar = legend
    # "Maximum relative error" (Between reactor model and surrogate model, over
    # all output values, for the intput values at which the optimization stopped)
    cbar_kws = dict(label="Maximum relative error")
    ax = sns.heatmap(
        error_array,
        cbar=cbar,
        cbar_kws=cbar_kws,
        square=True,
        cmap="Reds",
    )
    # Change "background color" used for missing data (where either full-space
    # or surrogate model failed)
    ax.set_facecolor("black")

    if legend:
        w, h = fig.get_size_inches()
        # Getting the legend to appear in the right location is really annoying...
        fig.set_size_inches(1.5*w, h)
        legend_handles = [Patch(color="black", label="Failed simulation")]
        ax.legend(
            handles=legend_handles,
            #ncol=1,
            handlelength=0.75,
            bbox_to_anchor=(2.4, 1.0),
            #loc="upper right",
            #borderaxespad=0,
        )

    xtick_positions = [i+0.5 for i in range(n_pressure)]
    ytick_positions = [i+0.5 for i in range(n_conversion)]
    xtick_labels = ["%1.2f" % (p / 1e6) if i%2 else "" for i, p in enumerate(pressure_list)]
    ytick_labels = ["%1.2f" % x if i%2 else "" for i, x in enumerate(conversion_list)]
    ax.set_xticks(xtick_positions, labels=xtick_labels)
    ax.set_yticks(ytick_positions, labels=ytick_labels, rotation=0)
    ax.set_xlabel("Pressure (MPa)")
    ax.set_ylabel("Conversion")

    plt.gca().invert_yaxis()

    return fig, ax


def main(args):
    df = pd.read_csv(args.error_fpath)

    fig, ax = plot_surrogate_error(
        df,
        legend=not args.no_legend,
        show_training_bounds=args.show_training_bounds,
    )

    if args.title is not None:
        ax.set_title(args.title)

    fig.tight_layout()

    if not args.no_save:
        if args.plot_fname is None:
            plot_fname = os.path.basename(args.error_fpath)
            data_ext = "." + plot_fname.split(".")[-1]
            ext_len = len(data_ext)
            plot_fname = plot_fname[:-ext_len] + ".pdf"
        else:
            plot_fname = args.plot_fname

        plot_fpath = os.path.join(args.results_dir, plot_fname)
        fig.savefig(plot_fpath, transparent=not args.opaque)

    if args.show:
        plt.show()


if __name__ == "__main__":
    argparser = config.get_plot_argparser()
    argparser.add_argument(
        "error_fpath", help="Path to CSV file containing surrogate errors to plot"
    )
    args = argparser.parse_args()
    main(args)
