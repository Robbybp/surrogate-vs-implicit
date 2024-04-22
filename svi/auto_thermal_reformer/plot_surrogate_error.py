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
import seaborn as sns
import svi.auto_thermal_reformer.config as config


def plot_surrogate_error(df, legend=True, title=None, show_training_bounds=False):
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

    ax = sns.heatmap(
        error_array,
    )

    if title is not None:
        ax.set_title(title)

    plt.gca().invert_yaxis()

    return fig, ax


def main(args):
    df = pd.read_csv(args.error_fpath)

    fig, ax = plot_surrogate_error(
        df,
        legend=not args.no_legend,
        title=args.title,
        show_training_bounds=args.show_training_bounds,
    )

    if not args.no_save:
        if args.plot_fname is None:
            plot_fname = os.path.basename(args.error_fpath)
            data_ext = "." + plot_fname.split(".")[-1]
            ext_len = len(data_ext)
            plot_fname = plot_fname[:-ext_len] + "-surrogate-error" + ".pdf"
        else:
            plot_fname = args.plot_fname

        plot_fpath = os.path.join(args.results_dir, plot_fname)
        fig.savefig(plot_fpath, transparent=True)

    if args.show:
        plt.show()


if __name__ == "__main__":
    argparser = config.get_plot_argparser()
    argparser.add_argument(
        "error_fpath", help="Path to CSV file containing surrogate errors to plot"
    )
    args = argparser.parse_args()
    main(args)
