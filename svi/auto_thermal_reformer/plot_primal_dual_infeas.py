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
#  ___________________________________________________________________________

import os
import svi.auto_thermal_reformer.config as config
import pandas as pd
import matplotlib.pyplot as plt


LABEL_LOOKUP = {
    "inf_pr": "Primal infeasibility",
    "inf_du": "Dual infeasibility",
}


def plot_trajectory(
    df,
    keys,
    labels=None,
):
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots()

    iterations = list(range(len(df)))

    for i, key in enumerate(keys):
        label = LABEL_LOOKUP[key] if key in LABEL_LOOKUP else key
        ax.plot(
            iterations,
            list(df[key]),
            label=label,
            linewidth=2,
        )
    ax.legend()
    ax.set_yscale("log")
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlabel("Iteration number")
    return fig, ax


def main(args):
    df = pd.read_csv(args.fpath)
    #keys = [key for key in args.keys.split(",") if key != ""]
    keys = ["inf_pr", "inf_du"]
    fig, ax = plot_trajectory(df, keys)

    if args.show:
        plt.show()

    if not args.no_save:
        fig.tight_layout()
        # Assume file name is NAME.ext
        name = os.path.basename(args.fpath).split(".")[0]
        fname = name + "-pdinfeas.pdf"
        fpath = os.path.join(args.results_dir, fname)
        fig.savefig(fpath, transparent=not args.opaque)


if __name__ == "__main__":
    argparser = config.get_plot_argparser()

    argparser.add_argument("fpath", help="File containing iterate data to plot")

    args = argparser.parse_args()
    main(args)
