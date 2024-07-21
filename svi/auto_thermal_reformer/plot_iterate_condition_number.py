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
    "condition-number": "Condition number",
    "fullspace": "Full-space",
    "implicit": "Implicit function",
    "nn-full": "Neural network",
    "alamo": "ALAMO surrogate",
}


FONTSIZE = 16


def plot_trajectory(
    df,
    keys,
    labels=None,
    fig=None,
    ax=None,
    iter_lim=None,
):
    plt.rcParams["font.size"] = FONTSIZE
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    iterations = list(range(len(df)))

    for i, key in enumerate(keys):
        if labels is None:
            label = LABEL_LOOKUP[key] if key in LABEL_LOOKUP else key
        else:
            label = labels[i]
        ax.plot(
            iterations,
            list(df[key]),
            label=label,
            linewidth=2,
        )
    if not args.no_legend:
        ax.legend()
    ax.set_yscale("log")
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlabel("Iteration number")
    if iter_lim is not None:
        ax.set_xlim(None, iter_lim)
    return fig, ax


def main(args):
    fpaths = args.fpaths.split(",")
    models = []
    for fpath in fpaths:
        model = None
        for name in config.CONSTRUCTOR_LOOKUP:
            if name in os.path.basename(fpath):
                model = name
                break
        if model is None:
            raise RuntimeError(
                f"Cannot infer model from fpath {fpath}"
            )
        models.append(model)

    dfs = [pd.read_csv(fpath) for fpath in fpaths]
    #keys = [key for key in args.keys.split(",") if key != ""]
    keys = ["condition-number"]
    labels = [LABEL_LOOKUP[model] for model in models]

    plt.rcParams["font.size"] = FONTSIZE
    fig, ax = plt.subplots()
    for i, df in enumerate(dfs):
        plot_trajectory(
            df,
            keys,
            # Use the labels arg here to override the default, which is to lookup
            # labels by keys.
            labels=[labels[i]],
            fig=fig,
            ax=ax,
            iter_lim=args.iter_lim,
        )

    if args.show:
        plt.show()

    if not args.no_save:
        fig.tight_layout()
        # Assume file name is NAME.ext
        if len(models) == 1:
            name = os.path.basename(args.fpath).split(".")[0]
            fname = name + "-condition.pdf"
            fpath = os.path.join(args.results_dir, fname)
        else:
            fname = "-".join(models) + "-iterates-condition.pdf"
            fpath = os.path.join(args.results_dir, fname)
        fig.savefig(fpath, transparent=not args.opaque)


if __name__ == "__main__":
    argparser = config.get_plot_argparser()

    argparser.add_argument(
        "fpaths",
        help="Comma-separated list of files containing iterate data to plot",
    )
    argparser.add_argument(
        "--iter-lim",
        type=int,
        default=None,
        help="Upper bound on x-axis",
    )

    args = argparser.parse_args()
    main(args)
