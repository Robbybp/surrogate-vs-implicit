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
from svi.nlp import project_onto
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


"""Script for plotting comparisons of state trajectories between different methods
"""


LABEL_LOOKUP = {
    "inf_pr": "Primal infeasibility",
    "inf_du": "Dual infeasibility",
    "fs.reformer_bypass.split_fraction[0.0,bypass_outlet]": "Split fraction",
    "fs.reformer_mix.steam_inlet_state[0.0].flow_mol": "Steam inlet",
    "fs.feed.properties[0.0].flow_mol": "Natural gas inlet",
}

KEY_LOOKUP = {
    # These are names that we can use in the CLI instead of writing out the keys
    "split-fraction": "fs.reformer_bypass.split_fraction[0.0,bypass_outlet]",
    "steam-inlet": "fs.reformer_mix.steam_inlet_state[0.0].flow_mol",
    "methane-inlet": "fs.feed.properties[0.0].flow_mol",
}

INDEX_LOOKUP = {
    "fs.reformer_bypass.split_fraction[0.0,bypass_outlet]": 0,
    "fs.reformer_mix.steam_inlet_state[0.0].flow_mol": 1,
    "fs.feed.properties[0.0].flow_mol": 2,
}


plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "serif"


def get_label(fpath):
    fname = os.path.basename(fpath)
    if "nn-full" in fname:
        return "Full-space NN"
    elif "nn-reduced" in fname:
        return "Reduced-space NN"
    elif "fullspace" in fname:
        return "Full-space"
    elif "implicit" in fname:
        return "Implicit"
    elif "alamo" in fname:
        return "ALAMO"
    else:
        raise NotImplementedError(f"Filepath {fpath} could not be recognized")


def plot_trajectory(
    df,
    keys,
    # Overrides default labels so we can label trajectories by file rather than by key
    labels=None,
    fig_ax=None,
    logscale=False,
):
    fig, ax = plt.subplots() if fig_ax is None else fig_ax

    iterations = list(range(len(df)))

    if labels is None:
        labels = [LABEL_LOOKUP.get(key, key) for key in keys]

    for i, key in enumerate(keys):
        label = labels[i]
        ax.plot(
            iterations,
            list(df[key]),
            label=label,
            linewidth=2,
        )
    ax.legend()
    if logscale:
        ax.set_yscale("log")
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)
    ax.set_xlabel("Iteration number")
    return fig, ax


def plot_2d_state(
    df,
    state1,
    state2,
    fig_ax=None,
    label=None,
    state1label=None,
    state2label=None,
    # Include initial and final points
    include_points=True,
):
    fig, ax = plt.subplots() if fig_ax is None else fig_ax
    state1label = LABEL_LOOKUP.get(state1, state1) if state1label is None else state1label
    state2label = LABEL_LOOKUP.get(state2, state2) if state2label is None else state2label

    xdata = list(df[state1])
    ydata = list(df[state2])
    ax.plot(
        xdata,
        ydata,
        label=label,
    )
    ax.set_xlabel(state1label)
    ax.set_ylabel(state2label)

    ax.legend()
    return fig, ax


def get_opt_xy(df, state1, state2):
    xdata = list(df[state1])
    ydata = list(df[state2])
    finalx = xdata[-1]
    finaly = ydata[-1]
    return (finalx, finaly)


def plot_points(
    df,
    state1,
    state2,
    fig_ax=None,
    include_final=True,
):
    fig, ax = plt.subplots() if fig_ax is None else fig_ax
    xdata = list(df[state1])
    ydata = list(df[state2])
    initx = xdata[0]
    inity = ydata[0]
    finalx = xdata[-1]
    finaly = ydata[-1]
    ax.scatter(
        [initx],
        [inity],
        marker='.',
        s=100,
        label="Initial",
        color="black",
        # Sufficiently high zorder that we put points on top of lines 
        zorder=10,
    )
    ax.scatter(
        [finalx],
        [finaly],
        marker='*',
        s=100,
        label="Optimal",
        color="black",
        zorder=10,
    )
    ax.legend()
    return fig, ax


def plot_contours(
    rh,
    center=(0.0, 0.0),
    fig_ax=None,
    levels=None,
):
    fig, ax = plt.subplots() if fig_ax is None else fig_ax
    #levels = [1.0, 2.0, 3.0] if levels is None else levels
    levels = [1.0] if levels is None else levels
    evals, evecs = np.linalg.eig(rh)
    # TODO: Make sure RH is 2x2?
    if not np.all(evals > 0):
        raise ValueError("Reduced hessian is not positive definite")
    for l in levels:
        width = 2 * (l / evals[0])**0.5
        height = 2 * (l / evals[1])**0.5
        angle = np.rad2deg(np.arctan2(*evecs[1]))
        ellipse = Ellipse(
            center,
            width=width,
            height=height,
            angle=angle,
            edgecolor=tuple([0.7]*3),
            facecolor="none",
        )
        ax.add_patch(ellipse)


def main(args):
    fpaths = args.fpaths.split(",")
    dfs = [pd.read_csv(fpath) for fpath in fpaths]

    # TODO: Handle multiple keys in a trajectory plot?
    keys = [args.state]
    # Get keys if any shorthands were used
    keys = [KEY_LOOKUP.get(key, key) for key in keys]
    fig, ax = plt.subplots()
    if args.state2 is None:
        for i, df in enumerate(dfs):
            label = get_label(fpaths[i])
            # I know that we're only plotting one key
            labels = [label]
            plot_trajectory(df, keys, labels=labels, fig_ax=(fig, ax))
        output_fname = "state-trajectory.pdf" if args.output_fname is None else args.output_fname
    else:
        for i, df in enumerate(dfs):
            label = get_label(fpaths[i])
            state2 = KEY_LOOKUP.get(args.state2, args.state2)
            plot_2d_state(
                df,
                keys[0],
                state2,
                fig_ax=(fig, ax),
                label=label,
            )
        # We plot the successful trajectory second
        plot_points(
            dfs[1],
            keys[0],
            state2,
            fig_ax=(fig, ax),
        )
        output_fname = "state.pdf" if args.output_fname is None else args.output_fname
        if args.rh is not None:
            coords = (INDEX_LOOKUP[keys[0]], INDEX_LOOKUP[state2])
            rh = np.load(args.rh)
            proj_rh = project_onto(rh, coords)
            levels = [0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
            center = get_opt_xy(dfs[1], keys[0], state2)
            plot_contours(
                proj_rh,
                center=center,
                fig_ax=(fig, ax),
                levels=levels,
            )
            #ax.set_xlim(0.0, 0.6)
            #ax.set_ylim(1000, 1400)
            ax.legend(loc="upper left")

    fig.tight_layout()

    if args.show:
        plt.show()

    if not args.no_save:
        # Assume file name is NAME.ext
        output_fpath = os.path.join(args.results_dir, output_fname)
        fig.savefig(output_fpath, transparent=not args.opaque)


if __name__ == "__main__":
    argparser = config.get_plot_argparser()

    # TODO: Get fpaths
    argparser.add_argument(
        "fpaths",
        help="Comma-separated list of files containing states to plot",
    )
    argparser.add_argument("state", help="State to plot. This can be anything we track, e.g. 'obj_value'")
    argparser.add_argument("--state2", help="Optional second state to plot", default=None)
    argparser.add_argument("--rh", default=None, help=".npy file containing reduced hessian")
    argparser.add_argument("--output-fname", default=None)

    args = argparser.parse_args()
    main(args)
