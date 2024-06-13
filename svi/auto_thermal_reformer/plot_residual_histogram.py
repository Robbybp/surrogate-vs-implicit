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
import numpy as np
from pyomo.core.base.constraint import Constraint


def main(args):
    df = pd.read_csv(args.fpath)

    model = None
    for key in config.CONSTRUCTOR_LOOKUP:
        if key in os.path.basename(args.fpath):
            model = key
    if model is None:
        raise RuntimeError("Could not infer model from filename")
    # Just choose some dummy values of conversion and pressure. We won't
    # actually use this model numerically
    X = 0.94
    P = 1550000.0
    m = config.CONSTRUCTOR_LOOKUP[model](X, P)
    connames = [con.name for con in m.component_data_objects(Constraint)]

    con_infeas_count = {}
    for name in connames:
        resid_array = np.array(df[name])
        violated = resid_array > args.infeas_threshold
        n_violated = sum(violated)
        con_infeas_count[name] = n_violated

    constraints_by_infeas = sorted(
        connames,
        reverse=True,
        key=lambda name: con_infeas_count[name],
    )
    infeas_counts = [con_infeas_count[name] for name in constraints_by_infeas]
    constraint_indices = list(range(len(constraints_by_infeas)))

    plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots()
    ax.bar(constraint_indices, infeas_counts)
    ax.set_ylabel("Number of\niterations", rotation=0) 
    ax.set_xlabel("Constraint indices")
    ax.yaxis.set_label_coords(-0.3, 0.5)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)

    w, h = fig.get_size_inches()
    fig.set_size_inches(w*1.3, h)

    fig.tight_layout()
    if args.show:
        plt.show()

    if not args.no_save:
        # Assume file name is NAME.ext
        name = os.path.basename(args.fpath).split(".")[0]
        fname = name + "-resid-histogram.pdf"
        fpath = os.path.join(args.results_dir, fname)
        fig.savefig(fpath, transparent=not args.opaque)


if __name__ == "__main__":
    argparser = config.get_plot_argparser()

    argparser.add_argument("fpath", help="File containing iterate data to plot")
    argparser.add_argument("--infeas-threshold", type=float, default=1e-3)

    args = argparser.parse_args()
    main(args)
