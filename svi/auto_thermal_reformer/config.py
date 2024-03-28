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
import argparse
import itertools
import pyomo.environ as pyo


filedir = os.path.dirname(__file__)


PARAM_SWEEP_KEYS = [
    'X', 'P', 'Termination', 'Time', 'Objective', 'Steam', 'Bypass Frac', 'CH4 Feed'
]


def get_optimization_solver(options=None):
    # Use cyipopt for everything for Ipopt version consistency among all
    # formulations
    solver = pyo.SolverFactory("cyipopt")
    if options is None:
        options = {}
    solver.config.options["max_iter"] = 300
    solver.config.options["linear_solver"] = "ma27"
    solver.config.options["tol"] = 1e-7
    solver.config.options["print_user_options"] = "yes"
    for key, val in options.items():
        solver.config.options[key] = val
    return solver


def get_data_dir():
    datadir = os.path.join(filedir, "data")
    if os.path.isfile(datadir):
        raise OSError(
            f"Default data dir {datadir} is already a file. Please specify a"
            " different data dir via the --data-dir argument."
        )
    elif not os.path.isdir(datadir):
        os.mkdir(datadir)
    return datadir


def get_results_dir():
    resultsdir = os.path.join(filedir, "results")
    if os.path.isfile(resultsdir):
        raise OSError(
            f"Default results dir {resultsdir} is already a file. Please specify a"
            " different results dir via the --results-dir argument."
        )
    elif not os.path.isdir(resultsdir):
        os.mkdir(resultsdir)
    return resultsdir


def get_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data-dir", required=False, default=get_data_dir())
    argparser.add_argument("--results-dir", required=False, default=get_results_dir())
    return argparser


def get_sweep_argparser():
    argparser = get_argparser()
    argparser.add_argument(
        "--n1", default=8, help="Default number of samples for conversion", type=int
    )
    argparser.add_argument(
        "--n2", default=8, help="Default number of samples for pressure", type=int
    )
    argparser.add_argument(
        "--subset",
        default=None,
        help="Comma-separated list of integers corresponding to samples to run in the sweep",
    )
    argparser.add_argument("--no-save", action="store_true", help="Don't save results")
    return argparser


def get_parameter_samples(args):
    x_lo = 0.90
    x_hi = 0.97
    p_lo = 1447379.0
    #p_hi = 1947379.0
    p_hi = 1937379.0
    
    n_x = args.n1
    n_p = args.n2
    dx = (x_hi - x_lo) / (n_x - 1)
    dp = (p_hi - p_lo) / (n_p - 1)
    x_list = [x_lo + i * dx for i in range(n_x)]
    p_list = [p_lo + i * dp for i in range(n_p)]

    xp_samples = list(itertools.product(x_list, p_list))
    if args.subset is not None:
        subset = args.subset.split(",")
        subset = [int(i) for i in subset]
        xp_samples = [xp_samples[i] for i in subset]
    return xp_samples
