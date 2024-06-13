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
from svi.cyipopt import TimedPyomoCyIpoptSolver, Callback
from svi.external import add_external_function_libraries_to_environment


"""Consistent-signature callbacks to construct each model

All models returned are initialized and ready to solve.

Note that these constructors use local imports to avoid circular dependencies.
The right solution here is probably to separate the single-model simulation drivers
(which depend on config) from the model constructing "library" functions (which we
need for these constructor callbacks).

"""

def fullspace_constructor(X, P, **kwds):
    from svi.auto_thermal_reformer.fullspace_flowsheet import make_optimization_model as create_fullspace_instance
    m = create_fullspace_instance(X, P)
    return m


def alamo_constructor(X, P, **kwds):
    from svi.auto_thermal_reformer.alamo_flowsheet import (
        create_instance as create_alamo_instance,
        initialize_alamo_atr_flowsheet,
    )
    surrogate_fname = kwds.pop("surrogate_fname", None)
    m = create_alamo_instance(X, P, surrogate_fname=surrogate_fname)
    initialize_alamo_atr_flowsheet(m)
    m.fs.reformer_bypass.inlet.temperature.unfix()
    m.fs.reformer_bypass.inlet.flow_mol.unfix()
    return m


def nn_full_constructor(X, P, **kwds):
    from svi.auto_thermal_reformer.nn_flowsheet import (
        create_instance as create_nn_instance,
        initialize_nn_atr_flowsheet,
    )
    surrogate_fname = kwds.pop("surrogate_fname", None)
    # Note that KerasSurrogate.Formulation.FULL_SPACE is the default
    m = create_nn_instance(X, P, surrogate_fname=surrogate_fname)
    initialize_nn_atr_flowsheet(m)
    m.fs.reformer_bypass.inlet.temperature.unfix()
    m.fs.reformer_bypass.inlet.flow_mol.unfix()
    return m


def implicit_constructor(X, P, **kwds):
    from svi.auto_thermal_reformer.fullspace_flowsheet import make_optimization_model as create_fullspace_instance
    from svi.auto_thermal_reformer.implicit_flowsheet import make_implicit
    m = create_fullspace_instance(X, P)
    add_external_function_libraries_to_environment(m)
    m_implicit = make_implicit(m)
    return m_implicit


filedir = os.path.dirname(__file__)


PARAM_SWEEP_KEYS = [
    'X',
    'P',
    'Termination',
    'Iterations',
    'Time',
    'function-time',
    'jacobian-time',
    'hessian-time',
    'Objective',
    'Steam',
    'Bypass Frac',
    'CH4 Feed',
]


CONSTRUCTOR_LOOKUP = {
    "fullspace": fullspace_constructor,
    "implicit": implicit_constructor,
    "alamo": alamo_constructor,
    "nn-full": nn_full_constructor,
}

def get_optimization_solver(options=None, iters=300, callback=None):
    # Use cyipopt for everything for Ipopt version consistency among all
    # formulations
    #solver = pyo.SolverFactory("cyipopt")
    # This is a very simple callback we just use to get iteration counts.
    if callback is None:
        callback = Callback()
    solver = TimedPyomoCyIpoptSolver(intermediate_callback=callback)
    if options is None:
        options = {}
    solver.config.options["max_iter"] = iters 
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

def get_plot_argparser():
    argparser = get_argparser()
    argparser.add_argument("--show", action="store_true", help="Flag to show the plot")
    argparser.add_argument("--no-save", action="store_true", help="Flag to not save the plot")
    argparser.add_argument("--plot-fname", default=None, help="Basename for plot file")
    argparser.add_argument("--no-legend", action="store_true", help="Flag to exclude a legend")
    argparser.add_argument("--title", default=None, help="Plot title")
    argparser.add_argument("--show-training-bounds", action="store_true")
    argparser.add_argument("--opaque", action="store_true", help="Not transparent")
    return argparser
