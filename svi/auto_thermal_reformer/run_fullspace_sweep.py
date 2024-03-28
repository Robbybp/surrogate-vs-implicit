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
import pyomo.environ as pyo
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from svi.auto_thermal_reformer.fullspace_flowsheet import (
    make_optimization_model,
    make_simulation_model,
)
import svi.auto_thermal_reformer.config as config
import pandas as pd
import numpy as np


df = {key: [] for key in config.PARAM_SWEEP_KEYS}


def main(X,P):
    m = make_optimization_model(X,P)

    # For instance 13 in the param sweep, these options give a quite interesting local
    # solution.
    solver = config.get_optimization_solver()
    timer = TicTocTimer()
    timer.tic("starting timer")
    print(f"Solving sample {i} with X={X}, P={P}")
    results = solver.solve(m, tee=True)
    dT = timer.toc("end timer")
    df[list(df.keys())[0]].append(X)
    df[list(df.keys())[1]].append(P)
    df[list(df.keys())[2]].append(results.solver.termination_condition)
    if pyo.check_optimal_termination(results.solver.termination_condition):
        df[list(df.keys())[3]].append(dT)
        df[list(df.keys())[4]].append(pyo.value(m.fs.product.mole_frac_comp[0,'H2']))
        df[list(df.keys())[5]].append(pyo.value(m.fs.reformer_mix.steam_inlet.flow_mol[0]))
        df[list(df.keys())[6]].append(pyo.value(m.fs.reformer_bypass.split_fraction[0,'bypass_outlet']))
        df[list(df.keys())[7]].append(pyo.value(m.fs.feed.outlet.flow_mol[0]))
    else:
        # If the solver didn't converge, we don't care about the solve time,
        # the objective, or any of the degree of freedom values.
        for i in range(3, 8):
            df[config.PARAM_SWEEP_KEYS[i]].append(None)


if __name__ == "__main__":
    argparser = config.get_sweep_argparser()
    argparser.add_argument(
        "--fname",
        default="fullspace-sweep.csv",
        help="Base file name for parameter sweep results"
    )
    args = argparser.parse_args()
    xp_samples = config.get_parameter_samples(args)

    fpath = os.path.join(args.data_dir, args.fname)

    for i, (X, P) in enumerate(xp_samples):
        print(f"Running sample {i} with X={X}, P={P}")
        try:
            main(X,P)
        except AssertionError:
             df[list(df.keys())[0]].append(X)
             df[list(df.keys())[1]].append(P)
             df[list(df.keys())[2]].append("AMPL Error")
             df[list(df.keys())[3]].append(999)
             df[list(df.keys())[4]].append(999)
             df[list(df.keys())[5]].append(999)
             df[list(df.keys())[6]].append(999)
             df[list(df.keys())[7]].append(999)
        except OverflowError:
             df[list(df.keys())[0]].append(X)
             df[list(df.keys())[1]].append(P)
             df[list(df.keys())[2]].append("Overflow Error")
             df[list(df.keys())[3]].append(999)
             df[list(df.keys())[4]].append(999)
             df[list(df.keys())[5]].append(999)
             df[list(df.keys())[6]].append(999)
             df[list(df.keys())[7]].append(999)
        except RuntimeError:
             df[list(df.keys())[0]].append(X)
             df[list(df.keys())[1]].append(P)
             df[list(df.keys())[2]].append("Runtime Error")
             df[list(df.keys())[3]].append(999)
             df[list(df.keys())[4]].append(999)
             df[list(df.keys())[5]].append(999)
             df[list(df.keys())[6]].append(999)
             df[list(df.keys())[7]].append(999)
   
    df = pd.DataFrame(df)
    print(df)
    if args.no_save:
        print(f"--no-save set. Not saving results")
    else:
        print(f"Writing sweep results to {fpath}")
        df.to_csv(fpath)
