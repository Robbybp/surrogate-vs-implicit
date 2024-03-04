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
import itertools
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.timing import TicTocTimer
from idaes.core.solvers import get_solver
from svi.auto_thermal_reformer.nn_flowsheet import (
    create_instance,
    initialize_nn_atr_flowsheet,
    # TODO: use function to get the surrogate fpath. This will require the ability to
    # pass in our parsed arguments. (Or make these arguments a global data structure)
    DEFAULT_SURROGATE_FNAME,
)
import svi.auto_thermal_reformer.config as config


def main():

    argparser = config.get_sweep_argparser()
    argparser.add_argument(
        "--fname",
        default="nn-sweep.csv",
        help="Base file name for parameter sweep results",
    )

    argparser.add_argument(
        "--surrogate_fname",
        default=DEFAULT_SURROGATE_FNAME,
        help="File name for the surrogate",
    )

    args = argparser.parse_args()

    surrogate_fname = os.path.join(args.data_dir, args.surrogate_fname)
    output_fpath = os.path.join(args.data_dir, args.fname, args.surrogate_fname)

    df = {'X':[], 'P':[], 'Termination':[], 'Time':[], 'Objective':[], 'Steam':[], 'Bypass Frac': [], 'CH4 Feed':[]}

    """
    The optimization problem to solve is the following:
    Maximize H2 composition in the product stream such that its minimum flow is 3500 mol/s, 
    its maximum N2 concentration is 0.3, the maximum reformer outlet temperature is 1200 K and 
    the maximum product temperature is 650 K.  
    """

    xp_samples = config.get_parameter_samples(args)

    for X, P in xp_samples:
        try: 
            m = create_instance(X, P, surrogate_fname=surrogate_fname)
            initialize_nn_atr_flowsheet(m)
            m.fs.reformer_bypass.inlet.temperature.unfix()
            m.fs.reformer_bypass.inlet.flow_mol.unfix()

            solver = get_solver()
            solver.options = {
                "tol": 1e-7,
                "max_iter": 300
            }
            timer = TicTocTimer()
            timer.tic('starting timer')
            results = solver.solve(m, tee=True)
            dT = timer.toc('end')
            df[list(df.keys())[0]].append(X)
            df[list(df.keys())[1]].append(P)
            df[list(df.keys())[2]].append(results.solver.termination_condition)
            df[list(df.keys())[3]].append(dT)
            df[list(df.keys())[4]].append(pyo.value(m.fs.product.mole_frac_comp[0, 'H2']))
            df[list(df.keys())[5]].append(pyo.value(m.fs.reformer_mix.steam_inlet.flow_mol[0]))
            df[list(df.keys())[6]].append(pyo.value(m.fs.reformer_bypass.split_fraction[0, 'bypass_outlet']))
            df[list(df.keys())[7]].append(pyo.value(m.fs.feed.outlet.flow_mol[0]))
        except ValueError:
            df[list(df.keys())[0]].append(X)
            df[list(df.keys())[1]].append(P)
            df[list(df.keys())[2]].append("ValueError")
            df[list(df.keys())[3]].append(999)
            df[list(df.keys())[4]].append(999)
            df[list(df.keys())[5]].append(999)
            df[list(df.keys())[6]].append(999)
            df[list(df.keys())[7]].append(pyo.value(m.fs.feed.outlet.flow_mol[0]))
            continue

    df = pd.DataFrame(df)
    df.to_csv(output_fpath)

if __name__ == "__main__":
    main()
