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
from pyomo.common.timing import TicTocTimer, HierarchicalTimer
from idaes.core.solvers import get_solver
from svi.auto_thermal_reformer.nn_flowsheet import (
    create_instance,
    initialize_nn_atr_flowsheet,
    # TODO: use function to get the surrogate fpath. This will require the ability to
    # pass in our parsed arguments. (Or make these arguments a global data structure)
    DEFAULT_SURROGATE_FNAME,
)
from idaes.core.surrogate.keras_surrogate import KerasSurrogate
import svi.auto_thermal_reformer.config as config


INVALID = None


FORMULATION_MAP = {
    "full": KerasSurrogate.Formulation.FULL_SPACE,
    "reduced": KerasSurrogate.Formulation.REDUCED_SPACE,
}


def main():

    argparser = config.get_sweep_argparser()
    argparser.add_argument(
        "--fname",
        # Default depends on full vs. reduced space and is defined below
        default=None,
        help="Base file name for parameter sweep results",
    )

    argparser.add_argument(
        "--surrogate_fname",
        default=DEFAULT_SURROGATE_FNAME,
        help="File name for the surrogate",
    )

    argparser.add_argument(
        "--formulation",
        default="full",
        help=(
            "Formulation for embedding neural network surrogate into optimization model."
            " Must be 'full' or 'reduced'."
        ),
    )

    args = argparser.parse_args()

    formulation = FORMULATION_MAP[args.formulation]

    if args.fname is None:
        sweep_fname = f"nn-sweep-{args.formulation}.csv"

    surrogate_fname = os.path.join(args.data_dir, args.surrogate_fname)
    output_fpath = os.path.join(args.data_dir, sweep_fname)

    df = {key: [] for key in config.PARAM_SWEEP_KEYS}

    """
    The optimization problem to solve is the following:
    Maximize H2 composition in the product stream such that its minimum flow is 3500 mol/s, 
    its maximum N2 concentration is 0.3, the maximum reformer outlet temperature is 1200 K and 
    the maximum product temperature is 650 K.  
    """

    xp_samples = config.get_parameter_samples(args)

    for X, P in xp_samples:
        try: 
            m = create_instance(X, P, surrogate_fname=surrogate_fname, formulation=formulation)
            initialize_nn_atr_flowsheet(m)
            m.fs.reformer_bypass.inlet.temperature.unfix()
            m.fs.reformer_bypass.inlet.flow_mol.unfix()

            solver = config.get_optimization_solver()
            intermediate_cb = solver.config.intermediate_callback
            htimer = HierarchicalTimer()
            timer = TicTocTimer()
            timer.tic('starting timer')
            results = solver.solve(m, tee=True, timer=htimer)
            dT = timer.toc('end')
            f_eval_time = htimer.timers["solve"].timers["function"].total_time
            j_eval_time = htimer.timers["solve"].timers["jacobian"].total_time
            h_eval_time = htimer.timers["solve"].timers["hessian"].total_time

            if results.solver.termination_condition == pyo.TerminationCondition.optimal:   
                df[list(df.keys())[0]].append(X)
                df[list(df.keys())[1]].append(P)
                df[list(df.keys())[2]].append(results.solver.termination_condition)
                df["Time"].append(dT)
                df["Objective"].append(pyo.value(m.fs.product.mole_frac_comp[0,'H2']))
                df["Steam"].append(pyo.value(m.fs.reformer_mix.steam_inlet.flow_mol[0]))
                df["Bypass Frac"].append(pyo.value(m.fs.reformer_bypass.split_fraction[0,'bypass_outlet']))
                df["CH4 Feed"].append(pyo.value(m.fs.feed.outlet.flow_mol[0]))
                df["Iterations"].append(len(intermediate_cb.iterate_data))
                df["function-time"].append(f_eval_time)
                df["jacobian-time"].append(j_eval_time)
                df["hessian-time"].append(h_eval_time)
            else:
                df[list(df.keys())[0]].append(X)
                df[list(df.keys())[1]].append(P)
                df[list(df.keys())[2]].append(results.solver.termination_condition)
                # If the solver didn't converge, we don't care about the solve time,
                # the objective, or any of the degree of freedom values.
                for key in config.PARAM_SWEEP_KEYS:
                    if key not in ("X", "P", "Termination"):
                        df[key].append(INVALID)
        
        except ValueError:
            df[list(df.keys())[0]].append(X)
            df[list(df.keys())[1]].append(P)
            df[list(df.keys())[2]].append("ValueError")
            # If the solver didn't converge, we don't care about the solve time,
            # the objective, or any of the degree of freedom values.
            for key in config.PARAM_SWEEP_KEYS:
                if key not in ("X", "P", "Termination"):
                    df[key].append(INVALID)

    df = pd.DataFrame(df)
    df.to_csv(output_fpath)

if __name__ == "__main__":
    main()
