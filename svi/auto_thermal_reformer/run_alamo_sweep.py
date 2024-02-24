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
from svi.auto_thermal_reformer.alamo_flowsheet import (
    create_instance,
    initialize_alamo_atr_flowsheet,
    # TODO: use function to get the surrogate fpath. This will require the ability to
    # pass in our parsed arguments. (Or make these arguments a global data structure)
    DEFAULT_SURROGATE_FNAME,
)
import svi.auto_thermal_reformer.config as config


def main():

    argparser = config.get_sweep_argparser()
    argparser.add_argument(
        "--fname",
        default="sweep_results_alamo.csv",
        help="Base file name for parameter sweep results",
    )
    args = argparser.parse_args()

    # TODO: This should be configurable by CLI
    surrogate_fname = os.path.join(args.data_dir, DEFAULT_SURROGATE_FNAME)
    output_fpath = os.path.join(args.data_dir, args.fname)

    df = {'X':[], 'P':[], 'Termination':[], 'Time':[], 'Objective':[], 'Steam':[], 'Bypass Frac': [], 'CH4 Feed':[]}

    """
    The optimization problem to solve is the following:
    Maximize H2 composition in the product stream such that its minimum flow is 3500 mol/s, 
    its maximum N2 concentration is 0.3, the maximum reformer outlet temperature is 1200 K and 
    the maximum product temperature is 650 K.  
    """

    xp_samples = config.get_parameter_samples(args)

    #x_lo = 0.90
    #x_hi = 0.97
    #p_lo = 1447379.0
    #p_hi = 1947379.0
    #
    #n_x = args.n1
    #n_p = args.n2
    #dx = (x_hi - x_lo) / (n_x - 1)
    #dp = (p_hi - p_lo) / (n_p - 1)
    #x_list = [x_lo + i * dx for i in range(n_x)]
    #p_list = [p_lo + i * dp for i in range(n_p)]

    #xp_samples = list(itertools.product(x_list, p_list))

    #for X in [0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97]:
    #for X in [0.95,0.96,0.97]:
    #    #for P in np.arange(1447379, 1947379, 70000):
    #    for P in [1450000, 1650000, 1850000]:
    for X, P in xp_samples:
        try: 
            m = create_instance(X, P, surrogate_fname=surrogate_fname)
            # Does this need to be applied after creating the surrogate? Why?
            initialize_alamo_atr_flowsheet(m)
            m.fs.reformer_bypass.inlet.temperature.unfix()
            m.fs.reformer_bypass.inlet.flow_mol.unfix()

            #m = make_simulation_model(X,P)

            ######## OBJECTIVE IS TO MAXIMIZE H2 COMPOSITION IN PRODUCT STREAM #######
            #m.fs.obj = pyo.Objective(expr = m.fs.product.mole_frac_comp[0, 'H2'], sense = pyo.maximize)

            ######## CONSTRAINTS #######

            ## Link outputs of ALAMO to inputs of reformer_recuperator 
            #@m.Constraint()
            #def link_T(m):
            #    return m.fs.reformer_recuperator.shell_inlet.flow_mol[0] == m.fs.reformer.out_flow_mol

            #@m.Constraint()
            #def link_F(m):
            #    return m.fs.reformer_recuperator.shell_inlet.temperature[0] == m.fs.reformer.out_temp

            #@m.Constraint()
            #def link_H2(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'H2'] == m.fs.reformer.out_H2

            #@m.Constraint()
            #def link_CO(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'CO'] == m.fs.reformer.out_CO

            #@m.Constraint()
            #def link_H2O(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'H2O'] == m.fs.reformer.out_H2O

            #@m.Constraint()
            #def link_CO2(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'CO2'] == m.fs.reformer.out_CO2

            #@m.Constraint()
            #def link_CH4(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'CH4'] == m.fs.reformer.out_CH4

            #@m.Constraint()
            #def link_C2H6(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'C2H6'] == m.fs.reformer.out_C2H6

            #@m.Constraint()
            #def link_C3H8(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'C3H8'] == m.fs.reformer.out_C3H8

            #@m.Constraint()
            #def link_C4H10(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'C4H10'] == m.fs.reformer.out_C4H10

            #@m.Constraint()
            #def link_N2(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'N2'] == m.fs.reformer.out_N2

            #@m.Constraint()
            #def link_O2(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'O2'] == m.fs.reformer.out_O2

            #@m.Constraint()
            #def link_Ar(m):
            #    return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'Ar'] == m.fs.reformer.out_Ar

            ## MINIMUM PRODUCT FLOW OF 3500 mol/s IN PRODUCT STREAM
            #@m.Constraint()
            #def min_product_flow_mol(m):
            #    return m.fs.product.flow_mol[0] >= 3500

            ## MAXIMUM N2 COMPOSITION OF 0.3 IN PRODUCT STREAM
            #@m.Constraint()
            #def max_product_N2_comp(m):
            #    return m.fs.product.mole_frac_comp[0, 'N2'] <= 0.3

            ## MAXIMUM REFORMER OUTLET TEMPERATURE OF 1200 K
            #@m.Constraint()
            #def max_reformer_outlet_temp(m):
            #    return m.fs.reformer.out_temp <= 1200

            ## MAXIMUM PRODUCT OUTLET TEMPERATURE OF 650 K
            #@m.Constraint()
            #def max_product_temp(m):
            #    return m.fs.product.temperature[0] <= 650

            #m.fs.feed.outlet.flow_mol[0].setlb(1120)
            #m.fs.feed.outlet.flow_mol[0].setub(1250)
            ## Unfix D.O.F. If you unfix these variables, inlet temperature, flow and composition
            ## to the Gibbs reactor will have to be determined by the optimization problem.
            #m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].unfix()
            #m.fs.feed.outlet.flow_mol.unfix()
            #m.fs.steam_feed.flow_mol.unfix() 

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
            #df[list(df.keys())[5]].append(pyo.value(m.fs.steam_feed.flow_mol[0]))
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
