import pyomo.environ as pyo
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from svi.auto_thermal_reformer.fullspace_flowsheet import (
    make_optimization_model,
    make_simulation_model,
)
import pandas as pd
import numpy as np


df = {'X':[], 'P':[], 'Termination':[], 'Time':[], 'Objective':[], 'Steam':[], 'Bypass Fraction':[], 'CH4 Feed':[]}


def main(X,P):
    m = make_optimization_model(X,P)
    solver = pyo.SolverFactory('ipopt')
    solver.options = {"tol": 1e-7, "max_iter": 300}
    timer = TicTocTimer()
    timer.tic("starting timer")
    results = solver.solve(m, tee=True)
    dT = timer.toc("end timer")
    df[list(df.keys())[0]].append(X)
    df[list(df.keys())[1]].append(P)
    df[list(df.keys())[2]].append(results.solver.termination_condition)
    df[list(df.keys())[3]].append(dT)
    df[list(df.keys())[4]].append(pyo.value(m.fs.product.mole_frac_comp[0,'H2']))
    df[list(df.keys())[5]].append(pyo.value(m.fs.reformer_mix.steam_inlet.flow_mol[0]))
    df[list(df.keys())[6]].append(pyo.value(m.fs.reformer_bypass.split_fraction[0,'bypass_outlet']))
    df[list(df.keys())[7]].append(pyo.value(m.fs.feed.outlet.flow_mol[0]))


if __name__ == "__main__":
    simulation = False
    optimization = not simulation
    visualize = False
    if optimization:
        for X in np.arange(0.90,0.98,0.01):
            for P in np.arange(1447379,1947379,70000):
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
    df.to_csv('fullspace_experiment.csv')

    if simulation:

        m = make_simulation_model(P = 3447379, initialize = True)
        calc_var_kwds = dict(eps=1e-7)
        solve_kwds = dict(tee=True)
        solver = pyo.SolverFactory("ipopt")
        solve_strongly_connected_components(
            m,
            solver=solver,
            calc_var_kwds=calc_var_kwds,
            solve_kwds=solve_kwds,
        )
        solver.solve(m, tee=True)

        m.fs.reformer.report()
        m.fs.reformer_recuperator.report()
        m.fs.product.report()
        m.fs.reformer_bypass.split_fraction.display()

    if visualize:
        m.fs.visualize("Auto-Thermal-Reformer-Flowsheet")
