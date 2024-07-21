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
from svi.cyipopt import ConditioningCallback
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from svi.auto_thermal_reformer.fullspace_flowsheet import (
    make_optimization_model,
    make_simulation_model,
)
from svi.auto_thermal_reformer.implicit_flowsheet import make_implicit
from svi.external import add_external_function_libraries_to_environment
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
    CyIpoptSolverWrapper
)

from svi.auto_thermal_reformer.alamo_flowsheet import (
    create_instance as create_alamo_instance,
    initialize_alamo_atr_flowsheet,
    DEFAULT_SURROGATE_FNAME as DEFAULT_ALAMO_SURROGATE_FNAME
)

from svi.auto_thermal_reformer.nn_flowsheet import (
    create_instance as create_nn_instance,
    initialize_nn_atr_flowsheet,
    DEFAULT_SURROGATE_FNAME as DEFAULT_NN_SURROGATE_FNAME,
)
import svi.auto_thermal_reformer.config as config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse 

# SUBSETS represent a tuple containing (X,P,iters), where iters = number of iterations 
# for the {full-space, implicit function, ALAMO, NN} formulations to successfully converge.
# For the unsuccessful instances, implicit function displays eval error after iter 12. The 
# full space and NN formulations do not find a solution after 300 iters.

data_dir = config.get_data_dir()

SUCCESS_SUBSETS_FULLSPACE = [
    (0.94, 1947379.0, 89),
    (0.90, 1727379.0, 123),
    (0.92, 1587379.0, 58),
    (0.93, 1657379.0, 52)
]

SUCCESS_SUBSETS_IMPLICIT = [
    (0.94, 1947379.0, 45),
    (0.90, 1727379.0, 35),
    (0.92, 1587379.0, 41),
    (0.93, 1657379.0, 37)
]

SUCCESS_SUBSETS_ALAMO = [
    (0.94, 1947379.0, 29),
    (0.90, 1727379.0, 26),
    (0.92, 1587379.0, 32),
    (0.93, 1657379.0, 27)
]

SUCCESS_SUBSETS_NN = [
    (0.94, 1947379.0, 53),
    (0.90, 1727379.0, 48),
    (0.92, 1587379.0, 47),
    (0.93, 1657379.0, 44)
]

NOSUCCESS_SUBSETS_FULLSPACE = [
    (0.96, 1447379.0, 150),
    (0.97, 1447379.0, 150),
    (0.96, 1587379.0, 150),
    (0.97, 1587379.0, 150)
]

NOSUCCESS_SUBSETS_IMPLICIT = [
    (0.96, 1447379.0, 12),
    (0.97, 1447379.0, 12),
    (0.96, 1587379.0, 12),
    (0.97, 1587379.0, 12)
]

NOSUCCESS_SUBSETS_NN = [
    (0.96, 1447379.0, 150),
    (0.97, 1447379.0, 150),
    (0.96, 1587379.0, 150),
    (0.97, 1587379.0, 150)
]

def solver(m, iters = 300):
    solver = config.get_optimization_solver(iters = iters)
    callback = ConditioningCallback()
    solver.config.intermediate_callback = callback
    solver.solve(m, tee=True)
    condition_numbers_list = callback.condition_numbers
    return condition_numbers_list[1:]

def full(X=0.94, P=1947379.0, iters=300):
    m = make_optimization_model(X, P)
    print(f"Solving sample with X={X}, P={P}")
    condition_numbers_list = solver(m, iters = iters)
    return condition_numbers_list

def implicit(X=0.94, P=1947379.0, iters=300):
    m = make_optimization_model(X, P)
    add_external_function_libraries_to_environment(m)
    m_implicit = make_implicit(m)
    print(f"Solving sample with X={X}, P={P}")
    condition_numbers_list = solver(m_implicit, iters = iters) 
    return condition_numbers_list

def alamo(X=0.94, P=1947379.0, iters=300):
    m = create_alamo_instance(X, P, surrogate_fname=os.path.join(data_dir, DEFAULT_ALAMO_SURROGATE_FNAME))
    initialize_alamo_atr_flowsheet(m)
    m.fs.reformer_bypass.inlet.temperature.unfix()
    m.fs.reformer_bypass.inlet.flow_mol.unfix()
    print(f"Solving sample with X={X}, P={P}")
    condition_numbers_list = solver(m, iters = iters)
    return condition_numbers_list 

def nn(X=0.94, P=1947379.0, iters=300):
    m = create_nn_instance(X, P, surrogate_fname=os.path.join(data_dir, DEFAULT_NN_SURROGATE_FNAME))
    initialize_nn_atr_flowsheet(m)
    m.fs.reformer_bypass.inlet.temperature.unfix()
    m.fs.reformer_bypass.inlet.flow_mol.unfix()
    print(f"Solving sample with X={X}, P={P}")
    condition_numbers_list = solver(m, iters = iters)
    return condition_numbers_list

def plot_subsets(SUBSETS_FULLSPACE, SUBSETS_IMPLICIT, SUBSETS_NN, SUBSETS_ALAMO=None, unsuccessful=False):
    plt.figure(figsize=(12, 8))

    for i, (full_subset, implicit_subset, nn_subset) in enumerate(zip(SUBSETS_FULLSPACE, SUBSETS_IMPLICIT, SUBSETS_NN), 1):
        plt.subplot(2, 2, i)
        full_list = full(X=full_subset[0], P=full_subset[1], iters=full_subset[2])
        implicit_list = implicit(X=implicit_subset[0], P=implicit_subset[1], iters=implicit_subset[2])
        nn_list = nn(X=nn_subset[0], P=nn_subset[1], iters=nn_subset[2])
        plt.plot(full_list, label='Full-space')
        plt.plot(implicit_list, label='Implicit function')
        plt.plot(nn_list, label='Neural Network surrogate')
        if SUBSETS_ALAMO:
            alamo_subset = SUBSETS_ALAMO[i - 1]
            alamo_list = alamo(X=alamo_subset[0], P=alamo_subset[1], iters=alamo_subset[2])
            plt.plot(alamo_list, label='ALAMO surrogate')
        plt.xlabel('Iteration')
        plt.ylabel('Condition number of constraint Jacobian')
        plt.title(f'Subset {i}: X={full_subset[0]}, P={full_subset[1]} Pa')
        if unsuccessful:
            plt.legend(loc="lower right")
        else:
            plt.legend(loc="upper right")
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot subsets")
    parser.add_argument("--unsuccessful", action="store_true", help="Whether to use an unsuccessful instance or not")
    args = parser.parse_args()

    if args.unsuccessful:
        SUBSETS_FULLSPACE = NOSUCCESS_SUBSETS_FULLSPACE
        SUBSETS_IMPLICIT = NOSUCCESS_SUBSETS_IMPLICIT
        SUBSETS_NN = NOSUCCESS_SUBSETS_NN
        fig = plot_subsets(SUBSETS_FULLSPACE, SUBSETS_IMPLICIT, SUBSETS_NN, unsuccessful=True)
    else:
        SUBSETS_FULLSPACE = SUCCESS_SUBSETS_FULLSPACE
        SUBSETS_IMPLICIT = SUCCESS_SUBSETS_IMPLICIT
        SUBSETS_ALAMO = SUCCESS_SUBSETS_ALAMO 
        SUBSETS_NN = SUCCESS_SUBSETS_NN
        fig = plot_subsets(SUBSETS_FULLSPACE, SUBSETS_IMPLICIT, SUBSETS_NN, SUBSETS_ALAMO)
