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
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
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

# This script will plot the condition number of the Jacobian vs Ipopt iteration
# for 4 successful instances of the full space and implicit formulations. 

# SUBSETS represent a tuple containing (X,P,iters), where iters = number of iterations 
# for the {full-space, implicit function, ALAMO, NN} formulations to successfully converge.

data_dir = config.get_data_dir()

SUBSETS_FULLSPACE = [
    (0.94, 1947379.0, 89),
    (0.90, 1727379.0, 123),
    (0.92, 1587379.0, 58),
    (0.93, 1657379.0, 52)
]

SUBSETS_IMPLICIT = [
    (0.94, 1947379.0, 45),
    (0.90, 1727379.0, 35),
    (0.92, 1587379.0, 41),
    (0.93, 1657379.0, 37)
]

SUBSETS_ALAMO = [
    (0.94, 1947379.0, 29),
    (0.90, 1727379.0, 26),
    (0.92, 1587379.0, 32),
    (0.93, 1657379.0, 27)
]

SUBSETS_NN = [
    (0.94, 1947379.0, 53),
    (0.90, 1727379.0, 48),
    (0.92, 1587379.0, 47),
    (0.93, 1657379.0, 44)
]

def calculate_condition_number(m):
    nlp = PyomoNLP(m)
    jac = nlp.evaluate_jacobian()
    cond_num = np.linalg.cond(jac.toarray())
    return cond_num

def full(X=0.94, P=1947379.0, iters=300):
    m = make_optimization_model(X, P)
    solver = config.get_optimization_solver(iters=iters)
    print(f"Solving sample with X={X}, P={P}")
    results = solver.solve(m, tee=True)
    cond_num = calculate_condition_number(m)
    return cond_num

def implicit(X=0.94, P=1947379.0, iters=300):
    m = make_optimization_model(X, P)
    add_external_function_libraries_to_environment(m)
    m_implicit = make_implicit(m)
    solver = config.get_optimization_solver(iters=iters)
    print(f"Solving sample with X={X}, P={P}")
    results = solver.solve(m_implicit, tee=True)
    cond_num = calculate_condition_number(m)
    return cond_num

def alamo(X=0.94, P=1947379.0, iters=300):
    m = create_alamo_instance(X, P, surrogate_fname=os.path.join(data_dir, DEFAULT_ALAMO_SURROGATE_FNAME))
    initialize_alamo_atr_flowsheet(m)
    m.fs.reformer_bypass.inlet.temperature.unfix()
    m.fs.reformer_bypass.inlet.flow_mol.unfix()
    solver = config.get_optimization_solver(iters=iters)
    print(f"Solving sample with X={X}, P={P}")
    results = solver.solve(m, tee=True)
    cond_num = calculate_condition_number(m)
    return cond_num

def nn(X=0.94, P=1947379.0, iters=300):
    m = create_nn_instance(X, P, surrogate_fname=os.path.join(data_dir, DEFAULT_NN_SURROGATE_FNAME))
    initialize_nn_atr_flowsheet(m)
    m.fs.reformer_bypass.inlet.temperature.unfix()
    m.fs.reformer_bypass.inlet.flow_mol.unfix()
    solver = config.get_optimization_solver(iters=iters)
    print(f"Solving sample with X={X}, P={P}")
    results = solver.solve(m, tee=True)
    cond_num = calculate_condition_number(m)
    return cond_num

plt.figure(figsize=(12, 8))

for i, (full_subset, implicit_subset, alamo_subset, nn_subset) in enumerate(zip(SUBSETS_FULLSPACE, 
                                                                                SUBSETS_IMPLICIT, 
                                                                                SUBSETS_ALAMO, 
                                                                                SUBSETS_NN), 1):
    plt.subplot(2, 2, i)
    full_list = []
    implicit_list = []
    alamo_list = []
    nn_list = []
    for iters in range(1, full_subset[2] + 1):
        full_list.append(full(X=full_subset[0], P=full_subset[1], iters=iters))
    for iters in range(1, implicit_subset[2] + 1):
        implicit_list.append(implicit(X=implicit_subset[0], P=implicit_subset[1], iters=iters))
    for iters in range(1, alamo_subset[2] + 1):
        alamo_list.append(alamo(X=alamo_subset[0], P=alamo_subset[1], iters=iters))
    for iters in range(1, nn_subset[2] + 1):
        nn_list.append(nn(X=nn_subset[0], P=nn_subset[1], iters=iters))
    plt.plot(full_list, label='Full-space')
    plt.plot(implicit_list, label='Implicit function')
    plt.plot(alamo_list, label='ALAMO surrogate')
    plt.plot(nn_list, label='Neural Network surrogate')
    plt.xlabel('Iteration')
    plt.ylabel('Condition number of constraint Jacobian')
    plt.title(f'Subset {i}: X={full_subset[0]}, P={full_subset[1]} Pa')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

plt.tight_layout()
plt.show()

