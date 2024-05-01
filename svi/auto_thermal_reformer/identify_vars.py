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
from svi.auto_thermal_reformer.fullspace_flowsheet import (
    make_optimization_model,
    make_simulation_model,
)
import pandas as pd
from svi.auto_thermal_reformer.implicit_flowsheet import make_implicit
from svi.external import add_external_function_libraries_to_environment
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
    CyIpoptSolverWrapper
)

from pyomo.core.expr.visitor import identify_variables
from pyomo.util.subsystems import create_subsystem_block
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
import svi.auto_thermal_reformer.config as config
import numpy as np

# SUBSETS_FULLSPACE represents tuples (X,P,iters), where iters = number of iterations 
# before the solve experiences a jump in the condition number.
# E.g., Instance X = 0.94, P = 1.94 MPa experiences a jump in iteration 12 + 1 = 13. 

SUBSETS_FULLSPACE = [
    (0.94, 1947379.0, 12),
    #(0.90, 1727379.0, 12),
    #(0.92, 1587379.0, 12),
    #(0.93, 1657379.0, 12)
]

def get_vars_related_to_block(m, block=524):
    input_and_block_vars = {"Variable":[], "Value":[]}
    nlp = PyomoNLP(m)
    igraph = IncidenceGraphInterface(m, include_inequality=False)
    vblocks, cblocks = igraph.block_triangularize()
    vblock = vblocks[block]
    cblock = cblocks[block]
    for con in cblock:
        for var in identify_variables(con.expr):
            if 'params' not in var.name:
                input_and_block_vars["Variable"].append(var.name)
                input_and_block_vars["Value"].append(var.value)
    return input_and_block_vars

def full(X=0.94, P=1947379.0, iters=300):
    m = make_optimization_model(X, P)
    solver = config.get_optimization_solver(iters=iters)
    print(f"Solving sample with X={X}, P={P}")
    results = solver.solve(m, tee=True)
    m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].fix(m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].value)
    m.fs.reformer_mix.steam_inlet.flow_mol.fix(m.fs.reformer_mix.steam_inlet.flow_mol[0].value)
    m.fs.feed.outlet.flow_mol.fix(m.fs.feed.outlet.flow_mol[0].value)
    result = get_vars_related_to_block(m)
    return result 

def implicit(X=0.94, P=1947379.0, iters=300):
    m = make_optimization_model(X, P)
    add_external_function_libraries_to_environment(m)
    m_implicit = make_implicit(m)
    solver = config.get_optimization_solver(iters=iters)
    print(f"Solving sample with X={X}, P={P}")
    results = solver.solve(m_implicit, tee=True)
    m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].fix(m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].value)
    m.fs.reformer_mix.steam_inlet.flow_mol.fix(m.fs.reformer_mix.steam_inlet.flow_mol[0].value)
    m.fs.feed.outlet.flow_mol.fix(m.fs.feed.outlet.flow_mol[0].value)
    cond_num = get_vars_related_to_block(m)
    return cond_num

def variable_with_jump(dict1, dict2):
    variables_with_jump = {"Variable":[], "Value_initial":[], "Value_final":[]}
    for var, value1, value2 in zip(dict1["Variable"], dict1["Value"], dict2["Value"]):
        if value2 != value1:
            percentage_difference = abs((value2 - value1) / value1) * 100
            if percentage_difference > 10:
                variables_with_jump["Variable"].append(var)
                variables_with_jump["Value_initial"].append(value1)
                variables_with_jump["Value_final"].append(value2)
    return variables_with_jump

def dict_to_csv(data, filename):
    df = pd.DataFrame.from_dict(data)
    df.to_csv(filename, index=False)

def main():
    for i, (full_subset) in enumerate(zip(SUBSETS_FULLSPACE), 1):
        before = implicit(X=full_subset[0][0], P=full_subset[0][1], iters=full_subset[0][2])
        after = implicit(X=full_subset[0][0], P=full_subset[0][1], iters=full_subset[0][2]+1)
    variables_with_jump = variable_with_jump(before, after)
    dict_to_csv(variables_with_jump, 'vars_jump_more_than_10pt.csv')


if __name__ == "__main__":
    main()
