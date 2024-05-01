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

def calculate_condition_number(m):
    result = {"block": [], "condition number": []}
    nlp = PyomoNLP(m)
    igraph = IncidenceGraphInterface(m, include_inequality=False)
    vblocks, cblocks = igraph.block_triangularize()
    for i, (vblock, cblock) in enumerate(zip(vblocks, cblocks)):
        submatrix = nlp.extract_submatrix_jacobian(vblock, cblock)
        cond = np.linalg.cond(submatrix.toarray())
        if cond > 1e6:
            result["block"].append(i)
            result["condition number"].append(cond)
    return result

def full(X=0.94, P=1947379.0, iters=300):
    m = make_optimization_model(X, P)
    solver = config.get_optimization_solver(iters=iters)
    print(f"Solving sample with X={X}, P={P}")
    results = solver.solve(m, tee=True)
    m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].fix(m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].value)
    m.fs.reformer_mix.steam_inlet.flow_mol.fix(m.fs.reformer_mix.steam_inlet.flow_mol[0].value)
    m.fs.feed.outlet.flow_mol.fix(m.fs.feed.outlet.flow_mol[0].value)
    result = calculate_condition_number(m)
    return result 

def identify_blocks_high_cn(dict1, dict2):
    increased_blocks = []
    for block1, cond1, cond2 in zip(dict1['block'], dict1['condition number'], dict2['condition number']):
        if cond2 > cond1:
            increased_blocks.append(block1)
    return increased_blocks

def main():
    for i, (full_subset) in enumerate(zip(SUBSETS_FULLSPACE), 1):
        before = full(X=full_subset[0][0], P=full_subset[0][1], iters=full_subset[0][2])
        after = full(X=full_subset[0][0], P=full_subset[0][1], iters=full_subset[0][2]+1)
    block_to_analyze = identify_blocks_high_cn(before, after)
    print(f"Block(s) causing the instance to enter a region of high condition number: {block_to_analyze}.")   

if __name__ == "__main__":
    main()
