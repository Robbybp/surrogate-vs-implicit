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

SUBSETS_FULLSPACE = [
    (0.94, 1947379.0, 12),
]

def full_space(X=0.94, P=1947379.0, iters=300):
    m = make_optimization_model(X, P)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
    solver = config.get_optimization_solver(iters=iters)
    print(f"Solving sample with X={X}, P={P}")
    results = solver.solve(m, tee=True)
    return m

def gradient_of_lagrangian(m):
    # The Lagrangian is defined as: L(x,s,y,z) = f(x) - y^T c_E(x) - z^T (c_I(x) - s)
    # Where "y" are duals for equality constraints and "z" are duals for inequality
    # constraints. In this case, inequalities are represented as equality constraints with
    # slack variables "s". 
    # The gradient of the lagrangian = grad_obj - jac_eq^T y - jac_ineq^T z and should = 0 at 
    # optimal solution.
    nlp = PyomoNLP(m)
    gradient_obj = nlp.evaluate_grad_objective()
    eq_constraint_jac = nlp.evaluate_jacobian_eq()
    ineq_constraint_jac = nlp.evaluate_jacobian_ineq()
    eq_duals = nlp.get_duals_eq()
    ineq_duals = nlp.get_duals_ineq()
    #print(eq_duals)
    jac_eqTy = eq_constraint_jac.transpose().dot(eq_duals)
    jac_ineqTz = ineq_constraint_jac.transpose().dot(ineq_duals)
    gradient_of_lagrangian = gradient_obj - jac_eqTy - jac_ineqTz
    #hess_lag = nlp.evaluate_hessian_lag()
    return gradient_of_lagrangian

def main():
    m = full_space(X=0.90, P=1447379.0, iters=300)
    print(np.round(gradient_of_lagrangian(m), decimals=2))

if __name__ == "__main__":
    main()

#m = pyo.ConcreteModel()
#m.x = pyo.Var(bounds=(-5, None))
#m.y = pyo.Var(initialize=2.5)
#m.obj = pyo.Objective(expr=m.x**2 + m.y**2, sense = pyo.maximize)
#m.c1 = pyo.Constraint(expr=m.y == -(m.x - 3)*2)
#m.c2 = pyo.Constraint(expr=m.y >= pyo.exp(2*m.x) - m.x**2)
#solver = config.get_optimization_solver(iters=22)
#solver.solve(m, tee=True)
#nlp = PyomoNLP(m)
#print(nlp.get_duals())


#m = pyo.ConcreteModel()
#
#m.v1 = pyo.Var(initialize=-2.0)
#m.v2 = pyo.Var(initialize=2.0)
#m.v3 = pyo.Var(initialize=2.0)
#m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
#
#m.v1.setlb(-10.0)
#m.v2.setlb(1.5)
#m.v1.setub(-1.0)
#m.v2.setub(10.0)
#
#m.eq_con = pyo.Constraint(expr=m.v1*m.v2*m.v3 - 2.0 == 0)
#
#obj_factor = 1 
#m.obj = pyo.Objective(
#        expr=obj_factor*(m.v1**2 + m.v2**2 + m.v3**2),
#        sense=pyo.minimize,
#        )
#solver = config.get_optimization_solver(iters=15)
#solver.solve(m, tee=True)
#nlp = PyomoNLP(m)
#print(nlp.get_duals())
#print(nlp.evaluate_jacobian())
##set_duals, suffix.
