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
from pyomo.util.subsystems import identify_external_functions


def get_external_function_libraries(model):
    # This implementation should be much faster, but it assumes that the model
    # has all relevant external functions on it. Also, in the old implementation,
    # I add a new library for each unique library-function pair. I'm not sure
    # if there was a reason for this, so I'll keep the old implementation around
    # for now.
    library_set = set()
    libraries = []
    for comp in model.component_data_objects(pyo.ExternalFunction):
        library = comp._library
        fcn = comp._function
        if library not in library_set:
            library_set.add(library)
            libraries.append(library)
    return libraries
    #ef_exprs = []
    #for comp in model.component_data_objects(
    #    (pyo.Constraint, pyo.Expression, pyo.Objective), active=True
    #):
    #    ef_exprs.extend(identify_external_functions(comp.expr))
    #unique_functions = []
    #fcn_set = set()
    #for expr in ef_exprs:
    #    fcn = expr._fcn
    #    data = (fcn._library, fcn._function)
    #    if data not in fcn_set:
    #        fcn_set.add(data)
    #        unique_functions.append(data)
    #unique_libraries = [library for library, function in unique_functions]
    return unique_libraries


def add_external_function_libraries_to_environment(model, envvar="AMPLFUNC"):
    libraries = get_external_function_libraries(model)
    lib_str = "\n".join(libraries)
    os.environ[envvar] = lib_str
