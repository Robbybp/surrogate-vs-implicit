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

# TODO
def create_implicit_model(model, external_variables, external_constraints):
    """Create a new Pyomo model based on the original, but with external
    variables defined implicitly by the external constraints

    """
    # - Identify input variables (non-external variables in external constraints)
    # - Identify "residual constraints" (non-external constraints containing
    #   external variables)
    # - Create ExternalPyomoModel
    # - Create new model with references variables/constraints from old model
    # - Somehow account for inequalities and bounds that contain external
    #   variables. It may be worthwhile to use ExternalGreyBoxBlock.outputs
    #   here.
    pass
