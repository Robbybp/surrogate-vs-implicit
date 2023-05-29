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

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
    ExternalPyomoModel
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock
)
from svi.distill import create_instance


def setup_implicit(instance, **kwds):
    """Return a Pyomo model where algebraic equations and variables haver been
    eliminated using an implicit function.

    Keyword arguments are sent straight to ExternalPyomoModel

    """

    diff_vars = [pyo.Reference(instance.x[i, :]) for i in instance.S_TRAYS]
    deriv_vars = [pyo.Reference(instance.dx[i, :]) for i in instance.S_TRAYS]
    disc_eqns = [pyo.Reference(instance.dx_disc_eq[i, :]) for i in instance.S_TRAYS]
    diff_eqns = [pyo.Reference(instance.diffeq[i, :]) for i in instance.S_TRAYS]

    n_diff = len(diff_vars)
    assert n_diff == len(deriv_vars)
    assert n_diff == len(disc_eqns)
    assert n_diff == len(diff_eqns)

    alg_vars = []
    alg_eqns = []
    alg_vars.extend(pyo.Reference(instance.y[i, :]) for i in instance.S_TRAYS)
    alg_eqns.extend(pyo.Reference(instance.mole_frac_balance[i, :])
            for i in instance.S_TRAYS)
    # Since we are not adding them to the reduced space model, alg vars do not
    # need to be references.
    alg_vars.append(instance.rr)
    alg_vars.append(instance.L)
    alg_vars.append(instance.V)
    alg_vars.append(instance.FL)

    alg_eqns.append(instance.reflux_ratio)
    alg_eqns.append(instance.flowrate_rectification)
    alg_eqns.append(instance.vapor_column)
    alg_eqns.append(instance.flowrate_stripping)

    input_vars = [pyo.Reference(instance.u1[:])]

    #
    # Create a block to hold the reduced space model
    #
    reduced_space = pyo.ConcreteModel()
    reduced_space.obj = pyo.Reference(instance.REDUCED_SPACE_OBJ)

    n_input = len(input_vars)

    def differential_block_rule(b, i):
        b.state = diff_vars[i]
        b.deriv = deriv_vars[i]
        b.disc = disc_eqns[i]

    def input_block_rule(b, i):
        b.var = input_vars[i]

    #
    # Add differential variables and equations to the reduced space model.
    # Any constraints that do not contain an "implicit variable" (one that we
    # solve for with our implicit function) can be added to the reduced space
    # model directly.
    #
    reduced_space.differential_block = pyo.Block(
        range(n_diff),
        rule=differential_block_rule,
    )
    reduced_space.input_block = pyo.Block(
        range(n_input),
        rule=input_block_rule,
    )

    #
    # Construct an ExternalGreyBoxBlock. This allows us to add equality
    # constraints to the model that are not defined explicitly. Instead,
    # they will be defined implicitly via an ExternalPyomoModel.
    #
    # ExternalGreyBoxBlock is just a special type of Pyomo Block. It holds
    # the variables that are inputs to or outputs from the external model,
    # and facilitates communication with solvers (CyIpopt).
    #
    reduced_space.external_block = ExternalGreyBoxBlock(instance.t)

    # Add reference to the constraint that specifies the initial conditions
    reduced_space.init_rule = pyo.Reference(instance.init_rule)

    for t in instance.t:
        if t == instance.t.first():
            reduced_space.external_block[t].deactivate()
            continue
        # Create and set external model for every external block
        reduced_space_vars = (
            list(reduced_space.input_block[:].var[t])
            + list(reduced_space.differential_block[:].state[t])
            + list(reduced_space.differential_block[:].deriv[t])
        )
        external_vars = [v[t] for v in alg_vars]
        residual_cons = [c[t] for c in diff_eqns]
        external_cons = [c[t] for c in alg_eqns]

        # This is how we add the ExternalPyomoModel to the ExternalGreyBoxBlock
        reduced_space.external_block[t].set_external_model(
            #
            # Construct an ExternalPyomoModel. This object accepts the inputs
            # specified (in the `inputs` arg) and is in charge of evaluating
            # equality constraints and their derivatives
            #
            ExternalPyomoModel(
                # Arguments are:

                # "Input variables"
                reduced_space_vars,

                # Variables that we eliminate by solving implicitly
                external_vars,

                # Constraints (equations) that will be exposed to the solver from
                # the ExternalPyomoModel. These are all constraints that *are not*
                # used to define the external vars, but that *do* contain some of
                # the external variables.
                residual_cons,

                # Equations that define the external variables (with input
                # variables specified)
                external_cons,

                **kwds,
            ),
            inputs=reduced_space_vars,
        )
    return reduced_space


def main():
    model = create_instance()
    m_implicit = setup_implicit(model)
    options = {"honor_original_bounds": "yes"}
    solver = pyo.SolverFactory("cyipopt", options=options)
    solver.solve(m_implicit, tee=True)


if __name__ == "__main__":
    main()
