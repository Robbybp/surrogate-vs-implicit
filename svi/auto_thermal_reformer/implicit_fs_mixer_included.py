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

import pandas as pd
import numpy as np
from pyomo.common.timing import TicTocTimer
import pyomo.environ as pyo
from pyomo.environ import (
    Constraint,
    Var,
    ConcreteModel,
    Expression,
    Objective,
    TransformationFactory,
    value,
    units as pyunits,
)
from pyomo.common.collections import ComponentSet
from pyomo.network import Arc, SequentialDecomposition
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.models.unit_models import (
    Mixer,
    Heater,
    HeatExchanger,
    PressureChanger,
    GibbsReactor,
    Separator,
    Feed,
    Product,
)
from idaes.core.util.initialization import propagate_state
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
from idaes.core.solvers import get_solver
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
    CyIpoptSolverWrapper
)

from svi.external import add_external_function_libraries_to_environment

def make_implicit(m):
    full_igraph = IncidenceGraphInterface(m)
    # m.fs.combined_reformer references the block containing the reformer_mix and the reformer.
    reformer_mixer_igraph = IncidenceGraphInterface(m.fs.combined_reformer, include_inequality=False)
    # Pressure is not an "external variable" (it is fixed), so the pressure
    # linking equation doesn't need to be included as "residual"
    to_exclude = ComponentSet([m.fs.REF_OUT_expanded.pressure_equality[0]])
    residual_eqns = [
        con for con in m.fs.REF_OUT_expanded.component_data_objects(
            pyo.Constraint, active=True
        )
        if con not in to_exclude
    ]

    input_vars = [
        ### steam
        m.fs.combined_reformer.reformer_mix.steam_inlet.flow_mol[0],
        ### natural gas
        m.fs.combined_reformer.reformer_mix.gas_inlet.temperature[0],
        m.fs.combined_reformer.reformer_mix.gas_inlet.flow_mol[0],
        m.fs.combined_reformer.reformer_mix.gas_inlet.pressure[0],                 
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "H2O"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "CO2"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "N2"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "O2"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "Ar"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "CH4"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "C2H6"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "C3H8"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "C4H10"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "CO"],
        m.fs.combined_reformer.reformer_mix.gas_inlet.mole_frac_comp[0, "H2"],
        ### air
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.temperature[0],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.flow_mol[0],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.pressure[0],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "H2O"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "CO2"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "N2"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "O2"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "Ar"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "CH4"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "C2H6"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "C3H8"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "C4H10"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "CO"],
        m.fs.combined_reformer.reformer_mix.oxygen_inlet.mole_frac_comp[0, "H2"],
        ### outlet reformer (downstream is reformer_recuperator)
        m.fs.reformer_recuperator.hot_side_inlet.temperature[0],
        m.fs.reformer_recuperator.hot_side_inlet.flow_mol[0],
    ]

    input_vars.extend(m.fs.reformer_recuperator.hot_side_inlet.mole_frac_comp[0, :])

    external_eqns = list(reformer_mixer_igraph.constraints)

    to_exclude = ComponentSet(input_vars)
    to_exclude.add(m.fs.combined_reformer.reformer.lagrange_mult[0, "N"])
    to_exclude.add(m.fs.combined_reformer.reformer.lagrange_mult[0, "Ar"])
    external_vars = [var for var in reformer_mixer_igraph.variables if var not in to_exclude]

    external_var_set = ComponentSet(external_vars)
    external_eqn_set = ComponentSet(external_eqns)
    residual_eqn_set = ComponentSet(residual_eqns)

    # Bounds on variables in the implicit function can lead to
    # undefined derivatives
    for i, var in enumerate(external_vars):
        var.setlb(None)
        var.setub(None)
        var.domain = pyo.Reals

    print("Length of inputs vars is", len(input_vars))
    for var in input_vars: print(var.name)
    print(".......................................")
    print("Length of external vars is", len(external_vars))
    for var in external_vars: print(var.name)
    print(".......................................")
    print("Length of residual eqns is", len(residual_eqns))
    print(".......................................")
    print("Length of external eqns is", len(external_eqns))
    for eqn in external_eqns: print(eqn.name)

    epm = ExternalPyomoModel(
        input_vars,
        external_vars,
        residual_eqns,
        external_eqns,
        # This forces us to use CyIpopt for the inner Newton solver
        #solver_options=dict(solver_class=CyIpoptSolverWrapper),
    )

    ########### CONNECT FLOWSHEET TO THE IMPLICIT AUTOTHERMAL REFORMER ###########

    m_implicit = ConcreteModel()
    m_implicit.egb = ExternalGreyBoxBlock()
    m_implicit.egb.set_external_model(epm, inputs=input_vars)

    fullspace_cons = [
        con
        for con in full_igraph.constraints
        if con not in residual_eqn_set and con not in external_eqn_set
    ]

    fullspace_vars = [
        var for var in full_igraph.variables if var not in external_var_set
    ]

    m_implicit.fullspace_cons = pyo.Reference(fullspace_cons)
    m_implicit.fullspace_vars = pyo.Reference(fullspace_vars)
    m_implicit.objective = pyo.Reference([m.fs.obj])

    return m_implicit

def main(X,P):
    from svi.auto_thermal_reformer.fullspace_fs_new_code_structure import make_optimization_model
    m = make_optimization_model(X, P)
    add_external_function_libraries_to_environment(m)
    m_implicit = make_implicit(m)
    solver = pyo.SolverFactory("cyipopt", options = {"tol": 1e-6, "max_iter": 90})
    results = solver.solve(m_implicit, tee=True)

    m.fs.combined_reformer.reformer_mix.report()
    m.fs.combined_reformer.reformer.report()
    m.fs.product.report()
    m.fs.NG_expander.report()
    m.fs.reformer_bypass.report()


if __name__ == "__main__":
    main(X = 0.90, P = 1447379)