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
from pyomo.common.timing import TicTocTimer, HierarchicalTimer
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
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom

import idaes
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
from pyomo.contrib.incidence_analysis.config import IncidenceMethod
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
    CyIpoptSolverWrapper
)

from svi.external import add_external_function_libraries_to_environment
from svi.auto_thermal_reformer.fullspace_flowsheet import make_optimization_model
import svi.auto_thermal_reformer.config as config

def make_implicit(m, **kwds):
    timer = kwds.pop("timer", HierarchicalTimer())
    ########### CREATE EXTERNAL PYOMO MODEL FOR THE AUTOTHERMAL REFORMER ###########
    full_igraph = IncidenceGraphInterface(m, method=IncidenceMethod.ampl_repn)
    reformer_igraph = IncidenceGraphInterface(
        m.fs.reformer, include_inequality=False, method=IncidenceMethod.ampl_repn
    )

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
        m.fs.reformer.inlet.temperature[0],
        m.fs.reformer.inlet.mole_frac_comp[0, "H2"],
        m.fs.reformer.inlet.mole_frac_comp[0, "CO"],
        m.fs.reformer.inlet.mole_frac_comp[0, "H2O"],
        m.fs.reformer.inlet.mole_frac_comp[0, "CO2"],
        m.fs.reformer.inlet.mole_frac_comp[0, "CH4"],
        m.fs.reformer.inlet.mole_frac_comp[0, "C2H6"],
        m.fs.reformer.inlet.mole_frac_comp[0, "C3H8"],
        m.fs.reformer.inlet.mole_frac_comp[0, "C4H10"],
        m.fs.reformer.inlet.mole_frac_comp[0, "N2"],
        m.fs.reformer.inlet.mole_frac_comp[0, "O2"],
        m.fs.reformer.inlet.mole_frac_comp[0, "Ar"],
        m.fs.reformer.inlet.flow_mol[0],
        m.fs.reformer.inlet.pressure[0],

        m.fs.reformer_recuperator.hot_side_inlet.temperature[0],
        m.fs.reformer_recuperator.hot_side_inlet.flow_mol[0],

        # NOTE:
        # - heat_duty is not necessary as it does not appear elsewhere
        #   in the model
        # - pressure is not an "input" as the pressure linking equation
        #   is not residual (because the reactor's outlet pressure is fixed)
    ]
    input_vars.extend(m.fs.reformer_recuperator.hot_side_inlet.mole_frac_comp[0, :])

    external_eqns = list(reformer_igraph.constraints)

    to_exclude = ComponentSet(input_vars)
    # These variables only appear in reformer_igraph due to a bug in how
    # coefficients with values of zero were handled. This should be fixed in
    # recent Pyomo main, but we leave this explicit filtering in here as we
    # often run using the latest release.
    #
    # NOTE: As of IDAES 2.5, lagrange multipliers for inert components are not
    # included in the Gibbs reactor.
    if (idaes.ver.package_version.major, idaes.ver.package_version.minor) < (2, 5):
        to_exclude.add(m.fs.reformer.lagrange_mult[0, "N"])
        to_exclude.add(m.fs.reformer.lagrange_mult[0, "Ar"])
    external_vars = [var for var in reformer_igraph.variables if var not in to_exclude]

    external_var_set = ComponentSet(external_vars)
    external_eqn_set = ComponentSet(external_eqns)
    residual_eqn_set = ComponentSet(residual_eqns)

    # Bounds on variables in the implicit function can lead to
    # undefined derivatives. However, removing these bounds may make
    # evaluation errors more likely.
    for i, var in enumerate(external_vars):
        var.setlb(None)
        var.setub(None)
        var.domain = pyo.Reals

    epm = ExternalPyomoModel(
        input_vars,
        external_vars,
        residual_eqns,
        external_eqns,
        timer=timer,
        solver_options=dict(use_calc_var=False),
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


if __name__ == "__main__":
    X = 0.95
    P = 1550000.0
    htimer = HierarchicalTimer()
    htimer.start("make-model")
    m = make_optimization_model(X,P)
    htimer.stop("make-model")
    add_external_function_libraries_to_environment(m)
    timer = TicTocTimer()
    timer.tic()
    htimer.start("make-implicit")
    m_implicit = make_implicit(m, timer=htimer)
    htimer.stop("make-implicit")
    timer.toc("made model")
    solver = config.get_optimization_solver()
    print(X,P)
    htimer.start("solve")
    results = solver.solve(m_implicit, tee=True)
    htimer.stop("solve")
    timer.toc("solved model")
    dT = timer.toc("end timer")
    print(htimer)
