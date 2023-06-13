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

######## IMPORT PACKAGES ########

import numpy as np
import random
import pandas as pd
import pyomo.environ as pyo
from idaes.core.util.exceptions import InitializationError
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
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.models.unit_models import (
    Feed,
    Mixer,
    Compressor,
    Heater,
    GibbsReactor,
    Product,
)

from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop

######## SOLVE AN INDIVIDUAL GIBBS REACTOR ########

m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

m.fs.R101 = GibbsReactor(
    property_package=m.fs.thermo_params,
    has_heat_transfer=True,
    has_pressure_change=False,
)

flow_H2O = 234  # mol/s
flow_CH4 = 75  # mol/s
total_flow_in = flow_H2O + flow_CH4

m.fs.R101.inlet.mole_frac_comp[0, "CH4"].fix(flow_CH4 / total_flow_in)
m.fs.R101.inlet.mole_frac_comp[0, "H2"].fix(9.9996e-06)
m.fs.R101.inlet.mole_frac_comp[0, "CO"].fix(9.9996e-06)
m.fs.R101.inlet.mole_frac_comp[0, "CO2"].fix(9.9996e-06)
m.fs.R101.inlet.mole_frac_comp[0, "H2O"].fix(flow_H2O / total_flow_in)

m.fs.R101.conversion = Var(bounds=(0, 1), units=pyunits.dimensionless)  # fraction

m.fs.R101.conv_constraint = Constraint(
    expr=m.fs.R101.conversion
    * m.fs.R101.inlet.flow_mol[0]
    * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
    == (
        m.fs.R101.inlet.flow_mol[0] * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
        - m.fs.R101.outlet.flow_mol[0] * m.fs.R101.outlet.mole_frac_comp[0, "CH4"]
    )
)

m.fs.R101.conversion.fix(0.9)
m.fs.R101.inlet.pressure.setub(1e6)  # Pa
m.fs.R101.inlet.temperature.setub(1000)  # K
m.fs.R101.inlet.flow_mol.fix(total_flow_in)

solver = get_solver()
results = solver.solve(m, tee=True)

########## CREATE AN IMPLICIT GIBBS REACTOR ###########

from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

igraph = IncidenceGraphInterface(m, include_inequality=False)

# Unfix inputs
m.fs.R101.inlet.temperature.unfix()
m.fs.R101.inlet.pressure.unfix()

m.outlet_mole_frac_comp = Var(m.fs.thermo_params.component_list)
for j in m.fs.thermo_params.component_list:
    m.outlet_mole_frac_comp[j] = m.fs.R101.outlet.mole_frac_comp[0, j]

m.outlet_temperature = Var(initialize=m.fs.R101.outlet.temperature[0].value)
m.outlet_heatDuty = Var(initialize=m.fs.R101.heat_duty[0].value)
m.outlet_flow_mol = Var(initialize=m.fs.R101.outlet.flow_mol[0].value)


@m.Constraint(m.fs.thermo_params.component_list)
def outlet_mole_frac_comp_eq(m, j):
    return m.outlet_mole_frac_comp[j] == m.fs.R101.outlet.mole_frac_comp[0, j]


@m.Constraint()
def outlet_temperature_eq(m):
    return m.outlet_temperature == m.fs.R101.outlet.temperature[0]


@m.Constraint()
def outlet_heatDuty_eq(m):
    return m.outlet_heatDuty == m.fs.R101.heat_duty[0]


@m.Constraint()
def outlet_flow_mol_eq(m):
    return m.outlet_flow_mol == m.fs.R101.outlet.flow_mol[0]

#
# Put bounds on residual vars. Then we can remove bounds on external variables
#
for j in m.outlet_mole_frac_comp:
    m.outlet_mole_frac_comp[j].setlb(m.fs.R101.outlet.mole_frac_comp[0, j].lb)
    m.outlet_mole_frac_comp[j].setub(m.fs.R101.outlet.mole_frac_comp[0, j].ub)
m.outlet_temperature.setlb(m.fs.R101.outlet.temperature[0].lb)
m.outlet_temperature.setub(m.fs.R101.outlet.temperature[0].ub)
m.outlet_heatDuty.setlb(m.fs.R101.heat_duty[0].lb)
m.outlet_heatDuty.setub(m.fs.R101.heat_duty[0].ub)

residual_eqns = [
    m.outlet_temperature_eq,
    m.outlet_heatDuty_eq,
    m.outlet_flow_mol_eq,
]
residual_eqns.extend(m.outlet_mole_frac_comp_eq.values())

input_vars = [
    m.fs.R101.inlet.temperature[0],
    m.fs.R101.inlet.pressure[0],
    m.outlet_temperature,
    m.outlet_heatDuty,
    m.outlet_flow_mol,
]
# Since we're not using outlet temperature or flow_mol, can we remove them from
# the "implicit function inputs" (as well as their linking ("residual")
# constraints)?
input_vars.extend(m.outlet_mole_frac_comp.values())

external_eqns = list(igraph.constraints)
external_vars = [
    var for var in igraph.variables if var is not m.fs.R101.inlet.temperature[0]
]
external_vars = [var for var in external_vars if var is not m.fs.R101.inlet.pressure[0]]

from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

epm = ExternalPyomoModel(
    input_vars,
    external_vars,
    residual_eqns,
    external_eqns,
)

########## CONNECT THE IMPLICIT GIBBS REACTOR TO THE ENTIRE FLOWSHEET ###########

m_implicit = ConcreteModel()

m_implicit.egb = ExternalGreyBoxBlock()

m_implicit.egb.set_external_model(epm, inputs=input_vars)

m_implicit.fs = FlowsheetBlock(dynamic=False)
thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
m_implicit.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

m_implicit.fs.CH4 = Feed(property_package=m_implicit.fs.thermo_params)
m_implicit.fs.H2O = Feed(property_package=m_implicit.fs.thermo_params)
m_implicit.fs.M101 = Mixer(
    property_package=m_implicit.fs.thermo_params,
    inlet_list=["methane_feed", "steam_feed"],
)
m_implicit.fs.H101 = Heater(
    property_package=m_implicit.fs.thermo_params,
    has_pressure_change=False,
    has_phase_equilibrium=False,
)

m_implicit.fs.C101 = Compressor(property_package=m_implicit.fs.thermo_params)

m_implicit.fs.s01 = Arc(
    source=m_implicit.fs.CH4.outlet, destination=m_implicit.fs.M101.methane_feed
)
m_implicit.fs.s02 = Arc(
    source=m_implicit.fs.H2O.outlet, destination=m_implicit.fs.M101.steam_feed
)
m_implicit.fs.s03 = Arc(
    source=m_implicit.fs.M101.outlet, destination=m_implicit.fs.C101.inlet
)
m_implicit.fs.s04 = Arc(
    source=m_implicit.fs.C101.outlet, destination=m_implicit.fs.H101.inlet
)

TransformationFactory("network.expand_arcs").apply_to(m_implicit)

m_implicit.fs.CH4.outlet.mole_frac_comp[0, "CH4"].fix(1)
m_implicit.fs.CH4.outlet.mole_frac_comp[0, "H2O"].fix(1e-5)
m_implicit.fs.CH4.outlet.mole_frac_comp[0, "H2"].fix(1e-5)
m_implicit.fs.CH4.outlet.mole_frac_comp[0, "CO"].fix(1e-5)
m_implicit.fs.CH4.outlet.mole_frac_comp[0, "CO2"].fix(1e-5)
m_implicit.fs.CH4.outlet.flow_mol.fix(75 * pyunits.mol / pyunits.s)
m_implicit.fs.CH4.outlet.temperature.fix(298.15 * pyunits.K)
m_implicit.fs.CH4.outlet.pressure.fix(1e5 * pyunits.Pa)

m_implicit.fs.H2O.outlet.mole_frac_comp[0, "CH4"].fix(1e-5)
m_implicit.fs.H2O.outlet.mole_frac_comp[0, "H2O"].fix(1)
m_implicit.fs.H2O.outlet.mole_frac_comp[0, "H2"].fix(1e-5)
m_implicit.fs.H2O.outlet.mole_frac_comp[0, "CO"].fix(1e-5)
m_implicit.fs.H2O.outlet.mole_frac_comp[0, "CO2"].fix(1e-5)
m_implicit.fs.H2O.outlet.flow_mol.fix(234 * pyunits.mol / pyunits.s)
m_implicit.fs.H2O.outlet.temperature.fix(373.15 * pyunits.K)
m_implicit.fs.H2O.outlet.pressure.fix(1e5 * pyunits.Pa)

m_implicit.compressor_efficiency = Constraint(
    expr=m_implicit.fs.C101.efficiency_isentropic[0] == 0.9
)


@m_implicit.Constraint()
def linking_T_to_egb(m_implicit):
    return m_implicit.fs.H101.outlet.temperature[0] == m_implicit.egb.inputs[0]


@m_implicit.Constraint()
def linking_P_to_egb(m_implicit):
    return m_implicit.fs.H101.outlet.pressure[0] == m_implicit.egb.inputs[1]


# set objective

m_implicit.fs.cooling_cost = Expression(expr=0.212e-7 * (m_implicit.egb.inputs[3]))
m_implicit.fs.heating_cost = Expression(expr=2.2e-7 * m_implicit.fs.H101.heat_duty[0])
m_implicit.fs.compression_cost = Expression(
    expr=0.12e-5 * m_implicit.fs.C101.work_isentropic[0]
)
m_implicit.fs.operating_cost = Expression(
    expr=(
        3600
        * 8000
        (value(m_implicit.fs.objective))
m_implicit.fs.objective.display()
print(m_implicit.fs.C101.report())
print(m_implicit.fs.H101.report())
m_implicit.egb.inputs.display()
