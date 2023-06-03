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


m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

m.fs.R101 = GibbsReactor(
    property_package=m.fs.thermo_params,
    has_heat_transfer=True,
    has_pressure_change=False,
)

Pin_min = 0.8e6  # Pa
Pin_max = 1.2e6  # Pa
Tin_min = 500  # K
Tin_max = 700  # K
conversion_CH4_min = 0.5
conversion_CH4_max = 0.98

flow_H2O = 234  # mol/s
flow_CH4 = 75  # mol/s
total_flow_in = flow_H2O + flow_CH4

m.fs.R101.inlet.mole_frac_comp[0, "CH4"].fix(flow_CH4 / total_flow_in)
m.fs.R101.inlet.mole_frac_comp[0, "H2"].fix(9.9996e-06)
m.fs.R101.inlet.mole_frac_comp[0, "CO"].fix(9.9996e-06)
m.fs.R101.inlet.mole_frac_comp[0, "CO2"].fix(9.9996e-06)
m.fs.R101.inlet.mole_frac_comp[0, "H2O"].fix(flow_H2O / total_flow_in)

m.fs.R101.conversion = Var(bounds=(0, 1), units=pyunits.dimensionless)

m.fs.R101.conv_constraint = Constraint(
    expr=m.fs.R101.conversion * total_flow_in * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
    == (
        total_flow_in * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
        - m.fs.R101.outlet.flow_mol[0] * m.fs.R101.outlet.mole_frac_comp[0, "CH4"]
    )
)

Tin = random.uniform(Tin_min, Tin_max)
Pin = random.uniform(Pin_min, Pin_max)
conversion_CH4 = random.uniform(conversion_CH4_min, conversion_CH4_max)
m.fs.R101.inlet.temperature.fix(Tin)
m.fs.R101.inlet.flow_mol.fix(total_flow_in)
m.fs.R101.inlet.pressure.fix(Pin)
m.fs.R101.conversion.fix(conversion_CH4)
m.fs.R101.initialize()
solver = get_solver()
results = solver.solve(m, tee=True)

m.fs.R101.inlet.pprint()
m.fs.R101.outlet.pprint()

m.fs.R101.outlet.mole_frac_comp.pprint()
m.fs.R101.outlet.flow_mol.pprint()
m.fs.R101.outlet.temperature.pprint()
m.fs.R101.outlet.pressure.pprint()

m.fs.R101.inlet.flow_mol.pprint()

from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

igraph = IncidenceGraphInterface(m, include_inequality=False)

# Unfix inputs
m.fs.R101.inlet.flow_mol.unfix()

# Define "auxiliary" outlet variables, which will be "inputs" to the
# system of constraints defined by the implicit function.
# Auxiliary variables are necessary because the original outlet variables
# are defined by the "external equations" (the equations of the reactor),
# so they can't also be "inputs" to the equations defined by the implicit
# function.
m.outlet_mole_frac_comp = pyo.Var(m.fs.thermo_params.component_list)
for j in m.fs.thermo_params.component_list:
    m.outlet_mole_frac_comp[j] = m.fs.R101.outlet.mole_frac_comp[0, j]

m.outlet_temperature = pyo.Var(initialize=m.fs.R101.outlet.temperature[0].value)
m.outlet_pressure = pyo.Var(initialize=m.fs.R101.outlet.pressure[0].value)
m.outlet_flow_mol = pyo.Var(initialize=m.fs.R101.outlet.flow_mol[0].value)

#
# Set up constraints to link the original outlet variables with the new
#
@m.Constraint(m.fs.thermo_params.component_list)
def outlet_mole_frac_comp_eq(m, j):
    return m.outlet_mole_frac_comp[j] == m.fs.R101.outlet.mole_frac_comp[0, j]

@m.Constraint()
def outlet_temperature_eq(m):
    return m.outlet_temperature == m.fs.R101.outlet.temperature[0]

@m.Constraint()
def outlet_pressure_eq(m):
    return m.outlet_pressure == m.fs.R101.outlet.pressure[0]

@m.Constraint()
def outlet_flow_mol_eq(m):
    return m.outlet_flow_mol == m.fs.R101.outlet.flow_mol[0]

#
# Define some simple optimization problem
#
# Note that optimization problem needs to be defined in terms of the
# "auxiliary" outlet variables.
obj_expr = 0.0
for idx in m.fs.R101.outlet.mole_frac_comp:
    t, j = idx
    obj_expr += (
        m.outlet_mole_frac_comp[j]
        - m.fs.R101.outlet.mole_frac_comp[idx].value
    )**2
obj_expr += (m.outlet_temperature - 1000.0)**2
obj_expr += (m.outlet_pressure - m.fs.R101.outlet.pressure[0].value)**2
obj_expr += (m.outlet_flow_mol - 410.0)**2

# Objective can be defined in terms of the original inlet variable
obj_expr += (m.fs.R101.inlet.flow_mol[0] - 350.0)**2

m.objective = pyo.Objective(expr=obj_expr)


# This was just to confirm that we can solve the original, full-space problem
#solver.solve(m, tee=True)


#
# Define components necessary for ExternalPyomoModel
#
residual_eqns = [
    m.outlet_temperature_eq,
    m.outlet_pressure_eq,
    m.outlet_flow_mol_eq,
]
residual_eqns.extend(m.outlet_mole_frac_comp_eq.values())

# Note that input variables contain both the inlets (only flow_mol is unfixed)
# and outlets. This is a bit nonintuitive, as we would expect that only the
# inlets are inputs to the reactor. However, outlets are inputs to the system
# of equations defined by the reactor. This is admittedly a fairly clumsy
# way to define an implicit function.
# A better way might be to explicitly support outputs.
input_vars = [
    m.fs.R101.inlet.flow_mol[0],
    m.outlet_temperature,
    m.outlet_pressure,
    m.outlet_flow_mol,
]
input_vars.extend(m.outlet_mole_frac_comp.values())

external_eqns = list(igraph.constraints)
external_vars = [var for var in igraph.variables if var is not m.fs.R101.inlet.flow_mol[0]]


from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
epm = ExternalPyomoModel(
    input_vars,
    external_vars,
    residual_eqns,
    external_eqns,
)

#
# Set up the "implicit formulation"
#
# Note that none of the bounds/inequalities on the model have been added
# to this formulation. This will need to change.
# If we had more than just the reactor in this model, we would need to add
# the rest of the model, not just the "input variables", to m_implicit
m_implicit = pyo.ConcreteModel()
m_implicit.input_vars = pyo.Reference(input_vars)
m_implicit.objective = pyo.Reference(m.objective)
m_implicit.egb = ExternalGreyBoxBlock()

m_implicit.egb.set_external_model(epm, inputs=input_vars)

solver = pyo.SolverFactory("cyipopt")
solver.solve(m_implicit, tee=True)

m.fs.R101.outlet.mole_frac_comp.pprint()
m.fs.R101.outlet.flow_mol.pprint()
m.fs.R101.outlet.temperature.pprint()
m.fs.R101.outlet.pressure.pprint()

m.fs.R101.inlet.flow_mol.pprint()
