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
import pyomo.environ as pyo
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

#################################################
################################################# Create Original Flowsheet
#################################################

m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

m.fs.CH4 = Feed(property_package=m.fs.thermo_params)
m.fs.H2O = Feed(property_package=m.fs.thermo_params)
m.fs.M101 = Mixer(
    property_package=m.fs.thermo_params, inlet_list=["methane_feed", "steam_feed"]
)
m.fs.H101 = Heater(
    property_package=m.fs.thermo_params,
    has_pressure_change=False,
    has_phase_equilibrium=False,
)
m.fs.C101 = Compressor(property_package=m.fs.thermo_params)

m.fs.R101 = GibbsReactor(
    property_package=m.fs.thermo_params,
    has_heat_transfer=True,
    has_pressure_change=False,
)

m.fs.s01 = Arc(source=m.fs.CH4.outlet, destination=m.fs.M101.methane_feed)
m.fs.s02 = Arc(source=m.fs.H2O.outlet, destination=m.fs.M101.steam_feed)
m.fs.s03 = Arc(source=m.fs.M101.outlet, destination=m.fs.C101.inlet)
m.fs.s04 = Arc(source=m.fs.C101.outlet, destination=m.fs.H101.inlet)
m.fs.s05 = Arc(source=m.fs.H101.outlet, destination=m.fs.R101.inlet)

TransformationFactory("network.expand_arcs").apply_to(m)

m.fs.CH4.outlet.mole_frac_comp[0, "CH4"].fix(1)
m.fs.CH4.outlet.mole_frac_comp[0, "H2O"].fix(1e-5)
m.fs.CH4.outlet.mole_frac_comp[0, "H2"].fix(1e-5)
m.fs.CH4.outlet.mole_frac_comp[0, "CO"].fix(1e-5)
m.fs.CH4.outlet.mole_frac_comp[0, "CO2"].fix(1e-5)
m.fs.CH4.outlet.flow_mol.fix(75 * pyunits.mol / pyunits.s)
m.fs.CH4.outlet.temperature.fix(298.15 * pyunits.K)
m.fs.CH4.outlet.pressure.fix(1e5 * pyunits.Pa)

m.fs.H2O.outlet.mole_frac_comp[0, "CH4"].fix(1e-5)
m.fs.H2O.outlet.mole_frac_comp[0, "H2O"].fix(1)
m.fs.H2O.outlet.mole_frac_comp[0, "H2"].fix(1e-5)
m.fs.H2O.outlet.mole_frac_comp[0, "CO"].fix(1e-5)
m.fs.H2O.outlet.mole_frac_comp[0, "CO2"].fix(1e-5)
m.fs.H2O.outlet.flow_mol.fix(234 * pyunits.mol / pyunits.s)
m.fs.H2O.outlet.temperature.fix(373.15 * pyunits.K)
m.fs.H2O.outlet.pressure.fix(1e5 * pyunits.Pa)
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

m.fs.CH4.initialize()
propagate_state(arc=m.fs.s01)

m.fs.H2O.initialize()
propagate_state(arc=m.fs.s02)

m.fs.M101.initialize()
propagate_state(arc=m.fs.s03)

m.fs.C101.initialize()
propagate_state(arc=m.fs.s04)

m.fs.H101.initialize()
propagate_state(arc=m.fs.s05)

m.fs.C101.efficiency_isentropic_constraint = Constraint(expr = m.fs.C101.efficiency_isentropic[0] == 0.9)
m.fs.R101.conversion_constraint = Constraint(expr = m.fs.R101.conversion == 0.9)

#################################################
################################################# Set Up the Implicit Model for the Gibbs Reactor
#################################################

## Here I changed part of your code. Instead of inlet flow being a DOF, it's T and P to the reactor.
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

igraph = IncidenceGraphInterface(m, include_inequality=False)

# Unfix inputs
m.fs.R101.inlet.temperature.unfix()
m.fs.R101.inlet.pressure.unfix()

# Define "auxiliary" outlet variables, which will be "inputs" to the
# system of constraints defined by the implicit function.
# Auxiliary variables are necessary because the original outlet variables
# are defined by the "external equations" (the equations of the reactor),
# so they can't also be "inputs" to the equations defined by the implicit
# function.
m.outlet_mole_frac_comp = Var(m.fs.thermo_params.component_list)
for j in m.fs.thermo_params.component_list:
    m.outlet_mole_frac_comp[j] = m.fs.R101.outlet.mole_frac_comp[0, j]

m.outlet_temperature = Var(initialize=m.fs.R101.outlet.temperature[0].value)
m.outlet_heatDuty = Var(initialize=m.fs.R101.heat_duty[0].value)
m.outlet_flow_mol = Var(initialize=m.fs.R101.outlet.flow_mol[0].value)

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
def outlet_heatDuty_eq(m):
    return m.outlet_heatDuty == m.fs.R101.heat_duty[0]

@m.Constraint()
def outlet_flow_mol_eq(m):
    return m.outlet_flow_mol == m.fs.R101.outlet.flow_mol[0]

residual_eqns = [
    m.outlet_temperature_eq,
    m.outlet_heatDuty_eq,
    m.outlet_flow_mol_eq,
]
residual_eqns.extend(m.outlet_mole_frac_comp_eq.values())

# Note that input variables contain both the inlets (only T and P are unfixed)
# and outlets. This is a bit nonintuitive, as we would expect that only the
# inlets are inputs to the reactor. However, outlets are inputs to the system
# of equations defined by the reactor. This is admittedly a fairly clumsy
# way to define an implicit function.
# A better way might be to explicitly support outputs.
input_vars = [
    m.fs.R101.inlet.temperature[0],
    m.fs.R101.inlet.pressure[0],
    m.outlet_temperature,
    m.outlet_heatDuty,
    m.outlet_flow_mol,
]
input_vars.extend(m.outlet_mole_frac_comp.values())

external_eqns = list(igraph.constraints)
external_vars = [var for var in igraph.variables if var is not m.fs.R101.inlet.temperature[0]]
external_vars = [var for var in external_vars if var is not m.fs.R101.inlet.pressure[0]]

from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
epm = ExternalPyomoModel(
    input_vars,
    external_vars,
    residual_eqns,
    external_eqns,
)

#################################################
################################################# Insert the Implicit Model for Gibbs into the entire flowsheet.
#################################################

# Set up the "implicit formulation"
# Note that none of the bounds/inequalities on the model have been added
# to this formulation. This will need to change.
# If we had more than just the reactor in this model, we would need to add
# the rest of the model, not just the "input variables", to m_implicit

m_implicit = ConcreteModel()

m_implicit.input_vars = pyo.Reference(input_vars)

m_implicit.egb = ExternalGreyBoxBlock()

m_implicit.egb.set_external_model(epm, inputs=input_vars)

m_implicit.fs = FlowsheetBlock(dynamic=False)
thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
m_implicit.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

m_implicit.fs.CH4 = Feed(property_package=m_implicit.fs.thermo_params)
m_implicit.fs.H2O = Feed(property_package=m_implicit.fs.thermo_params)
m_implicit.fs.M101 = Mixer(
    property_package=m_implicit.fs.thermo_params, inlet_list=["methane_feed", "steam_feed"]
)
m_implicit.fs.H101 = Heater(
    property_package=m_implicit.fs.thermo_params,
    has_pressure_change=False,
    has_phase_equilibrium=False,
)

m_implicit.fs.C101 = Compressor(property_package=m_implicit.fs.thermo_params)

m_implicit.fs.s01 = Arc(source=m_implicit.fs.CH4.outlet, destination=m_implicit.fs.M101.methane_feed)
m_implicit.fs.s02 = Arc(source=m_implicit.fs.H2O.outlet, destination=m_implicit.fs.M101.steam_feed)
m_implicit.fs.s03 = Arc(source=m_implicit.fs.M101.outlet, destination=m_implicit.fs.C101.inlet)
m_implicit.fs.s04 = Arc(source=m_implicit.fs.C101.outlet, destination=m_implicit.fs.H101.inlet)

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

m_implicit.conversion = Var(bounds=(0, 1), units=pyunits.dimensionless)  # fraction

m_implicit.conv_constraint = Constraint(
    expr=m_implicit.conversion
    * m_implicit.fs.H101.outlet.flow_mol[0]
    * m_implicit.fs.H101.outlet.mole_frac_comp[0, "CH4"]
    == (
        m_implicit.fs.H101.outlet.flow_mol[0] * m_implicit.fs.H101.outlet.mole_frac_comp[0, "CH4"]
        - m_implicit.egb.inputs[4].value * m_implicit.egb.inputs[5].value
    )
)

# m_implicit.input_vars[4].value == Outlet Molar Flow Rate of Implicit Gibbs 
# m_implicit.input_vars[5].value == Outlet Molar CH4 Composition of Implicit Gibbs 

m_implicit.compressor_efficiency = Constraint(expr = m_implicit.fs.C101.efficiency_isentropic[0] == 0.9)
m_implicit.conversion1 = Constraint(expr = m_implicit.conversion == 0.9)

# set objective

m_implicit.fs.cooling_cost = Expression(expr=0.212e-7 * (m_implicit.egb.inputs[3].value))  
m_implicit.fs.heating_cost = Expression(expr=2.2e-7 * m_implicit.fs.H101.heat_duty[0])
m_implicit.fs.compression_cost = Expression(expr=0.12e-5 * m_implicit.fs.C101.work_isentropic[0])
m_implicit.fs.operating_cost = Expression(expr=(3600 * 8000 * (m_implicit.fs.heating_cost + m_implicit.fs.cooling_cost + m_implicit.fs.compression_cost)))
m_implicit.fs.objective = Objective(expr=m_implicit.fs.operating_cost)

# Link inputs to the implicit reactor to the outputs of H101.
m_implicit.link_R101_imp1 = Constraint(expr = m.fs.R101.inlet.temperature[0] == m_implicit.fs.H101.outlet.temperature[0])
m_implicit.link_R101_imp2= Constraint(expr = m.fs.R101.inlet.pressure[0] == m_implicit.fs.H101.outlet.pressure[0])
#m_implicit.link_R101_imp3 = Constraint(expr = m.fs.R101.inlet.flow_mol[0] == m_implicit.fs.H101.outlet.flow_mol[0])
# @m_implicit.Constraint(m_implicit.fs.thermo_params.component_list)
# def outlet_mole_frac_comp_eq(m_implicit, j):
#     return m_implicit.fs.H101.outlet.mole_frac_comp[0, j] == m.fs.R101.inlet.mole_frac_comp[0, j]

# Initialize and solve each unit operation
m_implicit.fs.CH4.initialize()
propagate_state(arc=m_implicit.fs.s01)

m_implicit.fs.H2O.initialize()
propagate_state(arc=m_implicit.fs.s02)

m_implicit.fs.M101.initialize()
propagate_state(arc=m_implicit.fs.s03)

m_implicit.fs.C101.initialize()
propagate_state(arc=m_implicit.fs.s04)

solver = pyo.SolverFactory("cyipopt")
solver.solve(m_implicit, tee=True)