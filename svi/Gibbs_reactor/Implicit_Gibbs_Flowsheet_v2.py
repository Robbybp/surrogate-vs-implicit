##########################
########################## FIRST: BUILD A GIBBS REACTOR
##########################
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
m.fs.R101.inlet.pressure.setub(1e6)
m.fs.R101.inlet.temperature.setub(1000)
m.fs.R101.inlet.flow_mol.fix(total_flow_in)

# YOU GET THESE RESULTS FOR R101:

# ====================================================================================
# Unit : fs.R101                                                             Time: 0.0
# ------------------------------------------------------------------------------------
#     Unit Performance

#     Variables: 

#     Key       : Value  : Units : Fixed : Bounds
#     Heat Duty : 0.0000 :  watt : False : (None, None)

# ------------------------------------------------------------------------------------
#     Stream Table
#                                 Units         Inlet     Outlet  
#     Total Molar Flowrate     mole / second     309.00     8000.0
#     Total Mole Fraction CH4  dimensionless    0.24272    0.20000
#     Total Mole Fraction H2O  dimensionless    0.75728    0.20000
#     Total Mole Fraction H2   dimensionless 9.9996e-06    0.20000
#     Total Mole Fraction CO   dimensionless 9.9996e-06    0.20000
#     Total Mole Fraction CO2  dimensionless 9.9996e-06    0.20000
#     Temperature                     kelvin     500.00     500.00
#     Pressure                        pascal 1.3000e+05 1.3000e+05
# ====================================================================================

##########################
########################## SECOND: BUILD FLOWSHEET AND LINK FLOWSHEET VARS TO THE IMPLICIT MODEL
##########################

from pyomo.contrib.incidence_analysis import IncidenceGraphInterface

igraph = IncidenceGraphInterface(m, include_inequality=False)

# Unfix inputs: In this case, the DOF of the optimization problem are T and P inlet to the reactor.
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

# Note that input variables contain both the inlets (only flow_mol is unfixed)
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

#
# Set up the "implicit formulation"
#
# Note that none of the bounds/inequalities on the model have been added
# to this formulation. This will need to change.
# If we had more than just the reactor in this model, we would need to add
# the rest of the model, not just the "input variables", to m_implicit

m_implicit = ConcreteModel()

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

m_implicit.compressor_efficiency = Constraint(expr = m_implicit.fs.C101.efficiency_isentropic[0] == 0.9)

# LINK HEAT EXCHANGER OUTPUTS TO THE INPUTS OF THE IMPLICIT BLOCK
@m_implicit.Constraint()
def linking_T_to_egb(m_implicit):
    return m_implicit.fs.H101.outlet.temperature[0] == m_implicit.egb.inputs[0].value

@m_implicit.Constraint()
def linking_P_to_egb(m_implicit):
    return m_implicit.fs.H101.outlet.pressure[0] == m_implicit.egb.inputs[1].value

# SET OBJECTIVE 
m_implicit.fs.cooling_cost = Expression(expr=0.212e-7 * (m_implicit.egb.inputs[3].value))  
m_implicit.fs.heating_cost = Expression(expr=2.2e-7 * m_implicit.fs.H101.heat_duty[0])
m_implicit.fs.compression_cost = Expression(expr=0.12e-5 * m_implicit.fs.C101.work_isentropic[0])
m_implicit.fs.operating_cost = Expression(expr=(3600 * 8000 * (m_implicit.fs.heating_cost + m_implicit.fs.cooling_cost + m_implicit.fs.compression_cost)))
m_implicit.fs.objective = Objective(expr=m_implicit.fs.operating_cost)

# INITIALIZE AND SOLVE EACH UNIT OPERATION
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

# THIS CODE ABOVE RUNS
###############################################################################
# THE SOLUTION OF THE IMPLICIT BLOCK IS:
# inputs : Size=10, Index=egb.inputs_index, ReferenceTo=[<pyomo.core.base.var.ScalarVar object at 0x000001337FB7F0D0>, <pyomo.core.base.var.ScalarVar object at 0x000001337FB7EEA0>, <pyomo.core.base.var.ScalarVar object at 0x000001337FB7C510>, <pyomo.core.base.var.ScalarVar object at 0x000001337FB7C040>, <pyomo.core.base.var.ScalarVar object at 0x000001337FA9B060>, <pyomo.core.base.var._GeneralVarData object at 0x000001337FB7C0B0>, <pyomo.core.base.var._GeneralVarData object at 0x000001337FB7C4A0>, <pyomo.core.base.var._GeneralVarData object at 0x000001337FB7C2E0>, <pyomo.core.base.var._GeneralVarData object at 0x000001337FB7C270>, <pyomo.core.base.var._GeneralVarData object at 0x000001337FB7C200>]
#     Key : Lower   : Value                : Upper     : Fixed : Stale : Domain
#       0 :  273.15 :      636.57500363425 :      1000 : False : False : NonNegativeReals
#       1 : 50000.0 :             525000.0 : 1000000.0 : False : False : NonNegativeReals
#       2 :    None :   1033.5142306594703 :      None : False : False :            Reals
#       3 :    None :    19853302.77135539 :      None : False : False :            Reals
#       4 :    None :       444.0092696292 :      None : False : False :            Reals
#       5 :    None : 0.016891539238051003 :      None : False : False :            Reals
#       6 :    None :  0.30999431995769683 :      None : False : False :            Reals
#       7 :    None :   0.5210763695902061 :      None : False : False :            Reals
#       8 :    None :  0.08703296105121716 :      None : False : False :            Reals
#       9 :    None :  0.06500481016282905 :      None : False : False :            Reals

# THIS SOLUTION MAKES SENSE. HOWEVER, IT SEEMS LIKE THE HEAT EXCHANGER IS NOT CONNECTING PROPERLY 
# TO THE IMPLICIT BLOCK.

# ====================================================================================
# Unit : fs.H101                                                             Time: 0.0
# ------------------------------------------------------------------------------------
#     Unit Performance

#     Variables: 

#     Key       : Value      : Units : Fixed : Bounds
#     Heat Duty : 1.3751e+06 :  watt : False : (None, None)

# ------------------------------------------------------------------------------------
#     Stream Table
#                                 Units         Inlet     Outlet  
#     Total Molar Flowrate     mole / second     309.01     309.01
#     Total Mole Fraction CH4  dimensionless    0.24272    0.24272
#     Total Mole Fraction H2O  dimensionless    0.75725    0.75725
#     Total Mole Fraction H2   dimensionless 9.9996e-06 9.9996e-06
#     Total Mole Fraction CO   dimensionless 9.9996e-06 9.9996e-06
#     Total Mole Fraction CO2  dimensionless 9.9996e-06 9.9996e-06
#     Temperature                     kelvin     379.09     500.00
#     Pressure                        pascal 1.3000e+05 1.3000e+05
# ====================================================================================

## THE OUTLET TEMPERATURE AND PRESSURE OF H101 IS 500 K and 130000 Pa, WHICH COINCIDENTALLY
## ARE THE SAME RESULTS THAT WE OBTAINED IN STEP 1 (SEE LINES 97 AND 98). HOWEVER,
## THE INLET TEMPERATURE AND PRESSURE TO THE IMPLICIT BLOCK IS 636.57 k AND 525000 Pa (SEE LINES 276-277)

## BOTTOM LINE, THERE IS NOT AN AGREEMENT BETWEEN THE INLET T AND P FOR THE IMPLICIT BLOCK, AND
## THE OUTLET T AND P OF H101.