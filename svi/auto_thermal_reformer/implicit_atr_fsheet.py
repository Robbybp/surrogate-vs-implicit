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

from idaes.models.unit_models import (
    Mixer,
    Heater,
    HeatExchanger,
    PressureChanger,
    GibbsReactor,
    Separator,
    Feed,
    Product)
from idaes.core.util.initialization import propagate_state
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
from idaes.core.solvers import get_solver
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

def build_atr_flowsheet(conversion):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    ########## ADD THERMODYNAMIC PROPERTIES ##########  
    components = ['H2', 'CO', "H2O", 'CO2', 'CH4', "C2H6", "C3H8", "C4H10",'N2', 'O2', 'Ar']
    thermo_props_config_dict = get_prop(components = components)
    m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

    ########## ADD FEED AND PRODUCT STREAMS ##########  
    m.fs.feed = Feed(property_package = m.fs.thermo_params)
    m.fs.product = Product(property_package = m.fs.thermo_params)

    ########## ADD UNIT MODELS ##########  
    m.fs.reformer_recuperator = HeatExchanger(
        delta_temperature_callback = delta_temperature_underwood_callback,
        hot_side_name="shell", # hot fluid enters shell
        cold_side_name="tube", # cold fluid enters tube
        shell = {"property_package": m.fs.thermo_params},
        tube = {"property_package": m.fs.thermo_params})

    m.fs.NG_expander = PressureChanger(
        compressor = False,
        property_package = m.fs.thermo_params,
        thermodynamic_assumption = ThermodynamicAssumption.isentropic)

    m.fs.reformer_bypass = Separator(
        outlet_list = ["reformer_outlet", "bypass_outlet"],
        property_package = m.fs.thermo_params)

    m.fs.air_compressor_s1 = PressureChanger(
        compressor = True,
        property_package = m.fs.thermo_params,
        thermodynamic_assumption = ThermodynamicAssumption.isentropic)

    m.fs.intercooler_s1 = Heater(
        property_package = m.fs.thermo_params,
        has_pressure_change = True)

    m.fs.air_compressor_s2 = PressureChanger(
        compressor = True,
        property_package = m.fs.thermo_params,
        thermodynamic_assumption = ThermodynamicAssumption.isentropic)

    m.fs.intercooler_s2 = Heater(
        property_package = m.fs.thermo_params,
        has_pressure_change = True)

    m.fs.reformer_mix = Mixer(
        inlet_list = ["gas_inlet", "oxygen_inlet", "steam_inlet"],
        property_package = m.fs.thermo_params)

    m.fs.reformer = GibbsReactor(
        has_heat_transfer = True,
        has_pressure_change = True,
        inert_species = ["N2", "Ar"],
        property_package =  m.fs.thermo_params)

    m.fs.bypass_rejoin = Mixer(
        inlet_list = ["syngas_inlet", "bypass_inlet"],
        property_package = m.fs.thermo_params)


    ########## CONNECT UNIT MODELS ##########  

    m.fs.RECUP_COLD_IN = Arc(source=m.fs.feed.outlet, destination=m.fs.reformer_recuperator.tube_inlet)
    m.fs.RECUP_COLD_OUT = Arc(source=m.fs.reformer_recuperator.tube_outlet, destination=m.fs.NG_expander.inlet)
    m.fs.NG_EXPAND_OUT = Arc(source=m.fs.NG_expander.outlet, destination=m.fs.reformer_bypass.inlet)
    m.fs.NG_TO_REF = Arc(source=m.fs.reformer_bypass.reformer_outlet, destination=m.fs.reformer_mix.gas_inlet)
    m.fs.STAGE_1_OUT = Arc(source=m.fs.air_compressor_s1.outlet, destination=m.fs.intercooler_s1.inlet)
    m.fs.IC_1_OUT = Arc(source=m.fs.intercooler_s1.outlet, destination=m.fs.air_compressor_s2.inlet)
    m.fs.STAGE_2_OUT = Arc(source=m.fs.air_compressor_s2.outlet, destination=m.fs.intercooler_s2.inlet)
    m.fs.IC_2_OUT = Arc(source=m.fs.intercooler_s2.outlet, destination=m.fs.reformer_mix.oxygen_inlet)
    m.fs.REF_IN = Arc(source=m.fs.reformer_mix.outlet, destination=m.fs.reformer.inlet)
    m.fs.REF_OUT = Arc(source=m.fs.reformer.outlet, destination=m.fs.reformer_recuperator.shell_inlet)
    m.fs.RECUP_HOT_OUT = Arc(source=m.fs.reformer_recuperator.shell_outlet, destination=m.fs.bypass_rejoin.syngas_inlet)
    m.fs.REF_BYPASS = Arc(source=m.fs.reformer_bypass.bypass_outlet, destination=m.fs.bypass_rejoin.bypass_inlet)
    m.fs.PRODUCT = Arc(source=m.fs.bypass_rejoin.outlet, destination=m.fs.product.inlet)

    ########## EXPAND ARCS ##########  

    pyo.TransformationFactory("network.expand_arcs").apply_to(m.fs)

    ########## NATURAL GAS FEED CONDITIONS ##########

    m.fs.feed.outlet.flow_mol.fix(1161.9)  # mol/s
    m.fs.feed.outlet.temperature.fix(288.15)  # K
    m.fs.feed.outlet.pressure.fix(3447379) # Pa
    m.fs.feed.outlet.mole_frac_comp[0, 'CH4'].fix(0.931)
    m.fs.feed.outlet.mole_frac_comp[0, 'C2H6'].fix(0.032)
    m.fs.feed.outlet.mole_frac_comp[0, 'C3H8'].fix(0.007)
    m.fs.feed.outlet.mole_frac_comp[0, 'C4H10'].fix(0.004)
    m.fs.feed.outlet.mole_frac_comp[0, 'CO'].fix(1e-5)
    m.fs.feed.outlet.mole_frac_comp[0, 'CO2'].fix(0.01)
    m.fs.feed.outlet.mole_frac_comp[0, 'H2'].fix(1e-5)
    m.fs.feed.outlet.mole_frac_comp[0, 'H2O'].fix(1e-5)
    m.fs.feed.outlet.mole_frac_comp[0, 'N2'].fix(0.016)
    m.fs.feed.outlet.mole_frac_comp[0, 'O2'].fix(1e-5)
    m.fs.feed.outlet.mole_frac_comp[0, 'Ar'].fix(1e-5)

    ########## REFORMER RECUPERATOR CONDITIONS ##########

    m.fs.reformer_recuperator.area.fix(4190) # m2
    m.fs.reformer_recuperator.overall_heat_transfer_coefficient.fix(80)

    ########## NATURAL GAS EXPANDER CONDITIONS ##########

    m.fs.NG_expander.outlet.pressure.fix(203396) # Pa
    m.fs.NG_expander.efficiency_isentropic.fix(0.9)

    ########## AIR CONDITIONS ##########

    m.fs.air_compressor_s1.inlet.flow_mol.fix(1332.9)  # mol/s
    m.fs.air_compressor_s1.inlet.temperature.fix(288.15)  # K
    m.fs.air_compressor_s1.inlet.pressure.fix(101353)  # Pa 
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'CO2'].fix(0.0003)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'H2O'].fix(0.0104)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'N2'].fix(0.7722)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'O2'].fix(0.2077)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'Ar'].fix(0.00939)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'CH4'].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'C2H6'].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'C3H8'].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'C4H10'].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'CO'].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'H2'].fix(1e-6)

    ########## AIR COMPRESSORS AND INTERCOOLERS ##########

    m.fs.air_compressor_s1.outlet.pressure.fix(144790) # Pa
    m.fs.air_compressor_s1.efficiency_isentropic.fix(0.84)

    m.fs.intercooler_s1.outlet.temperature.fix(310.93) # K
    m.fs.intercooler_s1.outlet.pressure.fix(141343) # Pa equivalent to a dP of -0.5 psi

    m.fs.air_compressor_s2.outlet.pressure.fix(206843) # Pa
    m.fs.air_compressor_s2.efficiency_isentropic.fix(0.84)

    m.fs.intercooler_s2.outlet.temperature.fix(310.93) # K
    m.fs.intercooler_s2.outlet.pressure.fix(203396) # Pa equivalent to a dP of -0.5 psi

    ########## STEAM CONDITIONS ##########

    m.fs.reformer_mix.steam_inlet.flow_mol.fix(464.77) # mol/s
    m.fs.reformer_mix.steam_inlet.temperature.fix(422) # K
    m.fs.reformer_mix.steam_inlet.pressure.fix(203396)  # Pa
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'H2O'].fix(0.9999)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'CO2'].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'N2'].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'O2'].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'Ar'].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'CH4'].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'C2H6'].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'C3H8'].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'C4H10'].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'CO'].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'H2'].fix(1e-6)

    ########## INITIALIZE FLOWSHEET ##########
    # initialize the reformer with random values.
    # this is only to get a good set of initial values such that 
    # IPOPT can then take over and solve this flowsheet for us.

    m.fs.reformer.inlet.flow_mol[0] = 2262.5
    m.fs.reformer.inlet.temperature[0] = 469.8  
    m.fs.reformer.inlet.pressure[0] = 203395.9
    m.fs.reformer.inlet.mole_frac_comp[0, 'CH4'] = 0.1912
    m.fs.reformer.inlet.mole_frac_comp[0, 'C2H6'] = 0.0066
    m.fs.reformer.inlet.mole_frac_comp[0, 'C3H8'] = 0.0014
    m.fs.reformer.inlet.mole_frac_comp[0, 'C4H10'] = 0.0008
    m.fs.reformer.inlet.mole_frac_comp[0, 'H2'] = 1e-5
    m.fs.reformer.inlet.mole_frac_comp[0, 'CO'] = 1e-5
    m.fs.reformer.inlet.mole_frac_comp[0, 'CO2'] = 0.0022
    m.fs.reformer.inlet.mole_frac_comp[0, 'H2O'] = 0.2116
    m.fs.reformer.inlet.mole_frac_comp[0, 'N2'] = 0.4582
    m.fs.reformer.inlet.mole_frac_comp[0, 'O2'] = 0.1224
    m.fs.reformer.inlet.mole_frac_comp[0, 'Ar'] = 0.0055

    m.fs.reformer.initialize() 
    m.fs.reformer_recuperator.initialize()
    m.fs.bypass_rejoin.initialize()
    m.fs.product.initialize()
    m.fs.reformer_mix.initialize()
    m.fs.feed.initialize()         
    m.fs.NG_expander.initialize()
    m.fs.air_compressor_s1.initialize()    
    m.fs.intercooler_s1.initialize()    
    m.fs.air_compressor_s2.initialize()    
    m.fs.intercooler_s2.initialize()    
    m.fs.reformer_bypass.initialize()    

    propagate_state(arc=m.fs.RECUP_COLD_IN)
    propagate_state(arc=m.fs.RECUP_COLD_OUT)
    propagate_state(arc=m.fs.NG_EXPAND_OUT)
    propagate_state(arc=m.fs.NG_TO_REF)
    propagate_state(arc=m.fs.STAGE_1_OUT)
    propagate_state(arc=m.fs.IC_1_OUT)
    propagate_state(arc=m.fs.STAGE_2_OUT)
    propagate_state(arc=m.fs.IC_2_OUT)
    propagate_state(arc=m.fs.REF_IN)
    propagate_state(arc=m.fs.REF_OUT)
    propagate_state(arc=m.fs.RECUP_HOT_OUT)
    propagate_state(arc=m.fs.REF_BYPASS)
    propagate_state(arc=m.fs.PRODUCT)

    ####### OBJECTIVE IS TO MAXIMIZE H2 COMPOSITION IN PRODUCT STREAM #######

    m.fs.obj = pyo.Objective(expr = m.fs.product.mole_frac_comp[0, 'H2'], sense = pyo.maximize)
    
    ####### CONSTRAINTS #######
    m.fs.reformer.conversion = Var(bounds=(0, 1), units=pyunits.dimensionless)  # fraction

    m.fs.reformer.conv_constraint = Constraint(
        expr=m.fs.reformer.conversion
        * m.fs.reformer.inlet.flow_mol[0]
        * m.fs.reformer.inlet.mole_frac_comp[0, "CH4"]
        == (
            m.fs.reformer.inlet.flow_mol[0] * m.fs.reformer.inlet.mole_frac_comp[0, "CH4"]
            - m.fs.reformer.outlet.flow_mol[0] * m.fs.reformer.outlet.mole_frac_comp[0, "CH4"]
        )
    )
    m.fs.reformer.conversion.fix(conversion) # ACHIEVE A CONVERSION OF 0.9 IN AUTOTHERMAL REFORMER

    # MINIMUM PRODUCT FLOW OF 3500 mol/s IN PRODUCT STREAM
    @m.Constraint()
    def min_product_flow_mol(m):
        return m.fs.product.flow_mol[0] >= 3500

    # MAXIMUM N2 COMPOSITION OF 0.3 IN PRODUCT STREAM
    @m.Constraint()
    def max_product_N2_comp(m):
        return m.fs.product.mole_frac_comp[0, 'N2'] <= 0.3

    # MAXIMUM REFORMER OUTLET TEMPERATURE OF 1200 K
    @m.Constraint()
    def max_reformer_outlet_temp(m):
        return m.fs.reformer.outlet.temperature[0] <= 1200

    # MAXIMUM PRODUCT OUTLET TEMPERATURE OF 650 K
    @m.Constraint()
    def max_product_temp(m):
        return m.fs.product.temperature[0] <= 650

    ########## REFORMER OUTLET PRESSURE ##########
    m.fs.reformer.outlet.pressure.fix(137895)

    # Unfix D.O.F. If you unfix these variables, inlet temperature, flow and composition
    # to the Gibbs reactor will have to be determined by the optimization problem. 
    m.fs.reformer_bypass.split_fraction[0, 'bypass_outlet'].unfix() 
    m.fs.reformer_mix.steam_inlet.flow_mol.unfix()

    return m
    
def make_implicit(m):
    ########### CREATE EXTERNAL PYOMO MODEL FOR THE AUTOTHERMAL REFORMER ###########
    igraph = IncidenceGraphInterface(m.fs.reformer, include_inequality=False)

    m.outlet_mole_frac_comp = Var(m.fs.thermo_params.component_list)
    for j in m.fs.thermo_params.component_list:
        m.outlet_mole_frac_comp[j] = m.fs.reformer.outlet.mole_frac_comp[0, j]

    m.outlet_temperature = Var(initialize=m.fs.reformer.outlet.temperature[0].value)
    m.heatDuty = Var(initialize=m.fs.reformer.heat_duty[0].value)
    m.outlet_flow_mol = Var(initialize=m.fs.reformer.outlet.flow_mol[0].value)

    @m.Constraint(m.fs.thermo_params.component_list)
    def outlet_mole_frac_comp_eq(m, j):
        return m.outlet_mole_frac_comp[j] == m.fs.reformer.outlet.mole_frac_comp[0, j]

    @m.Constraint()
    def outlet_temperature_eq(m):
        return m.outlet_temperature == m.fs.reformer.outlet.temperature[0]

    @m.Constraint()
    def heatDuty_eq(m):
        return m.heatDuty == m.fs.reformer.heat_duty[0]

    @m.Constraint()
    def outlet_flow_mol_eq(m):
        return m.outlet_flow_mol == m.fs.reformer.outlet.flow_mol[0]

    residual_eqns = [
        m.outlet_temperature_eq,
        m.heatDuty_eq,
        m.outlet_flow_mol_eq,
    ]
    residual_eqns.extend(m.outlet_mole_frac_comp_eq.values())

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
        m.outlet_temperature,
        m.heatDuty,
        m.outlet_flow_mol,
    ]

    input_vars.extend(m.outlet_mole_frac_comp.values())    

    external_eqns = list(igraph.constraints)
    
    external_vars = [var for var in igraph.variables if var is not m.fs.reformer.inlet.temperature[0]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "H2"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "CO"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "H2O"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "CO2"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "CH4"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "C2H6"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "C3H8"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "C4H10"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "N2"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "O2"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.mole_frac_comp[0, "Ar"]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.flow_mol[0]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.inlet.pressure[0]]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.lagrange_mult[0.0,'N']]
    external_vars = [var for var in external_vars if var is not m.fs.reformer.lagrange_mult[0.0,'Ar']]

    external_var_set = ComponentSet(external_vars)
    external_eqn_set = ComponentSet(external_eqns)
    residual_eqn_set = ComponentSet(residual_eqns)

    epm = ExternalPyomoModel(
        input_vars,
        external_vars,
        residual_eqns,
        external_eqns,
    )

    ########### CONNECT FLOWSHEET TO THE IMPLICIT AUTOTHERMAL REFORMER ###########

    m_implicit = ConcreteModel()
    m_implicit.egb = ExternalGreyBoxBlock()
    m_implicit.egb.set_external_model(epm, inputs=input_vars)

    # Link the outlet of reformer_mix to the inlet of the autothermal reformer
    @m_implicit.Constraint()
    def linking_T_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.temperature[0] == m_implicit.egb.inputs[0]

    @m_implicit.Constraint()
    def linking_H2_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "H2"] == m_implicit.egb.inputs[1]
    
    @m_implicit.Constraint()
    def linking_CO_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "CO"] == m_implicit.egb.inputs[2]

    @m_implicit.Constraint()
    def linking_H2O_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "H2O"] == m_implicit.egb.inputs[3]

    @m_implicit.Constraint()
    def linking_CO2_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "CO2"] == m_implicit.egb.inputs[4]

    @m_implicit.Constraint()
    def linking_CH4_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "CH4"] == m_implicit.egb.inputs[5]
    
    @m_implicit.Constraint()
    def linking_C2H6_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "C2H6"] == m_implicit.egb.inputs[6]
    
    @m_implicit.Constraint()
    def linking_C3H8_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "C3H8"] == m_implicit.egb.inputs[7]

    @m_implicit.Constraint()
    def linking_C4H10_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "C4H10"] == m_implicit.egb.inputs[8]

    @m_implicit.Constraint()
    def linking_N2_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "N2"] == m_implicit.egb.inputs[9]

    @m_implicit.Constraint()
    def linking_O2_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "O2"] == m_implicit.egb.inputs[10]

    @m_implicit.Constraint()
    def linking_Ar_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.mole_frac_comp[0, "Ar"] == m_implicit.egb.inputs[11]

    @m_implicit.Constraint()
    def linking_flow_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.flow_mol[0] == m_implicit.egb.inputs[12]

    @m_implicit.Constraint()
    def linking_P_to_egb(m_implicit):
        return m.fs.reformer_mix.outlet.pressure[0] == m_implicit.egb.inputs[13]

    full_igraph = IncidenceGraphInterface(m)
    fullspace_cons = [
        con for con in full_igraph.constraints
        if con not in residual_eqn_set and con not in external_eqn_set
    ]
    fullspace_vars = [
        var for var in full_igraph.variables
        if var not in external_var_set
    ]

    m_implicit.fullspace_cons = pyo.Reference(fullspace_cons)
    m_implicit.fullspace_vars = pyo.Reference(fullspace_vars)

    m_implicit.objective = pyo.Reference(m.fs.obj)

    solver = pyo.SolverFactory("cyipopt")
    solver.solve(m_implicit, tee=True)
    return m_implicit.objective.value

if __name__ == "__main__":
    m = build_atr_flowsheet(conversion = 0.9)
    m_implicit = make_implicit(m)