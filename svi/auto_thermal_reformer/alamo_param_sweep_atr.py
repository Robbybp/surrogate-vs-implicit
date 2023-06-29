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
import os
import numpy as np
import pandas as pd
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
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale

from idaes.models.unit_models import (
    Mixer,
    HeatExchanger,
    PressureChanger,
    Separator,
    Feed,
    Product)
from idaes.core.util.initialization import propagate_state
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
from idaes.core.solvers import get_solver
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from idaes.core.surrogate.alamopy import AlamoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock

def build_alamo_atr_flowsheet(m, alamo_surrogate_dict, conversion):
   ########## ADD THERMODYNAMIC PROPERTIES ##########  
    components = ['H2', 'CO', "H2O", 'CO2', 'CH4', "C2H6", "C3H8", "C4H10",'N2', 'O2', 'Ar']
    thermo_props_config_dict = get_prop(components = components)
    m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

    ########## ADD FEED AND PRODUCT STREAMS ##########  
    m.fs.feed = Feed(property_package = m.fs.thermo_params)
    m.fs.product = Product(property_package = m.fs.thermo_params)
    m.fs.steam_feed = Feed(property_package = m.fs.thermo_params)

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

    ########## DEFINE SURROGATE BLOCK FOR THE ATR ##########

    m.fs.reformer = SurrogateBlock() 
    m.fs.reformer.conversion = Var(bounds=(0, 1), units=pyunits.dimensionless) 
    m.fs.reformer.conversion.fix(conversion) # ACHIEVE A CONVERSION OF 0.95 IN ATR

    ########## CREATE OUTLET VARS FOR ATR SURROGATE ##########

    m.fs.reformer.heat_duty = Var(initialize = 43262357) # W
    m.fs.reformer.out_flow_mol = Var(initialize = 3217) # mol/s
    m.fs.reformer.out_temp = Var(initialize = 998.8) # K
    m.fs.reformer.out_H2 = Var(initialize = 0.415311)
    m.fs.reformer.out_CO = Var(initialize = 0.169659)
    m.fs.reformer.out_H2O = Var(initialize = 0.042004)
    m.fs.reformer.out_CO2 = Var(initialize = 0.024803)
    m.fs.reformer.out_CH4 = Var(initialize = 0.021087)
    m.fs.reformer.out_C2H6 = Var(initialize = 0.000000194)
    m.fs.reformer.out_C3H8 = Var(initialize = 0.00000000000705)
    m.fs.reformer.out_C4H10 = Var(initialize = 0.000000000000000241)
    m.fs.reformer.out_N2 = Var(initialize = 0.323243566)
    m.fs.reformer.out_O2 = Var(initialize = 1e-19)
    m.fs.reformer.out_Ar = Var(initialize = 0.003892586)

    # define the inputs to the surrogate models
    inputs = [m.fs.reformer_bypass.reformer_outlet.flow_mol[0], 
                m.fs.reformer_bypass.reformer_outlet.temperature[0], 
                m.fs.steam_feed.flow_mol[0],
                m.fs.reformer.conversion]

    # define the outputs of the surrogate models
    outputs = [m.fs.reformer.heat_duty, m.fs.reformer.out_flow_mol, m.fs.reformer.out_temp, m.fs.reformer.out_H2,
                m.fs.reformer.out_CO, m.fs.reformer.out_H2O, m.fs.reformer.out_CO2, m.fs.reformer.out_CH4, m.fs.reformer.out_C2H6,
                m.fs.reformer.out_C3H8, m.fs.reformer.out_C4H10, m.fs.reformer.out_N2, m.fs.reformer.out_O2, m.fs.reformer.out_Ar]

    # build the surrogate for the Gibbs Reactor using the JSON file obtained before
    surrogate = AlamoSurrogate.load_from_file(alamo_surrogate_dict)
    m.fs.reformer.build_model(surrogate, input_vars=inputs, output_vars=outputs)

    m.fs.bypass_rejoin = Mixer(
        inlet_list = ["syngas_inlet", "bypass_inlet"],
        property_package = m.fs.thermo_params)


    ########## CONNECT UNIT MODELS UPSTREAM OF SURROGATE REFORMER ##########  

    m.fs.RECUP_COLD_IN = Arc(source=m.fs.feed.outlet, destination=m.fs.reformer_recuperator.tube_inlet)
    m.fs.RECUP_COLD_OUT = Arc(source=m.fs.reformer_recuperator.tube_outlet, destination=m.fs.NG_expander.inlet)
    m.fs.NG_EXPAND_OUT = Arc(source=m.fs.NG_expander.outlet, destination=m.fs.reformer_bypass.inlet)

    ########## CONNECT OUTPUTS OF SURROGATE TO RECUPERATOR SHELL INLET ##########  

    m.fs.reformer_recuperator.shell_inlet.flow_mol[0].set_value(value(m.fs.reformer.out_flow_mol))
    m.fs.reformer_recuperator.shell_inlet.temperature[0].set_value(value(m.fs.reformer.out_temp))
    m.fs.reformer_recuperator.shell_inlet.pressure.fix(137895)
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'H2'].set_value(value(m.fs.reformer.out_H2))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'CO'].set_value(value(m.fs.reformer.out_CO))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'H2O'].set_value(value(m.fs.reformer.out_H2O))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'CO2'].set_value(value(m.fs.reformer.out_CO2))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'CH4'].set_value(value(m.fs.reformer.out_CH4))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'C2H6'].set_value(value(m.fs.reformer.out_C2H6))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'C3H8'].set_value(value(m.fs.reformer.out_C3H8))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'C4H10'].set_value(value(m.fs.reformer.out_C4H10))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'N2'].set_value(value(m.fs.reformer.out_N2))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'O2'].set_value(value(m.fs.reformer.out_O2))
    m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'Ar'].set_value(value(m.fs.reformer.out_Ar))

    ########## CONNECT UNIT MODELS DOWNSTREAM OF SURROGATE REFORMER ##########  

    m.fs.RECUP_HOT_OUT = Arc(source=m.fs.reformer_recuperator.shell_outlet, destination=m.fs.bypass_rejoin.syngas_inlet)
    m.fs.REF_BYPASS = Arc(source=m.fs.reformer_bypass.bypass_outlet, destination=m.fs.bypass_rejoin.bypass_inlet)
    m.fs.PRODUCT = Arc(source=m.fs.bypass_rejoin.outlet, destination=m.fs.product.inlet)

    ########## EXPAND ARCS ##########  

    pyo.TransformationFactory("network.expand_arcs").apply_to(m.fs)

def set_alamo_atr_flowsheet_inputs(m, T, P):
    # natural gas feed conditions

    m.fs.feed.outlet.flow_mol.fix(1161.9)  # mol/s
    m.fs.feed.outlet.temperature.fix(T)  # K
    m.fs.feed.outlet.pressure.fix(P) # Pa
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

    # recuperator conditions

    m.fs.reformer_recuperator.area.fix(4190) # m2
    m.fs.reformer_recuperator.overall_heat_transfer_coefficient.fix(80) # W/m2K # it was 80e-3 # potential bug

    # natural gas expander conditions

    m.fs.NG_expander.outlet.pressure.fix(203396) # Pa
    m.fs.NG_expander.efficiency_isentropic.fix(0.9)

    # steam conditions

    m.fs.steam_feed.flow_mol.fix(250) # mol/s, this value will be unfixed. It's just to initialize.
    m.fs.steam_feed.temperature.fix(422) # K
    m.fs.steam_feed.pressure.fix(203396)  # Pa
    m.fs.steam_feed.mole_frac_comp[0, 'H2O'].fix(0.9999)
    m.fs.steam_feed.mole_frac_comp[0, 'CO2'].fix(1e-6)
    m.fs.steam_feed.mole_frac_comp[0, 'N2'].fix(1e-6)
    m.fs.steam_feed.mole_frac_comp[0, 'O2'].fix(1e-6)
    m.fs.steam_feed.mole_frac_comp[0, 'Ar'].fix(1e-6)
    m.fs.steam_feed.mole_frac_comp[0, 'CH4'].fix(1e-6)
    m.fs.steam_feed.mole_frac_comp[0, 'C2H6'].fix(1e-6)
    m.fs.steam_feed.mole_frac_comp[0, 'C3H8'].fix(1e-6)
    m.fs.steam_feed.mole_frac_comp[0, 'C4H10'].fix(1e-6)
    m.fs.steam_feed.mole_frac_comp[0, 'CO'].fix(1e-6)
    m.fs.steam_feed.mole_frac_comp[0, 'H2'].fix(1e-6)

    # initial bypass fraction for reformer bypass. This value will be unfixed. It's just to initialize.

    m.fs.reformer_bypass.split_fraction[0, 'bypass_outlet'].fix(0.23)

def initialize_alamo_atr_flowsheet(m):
    ########## INITIALIZE AND PROPAGATE STATES ##########

    m.fs.feed.initialize()    
    propagate_state(arc=m.fs.RECUP_COLD_IN)

    m.fs.reformer_recuperator.initialize()
    propagate_state(arc=m.fs.RECUP_COLD_OUT)

    m.fs.NG_expander.initialize()
    propagate_state(arc=m.fs.NG_EXPAND_OUT)

    m.fs.reformer_bypass.initialize() 
    propagate_state(arc=m.fs.RECUP_HOT_OUT)
    propagate_state(arc=m.fs.REF_BYPASS)

    m.fs.bypass_rejoin.initialize()

    m.fs.product.initialize()
    propagate_state(arc=m.fs.PRODUCT)

df = {'T':[], 'P':[], 'Termination':[], 'Time':[], 'Objective':[], 'Steam':[], 'Bypass Frac': [],
      'TinCH4':[], 'FinCH4':[]}

if __name__ == "__main__":
    """
    The optimization problem to solve is the following:
    Maximize H2 composition in the product stream such that its minimum flow is 3500 mol/s, 
    its maximum N2 concentration is 0.3, the maximum reformer outlet temperature is 1200 K and 
    the maximum product temperature is 650 K.  
    """
    for T in np.linspace(288.15,338.15,8):   
        for P in np.linspace(1447379,1947379,8): 
            m = pyo.ConcreteModel(name='ALAMO_ATR_Flowsheet')
            m.fs = FlowsheetBlock(dynamic = False)
            dirname = os.path.dirname(__file__)
            basename = "alamo_surrogate_atr.json"
            fname = os.path.join(dirname,basename)
            build_alamo_atr_flowsheet(m, alamo_surrogate_dict = fname, conversion=0.94)
            set_alamo_atr_flowsheet_inputs(m, T, P)
            initialize_alamo_atr_flowsheet(m)

            ####### OBJECTIVE IS TO MAXIMIZE H2 COMPOSITION IN PRODUCT STREAM #######
            m.fs.obj = pyo.Objective(expr = m.fs.product.mole_frac_comp[0, 'H2'], sense = pyo.maximize)

            ####### CONSTRAINTS #######

            # Link outputs of ALAMO to inputs of reformer_recuperator 
            @m.Constraint()
            def link_T(m):
                return m.fs.reformer_recuperator.shell_inlet.flow_mol[0] == m.fs.reformer.out_flow_mol

            @m.Constraint()
            def link_F(m):
                return m.fs.reformer_recuperator.shell_inlet.temperature[0] == m.fs.reformer.out_temp

            @m.Constraint()
            def link_H2(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'H2'] == m.fs.reformer.out_H2

            @m.Constraint()
            def link_CO(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'CO'] == m.fs.reformer.out_CO

            @m.Constraint()
            def link_H2O(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'H2O'] == m.fs.reformer.out_H2O

            @m.Constraint()
            def link_CO2(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'CO2'] == m.fs.reformer.out_CO2

            @m.Constraint()
            def link_CH4(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'CH4'] == m.fs.reformer.out_CH4

            @m.Constraint()
            def link_C2H6(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'C2H6'] == m.fs.reformer.out_C2H6

            @m.Constraint()
            def link_C3H8(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'C3H8'] == m.fs.reformer.out_C3H8

            @m.Constraint()
            def link_C4H10(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'C4H10'] == m.fs.reformer.out_C4H10

            @m.Constraint()
            def link_N2(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'N2'] == m.fs.reformer.out_N2

            @m.Constraint()
            def link_O2(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'O2'] == m.fs.reformer.out_O2

            @m.Constraint()
            def link_Ar(m):
                return m.fs.reformer_recuperator.shell_inlet.mole_frac_comp[0, 'Ar'] == m.fs.reformer.out_Ar

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
                return m.fs.reformer.out_temp <= 1200

            # MAXIMUM PRODUCT OUTLET TEMPERATURE OF 650 K
            @m.Constraint()
            def max_product_temp(m):
                return m.fs.product.temperature[0] <= 650

            # Unfix D.O.F. If you unfix these variables, inlet temperature, flow and composition
            # to the ATR will have to be determined by the optimization problem. 
            m.fs.reformer_bypass.split_fraction[0, 'bypass_outlet'].unfix() 
            m.fs.steam_feed.flow_mol.unfix() 

            solver = get_solver()
            solver.options = {
                "tol": 1e-8,
                "max_iter": 1000
            }
            timer = TicTocTimer()
            timer.tic('starting timer')
            results = solver.solve(m, tee=True)
            dT = timer.toc('end')
            df[list(df.keys())[0]].append(T)
            df[list(df.keys())[1]].append(P)
            df[list(df.keys())[2]].append(results.solver.termination_condition)
            df[list(df.keys())[3]].append(dT)
            df[list(df.keys())[4]].append(value(m.fs.product.mole_frac_comp[0, 'H2']))
            df[list(df.keys())[5]].append(value(m.fs.steam_feed.flow_mol[0]))
            df[list(df.keys())[6]].append(value(m.fs.reformer_bypass.split_fraction[0, 'bypass_outlet']))
            df[list(df.keys())[7]].append(value(m.fs.reformer_bypass.reformer_outlet.temperature[0]))
            df[list(df.keys())[8]].append(value(m.fs.reformer_bypass.reformer_outlet.flow_mol[0]))


df = pd.DataFrame(df)
df.to_csv('param_sweep_alamo_atr.csv')