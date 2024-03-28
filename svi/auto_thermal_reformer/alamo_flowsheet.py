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
    Heater,
    Feed,
    Product)
from idaes.core.util.initialization import propagate_state
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
from idaes.core.solvers import get_solver
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from idaes.core.surrogate.alamopy import AlamoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock

import svi.auto_thermal_reformer.fullspace_flowsheet as fullspace
import svi.auto_thermal_reformer.config as config


DEFAULT_SURROGATE_FNAME = "alamo_surrogate_atr.json"


def _get_alamo_surrogate_fname():
    # TODO: Accept arguments so we can override the default results dir.
    # Note that this function is essentially hard-coding the default
    # surrogate file.
    default_results_dir = config.get_results_dir()
    return os.path.join(default_results_dir, DEFAULT_SURROGATE_FNAME)


def create_instance(
    conversion,
    pressure,
    initialize=True,
    surrogate_fname=None,
):
    if surrogate_fname is None:
        surrogate_fname = _get_alamo_surrogate_fname()

    # Create a simulation model so we can explicitly ensure we have zero degrees
    # of freedom after replacing the reformer with the ALAMO surrogate.
    # Note that this fixes conversion to 0.95. We will have to set this later.
    m = fullspace.make_simulation_model(pressure, initialize=True)

    # Deactivate constraints in reformer and pre-reformer-mixer
    # Note that we will re-use variables in these blocks as part of the
    # surrogate model.
    for con in m.fs.reformer.component_objects(pyo.Constraint):
        con.deactivate()
    for con in m.fs.reformer_mix.component_objects(pyo.Constraint):
        con.deactivate()
    # Deactivate the arc between mixer and reformer
    m.fs.REF_IN_expanded.deactivate()

    # To account for the deactivated reformer_mix, we create a new steam feed
    #m.fs.steam_feed = Feed(property_package = m.fs.thermo_params)

    ########## DEFINE SURROGATE BLOCK FOR THE ATR ##########
    m.fs.reformer_surrogate = SurrogateBlock()

    # Fix conversion to specified value
    m.fs.reformer.conversion.fix(conversion)
    # Create a reference to conversion on the surrogate block
    m.fs.reformer_surrogate.conversion = pyo.Reference(m.fs.reformer.conversion)

    ########## CREATE OUTLET VARS FOR ATR SURROGATE ##########
    m.fs.reformer_surrogate.heat_duty = pyo.Reference(m.fs.reformer.heat_duty)
    m.fs.reformer_surrogate.out_flow_mol = pyo.Reference(m.fs.reformer.outlet.flow_mol)
    m.fs.reformer_surrogate.out_temp = pyo.Reference(m.fs.reformer.outlet.temperature)
    m.fs.reformer_surrogate.out_mole_frac_comp = pyo.Reference(m.fs.reformer.outlet.mole_frac_comp)

    # define the inputs to the surrogate models
    inputs = [
        m.fs.reformer_bypass.reformer_outlet.flow_mol[0], 
        m.fs.reformer_bypass.reformer_outlet.temperature[0], 
        m.fs.reformer_mix.steam_inlet.flow_mol,
        # This is now fixed?
        # m.fs.intercooler_s2.outlet.flow_mol[0],
        m.fs.reformer_surrogate.conversion,
    ]

    # define the outputs of the surrogate models
    outputs = [
        m.fs.reformer_surrogate.heat_duty[0],
        m.fs.reformer_surrogate.out_flow_mol[0],
        m.fs.reformer_surrogate.out_temp[0],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "H2"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "CO"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "H2O"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "CO2"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "CH4"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "C2H6"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "C3H8"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "C4H10"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "N2"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "O2"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "Ar"],
    ]

    # build the surrogate for the Gibbs Reactor using the JSON file obtained before
    surrogate = AlamoSurrogate.load_from_file(surrogate_fname)

    # The reformer surrogate contains 14 eq. constraints and 15 vars (one fixed)
    m.fs.reformer_surrogate.build_model(
        surrogate, input_vars=inputs, output_vars=outputs
    )

    # TODO: Toggle whether we apply enforce the surrogate training bounds
    m.fs.reformer_bypass.reformer_outlet_state[0.0].flow_mol.setlb(0.0)
    m.fs.reformer_bypass.reformer_outlet_state[0.0].flow_mol.setub(50000.0)
    m.fs.reformer_surrogate.conversion.setlb(0.0)
    m.fs.reformer_surrogate.conversion.setub(1.0)

    # At this point we have a square (simulation) model. Now we must
    # unfix degrees of freedom, add an objective function, and add specification
    # constraints.
    fullspace.add_obj_and_constraints(m)

    return m


def build_alamo_atr_flowsheet(m, alamo_surrogate_dict, conversion):
    # TODO: This flowsheet should re-use the fullspace flowsheet.
    # - construct full-space flowsheet (including Gibbs reactor and mixer)
    # - deactivate reactor, mixer, and arcs linking them to rest of flowsheet
    # - Construct block based on surrogate model

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

    m.fs.air_compressor_s1 = PressureChanger(
        compressor=True,
        property_package=m.fs.thermo_params,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    m.fs.intercooler_s1 = Heater(
        property_package=m.fs.thermo_params, has_pressure_change=True
    )

    m.fs.air_compressor_s2 = PressureChanger(
        compressor=True,
        property_package=m.fs.thermo_params,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    m.fs.intercooler_s2 = Heater(
        property_package=m.fs.thermo_params, has_pressure_change=True
    )

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
    inputs = [
        m.fs.reformer_bypass.reformer_outlet.flow_mol[0], 
        m.fs.reformer_bypass.reformer_outlet.temperature[0], 
        m.fs.steam_feed.flow_mol[0],
        # m.fs.intercooler_s2.outlet.flow_mol[0],
        m.fs.reformer.conversion,
    ]

    # define the outputs of the surrogate models
    outputs = [m.fs.reformer.heat_duty, m.fs.reformer.out_flow_mol, m.fs.reformer.out_temp, m.fs.reformer.out_H2,
                m.fs.reformer.out_CO, m.fs.reformer.out_H2O, m.fs.reformer.out_CO2, m.fs.reformer.out_CH4, m.fs.reformer.out_C2H6,
                m.fs.reformer.out_C3H8, m.fs.reformer.out_C4H10, m.fs.reformer.out_N2, m.fs.reformer.out_O2, m.fs.reformer.out_Ar]

    # build the surrogate for the Gibbs Reactor using the JSON file obtained before
    surrogate = AlamoSurrogate.load_from_file(alamo_surrogate_dict)
    m.fs.reformer.build_model(surrogate, input_vars=inputs, output_vars=outputs)

    # TODO: Toggle whether we apply enforce the surrogate training bounds
    m.fs.reformer_bypass.reformer_outlet_state[0.0].flow_mol.setlb(0.0)
    m.fs.reformer_bypass.reformer_outlet_state[0.0].flow_mol.setub(50000.0)
    m.fs.reformer.conversion.setlb(0.0)
    m.fs.reformer.conversion.setub(1.0)

    m.fs.bypass_rejoin = Mixer(
        inlet_list = ["syngas_inlet", "bypass_inlet"],
        property_package = m.fs.thermo_params)


    ########## CONNECT UNIT MODELS UPSTREAM OF SURROGATE REFORMER ##########  

    m.fs.RECUP_COLD_IN = Arc(source=m.fs.feed.outlet, destination=m.fs.reformer_recuperator.tube_inlet)
    m.fs.RECUP_COLD_OUT = Arc(source=m.fs.reformer_recuperator.tube_outlet, destination=m.fs.NG_expander.inlet)
    m.fs.NG_EXPAND_OUT = Arc(source=m.fs.NG_expander.outlet, destination=m.fs.reformer_bypass.inlet)
    m.fs.STAGE_1_OUT = Arc(source=m.fs.air_compressor_s1.outlet, destination=m.fs.intercooler_s1.inlet)
    m.fs.IC_1_OUT = Arc(source=m.fs.intercooler_s1.outlet, destination=m.fs.air_compressor_s2.inlet)
    m.fs.STAGE_2_OUT = Arc(source=m.fs.air_compressor_s2.outlet, destination=m.fs.intercooler_s2.inlet)

    ########## CONNECT OUTPUTS OF SURROGATE TO RECUPERATOR SHELL INLET ##########  

    m.fs.reformer_recuperator.shell_inlet.pressure[0].fix(137895)
    m.fs.reformer_recuperator.shell_inlet.flow_mol[0].set_value(value(m.fs.reformer.out_flow_mol))
    m.fs.reformer_recuperator.shell_inlet.temperature[0].set_value(value(m.fs.reformer.out_temp))
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

def set_alamo_atr_flowsheet_inputs(m,P):
    # natural gas feed conditions

    m.fs.feed.outlet.flow_mol.fix(1161.9)  # mol/s
    m.fs.feed.outlet.temperature.fix(288.15)  # K
    m.fs.feed.outlet.pressure.fix(P*pyo.units.Pa) # Pa
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

    # air conditions

    m.fs.air_compressor_s1.inlet.flow_mol.fix(1332.9)  # mol/s
    m.fs.air_compressor_s1.inlet.temperature.fix(288.15)  # K
    m.fs.air_compressor_s1.inlet.pressure.fix(101353)  # Pa
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "CO2"].fix(0.0003)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "H2O"].fix(0.0104)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "N2"].fix(0.7722)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "O2"].fix(0.2077)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "Ar"].fix(0.00939)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "CH4"].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "C2H6"].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "C3H8"].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "C4H10"].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "CO"].fix(1e-6)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, "H2"].fix(1e-6)

    # air compressors and intercoolers

    m.fs.air_compressor_s1.outlet.pressure.fix(144790)  # Pa
    m.fs.air_compressor_s1.efficiency_isentropic.fix(0.84)

    m.fs.intercooler_s1.outlet.temperature.fix(310.93)  # K
    m.fs.intercooler_s1.outlet.pressure.fix(141343)  # Pa equivalent to a dP of -0.5 psi

    m.fs.air_compressor_s2.outlet.pressure.fix(206843)  # Pa
    m.fs.air_compressor_s2.efficiency_isentropic.fix(0.84)

    m.fs.intercooler_s2.outlet.temperature.fix(310.93)  # K
    m.fs.intercooler_s2.outlet.pressure.fix(203396)  # Pa equivalent to a dP of -0.5 psi

    # steam conditions

    m.fs.steam_feed.flow_mol.fix(464.77) # mol/s, this value will be unfixed. It's just to initialize.
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


def initialize_alamo_atr_flowsheet(m):
    ########## INITIALIZE AND PROPAGATE STATES ##########
    m.fs.reformer_recuperator.initialize()
    m.fs.bypass_rejoin.initialize()
    m.fs.product.initialize()
    m.fs.feed.initialize()
    m.fs.NG_expander.initialize()
    m.fs.air_compressor_s1.initialize()
    m.fs.intercooler_s1.initialize()
    m.fs.air_compressor_s2.initialize()
    m.fs.intercooler_s2.initialize()
    m.fs.reformer_bypass.inlet.flow_mol.fix(1161.9)
    m.fs.reformer_bypass.inlet.temperature.fix(700)  # K
    m.fs.reformer_bypass.initialize()


def make_simulation_model(X, P, surrogate_fname=None):
    if surrogate_fname is None:
        surrogate_fname = _get_alamo_surrogate_fname()
    m = pyo.ConcreteModel(name="ATR_Flowsheet")
    m.fs = FlowsheetBlock(dynamic=False)
    build_alamo_atr_flowsheet(m, alamo_surrogate_fname, conversion = X)
    set_alamo_atr_flowsheet_inputs(m, P)
    initialize_alamo_atr_flowsheet(m)
    m.fs.reformer_bypass.inlet.temperature.unfix()
    m.fs.reformer_bypass.inlet.flow_mol.unfix()
    return m


if __name__ == "__main__":
    """
    The optimization problem to solve is the following:
    Maximize H2 composition in the product stream such that its minimum flow is 3500 mol/s, 
    its maximum N2 concentration is 0.3, the maximum reformer outlet temperature is 1200 K and 
    the maximum product temperature is 650 K.  
    """
    X = 0.95
    P = 1650000
    argparser = config.get_argparser()
    args = argparser.parse_args()
    surrogate_fname = os.path.join(args.data_dir, DEFAULT_SURROGATE_FNAME)
    m = create_instance(X, P, surrogate_fname=surrogate_fname)
    # Does this need to be applied after creating the surrogate? Why?
    initialize_alamo_atr_flowsheet(m)
    m.fs.reformer_bypass.inlet.temperature.unfix()
    m.fs.reformer_bypass.inlet.flow_mol.unfix()

    solver = config.get_optimization_solver()
    solver.config.options["honor_original_bounds"] = "yes"
    timer = TicTocTimer()
    timer.tic('starting timer')
    results = solver.solve(m, tee=True)
    dT = timer.toc('end')
