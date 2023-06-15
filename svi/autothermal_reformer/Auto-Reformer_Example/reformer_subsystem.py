##############################################################################
# The development of this flowsheet/code is funded by the ARPA-E DIFFERENTIATE
# project: “Machine Learning for Natural Gas to Electric Power System Design”
# Project number: DE-FOA-0002107-1625.
# This project is a collaborative effort between the Pacific Northwest National
# Laboratory, the National Energy Technology Laboratory, and the University of
# Washington to design NGFC systems with high efficiencies and low CO2
# emissions.
##############################################################################
##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Flowsheet for testing superstructure optimization of an NGFC reformer section.

System bounds: reformer section of NGFC flowsheet without CCS

ROM: none

Property package: natural gas PR with scaled units

Dummy objective function: temperature to power island

Disjunctions:
    1. external reformer (full reformer, heater & turbine, only heater)

Continuous variables: none
"""

import os
import sys

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.network import Arc
from pyomo.network.plugins import expand_arcs
from pyomo.common.fileutils import this_file_dir
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.gdp import Disjunct, Disjunction

# IDAES Imports
from idaes.core import FlowsheetBlock  # Flowsheet class
from idaes.core.util import copy_port_values
from idaes.core.util import model_serializer as ms
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.tables import create_stream_table_dataframe
import idaes.core.util.scaling as iscale
from idaes.core.util.misc import svg_tag

from idaes.generic_models.properties.core.generic.generic_property import (
    GenericParameterBlock, GenericStateBlockData)
from idaes.generic_models.properties.core.generic.generic_reaction import (
    GenericReactionParameterBlock)
from idaes.generic_models.unit_models import (
    Mixer,
    Heater,
    HeatExchanger,
    PressureChanger,
    GibbsReactor,
    StoichiometricReactor,
    Separator,
    Translator,
    Feed,
    Product)
from idaes.generic_models.unit_models.heat_exchanger import \
    delta_temperature_underwood_callback
from idaes.generic_models.unit_models.pressure_changer import \
    ThermodynamicAssumption
from idaes.generic_models.unit_models.separator import SplittingType
from idaes.generic_models.unit_models.mixer import MomentumMixingType
from natural_gas_PR_scaled_units import get_NG_properties

__author__ = "ARPA-E DIFFERENTIATE IDAES Team (A. Noring and M. Zamarripa)"
__version__ = "1.0.0"
# %% Reformer Section


def build_reformer(m):
    # build property package
    natural_gas_config = get_NG_properties(
        components=['H2', 'CO', "H2O", 'CO2', 'CH4', "C2H6", "C3H8", "C4H10",
                    'N2', 'O2', 'Ar'])
    # ['CH4', 'CO', 'CO2', 'H2', 'H2O', 'N2', 'O2', 'Ar'])
    m.fs.natural_gas_props = GenericParameterBlock(default=natural_gas_config)

    # feed and product streams
    m.fs.feed = Feed(default={'property_package': m.fs.natural_gas_props})
    m.fs.product = Product(default={'property_package':
                                    m.fs.natural_gas_props})

    # unit models
    m.fs.reformer_recuperator = HeatExchanger(
        default={"delta_temperature_callback":
                 delta_temperature_underwood_callback,
                 "shell": {"property_package": m.fs.natural_gas_props},
                 "tube": {"property_package": m.fs.natural_gas_props}})

    m.fs.NG_expander = PressureChanger(
        default={'compressor': False,
                 'property_package': m.fs.natural_gas_props,
                 'thermodynamic_assumption':
                     ThermodynamicAssumption.isentropic})

    m.fs.reformer_bypass = Separator(
        default={"outlet_list": ["reformer_outlet", "bypass_outlet"],
                 "property_package": m.fs.natural_gas_props})

    m.fs.air_compressor_s1 = PressureChanger(
        default={"compressor": True,
                 "property_package": m.fs.natural_gas_props,
                 "thermodynamic_assumption":
                     ThermodynamicAssumption.isentropic})

    m.fs.intercooler_s1 = Heater(
        default={"property_package": m.fs.natural_gas_props,
                 "has_pressure_change": True})

    m.fs.air_compressor_s2 = PressureChanger(
        default={"compressor": True,
                 "property_package": m.fs.natural_gas_props,
                 "thermodynamic_assumption":
                     ThermodynamicAssumption.isentropic})

    m.fs.intercooler_s2 = Heater(
        default={"property_package": m.fs.natural_gas_props,
                 "has_pressure_change": True})

    m.fs.reformer_mix = Mixer(
        default={"inlet_list": ["gas_inlet", "oxygen_inlet", "steam_inlet"],
                 "property_package": m.fs.natural_gas_props})

    m.fs.reformer = GibbsReactor(
        default={"has_heat_transfer": True,
                 "has_pressure_change": True,
                 "inert_species": ["N2", "Ar"],
                 "property_package": m.fs.natural_gas_props})

    m.fs.bypass_rejoin = Mixer(
        default={"inlet_list": ["syngas_inlet", "bypass_inlet"],
                 "property_package": m.fs.natural_gas_props})

    # arcs
    m.fs.RECUP_COLD_IN = Arc(
        source=m.fs.feed.outlet,
        destination=m.fs.reformer_recuperator.tube_inlet)

    m.fs.RECUP_COLD_OUT = Arc(
        source=m.fs.reformer_recuperator.tube_outlet,
        destination=m.fs.NG_expander.inlet)

    m.fs.NG_EXPAND_OUT = Arc(
        source=m.fs.NG_expander.outlet,
        destination=m.fs.reformer_bypass.inlet)

    m.fs.NG_TO_REF = Arc(
        source=m.fs.reformer_bypass.reformer_outlet,
        destination=m.fs.reformer_mix.gas_inlet)

    m.fs.STAGE_1_OUT = Arc(
        source=m.fs.air_compressor_s1.outlet,
        destination=m.fs.intercooler_s1.inlet)

    m.fs.IC_1_OUT = Arc(
        source=m.fs.intercooler_s1.outlet,
        destination=m.fs.air_compressor_s2.inlet)

    m.fs.STAGE_2_OUT = Arc(
        source=m.fs.air_compressor_s2.outlet,
        destination=m.fs.intercooler_s2.inlet)

    m.fs.IC_2_OUT = Arc(
        source=m.fs.intercooler_s2.outlet,
        destination=m.fs.reformer_mix.oxygen_inlet)

    m.fs.REF_IN = Arc(
        source=m.fs.reformer_mix.outlet,
        destination=m.fs.reformer.inlet)

    m.fs.REF_OUT = Arc(
        source=m.fs.reformer.outlet,
        destination=m.fs.reformer_recuperator.shell_inlet)

    m.fs.RECUP_HOT_OUT = Arc(
        source=m.fs.reformer_recuperator.shell_outlet,
        destination=m.fs.bypass_rejoin.syngas_inlet)

    m.fs.REF_BYPASS = Arc(
        source=m.fs.reformer_bypass.bypass_outlet,
        destination=m.fs.bypass_rejoin.bypass_inlet)

    m.fs.PRODUCT = Arc(
        source=m.fs.bypass_rejoin.outlet,
        destination=m.fs.product.inlet)

    # expand arcs
    pyo.TransformationFactory("network.expand_arcs").apply_to(m.fs)

    print("Degrees of Freedom = %d" % degrees_of_freedom(m))


def set_reformer_inputs(m):
    # natural gas feed conditions
    m.fs.feed.outlet.flow_mol.fix(1161.9e-3)  # kmol/s
    m.fs.feed.outlet.temperature.fix(288.15)  # K
    m.fs.feed.outlet.pressure.fix(3447379e-3)  # kPa, equal to 500 psia
    m.fs.feed.outlet.mole_frac_comp[0, 'CH4'].fix(0.974) # 0.931 when C2+ included
    m.fs.feed.outlet.mole_frac_comp[0, 'C2H6'].fix(0.032)
    m.fs.feed.outlet.mole_frac_comp[0, 'C3H8'].fix(0.007)
    m.fs.feed.outlet.mole_frac_comp[0, 'C4H10'].fix(0.004)
    m.fs.feed.outlet.mole_frac_comp[0, 'CO'].fix(0)
    m.fs.feed.outlet.mole_frac_comp[0, 'CO2'].fix(0.01)
    m.fs.feed.outlet.mole_frac_comp[0, 'H2'].fix(0)
    m.fs.feed.outlet.mole_frac_comp[0, 'H2O'].fix(0)
    m.fs.feed.outlet.mole_frac_comp[0, 'N2'].fix(0.016)
    m.fs.feed.outlet.mole_frac_comp[0, 'O2'].fix(0)
    m.fs.feed.outlet.mole_frac_comp[0, 'Ar'].fix(0)

    # recuperator conditions
    # m.fs.reformer_recuperator.tube_outlet.temperature.fix(1010.93)
    # I fixed area  to improve initialization
    m.fs.reformer_recuperator.area.fix(4190)
    m.fs.reformer_recuperator.overall_heat_transfer_coefficient.fix(80e-3)

    # natural gas expander conditions
    m.fs.NG_expander.outlet.pressure.fix(206843e-3)  # equal to 30 psia
    m.fs.NG_expander.efficiency_isentropic.fix(0.9)

    # internal reformation percentage
    m.fs.reformer_bypass.split_fraction[0, 'bypass_outlet'].fix(0.6)

    # air to reformer
    m.fs.air_compressor_s1.inlet.flow_mol.fix(1332.9e-3)  # kmol/s
    m.fs.air_compressor_s1.inlet.temperature.fix(288.15)  # K
    m.fs.air_compressor_s1.inlet.pressure.fix(101353e-3)  # kPa, = 14.7 psia
    m.fs.air_compressor_s1.inlet.mole_frac_comp.fix(0)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'CO2'].fix(0.0003)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'H2O'].fix(0.0104)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'N2'].fix(0.7722)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'O2'].fix(0.2077)
    m.fs.air_compressor_s1.inlet.mole_frac_comp[0, 'Ar'].fix(0.0094)

    # air compressors and intercoolers
    m.fs.air_compressor_s1.outlet.pressure.fix(144790e-3)  # kPa, = 21 psia
    m.fs.air_compressor_s1.efficiency_isentropic.fix(0.84)

    m.fs.intercooler_s1.outlet.temperature.fix(310.93)  # K, equal to 100 F
    m.fs.intercooler_s1.outlet.pressure.fix(141343e-3)
    # m.fs.intercooler_s1.deltaP.fix(-3447)  # kPa, equal to -0.5 psi

    m.fs.air_compressor_s2.outlet.pressure.fix(206843e-3)  # kPa, = 30 psia
    m.fs.air_compressor_s2.efficiency_isentropic.fix(0.84)

    m.fs.intercooler_s2.outlet.temperature.fix(310.93)  # K, equal to 100 F
    m.fs.intercooler_s2.outlet.pressure.fix(203396e-3)
    # m.fs.intercooler_s2.deltaP.fix(-3447)  # Pa, equal to -0.5 psi

    # steam to reformer
    m.fs.reformer_mix.steam_inlet.flow_mol.fix(464.77e-3)  # kmol/s
    m.fs.reformer_mix.steam_inlet.temperature.fix(422)  # K
    m.fs.reformer_mix.steam_inlet.pressure.fix(206843e-3)  # kPa, = 30 psia
    m.fs.reformer_mix.steam_inlet.mole_frac_comp.fix(0)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, 'H2O'].fix(1)

    # reformer outlet pressure
    m.fs.reformer.outlet.pressure.fix(137895e-3)  # kPa, equal to 20 Psi
    m.fs.reformer.outlet.temperature.fix(1060.93)  # K, equal to 1450 F

    print("Degrees of Freedom = %d" % degrees_of_freedom(m))


def apply_reformer_scaling(m):
    # property package scaling factors
    m.fs.natural_gas_props.set_default_scaling("flow_mol", 1)
    m.fs.natural_gas_props.set_default_scaling("flow_mol_phase", 1)
    m.fs.natural_gas_props.set_default_scaling("temperature", 1e-2)
    m.fs.natural_gas_props.set_default_scaling("pressure", 1e-2)
    m.fs.natural_gas_props.set_default_scaling("mole_frac_comp", 10)
    # m.fs.natural_gas_props.set_default_scaling("mole_frac_phase_comp", 10)
    m.fs.natural_gas_props.set_default_scaling("enth_mol", 1e-3)
    m.fs.natural_gas_props.set_default_scaling("enth_mol_phase", 1e-3)
    m.fs.natural_gas_props.set_default_scaling("entr_mol", 1e-1)
    m.fs.natural_gas_props.set_default_scaling("entr_mol_phase", 1e-1)

    iscale.set_scaling_factor(
        m.fs.reformer.control_volume.elemental_flow_in, 1)
    iscale.set_scaling_factor(
        m.fs.reformer.lagrange_mult, 1e-3)

    iscale.calculate_scaling_factors(m.fs)
    iscale.calculate_scaling_factors(m.fs.exp_off)
    iscale.calculate_scaling_factors(m.fs.ref_off)
    iscale.calculate_scaling_factors(m.fs.ref_on)


def initialize_reformer(m):
    # full reformer #

    # use reformer inlet as tear stream
    # m.fs.reformer.inlet.flow_mol.fix(2262e-3)  # kmol/s
    # m.fs.reformer.inlet.temperature.fix(462)  # K
    # m.fs.reformer.inlet.pressure.fix(203396e-3)  # kPa
    # m.fs.reformer.inlet.mole_frac_comp[0, 'CH4'].fix(0.200)
    # # m.fs.reformer.inlet.mole_frac_comp[0, 'C2H6'].fix(0.006)
    # # m.fs.reformer.inlet.mole_frac_comp[0, 'C3H8'].fix(0.002)
    # # m.fs.reformer.inlet.mole_frac_comp[0, 'C4H10'].fix(0.001)
    # m.fs.reformer.inlet.mole_frac_comp[0, 'H2'].fix(0)
    # m.fs.reformer.inlet.mole_frac_comp[0, 'CO'].fix(0)
    # m.fs.reformer.inlet.mole_frac_comp[0, 'CO2'].fix(0.002)
    # m.fs.reformer.inlet.mole_frac_comp[0, 'H2O'].fix(0.212)
    # m.fs.reformer.inlet.mole_frac_comp[0, 'N2'].fix(0.458)
    # m.fs.reformer.inlet.mole_frac_comp[0, 'O2'].fix(0.122)
    # m.fs.reformer.inlet.mole_frac_comp[0, 'Ar'].fix(0.006)
    m.fs.reformer.inlet.flow_mol[0] = 2262.5e-3  # kmol/s
    m.fs.reformer.inlet.temperature[0] = 469.8  # K
    m.fs.reformer.inlet.pressure[0] = 203395.9e-3  # kPa
    m.fs.reformer.inlet.mole_frac_comp[0, 'CH4'] = 0.1912
    m.fs.reformer.inlet.mole_frac_comp[0, 'C2H6'] = 0.0066
    m.fs.reformer.inlet.mole_frac_comp[0, 'C3H8'] = 0.0014
    m.fs.reformer.inlet.mole_frac_comp[0, 'C4H10'] = 0.0008
    m.fs.reformer.inlet.mole_frac_comp[0, 'H2'] = 0
    m.fs.reformer.inlet.mole_frac_comp[0, 'CO'] = 0
    m.fs.reformer.inlet.mole_frac_comp[0, 'CO2'] = 0.0022
    m.fs.reformer.inlet.mole_frac_comp[0, 'H2O'] = 0.2116
    m.fs.reformer.inlet.mole_frac_comp[0, 'N2'] = 0.4582
    m.fs.reformer.inlet.mole_frac_comp[0, 'O2'] = 0.1224
    m.fs.reformer.inlet.mole_frac_comp[0, 'Ar'] = 0.0055
    m.fs.reformer.inlet.flow_mol[0].fix()  # mol/s
    m.fs.reformer.inlet.temperature[0].fix()  # K
    m.fs.reformer.inlet.pressure[0].fix()  # Pa
    m.fs.reformer.inlet.mole_frac_comp[:, :].fix()
    m.fs.reformer.gibbs_scaling = 1e-5

    m.fs.reformer.lagrange_mult[0, 'C'] = 39230.1
    m.fs.reformer.lagrange_mult[0, 'H'] = 81252.4
    m.fs.reformer.lagrange_mult[0, 'O'] = 315048.8

    m.fs.reformer.outlet.flow_mol[0] = 2942.1e-3
    m.fs.reformer.outlet.mole_frac_comp[0, 'O2'] = 1e-12
    m.fs.reformer.outlet.mole_frac_comp[0, 'Ar'] = 0.004
    m.fs.reformer.outlet.mole_frac_comp[0, 'CH4'] = 0.0005
    m.fs.reformer.outlet.mole_frac_comp[0, 'C2H6'] = 1e-10
    m.fs.reformer.outlet.mole_frac_comp[0, 'C3H8'] = 1e-10
    m.fs.reformer.outlet.mole_frac_comp[0, 'C4H10'] = 1e-10
    m.fs.reformer.inlet.mole_frac_comp[0, 'H2'] = 0.3401
    m.fs.reformer.inlet.mole_frac_comp[0, 'CO'] = 0.1123
    m.fs.reformer.inlet.mole_frac_comp[0, 'CO2'] = 0.0518
    m.fs.reformer.outlet.pressure.fix(137895e-3)  # kPa, equal to 20 Psi
    m.fs.reformer.outlet.temperature.fix(1060.93)  # K, equal to 1450 F
    print("Degrees of Freedom = %d" % degrees_of_freedom(m.fs.reformer))
    options = {
        # 'tol': 1e-8,
        # 'bound_push': 1e-23,
        "max_iter": 100,
        # "halt_on_ampl_error": "yes",
    }
    # tried passing options and it isnt working
    m.fs.reformer.initialize(outlvl=0)  # , optarg=options)

    solver = pyo.SolverFactory('ipopt')
    solver.options = {
        "tol": 1e-8,
        'bound_push': 1e-23,
        "max_iter": 40,
        # "halt_on_ampl_error": "yes",
    }
    solver.solve(m.fs.reformer, tee=True)

    m.fs.reformer.inlet.unfix()

    # reformer recuperator
    copy_port_values(
        m.fs.reformer_recuperator.shell_inlet,
        m.fs.reformer.outlet)

    copy_port_values(
        m.fs.reformer_recuperator.tube_inlet,
        m.fs.feed.outlet)

    m.fs.reformer_recuperator.initialize()

    # NG expander
    copy_port_values(
        m.fs.NG_expander.inlet,
        m.fs.reformer_recuperator.tube_outlet)

    m.fs.NG_expander.initialize()

    # reformer bypass
    copy_port_values(
        m.fs.reformer_bypass.inlet, m.fs.NG_expander.outlet)

    m.fs.reformer_bypass.initialize()

    # air compressor train
    m.fs.air_compressor_s1.initialize()

    copy_port_values(
        m.fs.intercooler_s1.inlet, m.fs.air_compressor_s1.outlet)

    m.fs.intercooler_s1.initialize()

    copy_port_values(
        m.fs.air_compressor_s2.inlet, m.fs.intercooler_s1.outlet)

    m.fs.air_compressor_s2.initialize()

    copy_port_values(
        m.fs.intercooler_s2.inlet, m.fs.air_compressor_s2.outlet)

    m.fs.intercooler_s2.initialize()

    # reformer mixer
    copy_port_values(
        m.fs.reformer_mix.oxygen_inlet,
        m.fs.intercooler_s2.outlet)

    copy_port_values(
        m.fs.reformer_mix.gas_inlet,
        m.fs.reformer_bypass.reformer_outlet)

    m.fs.reformer_mix.initialize()

    # bypass rejoin
    copy_port_values(
        m.fs.bypass_rejoin.syngas_inlet,
        m.fs.reformer_recuperator.shell_outlet)

    copy_port_values(
        m.fs.bypass_rejoin.bypass_inlet,
        m.fs.reformer_bypass.bypass_outlet)

    m.fs.bypass_rejoin.initialize()


def add_reformer_var_bounds(m):
    # add upper bound to vars within a GenericStateBlock
    def set_GenericStateBlock_bounds(b):
        b.mole_frac_comp.setub(1)
        b.mole_frac_phase_comp.setub(1)
        b.phase_frac.setub(1)

    def set_ControlVolume_bounds(control_volume):
        set_GenericStateBlock_bounds(control_volume.properties_in[0])
        set_GenericStateBlock_bounds(control_volume.properties_out[0])

        if hasattr(control_volume, 'heat'):
            control_volume.heat.setlb(-1e7)
            control_volume.heat.setub(1e7)

        if hasattr(control_volume, 'deltaP'):
            control_volume.deltaP.setlb(-1e4)
            control_volume.deltaP.setub(1e4)

        if hasattr(control_volume, 'work'):
            control_volume.work.setlb(-1e7)
            control_volume.work.setub(1e7)

    # feed and product
    streams = [m.fs.feed,
               m.fs.product]

    for stream in streams:
        set_GenericStateBlock_bounds(stream.properties[0])

    # heaters
    heaters = [m.fs.heater,
               m.fs.heater,
               m.fs.intercooler_s1,
               m.fs.intercooler_s2]

    for unit in heaters:
        set_ControlVolume_bounds(unit.control_volume)

    # heat exchangers
    heat_exchangers = [m.fs.reformer_recuperator]

    for unit in heat_exchangers:
        unit.delta_temperature_in.setlb(-1000)
        unit.delta_temperature_in.setub(1000)
        unit.delta_temperature_out.setlb(-1000)
        unit.delta_temperature_out.setub(1000)

        set_ControlVolume_bounds(unit.shell)
        set_ControlVolume_bounds(unit.tube)

    # pressure changers
    pressure_changers = [m.fs.NG_expander,
                         m.fs.NG_expander,
                         m.fs.air_compressor_s1,
                         m.fs.air_compressor_s2]

    for unit in pressure_changers:
        unit.work_mechanical[0].setlb(-1e7)
        unit.work_mechanical[0].setub(1e7)
        unit.deltaP[0].setlb(-1e4)
        unit.deltaP[0].setub(1e4)
        unit.ratioP[0].setlb(0)
        unit.ratioP[0].setub(10)
        unit.efficiency_isentropic[0].setlb(0)
        unit.efficiency_isentropic[0].setub(1)
        unit.work_isentropic[0].setlb(-1e7)
        unit.work_isentropic[0].setub(1e7)

        set_ControlVolume_bounds(unit.control_volume)

        set_GenericStateBlock_bounds(unit.properties_isentropic[0])

    # Gibbs reactors
    gibbs = [m.fs.reformer]

    for unit in gibbs:
        set_ControlVolume_bounds(unit.control_volume)
        unit.lagrange_mult.setlb(0)
        unit.lagrange_mult.setub(1e6)

    # mixers
    mixers = [m.fs.bypass_rejoin,
              m.fs.reformer_mix]

    for unit in mixers:
        unit.minimum_pressure.setlb(0)
        unit.minimum_pressure.setub(1e7)
        for o in unit.component_data_objects(active=True, descend_into=False):
            if type(o) == GenericStateBlockData:
                set_GenericStateBlock_bounds(o)

    # separators
    separators = [m.fs.reformer_bypass]

    for unit in separators:
        unit.split_fraction.setlb(0)
        unit.split_fraction.setub(1)
        for o in unit.component_data_objects(active=True, descend_into=False):
            if type(o) == GenericStateBlockData:
                set_GenericStateBlock_bounds(o)


# %% Main Script

if __name__ == "__main__":
    # create model and flowsheet
    m = pyo.ConcreteModel(name='reformer superstructure')
    m.fs = FlowsheetBlock(default={"dynamic": False})

    build_reformer(m)
    set_reformer_inputs(m)

    fname = "reformer_init.json"
    if os.path.exists(fname):
        ms.from_json(m, fname=fname)
        print('Flowsheet initialized')
    else:
        initialize_reformer(m)
        ms.to_json(m, fname=fname)

    ## apply_reformer_scaling(m)
    ## add_reformer_var_bounds(m)

    m.fs.NGtoAir_ratio = pyo.Var(initialize=2.868,
                                  doc='natural gas to air ratio')
    m.fs.NGtoAir_ratio.fix()
    m.fs.NGtoSteam_ratio = pyo.Var(initialize=1,
                                    doc='natural gas to steam ratio')
    m.fs.NGtoSteam_ratio.fix()
    # sets steam flow to reformer based on NG flow
    @m.fs.Constraint()
    def reformer_steam_flow(fs):
        ng_flow = m.fs.reformer_mix.gas_inlet.flow_mol[0]
        steam_flow = m.fs.reformer_mix.steam_inlet.flow_mol[0]
        # IR = m.fs.reformer_bypass.split_fraction[0, 'bypass_outlet']
        return steam_flow == m.fs.NGtoSteam_ratio*ng_flow  # *(1 - IR)*ng_flow
    m.fs.reformer_mix.steam_inlet.flow_mol.unfix()
    # m.fs.reformer_mix.steam_inlet.flow_mol.fix(464.77e-3)  # kmol/s

    # sets air flow to reformer based on NG flow
    @m.fs.Constraint()
    def reformer_air_rule(fs):
        ng_flow = m.fs.reformer_mix.gas_inlet.flow_mol[0]
        air_flow = m.fs.air_compressor_s1.inlet.flow_mol[0]
        # IR = m.fs.reformer_bypass.split_fraction[0, 'bypass_outlet']
        return air_flow == m.fs.NGtoAir_ratio*ng_flow  # *(1 - IR)*ng_flow
    # m.fs.air_compressor_s1.inlet.flow_mol.fix(1332.9e-3)  # kmol/s
    m.fs.air_compressor_s1.inlet.flow_mol.unfix()
    m.fs.obj = pyo.Objective(expr=(m.fs.product.inlet.temperature[0]),
                              sense=pyo.minimize)

    solver = pyo.SolverFactory('ipopt')
    solver.options = {
        "tol": 1e-8,
        'bound_push': 1e-23,
        "max_iter": 40,
        # "halt_on_ampl_error": "yes",
    }
    res = solver.solve(m, tee=True)
