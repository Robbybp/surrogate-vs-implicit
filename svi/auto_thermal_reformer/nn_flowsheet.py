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
import pyomo.environ as pyo
from pyomo.common.timing import TicTocTimer
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
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.sampling.scaling import OffsetScaler
from idaes.core.surrogate.keras_surrogate import (
    KerasSurrogate,
    save_keras_json_hd5,
    load_keras_json_hd5,
)

import svi.auto_thermal_reformer.fullspace_flowsheet as fullspace
import svi.auto_thermal_reformer.config as config


DEFAULT_SURROGATE_FNAME = "keras_surrogate_high_rel"


def _get_nn_surrogate_fname():
    # TODO: Accept arguments so we can override the default results dir.
    # Note that this function is essentially hard-coding the default
    # surrogate file.
    default_results_dir = config.get_results_dir()
    return os.path.join(default_results_dir, DEFAULT_SURROGATE_FNAME)


def create_instance(
    conversion,
    pressure,
    initialize=True,
    surrogate_fname=None
):

    if surrogate_fname is None:
        surrogate_fname = _get_nn_surrogate_fname()

    m = fullspace.make_simulation_model(pressure, initialize = True)

    for con in m.fs.reformer.component_objects(pyo.Constraint):
        con.deactivate()
    for con in m.fs.reformer_mix.component_objects(pyo.Constraint):
        con.deactivate()

    m.fs.REF_IN_expanded.deactivate()

    ########## DEFINE SURROGATE BLOCK FOR THE ATR ##########
    m.fs.reformer_surrogate = SurrogateBlock()

    m.fs.reformer.conversion.fix(conversion)

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
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "N2"],
        m.fs.reformer_surrogate.out_mole_frac_comp[0, "Ar"],
    ]

    #dirname = os.path.dirname(__file__)
    #data_dir = os.path.join(dirname, "data")
    #basename = "keras_surrogate_low_rel"
    #keras_surrogate = os.path.join(data_dir, basename)

    keras_surrogate = KerasSurrogate.load_from_folder(surrogate_fname)

    m.fs.reformer_surrogate.build_model(
        keras_surrogate,
        formulation=KerasSurrogate.Formulation.FULL_SPACE,
        input_vars=inputs,
        output_vars=outputs,
    )

    m.fs.reformer_bypass.reformer_outlet_state[0.0].flow_mol.setlb(0.0)
    m.fs.reformer_bypass.reformer_outlet_state[0.0].flow_mol.setub(50000.0)
    m.fs.reformer_surrogate.conversion.setlb(0.0)
    m.fs.reformer_surrogate.conversion.setub(1.0)

    fullspace.add_obj_and_constraints(m)

    return m


def initialize_nn_atr_flowsheet(m):
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
    m.fs.reformer_bypass.inlet.temperature.fix(700)
    m.fs.reformer_bypass.initialize()


if __name__ == "__main__":

    X = 0.95
    P = 1650000.0
    m = create_instance(X, P) 
    initialize_nn_atr_flowsheet(m)
    m.fs.reformer_bypass.inlet.temperature.unfix()
    m.fs.reformer_bypass.inlet.flow_mol.unfix()

    solver = config.get_optimization_solver()
    timer = TicTocTimer()
    timer.tic('starting timer')
    results = solver.solve(m, tee=True)
    dT = timer.toc('end')
