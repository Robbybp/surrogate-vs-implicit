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
from pyomo.common.collections import ComponentSet
from pyomo.network import Arc, SequentialDecomposition
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
    Product,
)
from idaes.core.util.initialization import propagate_state
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
from idaes.core.solvers import get_solver
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
    CyIpoptSolverWrapper
)

import os
import pyomo.environ as pyo
from pyomo.util.subsystems import identify_external_functions


def get_external_function_libraries(model):
    ef_exprs = []
    for comp in model.component_data_objects(
        (pyo.Constraint, pyo.Expression, pyo.Objective), active=True
    ):
        ef_exprs.extend(identify_external_functions(comp.expr))
    unique_functions = []
    fcn_set = set()
    for expr in ef_exprs:
        fcn = expr._fcn
        data = (fcn._library, fcn._function)
        if data not in fcn_set:
            fcn_set.add(data)
            unique_functions.append(data)
    unique_libraries = [library for library, function in unique_functions]
    return unique_libraries


def add_external_function_libraries_to_environment(model, envvar="AMPLFUNC"):
    libraries = get_external_function_libraries(model)
    lib_str = "\n".join(libraries)
    os.environ[envvar] = lib_str

def make_implicit(m):
    ########### CREATE EXTERNAL PYOMO MODEL FOR THE AUTOTHERMAL REFORMER ###########
    full_igraph = IncidenceGraphInterface(m)
    reformer_igraph = IncidenceGraphInterface(m.fs.reformer, include_inequality=False)

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

        #m.outlet_temperature,
        #m.heatDuty,
        #m.outlet_flow_mol,
        m.fs.reformer_recuperator.hot_side_inlet.temperature[0],
        m.fs.reformer_recuperator.hot_side_inlet.flow_mol[0],
        # Note that heat_duty is not necessary as it does not appear elsewhere
        # in the model
        #
        # Note that pressure is not an "input" as the pressure linking equation
        # is not residual (because the reactor's outlet pressure is fixed)
        #m.fs.reformer_recuperator.hot_side_inlet.pressure[0],
    ]

    #input_vars.extend(m.outlet_mole_frac_comp.values())
    input_vars.extend(m.fs.reformer_recuperator.hot_side_inlet.mole_frac_comp[0, :])

    external_eqns = list(reformer_igraph.constraints)

    to_exclude = ComponentSet(input_vars)
    to_exclude.add(m.fs.reformer.lagrange_mult[0, "N"])
    to_exclude.add(m.fs.reformer.lagrange_mult[0, "Ar"])
    external_vars = [var for var in reformer_igraph.variables if var not in to_exclude]

    external_var_set = ComponentSet(external_vars)
    external_eqn_set = ComponentSet(external_eqns)
    residual_eqn_set = ComponentSet(residual_eqns)

    # Bounds on variables in the implicit function can lead to
    # undefined derivatives
    for i, var in enumerate(external_vars):
        var.setlb(None)
        var.setub(None)
        var.domain = pyo.Reals

    epm = ExternalPyomoModel(
        input_vars,
        external_vars,
        residual_eqns,
        external_eqns,
        # This forces us to use CyIpopt for the inner Newton solver
        #solver_options=dict(solver_class=CyIpoptSolverWrapper),
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

def build_atr_flowsheet(m):
    ########## ADD THERMODYNAMIC PROPERTIES ##########
    components = [
        "H2",
        "CO",
        "H2O",
        "CO2",
        "CH4",
        "C2H6",
        "C3H8",
        "C4H10",
        "N2",
        "O2",
        "Ar",
    ]
    thermo_props_config_dict = get_prop(components=components)
    m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

    ########## ADD FEED AND PRODUCT STREAMS ##########
    m.fs.feed = Feed(property_package=m.fs.thermo_params)
    m.fs.product = Product(property_package=m.fs.thermo_params)

    ########## ADD UNIT MODELS ##########
    m.fs.reformer_recuperator = HeatExchanger(
        delta_temperature_callback=delta_temperature_underwood_callback,
        hot_side_name="shell",  # hot fluid enters shell
        cold_side_name="tube",  # cold fluid enters tube
        shell={"property_package": m.fs.thermo_params},
        tube={"property_package": m.fs.thermo_params},
    )

    m.fs.NG_expander = PressureChanger(
        compressor=False,
        property_package=m.fs.thermo_params,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    m.fs.reformer_bypass = Separator(
        outlet_list=["reformer_outlet", "bypass_outlet"],
        property_package=m.fs.thermo_params,
    )

    m.fs.reformer_mix = Mixer(
        inlet_list=["gas_inlet", "oxygen_inlet", "steam_inlet"],
        property_package=m.fs.thermo_params,
    )

    m.fs.reformer = GibbsReactor(
        has_heat_transfer=True,
        has_pressure_change=True,
        inert_species=["N2", "Ar"],
        property_package=m.fs.thermo_params,
    )

    m.fs.bypass_rejoin = Mixer(
        inlet_list=["syngas_inlet", "bypass_inlet"], property_package=m.fs.thermo_params
    )

    ########## CONNECT UNIT MODELS ##########

    m.fs.RECUP_COLD_IN = Arc(
        source=m.fs.feed.outlet, destination=m.fs.reformer_recuperator.tube_inlet
    )
    m.fs.RECUP_COLD_OUT = Arc(
        source=m.fs.reformer_recuperator.tube_outlet, destination=m.fs.NG_expander.inlet
    )
    m.fs.NG_EXPAND_OUT = Arc(
        source=m.fs.NG_expander.outlet, destination=m.fs.reformer_bypass.inlet
    )
    m.fs.NG_TO_REF = Arc(
        source=m.fs.reformer_bypass.reformer_outlet,
        destination=m.fs.reformer_mix.gas_inlet,
    )
    
    m.fs.REF_IN = Arc(source=m.fs.reformer_mix.outlet, destination=m.fs.reformer.inlet)
    m.fs.REF_OUT = Arc(
        source=m.fs.reformer.outlet, destination=m.fs.reformer_recuperator.shell_inlet
    )
    m.fs.RECUP_HOT_OUT = Arc(
        source=m.fs.reformer_recuperator.shell_outlet,
        destination=m.fs.bypass_rejoin.syngas_inlet,
    )
    m.fs.REF_BYPASS = Arc(
        source=m.fs.reformer_bypass.bypass_outlet,
        destination=m.fs.bypass_rejoin.bypass_inlet,
    )
    m.fs.PRODUCT = Arc(source=m.fs.bypass_rejoin.outlet, destination=m.fs.product.inlet)

    ########## EXPAND ARCS ##########

    pyo.TransformationFactory("network.expand_arcs").apply_to(m.fs)


def set_atr_flowsheet_inputs(m,T,P):
    # natural gas feed conditions

    m.fs.feed.outlet.flow_mol.fix(1161.9)  # mol/s
    m.fs.feed.outlet.temperature.fix(T)  # K
    m.fs.feed.outlet.pressure.fix(P)  # Pa
    m.fs.feed.outlet.mole_frac_comp[0, "CH4"].fix(0.931)
    m.fs.feed.outlet.mole_frac_comp[0, "C2H6"].fix(0.032)
    m.fs.feed.outlet.mole_frac_comp[0, "C3H8"].fix(0.007)
    m.fs.feed.outlet.mole_frac_comp[0, "C4H10"].fix(0.004)
    m.fs.feed.outlet.mole_frac_comp[0, "CO"].fix(1e-5)
    m.fs.feed.outlet.mole_frac_comp[0, "CO2"].fix(0.01)
    m.fs.feed.outlet.mole_frac_comp[0, "H2"].fix(1e-5)
    m.fs.feed.outlet.mole_frac_comp[0, "H2O"].fix(1e-5)
    m.fs.feed.outlet.mole_frac_comp[0, "N2"].fix(0.016)
    m.fs.feed.outlet.mole_frac_comp[0, "O2"].fix(1e-5)
    m.fs.feed.outlet.mole_frac_comp[0, "Ar"].fix(1e-5)

    # recuperator conditions

    m.fs.reformer_recuperator.area.fix(4190)  # m2
    m.fs.reformer_recuperator.overall_heat_transfer_coefficient.fix(
        80
    )  # W/m2K # it was 80e-3 # potential bug

    # natural gas expander conditions

    m.fs.NG_expander.outlet.pressure.fix(203396)  # Pa
    m.fs.NG_expander.efficiency_isentropic.fix(0.9)

    # steam conditions

    m.fs.reformer_mix.steam_inlet.flow_mol.fix(464.77)  # mol/s
    m.fs.reformer_mix.steam_inlet.temperature.fix(422)  # K
    m.fs.reformer_mix.steam_inlet.pressure.fix(203396)  # Pa
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "H2O"].fix(0.9999)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "CO2"].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "N2"].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "O2"].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "Ar"].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "CH4"].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "C2H6"].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "C3H8"].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "C4H10"].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "CO"].fix(1e-6)
    m.fs.reformer_mix.steam_inlet.mole_frac_comp[0, "H2"].fix(1e-6)

    m.fs.reformer_mix.oxygen_inlet.flow_mol.fix(1332.9)  # mol/s
    m.fs.reformer_mix.oxygen_inlet.temperature.fix(310.93)  # K
    m.fs.reformer_mix.oxygen_inlet.pressure.fix(203396)  # Pa
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "H2O"].fix(0.0104)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "CO2"].fix(0.0003)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "N2"].fix(0.7722)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "O2"].fix(0.2077)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "Ar"].fix(0.00939)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "CH4"].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "C2H6"].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "C3H8"].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "C4H10"].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "CO"].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, "H2"].fix(1e-6)

    # reformer outlet pressure

    m.fs.reformer.outlet.pressure[0].fix(
        137895
    )  # Pa, our Gibbs Reactor has pressure change


def initialize_atr_flowsheet(m):
    ####### PROPAGATE STATES #######

    # initialize the reformer with random values.
    # this is only to get a good set of initial values such that
    # IPOPT can then take over and solve this flowsheet for us.

    m.fs.reformer.inlet.flow_mol[0] = 2262.5
    m.fs.reformer.inlet.temperature[0] = 469.8
    m.fs.reformer.inlet.pressure[0] = 203395.9
    m.fs.reformer.inlet.mole_frac_comp[0, "CH4"] = 0.1912
    m.fs.reformer.inlet.mole_frac_comp[0, "C2H6"] = 0.0066
    m.fs.reformer.inlet.mole_frac_comp[0, "C3H8"] = 0.0014
    m.fs.reformer.inlet.mole_frac_comp[0, "C4H10"] = 0.0008
    m.fs.reformer.inlet.mole_frac_comp[0, "H2"] = 1e-5
    m.fs.reformer.inlet.mole_frac_comp[0, "CO"] = 1e-5
    m.fs.reformer.inlet.mole_frac_comp[0, "CO2"] = 0.0022
    m.fs.reformer.inlet.mole_frac_comp[0, "H2O"] = 0.2116
    m.fs.reformer.inlet.mole_frac_comp[0, "N2"] = 0.4582
    m.fs.reformer.inlet.mole_frac_comp[0, "O2"] = 0.1224
    m.fs.reformer.inlet.mole_frac_comp[0, "Ar"] = 0.0055
    
    m.fs.reformer.initialize()
    m.fs.reformer_recuperator.initialize()
    m.fs.bypass_rejoin.initialize()
    m.fs.product.initialize()
    m.fs.reformer_mix.initialize()
    m.fs.feed.initialize()
    m.fs.NG_expander.initialize()
    m.fs.reformer_bypass.initialize()

    propagate_state(arc=m.fs.RECUP_COLD_IN)
    propagate_state(arc=m.fs.RECUP_COLD_OUT)
    propagate_state(arc=m.fs.NG_EXPAND_OUT)
    propagate_state(arc=m.fs.NG_TO_REF)
    propagate_state(arc=m.fs.REF_IN)


def make_simulation_model(T,P,initialize=True):
    m = pyo.ConcreteModel(name="ATR_Flowsheet")
    m.fs = FlowsheetBlock(dynamic=False)
    build_atr_flowsheet(m)
    set_atr_flowsheet_inputs(m,T,P)
    if initialize:
        initialize_atr_flowsheet(m)
    
    m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].fix(0.3)
    m.fs.reformer_mix.steam_inlet.flow_mol.fix(466.7)
    solver = get_solver()
    solver.solve(m, tee=True)
    return m


def make_optimization_model(T,P,initialize=True):
    """
    The optimization problem to solve is the following:

    Maximize H2 composition in the product stream such that its minimum flow is 3500 mol/s,
    its maximum N2 concentration is 0.3, the maximum reformer outlet temperature is 1200 K and
    the maximum product temperature is 650 K.
    """
    m = make_simulation_model(T,P,initialize=initialize)

    # TODO: Optionally solve the simulation model at this point so we start
    # the optimization problem with no primal infeasibility (other than due
    # to any operational constraints we might impose).

    ####### OBJECTIVE IS TO MAXIMIZE H2 COMPOSITION IN PRODUCT STREAM #######
    m.fs.obj = pyo.Objective(
        expr=m.fs.product.mole_frac_comp[0, "H2"], sense=pyo.maximize
    )

    ####### CONSTRAINTS #######
    m.fs.reformer.conversion = Var(
        bounds=(0, 1), units=pyunits.dimensionless
    )  # fraction

    m.fs.reformer.conv_constraint = Constraint(
        expr=m.fs.reformer.conversion
        * m.fs.reformer.inlet.flow_mol[0]
        * m.fs.reformer.inlet.mole_frac_comp[0, "CH4"]
        == (
            m.fs.reformer.inlet.flow_mol[0]
            * m.fs.reformer.inlet.mole_frac_comp[0, "CH4"]
            - m.fs.reformer.outlet.flow_mol[0]
            * m.fs.reformer.outlet.mole_frac_comp[0, "CH4"]
        )
    )
    # ACHIEVE A CONVERSION OF 0.94 IN AUTOTHERMAL REFORMER
    m.fs.reformer.conversion.fix(0.94)

    # MINIMUM PRODUCT FLOW OF 3500 mol/s IN PRODUCT STREAM
    @m.Constraint()
    def min_product_flow_mol(m):
        return m.fs.product.flow_mol[0] >= 3500

    # MAXIMUM N2 COMPOSITION OF 0.3 IN PRODUCT STREAM
    @m.Constraint()
    def max_product_N2_comp(m):
        return m.fs.product.mole_frac_comp[0, "N2"] <= 0.3

    # MAXIMUM REFORMER OUTLET TEMPERATURE OF 1200 K
    @m.Constraint()
    def max_reformer_outlet_temp(m):
        #return m.fs.reformer.outlet.temperature[0] <= 1200
        return m.fs.reformer_recuperator.hot_side_inlet.temperature[0] <= 1200.0

    # MAXIMUM PRODUCT OUTLET TEMPERATURE OF 650 K
    @m.Constraint()
    def max_product_temp(m):
        return m.fs.product.temperature[0] <= 650

    # Unfix D.O.F. If you unfix these variables, inlet temperature, flow and composition
    # to the Gibbs reactor will have to be determined by the optimization problem.
    m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].unfix()
    m.fs.reformer_mix.steam_inlet.flow_mol.unfix()

    return m

df = {'T':[], 'P':[], 'Termination':[], 'Time':[], 'Objective':[], 'Steam':[], 'Bypass Frac': []}

def main(T,P):
    m = make_optimization_model(T,P)
    add_external_function_libraries_to_environment(m)
    m_implicit = make_implicit(m)
    solver = pyo.SolverFactory("cyipopt", options = {"tol": 1e-6,"max_iter": 120})
    timer = TicTocTimer()
    timer.tic('starting timer')
    results = solver.solve(m_implicit, tee=True)
    dT = timer.toc('end')
    df[list(df.keys())[0]].append(T)
    df[list(df.keys())[1]].append(P)
    df[list(df.keys())[2]].append(results.solver.termination_condition)
    df[list(df.keys())[3]].append(dT)
    df[list(df.keys())[4]].append(value(m.fs.product.mole_frac_comp[0, 'H2']))
    df[list(df.keys())[5]].append(value(m.fs.reformer_mix.steam_inlet.flow_mol[0]))
    df[list(df.keys())[6]].append(value(m.fs.reformer_bypass.split_fraction[0, 'bypass_outlet']))

if __name__ == "__main__":
    for T in np.arange(288.15,338.15,7):
        for P in np.arange(1447379,1947379,70000):
            try:
                main(T,P)
            except AssertionError:
                df[list(df.keys())[0]].append(T)
                df[list(df.keys())[1]].append(P)
                df[list(df.keys())[2]].append("AMPL Error")
                df[list(df.keys())[3]].append("999")
                df[list(df.keys())[4]].append("999")
                df[list(df.keys())[5]].append("999")
                df[list(df.keys())[6]].append("999")
                continue
            except OverflowError:
                df[list(df.keys())[0]].append(T)
                df[list(df.keys())[1]].append(P)
                df[list(df.keys())[2]].append("Overflow Error")
                df[list(df.keys())[3]].append("999")
                df[list(df.keys())[4]].append("999")
                df[list(df.keys())[5]].append("999")
                df[list(df.keys())[6]].append("999")
                continue
            except RuntimeError:
                df[list(df.keys())[0]].append(T)
                df[list(df.keys())[1]].append(P)
                df[list(df.keys())[2]].append("Runtime Error")
                df[list(df.keys())[3]].append("999")
                df[list(df.keys())[4]].append("999")
                df[list(df.keys())[5]].append("999")
                df[list(df.keys())[6]].append("999")
                continue

df = pd.DataFrame(df)
df.to_csv("implicit_sweep_test.csv")