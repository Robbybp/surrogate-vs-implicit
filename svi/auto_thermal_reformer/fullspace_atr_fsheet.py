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
from pyomo.network import Arc, SequentialDecomposition
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale

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

from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from pyomo.util.subsystems import TemporarySubsystemManager

#
# For debugging
#
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP


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
    m.fs.STAGE_1_OUT = Arc(
        source=m.fs.air_compressor_s1.outlet, destination=m.fs.intercooler_s1.inlet
    )
    m.fs.IC_1_OUT = Arc(
        source=m.fs.intercooler_s1.outlet, destination=m.fs.air_compressor_s2.inlet
    )
    m.fs.STAGE_2_OUT = Arc(
        source=m.fs.air_compressor_s2.outlet, destination=m.fs.intercooler_s2.inlet
    )
    m.fs.IC_2_OUT = Arc(
        source=m.fs.intercooler_s2.outlet, destination=m.fs.reformer_mix.oxygen_inlet
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


def set_atr_flowsheet_inputs(m):
    # natural gas feed conditions

    m.fs.feed.outlet.flow_mol.fix(1161.9)  # mol/s
    m.fs.feed.outlet.temperature.fix(288.15)  # K
    m.fs.feed.outlet.pressure.fix(3447379)  # Pa
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


def make_simulation_model(initialize=True):
    m = pyo.ConcreteModel(name="ATR_Flowsheet")
    m.fs = FlowsheetBlock(dynamic=False)
    build_atr_flowsheet(m)
    set_atr_flowsheet_inputs(m)
    if initialize:
        initialize_atr_flowsheet(m)

    #
    # Fix degrees of freedom for simulation model
    # (Fix one variable, add one constraint)
    #
    m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].fix()
    m.fs.reformer.conversion = Var(bounds=(0, 1), units=pyunits.dimensionless)
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
    # Fix conversion to 0.95
    m.fs.reformer.conversion.fix(0.95)

    return m


def make_optimization_model(initialize=True):
    """
    The optimization problem to solve is the following:

    Maximize H2 composition in the product stream such that its minimum flow is 3500 mol/s,
    its maximum N2 concentration is 0.3, the maximum reformer outlet temperature is 1200 K and
    the maximum product temperature is 650 K.
    """
    m = make_simulation_model(initialize=initialize)

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
    # ACHIEVE A CONVERSION OF 0.95 IN AUTOTHERMAL REFORMER
    m.fs.reformer.conversion.fix(0.95)

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
        return m.fs.reformer.outlet.temperature[0] <= 1200

    # MAXIMUM PRODUCT OUTLET TEMPERATURE OF 650 K
    @m.Constraint()
    def max_product_temp(m):
        return m.fs.product.temperature[0] <= 650

    # Unfix D.O.F. If you unfix these variables, inlet temperature, flow and composition
    # to the Gibbs reactor will have to be determined by the optimization problem.
    m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"].unfix()
    m.fs.reformer_mix.steam_inlet.flow_mol.unfix()

    return m


if __name__ == "__main__":
    simulation = False
    optimization = not simulation
    visualize = False

    if optimization:
        m = make_optimization_model()
        #m.fs.reformer.conversion.fix(0.9999)

        #
        # Initialize square model
        #
        to_fix = [
            m.fs.reformer_bypass.split_fraction[0, "bypass_outlet"],
            m.fs.reformer_mix.steam_inlet.flow_mol[0],
        ]
        with TemporarySubsystemManager(to_fix=to_fix):
            calc_var_kwds = dict(eps=1e-7)
            solve_kwds = dict(tee=True)
            solver = get_solver()
            solver.options["max_iter"] = 1000
            solve_strongly_connected_components(
                m,
                solver=solver,
                calc_var_kwds=calc_var_kwds,
                solve_kwds=solve_kwds,
            )
        ###

        solver = get_solver()
        solver.options = {"tol": 1e-8, "max_iter": 1000}

        solver.solve(m, tee=True)

        m.fs.reformer.report()
        m.fs.reformer_recuperator.report()
        m.fs.product.report()
        m.fs.reformer_bypass.split_fraction.display()

    if simulation:
        m = make_simulation_model()
        #m.fs.reformer.conversion.fix(0.9999)

        # NOTE: This relies on recent Pyomo PRs
        calc_var_kwds = dict(eps=1e-7)
        solve_kwds = dict(tee=True)
        solver = get_solver()
        solver.options["max_iter"] = 1000
        solve_strongly_connected_components(
            m,
            solver=solver,
            calc_var_kwds=calc_var_kwds,
            solve_kwds=solve_kwds,
        )
        solver.solve(m, tee=True)

        m.fs.reformer.report()
        m.fs.reformer_recuperator.report()
        m.fs.product.report()
        m.fs.reformer_bypass.split_fraction.display()

    if visualize:
        m.fs.visualize("Auto-Thermal-Reformer-Flowsheet")
