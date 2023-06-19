###############################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
###############################################################################

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

def fullspace_gibbs_flowsheet(conversion):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
    m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

    m.fs.CH4 = Feed(property_package=m.fs.thermo_params)
    m.fs.H2O = Feed(property_package=m.fs.thermo_params)
    m.fs.PROD = Product(property_package=m.fs.thermo_params)
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
    m.fs.s06 = Arc(source=m.fs.R101.outlet, destination=m.fs.PROD.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.hyd_prod = Expression(
        expr=pyunits.convert(
            m.fs.PROD.inlet.flow_mol[0]
            * m.fs.PROD.inlet.mole_frac_comp[0, "H2"]
            * m.fs.thermo_params.H2.mw,  # MW defined in properties as kg/mol
            to_units=pyunits.Mlb / pyunits.yr,
        )
    )  # converting kg/s to MM lb/year

    m.fs.cooling_cost = Expression(expr=0.212e-7 * (m.fs.R101.heat_duty[0]))  # the reaction is endothermic, so R101 duty is positive
    m.fs.heating_cost = Expression(expr=2.2e-7 * m.fs.H101.heat_duty[0])  # the stream must be heated to T_rxn, so H101 duty is positive
    m.fs.compression_cost = Expression(expr=0.12e-5 * m.fs.C101.work_isentropic[0])  # the stream must be pressurized, so the C101 work is positive
    m.fs.operating_cost = Expression(expr=(3600 * 8000 * (m.fs.heating_cost + m.fs.cooling_cost + m.fs.compression_cost)))

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

    m.fs.C101.efficiency_isentropic.fix(0.90) #

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

    # Initialize and solve each unit operation
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

    m.fs.R101.initialize()
    propagate_state(arc=m.fs.s06)

    m.fs.PROD.initialize()

    m.fs.objective = Objective(expr=m.fs.operating_cost)
    m.fs.R101.conversion.fix(conversion)
    m.fs.C101.outlet.pressure.unfix()
    m.fs.C101.outlet.pressure[0].setlb(pyunits.convert(1 * pyunits.bar, to_units=pyunits.Pa))  # equals inlet pressure
    m.fs.C101.outlet.pressure[0].setlb(pyunits.convert(10 * pyunits.bar, to_units=pyunits.Pa))  # at most, pressurize to 1 bar
    m.fs.H101.outlet.temperature.unfix()
    m.fs.H101.heat_duty[0].setlb(0 * pyunits.J / pyunits.s)  # ensures outlet is equal to or greater than inlet temperature
    m.fs.H101.outlet.temperature[0].setub(1000 * pyunits.K)  # at most, heat to 1000 K

    assert degrees_of_freedom(m) == 2

    return m

def main():
    m = fullspace_gibbs_flowsheet(conversion = 0.9)
    solver = get_solver()
    solver.solve(m, tee=True)

    print("Gibbs Reactor Report")
    print()
    print(f"Inlet Temperature: {value(m.fs.H101.outlet.temperature[0]):1.2f} K.")
    print(f"Inlet Pressure: {value(m.fs.H101.outlet.pressure[0]):1.2f} Pa.")
    print()
    print(f"Heat Duty: {value(m.fs.R101.heat_duty[0]):1.2f} W.")
    print(f"Outlet Molar Flow Rate: {value(m.fs.R101.outlet.flow_mol[0]):1.2f} mol/s.")
    print(f"Outlet Temperature: {value(m.fs.R101.outlet.temperature[0]):1.2f} K.")
    print(f"Outlet H2 Composition: {value(m.fs.R101.outlet.mole_frac_comp[0, 'H2']):1.5f}.")
    print(f"Outlet H2O Composition: {value(m.fs.R101.outlet.mole_frac_comp[0, 'H2O']):1.5f}.")
    print(f"Outlet CO2 Composition: {value(m.fs.R101.outlet.mole_frac_comp[0, 'CO2']):1.5f}.")
    print(f"Outlet CO Composition: {value(m.fs.R101.outlet.mole_frac_comp[0, 'CO']):1.5f}.")
    print(f"Outlet CH4 Composition: {value(m.fs.R101.outlet.mole_frac_comp[0, 'CH4']):1.5f}.")
    print()
    print(f"Minimum operating cost per year: USD {value(m.fs.objective)/1e6:1.2f} million/yr.")

if __name__ == "__main__":
    main()

    





