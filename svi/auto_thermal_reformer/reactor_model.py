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
from idaes.core.util.exceptions import InitializationError
from pyomo.environ import (
    Constraint,
    Var,
    ConcreteModel,
    units as pyunits,
    TransformationFactory
)

from idaes.core import FlowsheetBlock
from pyomo.network import Arc
from idaes.core.util.initialization import propagate_state
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.models.unit_models import GibbsReactor, Mixer
from idaes.core.solvers import get_solver
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop



def add_reactor_model(
    m,
    conversion=0.95,
    flow_mol_h2o=300,
    flow_mol_gas=750,
    temp_gas=750,
    initialize=True,
):
    """Add reactor unit model to provided Pyomo/IDAES model

    Here we assume the model has been set up with a flowsheet, property package,
    and whatever other "global" components are necessary to add the reactor model.
    This function is suitable for constructing a standalone reactor model. That
    is, it does not attempt to add any arcs to connect to the rest of the
    flowsheet.

    """
    m.fs.reformer = GibbsReactor(
    has_heat_transfer = True,
    has_pressure_change = True,
    inert_species = ["N2", "Ar"],
    property_package =  m.fs.thermo_params)

    m.fs.reformer_mix = Mixer(
        inlet_list = ["gas_inlet", "oxygen_inlet", "steam_inlet"],
        property_package = m.fs.thermo_params)

    m.fs.connect = Arc(source=m.fs.reformer_mix.outlet, destination=m.fs.reformer.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    ######### DEFINE INLET CONDITIONS FOR THE STEAM #########
    #Fin_min_H2O = 200 # mol/s
    #Fin_max_H2O = 350 # mol/s
    Fin_H2O = flow_mol_h2o

    m.fs.reformer_mix.steam_inlet.flow_mol.fix(Fin_H2O) 
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

    ######### DEFINE INLET CONDITIONS FOR THE AIR #########
    m.fs.reformer_mix.oxygen_inlet.flow_mol.fix(1332.9)  # mol/s
    m.fs.reformer_mix.oxygen_inlet.temperature.fix(310.93)  # K
    m.fs.reformer_mix.oxygen_inlet.pressure.fix(203396)  # Pa 
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'CO2'].fix(0.0003)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'H2O'].fix(0.0104)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'N2'].fix(0.7722)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'O2'].fix(0.2077)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'Ar'].fix(0.00939)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'CH4'].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'C2H6'].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'C3H8'].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'C4H10'].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'CO'].fix(1e-6)
    m.fs.reformer_mix.oxygen_inlet.mole_frac_comp[0, 'H2'].fix(1e-6)

    ######### DEFINE INLET CONDITIONS FOR THE NATURAL GAS #########
    Fin_min_gas = 600 # mol/s
    Fin_max_gas = 900 # mol/s

    Tin_min_gas = 600  # K
    Tin_max_gas = 900  # K

    Fin_gas = flow_mol_gas
    Tin_gas = temp_gas

    m.fs.reformer_mix.gas_inlet.flow_mol.fix(Fin_gas)  
    m.fs.reformer_mix.gas_inlet.temperature.fix(Tin_gas)  
    m.fs.reformer_mix.gas_inlet.pressure.fix(203396) # Pa
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'CH4'].fix(0.931)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'C2H6'].fix(0.032)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'C3H8'].fix(0.007)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'C4H10'].fix(0.004)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'CO'].fix(1e-5)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'CO2'].fix(0.01)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'H2'].fix(1e-5)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'H2O'].fix(1e-5)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'N2'].fix(0.016)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'O2'].fix(1e-5)
    m.fs.reformer_mix.gas_inlet.mole_frac_comp[0, 'Ar'].fix(1e-5)

    ######### INITIALIZE UNITS AND PROPAGATE STATES #########
    m.fs.reformer_mix.initialize()  
    propagate_state(arc=m.fs.connect)

    if initialize:
        m.fs.reformer.initialize()  

    ######### SET REFORMER OUTLET PRESSURE #########
    m.fs.reformer.outlet.pressure[0].fix(137895)
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

    m.fs.reformer.conversion.fix(conversion)


def create_instance(
    conversion=0.95,
    flow_mol_h2o=300,
    flow_mol_gas=750,
    temp_gas=750,
    initialize=True,
):
    # Set up global information
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    components = ['H2', 'CO', "H2O", 'CO2', 'CH4', "C2H6", "C3H8", "C4H10",'N2', 'O2', 'Ar']
    thermo_props_config_dict = get_prop(components = components)
    m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

    add_reactor_model(
        m,
        conversion=conversion,
        flow_mol_h2o=flow_mol_h2o,
        flow_mol_gas=flow_mol_gas,
        temp_gas=temp_gas,
        initialize=initialize,
    )

    return m


if __name__ == "__main__":
    from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
    from svi.validate import validate_solution
    m = create_instance()
    igraph = IncidenceGraphInterface(m)
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()
    assert not var_dmp.unmatched
    assert not con_dmp.unmatched

    solver = pyo.SolverFactory("ipopt")
    solver.solve(m, tee=True)
    tol = 1e-5
    validate_solution(m, tolerance=tol)
