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
import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.models.unit_models import (
    Feed,
    Mixer,
    Compressor,
    Heater,
    GibbsReactor,
)

from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

def make_fullspace_gibbs_flowsheet(conversion):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
    m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

    m.fs.CH4 = Feed(property_package=m.fs.thermo_params)
    m.fs.H2O = Feed(property_package=m.fs.thermo_params)
    m.fs.M101 = Mixer(property_package=m.fs.thermo_params, inlet_list=["methane_feed", "steam_feed"])
    m.fs.C101 = Compressor(property_package=m.fs.thermo_params)
    m.fs.H101 = Heater(
        property_package=m.fs.thermo_params,
        has_pressure_change=False,
        has_phase_equilibrium=False,
    )
    
    m.fs.R101 = GibbsReactor(
        property_package=m.fs.thermo_params,
        has_heat_transfer=True,
        has_pressure_change=False,
    )

    m.fs.s01 = Arc(source=m.fs.CH4.outlet, destination=m.fs.M101.methane_feed)
    m.fs.s02 = Arc(source=m.fs.H2O.outlet, destination=m.fs.M101.steam_feed)
    m.fs.s03 = Arc(source=m.fs.M101.outlet, destination=m.fs.C101.inlet)
    m.fs.s04 = Arc(source=m.fs.C101.outlet, destination=m.fs.H101.inlet)
    # m.fs.s05 = Arc(source=m.fs.H101.outlet, destination=m.fs.R101.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

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

    m.fs.R101.conversion = Var(bounds=(0, 1), units=pyunits.dimensionless)  # fraction

    ########### R101 CONVERSION CONSTRAINT ###########
    m.fs.R101.conv_constraint = Constraint(
        expr=m.fs.R101.conversion
        * m.fs.R101.inlet.flow_mol[0]
        * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
        == (
            m.fs.R101.inlet.flow_mol[0] * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
            - m.fs.R101.outlet.flow_mol[0] * m.fs.R101.outlet.mole_frac_comp[0, "CH4"]
        )
    )

    m.fs.R101.conversion.fix(conversion)

    ########### OBJECTIVE AND CONSTRAINTS ###########
    m.fs.objective = Objective(expr=m.fs.operating_cost) 
    m.fs.C101_min_outlet_P = Constraint(expr = m.fs.C101.outlet.pressure[0] >= 1e6)
    m.fs.HeatDuty_H101 = Constraint(expr = m.fs.H101.heat_duty[0] >= 0)
    m.fs.H101_max_outlet_T = Constraint(expr = m.fs.H101.outlet.temperature[0] <= 1000)
    m.fs.C101_efficiency = Constraint(expr = m.fs.C101.efficiency_isentropic[0] == 0.90)

    ########### INITIALIZE AND SOLVE EACH UNIT OPERATION ###########
    m.fs.CH4.initialize()
    propagate_state(arc=m.fs.s01)

    m.fs.H2O.initialize()
    propagate_state(arc=m.fs.s02)

    m.fs.M101.initialize()
    propagate_state(arc=m.fs.s03)

    m.fs.C101.initialize()
    propagate_state(arc=m.fs.s04)

    m.fs.H101.initialize()
    # propagate_state(arc=m.fs.s05)

    ########### PROPAGATE STATE FROM H101 TO R101 ###########
    m.fs.R101.inlet.flow_mol.fix(m.fs.H101.outlet.flow_mol[0])
    m.fs.R101.inlet.mole_frac_comp[0, "CH4"].fix(m.fs.H101.outlet.mole_frac_comp[0, "CH4"])
    m.fs.R101.inlet.mole_frac_comp[0, "H2O"].fix(m.fs.H101.outlet.mole_frac_comp[0, "H2O"])
    m.fs.R101.inlet.mole_frac_comp[0, "H2"].fix(m.fs.H101.outlet.mole_frac_comp[0, "H2"])
    m.fs.R101.inlet.mole_frac_comp[0, "CO"].fix(m.fs.H101.outlet.mole_frac_comp[0, "CO"])
    m.fs.R101.inlet.mole_frac_comp[0, "CO2"].fix(m.fs.H101.outlet.mole_frac_comp[0, "CO2"])
    
    return m

def make_implicit(model):
    ########### CREATE EXTERNAL PYOMO MODEL FOR GIBBS REACTOR ###########
    igraph = IncidenceGraphInterface(model.fs.R101, include_inequality=False)

    # Unfix inputs, which are the degrees of freedom of the optimization problem
    model.fs.R101.inlet.temperature.unfix()
    model.fs.R101.inlet.pressure.unfix()

    model.outlet_mole_frac_comp = Var(model.fs.thermo_params.component_list)
    for j in model.fs.thermo_params.component_list:
        model.outlet_mole_frac_comp[j] = model.fs.R101.outlet.mole_frac_comp[0, j]

    model.outlet_temperature = Var(initialize=model.fs.R101.outlet.temperature[0].value)
    model.heatDuty = Var(initialize=model.fs.R101.heat_duty[0].value)
    model.outlet_flow_mol = Var(initialize=model.fs.R101.outlet.flow_mol[0].value)

    @model.Constraint(model.fs.thermo_params.component_list)
    def outlet_mole_frac_comp_eq(model, j):
        return model.outlet_mole_frac_comp[j] == model.fs.R101.outlet.mole_frac_comp[0, j]

    @model.Constraint()
    def outlet_temperature_eq(model):
        return model.outlet_temperature == model.fs.R101.outlet.temperature[0]

    @model.Constraint()
    def heatDuty_eq(model):
        return model.heatDuty == model.fs.R101.heat_duty[0]

    @model.Constraint()
    def outlet_flow_mol_eq(model):
        return model.outlet_flow_mol == model.fs.R101.outlet.flow_mol[0]

    residual_eqns = [
        model.outlet_temperature_eq,
        model.heatDuty_eq,
        model.outlet_flow_mol_eq,
    ]
    residual_eqns.extend(model.outlet_mole_frac_comp_eq.values())

    input_vars = [
        model.fs.R101.inlet.temperature[0],
        model.fs.R101.inlet.pressure[0],
        model.outlet_temperature,
        model.heatDuty,
        model.outlet_flow_mol,
    ]

    input_vars.extend(model.outlet_mole_frac_comp.values())

    external_eqns = list(igraph.constraints)
    external_vars = [var for var in igraph.variables if var is not model.fs.R101.inlet.temperature[0]]
    external_vars = [var for var in external_vars if var is not model.fs.R101.inlet.pressure[0]]

    external_var_set = ComponentSet(external_vars)
    external_eqn_set = ComponentSet(external_eqns)
    residual_eqn_set = ComponentSet(residual_eqns)

    epm = ExternalPyomoModel(
        input_vars,
        external_vars,
        residual_eqns,
        external_eqns,
    )

    ########### CONNECT FLOWSHEET TO THE IMPLICIT GIBBS REACTOR ###########
    m_implicit = ConcreteModel()
    m_implicit.egb = ExternalGreyBoxBlock()
    m_implicit.egb.set_external_model(epm, inputs=input_vars)

    m_implicit.link_T_EGB = Constraint(expr = model.fs.H101.outlet.temperature[0] == m_implicit.egb.inputs[0])
    m_implicit.link_P_EGB = Constraint(expr = model.fs.H101.outlet.pressure[0] == m_implicit.egb.inputs[1])
    m_implicit.link_HeatDuty_EGB = Constraint(expr = model.fs.R101.heat_duty[0] == m_implicit.egb.inputs[3])

    # TODO: Attach rest of flowsheet model to m_implicit
    full_igraph = IncidenceGraphInterface(model)
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
    m_implicit.objective = pyo.Reference(model.fs.objective)

    for con in full_igraph.get_adjacent_to(model.fs.R101.control_volume.heat[0.0]):
        print(con.name)

    solver = pyo.SolverFactory("cyipopt")
    solver.solve(m_implicit, tee=True)
    return m_implicit.egb.inputs.display()

if __name__ == "__main__":
    m1 = make_fullspace_gibbs_flowsheet(conversion=0.9)
    solver = pyo.SolverFactory("ipopt")
    solver.solve(m1, tee=True)
    # Print results

    model = make_fullspace_gibbs_flowsheet(conversion = 0.9)
    m_implicit = make_implicit(model)
