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

########### CREATE AN INDIVIDUAL GIBBS REACTOR ###########
def make_gibbs(conversion):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
    m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

    m.fs.R101 = GibbsReactor(
        property_package=m.fs.thermo_params,
        has_heat_transfer=True,
        has_pressure_change=False,
    )

    m.R101_conversion = Var(bounds=(0, 1), units=pyunits.dimensionless)  # fraction

    m.fs.R101.conv_constraint = Constraint(
        expr=m.R101_conversion
        * m.fs.R101.inlet.flow_mol[0]
        * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
        == (
            m.fs.R101.inlet.flow_mol[0] * m.fs.R101.inlet.mole_frac_comp[0, "CH4"]
            - m.fs.R101.outlet.flow_mol[0] * m.fs.R101.outlet.mole_frac_comp[0, "CH4"]
        )
    )

    m.R101_conversion.fix(conversion)
    m.fs.R101.initialize()

    return m

########### CREATE EXTERNAL PYOMO MODEL FOR GIBBS REACTOR ###########
def make_implicit_gibbs_unit(model):
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
        model.fs.R101.inlet.mole_frac_comp[0, "CH4"],
        model.fs.R101.inlet.mole_frac_comp[0, "H2"],
        model.fs.R101.inlet.mole_frac_comp[0, "CO"],
        model.fs.R101.inlet.mole_frac_comp[0, "CO2"],
        model.fs.R101.inlet.mole_frac_comp[0, "H2O"],
        model.fs.R101.inlet.flow_mol[0],
        model.outlet_temperature,
        model.heatDuty,
        model.outlet_flow_mol,
    ]

    input_vars.extend(model.outlet_mole_frac_comp.values())

    external_eqns = list(igraph.constraints)
    
    external_vars = [var for var in igraph.variables if var is not model.fs.R101.inlet.temperature[0]]
    external_vars = [var for var in external_vars if var is not model.fs.R101.inlet.pressure[0]]
    external_vars = [var for var in external_vars if var is not model.fs.R101.inlet.mole_frac_comp[0, "CH4"]]
    external_vars = [var for var in external_vars if var is not model.fs.R101.inlet.mole_frac_comp[0, "H2"]]
    external_vars = [var for var in external_vars if var is not model.fs.R101.inlet.mole_frac_comp[0, "CO"]]
    external_vars = [var for var in external_vars if var is not model.fs.R101.inlet.mole_frac_comp[0, "CO2"]]
    external_vars = [var for var in external_vars if var is not model.fs.R101.inlet.mole_frac_comp[0, "H2O"]]
    external_vars = [var for var in external_vars if var is not model.fs.R101.inlet.flow_mol[0]]
    
    epm = ExternalPyomoModel(
        input_vars,
        external_vars,
        residual_eqns,
        external_eqns,
    )
    return epm

########### CREATE THE FLOWSHEET AND CONNECT THE IMPLICIT GIBBS REACTOR ###########
def make_implicit_gibbs_flowsheet(epm):  
    m_implicit = ConcreteModel()
    m_implicit.egb = ExternalGreyBoxBlock()
    m_implicit.egb.set_external_model(epm, inputs=epm.input_vars)

    m_implicit.fs = FlowsheetBlock(dynamic=False)
    thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
    m_implicit.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

    m_implicit.fs.CH4 = Feed(property_package=m_implicit.fs.thermo_params)
    m_implicit.fs.H2O = Feed(property_package=m_implicit.fs.thermo_params)
    m_implicit.fs.M101 = Mixer(
        property_package=m_implicit.fs.thermo_params,
        inlet_list=["methane_feed", "steam_feed"],
    )
    m_implicit.fs.H101 = Heater(
        property_package=m_implicit.fs.thermo_params,
        has_pressure_change=False,
        has_phase_equilibrium=False,
    )

    m_implicit.fs.C101 = Compressor(property_package=m_implicit.fs.thermo_params)

    m_implicit.fs.s01 = Arc(
        source=m_implicit.fs.CH4.outlet, destination=m_implicit.fs.M101.methane_feed
    )
    m_implicit.fs.s02 = Arc(
        source=m_implicit.fs.H2O.outlet, destination=m_implicit.fs.M101.steam_feed
    )
    m_implicit.fs.s03 = Arc(
        source=m_implicit.fs.M101.outlet, destination=m_implicit.fs.C101.inlet
    )
    m_implicit.fs.s04 = Arc(
        source=m_implicit.fs.C101.outlet, destination=m_implicit.fs.H101.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m_implicit)

    m_implicit.fs.CH4.outlet.mole_frac_comp[0, "CH4"].fix(1)
    m_implicit.fs.CH4.outlet.mole_frac_comp[0, "H2O"].fix(1e-5)
    m_implicit.fs.CH4.outlet.mole_frac_comp[0, "H2"].fix(1e-5)
    m_implicit.fs.CH4.outlet.mole_frac_comp[0, "CO"].fix(1e-5)
    m_implicit.fs.CH4.outlet.mole_frac_comp[0, "CO2"].fix(1e-5)
    m_implicit.fs.CH4.outlet.flow_mol.fix(75 * pyunits.mol / pyunits.s)
    m_implicit.fs.CH4.outlet.temperature.fix(298.15 * pyunits.K)
    m_implicit.fs.CH4.outlet.pressure.fix(1e5 * pyunits.Pa)

    m_implicit.fs.H2O.outlet.mole_frac_comp[0, "CH4"].fix(1e-5)
    m_implicit.fs.H2O.outlet.mole_frac_comp[0, "H2O"].fix(1)
    m_implicit.fs.H2O.outlet.mole_frac_comp[0, "H2"].fix(1e-5)
    m_implicit.fs.H2O.outlet.mole_frac_comp[0, "CO"].fix(1e-5)
    m_implicit.fs.H2O.outlet.mole_frac_comp[0, "CO2"].fix(1e-5)
    m_implicit.fs.H2O.outlet.flow_mol.fix(234 * pyunits.mol / pyunits.s)
    m_implicit.fs.H2O.outlet.temperature.fix(373.15 * pyunits.K)
    m_implicit.fs.H2O.outlet.pressure.fix(1e5 * pyunits.Pa)

    m_implicit.compressor_efficiency = Constraint(expr=m_implicit.fs.C101.efficiency_isentropic[0] == 0.9)

    # Link the flowsheet to the implicit Gibbs reactor
    @m_implicit.Constraint()
    def linking_T_to_egb(m_implicit):
        return m_implicit.fs.H101.outlet.temperature[0] == m_implicit.egb.inputs[0]

    @m_implicit.Constraint()
    def linking_P_to_egb(m_implicit):
        return m_implicit.fs.H101.outlet.pressure[0] == m_implicit.egb.inputs[1]
    
    @m_implicit.Constraint()
    def linking_CH4_to_egb(m_implicit):
        return m_implicit.fs.H101.outlet.mole_frac_comp[0, "CH4"] == m_implicit.egb.inputs[2]

    @m_implicit.Constraint()
    def linking_H2_to_egb(m_implicit):
        return m_implicit.fs.H101.outlet.mole_frac_comp[0, "H2"] == m_implicit.egb.inputs[3]

    @m_implicit.Constraint()
    def linking_CO_to_egb(m_implicit):
        return m_implicit.fs.H101.outlet.mole_frac_comp[0, "CO"] == m_implicit.egb.inputs[4]

    @m_implicit.Constraint()
    def linking_CO2_to_egb(m_implicit):
        return m_implicit.fs.H101.outlet.mole_frac_comp[0, "CO2"] == m_implicit.egb.inputs[5]
    
    @m_implicit.Constraint()
    def linking_H2O_to_egb(m_implicit):
        return m_implicit.fs.H101.outlet.mole_frac_comp[0, "H2O"] == m_implicit.egb.inputs[6]
    
    @m_implicit.Constraint()
    def linking_flow_to_egb(m_implicit):
        return m_implicit.fs.H101.outlet.flow_mol[0] == m_implicit.egb.inputs[7]

    # Add bounds on auxiliary input variables
    @m_implicit.Constraint()
    def maximum_temperature_H101(m_implicit):
        return m_implicit.fs.H101.outlet.temperature[0] <= 1000

    @m_implicit.Constraint()
    def minimum_pressure_C101(m_implicit):
        return m_implicit.fs.C101.outlet.pressure[0] >= 1e6

    @m_implicit.Constraint()
    def minimum_heatDuty_H101(m_implicit):
        return m_implicit.fs.H101.heat_duty[0] >= 0

    # Set objective 
    m_implicit.fs.cooling_cost = Expression(expr=0.212e-7 * (m_implicit.egb.inputs[9]))
    m_implicit.fs.heating_cost = Expression(expr=2.2e-7 * m_implicit.fs.H101.heat_duty[0])
    m_implicit.fs.compression_cost = Expression(expr=0.12e-5 * m_implicit.fs.C101.work_isentropic[0])
    m_implicit.fs.operating_cost = Expression(expr=(3600 * 8000 * (m_implicit.fs.heating_cost + m_implicit.fs.cooling_cost + m_implicit.fs.compression_cost)))
    m_implicit.fs.objective = Objective(expr=m_implicit.fs.operating_cost)

    # Initialize and solve each unit operation
    m_implicit.fs.CH4.initialize()
    propagate_state(arc=m_implicit.fs.s01)

    m_implicit.fs.H2O.initialize()
    propagate_state(arc=m_implicit.fs.s02)

    m_implicit.fs.M101.initialize()
    propagate_state(arc=m_implicit.fs.s03)

    m_implicit.fs.C101.initialize()
    propagate_state(arc=m_implicit.fs.s04)

    m_implicit.fs.H101.initialize()

    # Solve the flowsheet

    solver = pyo.SolverFactory("cyipopt")
    solver.solve(m_implicit, tee=True)

    return m_implicit.egb.inputs.display()


if __name__ == "__main__":
    model = make_gibbs(conversion = 0.9)
    epm = make_implicit_gibbs_unit(model)
    make_implicit_gibbs_flowsheet(epm)
