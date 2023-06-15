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

#### Data Generation for a Gibbs reactor based on the example given in: 
#### https://idaes.github.io/examples-pse/latest/Examples/UnitModels/Reactors/gibbs_reactor_doc.html

######## IMPORT PACKAGES ########

import random 
import pandas as pd
import pyomo.environ as pyo
from idaes.core.util.exceptions import InitializationError
from pyomo.environ import (
    Constraint,
    Var,
    ConcreteModel,
    units as pyunits,
)

from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.models.unit_models import GibbsReactor
from idaes.core.solvers import get_solver
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop

######## FUNCTION TO GENERATE DATA ########

# T: [K]
# P: [Pa]
# F: [mol/s]
# Heat Duty: [W]

def GibbsDataGen(num_samples, case):
    df = {'Tin':[], 'Pin':[], 'Conversion':[], 'Fin':[], 'HeatDuty':[], 'Fout':[], 
      'Tout':[], 'H2':[], 'H2O':[], 'CO2':[], 'CO':[], 'CH4':[]}
    if case == 1: # more cases could be added to this function depending on what is needed in the optimization problem
        print("Degrees of Freedom are Conversion and Inlet Temperature and Pressure to the Reactor.")
        for _ in range(num_samples):
            try:
                m = ConcreteModel()
                m.fs = FlowsheetBlock(dynamic=False)
                thermo_props_config_dict = get_prop(components=["CH4", "H2O", "H2", "CO", "CO2"])
                m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)

                m.fs.R101 = GibbsReactor(property_package=m.fs.thermo_params,
                                        has_heat_transfer=True,
                                        has_pressure_change=False)
                
                Pin_min = 0.8e6 
                Pin_max = 1.2e6 
                Tin_min = 500 
                Tin_max = 700 
                conversion_CH4_min = 0.5
                conversion_CH4_max = 0.98

                flow_H2O = 234 
                flow_CH4 = 75 
                total_flow_in = flow_H2O + flow_CH4

                m.fs.R101.inlet.mole_frac_comp[0, 'CH4'].fix(flow_CH4/total_flow_in)
                m.fs.R101.inlet.mole_frac_comp[0, 'H2'].fix(9.9996e-06)
                m.fs.R101.inlet.mole_frac_comp[0, 'CO'].fix(9.9996e-06)
                m.fs.R101.inlet.mole_frac_comp[0, 'CO2'].fix(9.9996e-06)
                m.fs.R101.inlet.mole_frac_comp[0, 'H2O'].fix(flow_H2O/total_flow_in)

                m.fs.R101.conversion = Var(bounds=(0, 1), units=pyunits.dimensionless) 

                m.fs.R101.conv_constraint = Constraint(expr=m.fs.R101.conversion * total_flow_in * m.fs.R101.inlet.mole_frac_comp[0, "CH4"] 
                                                    == (total_flow_in * m.fs.R101.inlet.mole_frac_comp[0, "CH4"] - m.fs.R101.outlet.flow_mol[0] 
                                                        * m.fs.R101.outlet.mole_frac_comp[0, "CH4"]))
                
                Tin = random.uniform(Tin_min, Tin_max)
                Pin = random.uniform(Pin_min, Pin_max)
                conversion_CH4 = random.uniform(conversion_CH4_min, conversion_CH4_max)
                m.fs.R101.inlet.temperature.fix(Tin)
                m.fs.R101.inlet.flow_mol.fix(total_flow_in)
                m.fs.R101.inlet.pressure.fix(Pin)
                m.fs.R101.conversion.fix(conversion_CH4)
                m.fs.R101.initialize()
                solver = get_solver()
                solver.solve(m, tee=False)
                df[list(df.keys())[0]].append(Tin)
                df[list(df.keys())[1]].append(Pin)
                df[list(df.keys())[2]].append(conversion_CH4)
                df[list(df.keys())[3]].append(total_flow_in)
                df[list(df.keys())[4]].append(pyo.value(m.fs.R101.heat_duty[0]))
                df[list(df.keys())[5]].append(pyo.value(m.fs.R101.outlet.flow_mol[0]))
                df[list(df.keys())[6]].append(pyo.value(m.fs.R101.outlet.temperature[0]))
                df[list(df.keys())[7]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "H2"]))
                df[list(df.keys())[8]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "H2O"]))
                df[list(df.keys())[9]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "CO2"]))
                df[list(df.keys())[10]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "CO"]))
                df[list(df.keys())[11]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "CH4"]))
            except InitializationError:
                continue
    df = pd.DataFrame(df)
    df.to_csv('data_gibbs.csv')
    return df

