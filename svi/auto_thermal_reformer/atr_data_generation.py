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

import random 
import pandas as pd
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

####### FUNCTION TO GENERATE DATA FOR THE AUTOTHERMAL REFORMER #######

# Inputs: Fin_CH4 [mol/s],Tin_CH4 [K],Fin_H2O [mol/s],Conversion
# Outputs: HeatDuty [W],Fout [mol/s],Tout[K],'H2','CO','H2O','CO2','CH4','C2H6','C3H8','C4H10','N2','O2','Ar'

def atr_data_gen(num_samples):
    df = {'Fin_CH4':[], 'Tin_CH4':[], 'Fin_H2O':[], 'Conversion':[], 'HeatDuty':[], 'Fout':[], 'Tout':[], 'H2':[], 
          'CO':[], 'H2O':[], 'CO2':[], 'CH4':[], 'C2H6':[], 'C3H8':[], 'C4H10':[], 'N2':[], 'O2':[], 'Ar':[]}
    for _ in range(num_samples):
        try:
            m = ConcreteModel()
            m.fs = FlowsheetBlock(dynamic=False)
            components = ['H2', 'CO', "H2O", 'CO2', 'CH4', "C2H6", "C3H8", "C4H10",'N2', 'O2', 'Ar']
            thermo_props_config_dict = get_prop(components = components)
            m.fs.thermo_params = GenericParameterBlock(**thermo_props_config_dict)
            
            m.fs.R101 = GibbsReactor(
            has_heat_transfer = True,
            has_pressure_change = True,
            inert_species = ["N2", "Ar"],
            property_package =  m.fs.thermo_params)

            m.fs.reformer_mix = Mixer(
                inlet_list = ["gas_inlet", "oxygen_inlet", "steam_inlet"],
                property_package = m.fs.thermo_params)

            m.fs.connect = Arc(source=m.fs.reformer_mix.outlet, destination=m.fs.R101.inlet)

            TransformationFactory("network.expand_arcs").apply_to(m)

            ######### DEFINE INLET CONDITIONS FOR THE STEAM #########
            Fin_min_H2O = 200 # mol/s
            Fin_max_H2O = 350 # mol/s

            Fin_H2O = random.uniform(Fin_min_H2O, Fin_max_H2O)

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

            Fin_gas = random.uniform(Fin_min_gas, Fin_max_gas)
            Tin_gas = random.uniform(Tin_min_gas, Tin_max_gas)

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

            m.fs.R101.initialize()  

            ######### SET REFORMER OUTLET PRESSURE #########
            m.fs.R101.outlet.pressure[0].fix(137895)
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

            conv = random.uniform(0.8, 0.95)
            m.fs.R101.conversion.fix(conv)

            ######### SOLVE #########
            solver = get_solver()
            solver.options = {
                "tol": 1e-8,
                "max_iter": 400
            }
            solver.solve(m, tee=False)

            df[list(df.keys())[0]].append(Fin_gas)
            df[list(df.keys())[1]].append(Tin_gas)
            df[list(df.keys())[2]].append(Fin_H2O)
            df[list(df.keys())[3]].append(conv)
            df[list(df.keys())[4]].append(m.fs.R101.heat_duty[0].value)
            df[list(df.keys())[5]].append(pyo.value(m.fs.R101.outlet.flow_mol[0]))
            df[list(df.keys())[6]].append(pyo.value(m.fs.R101.outlet.temperature[0]))
            df[list(df.keys())[7]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "H2"]))
            df[list(df.keys())[8]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "CO"]))
            df[list(df.keys())[9]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "H2O"]))
            df[list(df.keys())[10]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "CO2"]))
            df[list(df.keys())[11]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "CH4"]))
            df[list(df.keys())[12]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "C2H6"]))
            df[list(df.keys())[13]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "C3H8"]))
            df[list(df.keys())[14]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "C4H10"]))
            df[list(df.keys())[15]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "N2"]))
            df[list(df.keys())[16]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "O2"]))
            df[list(df.keys())[17]].append(pyo.value(m.fs.R101.outlet.mole_frac_comp[0, "Ar"]))
        except InitializationError:
            continue
        except ValueError:
            continue
    df = pd.DataFrame(df)
    df.to_csv('data_atr.csv')
    return df

if __name__ == "__main__":
    atr_data_gen(num_samples=600)