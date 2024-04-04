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
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from idaes.core.util.initialization import propagate_state
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.models.unit_models import GibbsReactor, Mixer
from idaes.core.solvers import get_solver
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop

from svi.auto_thermal_reformer.reactor_model import create_instance
import svi.auto_thermal_reformer.config as config

import time

####### FUNCTION TO GENERATE DATA FOR THE AUTOTHERMAL REFORMER #######

# Inputs: Fin_CH4 [mol/s],Tin_CH4 [K],Fin_H2O [mol/s],Conversion
# Outputs: HeatDuty [W],Fout [mol/s],Tout[K],'H2','CO','H2O','CO2','CH4','C2H6','C3H8','C4H10','N2','O2','Ar'

argparser = config.get_argparser()
argparser.add_argument("--fname", default="training-data.csv", help="Basename for data file for surrogate training")

def atr_data_gen(num_samples = 600):
    df = {'Fin_CH4':[], 'Tin_CH4':[], 'Fin_H2O':[], 'Conversion': [], 'HeatDuty':[], 'Fout':[], 'Tout':[], 'H2':[], 
      'CO':[], 'H2O':[], 'CO2':[], 'CH4':[], 'C2H6':[], 'C3H8':[], 'C4H10':[], 'N2':[], 'O2':[], 'Ar':[]}

    t_start = time.time()
    for _ in range(num_samples):
        try: 
            conversion = random.uniform(0.80,0.96)
            flow_mol_h2o = random.uniform(200,350)
            flow_mol_gas = random.uniform(600,900)
            temp_gas = random.uniform(600,900)

            m = create_instance(
                conversion,
                flow_mol_h2o,
                flow_mol_gas,
                temp_gas,
            )

            ######### SOLVE #########
            solver = get_solver()
            solver.options = {
                "tol": 1e-8,
                "max_iter": 400
            }
            #scc_solver = pyo.SolverFactory("ipopt")
            #solve_strongly_connected_components(
            #    m, solver=scc_solver, use_calc_var=False
            #)
            solver.solve(m, tee=True)

            df[list(df.keys())[0]].append(flow_mol_gas)
            df[list(df.keys())[1]].append(temp_gas)
            df[list(df.keys())[2]].append(flow_mol_h2o)
            df[list(df.keys())[3]].append(conversion)
            df[list(df.keys())[4]].append(m.fs.reformer.heat_duty[0].value)
            df[list(df.keys())[5]].append(pyo.value(m.fs.reformer.outlet.flow_mol[0]))
            df[list(df.keys())[6]].append(pyo.value(m.fs.reformer.outlet.temperature[0]))
            df[list(df.keys())[7]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "H2"]))
            df[list(df.keys())[8]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "CO"]))
            df[list(df.keys())[9]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "H2O"]))
            df[list(df.keys())[10]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "CO2"]))
            df[list(df.keys())[11]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "CH4"]))
            df[list(df.keys())[12]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "C2H6"]))
            df[list(df.keys())[13]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "C3H8"]))
            df[list(df.keys())[14]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "C4H10"]))
            df[list(df.keys())[15]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "N2"]))
            df[list(df.keys())[16]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "O2"]))
            df[list(df.keys())[17]].append(pyo.value(m.fs.reformer.outlet.mole_frac_comp[0, "Ar"]))
        except InitializationError as err:
            print(err)
            continue
        except ValueError as err:
            print(err)
            continue
    t_generate_samples = time.time() - t_start
    print(f"Time to sample inputs and generate data: {t_generate_samples}")
    df = pd.DataFrame(df)
    return df

if __name__ == "__main__":
    args = argparser.parse_args()
    df = atr_data_gen(num_samples = 600)
    fpath = os.path.join(args.data_dir, args.fname)
    df.to_csv(fpath)
