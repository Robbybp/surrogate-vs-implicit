##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Task: Artificial Intelligence/Machine Learning
Scenario: Gibbs Reactor for Syngas Reformation
Author: B. Paul and M. Zamarripa
"""

import os
import sys

from collections import OrderedDict

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.network.plugins import expand_arcs
from pyomo.common.fileutils import this_file_dir
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.gdp import Disjunct, Disjunction

# IDAES Imports
from idaes.core import FlowsheetBlock
from idaes.core.util import copy_port_values
from idaes.core.util import model_serializer as ms
from idaes.core.util.misc import svg_tag
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.tables import create_stream_table_dataframe

import idaes.core.util.scaling as iscale
from idaes.generic_models.properties.core.generic.generic_property import (
    GenericParameterBlock, GenericStateBlockData)
from idaes.generic_models.properties.core.generic.generic_reaction import (
    GenericReactionParameterBlock)

from idaes.generic_models.unit_models import (
    Mixer,
    Heater,
    HeatExchanger,
    PressureChanger,
    GibbsReactor,
    StoichiometricReactor,
    Separator,
    Translator,
    Feed)

# imports from other files
#sys.path.append(os.path.abspath("C:/Users/MZamarripa/Documents/#IDAES/ARPA_E/Differentiate/code/flowsheets_5182021/ROM_library")) #local folder
#from SOFC_DNN_ROM_builder import build_SOFC_ROM, #initialize_SOFC_ROM

#sys.path.append(os.path.abspath("C:/Users/bpaul/GitHub/Keras-files")) #local folder
sys.path.append(os.path.abspath(".../Core-tasks-2.5")) #working directory
from natural_gas_PR_scaled_units import get_NG_properties, rxn_configuration

# create model and flowsheet
print("Test case '1' fixes outlet temperature, with heat duty and outlet flow as outputs.")
print("Test case '2' fixes heat duty to 0, with outlet temperature, outlet flow and outlet mole fractions as outputs.")
case_number = input("Enter case number (invalid value will default to case 1):")
if case_number == '1':
    print('Running case ',case_number, '...')
    print("Initializing run...")
elif case_number == '2':
    print('Running case ',case_number, '...')
    print("Initializing run...")
else:
    print("Valid value not entered, defaulting to case 1...")
    print("Initializing run...")
    case_number = '1'

m = pyo.ConcreteModel()
m.fs = FlowsheetBlock(default={"dynamic": False})
# create property packages - 3 property packages and 1 reaction
NG_config = get_NG_properties(
    components=['H2', 'CO', "H2O", 'CO2', 'CH4', "C2H6", "C3H8", "C4H10",
                'N2', 'O2', 'Ar'])
component_list = ['H2', 'CO', "H2O", 'CO2', 'CH4', "C2H6", "C3H8", "C4H10",
                'N2', 'O2', 'Ar']
m.fs.NG_props = GenericParameterBlock(default=NG_config)


m.fs.ref_on = pyo.Block()
m.fs.ref_on.reformer = GibbsReactor(
    default={"has_heat_transfer": True,
             "has_pressure_change": True,
             "inert_species": ["N2", "Ar"],
             "property_package": m.fs.NG_props})

print("The model has ",degrees_of_freedom(m), " degrees of freedom, assigning inlet values...")

m.fs.ref_on.reformer.inlet.flow_mol[0] = 2262.5e-3  # mol/s
m.fs.ref_on.reformer.inlet.temperature[0] = 469.8  # K
m.fs.ref_on.reformer.inlet.pressure[0] = 203395.9e-3  # Pa
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'CH4'] = 0.1912
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'C2H6'] = 0.0066
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'C3H8'] = 0.0014
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'C4H10'] = 0.0008
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'H2'] = 0
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'CO'] = 0
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'CO2'] = 0.0022
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'H2O'] = 0.2116
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'N2'] = 0.4582
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'O2'] = 0.1224
m.fs.ref_on.reformer.inlet.mole_frac_comp[0, 'Ar'] = 0.0055
print("The model has ",degrees_of_freedom(m), " degrees of freedom, assigning outlet values...")
m.fs.ref_on.reformer.lagrange_mult[0, 'C'] = 39230.1
m.fs.ref_on.reformer.lagrange_mult[0, 'H'] = 81252.4
m.fs.ref_on.reformer.lagrange_mult[0, 'O'] = 315048.8

m.fs.ref_on.reformer.outlet.flow_mol[0] = 2942.1e-3
m.fs.ref_on.reformer.outlet.mole_frac_comp[0, 'O2'] = 0
m.fs.ref_on.reformer.outlet.mole_frac_comp[0, 'Ar'] = 0.004
m.fs.ref_on.reformer.outlet.mole_frac_comp[0, 'CH4'] = 0.0005
m.fs.ref_on.reformer.outlet.mole_frac_comp[0, 'C2H6'] = 0
m.fs.ref_on.reformer.outlet.mole_frac_comp[0, 'C3H8'] = 0
m.fs.ref_on.reformer.outlet.mole_frac_comp[0, 'C4H10'] = 0

print("The model has ",degrees_of_freedom(m), " degrees of freedom, fixing values...")

# iscale.calculate_scaling_factors(m)
#init_fname = 'NGFC_master_superstructure_init.json'
#ms.from_json(m, fname=init_fname)
m.fs.ref_on.reformer.inlet.flow_mol[0].fix()  # mol/s
m.fs.ref_on.reformer.inlet.temperature[0].fix()  # K
m.fs.ref_on.reformer.inlet.pressure[0].fix()  # Pa
m.fs.ref_on.reformer.inlet.mole_frac_comp[:,:].fix()

# reformer outlet pressure
m.fs.ref_on.reformer.outlet.pressure.fix(137895e-3)  # Pa, equal to 20 Psi

print("The model has ",degrees_of_freedom(m), " degrees of freedom, adding constraints...")

# constraint on mole fractions
#m.fs.ref_on.reformer.frac_constraint = pyo.Constraint(
#    expr = sum(m.fs.ref_on.reformer.inlet.mole_frac_comp[0, i] for i in component_list) == \
#        sum(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, i] for i in component_list))

# solver setup
solver = pyo.SolverFactory("ipopt")
solver.options = {
        "tol": 1e-8,
        'bound_push': 1e-23,
        "max_iter": 40,
        # "halt_on_ampl_error": "yes",
    }

print("The model has ",degrees_of_freedom(m), " degrees of freedom, fixing specific case values...")

# case-specific variable settings
if case_number == '1':
    m.fs.ref_on.reformer.outlet.temperature.fix(1060.93)  # K, equal to 1450 F
    print("The model has ",degrees_of_freedom(m), " degrees of freedom, initializing...")
    m.fs.ref_on.reformer.initialize()
    print("The model has ",degrees_of_freedom(m), " degrees of freedom, solving...")
    solver.solve(m, tee=True)
    
    outlet_list = dict([
     ('heat_duty', pyo.value(m.fs.ref_on.reformer.heat_duty[0])),
     ('H2', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "H2"])),
     ('CO2', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "CO2"])),
     ('CO', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "CO"])),
     ('H2O', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "H2O"])),
     ('N2', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "N2"])),
     ('Tout', pyo.value(m.fs.ref_on.reformer.outlet.temperature[0])),
     ('Fout', pyo.value(m.fs.ref_on.reformer.outlet.flow_mol[0]))
 ])
    print() # extra line for readability
    print("The heat duty is ", outlet_list["heat_duty"], " kJ/s")
    print("The outlet flow is ", outlet_list["Fout"], " kmol/s")
    
elif case_number == '2':
    m.fs.ref_on.reformer.heat_duty[0].fix(0)
    m.fs.ref_on.reformer.outlet.temperature.unfix()
    print("The model has ",degrees_of_freedom(m), " degrees of freedom, initializing...")
    m.fs.ref_on.reformer.initialize()
    print("The model has ",degrees_of_freedom(m), " degrees of freedom, solving...")
    solver.solve(m, tee=True)
    
outlet_list = dict([
     ('heat_duty', pyo.value(m.fs.ref_on.reformer.heat_duty[0])),
     ('H2', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "H2"])),
     ('CO2', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "CO2"])),
     ('CO', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "CO"])),
     ('H2O', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "H2O"])),
     ('N2', pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "N2"])),
     ('Tout', pyo.value(m.fs.ref_on.reformer.outlet.temperature[0])),
     ('Fout', pyo.value(m.fs.ref_on.reformer.outlet.flow_mol[0]))
 ])
print() # extra line for readability
print("The outlet flow is ", outlet_list["Fout"], " kmol/s")
print("The outlet temperature is ", outlet_list["Tout"], " K")
print("The outlet H2 mole fraction is ", outlet_list["H2"], "H2")
print("The outlet CO2 mole fraction is ", outlet_list["CO2"], "CO2")
print("The outlet CO mole fraction is ", outlet_list["CO"], "CO")
print("The outlet H2O mole fraction is ", outlet_list["H2O"], "H2O")
print("The outlet N2 mole fraction is ", outlet_list["N2"], "N2")


#m.fs.ref_on.reformer.initialize()
#solver.solve(m, tee=True)
# removed 5/25/2021 after discussion with Alex
#  m.fs.ref_on.reformer.heat_duty[0].fix(0)
#  m.fs.ref_on.reformer.outlet.temperature.unfix()
#solver.solve(m, tee=True)
#sys.path.append(os.path.abspath("C:/Users/MZamarripa/Documents/IDAES/ARPA_E/Differentiate/code/5_9_2021/superstructure"))

#ms.from_json(m, fname='initfunc.json')
#m.fs.ref_on.reformer.inlet.flow_mol[0].fix(x["Fin"])
#m.fs.ref_on.reformer.inlet.temperature[0].fix(x["Tin"])

#solver.solve(m, tee=True)
#f["heat_duty"] = (pyo.value(m.fs.ref_on.reformer.heat_duty[0]))
#f["H2"] = (pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "H2"]))
#f["CO2"] = (pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "CO2"]))
#f["CO"] = (pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "CO"]))
#f["H2O"] = (pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "H2O"]))
#f["N2"] = (pyo.value(m.fs.ref_on.reformer.outlet.mole_frac_comp[0, "N2"]))
#f["Tout"] = pyo.value(m.fs.ref_on.reformer.outlet.temperature[0])
#f["Fout"] = pyo.value(m.fs.ref_on.reformer.outlet.flow_mol[0])
