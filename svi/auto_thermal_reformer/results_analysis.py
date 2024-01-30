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
#  __________________________________________________________________________

######## IMPORT PACKAGES ########
import pyomo.environ as pyo
import pandas as pd
import numpy as np
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

from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from svi.auto_thermal_reformer.fullspace_flowsheet import make_simulation_model

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns

###### FUNCTION TO VALIDATE ALAMO AND NEURAL NETWORK RESULTS ######

def validate_alamo_or_nn(fname = "alamo_experiment.csv"):
    df = pd.read_csv(fname)
    
    # Parse the data needed for validation
    validation_inputs = df[['X', 'P', 'Steam', 'Bypass Frac', 'CH4 Feed']]
    
    # Create an empty dataframe to store the objective value from the validation process.
    # df_val_res stores the result of the full space simulation that takes surrogate DOF
    # as inputs.

    df_val_res = {'X':[], 'P':[], 'Objective':[]} 

    for index, row in validation_inputs.iterrows():
        X = row['X']
        P = row['P']
        Steam = row['Steam']
        Bypass_Frac = row['Bypass Frac']
        CH4_Feed = row['CH4 Feed']

        m = make_simulation_model(X,P,Steam,Bypass_Frac,CH4_Feed)
        try:
            calc_var_kwds = dict(eps=1e-7)
            solve_kwds = dict(tee=True)
            solver = pyo.SolverFactory("ipopt")
            solve_strongly_connected_components(
                m,
                solver=solver,
                calc_var_kwds=calc_var_kwds,
                solve_kwds=solve_kwds,
            )
            
            solver.solve(m, tee=True)
            df_val_res['X'].append(X)
            df_val_res['P'].append(P)
            df_val_res['Objective'].append(value(m.fs.product.mole_frac_comp[0,'H2']))
        except ValueError:
            df_val_res['X'].append(X)
            df_val_res['P'].append(P)
            df_val_res['Objective'].append(999)

    df_val_res = pd.DataFrame(df_val_res)

    if 'alamo' in fname:
        df_val_res.to_csv("alamo_validation.csv")
    elif 'nn' in fname:
        df_val_res.to_csv("nn_validation.csv")

    return df_val_res

def calculate_error_in_objectives(fname_1 = "implicit_experiment.csv",
                                  fname_2 = "alamo_validation.csv",
                                  fname_3 = "alamo_experiment.csv"):
    
    df_implicit = pd.read_csv(fname_1)
    df_val_res = pd.read_csv(fname_2)
    df_surrogate = pd.read_csv(fname_3)
    
    list_of_optimalsurr_results = list()
    list_of_optimalimp_results = list()
    intersected_list = list()
    list_of_invalid_ind = list()

    for index, row in df_surrogate.iterrows():
        if row['Termination'] == "optimal":
            list_of_optimalsurr_results.append(index)
    
    for index, row in df_implicit.iterrows():
        if row['Termination'] == 'optimal':
            list_of_optimalimp_results.append(index)

    for index, row in df_val_res.iterrows():
        if row['Objective'] == 999:
            list_of_invalid_ind.append(index)

    for e in list_of_optimalsurr_results:
        if e in list_of_optimalimp_results:
            intersected_list.append(e)

    for e in list_of_invalid_ind:
        if e in intersected_list:
            intersected_list.remove(e)

    df_implicit_filtered = df_implicit.iloc[intersected_list]
    df_val_res_filtered = df_val_res.iloc[intersected_list]
    
    errors  = (abs(df_implicit_filtered['Objective'] - df_val_res_filtered['Objective']) / df_implicit_filtered['Objective']).tolist()
    average_error = sum(errors) * 100 / len(df_val_res_filtered.index)

    if 'alamo' in fname_2:
        print("The average error in ALAMO is:",average_error, "%.")
    if 'nn' in fname_2:
        print("The average error in NN is:", average_error, "%.")

def plot_convergence_reliability(fname = 'implicit_experiment.csv'):
    data = pd.read_csv(fname)
    
    if 'full' in fname:
        name = 'Full Space'
    if 'implicit' in fname:
        name = 'Implicit Function'
    if 'alamo' in fname:
        name = 'ALAMO Surrogate'
    if 'nn' in fname:
        name = 'Neural Network Surrogate'

    condition = data['Termination'] == 'optimal'
    data.loc[condition,'Termination'] = 1
    data.loc[~condition,'Termination'] = 0
    
    data = data.drop('Unnamed: 0', axis = 1)
    data['Termination'] = data['Termination'].astype(float)

    data['P'] = data['P'] / 1e6
    data['P'] = data['P'].round(2)

    df_for_plotting = data[['X','P','Termination']]
    pivoting = np.round(pd.pivot_table(df_for_plotting,
                                       values = 'Termination',
                                       index = 'X',
                                       columns = 'P',
                                       aggfunc = 'first'),2)

    fig = plt.figure(figsize = (7,7))
    ax = sns.heatmap(pivoting, linewidths = 1, linecolor = 'darkgray', 
                     linewidth = 1, cbar = False, cmap = ListedColormap(['black','bisque']))

    plt.title(name+' Formulation', fontsize = 18.5)
    plt.xlabel("Pressure (MPa)", fontsize = 18.5)
    plt.ylabel("Conversion", fontsize = 18.5)

    original_labels_conversion = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
    new_labels_conversion = [0.91, 0.93, 0.95, 0.97]
    labels_conversion_plotting = [label if label in new_labels_conversion else "" for label in original_labels_conversion]
    ax.set_yticklabels(labels_conversion_plotting, fontsize = 16.5)

    original_labels_pressure = [1.45, 1.52, 1.59, 1.66, 1.73, 1.80, 1.87, 1.94]
    new_labels_pressure = [1.52, 1.66, 1.80, 1.94]
    labels_pressure_plotting = [label if label in new_labels_pressure else "" for label in original_labels_pressure]
    ax.set_xticklabels(labels_pressure_plotting, fontsize = 16.5)

    legend_handles = [Patch(color = 'bisque', label = 'Successful'),
                      Patch(color = 'black', label = 'Unsuccessful')]

    plt.legend(handles = legend_handles, ncol = 1, fontsize = 16, handlelength = .8, bbox_to_anchor = (1.05,1),
               loc = 'upper left', borderaxespad = 0)

    plt.gca().invert_yaxis()
    fig.savefig(name + ' Plot', bbox_inches='tight')

if __name__ == "__main__":
    validate_alamo_or_nn(fname = "nn_experiment.csv")
    calculate_error_in_objectives(fname_1 = "implicit_experiment.csv", fname_2 = "nn_validation.csv", fname_3 = "nn_experiment.csv")
    plot_convergence_reliability(fname = 'nn_experiment.csv')
