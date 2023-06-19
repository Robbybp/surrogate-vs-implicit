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

import os
import pandas as pd
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.alamopy import AlamoTrainer, AlamoSurrogate
from idaes.core.surrogate.plotting.sm_plotter import surrogate_parity

######## FUNCTION TO GENERATE ALAMO SURROGATES ########

def gibbs_to_alamo(file_path, show_surrogates = False, create_plots = False):
    try:
        df = pd.read_csv(file_path) # load data generated from Gibbs_data_generation.py
        if 'Unnamed: 0' in df.columns:
             df = df.drop('Unnamed: 0', axis=1)
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")

    # Recall that the input data to the surrogate models is methane  
    # conversion and inlet temperature and pressure to the reactor. 
    input_data = df.iloc[:, :3]  

    # Recall that the output data of the surrogate models is outlet  
    # temperature and flow rate, heat duty, and molar compositions of
    # H2, H2O, CO, CO2, CH4
    output_data = df.iloc[:, 4:]    
    
    # Split training and validation data
    input_labels = input_data.columns
    output_labels = output_data.columns

    n_data = df[input_labels[0]].size
    data_training, data_validation = split_training_validation(df, 0.8, seed=n_data)

    trainer = AlamoTrainer(
        input_labels=input_labels,
        output_labels=output_labels,
        training_dataframe=data_training,
    )

    trainer.config.constant = True
    trainer.config.linfcns = True
    trainer.config.monomialpower = [2] # keep surrogate models as simple as possible, as long as parity plots show good correlations

    _, alm_surr, _ = trainer.train_surrogate()

    alm_surr.save_to_file("alamo_surrogate.json", overwrite=True)

    surrogate_expressions = trainer._results["Model"]

    if show_surrogates == True:
        for i in surrogate_expressions:
            print(surrogate_expressions[i])

    input_labels = trainer._input_labels
    output_labels = trainer._output_labels

    bounds = df[['Tin','Pin','Conversion']].agg(['min', 'max']).T
    input_bounds = {index: (row['min'], row['max']) for index, row in bounds.iterrows()}

    alm_surr = AlamoSurrogate(
        surrogate_expressions, input_labels, output_labels, input_bounds
    )

    if create_plots == True:
        surrogate_parity(alm_surr, data_training, filename='parity_train.pdf')
        surrogate_parity(alm_surr, data_validation, filename='parity_val.pdf')

    file_dir = os.path.dirname(__file__)
    fname_surrogates = os.path.join(file_dir, 'alamo_surrogate.json')
    fname_train_plot = os.path.join(file_dir, 'parity_train.pdf')
    fname_val_plot = os.path.join(file_dir, 'parity_val.pdf')   
    
    return fname_surrogates, fname_train_plot, fname_val_plot
