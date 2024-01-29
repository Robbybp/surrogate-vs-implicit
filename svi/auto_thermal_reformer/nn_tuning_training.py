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
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.sampling.scaling import OffsetScaler
from idaes.core.surrogate.keras_surrogate import (
    KerasSurrogate,
    save_keras_json_hd5,
    load_keras_json_hd5,
)

from idaes.core.surrogate.alamopy import AlamoSurrogate
from idaes.core.surrogate.keras_surrogate import KerasSurrogate
from sklearn.metrics import r2_score

np.random.seed(46)
random.seed(1342)
tf.random.set_seed(62)
tf.keras.backend.set_floatx('float64')

# This function is to tune and train the neural network

def create_nn_and_compare(fname = "data_atr.csv", 
                          tune = False, 
                          alamo_surr = 'alamo_surrogate_atr.json', 
                          compare_r2 = True):

    csv_data = pd.read_csv(fname) 
    if 'Unnamed: 0' in csv_data.columns:
        csv_data = csv_data.drop('Unnamed: 0', axis=1)

    # Drop columns that contain values less than 1e-14 so that the NN can train properly
    csv_data = csv_data.drop(['C4H10','O2'], axis=1)

    data = csv_data

    input_data = data.iloc[:, :4]
    output_data = data.iloc[:, 4:]

    input_labels = input_data.columns
    output_labels = output_data.columns

    n_data = data[input_labels[0]].size
    data_training, data_validation = split_training_validation(
        data, 0.8, seed=n_data
    ) 

    # Save the data_validation file
    data_validation.to_csv("data_validation.csv")

    # Define the parameter values to try
    activations = ["sigmoid", "tanh"]
    optimizers = ["Adam"]
    n_hidden_layers_values = np.arange(2,5,1).tolist()
    n_nodes_per_layer_values = np.arange(20,31,1).tolist()
    epochs = 500

    loss, metrics = "mse", ["mae", "mse"]
    
    best_val_loss = float("inf")  # Variable to store the best validation loss
    best_model = None  # Variable to store the best model
    
    # Create data objects for training using scalar normalization
    n_inputs = len(input_labels)
    n_outputs = len(output_labels)
    x = input_data
    y = output_data
    
    input_scaler = None
    output_scaler = None
    input_scaler = OffsetScaler.create_normalizing_scaler(x)
    output_scaler = OffsetScaler.create_normalizing_scaler(y)
    x = input_scaler.scale(x)
    y = output_scaler.scale(y)
    x = x.to_numpy()
    y = y.to_numpy()

    bounds = csv_data[['Fin_CH4','Tin_CH4','Fin_H2O','Conversion']].agg(['min', 'max']).T
    input_bounds = {index: (row['min'], row['max']) for index, row in bounds.iterrows()}

    if tune == True:
        # Iterate over the parameter combinations
        for activation in activations:
            for optimizer in optimizers:
                for n_hidden_layers in n_hidden_layers_values:
                    for n_nodes_per_layer in n_nodes_per_layer_values:
                        model = tf.keras.Sequential()
                        model.add(tf.keras.layers.Dense(units=n_nodes_per_layer, input_dim=n_inputs, activation=activation))
                        for _ in range(1, n_hidden_layers):
                            model.add(tf.keras.layers.Dense(units=n_nodes_per_layer, activation=activation))
                        model.add(tf.keras.layers.Dense(units=n_outputs))
        
                        # Train surrogate
                        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
                        mcp_save = tf.keras.callbacks.ModelCheckpoint(".mdl_wts.hdf5", save_best_only=True, monitor="val_loss", mode="min")
                        history = model.fit(x=x, y=y, validation_split=0.2, verbose=1, epochs=epochs, callbacks=[mcp_save])
        
                        val_loss = history.history["val_loss"][-1]  # Get the final validation loss
        
                        # Check if this model has the best validation loss so far
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model = model
        
                            keras_surrogate = KerasSurrogate(
                                best_model,
                                input_labels=list(input_labels),
                                output_labels=list(output_labels),
                                input_bounds=input_bounds,
                                input_scaler=input_scaler,
                                output_scaler=output_scaler,
                            )
        
                            keras_surrogate.save_to_folder("keras_surrogate")
    
    if compare_r2 == True:
        
        # Load the NN
        model = keras.models.load_model('keras_surrogate')

        keras_surrogate = KerasSurrogate(
            model,
            input_labels=list(input_labels),
            output_labels=list(output_labels),
            input_bounds=input_bounds,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
        )

        # Load the ALAMO surrogate
        alamo_surrogate = AlamoSurrogate.load_from_file(alamo_surr)

        # Predict output data with NN and ALAMO
        predicted_output_data_nn = KerasSurrogate.evaluate_surrogate(keras_surrogate, data_validation.iloc[:,:4])
        predicted_output_data_alamo = AlamoSurrogate.evaluate_surrogate(alamo_surrogate, data_validation.iloc[:,:4])

        # Calculate R2 for NN and ALAMO parity plots
        r2_nn = list()
        r2_alamo = list()

        for component in output_labels:
            if component not in ['C4H10','O2']:
                y_pred_nn = predicted_output_data_nn[component].values
                y_true_nn = data_validation.iloc[:,4:][component].values
                r2_nn.append(r2_score(y_true_nn,y_pred_nn))
                y_pred_alamo = predicted_output_data_alamo[component].values
                y_true_alamo = data_validation.iloc[:,4:][component].values
                r2_alamo.append(r2_score(y_true_alamo,y_pred_alamo))
        
        r2_df = pd.DataFrame({'R2-ALAMO': r2_alamo, 'R2-NN': r2_nn})
        r2_df.to_csv('R2_values.csv')

if __name__ == "__main__":
    create_nn_and_compare(fname = "data_atr.csv", 
                          tune = False, 
                          alamo_surr = 'alamo_surrogate_atr.json', 
                          compare_r2 = True)




















