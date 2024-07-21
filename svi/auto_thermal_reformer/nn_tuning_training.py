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
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.sampling.scaling import OffsetScaler
from idaes.core.surrogate.keras_surrogate import KerasSurrogate
import svi.auto_thermal_reformer.config as config
import time

np.random.seed(46)
random.seed(1342)
tf.random.set_seed(62)
tf.keras.backend.set_floatx('float64')

DEFAULT_DATA_FILE = "data_atr.csv"
DEFAULT_SURR_NAME = "keras_surrogate_high_rel"


def gibbs_to_nn(
    data, 
    surrogate_fname,
    tune,
    activation,
    layers,
    neurons,
):

    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)

    # Drop columns that contain values less than 1e-16 so that the NN can train properly
    data = data.drop(['C4H10','O2'], axis=1)

    input_data = data.iloc[:, :4]
    output_data = data.iloc[:, 4:]

    input_labels = input_data.columns
    output_labels = output_data.columns

    n_data = data[input_labels[0]].size
    
    data_training, data_validation = split_training_validation(data, 0.8, seed=n_data) 
    
    optimizers = ["Adam"]
    epochs = 400 

    # Define the parameter values to try
    activations = ["sigmoid", "tanh"]
    n_hidden_layers_values = np.arange(2,5,1).tolist()
    n_nodes_per_layer_values = np.arange(20,33,1).tolist()
    
    if not tune:
        activations = [activation]
        n_hidden_layers_values = [layers]
        n_nodes_per_layer_values = [neurons]
    
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

    bounds = data[['Fin_CH4','Tin_CH4','Fin_H2O','Conversion']].agg(['min', 'max']).T
    input_bounds = {index: (row['min'], row['max']) for index, row in bounds.iterrows()}

    t0 = time.time()
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
                        
                        keras_surrogate.save_to_folder(surrogate_fname)
                        print(f"Saved NN surrogate model to {surrogate_fname}")
                        print(
                            f"Parameters of model that was just saved:"
                            f"activation={activation},"
                            f"optimizer={optimizer},"
                            f"n_hidden_layers={n_hidden_layers},"
                            f"n_nodes_per_payer={n_nodes_per_layer},"
                        )

    t1 = time.time()
    total_time = t1 - t0
    print("Total time: ", total_time)


def main():

    argparser = config.get_argparser()

    argparser.add_argument(
        "fpath", help="Base file name for training the neural network",
    )

    argparser.add_argument(
        "--surrogate_fname",
        default=DEFAULT_SURR_NAME,
        help="File name for the neural network",
    )

    argparser.add_argument(
        "--tune",
        action="store_true",
        help="If not set, you just train with tanh, 3 hidden layers, 32 neurons.",
    )

    args = argparser.parse_args()

    surrogate_fname = os.path.join(args.data_dir, args.surrogate_fname)

    data = pd.read_csv(args.fpath)

    gibbs_to_nn(
        data,
        surrogate_fname=surrogate_fname,
        tune=args.tune,
        activation="tanh",
        layers=3,
        neurons=32,
    )


if __name__ == "__main__":
    main()
