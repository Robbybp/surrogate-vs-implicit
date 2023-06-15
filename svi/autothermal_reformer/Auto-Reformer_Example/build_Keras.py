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
Subtask: General Unified Surrogate Object - Keras Build Method
Author: B. Paul
"""
import os
import numpy as np
import random as rn
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from math import sqrt


os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(46)
rn.seed(1342)
tf.random.set_seed(62)


def build_Keras(data, xtest, n_hidden=1, n_neurons=150, activation='relu',
                optimizer='SGD', loss='mse', metrics=['mae', 'mse']):
    """
    Generates and trains a single Keras model from given input settings.
    """
    xdata = data[:, :-1]
    zdata = data[:, -1]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=np.shape(xdata)[1]))

    i = 1
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons/i, activation=activation))
        if n_neurons > 20:
            model.add(tf.keras.layers.Dropout(rate=0.2))
            i = i + 2

    model.add(tf.keras.layers.Dense(1, activation=activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit(xdata, zdata, epochs=500, verbose=0)
    zfit = model.predict(xtest)

    return model, zfit


def auto_Keras(data, xtest, options=None):
    """
    Automatically runs Keras training for a range of settings and selects
    the best fit with the minimum root mean squared error.
    """
    zdata = data[:, -1]

    # procedurally iterate through all possible regression runs
    min_er = 1E9  # higher than any possible error value (to check improvement)
    for act in options['act']:
        for opt in options['opt']:
            for hid in options['hid']:
                for neu in options['neu']:
                    print("Trying", act, "with", opt, "with", hid,
                          "hidden layers and", neu, "neurons")
                    model, zfit = build_Keras(data, xtest,
                                              activation=act,
                                              optimizer=opt,
                                              n_hidden=hid,
                                              n_neurons=neu,
                                              loss=options['loss'],
                                              metrics=options['metrics'])
                    model_error = sqrt(mean_squared_error(zdata, zfit))

                    if model_error <= min_er:
                        min_er = model_error
                        final_model = model  # better model found, save it
                        final_fit = zfit[:, 0]  # save model output values
                        best = [act, opt, hid, neu]  # save best options

    # identify the best run and return selected model and fit values

    print("Best fit obtained with", best[0], "and", best[1], "with",
          best[2], "hidden layers and", best[3], "neurons")

    return final_model, final_fit
