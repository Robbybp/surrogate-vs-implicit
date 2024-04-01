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

import os
import tensorflow as tf
from tensorflow import keras

dirname = os.path.dirname(__file__)
data_dir = os.path.join(dirname, "data")
basename_h = "keras_surrogate_high_rel"
keras_surrogate_h = os.path.join(data_dir, basename_h)

modelh = keras.models.load_model(keras_surrogate_h)

print("Layers:")
for layer in modelh.layers:
    print(layer.name, layer.output_shape, layer.activation)

print("Model Summary:")
print(modelh.summary())

