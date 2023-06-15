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
Subtask: General Unified Surrogate Object - Auto-reformer Example
Author: B. Paul
"""
# Import statements
import numpy as np
import pandas as pd

import sm_plotter as splot

from idaes.surrogate.pysmo import sampling as sp

from build_Alamopy import build_Alamopy as abuild
from build_Keras import auto_Keras as autokbuild
from build_PySMO import build_PySMO_poly as ppolybuild
from build_PySMO import build_PySMO_rbf as prbfbuild
from build_PySMO import build_PySMO_krig as pkrigbuild

# Import Auto-reformer training data and generate test data samples
np.set_printoptions(precision=6, suppress=True)

data = pd.read_csv(r'reformer-data.csv')
train_data = data.iloc[::28, :]

xdata = train_data.iloc[:, :2]
zdata = train_data.iloc[:, 2:]
xlabels = xdata.columns
zlabels = zdata.columns

sample = input("Enter '1' to generate plots from sampled xtest data: ")

if sample == '1':
    bounds_min = xdata.min(axis=0)
    bounds_max = xdata.max(axis=0)
    bounds_list = [list(bounds_min), list(bounds_max)]
    space_init = sp.LatinHypercubeSampling(bounds_list,
                                           sampling_type='creation',
                                           number_of_samples=100)
    xtest = np.array(space_init.sample_points())
else:
    xtest = np.array(xdata)

# Alamopy options
aoptlabels = ['constant', 'linfcns', 'multi2power', 'monomialpower',
              'ratiopower', 'maxterms', 'filename', 'overwrite_files']
aoptvals = [True, True, (1, 2), (2, 3, 4, 5, 6), (1, 2), [10] * len(zlabels),
            'alamo_run', True]
alamopy_options = dict(zip(aoptlabels, aoptvals))

# PySMO Poly options
ppolylabels = ['maximum_polynomial_order', 'multinomials', 'training_split',
               'number_of_crossvalidations', 'overwrite']
ppolyvals = [6, 1, 0.8, 10, True]
ppoly_options = dict(zip(ppolylabels, ppolyvals))

# PySMO RBF options
prbflabels = ['basis_function', 'solution_method', 'regularization',
              'fname', 'overwrite']
prbfvals = [('gaussian'), 'pyomo', True, None, True]
prbf_options = dict(zip(prbflabels, prbfvals))

# PySMO Krig options
pkriglabels = ['numerical_gradients', 'regularization', 'fname', 'overwrite']
pkrigvals = [True, True, None, True]
pkrig_options = dict(zip(pkriglabels, pkrigvals))

# Keras options
koptlabels = ['act', 'opt', 'hid', 'neu', 'loss', 'metrics']
koptvals = [['relu', 'sigmoid'], ['RMSprop', 'Adam'], [1], [300], ['mse'],
            ['mae', 'mse']]
keras_options = dict(zip(koptlabels, koptvals))

# Calling surrogate methods to obtains models (and zfit data)

print('The input variables are ', str(xlabels))

SM = input("Enter '1' to fit with ALAMOpy, '2' to fit with PySMO, and any "
           "other character to fit with Keras: ")

if SM == '1':
    print("Fitting with ALAMOpy, calling, method...")

    print('Generating surrogates for outputs ', str(zlabels))

    xmin, xmax = [0.1, 0.8], [0.8, 1.2]
    res, zfit = abuild(np.transpose(np.array(xdata)),
                       np.transpose(np.array(zdata)),
                       xtest, xmin=xmin, xmax=xmax,
                       options=alamopy_options)
    model = res['Model']
    name = 'alamopy'

elif SM == '2':
    print("Fitting with PySMO, calling method...")
    pysmo = input("Enter '1' for Poly, '2' for RBF, other for Kriging: ")

    zfit = np.empty(np.shape(zdata))
    model = [None] * len(zlabels)
    for j in range(len(zlabels)):
        data_j = np.array(pd.concat([xdata, zdata.iloc[:, j]], axis=1))
        print('Generating surrogate for output ', str(j+1), ' of ',
              str(len(zlabels)), ': ', str(zlabels[j]))

        if pysmo == '1':
            model[j], zfit[:, j] = ppolybuild(data_j, xtest, ppoly_options)
            name = 'ppoly'
        elif pysmo == '2':
            model[j], zfit[:, j] = prbfbuild(data_j, xtest, prbf_options)
            name = 'prbf'
        else:
            model[j], zfit[:, j] = pkrigbuild(data_j, xtest, pkrig_options)
            name = 'pkrig'
else:
    print("Fitting with Keras: calling method...")

    zfit = np.empty(np.shape(zdata))
    model = [None] * len(zlabels)
    for j in range(len(zlabels)):
        data_j = np.array(pd.concat([xdata, zdata.iloc[:, j]], axis=1))
        print('Generating surrogate for output ', str(j+1), ' of ',
              str(len(zlabels)), ': ', str(zlabels[j]))
        model[j], zfit[:, j] = autokbuild(data_j, xtest, keras_options)
        name = 'keras'

# data for plot inputs
xtest = splot.extractData(xtest)
xdata = splot.extractData(xdata)
zdata = splot.extractData(zdata)
zfit = splot.extractData(zfit)
e = zfit - zdata
rele = np.divide(zfit - zdata, zdata)

# generate surrogate results plots

splot.scatter2D(xdata, zdata, xtest, zfit, xlabels=xlabels, zlabels=zlabels,
                PDF=True, filename=(name + '_scatter2D.pdf'))
splot.scatter3D(xdata, zdata, xtest, zfit, xlabels=xlabels, zlabels=zlabels,
                PDF=True, filename=(name + '_scatter3D.pdf'))
splot.parity(zdata, zfit, zlabels=zlabels, PDF=True,
             filename=(name + '_parity.pdf'))
splot.residual(zdata, e, zlabels=zlabels, PDF=True,
               filename=(name + '_residual.pdf'))
splot.residual(zdata, rele, zlabels=zlabels, elabel='Relative Model error',
               PDF=True, filename=(name + '_relresidual.pdf'))
