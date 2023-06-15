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
Subtask: General Unified Surrogate Object - Alamopy Example w/ Parity Plot
Author: B. Paul
"""
import numpy as np
import pandas as pd
import sm_plotter as splot
from build_Alamopy import build_Alamopy as abuild
from idaes.surrogate.pysmo import sampling as sp

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

print("Fitting with ALAMOpy, calling, method...")

print('Generating surrogates for outputs ', str(zlabels))

xmin, xmax = [0.1, 0.8], [0.8, 1.2]
res, zfit = abuild(np.transpose(np.array(xdata)),
                   np.transpose(np.array(zdata)),
                   xtest, xmin=xmin, xmax=xmax,
                   options=alamopy_options)
model = res['Model']
name = 'alamopy_example'

# Ensure data is in correct format for visualization utilities (2D arrays)
xdata = splot.extractData(xdata)
zdata = splot.extractData(zdata)
zfit = splot.extractData(zfit)

splot.parity(zdata, zfit, zlabels=zlabels, PDF=True,
             filename=(name + '_parity.pdf'))
