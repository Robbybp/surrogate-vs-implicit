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
Subtask: General Unified Surrogate Object - ALAMOpy Build Method
Author: B. Paul
"""
import numpy as np
from idaes.surrogate.alamopy_new import AlamoTrainer, AlamoObject


def build_Alamopy(xdata, zdata, xtest, xmin=None, xmax=None, xlabels=None,
                  zlabels=None, options=None):
    """
    Calls Alamopy wrapper to train a single model from given input settings.
    """
    numins = np.shape(xdata)[0]
    numouts = np.shape(zdata)[0]

    trainer = AlamoTrainer()
    trainer._n_inputs = numins
    trainer._n_outputs = numouts
    trainer._rdata_in = xdata
    trainer._rdata_out = zdata

    if xmin is None:
        xmin = list(xdata.min(axis=0))
    if xmax is None:
        xmax = list(xdata.max(axis=0))
    if xlabels is None:
        xlabels = ['x' + str(i+1) for i in range(numins)]
    if zlabels is None:
        zlabels = ['z' + str(j+1) for j in range(numouts)]

    trainer._input_min = [xmin[i] for i in range(len(xmin))]
    trainer._input_max = [xmax[i] for i in range(len(xmax))]
    trainer._input_labels = [xlabels[i] for i in range(len(xlabels))]
    trainer._output_labels = [zlabels[i] for i in range(len(zlabels))]

    print(trainer._rdata_in)
    print(trainer._input_min)
    print(trainer._input_max)
    print(trainer._input_labels)
    print(trainer._output_labels)
    if options is not None:
        for entry in options:
            setattr(trainer.config, entry, options[entry])

    trainer.train_surrogate()

    surrogate = trainer._results['Model']
    input_labels = trainer._input_labels
    output_labels = trainer._output_labels
    input_bounds = {xlabels[i]: (xmin[i], xmax[i])
                    for i in range(len(xlabels))}

    alm_surr = AlamoObject(surrogate, input_labels,
                           output_labels, input_bounds)
    zfit = np.transpose(alm_surr.evaluate_surrogate(np.transpose(xtest)))

    return trainer._results, zfit
