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
Subtask: General Unified Surrogate Object - PySMO Build Method
Author: B. Paul
"""
#  import sys
#  import os
from idaes.surrogate.pysmo.polynomial_regression import PolynomialRegression
from idaes.surrogate.pysmo.radial_basis_function import RadialBasisFunctions
from idaes.surrogate.pysmo.kriging import KrigingModel


def build_PySMO_poly(data, xtest, options=None):
    """
    Calls Alamopy wrapper to train a single model from given input settings.
    """
    poly_class = PolynomialRegression(original_data_input=data,
                                      regression_data_input=data,
                                      **options)
    features = poly_class.get_feature_vector()
    poly_fit = poly_class.training()
    list_vars = []
    for i in features.keys():
        list_vars.append(features[i])
    model = poly_fit.generate_expression(list_vars)

    zfit = [poly_fit.predict_output(xtest)[w] for w in range(len(xtest))]

    return model, zfit


def build_PySMO_rbf(data, xtest, options=None):
    """
    Calls Alamopy wrapper to train a single model from given input settings.
    """
    rbf_class = RadialBasisFunctions(XY_data=data, **options)
    features = rbf_class.get_feature_vector()
    rbf_fit = rbf_class.training()
    list_vars = []
    for i in features.keys():
        list_vars.append(features[i])
    model = rbf_fit.generate_expression(list_vars)

    zfit = [rbf_fit.predict_output(xtest)[w] for w in range(len(xtest))]

    return model, zfit


def build_PySMO_krig(data, xtest, options=None):
    """
    Calls Alamopy wrapper to train a single model from given input settings.
    """
    krig_class = KrigingModel(XY_data=data, **options)
    features = krig_class.get_feature_vector()
    krig_fit = krig_class.training()
    list_vars = []
    for i in features.keys():
        list_vars.append(features[i])
    model = krig_fit.generate_expression(list_vars)

    zfit = [krig_fit.predict_output(xtest)[w] for w in range(len(xtest))]

    return model, zfit
