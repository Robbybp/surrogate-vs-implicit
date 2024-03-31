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
import pandas as pd
import numpy as np
from pyomo.common.timing import TicTocTimer
import pyomo.environ as pyo
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
from pyomo.common.collections import ComponentSet
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties import GenericParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.models.unit_models import (
    Mixer,
    Heater,
    HeatExchanger,
    PressureChanger,
    GibbsReactor,
    Separator,
    Feed,
    Product,
)
from idaes.core.util.initialization import propagate_state
from idaes.models_extra.power_generation.properties.natural_gas_PR import get_prop
from idaes.core.solvers import get_solver
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import ExternalPyomoModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
    CyIpoptSolverWrapper
)

from svi.external import add_external_function_libraries_to_environment
from svi.auto_thermal_reformer.fullspace_flowsheet import make_optimization_model
from svi.auto_thermal_reformer.implicit_flowsheet import make_implicit
import svi.auto_thermal_reformer.config as config


df = {key: [] for key in config.PARAM_SWEEP_KEYS}


INVALID = None


def main(X,P):
    m = make_optimization_model(X,P)
    add_external_function_libraries_to_environment(m)
    m_implicit = make_implicit(m)
    #solver = pyo.SolverFactory("cyipopt", options = {"tol": 1e-6, "max_iter": 100})
    solver = config.get_optimization_solver()
    timer = TicTocTimer()
    timer.tic("starting timer")
    print(X,P)
    results = solver.solve(m_implicit, tee=True)
    dT = timer.toc("end timer")
    df[list(df.keys())[0]].append(X)
    df[list(df.keys())[1]].append(P)
    df[list(df.keys())[2]].append(results.solver.termination_condition)
    df[list(df.keys())[3]].append(dT)
    df[list(df.keys())[4]].append(value(m.fs.product.mole_frac_comp[0,'H2']))
    df[list(df.keys())[5]].append(value(m.fs.reformer_mix.steam_inlet.flow_mol[0]))
    df[list(df.keys())[6]].append(value(m.fs.reformer_bypass.split_fraction[0,'bypass_outlet']))
    df[list(df.keys())[7]].append(value(m.fs.feed.outlet.flow_mol[0]))


if __name__ == "__main__":
    argparser = config.get_sweep_argparser()
    argparser.add_argument(
        "--fname",
        default="implicit-sweep.csv",
        help="Basename for parameter sweep results file",
    )
    args = argparser.parse_args()
    xp_samples = config.get_parameter_samples(args)

    #for X in np.arange(0.90,0.98,0.01):
    #    for P in np.arange(1447379,1947379,70000):
    for X, P in xp_samples:
        try:
            main(X,P)
        except PyNumeroEvaluationError as err:
            # Even though, during the Ipopt algorithm, this is caught by
            # CyIpopt, we still hit these errors sometimes. My best guess
            # is that, if we converge with restoration/line search failures
            # due to repeated evaluation errors, we will fail when trying to
            # load the solution back into the model (which involves an implicit
            # function solve, which uses scipy.fsolve, which doesn't handle
            # evaluation errors).
            print("WARNING: Error occured:")
            print(err)
            df[list(df.keys())[0]].append(X)
            df[list(df.keys())[1]].append(P)
            df[list(df.keys())[2]].append("Evaluation Error")
            df[list(df.keys())[3]].append(INVALID)
            df[list(df.keys())[4]].append(INVALID)
            df[list(df.keys())[5]].append(INVALID)
            df[list(df.keys())[6]].append(INVALID)
            df[list(df.keys())[7]].append(INVALID)
        except RuntimeError as err:
            # Not sure why we would hit this. Maybe a tolerance-not-met
            # error from scipy.fsolve?
            print("WARNING: Error occured:")
            print(err)
            df[list(df.keys())[0]].append(X)
            df[list(df.keys())[1]].append(P)
            df[list(df.keys())[2]].append("RuntimeError")
            df[list(df.keys())[3]].append(INVALID)
            df[list(df.keys())[4]].append(INVALID)
            df[list(df.keys())[5]].append(INVALID)
            df[list(df.keys())[6]].append(INVALID)
            df[list(df.keys())[7]].append(INVALID)
        except OverflowError as err:
            # Not sure why these happen, but they do...
            print("WARNING: Error occured:")
            print(err)
            df[list(df.keys())[0]].append(X)
            df[list(df.keys())[1]].append(P)
            df[list(df.keys())[2]].append("OverflowError")
            df[list(df.keys())[3]].append(INVALID)
            df[list(df.keys())[4]].append(INVALID)
            df[list(df.keys())[5]].append(INVALID)
            df[list(df.keys())[6]].append(INVALID)
            df[list(df.keys())[7]].append(INVALID)

    df = pd.DataFrame(df)
    fpath = os.path.join(args.data_dir, args.fname)
    df.to_csv(fpath)
