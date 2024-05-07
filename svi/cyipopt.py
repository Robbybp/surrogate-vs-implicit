import sys
import io
import logging
import os
from pyomo.common.tee import redirect_fd, TeeStream
from pyomo.common.timing import HierarchicalTimer, TicTocTimer
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    PyomoCyIpoptSolver,
    _cyipopt_status_enum,
    _ipopt_term_cond,
)
from pyomo.core.base import Block, Objective, minimize
from pyomo.opt import SolverStatus, SolverResults, TerminationCondition, ProblemSense
from pyomo.opt.results.solution import Solution
import numpy as np
from scipy import sparse

pyomo_nlp = attempt_import("pyomo.contrib.pynumero.interfaces.pyomo_nlp")[0]
pyomo_grey_box = attempt_import("pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp")[
    0
]
egb = attempt_import("pyomo.contrib.pynumero.interfaces.external_grey_box")[0]

# Defer this import so that importing this module (PyomoCyIpoptSolver in
# particular) does not rely on an attempted cyipopt import.
cyipopt_interface, _ = attempt_import(
    "pyomo.contrib.pynumero.interfaces.cyipopt_interface"
)

logger = logging.getLogger(__name__)


class TimedCyIpoptNLP(CyIpoptNLP):

    def __init__(self, nlp, **kwds):
        timer = kwds.pop("timer", None)
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer

        super().__init__(nlp, **kwds)

    def solve(self, x, lagrange=None, zl=None, zu=None):
        self._timer.start("solve")
        data = super().solve(x, lagrange=lagrange, zl=zl, zu=zu)
        self._timer.stop("solve")
        return data

    def objective(self, x):
        self._timer.start("function")
        try:
            data = super().objective(x)
        finally:
            self._timer.stop("function")
        return data

    def gradient(self, x):
        self._timer.start("jacobian")
        try:
            data = super().gradient(x)
        finally:
            self._timer.stop("jacobian")
        return data

    def constraints(self, x):
        self._timer.start("function")
        try:
            data = super().constraints(x)
        finally:
            self._timer.stop("function")
        return data

    def jacobian(self, x):
        self._timer.start("jacobian")
        try:
            data = super().jacobian(x)
        finally:
            self._timer.stop("jacobian")
        return data

    def hessian(self, x, y, obj_factor):
        self._timer.start("hessian")
        try:
            data = super().hessian(x, y, obj_factor)
        finally:
            self._timer.stop("hessian")
        return data

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        self._timer.start("intermediate-cb")
        try:
            if self._intermediate_callback is not None:
                if hasattr(self, "_use_13arg_callback") and self._use_13arg_callback:
                    # This is the callback signature expected as of Pyomo vTBD
                    ret = self._intermediate_callback(
                        self._nlp,
                        self,
                        alg_mod,
                        iter_count,
                        obj_value,
                        inf_pr,
                        inf_du,
                        mu,
                        d_norm,
                        regularization_size,
                        alpha_du,
                        alpha_pr,
                        ls_trials,
                    )
                else:
                    # This is the callback signature expected pre-Pyomo vTBD and
                    # is supported for backwards compatibility.
                    ret = self._intermediate_callback(
                        self._nlp,
                        alg_mod,
                        iter_count,
                        obj_value,
                        inf_pr,
                        inf_du,
                        mu,
                        d_norm,
                        regularization_size,
                        alpha_du,
                        alpha_pr,
                        ls_trials,
                    )
            else:
                ret = True
        finally:
            self._timer.stop("intermediate-cb")
        return ret


class TimedPyomoCyIpoptSolver(PyomoCyIpoptSolver):

    def solve(self, model, **kwds):
        htimer = kwds.pop("timer", None)
        config = self.config(kwds, preserve_implicit=True)

        if not isinstance(model, Block):
            raise ValueError(
                "PyomoCyIpoptSolver.solve(model): model must be a Pyomo Block"
            )

        # If this is a Pyomo model / block, then we need to create
        # the appropriate PyomoNLP, then wrap it in a CyIpoptNLP
        grey_box_blocks = list(
            model.component_data_objects(egb.ExternalGreyBoxBlock, active=True)
        )
        if grey_box_blocks:
            # nlp = pyomo_nlp.PyomoGreyBoxNLP(model)
            nlp = pyomo_grey_box.PyomoNLPWithGreyBoxBlocks(model)
        else:
            nlp = pyomo_nlp.PyomoNLP(model)

        problem = TimedCyIpoptNLP(
            nlp,
            intermediate_callback=config.intermediate_callback,
            halt_on_evaluation_error=config.halt_on_evaluation_error,
            timer=htimer,
        )
        ng = len(problem.g_lb())
        nx = len(problem.x_lb())
        cyipopt_solver = problem

        # check if we need scaling
        obj_scaling, x_scaling, g_scaling = problem.scaling_factors()
        if any(_ is not None for _ in (obj_scaling, x_scaling, g_scaling)):
            # need to set scaling factors
            if obj_scaling is None:
                obj_scaling = 1.0
            if x_scaling is None:
                x_scaling = np.ones(nx)
            if g_scaling is None:
                g_scaling = np.ones(ng)
            try:
                set_scaling = cyipopt_solver.set_problem_scaling
            except AttributeError:
                # Fall back to pre-1.0.0 API
                set_scaling = cyipopt_solver.setProblemScaling
            set_scaling(obj_scaling, x_scaling, g_scaling)

        # add options
        try:
            add_option = cyipopt_solver.add_option
        except AttributeError:
            # Fall back to pre-1.0.0 API
            add_option = cyipopt_solver.addOption
        for k, v in config.options.items():
            add_option(k, v)

        timer = TicTocTimer()
        try:
            # We preemptively set up the TeeStream, even if we aren't
            # going to use it: the implementation is such that the
            # context manager does nothing (i.e., doesn't start up any
            # processing threads) until after a client accesses
            # STDOUT/STDERR
            with TeeStream(sys.stdout) as _teeStream:
                if config.tee:
                    try:
                        fd = sys.stdout.fileno()
                    except (io.UnsupportedOperation, AttributeError):
                        # If sys,stdout doesn't have a valid fileno,
                        # then create one using the TeeStream
                        fd = _teeStream.STDOUT.fileno()
                else:
                    fd = None
                with redirect_fd(fd=1, output=fd, synchronize=False):
                    x, info = cyipopt_solver.solve(problem.x_init())
            solverStatus = SolverStatus.ok
        except:
            msg = "Exception encountered during cyipopt solve:"
            logger.error(msg, exc_info=sys.exc_info())
            solverStatus = SolverStatus.unknown
            raise

        wall_time = timer.toc(None)

        results = SolverResults()

        if config.load_solutions:
            nlp.set_primals(x)
            nlp.set_duals(info["mult_g"])
            nlp.load_state_into_pyomo(
                bound_multipliers=(info["mult_x_L"], info["mult_x_U"])
            )
        else:
            soln = Solution()
            sm = nlp.symbol_map
            soln.variable.update(
                (sm.getSymbol(i), {'Value': j, 'ipopt_zL_out': zl, 'ipopt_zU_out': zu})
                for i, j, zl, zu in zip(
                    nlp.get_pyomo_variables(), x, info['mult_x_L'], info['mult_x_U']
                )
            )
            soln.constraint.update(
                (sm.getSymbol(i), {'Dual': j})
                for i, j in zip(nlp.get_pyomo_constraints(), info['mult_g'])
            )
            model.solutions.add_symbol_map(sm)
            results._smap_id = id(sm)
            results.solution.insert(soln)

        results.problem.name = model.name
        obj = next(model.component_data_objects(Objective, active=True))
        if obj.sense == minimize:
            results.problem.sense = ProblemSense.minimize
            results.problem.upper_bound = info["obj_val"]
        else:
            results.problem.sense = ProblemSense.maximize
            results.problem.lower_bound = info["obj_val"]
        results.problem.number_of_objectives = 1
        results.problem.number_of_constraints = ng
        results.problem.number_of_variables = nx
        results.problem.number_of_binary_variables = 0
        results.problem.number_of_integer_variables = 0
        results.problem.number_of_continuous_variables = nx
        # TODO: results.problem.number_of_nonzeros

        results.solver.name = "cyipopt"
        results.solver.return_code = info["status"]
        results.solver.message = info["status_msg"]
        results.solver.wallclock_time = wall_time
        status_enum = _cyipopt_status_enum[info["status_msg"]]
        results.solver.termination_condition = _ipopt_term_cond[status_enum]
        results.solver.status = TerminationCondition.to_solver_status(
            results.solver.termination_condition
        )

        problem.close()

        if config.return_nlp:
            return results, nlp

        return results


class Callback:

    def __init__(self):
        self.iterate_data = []

    def __call__(
        self,
        nlp,
        # Don't include this argument, for compatibility with current CyIpopt
        # interface
        #ipopt_problem,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        self.iterate_data.append(
            (
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials,
            )
        )

class ConditioningCallback:

    def __init__(self):
        self.iterate_data = []
        self.condition_numbers = []

    def __call__(
        self,
        nlp,
        # Don't include this argument, for compatibility with current CyIpopt
        # interface
        #ipopt_problem,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        self.iterate_data.append(
            (
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials,
            )
        )
        jac = nlp.evaluate_jacobian()
        cond = np.linalg.cond(jac.toarray())
        self.condition_numbers.append(cond)

def get_gradient_of_lagrangian(
    nlp,
    primal_lb_multipliers,
    primal_ub_multipliers,
):
    # PyNumero NLPs contain constraint multipliers, but does not define a convention.
    # We still need:
    # - primal LB/UB multipliers
    # We should not need slack multipliers (Ipopt should take care of this...)
    grad_obj = nlp.evaluate_grad_objective()

    # There is no way this works. We will probably need to separate equality and
    # inequality multipliers.
    jac = nlp.evaluate_jacobian()
    duals = nlp.get_duals()
    # Each constraint gradient times its multiplier
    conjac_term = jac.transpose().dot(duals)

    grad_lag = (
        - grad_obj
        - conjac_term
        + primal_lb_multipliers
        - primal_ub_multipliers
    )
    return grad_lag
