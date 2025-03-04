from pyomo.common.collections import ComponentSet
import numpy as np
import scipy.sparse as sps


def get_reduced_hessian(nlp, dof_vars, lbmult, ubmult):
    dofvarset = ComponentSet(dof_vars)
    other_vars = [var for var in nlp.get_pyomo_variables() if var not in dofvarset]
    eqcons = nlp.get_pyomo_equality_constraints()
    dof_jac = nlp.extract_submatrix_jacobian(dof_vars, eqcons)
    other_jac = nlp.extract_submatrix_jacobian(other_vars, eqcons)
    ndof = len(dof_vars)
    dof_nullbasis = np.identity(ndof)
    lu = sps.linalg.splu(other_jac)
    other_nullbasis = - lu.solve(dof_jac.toarray())
    nullbasis = np.vstack((dof_nullbasis, other_nullbasis))

    varorder = dof_vars + other_vars
    varindices = nlp.get_primal_indices(varorder)

    ineqcons = nlp.get_pyomo_inequality_constraints()
    ineq_jac = nlp.extract_submatrix_jacobian(varorder, ineqcons)
    ineq_val = np.diag(nlp.evaluate_ineq_constraints())
    ineq_val_inv = np.diag(1 / nlp.evaluate_ineq_constraints())
    ineq_duals = np.diag(nlp.get_duals_ineq())
    # This term is in the right order
    ineq_term = ineq_jac.transpose() @ ineq_val_inv @ ineq_duals @ ineq_jac

    lbmult = np.diag(lbmult)
    ubmult = np.diag(ubmult)
    # TODO: Have to make sure values and multipliers are in the right order
    primal_val = nlp.get_primals()[varindices]
    primal_lb = nlp.primals_lb()[varindices]
    primal_ub = nlp.primals_ub()[varindices]
    # Have some variables equal to their (lower) bounds. These must not participate
    # in the optimization.
    active_lbs = np.where(primal_lb == primal_val)[0]
    active_ubs = np.where(primal_ub == primal_val)[0]
    primal_lb[active_lbs] -= 1e-8
    primal_ub[active_ubs] += 1e-8
    primal_ubslack_inv = np.diag(1 / (primal_ub - primal_val))
    primal_lbslack_inv = np.diag(1 / (primal_val - primal_lb))
    # TODO: Are these the right signs?
    bound_term = lbmult @ primal_ubslack_inv + ubmult @ primal_lbslack_inv

    # Get Hessian in the proper order
    # I'm just assuming that mutipliers here are using the right convention.
    # This appears to be correct based on my grad-lag calculation.
    hess = nlp.extract_submatrix_hessian_lag(varorder, varorder)
    # Add terms for inequalities and bounds
    hess += - ineq_term + bound_term

    rh = nullbasis.transpose() @ hess @ nullbasis
    return rh


def project_onto(mat, coords):
    # Schur complement of a dense matrix wrt the complement of coords
    assert mat.shape[0] == mat.shape[1]
    all_coords = np.arange(mat.shape[0])
    other_coords = all_coords[~np.isin(all_coords, coords)]
    # Is this the right way to slice this matrix?
    mat_11 = mat[coords, :][:, coords]
    mat_22 = mat[other_coords, :][:, other_coords]
    mat_12 = mat[coords, :][:, other_coords]
    mat_21 = mat[other_coords, :][:, coords]
    proj = mat_11 - mat_12.dot(np.linalg.solve(mat_22, mat_21))
    return proj


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
