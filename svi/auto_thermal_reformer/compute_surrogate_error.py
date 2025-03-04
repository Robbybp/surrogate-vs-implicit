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

import pyomo.environ as pyo
from pyomo.common.timing import TicTocTimer
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.contrib.incidence_analysis import solve_strongly_connected_components
from idaes.core.util.exceptions import InitializationError
from svi.auto_thermal_reformer.reactor_model import create_instance


def compute_surrogate_error(model):
    model_surrogate = model.fs.reformer_surrogate

    model_surrogate_outputs = {
        key: var.value for key, var in model_surrogate.output_vars_as_dict().items()
    }

    inputs = [
        model.fs.reformer_bypass.reformer_outlet.flow_mol[0], 
        model.fs.reformer_bypass.reformer_outlet.temperature[0], 
        model.fs.reformer_mix.steam_inlet.flow_mol[0],
        model.fs.reformer.conversion,
    ]

    scc_solver = pyo.SolverFactory("ipopt")

    timer = TicTocTimer()
    timer.tic()

    with TemporarySubsystemManager(to_fix=inputs):
        surrogate_copy = model_surrogate.clone()
        timer.toc("clone-surrogate")
        solve_strongly_connected_components(surrogate_copy, solver=scc_solver)
        timer.toc("solve-scc-surrogate")
        surrogate_res = scc_solver.solve(surrogate_copy)
    if not pyo.check_optimal_termination(surrogate_res):
        print("WARNING: Surrogate model failed to simulate")
        raise ValueError("Surrogate reactor model failed to simulate")

    # If we have converged, these newly computed surrogate outputs should be
    # the same as our "model surrogate outputs" above. If we did not converge,
    # they may be different.
    surrogate_outputs = {
        key: var.value for key, var in surrogate_copy.output_vars_as_dict().items()
    }

    try:
        fullspace_model = create_instance(
            model.fs.reformer.conversion.value,
            model.fs.reformer_mix.steam_inlet.flow_mol[0].value,
            # Note that "reformer_outlet" here means "the outlet of the bypass splitter
            # that goes to the reformer". The other outlet is called "bypass_outlet".
            # The inlet is simply called "inlet".
            model.fs.reformer_bypass.reformer_outlet.flow_mol[0].value,
            model.fs.reformer_bypass.reformer_outlet.temperature[0].value,
            initialize=True,
        )
        timer.toc("create-instance-fullspace")
    except InitializationError:
        print("WARNING: Full-space model failed to initialize. Trying to continue.")
        fullspace_model = create_instance(
            model.fs.reformer.conversion.value,
            model.fs.reformer_mix.steam_inlet.flow_mol[0].value,
            model.fs.reformer_bypass.reformer_outlet.flow_mol[0].value,
            model.fs.reformer_bypass.reformer_outlet.temperature[0].value,
            initialize=False,
        )

    solve_strongly_connected_components(
        fullspace_model,
        solver=scc_solver,
        use_calc_var=False,
    )
    timer.toc("solve-scc-fullspace")
    fullspace_res = scc_solver.solve(fullspace_model, tee=True)
    timer.toc("solve-fullspace")

    if not pyo.check_optimal_termination(fullspace_res):
        # In parameter sweep scripts, this ValueError is caught and we
        # write an empty row in the surrogate-error file.
        # But we don't want to count a failure for the surrogate if this
        # fails... this can be handled by the caller.
        print("WARNING: Full-space model failed to simulate")
        raise ValueError("Full-space reactor model failed to simulate")

    fullspace_output = {
        "Fout": fullspace_model.fs.reformer.outlet.flow_mol[0].value,
        "Tout": fullspace_model.fs.reformer.outlet.temperature[0].value,
        "HeatDuty": fullspace_model.fs.reformer.heat_duty[0].value,
    }
    for key in surrogate_outputs:
        # The keys we have not added so far are component names, which may
        # be used as indices to mole_frac_comp
        if key not in fullspace_output:
            fullspace_output[key] = fullspace_model.fs.reformer.outlet.mole_frac_comp[0, key].value

    relative_errors = {
        key: (
            abs(fullspace_output[key] - surrogate_outputs[key])
            / max(1, fullspace_output[key], surrogate_outputs[key])
        )
        for key in surrogate_outputs
    }
    #max_relative_error = max(relative_errors.values())
    #ave_relative_error = sum(relative_errors.values()) / len(relative_errors)
    timer.toc("done")

    return relative_errors
