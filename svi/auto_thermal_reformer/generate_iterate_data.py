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
import svi.auto_thermal_reformer.config as config
from svi.cyipopt import FullStateCallback
import pandas as pd


def main(args):
    if args.sample is None:
        conversion = 0.94
        pressure = 1550000.0
    else:
        xp_samples = config.get_parameter_samples(args)
        conversion, pressure = xp_samples[args.sample]
    m = config.CONSTRUCTOR_LOOKUP[args.model](conversion, pressure)

    dof_varnames = [
        "fs.reformer_bypass.split_fraction[0.0,bypass_outlet]",
        "fs.reformer_mix.steam_inlet_state[0.0].flow_mol",
        "fs.feed.properties[0.0].flow_mol",
    ]
    callback = FullStateCallback(
        include_condition=args.include_condition,
        include_block_condition=args.include_block_condition,
        dof_varnames=dof_varnames,
    )
    solver = config.get_optimization_solver(callback=callback)

    results = solver.solve(m, tee=True)

    if args.fname is None:
        fname = f"{args.model}-iterates"
        if args.sample is not None:
            fname += f"-{args.sample}"
        fname += ".csv"
    else:
        fname = args.fname
    fpath = os.path.join(args.data_dir, fname)

    iterate_data = dict(callback.iterate_data)
    # TODO: Should the keys here be updated to indicate that they are primal
    # values and residuals? Probably, but I'll deal with that later.
    iterate_data.update(callback.primal_values)
    iterate_data.update(callback.primal_residuals)
    # Make sure nothing pathological happened, like we had a variable named "inf_du"
    assert len(iterate_data) == (
        len(callback.iterate_data) + len(callback.primal_values) + len(callback.primal_residuals)
    )

    if args.include_condition:
        iterate_data["condition-number"] = list(callback.condition_numbers)
    if args.include_block_condition:
        iterate_data.update(callback.block_condition_numbers)

    df = pd.DataFrame(iterate_data)
    if not args.no_save:
        print(f"Saving iterate data to {fpath}")
        df.to_csv(fpath)
    else:
        print("--no-save is set. Not saving iterate data")


if __name__ == "__main__":
    argparser = config.get_sweep_argparser()

    argparser.add_argument(
        "--model",
        default="fullspace",
        help="Options are 'fullspace', 'implcit', 'alamo', or 'nn-full'. Default is 'fullspace'.",
    )
    argparser.add_argument(
        "--fname",
        default=None,
        help="Basename for iterate data CSV file. Default is: {model}-iterates-{sample}.csv",
    )
    argparser.add_argument(
        "--sample",
        default=None,
        type=int,
        help="Index of conversion/pressure parameter sample to use. Default is X=0.95, P=1.55 MPa",
    )
    argparser.add_argument(
        "--include-condition",
        action="store_true",
    )
    argparser.add_argument(
        "--include-block-condition",
        action="store_true",
    )

    args = argparser.parse_args()
    if args.subset is not None:
        raise RuntimeError("--subset cannot be provided. Use --sample instead")
    main(args)
