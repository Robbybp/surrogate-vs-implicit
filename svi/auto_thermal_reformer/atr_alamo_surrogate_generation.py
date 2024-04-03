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

######## IMPORT PACKAGES ########

import os
import pandas as pd
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.alamopy import AlamoTrainer, AlamoSurrogate
from idaes.core.surrogate.plotting.sm_plotter import surrogate_parity
import svi.auto_thermal_reformer.config as config

######## FUNCTION TO GENERATE ALAMO SURROGATES ########

DEFAULT_DATA_FILE = "data_atr.csv"
DEFAULT_SURR_NAME = "alamo_surrogate_atr.json"
TRAIN_PLOT_NAME = "parity_train_atr.pdf"
VAL_PLOT_NAME = "parity_val_atr.pdf"

def gibbs_to_alamo(fname, 
                   surrogate_fname,
                   train_plot,
                   val_plot,
                   show_surrogates = False, 
                   create_plots = False):
    
    df = pd.read_csv(fname) 
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    input_data = df.iloc[:, :4] 
    output_data = df.iloc[:, 4:]    
    
    # Split training and validation data
    input_labels = input_data.columns
    output_labels = output_data.columns

    n_data = df[input_labels[0]].size
    data_training, data_validation = split_training_validation(df, 0.8, seed=n_data)

    trainer = AlamoTrainer(
        input_labels=input_labels,
        output_labels=output_labels,
        training_dataframe=data_training,
    )

    trainer.config.constant = True
    trainer.config.linfcns = True
    trainer.config.monomialpower = [2,3]

    _, alm_surr, _ = trainer.train_surrogate()

    alm_surr.save_to_file(surrogate_fname, overwrite=True)

    surrogate_expressions = trainer._results["Model"]

    if show_surrogates == True:
        for i in surrogate_expressions:
            print(surrogate_expressions[i])

    input_labels = trainer._input_labels
    output_labels = trainer._output_labels

    bounds = df[['Fin_CH4','Tin_CH4','Fin_H2O','Conversion']].agg(['min', 'max']).T
    input_bounds = {index: (row['min'], row['max']) for index, row in bounds.iterrows()}

    alm_surr = AlamoSurrogate(
        surrogate_expressions, input_labels, output_labels, input_bounds
    )

    surrogate_parity(alm_surr, data_training, filename=train_plot, show = False)
    surrogate_parity(alm_surr, data_validation, filename=val_plot, show = False)

def main():
    
    argparser = config.get_argparser()

    argparser.add_argument(
        "--fname",
        default=DEFAULT_DATA_FILE,
        help="Base file name for training the ALAMO surrogate",
    )
    
    argparser.add_argument(
        "--surrogate_fname",
        default=DEFAULT_SURR_NAME,
        help="File name for the ALAMO surrogate",
    )
    
    argparser.add_argument(
        "--train_plot",
        default=TRAIN_PLOT_NAME,
        help="Base file name for training plot",
    )
    
    argparser.add_argument(
        "--val_plot",
        default=VAL_PLOT_NAME,
        help="Base file name validation plot",
    )
    
    args = argparser.parse_args()

    surrogate_fname = os.path.join(args.data_dir, args.surrogate_fname)
    fname = os.path.join(args.data_dir, args.fname)
    train_plot = os.path.join(args.results_dir, args.train_plot)
    val_plot = os.path.join(args.results_dir, args.val_plot)

    gibbs_to_alamo(fname=fname,
                   surrogate_fname=surrogate_fname,
                   train_plot = train_plot,
                   val_plot = val_plot,
                   show_surrogates = False, 
                   create_plots = False)

if __name__ == "__main__":
    main()
