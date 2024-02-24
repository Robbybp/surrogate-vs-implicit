import os
import argparse
import itertools


filedir = os.path.dirname(__file__)


def get_data_dir():
    return os.path.join(filedir, "data")


def get_results_dir():
    return os.path.join(filedir, "results")


def get_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data-dir", required=False, default=get_data_dir())
    argparser.add_argument("--results-dir", required=False, default=get_results_dir())
    return argparser


def get_sweep_argparser():
    argparser = get_argparser()
    argparser.add_argument(
        "--n1", default=8, help="Default number of samples for conversion", type=int
    )
    argparser.add_argument(
        "--n2", default=8, help="Default number of samples for pressure", type=int
    )
    return argparser


def get_parameter_samples(args):
    x_lo = 0.90
    x_hi = 0.97
    p_lo = 1447379.0
    p_hi = 1947379.0
    
    n_x = args.n1
    n_p = args.n2
    dx = (x_hi - x_lo) / (n_x - 1)
    dp = (p_hi - p_lo) / (n_p - 1)
    x_list = [x_lo + i * dx for i in range(n_x)]
    p_list = [p_lo + i * dp for i in range(n_p)]

    xp_samples = list(itertools.product(x_list, p_list))
    return xp_samples