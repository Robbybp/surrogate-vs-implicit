import os
import argparse


filedir = os.path.dirname(__file__)


def get_data_dir():
    return os.path.join(filedir, "data")


def get_results_dir():
    return os.path.join(filedir, "results")


def get_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_args("--data-dir", required=False, default=get_data_dir())
    argparser.add_args("--results-dir", required=False, default=get_results_dir())
    return argparser
