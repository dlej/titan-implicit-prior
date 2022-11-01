# Author: Ali Siahkoohi, alisk@rice.edu
# Date: October 2022
"""Utility functions for reading input arguments and naming experiments.

Typical usage example:

# Read config json file from `configsdir()`.
args = read_config(configsdir(super-resolution.json))

# Parse input command line argument to overwrite default values in the config
json file.
parse_input_args(args)

# Use the parsed input arguments to create experiment name.
make_experiment_name(args)
"""

import argparse
import json


def read_config(filename):
    """Read input variables and values from a json file."""
    with open(filename) as f:
        configs = json.load(f)
    return configs


def parse_input_args(args):
    "Use variables in args to create command line input parser."
    parser = argparse.ArgumentParser(description='')
    for key, value in args.items():
        parser.add_argument('--' + key, default=value, type=type(value))
    return parser.parse_args()


def make_experiment_name(args):
    """Make experiment name based on input arguments"""
    experiment_name = ''
    for key, value in vars(args).items():
        experiment_name += key + '-{}_'.format(value)
    return experiment_name[:-1].replace(' ', '')
