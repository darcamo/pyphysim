#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that reads two files with saved SimulationResults and combine
them into a new SimulationResults which is saved to a new file.
"""

__revision__ = "$Revision$"


# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import argparse

from pyphysim.simulations.results import SimulationResults, combine_simulation_results


def main():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('first', help="The name of the first SimulationResults file.")
    parser.add_argument('second', help="The name of the second SimulationResults file.")
    parser.add_argument(
        'output',
        help="The name that will be used to save the combinedSimulationResults file.")

    args = parser.parse_args()

    first = SimulationResults.load_from_file(args.first)
    second = SimulationResults.load_from_file(args.first)

    print first
    print second

    union = combine_simulation_results(first, second)

    union.save_to_file(args.output)


if __name__ == '__main__':
    main()
