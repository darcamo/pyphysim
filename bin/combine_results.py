#!/usr/bin/env python
"""
Script that reads two files with saved SimulationResults and combine
them into a new SimulationResults which is saved to a new file.
"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import argparse

from pyphysim.simulations.results import SimulationResults, \
    combine_simulation_results
from pyphysim.util.misc import replace_dict_values


def main():
    """Main function of the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('first',
                        help="The name of the first SimulationResults file.")
    parser.add_argument('second',
                        help="The name of the second SimulationResults file.")
    parser.add_argument('output',
                        help=("The name that will be used to save the combined "
                              "SimulationResults file."),
                        nargs='?')

    args = parser.parse_args()

    first = SimulationResults.load_from_file(args.first)
    second = SimulationResults.load_from_file(args.second)
    union = combine_simulation_results(first, second)

    if args.output is None:
        output = replace_dict_values(first.original_filename,
                                     union.params.parameters,
                                     filename_mode=True)
    else:
        output = args.output

    if output == args.first or output == args.second:
        raise RuntimeError(
            "output filename must be different from the filename of either"
            " of the two SimulationResults.")

    # Finally save to the output file
    union.save_to_file(output)


if __name__ == '__main__':
    main()
