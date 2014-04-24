#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that reads a SimulationResults file and save the corresponding
partial results.
"""

__revision__ = "$Revision$"

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import argparse

from pyphysim.simulations.results import SimulationResults
from pyphysim.simulations.runner import get_partial_results_filename
from pyphysim.util.misc import replace_dict_values


def main():
    """Main function of the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="The name of the SimulationResults file.")
    parser.add_argument('folder', help="The name of the second SimulationResults file.",
                        nargs='?')

    args = parser.parse_args()

    name = args.name

    if args.folder is None:
        folder = 'subsolfer'
    else:
        folder = args.folder

    print name
    print folder

    results = SimulationResults.load_from_file(name)
    original_filename = results.original_filename

    for i, p in enumerate(results.params.get_unpacked_params_list()):
        print i
        partial_filename = get_partial_results_filename(original_filename, p)

        partial_filename_with_replacements = replace_dict_values(
            partial_filename,
            results.params.parameters,
            filename_mode=True)
        if partial_filename_with_replacements == name:
            raise RuntimeError('invalid name')
        print partial_filename_with_replacements


    # TODO: Finish the implementation
    # Save the partial results

if __name__ == '__main__':
    main()
