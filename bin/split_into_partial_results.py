#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that reads a SimulationResults file and save the corresponding
partial results.
"""

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


# TODO: Test if the partial results saved by this script are correct
def main():
    """Main function of the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="The name of the SimulationResults file.")
    parser.add_argument('folder',
                        help="The name of the second SimulationResults file.",
                        nargs='?')

    args = parser.parse_args()

    name = args.name
    folder = args.folder  # This will be None, if not provided

    results = SimulationResults.load_from_file(name)

    original_filename = results.original_filename

    # xxxxxxxxxx Remove the .pickle file extension xxxxxxxxxxxxxxxxxxxxxxxx
    # If partial_filename has a 'pickle' extension we remove it from the name
    original_filename_no_ext, ext = os.path.splitext(original_filename)
    if ext == '.pickle':
        original_filename = original_filename_no_ext
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    unpacked_params_list = results.params.get_unpacked_params_list()
    all_results_names = results.get_result_names()

    for i, p in enumerate(results.params.get_unpacked_params_list()):
        partial_filename = get_partial_results_filename(
            original_filename, p, folder)
        partial_filename_with_replacements = replace_dict_values(
            partial_filename,
            results.params.parameters,
            filename_mode=True)

        if partial_filename_with_replacements == name:
            raise RuntimeError('invalid name')

        partial_param = unpacked_params_list[i]
        partial_result = SimulationResults()
        partial_result.set_parameters(partial_param)

        for result_name in all_results_names:
            partial_result.add_result(results[result_name][i])

        partial_result.current_rep = results.runned_reps[i]

        # Try to save the partial results. If we get an error
        try:
            # If we get an IOError exception here, maybe we are trying to
            # save to a folder and the folder does not exist yet.
            partial_result.save_to_file(partial_filename_with_replacements)
        except IOError as e:
            if folder is not None:
                # Lets create the folder...
                os.mkdir(folder)
                # ... and try to save the file again.
                partial_result.save_to_file(partial_filename_with_replacements)
            else:
                raise e


if __name__ == '__main__':
    main()
