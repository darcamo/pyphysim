#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See the link below for an "argparse + configobj" option
# http://mail.scipy.org/pipermail/numpy-discussion/2011-November/059332.html

from configobj import ConfigObj, flatten_errors
import validate
from validate import Validator, VdtTypeError
import numpy as np

def _parse_range_expr(value):
    """Parse a string in the form of min:max or min:step:max and return a numpy
    array.

    """
    try:
        limits = value.split(':')
        limits = [float(i) for i in limits]
        if len(limits) == 2:
            value = np.arange(limits[0], limits[1])
        elif len(limits) == 3:
            value = np.arange(limits[0], limits[2], limits[1])
    except Exception:
        raise VdtTypeError(value)

    return value


def real_numpy_array_check(value, minv=None, maxv=None):
    """Value can be either a single number, a range expression in the form of
    min:max or min:step:max, or even a list containing numbers and range
    expressions.

    """
    # Test if it is a list or not
    if isinstance(value, list):
        # If it is a list, each element can be either a number of a 'range
        # expression' that can be parsed with _parse_range_expr. We simple
        # apply real_numpy_array_check on each element in the list to do
        # the work and stack horizontally all the results.
        value = [real_numpy_array_check(a) for a in value]
        value = np.hstack(value)

    else:
        # It its not a list, it can be either a single number of a 'range
        # expression' that can be parsed with _parse_range_expr
        try:
            value = validate.is_float(value)
            value=np.array([value])
        except VdtTypeError:
            value = _parse_range_expr(value)

    return value


if __name__ == '__main__':
    config_file_name = 'psk_simulation_config.txt'

    # Spec could be also the name of a file containing the string below
    spec = """[Simulation]
    SNR=real_numpy_array
    M=integer(min=4, max=512, default=4)
    NSymbs=integer(min=10, max=1000000, default=500)
    rep_max=integer(min=1, default=5000)
    max_bit_errors=integer(min=1, default=300)""".split("\n")

    conf_file_parser = ConfigObj(
        config_file_name,
        list_values=True,
        configspec=spec)

    # Dictionary with custom validation functions
    fdict = {'real_numpy_array': real_numpy_array_check}
    validator = Validator(fdict)

    # The 'copy' argument indicates that if we save the ConfigObj object to
    # a file after validating, the default values will also be written to
    # the file.
    result = conf_file_parser.validate(validator, preserve_errors=True, copy=True)

    # xxxxxxxxxx Test if there was some error in parsing the file xxxxxxxxx
    errors_list = flatten_errors(conf_file_parser, result)

    if len(errors_list) != 0:
        first_error = errors_list[0]
        # The exception will only describe the error for the first
        # incorrect parameter.
        raise Exception("Parameter {0} in section {1} is incorrect.\nMessage: {2}".format(first_error[1], first_error[0], first_error[2].message.capitalize()))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print 'Filename: {0}'.format(config_file_name)
    print "Valid_config_file: {0}".format(result)


    # if result != True:
    #     print 'Config file validation failed!'
    #     sys.exit(1)

    SimulationSection = conf_file_parser['Simulation']
    SNR = SimulationSection['SNR']
    M = SimulationSection['M']
    NSymbs = SimulationSection['NSymbs']
    rep_max = SimulationSection['rep_max']
    max_bit_errors = SimulationSection['max_bit_errors']
    #dummy = SimulationSection['dummy']

    print "Parameters: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    print "SNR: {0}".format(SNR)
    print "M: {0}".format(M)
    print "NSymbs: {0}".format(NSymbs)
    print "rep_max: {0}".format(rep_max)
    print "max_bit_errors: {0}".format(max_bit_errors)
