#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See the link below for an "argparse + configobj" option
# http://mail.scipy.org/pipermail/numpy-discussion/2011-November/059332.html

from configobj import ConfigObj, flatten_errors
import validate
from validate import Validator, VdtTypeError
import numpy as np


def _parse_range_expr(value):
    """
    Parse a string in the form of min:max or min:step:max and return a
    numpy array.
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


# TODO: Check agains the 'min' and 'max'. For now they are ignored.
def real_numpy_array_check(value, min=None, max=None):
    """Parse and validate `value` as a numpy array.

    Value can be either a single number, a range expression in the form of
    min:max or min:step:max, or even a list containing numbers and range
    expressions.

    Notes:
    ------
    You can either separate the values with commas or spaces. However, if
    you separate with spaces the values should be brackets, while if you
    separate with commands there should be no brackets.
    .. code::
        SNR = 0,5,10:20
        SNR = [0 5 10:20]
    """
    if isinstance(value, str):
        # Remove '[' and ']' if they exist.
        if value[0] == '[' and value[-1] == ']':
            value = value[1:-1].strip()
            value = value.split()

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
            value = np.array([value])
        except VdtTypeError:
            value = _parse_range_expr(value)

    return value


if __name__ == '__main__':
    config_file_name = 'psk_simulation_config.txt'

    # Spec could be also the name of a file containing the string below
    spec = """[Scenario]
    SNR=real_numpy_array(default=15)
    modulator=option('PSK', 'QAM', 'BPSK', default="PSK")
    M=integer(min=4, max=512, default=4)
    NSymbs=integer(min=10, max=1000000, default=200)
    K=integer(min=2,default=3)
    Nr=integer(min=2,default=2)
    Nt=integer(min=2,default=2)
    Ns=integer(min=1,default=1)
    [IA Algorithm]
    max_iterations=integer(min=1, default=60)
    [General]
    rep_max=integer(min=1, default=2000)
    max_bit_errors=integer(min=1, default=3000)
    unpacked_parameters=string_list(default=list('SNR'))
    """.split("\n")

    conf_file_parser = ConfigObj(
        config_file_name,
        list_values=True,
        configspec=spec)

    #conf_file_parser.write()

    # Dictionary with custom validation functions
    fdict = {'real_numpy_array': real_numpy_array_check}
    validator = Validator(fdict)

    # The 'copy' argument indicates that if we save the ConfigObj object to
    # a file after validating, the default values will also be written to
    # the file.
    result = conf_file_parser.validate(validator, preserve_errors=True, copy=True)

    # Note that if thare was no parsing errors, then "result" will be True.
    # It there was an error, then result will be a dictionary with each
    # parameter as a key. The value of each key will be either 'True' if
    # that parameter was parsed without error or a "validate.something"
    # object (since we set preserve_errors to True) describing the error.

    # if result != True:
    #     print 'Config file validation failed!'
    #     sys.exit(1)

    # xxxxxxxxxx Test if there was some error in parsing the file xxxxxxxxx
    # The flatten_errors function will return only the parameters whose
    # parsing failed.
    errors_list = flatten_errors(conf_file_parser, result)

    if len(errors_list) != 0:
        first_error = errors_list[0]
        # The exception will only describe the error for the first
        # incorrect parameter.
        if first_error[2] is False:
            raise Exception("Parameter '{0}' in section '{1}' must be provided.".format(first_error[1], first_error[0][0]))
        else:
            raise Exception("Parameter '{0}' in section '{1}' is invalid. {2}".format(first_error[1], first_error[0][0], first_error[2].message.capitalize()))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print 'Filename: {0}'.format(config_file_name)
    print "Valid_config_file: {0}".format(result)

    # Simulation Section
    SimulationSection = conf_file_parser['Scenario']
    SNR = SimulationSection['SNR']
    M = SimulationSection['M']
    NSymbs = SimulationSection['NSymbs']
    modulator_name = SimulationSection['modulator']

    # IA Section
    IASection = conf_file_parser['IA Algorithm']
    max_iterations = IASection['max_iterations']

    # General Section
    GeneralSection = conf_file_parser['General']
    rep_max = GeneralSection['rep_max']
    max_bit_errors = GeneralSection['max_bit_errors']
    unpacked_parameters = GeneralSection['unpacked_parameters']

    print "Parameters: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    print "SNR: {0}".format(SNR)
    print "M: {0}".format(M)
    print "Modulator: {0}".format(modulator_name)
    print "NSymbs: {0}".format(NSymbs)
    print "rep_max: {0}".format(rep_max)
    print "max_bit_errors: {0}".format(max_bit_errors)
    print "unpacked_parameters: {0}".format(unpacked_parameters)

    conf_file_parser.write()
