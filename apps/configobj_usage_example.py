#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See the link below for an "argparse + configobj" option
# http://mail.scipy.org/pipermail/numpy-discussion/2011-November/059332.html

from configobj import ConfigObj
import validate
from validate import Validator, VdtTypeError
import numpy as np


def real_numpy_array_check(value, minv=None, maxv=None):
    # The value can be a single number or a list of numbers
    try:
        # Lets fist try to interpret it as a list of integers
        value = [validate.is_float(value)]
    except VdtTypeError:
        # It is not a single number
        # Lets try to interpret it as a list of floats
        value = validate.is_float_list(value, minv, maxv)

    return np.array(value)


if __name__ == '__main__':
    config_file_name = 'psk_simulation_config.txt'

    # Spec could be also the name of a file containing the string below
    spec = """[Simulation]
    SNR=real_numpy_array
    M=integer(min=4, max=512, default=4)
    NSymbs=integer(min=10, max=1000000, default=1000)
    rep_max=integer(min=1, max=1000000, default=5000)
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
