#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module implementing validation functions to define "specs" for configobj
validation.

This module is not intended to be used directly. The functions defined here
are used in the "simulations" module.
"""

__revision__ = "$Revision$"

import numpy as np

def _parse_range_expr(value, converter=float):
    """
    Parse a string in the form of min:max or min:step:max and return a
    numpy array.
    """
    import validate
    try:
        limits = value.split(':')
        limits = [converter(i) for i in limits]
        if len(limits) == 2:
            value = np.arange(limits[0], limits[1])
        elif len(limits) == 3:
            value = np.arange(limits[0], limits[2], limits[1])
    except Exception:
        raise validate.VdtTypeError(value)

    return value


def _parse_float_range_expr(value):
    """
    Parse a string in the form of min:max or min:step:max and return a
    numpy array (of floats).
    """
    return _parse_range_expr(value, float)


def _parse_int_range_expr(value):
    """
    Parse a string in the form of min:max or min:step:max and return a
    numpy array (of integers).
    """
    return _parse_range_expr(value, int)

# pylint: disable= W0622
def _real_numpy_array_check(value, min=None, max=None):
    """
    Parse and validate `value` as a numpy array (of floats).

    Value can be either a single number, a range expression in the form of
    min:max or min:step:max, or even a list containing numbers and range
    expressions.

    Notes
    -----
    You can either separate the values with commas or spaces (any comma
    will have the same effect as a space). However, if you separate with
    spaces the values should be brackets, while if you separate with
    commands there should be no brackets.
    .. code::
        SNR = 0,5,10:20
        SNR = [0 5 10:20]
    """
    import validate
    if isinstance(value, str):
        # Remove '[' and ']' if they exist.
        if value[0] == '[' and value[-1] == ']':
            value = value[1:-1].strip()
            value = value.replace(',', ' ')  # Replace any commas by a space
            value = value.split()  # Split based on spaces

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Test if it is a list or not
    if isinstance(value, list):
        # If it is a list, each element can be either a number of a 'range
        # expression' that can be parsed with _parse_float_range_expr. We
        # simple apply _real_numpy_array_check on each element in the list
        # to do the work and stack horizontally all the results.
        value = [_real_numpy_array_check(a, min, max) for a in value]
        value = np.hstack(value)

    else:
        # It its not a list, it can be either a single number of a 'range
        # expression' that can be parsed with _parse_float_range_expr
        try:
            value = validate.is_float(value)
            value = np.array([value])
        except validate.VdtTypeError:
            value = _parse_float_range_expr(value)

    # xxxxxxxxxx Validate if minimum and maximum allowed values xxxxxxxxxxx
    if min is not None:
        # maybe "min" was passed as a string and thus we need to convert it
        # to a float
        min = float(min)
        if value.min() < min:
            raise validate.VdtValueTooSmallError(value.min())

    if max is not None:
        # maybe "min" was passed as a string and thus we need to convert it
        # to a float
        max = float(max)
        if value.max() > max:
            raise validate.VdtValueTooBigError(value.max())
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return value


def _integer_numpy_array_check(value, min=None, max=None):
    """
    Parse and validate `value` as a numpy array (of integers).

    Value can be either a single number, a range expression in the form of
    min:max or min:step:max, or even a list
    containing numbers and range expressions.

    Notes
    -----
    You can either separate the values with commas or spaces (any comma
    will have the same effect as a space). However, if you separate with
    spaces the values should be brackets, while if you separate with
    commands there should be no brackets.
    .. code::
        max_iter = 5,10:20
        max_iter = [0 5 10:20]
    """
    import validate
    if isinstance(value, str):
        # Remove '[' and ']' if they exist.
        if value[0] == '[' and value[-1] == ']':
            value = value[1:-1].strip()
            value = value.replace(',', ' ')  # Replace any commas by a space
            value = value.split()  # Split based on spaces

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Test if it is a list or not
    if isinstance(value, list):
        # If it is a list, each element can be either a number of a 'range
        # expression' that can be parsed with _parse_int_range_expr. We simple
        # apply _integer_numpy_array_check on each element in the list to do
        # the work and stack horizontally all the results.
        value = [_integer_numpy_array_check(a, min, max) for a in value]
        value = np.hstack(value)

    else:
        # It its not a list, it can be either a single number of a 'range
        # expression' that can be parsed with _parse_int_range_expr
        try:
            value = validate.is_integer(value)
            value = np.array([value])
        except validate.VdtTypeError:
            value = _parse_int_range_expr(value)

    # xxxxxxxxxx Validate if minimum and maximum allowed values xxxxxxxxxxx
    if min is not None:
        # maybe "min" was passed as a string and thus we need to convert it
        # to a integer
        min = int(min)
        if value.min() < min:
            raise validate.VdtValueTooSmallError(value.min())

    if max is not None:
        # maybe "min" was passed as a string and thus we need to convert it
        # to a integer
        max = int(max)
        if value.max() > max:
            raise validate.VdtValueTooBigError(value.max())
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return value
