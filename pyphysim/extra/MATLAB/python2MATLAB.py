#!/usr/bin/env python
"""Module with functions to easily moving data from python to MATLAB."""

import numpy as np

__all__ = ["to_mat_str"]


def to_mat_str(  # type:ignore
        x: np.ndarray, format_string: str = '+.12e') -> str:
    """Convert the ndarray 'x' to a string corresponding to the MATLAB
    representation of `x`.

    The to_mat_str function formats numpy arrays of arbitrary dimension in
    a way which can easily copied and pasted into an interactive MATLAB
    session

    Parameters
    ----------
    x : numpy array
        The numpy array to be represented as a MATLAB type.
    format_string : str, optional
        The format_string string to convert each element in `x`.

    Returns
    -------
    converted_string : str
        A string that represents the converted numpy array. You can copy
        this string and past it into a MATLAB session.

    Examples
    --------
    >>> a=np.arange(1,10)
    >>> a.shape=(3,3)
    >>> # Print as a numpy matrix
    >>> print(a)
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    >>> # Call to_mat_str(a) to print the string representation of the
    >>> # converted matrix
    >>> print(to_mat_str(a))
    [\
+1.000000000000e+00, +2.000000000000e+00, +3.000000000000e+00; \
+4.000000000000e+00, +5.000000000000e+00, +6.000000000000e+00; \
+7.000000000000e+00, +8.000000000000e+00, +9.000000000000e+00]

    """

    # noinspection PyShadowingNames
    def convert_row_or_col(numpy_array: np.array,
                           format_string: str,
                           separator: str = ', ') -> str:
        """Convert a one-dimensional numpy array to its MATLAB
        representation

        Parameters
        ----------
        numpy_array : numpy array
            The array to be converted.
        format_string : str
            The format string for the conversion.
        separator : str
            The separator for 2 elements.

        """
        # {0:+.12e}
        # +.12e
        output = []
        if numpy_array.dtype == 'complex':
            format_string = "{{0:{0}}}{{1:{0}}}j".format(  # type: ignore
                format_string, format_string)
            for i in numpy_array:
                output.append(format_string.format(i.real, i.imag))
        else:
            format_string = '{{0:{0}}}'.format(format_string)
            for i in numpy_array:
                output.append(format_string.format(i))
        return separator.join(output)

    if x.ndim == 1:
        # 1d input
        return f'[{convert_row_or_col(x, format_string)}]'

    if x.ndim == 2:
        if x.shape[1] == 1:
            # This is a Column vector
            return f"[{convert_row_or_col(np.reshape(x, x.size), format_string, separator='; ')}]"
        else:
            # This is not a column vector
            output = []
            for row in x:
                output.append(convert_row_or_col(row, format_string))
            return f"[{'; '.join(output)}]"

    if x.ndim > 2:  # pragma: no cover
        raise NotImplementedError('This case is not implemented')
