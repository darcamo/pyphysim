#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module containing function related to serialization.
"""


import numpy as np
import json


class NumpyOrSetEncoder(json.JSONEncoder):
    """
    Json encoder for numpy arrays.
    """
    def default(self, obj):
        """
        If input object is an ndarray it will be converted into a dict holding
        data, dtype, _is_numpy_array and shape.

        Parameters
        ----------
        obj : np.ndarray | any

        Returns
        -------
        Serialized Data
        """
        # Case for numpy arrays
        if isinstance(obj, np.ndarray):
            return {'data': obj.tolist(),
                    'dtype': str(obj.dtype),
                    '_is_numpy_array': True,
                    'shape': obj.shape}
        # Case for numpy scalars
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64, np.float128)):
            return int(obj)

        # Case for built-in Python sets
        if isinstance(obj, set):
            return {'data': list(obj),
                    '_is_set': True}

        # If it is not a numpy array we fall back to base class encoder
        return json.JSONEncoder(self, obj)


def json_numpy_or_set_obj_hook(dct):
    """
    Decodes a previously encoded numpy array.

    Parameters
    ----------
    dct : dict
        The JSON encoded numpy array.

    Returns
    -------
    np.ndarray | set | dict
        The decoded numpy array or None if the encoded json data was not an
        encoded numpy array.
    """
    if isinstance(dct, dict) and '_is_numpy_array' in dct:
        if dct['_is_numpy_array'] is True:
            data = dct['data']
            return np.array(data)
        else:  # pragma: no cover
            raise ValueError(
                'Json representation contains the "_is_numpy_array" key '
                'indicating that the object should be a numpy array, but it '
                'was set to False, which is not valid.')
    if isinstance(dct, dict) and '_is_set' in dct:
        if dct['_is_set'] is True:
            data = dct['data']
            return set(data)
        else:  # pragma: no cover
            raise ValueError(
                'Json representation contains the "_is_set" key '
                'indicating that the object should be python set, but it '
                'was set to False, which is not valid.')
    return dct


# xxxxxxxxxx Test and Example Usage xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    expected = np.arange(100, dtype=np.float)
    dumped = json.dumps(expected, cls=NumpyOrSetEncoder)
    result = json.loads(dumped, object_hook=json_numpy_or_set_obj_hook)
    print(type(result))
    print(result)
