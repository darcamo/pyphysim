#!/usr/bin/env python
"""
Module containing function related to serialization.
"""

import json
from typing import Any, Dict, List, Union

import numpy as np

Serializable = Union[np.ndarray, np.int32, np.int64, np.float32, np.float64,
                     np.float128, set]

# A type corresponding to the JSON representation of the object. For a lack of
# a better option we use Any
JsonRepresentation = Any


class NumpyOrSetEncoder(json.JSONEncoder):
    """
    JSON encoder for numpy arrays.

    Pass this class to json.dumps when converting a dictionary to json so
    that any field which with a numpy array as value will be properly
    converted.

    This encoder will also handle numpy scalars and the native python set
    types.

    When you need to convert the json representation back, use the
    `json_numpy_or_set_obj_hook` function.

    See Also
    --------
    json_numpy_or_set_obj_hook
    """
    def default(self, obj: Serializable) -> JsonRepresentation:
        """
        If input object is an ndarray it will be converted into a dict holding
        data, dtype, _is_numpy_array and shape.

        Parameters
        ----------
        obj : Serializable

        Returns
        -------
        Serialized Data
        """
        # Case for numpy arrays
        if isinstance(obj, np.ndarray):
            return {
                'data': obj.tolist(),
                'dtype': str(obj.dtype),
                '_is_numpy_array': True,
                'shape': obj.shape
            }
        # Case for numpy scalars
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64, np.float128)):
            return int(obj)

        # Case for built-in Python sets
        if isinstance(obj, set):
            return {'data': list(obj), '_is_set': True}

        # If it is not a numpy array we fall back to base class encoder
        return json.JSONEncoder(self, obj)  # type: ignore


def json_numpy_or_set_obj_hook(
        dct: Dict[str, JsonRepresentation]) -> Serializable:
    """
    Decodes a previously encoded numpy array.

    Parameters
    ----------
    dct : dict
        The JSON encoded numpy array.

    Returns
    -------
    np.ndarray | set | dict, optional
        The decoded numpy array or None if the encoded json data was not an
        encoded numpy array.

    See Also
    --------
    NumpyOrSetEncoder
    """
    if isinstance(dct, dict) and '_is_numpy_array' in dct:
        if dct['_is_numpy_array'] is True:
            data = dct['data']
            return np.array(data)

        raise ValueError(
            'Json representation contains the "_is_numpy_array" key '
            'indicating that the object should be a numpy array, but it '
            'was set to False, which is not valid.')
    if isinstance(dct, dict) and '_is_set' in dct:
        if dct['_is_set'] is True:
            data = dct['data']
            return set(data)

        raise ValueError(
            'Json representation contains the "_is_set" key '
            'indicating that the object should be python set, but it '
            'was set to False, which is not valid.')
    return dct


class JsonSerializable:
    """
    Base class for classes you want to be JSON serializable (convert
    to/from JSON).

    You can call the methods `to_json` and `from_json` methods (the later
    is a staticmethod).

    Note that a subclass must implement the `_to_dict` and `_from_dict` methods.
    """
    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            The dictionary representation of the object.
        """
        raise NotImplementedError("Implement in a subclass")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            The dictionary representation of the object.
        """
        return self._to_dict()

    @staticmethod
    def _from_dict(d: Dict[str, Any]) -> Any:
        """
        Convert from a dictionary to an object.

        Parameters
        ----------
        d : dict
            The dictionary representing the object.

        Returns
        -------
        Result
            The converted object.
        """
        raise NotImplementedError("Implement in a subclass")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Any:
        """
        Convert from a dictionary to an object.

        Parameters
        ----------
        d : dict
            The dictionary representing the Result.

        Returns
        -------
        Result
            The converted object.
        """
        return cls._from_dict(d)

    def to_json(self) -> JsonRepresentation:
        """
        Convert the object to JSON.

        Returns
        -------
        str
            JSON representation of the object.
        """
        return json.dumps(self._to_dict(), cls=NumpyOrSetEncoder)

    @classmethod
    def from_json(cls, data: JsonRepresentation) -> Any:
        """
        Convert a JSON representation of the object to an actual object.

        Parameters
        ----------
        data : str
            The JSON representation of the object.

        Returns
        -------
        any
            The actual object
        """
        d = json.loads(data, object_hook=json_numpy_or_set_obj_hook)
        return cls._from_dict(d)


# xxxxxxxxxx Test and Example Usage xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    expected = np.arange(100, dtype=np.float)
    dumped = json.dumps(expected, cls=NumpyOrSetEncoder)
    result = json.loads(dumped, object_hook=json_numpy_or_set_obj_hook)
    print(type(result))
    print(result)
