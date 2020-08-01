#!/usr/bin/env python
"""Module containing simulation result classes."""

import os.path
from collections.abc import Iterable
from typing import Any, Dict, Iterator, List, Optional, cast

import numpy as np

from ..util.misc import (calc_confidence_interval, equal_dicts,
                         replace_dict_values)
from ..util.serialize import JsonSerializable
from .parameters import SimulationParameters, combine_simulation_parameters

try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle  # type: ignore

try:
    # noinspection PyUnresolvedReferences
    import pandas as pd
    DataFrame = pd.DataFrame
except ImportError:  # pragma: no cover
    # This will be used just for type checking, since pandas is not installed
    from typing import Any as DataFrame

# One of SUMTYPE, RATIOTYPE, MISCTYPE or CHOICETYPE
ResultType = int

__all__ = ["combine_simulation_results", "SimulationResults", "Result"]


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Module Functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def combine_simulation_results(
        simresults1: "SimulationResults",
        simresults2: "SimulationResults") -> "SimulationResults":
    """
    Combine two SimulationResults objects with different parameters values.

    For this function to work both simulation objects need to have exact
    the same parameters and only the values of the parameters set to be
    unpacked can be different.

    Parameters
    ----------
    simresults1 : SimulationResults
        The first SimulationResults object to be combined.
    simresults2 : SimulationResults
        The second SimulationResults object to be combined.

    Returns
    -------
    SimulationResults
        The combined SimulationResults object.

    Examples
    --------
    If the first SimulationResults object was obtained for the parameters
    "p1 = 10" and "p2 = [1, 2, 3]", while the second SimulationResults
    object was obtained for the parameters "p1 = 10" and "p2 = [2, 4, 6]"
    and p2 was marked to be unpacked in both of them, then the returned
    combined SimulationResults object will have parameters "p1 = 10" and
    "p2 = [1, 2, 3, 4, 6]" with p2 marked to be unpacked.

    Note that the results for the values of p2 equal to "2" and "4" exist
    in both objects and will be merged together.
    """
    # Create the combined simulation parameters
    combined_params = combine_simulation_parameters(simresults1.params,
                                                    simresults2.params)

    result_names = simresults1.get_result_names()
    if set(result_names) != set(simresults2.get_result_names()):
        raise RuntimeError(
            'Both SimulationResults objects must have the same results.')

    union = SimulationResults()
    union.set_parameters(combined_params)

    for name in result_names:
        result_list1 = simresults1[name]
        result_list2 = simresults2[name]
        type_code = result_list1[0].type_code
        for unpack in combined_params.get_unpacked_params_list():
            # Create an empty Result object.
            result_object = Result(name, type_code)

            # Dictionary with the current unpack variation
            fixed_parameters = unpack.parameters

            try:
                index1 = simresults1.params.get_pack_indexes(fixed_parameters)
                result_object.merge(result_list1[index1[0]])
            except ValueError:
                pass

            try:
                index2 = simresults2.params.get_pack_indexes(fixed_parameters)
                result_object.merge(result_list2[index2[0]])
            except ValueError:
                pass

            union.append_result(result_object)

    return union


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Result - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Result(JsonSerializable):
    """Class to store a single simulation result.

    A simulation result can be anything, such as the number of errors,
    a string, an error rate, etc. When creating a `Result` object one
    needs to specify only the `name` of the stored result and the result
    `type`.

    The different types indicate how multiple samples (from multiple
    iterations) of the same Result can be merged (usually to get a result
    with more statistical reliability). The possible values are SUMTYPE,
    RATIOTYPE and MISCTYPE.

    In the `SUMTYPE` the new value should be added to current one in update
    function.

    In the `RATIOTYPE` the new value should be added to current one and
    total should be also updated in the update function. One caveat is that
    rates are stored as a number (numerator) and a total (denominator)
    instead of as a float. For instance, if you need to store a result such
    as a bit error rate, then you could use the a Result with the RATIOTYPE
    type and when updating the result, pass the number of bit errors and
    the number of simulated bits.

    The `MISCTYPE` type can store anything and the update will simple
    replace the stored value with the current value.

    Parameters
    ----------
    name : str
        Name of the Result.
    update_type_code : int
        Type of the result. It must be one of the elements in
        {Result.SUMTYPE, Result.RATIOTYPE,
        Result.MISCTYPE, Result.CHOICETYPE}.
    accumulate_values : bool
        If True, then the values `value` and `total` will be
        accumulated in the `update` (and merge) method(s). This means
        that the Result object will use more memory as more and more
        values are accumulated, but having all values sometimes is
        useful to perform statistical calculations. This is useful for
        debugging/testing.
    choice_num : int
        Number of different choices for the CHOICETYPE type. This is a
        required parameter for the CHOICETYPE type, but it is ignored
        for the other types

    Examples
    --------
    - Example of the SUMTYPE result.

      >>> result1 = Result("name", Result.SUMTYPE)
      >>> result1.update(13)
      >>> result1.update(4)
      >>> result1.get_result()
      17
      >>> result1.num_updates
      2
      >>> result1
      Result -> name: 17
      >>> result1.type_name
      'SUMTYPE'
      >>> result1.type_code
      0

    - Example of the RATIOTYPE result.

      >>> result2 = Result("name2", Result.RATIOTYPE)
      >>> result2.update(4,10)
      >>> result2.update(3,4)
      >>> result2.get_result()
      0.5
      >>> result2.type_name
      'RATIOTYPE'
      >>> result2.type_code
      1
      >>> result2_other = Result("name2", Result.RATIOTYPE)
      >>> result2_other.update(3,11)
      >>> result2_other.merge(result2)
      >>> result2_other.get_result()
      0.4
      >>> result2_other.num_updates
      3
      >>> result2_other._value
      10
      >>> result2_other._total
      25
      >>> result2.get_result()
      0.5
      >>> print(result2_other)
      Result -> name2: 10/25 -> 0.4

    - Example of the MISCTYPE result.

      The MISCTYPE result 'merge' process in fact simple replaces the
      current stored value with the new value.

    """
    # Like an Enumeration for the type of results.
    (SUMTYPE, RATIOTYPE, MISCTYPE, CHOICETYPE) = range(4)
    _all_types = {
        SUMTYPE: "SUMTYPE",
        RATIOTYPE: "RATIOTYPE",
        MISCTYPE: "MISCTYPE",
        CHOICETYPE: "CHOICETYPE",
    }

    def __init__(self,
                 name: str,
                 update_type_code: ResultType,
                 accumulate_values: bool = False,
                 choice_num: Optional[int] = None) -> None:
        """
        Constructor for the result object.

        Parameters
        ----------
        name : str
            Name of the Result.
        update_type_code : int
            Type of the result. It must be one of the elements in
            {Result.SUMTYPE, Result.RATIOTYPE,
            Result.MISCTYPE, Result.CHOICETYPE}.
        accumulate_values : bool
            If True, then the values `value` and `total` will be
            accumulated in the `update` (and merge) method(s). This means
            that the Result object will use more memory as more and more
            values are accumulated, but having all values sometimes is
            useful to perform statistical calculations. This is useful for
            debugging/testing.
        choice_num : int
            Number of different choices for the CHOICETYPE type. This is a
            required parameter for the CHOICETYPE type, but it is ignored
            for the other types
        """
        self.name: str = name
        self._update_type_code = update_type_code
        self._value: Any = 0
        self._total: Any = 0
        # At each update the current result will be added to this variable
        self._result_sum: float = 0.0
        # At each update the square of the current result will be added to
        # this variable.
        self._result_squared_sum: float = 0.0
        # Number of times the Result object was updated
        self.num_updates: int = 0

        if update_type_code == Result.CHOICETYPE:
            if not isinstance(choice_num, int):
                raise RuntimeError(
                    "'choice_num' argument for the Result object must be "
                    "an integer for the CHOICETYPE type.")

            self._value = np.zeros(choice_num, dtype=int)

        # Accumulation of values: This is useful for debugging/testing
        self._accumulate_values_bool: bool = accumulate_values
        self._value_list: List[Any] = []
        self._total_list: List[Any] = []

    def __eq__(self, other: Any) -> bool:
        """
        Compare two Result objects.

        Two Result objects are considered equal if all attributes in both
        of them are equal, with the exception of the 'num_updates' member
        variable which is ignored in the comparison.

        Parameters
        ----------
        other : Result | any
            The other Result object. It it is not a Result object then the
            comparison will always yield False.

        Returns
        -------
        bool
            True if `other` is equal to `self`, False otherwise.
        """
        # All class attributes with the exception of 'num_updates' and
        # '_value'. The value of 'num_updates' is not important for
        # equality comparison. The value of '_value' is important, but it
        # is not included in 'attributes' because it will be explicitly
        # tested later
        attributes = [
            'name', '_update_type_code', '_total', '_accumulate_values_bool',
            '_value_list', '_total_list', '_result_squared_sum', '_result_sum'
        ]
        if self is other:  # pragma: no cover
            return True

        if not isinstance(other, self.__class__):
            return False

        result = True
        for att in attributes:
            if getattr(self, att) != getattr(other, att):
                result = False

        # Test if the '_value' fields are equal in both objects
        if self._update_type_code == Result.CHOICETYPE:
            # For the CHOICETYPE _value is a numpy array and thus we need
            # to use 'all_true'
            if np.array_equal(self._value, other._value) is False:
                result = False
        else:
            if self._value != other._value:
                result = False

        return result

    def __ne__(self, other: Any) -> bool:
        """
        Compare two Result objects.

        Two Result objects are considered equal if all attributes in both
        of them are equal, with the exception of the 'num_updates' member
        variable which is ignored in the comparison.

        Parameters
        ----------
        other : Result
            The other Result object.

        Returns
        -------
        bool
            False if `other` is equal to `self`, True otherwise.
        """
        return not self.__eq__(other)

    @property
    def accumulate_values_bool(self) -> bool:
        """
        Property to see if values are accumulated of not during a call
        to the `update` method.
        """
        return self._accumulate_values_bool

    @staticmethod
    def create(name: str,
               update_type: ResultType,
               value: Any,
               total: int = 0,
               accumulate_values: bool = False) -> "Result":
        """
        Create a Result object and update it with `value` and `total` at
        the same time.

        Equivalent to creating the object and then calling its
        :meth:`update` method.

        Parameters
        ----------
        name : str
            Name of the Result.
        update_type : int
            Type of the result. It must be one of the elements in
            {Result.SUMTYPE, Result.RATIOTYPE,
            Result.MISCTYPE, Result.CHOICETYPE}.
        value : any
            Value of the result.
        total : any | int | float
            Total value of the result (used only for the RATIOTYPE and
            CHOICETYPE). For the CHOICETYPE it is interpreted as the number
            of different choices if it is an integer or the current value of
            each choice if it is a list.
        accumulate_values : bool
            If True, then the values `value` and `total` will be
            accumulated in the `update` (and merge) method(s). This means
            that the Result object will use more memory as more and more
            values are accumulated, but having all values sometimes is
            useful to perform statistical calculations. This is useful for
            debugging/testing.

        Returns
        -------
        Result
            The new Result object.

        Notes
        -----
        Even if accumulate_values is True the values will not be
        accumulated for the MISCTYPE.

        See also
        --------
        update
        """
        if update_type == Result.CHOICETYPE:
            if total == 0:
                raise RuntimeError(
                    "When creating a new Result of CHOICETYPE you must "
                    "provide the 'total' as well as the 'value.")
            result = Result(name,
                            update_type,
                            accumulate_values,
                            choice_num=total)
            result.update(value)
        else:
            result = Result(name, update_type, accumulate_values)
            result.update(value, total)
        return result

    @property
    def type_name(self) -> str:
        """
        Get the Result type name.

        Returns
        -------
        type_name : str
            The result type string (SUMTYPE, RATIOTYPE, MISCTYPE or
            CHOICETYPE).
        """
        return Result._all_types[self._update_type_code]

    @property
    def type_code(self) -> ResultType:
        """
        Get the Result type.

        Returns
        -------
        type_code : int
            The returned value is a number corresponding to one of the
            types SUMTYPE, RATIOTYPE, MISCTYPE or CHOICETYPE.
        """
        return self._update_type_code

    def __repr__(self) -> str:
        if self._update_type_code == Result.RATIOTYPE:
            v = self._value
            t = self._total
            if t != 0:
                return "Result -> {0}: {1}/{2} -> {3}".format(
                    self.name, v, t, v / t)
            return "Result -> {0}: {1}/{2} -> NaN".format(self.name, v, t)

        return "Result -> {0}: {1}".format(self.name, self.get_result())

    def update(self, value: Any, total: Optional[Any] = None) -> None:
        """
        Update the current value.

        Parameters
        ----------
        value : anything, but usually a number
            Value to be added to (or replaced) the current value

        total : same type as `value`
            Value to be added to the current total (only useful for the
            RATIOTYPE update type)

        Notes
        -----
        The way how this update process depends on the Result type and is
        described below

        - RATIOTYPE: Add "value" to current value and "total" to current
          total.
        - SUMTYPE: Add "value" to current value. "total" is ignored.
        - MISCTYPE: Replace the current value with "value".
        - CHOICETYPE: Update the choice "value" and the total by 1.

        See also
        --------
        create
        """
        self.num_updates += 1

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Python does not have a switch statement. We use dictionaries as
        # the equivalent of a switch statement.
        # First we define a function for each possibility.
        def __default_update(
                *_: Any) -> None:  # "*_" denotes the two unused args here
            """Default update method.

            This will only be called when the update type is not one of the
            available types. Thus, an exception will be raised.

            """
            msg = "Can't update a Result object of type '{0}'"
            raise ValueError(msg.format(self._update_type_code))

        def __update_SUMTYPE_value(p_value: Any, _: Any) -> None:
            """Update the Result object when its type is SUMTYPE."""
            self._value += p_value
            self._result_sum += p_value
            self._result_squared_sum += p_value**2
            if self._accumulate_values_bool is True:
                self._value_list.append(p_value)

        def __update_RATIOTYPE_value(p_value: Any, p_total: Any) -> None:
            """Update the Result object when its type is RATIOTYPE.

            Raises
            ------
            ValueError
                If the `p_total` parameter is None (not provided).
            """
            if p_total is None:
                msg = ("A 'p_value' and a 'p_total' are required when "
                       "updating a Result object of the RATIOTYPE type.")
                raise ValueError(msg)

            self._value += p_value
            self._total += p_total

            result = p_value / p_total
            self._result_sum += result
            self._result_squared_sum += result**2

            if self._accumulate_values_bool is True:
                self._value_list.append(p_value)
                self._total_list.append(p_total)

        def __update_by_replacing_current_value(p_value: Any, _: Any) -> None:
            """Update the Result object when its type is MISCTYPE."""
            self._value = p_value
            if self._accumulate_values_bool is True:
                self._value_list.append(p_value)

        def __update_CHOICETYPE_value(p_value: Any, _: Any) -> None:
            """Update the Result object when its type is CHOICETYPE."""
            # The provided 'p_value' is used as an index to increase the
            # choice in self._value, which is stored as a numpy array.
            assert isinstance(
                p_value,
                (int, np.int, np.int32,
                 np.int64)), ("Value for the CHOICETYPE must be an integer.")

            self._value[p_value] += 1
            self._total += 1
            if self._accumulate_values_bool is True:
                self._value_list.append(p_value)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Now we fill the dictionary with the functions
        possible_updates = {
            Result.RATIOTYPE: __update_RATIOTYPE_value,
            Result.MISCTYPE: __update_by_replacing_current_value,
            Result.SUMTYPE: __update_SUMTYPE_value,
            Result.CHOICETYPE: __update_CHOICETYPE_value
        }

        # Call the appropriated update method. If self._update_type_code
        #  does not contain a key in the possible_updates dictionary (
        # that is, a valid update type), then the function
        # __default_update is called.
        possible_updates.get(self._update_type_code, __default_update)(value,
                                                                       total)

    def merge(self, other: "Result") -> None:
        """
        Merge the result from other with self.

        Parameters
        ----------
        other : Result
            Another Result object.
        """
        assert (isinstance(other, self.__class__))
        # pylint: disable=W0212
        assert self._update_type_code == other._update_type_code, (
            "Can only merge two objects with the same name and type")
        assert self._update_type_code != Result.MISCTYPE, (
            "Cannot merge results of the MISCTYPE type")
        assert self.name == other.name, (
            "Can only merge two objects with the same name and type")

        if self.accumulate_values_bool is True:
            # The second object must also have been set to accumulate
            # values
            msg = ("The merged Result also must have been set to "
                   "accumulate values.")
            assert other.accumulate_values_bool is True, msg

            self._value_list.extend(other._value_list)
            self._total_list.extend(other._total_list)

        self.num_updates += other.num_updates
        self._value += other._value
        self._total += other._total
        self._result_sum += other._result_sum
        self._result_squared_sum += other._result_squared_sum

    def get_result(self) -> Any:
        """
        Get the result stored in the Result object.

        Returns
        -------
        results : anything, but usually a number
            For the RATIOTYPE type get_result will return the
            `value/total`, while for the other types it will return
            `value`.
        """
        if self.num_updates == 0:
            return "Nothing yet"

        if self._update_type_code == Result.RATIOTYPE:
            return self._value / self._total

        if self._update_type_code == Result.CHOICETYPE:
            return self._value / self._total

        return self._value

    def get_result_accumulated_values(self) -> List[Any]:  # pragma: no cover
        """
        Return the accumulated values.

        Note that in case the result if of type RATIOTYPE this you probably
        want to call the get_result_accumulated_totals function to also get
        the totals.
        """
        return self._value_list

    def get_result_accumulated_totals(self) -> List[Any]:  # pragma: no cover
        """
        Return the accumulated values.

        Note that in case the result if of type RATIOTYPE this you probably
        want to call the get_result_accumulated_values function to also get
        the values.
        """
        return self._total_list

    def get_result_mean(self) -> float:
        """Get the mean of all the updated results.

        Returns
        -------
        float
            The mean of the result.
        """
        # self._fix_old_version()  # Remove this line in the future

        return self._result_sum / self.num_updates

    def get_result_var(self) -> float:
        """
        Get the variance of all updated results.

        Returns
        -------
        float
            The variance of the results.
        """
        # self._fix_old_version()  # Remove this line in the future

        return ((self._result_squared_sum / self.num_updates) -
                (self.get_result_mean())**2)

    def get_confidence_interval(self, P: float = 95.0) -> np.ndarray:
        """
        Get the confidence interval that contains the true result with a
        given probability `P`.

        Parameters
        ----------
        P : float
            The desired confidence (probability in %) that true value is
            inside the calculated interval. The possible values are
            described in the documentation of the
            :func:`.calc_confidence_interval` function`

        Returns
        -------
        Interval : np.ndarray
            Numpy (float) array with two elements.

        See also
        --------
        .calc_confidence_interval
        """
        if self._update_type_code == Result.MISCTYPE:
            message = ("Calling get_confidence_interval is not valid for "
                       "the MISC update type.")
            raise RuntimeError(message)

        mean = self.get_result_mean()
        std = np.sqrt(self.get_result_var())
        n = self.num_updates
        return calc_confidence_interval(mean, std, n, P)

    # Overwrite version in  JsonSerializable
    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert the Result object to a dictionary representation.

        Returns
        -------
        dict
            The dictionary representation of the object.
        """
        d = {
            'name': self.name,
            'update_type_code': self._update_type_code,
            'value': self._value,
            'total': self._total,
            'result_sum': self._result_sum,
            'result_squared_sum': self._result_squared_sum,
            'num_updates': self.num_updates,
            'accumulate_values_bool': self._accumulate_values_bool,
            'value_list': self._value_list,
            'total_list': self._total_list
        }
        return d

    @staticmethod
    def _from_dict(d: Dict[str, Any]) -> "Result":
        """
        Convert from a dictionary to a Result object.

        Parameters
        ----------
        d : dict
            The dictionary representing the Result.

        Returns
        -------
        Result
            The converted object.
        """
        if isinstance(d['value'], Iterable) and \
           d['update_type_code'] == Result.CHOICETYPE:

            values = d['value']

            r = Result(name=d['name'],
                       update_type_code=d['update_type_code'],
                       accumulate_values=d['accumulate_values_bool'],
                       choice_num=len(values))

            for i, v in enumerate(values):
                for _ in range(v):
                    r.update(i)

        else:
            r = Result.create(name=d['name'],
                              update_type=d['update_type_code'],
                              value=d['value'],
                              total=d['total'],
                              accumulate_values=d['accumulate_values_bool'])
            r._value_list = d['value_list']
            r._total_list = d['total_list']
            r.num_updates = d['num_updates']
            r._result_sum = d['result_sum']
            r._result_squared_sum = d['result_squared_sum']
        return r


# xxxxxxxxxx Result - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationResults - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SimulationResults(JsonSerializable):
    """Store results from simulations.

    This class is used in the :class:`.SimulationRunner` class in order to store
    results from a simulation. It is able to combine the results from
    multiple iterations (of the :meth:`.SimulationRunner._run_simulation`
    method in the :class:`SimulationRunner` class) as well as append
    results for different simulation parameters configurations.

    .. note::

       Each result stored in the :class:`SimulationResults` object is in
       fact an object of the :class:`Result` class. This is required so
       that multiple :class:`SimulationResults` objects can be merged
       together, since the logic to merge each individual result is in the
       the :class:`Result` class.

    Examples
    --------
    - Creating a SimulationResults object and adding a few results to it

      .. code-block:: python

         result1 = Result.create(...)  # See the Result class for details
         result2 = Result.create(...)
         result3 = Result.create(...)
         simresults = SimulationResults()
         simresults.add_result(result1)
         simresults.add_result(result2)
         simresults.add_result(result3)

      Instead of explicitly create a Result object and add it to the
      SimulationResults object, we can also create the Result object
      on-the-fly when adding it to the SimulationResults object by using
      the :meth:`add_new_result` method.

      That is

      .. code-block:: python

         simresults = SimulationResults()
         simresults.add_new_result(...)
         simresults.add_new_result(...)
         simresults.add_new_result(...)

    - Merging multiple SimulationResults objects

      .. code-block:: python

         # First SimulationResults object
         simresults = SimulationResults()
         # Create a Result object
         result = Result.create('some_name', Result.SUMTYPE, 4)
         # and add it to the SimulationResults object.
         simresults.add_result(result)

         # Second SimulationResults object
         simresults2 = SimulationResults()
         # We can also create the Result object on-the-fly when adding it
         # to the SimulationResults object to save one line.
         simresults2.add_new_result('some_name', Result.SUMTYPE, 6)

         # We can merge the results in the second SimulationResults object.
         # Since the update type of the single result stored is SUMTYPE,
         # then the simresults will now have a single Result of SUMTYPE
         # type with a value of 10.
         simresults.merge_all_results(simresults)

    See Also
    --------
    .runner.SimulationRunner : Base class to implement Monte Carlo simulations.
    .parameters.SimulationParameters : Class to store the simulation parameters.
    Result : Class to store a single simulation result.

    """
    def __init__(self) -> None:
        super().__init__()
        self._results: Dict[str, List[Result]] = dict()

        # This will store the simulation parameters used in the simulation
        # that resulted in the results. This should be set by calling the
        # set_parameters method.
        self._params = SimulationParameters()

        # Don't change this manually. This will be set in the
        # SimulationRunner class in the end of the simulation.
        self.runned_reps: Optional[int] = None

        # When the SimulationResults object is saved to a file with the method
        # 'save_to_file', this variable will be set to the used filename
        # (before any string replacements). This is useful when this file is
        # loaded to recover the SimulationResults object.
        self.original_filename: Optional[str] = None

        # The SimulationResults will set and retrieve this value
        self.current_rep = -1

    def __eq__(self, other: Any) -> bool:
        """
        Compare two SimulationResults objects.

        Two SimulationResults objects are considered equal if all Result
        objects in both of them are equal, with the exception of the
        'elapsed_time' Result, which is ignored in the comparison.

        Note that the 'original_filename' variable is also ignored in the
        comparison.

        Parameters
        ----------
        other : w
            The other SimulationResults object.

        Returns
        -------
        bool
            True if `other` is equal to `self`, False otherwise.
        """
        if self is other:  # pragma: no cover
            return True

        if not isinstance(other, self.__class__):
            return False

        aux = equal_dicts(
            self.__dict__,
            other.__dict__,
            ignore_keys=['elapsed_time', '_results', 'original_filename'])

        if aux is False:
            return False

        # pylint: disable=W0212
        if self._results.keys() != other._results.keys():
            return False

        return all([
            self[k] == other[k] for k in self._results.keys()
            if k != 'elapsed_time'
        ])

    def __ne__(self, other: Any) -> bool:
        """
        Compare two SimulationResults objects.

        Two SimulationResults objects are considered equal if all Result
        objects in both of them are equal, with the exception of the
        'elapsed_time' Result, which is ignored in the comparison.

        Parameters
        ----------
        other : SimulationResults
            The other SimulationResults object.

        Returns
        -------
        bool
            False if `other` is equal to `self`, True otherwise.
        """
        return not self.__eq__(other)

    @property
    def params(self) -> SimulationParameters:
        """Get method for the params property."""
        return self._params

    def set_parameters(self, params: SimulationParameters) -> None:
        """
        Set the parameters of the simulation used to generate the
        simulation results stored in the SimulationResults object.

        Parameters
        ----------
        params : SimulationParameters
            A SimulationParameters object containing the simulation
            parameters.

        """
        if not isinstance(params, SimulationParameters):
            raise ValueError('params must be a SimulationParameters object')
        self._params = params

    def __repr__(self) -> str:
        """
        String representation of the SimulationResults object.

         Returns
         -------
         str
            The string representation of the SimulationResults object.
         """
        list_of_names = self._results.keys()
        repr_string = "SimulationResults: {0}".format(sorted(list_of_names))
        return repr_string

    def add_result(self, result: Result) -> None:
        """
        Add a result object to the SimulationResults object.

        .. note::

           If there is already a result stored with the same name, this
           will replace it.

        Parameters
        ----------
        result : Result
            The Result object to add to the simulation results.
        """
        # Added as a list with a single element
        self._results[result.name] = [result]

    def add_new_result(self,
                       name: str,
                       update_type: ResultType,
                       value: Any,
                       total: Any = 0) -> None:
        """Create a new Result object on the fly and add it to the
        SimulationResults object.

        .. note::

           This is Equivalent to the code below,

           .. code-block:: python

              result = Result.create(name, update_type, value, total)
              self.add_result(result)

           which in fact is exactly how this method is implemented.

        Parameters
        ----------
        name : str
            Name of the Result.
        update_type : int
            Type of the result (SUMTYPE, RATIOTYPE, MISCTYPE or
            CHOICETYPE).
        value : any
            Value of the result.
        total : any | int
            Total value of the result (used only for the RATIOTYPE and
            ignored for the other types).
        """
        result = Result.create(name, update_type, value, total)
        self.add_result(result)

    def append_result(self, result: Result) -> None:
        """
        Append a result to the SimulationResults object.

        This effectively means that the SimulationResults object will
        now store a list for the given result name. This allow you,
        for instance, to store multiple bit error rates with the 'BER'
        name such that simulation_results_object['BER'] will return a
        list with the Result objects for each value.

        Parameters
        ----------
        result : Result
            The Result object to append to the simulation results.

        Notes
        -----
        If multiple values for some Result are stored, then only the last
        value can be updated with :meth:`merge_all_results`.

        Raises
        ------
        ValueError
            If the `result` has a different type from the result previously
            stored.

        See also
        --------
        append_all_results, merge_all_results
        """
        if result.name in self._results.keys():
            update_type_code = self._results[result.name][0].type_code
            if update_type_code == result.type_code:
                self._results[result.name].append(result)
            else:
                raise ValueError("Can only append to results of the same type")
        else:
            self.add_result(result)

    def append_all_results(self, other: "SimulationResults") -> None:
        """
        Append all the results of the other SimulationResults object
        with self.

        Parameters
        ----------
        other : SimulationResults
            Another SimulationResults object

        See also
        --------
        append_result, merge_all_results
        """
        for results in other:
            # There can be more then one value for the same result name
            for result in results:
                self.append_result(result)

    def merge_all_results(self, other: "SimulationResults") -> None:
        """
        Merge all the results of the other SimulationResults object with the
        results in self.

        When there is more then one result with the same name stored in
        self (for instance two bit error rates -> for different parameters)
        then only the last one will be merged with the one in `other`. That
        also means that only one result for that name should be stored in
        `other`.

        Parameters
        ----------
        other : SimulationResults
            Another SimulationResults object

        See also
        --------
        append_result, append_all_results

        Notes
        -----
        This method is used in the SimulationRunner class to combine
        results of two simulations for the exact same parameters.
        """
        # If the current SimulationResults object is empty, we basically
        # copy the Result objects from other
        if len(self) == 0:
            for name in other.get_result_names():
                self._results[name] = other[name]
        # Otherwise, we merge each Result from `self` with the Result from
        # `other`
        else:
            for item in self.get_result_names():
                # The 'num_skipped_reps' result is different from the other
                # results in the sense that it is created by the
                # SimulationRunner class to count how many times a
                # SkipThisOne exception is raised. It is not created at the
                # same time as the other Result objects, but we want to
                # allow merging two SimulationResults objects even if one
                # of them does not have a 'num_skipped_reps' Result object.
                if item != 'num_skipped_reps':
                    self._results[item][-1].merge(other[item][-1])

            # Merge the 'num_skipped_reps' Result if the second object has
            # it.
            if 'num_skipped_reps' in other.get_result_names():
                # It the second SimulationResults has the the
                # 'num_skipped_reps' Result, but the first one has not,
                # then first we create a 'num_skipped_reps' Result for the
                # first SimulationResults object.
                if 'num_skipped_reps' not in self.get_result_names():
                    self.add_new_result('num_skipped_reps', Result.SUMTYPE, 0)

                # Now we merge 'num_skipped_reps' from both of them
                self._results['num_skipped_reps'][-1].merge(
                    other['num_skipped_reps'][-1])

    def get_result_names(self) -> List[str]:
        """
        Get the names of all results stored in the SimulationResults
        object.

        Returns
        -------
        names : list[str]
            The names of the results stored in the SimulationResults object.
        """
        return list(self._results.keys())

    def get_result_values_list(
            self,
            result_name: str,
            fixed_params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Get the values for the results with name `result_name`.

        Returns a list with the values.

        Parameters
        ----------
        result_name : str
            The name of the desired result.
        fixed_params : dict
            A python dictionary containing the fixed parameters. If
            `fixed_params` is provided then the returned list will be only
            a subset of the results that match the fixed values of the
            parameters in the `fixed_params` argument, where the key is the
            parameter's name and the value is the fixed value. See the
            notes for an example.

        Returns
        -------
        result_list : List
            A list with the stored values for the result with name
            `result_name`

        Notes
        -----
        As an example of the usage of the `fixed_params` argument, suppose
        the results where obtained in a simulation for three parameters:
        'first', with value 'A', 'second' with value '[1, 2, 3]' and
        'third' with value '[B, C]', where the 'second' and 'third' were
        set to be unpacked. In that case the returned result list would
        have a length of 6 (the number of possible combinations of the
        parameters to be unpacked). If fixed_params is provided with the
        value of "{'second': 2}" that means that only the subset of results
        which corresponding to the second parameters having the value of
        '2' will be provided and the returned list will have a length of
        2. If fixed_params is provided with the value of "{'second': '1',
        'third': 'C'}" then a single result will be provided instead of a
        list.
        """
        if fixed_params is None:
            fixed_params = {}

        # If the dictionary is not empty
        if fixed_params:
            indexes = self.params.get_pack_indexes(fixed_params)
            out = [
                v.get_result() for i, v in enumerate(self[result_name])
                if i in indexes
            ]
        else:
            # If fixed_params is an empty dictionary (default value) then
            # we return the full list of results
            out = [v.get_result() for v in self[result_name]]
        return out

    def get_result_values_confidence_intervals(
            self,
            result_name: str,
            P: float = 95.0,
            fixed_params: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """
        Get the values for the results with name `result_name`.

        This method is similar to the `get_result_values_list` method, but
        instead of returning a list with the values it will return a list
        with the confidence intervals for those values.

        Parameters
        ----------
        result_name : str
            The name of the desired result.
        P : float
        fixed_params : dict
            A python dictionary containing the fixed parameters. If
            `fixed_params` is provided then the returned list will be only
            a subset of the results that match the fixed values of the
            parameters in the `fixed_params` argument, where the key is the
            parameter's name and the value is the fixed value. See the
            notes in the documentation of :meth:`get_result_values_list`
            for an example.

        Returns
        -------
        confidence_interval_list : list[np.ndarray]
            A list of Numpy (float) arrays. Each element in the list is an
            array with two elements, corresponding to the lower and upper
            limits of the confidence interval.8

        See also
        --------
        .calc_confidence_interval
        """
        if fixed_params is None:
            fixed_params = {}

        if fixed_params:
            indexes = self.params.get_pack_indexes(fixed_params)

            # If indexes is just an integer, make it an iterable
            if not isinstance(indexes, Iterable):
                indexes = [indexes]

            out = [
                v.get_confidence_interval(P)
                for i, v in enumerate(self[result_name]) if i in indexes
            ]
        else:
            # If fixed_params is an empty dictionary (default value) then
            # we return the full list of results
            out = [i.get_confidence_interval(P) for i in self[result_name]]
        return out

    def __getitem__(self, key: str) -> List[Result]:
        """
        Get the value of the desired result.

        Parameters
        ----------
        key : str
            Name of the desired result.

        Returns
        -------
        List[Result]
            The desired results.
        """
        # if key in self._results.keys():
        return self._results[key]
        # else:
        #     raise KeyError("Invalid key: %s" % key)

    def __len__(self) -> int:
        """Get the number of results stored in self.

        Returns
        -------
        length : int
            Number of results stored in self.
        """
        return len(self._results)

    def __iter__(self) -> Iterator[Result]:  # pragma: no cover
        # """Get an iterator to the internal dictionary. Therefore iterating
        # through this will iterate through the dictionary keys, that is, the
        # name of the results stored in the SimulationResults object.
        # """
        """
        Get an iterator to the results stored in the SimulationResults
        object.
        """
        return iter(self._results.values())

    def get_filename_with_replaced_params(self, filename: str) -> str:
        """
        Perform the string replacements in filename with simulation parameters.

        Parameters
        ----------
        filename : str
            Name of the file to save the results. This can have string
            placements for replacements of simulation parameters. For
            instance, is `filename` is "somename_{age}.pickle" and the
            value of an 'age' parameter is '3', then the actual name used
            to save the file will be "somename_3.pickle"

        Returns
        -------
        string
            The name of the file where the results were saved. This will be
            equivalent to `filename` after string replacements (with the
            simulation parameters) are done.
        """
        # If filename has some replacements that cannot be done, which
        # would raise an exception, then we will save to the filename
        # without string replacements so that at least we don't lose
        # simulation results.
        try:
            filename = replace_dict_values(filename, self.params.parameters,
                                           True)
        except KeyError:  # pragma: nocover
            pass
        return filename

    # noinspection PyMethodMayBeStatic
    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert the SimulationResults object to a dictionary representation.

        Returns
        -------
        dict
            The dictionary representation of the SimulationResults object.
        """

        # -----------------------------------------------------------------
        def list_of_results_to_list_of_dicts(
                result_list: List[Result]) -> List[Dict[str, Any]]:
            """
            Convert a list of Result objects into a list of dictionary
            representations ob Result objects.

            Parameters
            ----------
            result_list : list[Result]
                List of Result objects.

            Returns
            -------
            list[dict]
                List of dictionary representations of the Result objects.
            """
            out = [r.to_dict() for r in result_list]
            return out

        # -----------------------------------------------------------------

        results = {
            n: list_of_results_to_list_of_dicts(v)
            for n, v in self._results.items()
        }

        d = {
            'params': self._params.to_dict(),
            'runned_reps': self.runned_reps,
            'original_filename': self.original_filename,
            'results': results
        }

        return d

    @staticmethod
    def _from_dict(d: Dict[str, Any]) -> "SimulationResults":
        """
        Convert from a dictionary to a SimulationResults object.

        Parameters
        ----------
        d : dict
            The dictionary representing the SimulationResults.

        Returns
        -------
        SimulationResults
            The converted object.
        """
        def list_of_dicts_to_list_of_results(
                result_list: Dict[str, Any]) -> List[Result]:
            """
            Convert a list of dictionary representations of Result objects to a
            list of Result objects.

            Parameters
            ----------
            result_list : list[dict]
                List of dictionary representations of the Result objects.

            Returns
            -------
            List[Result]
                List of Result objects.
            """
            out = [Result.from_dict(r) for r in result_list]
            return out

        results = {
            n: list_of_dicts_to_list_of_results(v)
            for n, v in d['results'].items()
        }

        simresults = SimulationResults()
        simresults._params = SimulationParameters.from_dict(d['params'])
        simresults.runned_reps = d['runned_reps']
        simresults.original_filename = d['original_filename']
        simresults._results = results

        return simresults

    def _save_to_pickle(self, filename: str) -> None:
        """
        Save the SimulationResults object to the pickle file with name
        `filename`.

        Parameters
        ----------
        filename : src
            Name of the file to save the SimulationResults object.
        """
        # For python3 compatibility the file must be opened in binary mode
        with open(filename, 'wb') as output:
            # We use the protocol version 2, since it is the highest
            # protocol that is supported by both python 2 and python
            # 3. Note that we still need to be careful when unpickling,
            # since a file pickled with python 2 might raise a
            # UnicodeDecodeError exception when unpickled with python 3. We
            # solve this in the `load_from_config_file` method by
            # specifying the encoding when unpickling the file.
            pickle.dump(self, output, protocol=2)

    def _save_to_json(self, filename: str) -> None:
        """
        Save the SimulationResults object to the json file with name
        `filename`.

        Parameters
        ----------
        filename : src
            Name of the file to save the SimulationResults object.
        """
        with open(filename, 'w') as output:
            output.write(self.to_json())

    def save_to_file(self, filename: str) -> str:
        """
        Save the SimulationResults to the file `filename`.

        The string in `filename` can have placeholders for string
        replacements with any parameter value.

        Parameters
        ----------
        filename : src
            Name of the file to save the results. This can have string
            placements for replacements of simulation parameters. For
            instance, is `filename` is "somename_{age}.pickle" and the
            value of an 'age' parameter is '3', then the actual name used
            to save the file will be "somename_3.pickle"

        Returns
        -------
        string
            The name of the file where the results were saved. This will be
            equivalent to `filename` after string replacements (with the
            simulation parameters) are done.
        """
        # Get the file extension (if there is any). If it is not equal to
        # '.pickle' that means we need to add the '.pickle' extension.
        ext = os.path.splitext(filename)[-1]
        if ext == '':
            filename = '{0}.pickle'.format(filename)
            ext = '.pickle'

        # Save the original filename before string replacements
        self.original_filename = filename

        # To get the actual filename we perform the parameter replacements
        filename = self.get_filename_with_replaced_params(filename)

        # xxxxxxxxxx Finally save to the appropriated file xxxxxxxxxxxxxxxx
        ext_to_save_func_mapping = {
            '.pickle': self._save_to_pickle,
            '.json': self._save_to_json
        }
        save_func = ext_to_save_func_mapping[ext]

        # Save the SimulationResults to the file with the desired format
        save_func(filename)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return filename

    @staticmethod
    def _load_from_pickle_file(filename: str) -> "SimulationResults":
        with open(filename, 'rb') as inputfile:
            try:
                obj = pickle.load(inputfile)
            except UnicodeDecodeError:  # pragma: nocover
                # If we pickle with python 2 and try to unpickle with
                # python 3 we might get a UnicodeDecodeError exception. In
                # that case, let's try to unpickle specifying the
                # 'iso-8859-1' encoding to see if it works.
                obj = pickle.load(inputfile, encoding='iso-8859-1')
            # except ValueError:
            #     # If we pickle with python 3 and try to unpickle with
            #     # python 2 we might get a ValueError (due to unsupported
            #     # pickle protocol).
            #     #
            #     # By raising an IOError (the same exception raised by
            #     # python when the file does not exist this will be
            #     # interpreted as "there is no partial file". This will then
            #     # overwrite the old partial result file in the first time
            #     # partial results are saved.
            #     raise IOError("Could not unpickle file '{0}'".format(
            #         filename))

        return obj

    @staticmethod
    def _load_from_json_file(filename: str) -> "SimulationResults":
        with open(filename, 'r') as inputfile:
            json_data = inputfile.read()
        obj = SimulationResults.from_json(json_data)
        return obj

    @staticmethod
    def load_from_file(filename: str) -> "SimulationResults":
        """
        Load the SimulationResults from the file `filename`.

        Parameters
        ----------
        filename : src
            Name of the file from where the results will be loaded.

        Returns
        -------
        SimulationResults
            The SimulationResults object loaded from the file `filename`.
        """
        ext = os.path.splitext(filename)[-1]
        if ext == '':
            filename = '{0}.pickle'.format(filename)
            ext = '.pickle'

        ext_to_load_func_mapping = {
            '.pickle': SimulationResults._load_from_pickle_file,
            '.json': SimulationResults._load_from_json_file
        }
        load_func = ext_to_load_func_mapping[ext]

        return load_func(filename)

    def to_dataframe(self) -> DataFrame:
        """
        Convert the SimulationResults object to a pandas DataFrame.
        """
        # The data dictionary that we will use to create the DataFrame
        data = {}
        all_params_list = self.params.get_unpacked_params_list()
        for name in self.params:
            data[name] = [a[name] for a in all_params_list]

        for res in self:
            name = res[0].name
            data[name] = [r.get_result() for r in res]

        try:
            data['runned_reps'] = self.runned_reps
        except AttributeError:  # pragma: no cover
            pass

        df = pd.DataFrame(data)
        return df

    def _repr_html_(self):
        """
        Method used by the jupyter rich display.

        We will simple use the HTML representation of the corresponding pandas
        dataframe.
        """
        return self.to_dataframe()._repr_html_()


# xxxxxxxxxx SimulationResults - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
