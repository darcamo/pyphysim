#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing simulation result classes."""

__revision__ = "$Revision$"


import numpy as np

from .parameters import SimulationParameters
from ..util.misc import calc_confidence_interval, equal_dicts

try:
    import cPickle as pickle
except ImportError as e:  # pragma: no cover
    import pickle

try:
    import tables as tb
except ImportError as e:
    pass

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationResults - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SimulationResults(object):
    """Store results from simulations.

    This class is used in the SimulationRunner class in order to store
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
    - Creating a SimulationResults onject and adding a few results to it

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
    SimulationRunner : Base class to implement Monte Carlo simulations.
    SimulationParameters : Class to store the simulation parameters.
    Result : Class to store a single simulation result.

    """
    def __init__(self):
        self._results = dict()

        # This will store the simulation parameters used in the simulation
        # that resulted in the results. This should be set by calling the
        # set_parameters method.
        self._params = SimulationParameters()

    def __eq__(self, other):
        """
        Compare two SimulationResults objects.

        Two SimulationResults objects are considered equal if all Result
        objects in both of them are equal, with the exception of the
        'elapsed_time' Result, which is ignored in the comparison.

        Parameters
        ----------
        other : SimulationResults object
            The other SimulationResults object.
        """
        if self is other:  # pragma: no cover
            return True

        aux = equal_dicts(self.__dict__, other.__dict__, ignore_keys=['elapsed_time', '_results'])
        if aux is False:
            return False

        # pylint: disable=W0212
        if self._results.keys() != other._results.keys():
            return False

        return all([self[k] == other[k] for k in self._results.keys() if k != 'elapsed_time'])

    def __ne__(self, other):
        """
        Compare two SimulationResults objects.

        Two SimulationResults objects are considered equal if all Result
        objects in both of them are equal, with the exception of the
        'elapsed_time' Result, which is ignored in the comparison.

        Parameters
        ----------
        other : SimulationResults object
            The other SimulationResults object.
        """
        return not self.__eq__(other)

    def _get_params(self):
        """Get method for the params property."""
        return self._params
    params = property(_get_params)

    def set_parameters(self, params):
        """Set the parameters of the simulation used to generate the
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

    def __repr__(self):
        list_of_names = self._results.keys()
        repr_string = "SimulationResults: {0}".format(sorted(list_of_names))
        return repr_string

    def add_result(self, result):
        """Add a result object to the SimulationResults object.

        .. note::

           If there is already a result stored with the same name, this
           will replace it.

        Parameters
        ----------
        result : An object of the :class:`Result` class
            Must be an object of the Result class.

        """
        # Added as a list with a single element
        self._results[result.name] = [result]

    def add_new_result(self, name, update_type, value, total=0):
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
        update_type : {SUMTYPE, RATIOTYPE, MISCTYPE}
            Type of the result (SUMTYPE, RATIOTYPE or MISCTYPE).
        value : anything, but usually a number
            Value of the result.
        total : same type as `value`
            Total value of the result (used only for the RATIOTYPE and
            ignored for the other types).

        """
        result = Result.create(name, update_type, value, total)
        self.add_result(result)

    def append_result(self, result):
        """Append a result to the SimulationResults object. This
        efectivelly means that the SimulationResults object will now store
        a list for the given result name. This allow you, for instance, to
        store multiple bit error rates with the 'BER' name such that
        simulation_results_object['BER'] will return a list with the Result
        objects for each value.

        Parameters
        ----------
        result : An object of the :class:`Result` class
            Must be an object of the Result class.

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

    def append_all_results(self, other):
        """Append all the results of the other SimulationResults object
        with self.

        Parameters
        ----------
        other : An object of the :class:`SimulationResults` class.
            Another SimulationResults object

        See also
        --------
        append_result, merge_all_results
        """
        for results in other:
            # There can be more then one value for the same result name
            for result in results:
                self.append_result(result)

    def merge_all_results(self, other):
        """Merge all the results of the other SimulationResults object with
        the results in self.

        When there is more then one result with the same name stored in
        self (for instance two bit error rates) then only the last one will
        be merged with the one in `other`. That also means that only one
        result for that name should be stored in `other`.

        Parameters
        ----------
        other : An object of the :class:`SimulationResults` class.
            Another SimulationResults object

        See also
        --------
        append_result, append_all_results

        """
        # If the current SimulationResults object is empty, we basically
        # copy the Result objects from other
        if len(self) == 0:
            for name in other.get_result_names():
                self._results[name] = other[name]
        # Otherwise, we merge each Result from `self` with the Result from
        # `other`
        else:
            for item in self._results.keys():
                self._results[item][-1].merge(other[item][-1])

    def get_result_names(self):
        """Get the names of all results stored in the SimulationResults
        object.

        Returns
        -------
        names : list
            The names of the results stored in the SimulationResults object.
        """
        return self._results.keys()

    def get_result_values_list(self, result_name, fixed_params=None):
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
        result_list : list
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

        # If the fictionary is not empty
        if fixed_params:
            # TODO: Test this part
            indexes = self.params.get_pack_indexes(fixed_params)
            out = [v.get_result() for i, v in enumerate(self[result_name])
                   if i in indexes]
        else:
            # If fixed_params is an empty dictionary (default value) then
            # we return the full list of results
            out = [v.get_result() for v in self[result_name]]
        return out

    def get_result_values_confidence_intervals(self,
                                               result_name,
                                               P=95,
                                               fixed_params=None):
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
        confidence_interval_list : list
            A list of Numpy (float) arrays. Each element in the list is an
            array with two elements, corresponding to the lower and upper
            limits of the confidence interval.8

        See also
        --------
        util.misc.calc_confidence_interval
        """
        if fixed_params is None:
            fixed_params = {}

        if fixed_params:
            # TODO: Test this part
            indexes = self.params.get_pack_indexes(fixed_params)
            out = [v.get_confidence_interval(P) for i, v in enumerate(self[result_name])
                   if i in indexes]
        else:
            # If fixed_params is an empty dictionary (default value) then
            # we return the full list of results
            out = [i.get_confidence_interval(P) for i in self[result_name]]
        return out

    def __getitem__(self, key):
        """Get the value of the desired result

        Parameters
        ----------
        key : str
            Name of the desired result.

        Returns
        -------
        value :
            The desired result.
        """

        # if key in self._results.keys():
        return self._results[key]
        # else:
        #     raise KeyError("Invalid key: %s" % key)

    def __len__(self):
        """Get the number of results stored in self.

        Returns
        -------
        length : int
            Number of results stored in self.
        """
        return len(self._results)

    def __iter__(self):  # pragma: no cover
        # """Get an iterator to the internal dictionary. Therefore iterating
        # through this will iterate through the dictionary keys, that is, the
        # name of the results stored in the SimulationResults object.
        # """
        """Get an iterator to the results stored in the SimulationResults
        object.
        """
        try:
            # This is for python 2
            iterator = self._results.itervalues()
        except AttributeError:
            # This is for python 3
            iterator = iter(self._results.values())

        return iterator

    def save_to_file(self, filename):
        """Save the SimulationResults to the file `filename`.

        Parameters
        ----------
        filename : src
            Name of the file to save the results.

        """
        # For python3 compatibility the file must be opened in binary mode
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(filename):
        """Load the SimulationResults from the file `filename`.

        Parameters
        ----------
        filename : src
            Name of the file from where the results will be loaded.

        Returns
        -------
        simresults : A SimulationResults object
            The SimulationResults object loaded from the file `filename`.
        """
        with open(filename, 'rb') as inputfile:
            obj = pickle.load(inputfile)
        return obj

    def save_to_hdf5_file(self, filename, attrs=None):
        """
        Save the SimulationResults to the file `filename` using the HDF5 format
        standard.

        Parameters
        ----------
        filename : src
            Name of the file to save the results.
        attrs : a dictionary
            Extra attributes to add to the HDF5 file.

        See also
        --------
        load_from_hdf5_file
        """
        if attrs is None:
            attrs = {}

        import h5py
        fid = h5py.File(filename, 'w')

        # Save the TITTLE attribute to be more consistent with what
        # Pytables would do.
        fid.attrs.create("TITLE", "Simulation Results file")

        # Add the attributes, if any
        if isinstance(attrs, dict):  # pragma: no cover
            # attr is a dictionary of attributes
            for name, value in attrs.items():
                fid.attrs.create(name, value)

        # xxxxxxxxxx Save the results in the 'results' group xxxxxxxxxxxxxx
        g = fid.create_group('results')
        # Save the TITTLE attribute to be more consistent with what
        # Pytables would do.
        g.attrs.create("TITLE", "Simulation Results")
        for r in self:
            Result.save_to_hdf5_dataset(g, r)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Save the parameters in the 'parameters' group xxxxxxxx
        pg = fid.create_group('parameters')
        # Save the TITTLE attribute to be more consistent with what
        # Pytables would do.
        pg.attrs.create("TITLE", "Simulation Parameters")
        self.params.save_to_hdf5_group(pg)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        fid.close()

    # TODO: Test if this method saves all the information that the
    # save_to_hdf5_file method saves.
    def save_to_pytables_file(self, filename, attrs=None):
        """
        Save the SimulationResults to the file `filename` using pytables.
        """
        if attrs is None:
            attrs = {}

        import tables as tb
        fid = tb.openFile(filename, 'w', title='Simulation Results file')

        # Add the attributes, if any
        if isinstance(attrs, dict):  # pragma: no cover
            # attr is a dictionary of attributes
            for name, value in attrs.items():
                fid.setNodeAttr(fid.root, name, value)

        # xxxxxxxxxx Save the results in the 'results' group xxxxxxxxxxxxxx
        g = fid.createGroup(fid.root, 'results', title="Simulation Results")
        for r in self:
            Result.save_to_pytables_table(g, r)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Save the parameters in the 'parameters' group xxxxxxxx
        pg = fid.createGroup(fid.root, 'parameters', title="Simulation Parameters")
        self.params.save_to_pytables_group(pg)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        fid.close()

    @staticmethod
    def load_from_hdf5_file(filename):
        """Load a SimulationResults object from an HDF5 file saved with the
        save_to_hdf5_file method.

        Parameters
        ----------
        filename : src
            Name of the file to save the results.

        Returns
        -------
        simresults : A SimulationResults object.
            The SimulationResults object loaded from the file.

        See also
        --------
        save_to_hdf5_file
        """
        simresults = SimulationResults()

        import h5py
        fid = h5py.File(filename, 'r')

        # xxxxxxxxxx Results group xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        rg = fid['results']

        for result_name in rg:
            ds = rg[result_name]
            #simresults._results[result_name] = Result.load_from_hdf5_dataset(ds)
            result = Result.load_from_hdf5_dataset(ds)[-1]
            simresults.add_result(result)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Parameters grop xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        try:
            # We only set the simulation parameters if it was stored in the
            # hdf5 file.
            pg = fid['parameters']
            # simresults._params = SimulationParameters.load_from_hdf5_group(pg)
            simresults.set_parameters(SimulationParameters.load_from_hdf5_group(pg))
        except KeyError:  # pragma: no cover
            pass

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        fid.close()
        return simresults

# xxxxxxxxxx SimulationResults - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Result - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Result(object):
    """Class to store a single simulation result.

    A simulation result can be anything, such as the number of errors, a
    string, an error rate, etc. When creating a `Result` object one needs to
    specify only the `name` of the stored result and the result `type`.

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
    # Since we only expect to have some very specific attributes for any
    # object of the Result class, we define the attributes here. This can
    # save memory when you have a lot of Result objects.
    #__slots__ = ('name', '_update_type_code', '_value', '_total', 'num_updates', '_accumulate_values_bool', '_value_list', '_total_list', '_result_sum', '_result_squared_sum')

    # Like an Enumeration for the type of results.
    (SUMTYPE, RATIOTYPE, MISCTYPE) = range(3)
    _all_types = {
        SUMTYPE: "SUMTYPE",
        RATIOTYPE: "RATIOTYPE",
        MISCTYPE: "MISCTYPE",
    }

    def __init__(self, name, update_type_code, accumulate_values=False):
        self.name = name
        self._update_type_code = update_type_code
        self._value = 0
        self._total = 0
        self._result_sum = 0.0  # At each update the current result will be
                                # added to this variable
        self._result_squared_sum = 0.0  # At each update the square of the
                                        # current result will be added to
                                        # this variable.
        self.num_updates = 0  # Number of times the Result object was
                              # updated

        # Accumulation of values: This is useful for debugging/testing
        self._accumulate_values_bool = accumulate_values
        self._value_list = []
        self._total_list = []

    def __eq__(self, other):
        """
        Compare two Result objects.

        Two Result objects are considered equal if all attributes in both
        of them are equal, with the exception of the 'num_updates' member
        variable which is ignored in the comparison.

        Parameters
        ----------
        other : Result object
            The other Result object.
        """
        # All class attributes with the exception of num_updates
        attributes = ['name', '_update_type_code', '_value', '_total', '_accumulate_values_bool', '_value_list', '_total_list', '_result_squared_sum', '_result_sum']
        if self is other:  # pragma: no cover
            return True

        result = True
        for att in attributes:
            if getattr(self, att) != getattr(other, att):
                result = False

        return result

    def __ne__(self, other):
        """
        Compare two Result objects.

        Two Result objects are considered equal if all attributes in both
        of them are equal, with the exception of the 'num_updates' member
        variable which is ignored in the comparison.

        Parameters
        ----------
        other : Result object
            The other Result object.
        """
        return not self.__eq__(other)

    @property
    def accumulate_values_bool(self):
        """
        Property to see if values are accumulated of not during a call to the
        `update` method.
        """
        return self._accumulate_values_bool

    @staticmethod
    def create(name, update_type, value, total=0, accumulate_values=False):
        """
        Create a Result object and update it with `value` and `total` at
        the same time.

        Equivalent to creating the object and then calling its
        :meth:`update` method.

        Parameters
        ----------
        name : str
            Name of the Result.
        update_type : {SUMTYPE, RATIOTYPE, MISCTYPE}
            Type of the result (SUMTYPE, RATIOTYPE or MISCTYPE).
        value : anything, but usually a number
            Value of the result.
        total : same type as `value`
            Total value of the result (used only for the RATIOTYPE and
            ignored for the other types).
        accumulate_values : bool
            If True, then the values `value` and `total` will be
            accumulated in the `update` (and merge) method(s). This means
            that the Result object will use more memory as more and more
            values are accumulated, but having all values sometimes is
            useful to perform statistical calculations. This is useful for
            debugging/testing.

        Returns
        -------
        result : A Result object.
            The new Result object.

        Notes
        -----
        Even if accumulate_values is True the values will not be
        accumulated for the MISCTYPE.

        See also
        --------
        update
        """
        result = Result(name, update_type, accumulate_values)
        result.update(value, total)
        return result

    @property
    def type_name(self):
        """Get the Result type name.

        Returns
        -------
        type_name : str
            The result type string (SUMTYPE, RATIOTYPE or MISCTYPE).
        """
        return Result._all_types[self._update_type_code]

    @property
    def type_code(self):
        """Get the Result type.

        Returns
        -------
        type_code : int
            The returned value is a number corresponding to one of the
            types SUMTYPE, RATIOTYPE or MISCTYPE.

        """
        return self._update_type_code

    def __repr__(self):
        if self._update_type_code == Result.RATIOTYPE:
            v = self._value
            t = self._total
            if t != 0:
                return "Result -> {0}: {1}/{2} -> {3}".format(
                    self.name, v, t, float(v) / t)
            else:
                return "Result -> {0}: {1}/{2} -> NaN".format(
                    self.name, v, t)
        else:
            return "Result -> {0}: {1}".format(self.name, self.get_result())

    def update(self, value, total=None):
        """Update the current value.

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

        - RATIOTYPE: Add "value" to current value and "total" to current total
        - SUMTYPE: Add "value" to current value. "total" is ignored.
        - MISCTYPE: Replace the current value with "value".

        See also
        --------
        create

        """
        self.num_updates += 1

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Python does not have a switch statement. We use dictionaries as
        # the equivalent of a switch statement.
        # First we define a function for each possibility.
        def __default_update(dummy1, dummy2):
            """Default update method.

            This will only be called when the update type is not one of the
            available types. Thus, an exception will be raised.

            """
            raise ValueError("Can't update a Result object of type '{0}'".format(self._update_type_code))

        def __update_SUMTYPE_value(value, dummy):
            """Update the Result object when its type is SUMTYPE."""
            self._value += value
            self._result_sum += value
            self._result_squared_sum += value**2
            if self._accumulate_values_bool is True:
                self._value_list.append(value)

        def __update_RATIOTYPE_value(value, total):
            """Update the Result object when its type is RATIOTYPE.

            Raises
            ------
            ValueError
                If the `total` parameter is None (not provided).
            """
            if total is None:
                raise ValueError("A 'value' and a 'total' are required when updating a Result object of the RATIOTYPE type.")

            self._value += value
            self._total += total

            result = float(value) / float(total)
            self._result_sum += result
            self._result_squared_sum += result**2

            if self._accumulate_values_bool is True:
                self._value_list.append(value)
                self._total_list.append(total)

        def __update_by_replacing_current_value(value, dummy):
            """Update the Result object when its type is MISCTYPE."""
            self._value = value
            if self._accumulate_values_bool is True:
                self._value_list.append(value)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Now we fill the dictionary with the functions
        possible_updates = {
            Result.RATIOTYPE: __update_RATIOTYPE_value,
            Result.MISCTYPE: __update_by_replacing_current_value,
            Result.SUMTYPE: __update_SUMTYPE_value}

        # Call the apropriated update method. If self._update_type_code does
        # not contain a key in the possible_updates dictionary (that is, a
        # valid update type), then the function __default_update is called.
        possible_updates.get(self._update_type_code,
                             __default_update)(value, total)

    def merge(self, other):
        """Merge the result from other with self.

        Parameters
        ----------
        other : Result object
            Another Result object.
        """
        # pylint: disable=W0212
        assert self._update_type_code == other._update_type_code, (
            "Can only merge two objects with the same name and type")
        assert self._update_type_code != Result.MISCTYPE, (
            "Cannot merge results of the MISCTYPE type")
        assert self.name == other.name, (
            "Can only merge two objects with the same name and type")

        if self._accumulate_values_bool is True:
            # The second object must also have been set to accumulate values
            assert other.accumulate_values_bool == True, (
            "The merged Result also must have been set to accumulate values.")

            self._value_list.extend(other._value_list)
            self._total_list.extend(other._total_list)

        self.num_updates += other.num_updates
        self._value += other._value
        self._total += other._total
        self._result_sum += other._result_sum
        self._result_squared_sum += other._result_squared_sum

    def get_result(self):
        """Get the result stored in the Result object.

        Returns
        -------
        results : anything, but usually a number
            For the RATIOTYPE type get_result will return the
            `value/total`, while for the other types it will return
            `value`.

        """
        if self.num_updates == 0:
            return "Nothing yet".format(self.name)
        else:
            if self._update_type_code == Result.RATIOTYPE:
                #assert self._total != 0, 'Total should not be zero'
                return float(self._value) / self._total
            else:
                return self._value

    # # Remove this in the future
    # def _fix_old_version(self):
    #     """
    #     """
    #     # xxxxxxxxxx REMOVE THIS IN THE FUTURE - START xxxxxxxxxxxxxxxxxxxx
    #     try:
    #         self._result_sum
    #     except AttributeError as _:
    #         if self.type_code == Result.RATIOTYPE:
    #             n = np.array(self._value_list, dtype=float)
    #             d = np.array(self._total_list, dtype=float)
    #             r = n / d
    #         elif self.type_code == Result.SUMTYPE:
    #             r = np.array(self._value_list, dtype=float)
    #         self._result_sum = np.sum(r)
    #         self._result_squared_sum = np.sum(r**2)
    #     # xxxxxxxxxx REMOVE THIS IN THE FUTURE - END xxxxxxxxxxxxxxxxxxxxxx

    def get_result_mean(self):
        """Get the mean of all the updated results.
        """
        # self._fix_old_version()  # Remove this line in the future

        return float(self._result_sum) / self.num_updates

    def get_result_var(self):
        """
        Get the variance of all updated results.
        """
        # self._fix_old_version()  # Remove this line in the future

        return (float(self._result_squared_sum) / self.num_updates) - (self.get_result_mean())**2

    def get_confidence_interval(self, P=95):
        """
        Get the confidence inteval that contains the true result with a given
        probability `P`.

        Parameters
        ----------
        P : float
            The desired confidence (probability in %) that true value is
            inside the calculated interval. The possible values are
            described in the documentaiton of the
            `util.misc.calc_confidence_interval` function`

        Returns
        -------
        Interval : Numpy (float) array with two elements.

        See also
        --------
        util.misc.calc_confidence_interval
        """
        if self._update_type_code == Result.MISCTYPE:
            message = "Calling get_confidence_interval is not valid for the MISC update type."
            raise RuntimeError(message)

        mean = self.get_result_mean()
        std = np.sqrt(self.get_result_var())
        n = self.num_updates
        return calc_confidence_interval(mean, std, n, P)

    # # Remove this in the future
    # def get_confidence_interval_old(self, P=95):
    #     """
    #     Get the confidence inteval that contains the true result with a given
    #     probability `P`.

    #     Parameters
    #     ----------
    #     P : float
    #         The desired confidence (probability in %) that true value is
    #         inside the calculated interval. The possible values are
    #         described in the documentaiton of the
    #         `util.misc.calc_confidence_interval` function`

    #     Returns
    #     -------
    #     Interval : Numpy (float) array with two elements.

    #     See also
    #     --------
    #     util.misc.calc_confidence_interval
    #     """
    #     if self._update_type_code == Result.MISCTYPE:
    #         message = "Calling get_confidence_interval is not valid for the MISC update type."
    #         raise RuntimeError(message)

    #     if len(self._value_list) == 0:
    #         if self._accumulate_values_bool is False:
    #             message = "get_confidence_interval: The accumulate_values option must be set to True."
    #         else:
    #             message = "get_confidence_interval: There are no stored values yet."
    #         raise RuntimeError(message)

    #     values = np.array(self._value_list, dtype=float)
    #     if self._update_type_code == Result.RATIOTYPE:
    #         values = values / np.array(self._total_list, dtype=float)

    #     mean = values.mean()
    #     std = values.std()
    #     n = values.size

    #     return calc_confidence_interval(mean, std, n, P)

    # TODO: Save the _value_list, _total_list and _accumulate_values_bool
    # variables
    @staticmethod
    def save_to_hdf5_dataset(parent, results_list):
        """Create an HDF5 dataset in `parent` and fill it with the Result
        objects in `results_list`.

        Parameters
        ----------
        parent : An HDF5 group (usually) or file.
            The parent that will contain the dataset.
        results_list : A python list of Result objects.
            A list of Result objects. All of these objects must have the
            same name and update type.

        Notes
        -----
        This method is called from the save_to_hdf5_file method in the
        SimulationResults class. It uses the python h5py library and
        `parent` is supposed to be an HDF5 group created with that library.

        See also
        --------
        load_from_hdf5_dataset

        """
        dtype = [('_value', float), ('_total', float), ('num_updates', int)]
        name = results_list[0].name
        size = len(results_list)
        ds = parent.create_dataset(name, shape=(size,), dtype=dtype)

        r = None
        for i, r in enumerate(results_list):
            # pylint: disable=W0212
            ds[i] = (r._value, r._total, r.num_updates)

        if r is not None:
            ds.attrs.create('update_type_code', data=r.type_code)
        # Save the TITTLE attribute to be more consistent with what
        # Pytables would do.
        ds.attrs.create("TITLE", name)

    # TODO: Save the _value_list, _total_list and _accumulate_values_bool
    # variables
    @staticmethod
    def save_to_pytables_table(parent, results_list):
        """
        Save the Result object.
        """
        pytables_file = parent._v_file
        name = results_list[0].name
        # pylint: disable= E1101
        description = {'_value': tb.FloatCol(), '_total': tb.FloatCol(), 'num_updates': tb.IntCol()}
        table = pytables_file.createTable(parent, name, description,
                                          title=name)
        row = table.row

        r = None
        for r in results_list:
            # pylint: disable=W0212
            row['_value'] = r._value
            row['_total'] = r._total
            row['num_updates'] = r.num_updates
            row.append()

        pytables_file.setNodeAttr(table, 'update_type_code', r.type_code)
        table.flush()

    @staticmethod
    def load_from_hdf5_dataset(ds):
        """Load a list of Rersult objects from an HDF5 dataset.

        This dataset was suposelly saved with the save_to_hdf5_dataset
        function.

        Parameters
        ----------
        ds : An HDF5 Dataset
            The dataset to be loaded.

        Returns
        -------
        results_list : A list of Result objects.
            The list of Result objects loaded from the dataset.

        Notes
        -----
        This method is called from the load_from_hdf5_file method in the
        SimulationResults class. It uses the python h5py library and
        `ds` is supposed to be an HDF5 dataset created with that library.

        See also
        --------
        save_to_hdf5_dataset

        """
        results_list = []

        name = ds.name.split('/')[-1]
        update_type_code = ds.attrs['update_type_code']
        for data in ds:
            r = Result.create(name,
                              update_type_code,
                              data['_value'],
                              data['_total'])
            r.num_updates = data['num_updates']
            results_list.append(r)
        return results_list

    # def to_pandas_series_of_dataframe(self):
    #     """
    #     Convert the Result object to a pandas DataFrame
    #     """
    #     if self.type_code == Result.RATIOTYPE:
    #         df = pd.DataFrame({'v': self._value_list, 't': self._total_list})
    #         return df
    #     else:
    #         s = pd.Series(self._value_list)
    #         return s

# xxxxxxxxxx Result - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
