#!/usr/bin/env python
"""Module containing simulation parameter classes."""

import copy
import functools
import itertools
import operator
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np

from ..util.serialize import JsonSerializable
from .configobjvalidation import (integer_numpy_array_check,
                                  integer_scalar_or_integer_numpy_array_check,
                                  real_numpy_array_check,
                                  real_scalar_or_real_numpy_array_check)

# TODO: Change configobj to yaml + voluptuous (for validation)
# https://pypi.org/project/voluptuous/

try:
    # noinspection PyUnresolvedReferences
    from configobj import ConfigObj, Section, flatten_errors
    # noinspection PyUnresolvedReferences
    from validate import Validator
except ImportError:  # pragma: no cover
    pass

try:
    import cPickle as pickle
except ImportError as e:  # pragma: no cover
    import pickle

try:
    # noinspection PyUnresolvedReferences
    import pandas as pd
    DataFrame = pd.DataFrame
except ImportError:  # pragma: no cover
    # This will be used just for type checking, since pandas is not installed
    from typing import Any as DataFrame

__all__ = ["combine_simulation_parameters", "SimulationParameters"]


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Module Functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def combine_simulation_parameters(params1, params2):
    """
    Combine two SimulationParameters objects and return a new
    SimulationParameters object corresponding to the union of them.

    The union is only valid if both objects have the same parameters and
    only the values of the unpacked parameters are different.

    Parameters
    ----------
    params1 : SimulationParameters
        The first SimulationParameters object.
    params2 : SimulationParameters
        The second SimulationParameters object.

    Returns
    -------
    union : SimulationParameters
        The union of 'params1' and 'params2'.
    """
    if set(params1.parameters.keys()) != set(params2.parameters.keys()):
        raise RuntimeError('Both SimulationParameters objects must have'
                           ' the same parameters.')

    if set(params1.unpacked_parameters) != set(params2.unpacked_parameters):
        raise RuntimeError('Both SimulationParameters objects must have'
                           ' the same unpacked parameters (only the values'
                           ' should can be different).')

    fixed_parameters = params1.fixed_parameters
    for key in fixed_parameters:
        if params1[key] != params2[key]:
            raise RuntimeError('The fixed parameters in both '
                               'SimulationParameters objects must have the'
                               ' same value.')

    union = SimulationParameters()

    # Add the fixed parameters to the 'union' object.
    for key in fixed_parameters:
        union.add(key, copy.copy(params1[key]))

    # Add the union of the unpacked parameters to the 'union' object.
    for key in params1.unpacked_parameters:
        union.add(key, np.union1d(params1[key], params2[key]))

    # Set the parameters to be unpacked.
    for p in params1.unpacked_parameters:
        union.set_unpack_parameter(p)

    return union


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationParameters - START xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SimulationParameters(JsonSerializable):
    """Class to store the simulation parameters.

    A SimulationParameters object acts as a container for all simulation
    parameters. To add a new parameter to the object just call the
    :meth:`add` method passing the name and the value of the parameter. The
    value can be anything as long as the
    :meth:`.SimulationRunner._run_simulation` method can understand it.

    Alternatively, you can create a SimulationParameters object with all
    the parameters with the static method :meth:`create`, which receives a
    dictionary with the parameter names as keys.

    Parameters can be marked to be "unpacked", as long as they are
    iterable, by calling the :meth:`set_unpack_parameter` method. Different
    simulations will be performed for every combination of parameters
    marked to be unpacked, with the other parameters kept constant.

    Examples
    --------

    - Create a new empty SimulationParameters object and add the individual
      parameters to it by calling its :meth:`add` method.

      .. code-block:: python

         params = SimulationParameters()
         params.add('p1', [1,2,3])
         params.add('p2', ['a','b'])
         params.add('p3', 15)

    - Creating a new SimulationParameters object with the static
      :meth:`create` function.

      .. code-block:: python

         p = {'p1':[1,2,3], 'p2':['a','b'],'p3':15}
         params=SimulationParameters.create(p)
         params.set_unpack_parameter('p1')
         params.set_unpack_parameter('p2')

    - We can then set some of the parameters to be unpacked with

      .. code-block:: python

         params.set_unpack_parameter('p1')


    See also
    --------
    .SimulationResults : Class to store simulation results.
    .SimulationRunner : Base class to implement Monte Carlo simulations.
    """
    def __init__(self):
        # Dictionary that will store the parameters. The key is the
        # parameter name and the value is the parameter value.
        self.parameters = {}

        # A set to store the names of the parameters that will be unpacked.
        # Note there is a property to get the parameters marked to be
        # unpacked, that is, the unpacked_parameters property.
        self._unpacked_parameters_set = set()

        # If this SimulationParameters object was unpacked into a list of
        # SimulationParameters objects then each of these new objects will
        # set this variable with its index in that list. In other words, if
        # this member variable in a SimulationParameters object was set to
        # a non-negative integer then that SimulationParameters object is
        # actually one of the unpacked variations of another
        # SimulationParameters object. The original SimulationParameters
        # object will be stored in the _original_sim_params member
        # variable.
        self._unpack_index = -1
        self._original_sim_params = None

    @property
    def unpack_index(self):
        """Get method for the unpack_index property.

        Returns
        -------
        int
            The unpack index. Ir this SimulationParameters object is not
            an unpacked SimulationParameters then -1 is returned.
        """
        return self._unpack_index

    @property
    def unpacked_parameters(self):
        """
        Names of the parameters marked to be unpacked.

        Returns
        -------
        list[str]
        """
        return sorted(self._unpacked_parameters_set)

    @property
    def fixed_parameters(self):
        """
        Names of the parameters which are NOT marked to be unpacked.

        Returns
        -------
        list[str]
            List with the names of the fixed parameter.
        """
        fixed_params = [
            name for name in self.parameters.keys()
            if name not in self._unpacked_parameters_set
        ]
        return fixed_params

    @staticmethod
    def _create(params_dict, unpack_index=-1, original_sim_params=None):
        """
        Creates a new SimulationParameters object.

        This static method provides a different way to create a
        SimulationParameters object, already containing the parameters in
        the `params_dict` dictionary.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters. Each dictionary key
            corresponds to a parameter's name, while the dictionary value
            corresponds to the actual parameter value..
        unpack_index : int
            Index of the created SimulationParameters object when it is
            part of the unpacked variations of another SimulationParameters
            object. See :meth:`get_unpacked_params_list`.
        original_sim_params : SimulationParameters
            The original SimulationParameters object from which the
            SimulationParameters object that will be created by this method
            came from.

        Returns
        -------
        sim_params : SimulationParameters
            The corresponding SimulationParameters object.
        """
        sim_params = SimulationParameters()
        sim_params.parameters = copy.deepcopy(params_dict)
        if unpack_index < 0:
            unpack_index = -1
        # pylint: disable=W0212
        sim_params._unpack_index = unpack_index
        # pylint: disable=W0212
        sim_params._original_sim_params = original_sim_params
        return sim_params

    @staticmethod
    def create(params_dict):
        """
        Creates a new SimulationParameters object.

        This static method provides a different way to create a
        SimulationParameters object, already containing the parameters in
        the `params_dict` dictionary.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters. Each dictionary key
            corresponds to a parameter's name, while the dictionary value
            corresponds to the actual parameter value..

        Returns
        -------
        sim_params : SimulationParameters
            The corresponding SimulationParameters object.
        """
        return SimulationParameters._create(params_dict)

    def add(self, name, value):
        """Adds a new parameter to the SimulationParameters object.

        If there is already a parameter with the same name it will be
        replaced.

        Parameters
        ----------
        name : str
            Name of the parameter.
        value : anything
            Value of the parameter.
        """
        self.parameters[name] = value

    def remove(self, name):
        """
        Remove the parameter with name `name` from the SimulationParameters
        object

        Parameters
        ----------
        name : str
            Name of the parameter to be removed.

        Raises
        ------
        KeyError
            If `name` is not in parameters.
        """
        del self.parameters[name]
        if name in self._unpacked_parameters_set:
            self._unpacked_parameters_set.remove(name)

    def set_unpack_parameter(self, name, unpack_bool=True):
        """Set the unpack property of the parameter with name `name`.

        The parameter `name` must be already added to the
        SimulationParameters object and be an iterable.

        This is used in the :class:`.SimulationRunner` class.

        Parameters
        ----------
        name : str
            Name of the parameter to be unpacked.
        unpack_bool : bool, optional
            True activates unpacking for `name`, False deactivates it.

        Raises
        ------
        ValueError
            If `name` is not in parameters or is not iterable.
        """
        if name in self.parameters.keys():
            if isinstance(self.parameters[name], Iterable):
                if unpack_bool is True:
                    self._unpacked_parameters_set.add(name)
                else:
                    self._unpacked_parameters_set.remove(name)
            else:
                raise ValueError("Parameter {0} is not iterable".format(name))
        else:
            raise ValueError("Unknown parameter: `{0}`".format(name))

    def __getitem__(self, name):
        """Return the parameter with name `name`.

        Easy access to a given parameter using the brackets syntax.

        Parameters
        ----------
        name : str
            Name of the desired parameter.

        Returns
        -------
        desired_param : anything
            The value of the parameter with name `name`.
        """
        return self.parameters[name]

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def __modify_name(self, p_name):
        """
        Add an * in p_name if it is set to be unpacked

        Parameters
        ----------
        p_name : str
            The original name.
        """
        if p_name in self._unpacked_parameters_set:
            p_name += '*'
        return p_name

    def __repr__(self):  # pragma: no cover
        """
        Get the object representation as a string.

        Returns
        -------
        str
            The object representation as a string.
        """
        repr_list = []
        for name, value in self.parameters.items():
            repr_list.append("'{0}': {1}".format(self.__modify_name(name),
                                                 value))
        return '{%s}' % ', '.join(repr_list)

    def __len__(self):
        """
        Get the number of different parameters stored in the
        SimulationParameters object.

        Returns
        -------
        length : int
            The number of different parameters stored in the
            SimulationParameters object
        """
        return len(self.parameters)

    def __iter__(self):  # pragma: no cover
        """
        Get an iterator to the parameters in the SimulationParameters
        object.

        Returns
        -------
        iterator
            An iterator to the SimulationParameters object.
        """
        return iter(self.parameters)

    def __eq__(self, other):
        """
        Check if two SimulationParameters objects are equal.

        Two simulation parameters objects are considered equal if all
        parameters stored in both objects are the same, except for a
        parameter object called 'rep_max'.

        Parameters
        ----------
        other: SimulationParameters
            The other SimulationParameters to be compared with self.

        Returns
        -------
        bool
            True if both objects are considered to be equal, returns
            False otherwise.

        Notes
        -----
        The main usage for comparing if two SimulationParameters objects
        are equal is when loading partial results in the :class:`.SimulationRunner`
        class, where we need to assure we are not combining results for
        different simulation parameters. The :class:`.SimulationRunner` class must
        check if the loaded results parameters match the current parameters
        to be simulated and thus require the "==" operator (or the "!="
        operator) to be implemented.

        However, it makes sense to ignore a parameter called 'rep_max',
        since it is not a parameter related to a 'scenario'. It is used in
        the :class:`.SimulationRunner` class to indicate the maximum number of
        iterations to perform and there is no problem when its value is
        different.
        """
        if self is other:  # pragma: no cover
            return True

        if not isinstance(other, self.__class__):
            return False

        # pylint: disable=W0212
        if self._unpacked_parameters_set != other._unpacked_parameters_set:
            return False

        if set(self.parameters.keys()) != set(other.parameters.keys()):
            return False

        if self._unpack_index != other._unpack_index:
            return False

        for key in self.parameters.keys():
            # We care about all keys, except for a key called 'rep_max'
            # whose value does not matter when comparing if two
            # SimulationResults objects are equal or not.
            if key != "rep_max":
                # noinspection PyTypeChecker
                if np.any(self.parameters[key] != other.parameters[key]):
                    return False

        # If we didn't return until we reach this point then the objects
        # are equal
        return True

    def __ne__(self, other):
        """
        Check if two SimulationParameters objects are different.

        See documentation for __eq__.

        Parameters
        ----------
        other: SimulationParameters
            The other SimulationParameters to be compared with self.

        Returns
        -------
        bool
            True if both objects are considered to be different, returns
            False otherwise.
        """

        return not self.__eq__(other)

    def get_num_unpacked_variations(self):
        """
        Get the number of variations when the parameters are unpacked.

        If no parameter was marked to be unpacked, then return 1.

        If this SimulationParameters object is actually a 'unpacked
        variation' of another SimulationParameters object, return the
        number of variations of the parent SimulationParameters object
        instead.

        Returns
        -------
        num : int
            The number of variations when the parameters are unpacked.
        """
        # If self._original_sim_params is not None, that means that this
        # SimulationParameters object is in fact one of the unpacked
        # variations of another SimulationParameters object. In that case,
        # we return the number of unpacked variations of the original
        # object.
        if self._original_sim_params is not None:
            return self._original_sim_params.get_num_unpacked_variations()

        if len(self._unpacked_parameters_set) == 0:
            return 1

        # Generator for the lengths of the parameters set to be
        # unpacked
        gen_values = (len(self.parameters[i])
                      for i in self._unpacked_parameters_set)
        # Just multiply all the lengths
        return functools.reduce(operator.mul, gen_values)

    def get_pack_indexes(self, fixed_params_dict=None):
        """
        When you call the function get_unpacked_params_list you get a
        list of SimulationParameters objects corresponding to all
        combinations of the parameters. The function get_pack_indexes
        allows you to provided all parameters marked to be unpacked except
        one, and returns the indexes of the list returned by
        get_unpacked_params_list that you want.

        Parameters
        ----------
        fixed_params_dict : dict[str, anything]
            A dictionary with the name of the fixed parameters as keys and
            the fixed value as value.

        Returns
        -------
        indexes : np.ndarray
            The desired indexes (1D numpy array or an integer).

        Examples
        --------
        Suppose we have

        >>> p={'p1':[1,2,3], 'p2':['a','b'],'p3':15}
        >>> params=SimulationParameters.create(p)
        >>> params.set_unpack_parameter('p1')
        >>> params.set_unpack_parameter('p2')

        If we call params.get_unpacked_params_list we will get a list of
        six SimulationParameters objects, one for each combination of the
        values of p1 and p2. That is,

        >>> len(params.get_unpacked_params_list())
        6

        Likewise, in the simulation the :class:`.SimulationRunner` object will return
        a list of results in the order corresponding to the order of the
        list of parameters. The get_pack_indexes is used to get the index
        of the results corresponding to a specific configuration of
        parameters. Suppose now you want the results when 'p2' is varying,
        but with the other parameters fixed to some specific value. For
        this create a dictionary specifying all parameters except 'p2' and
        call get_pack_indexes with this dictionary. You will get an array
        of indexes that can be used in the results list to get the desired
        results. For instance

        >>> fixed={'p1':3,'p3':15}
        >>> indexes = params.get_pack_indexes(fixed)
        >>> index0 = indexes[0]
        >>> index1 = indexes[1]
        >>> unpacked_list = params.get_unpacked_params_list()
        >>> unpacked_list[index0]['p1']
        3
        >>> unpacked_list[index0]['p3']
        15
        >>> unpacked_list[index1]['p1']
        3
        >>> unpacked_list[index1]['p3']
        15
        """
        if fixed_params_dict is None:  # pragma: no cover
            fixed_params_dict = {}

        # Get the only parameter that was not fixed
        varying_param = list(self._unpacked_parameters_set -
                             set(fixed_params_dict.keys()))

        # List to store the indexes (as strings) of the fixed parameters,
        # as well as ":" for the varying parameter,
        param_indexes = []
        # Note that self.unpacked_parameters is a sorted list, which
        # guarantees a predictable order
        for i in self.unpacked_parameters:
            if i in varying_param:
                param_indexes.append(':')
            else:
                fixed_param_value_index = list(self.parameters[i]).index(
                    fixed_params_dict[i])
                param_indexes.append(str(fixed_param_value_index))

        # xxxxx Get the indexes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # For this we create a auxiliary numpy array going from 0 to the
        # number of unpack variations. Then we use param_indexes to build a
        # string that we can evaluate using the auxiliary numpy array in
        # order to get the linear indexes.

        # Get the lengths of the parameters marked to be unpacked
        dimensions = [
            len(self.parameters[i]) for i in self.unpacked_parameters
        ]
        aux = np.arange(0, self.get_num_unpacked_variations())
        aux.shape = dimensions
        indexes = eval("aux" +
                       "[{0}]".format(",".join(param_indexes))).flatten()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # if indexes.size == 1:
        #     indexes = indexes[0]

        return indexes

    def get_unpacked_params_list(self):
        """
        Get a list of SimulationParameters objects, each one
        corresponding to a possible combination of (unpacked) parameters.

        Returns
        -------
        list[SimulationParameters]
           A list of SimulationParameters objects.

        Examples
        --------
        Suppose you have a SimulationParameters object with the parameters
        'a', 'b', 'c' and 'd' as below

        >>> simparams = SimulationParameters()
        >>> simparams.add('a', 1)
        >>> simparams.add('b', 2)
        >>> simparams.add('c', [3, 4])
        >>> simparams.add('d', [5, 6])

        and the parameters 'c' and 'd' were set to be unpacked.

        >>> simparams.set_unpack_parameter('c')
        >>> simparams.set_unpack_parameter('d')

        Then get_unpacked_params_list would return a list of four
        SimulationParameters objects as below

        >>> len(simparams.get_unpacked_params_list())
        4

        That is

        .. code-block:: python

           [{'a': 1, 'c': 3, 'b': 2, 'd': 5},
            {'a': 1, 'c': 3, 'b': 2, 'd': 6},
            {'a': 1, 'c': 4, 'b': 2, 'd': 5},
            {'a': 1, 'c': 4, 'b': 2, 'd': 6}]
        """
        # If unpacked_parameters is empty, return self
        if not self._unpacked_parameters_set:
            return [self]

        # Lambda function to get an iterator to a (iterable) parameter
        # given its name. This only works if self.parameters[name] is an
        # iterable.
        def get_iter_from_name(name):
            return iter(self.parameters[name])

        # Dictionary that stores the name and an iterator of a parameter
        # marked to be unpacked
        unpacked_params_iter_dict = OrderedDict()
        # The sorted function is important to guarantee that the keys
        # returned from _unpacked_parameters_set will have a predictable
        # order
        for i in sorted(self._unpacked_parameters_set):
            unpacked_params_iter_dict[i] = get_iter_from_name(i)
        keys = list(unpacked_params_iter_dict.keys())

        # Using itertools.product we can convert the multiple iterators
        # (for the different parameters marked to be unpacked) to a single
        # iterator that returns all the possible combinations (cartesian
        # product) of the individual iterators.
        all_combinations = itertools.product(
            *(unpacked_params_iter_dict.values()))

        # Names of the parameters that don't need to be unpacked
        regular_params = (set(self.parameters.keys()) -
                          self._unpacked_parameters_set)

        # Constructs a list with dictionaries, where each dictionary
        # corresponds to a possible parameters combination
        unpack_params_length = len(self._unpacked_parameters_set)
        all_possible_dicts_list = []
        for comb in all_combinations:
            new_dict = {}
            # Add current combination of the unpacked parameters
            for index in range(unpack_params_length):
                new_dict[keys[index]] = comb[index]
            # Add the regular parameters
            for param in regular_params:
                new_dict[param] = self.parameters[param]
            all_possible_dicts_list.append(new_dict)

        # Map the list of dictionaries to a list of SimulationParameters
        # objects and return it.
        #
        # Note that since we are passing the index "i" to each new object
        # in the list as well as the original SimulationParameters object
        # "self", then each SimulationParameters object in the returned
        # list will know its index in that list (the _unpack_index
        # variable) as well as the original SimulationParameters object
        # from where it came from (stored in the _original_sim_params
        # variable).
        sim_params_list = [
            SimulationParameters._create(v, i, self)
            for i, v in enumerate(all_possible_dicts_list)
        ]
        return sim_params_list

    def save_to_pickled_file(self, filename):
        """
        Save the SimulationParameters object to the file `filename` using
        pickle.

        Parameters
        ----------
        filename : str
            Name of the file to save the parameters.
        """
        with open(filename, 'wb') as output:
            pickle.dump(self, output, protocol=2)

    @staticmethod
    def load_from_pickled_file(filename):
        """
        Load the SimulationParameters from the file 'filename' previously
        stored (using pickle) with `save_to_pickled_file`.

        Parameters
        ----------
        filename : str
            Name of the file from where the results will be loaded.

        Returns
        -------
        SimulationParameters
            The loaded SimulationParameters object.
        """
        with open(filename, 'rb') as input_file:
            obj = pickle.load(input_file)
        return obj

    @staticmethod
    def load_from_config_file(filename, spec=None, save_parsed_file=False):
        """
        Load the SimulationParameters from a config file using the
        configobj module.

        If the config file has a parameter called ``unpacked_parameters``,
        which should be a list of strings with the names of other
        parameters, then these parameters will be set to be unpacked.

        Parameters
        ----------
        filename : str
            Name of the file from where the results will be loaded.
        spec : list[str], optional
            A list of strings with the config spec. See "validation" in the
            configobj module documentation for more info.
        save_parsed_file : bool
            If `save_parsed_file` is True, then the parsed config file will
            be written back to disk. This will add any missing values in
            the config file whose default values are provided in the
            `spec`. This will even create the file if all default values
            are provided in `spec` and the file does not exist yet.

        Notes
        -----
        Besides the usual checks that the configobj validation has such as
        ``integer``, ``string``, ``option``, etc., you can also use
        `real_numpy_array` for numpy float arrays. Note that when this
        validation function is used you can set the array in the config
        file in several ways such as

            SNR=0,5,10,15:20

        for instance.
        """
        if spec is None:
            spec = []

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        def add_params(simulation_params, config):
            """
            Add the parameters in `config`.

            Parameters
            ----------
            simulation_params : SimulationParameters
                The SimulationParameters object where the parameters will
                be added.
            config : configobj.ConfigObj | configobj.Section
                A ConfigObj object or a Section object. The `config` object
                can contain parameters (called scalars) or sections which
                can contain either parameters or other sections.
            """
            # Add scalar parameters
            for v in config.scalars:
                simulation_params.add(v, config[v])

            for s in config.sections:
                add_params(simulation_params, config[s])

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        conf_file_parser = ConfigObj(filename,
                                     list_values=True,
                                     configspec=spec)

        # Dictionary with custom validation functions. Here we add a
        # validation function for numpy float arrays.
        fdict = {
            'real_numpy_array':
            real_numpy_array_check,
            'integer_numpy_array':
            integer_numpy_array_check,
            'integer_scalar_or_integer_numpy_array_check':
            integer_scalar_or_integer_numpy_array_check,
            'real_scalar_or_real_numpy_array_check':
            real_scalar_or_real_numpy_array_check
        }
        validator = Validator(fdict)

        # The 'copy' argument indicates that if we save the ConfigObj
        # object to a file after validating, the default values will also
        # be written to the file.
        result = conf_file_parser.validate(validator,
                                           preserve_errors=True,
                                           copy=True)

        # Note that if there was no parsing errors, then "result" will be
        # 'True'.  It there was an error, then result will be a dictionary
        # with each parameter as a key. The value of each key will be
        # either 'True' if that parameter was parsed without error or a
        # "validate.something" object (since we set preserve_errors to
        # True) describing the error.

        # if result != True:
        #     print 'Config file validation failed!'
        #     sys.exit(1)

        # xxxxx Test if there was some error in parsing the file xxxxxxxxxx
        # The flatten_errors function will return only the parameters whose
        # parsing failed.
        errors_list = flatten_errors(conf_file_parser, result)
        if len(errors_list) != 0:
            first_error = errors_list[0]
            # The exception will only describe the error for the first
            # incorrect parameter.
            if first_error[2] is False:
                msg = ("Error loading file {0}. Parameter '{1}' in section "
                       "'{2}' must be provided.")
                raise Exception(
                    msg.format(filename, first_error[1], first_error[0][0]))

            msg = ("Error loading file {0}. Parameter '{1}' in section "
                   "'{2}' is invalid. {3}")
            raise Exception(
                msg.format(filename, first_error[1], first_error[0][0],
                           str(first_error[2]).capitalize()))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Finally add the parsed parameters to the params object xxxx
        params = SimulationParameters()
        add_params(params, conf_file_parser)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        if save_parsed_file is True:  # pragma: no cover
            # xxxxx Write the parsed config file to disk xxxxxxxxxxxxxxxxxx
            # This will add the default values if they are not present. If
            # the file does not exist yet and all default values are
            # provided in the spec then the file will be created. If some
            # parameter without a default value was not provided then when
            # exception would already have been thrown and we wouldn't be
            # here.
            conf_file_parser.write()
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # If the special parameter 'unpacked_parameters' is in the config
        # file, then we will set the parameters whose name are in it to be
        # unpacked
        try:
            unpacked_parameters_list = params['unpacked_parameters']
        except KeyError:
            unpacked_parameters_list = []
        for name in unpacked_parameters_list:
            params.set_unpack_parameter(name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return params

    def _to_dict(self):
        """
        Convert the SimulationParameters object to a dictionary for easier
        further serialization.
        """
        original_sim_params: "SimulationParameters" = self._original_sim_params

        if original_sim_params is not None:
            original_sim_params = original_sim_params._to_dict()

        out = {
            'parameters': self.parameters,
            'unpacked_parameters_set': self._unpacked_parameters_set,
            'unpack_index': self.unpack_index,
            'original_sim_params': original_sim_params
        }
        return out

    @staticmethod
    def _from_dict(d):
        """
        Create a new SimulationParameters object from a dictionary.

        Parameters
        ----------
        d : dict
            The dictionary with the data.

        Returns
        -------
        SimulationParameters
            The new SimulationParameters object.
        """
        sim_params = SimulationParameters()
        sim_params.parameters = d['parameters']
        sim_params._unpacked_parameters_set = d['unpacked_parameters_set']
        sim_params._unpack_index = d['unpack_index']

        original_sim_params = d['original_sim_params']
        if original_sim_params is not None:
            original_sim_params = SimulationParameters._from_dict(
                original_sim_params)

        sim_params._original_sim_params = original_sim_params
        return sim_params

    def to_dataframe(self):
        """
        Convert the SimulationParamters object to a pandas DataFrame.
        """
        data = {}
        all_params_list = self.get_unpacked_params_list()
        for name in self:
            data[name] = [a[name] for a in all_params_list]
        return pd.DataFrame(data)

    def _repr_html_(self):
        """
        Method used by the jupyter rich display.

        We will simple use the HTML representation of the
        corresponding pandas
        dataframe.
        """
        return self.to_dataframe()._repr_html_()


# xxxxxxxxxx SimulationParameters - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
