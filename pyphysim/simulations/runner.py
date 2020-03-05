#!/usr/bin/env python
"""Module containing the simulation runner."""

import itertools
import os
import sys
from argparse import ArgumentParser
from time import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from pyphysim.simulations.progressbar import ProgressbarZMQClient, ProgressBarBase

from ..util.misc import pretty_time
from .parameters import SimulationParameters
from .progressbar import (ProgressBarIPython, ProgressbarText,
                          ProgressbarText2, ProgressbarText3,
                          ProgressbarZMQServer)
from .results import Result, SimulationResults

try:
    # noinspection PyUnresolvedReferences
    from ipyparallel import LoadBalancedView, DirectView
except ImportError:  # pragma: no cover
    LoadBalancedView = Any
    DirectView = Any

__all__ = ["get_partial_results_filename", "SimulationRunner", "SkipThisOne"]

UpdateFunction = Callable[[int], None]

ProxybarData = Tuple[int, str, int]

# A view form ipyparallel
ParallelView = Union[LoadBalancedView, DirectView]


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Command Line Argument Parser xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def get_common_parser() -> ArgumentParser:
    """
    Get the command line parser that can be used.

    The global parser is a singleton object of the ArgumentParser class which is
    used in the `simulate_do_what_i_mean` function to parse command line
    arguments.

    It already has two arguments, "index" and "config" configured. If you need
    more then that, you can get the global parser object by calling this
    function and then calling the `add_argument` method of the returned object.
    See the documentation of ArgumentParser for more.

    Returns
    -------
    ArgumentParser
        The command line parser.
    """
    if get_common_parser.parser is None:  # type: ignore
        parser = ArgumentParser()
        group = parser.add_argument_group('General')

        help_msg = ('An index for the parameters variations. If provided, '
                    'only that variation will be simulated.')
        group.add_argument(
            '-i',  # short version to specify the opt
            '--index',  # Long version to specify the option
            help=help_msg,
            metavar='VARIATION INDEX',
            type=int,
            nargs='?')

        help_msg = 'Name of the file with the simulation parameters'
        group.add_argument(
            '-c',  # short version to specify the opt
            '--config',  # Long version to specify the opt
            help=help_msg,
            metavar='CONFIG FILENAME',
            # default=default_config_file,
            type=str,
            nargs='?')

        help_msg = ('Instead of running the simulation, return the '
                    'number of variations.')
        group.add_argument(
            # short version to specify the option
            '-n',
            # Long version to specify the option
            '--number_variations',
            help=help_msg,
            action='store_true')

        get_common_parser.parser = parser  # type: ignore

    return get_common_parser.parser  # type: ignore


get_common_parser.parser = None  # type: ignore

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Module Functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def get_partial_results_filename(results_base_filename: str,
                                 current_params: SimulationParameters,
                                 partial_results_folder: Optional[str] = None
                                 ) -> str:
    """
    Get the name of the file where the partial result will be saved for
    the unpacked result with index `unpack_index`.

    Parameters
    ----------
    results_base_filename : str
        Base name for partial result file.
    current_params : SimulationParameters
        The current parameters, which must be a "unpacked variation" of
        another SimulationParameters object.
    partial_results_folder : str, optional
        The folder where the partial results will be stored.

    Returns
    -------
    partial_results_filename : str
        The name of the partial results file.
    """
    # This will get the number of unpacked variations of the parent
    # SimulationParameters object.
    total_unpacks = current_params.get_num_unpacked_variations()
    num_digits = len(str(total_unpacks))
    unpack_index_str = str(current_params.unpack_index).zfill(num_digits)

    partial_results_filename = '{0}_unpack_{1}.pickle'.format(
        results_base_filename, unpack_index_str)

    if partial_results_folder is not None:
        partial_results_filename = os.path.join(partial_results_folder,
                                                partial_results_filename)

    return partial_results_filename


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Exception xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SkipThisOne(Exception):
    """
    Exception that can be raised in the `_run_simulation` method to skip
    the current repetition.

    The `simulate` method will not count a run of `_run_simulation` if it
    throws a `SkipThisOne` exception.

    Parameters
    ----------
    msg : str
        The message with more information on why the exception was
        raised.
    """
    def __init__(self, msg: str) -> None:
        """
        Parameters
        ----------
        msg : str
            The message with more information on why the exception was
            raised.
        """
        super().__init__()
        self.msg = msg

    def __str__(self) -> str:  # pragma: nocover
        """
        Convert the exception object to a suitable string representation.

        Returns
        -------
        str
            String representation of the object.
        """
        return "SkipThisOne: {0}".format(self.msg)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationRunner - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# pylint: disable=R0902
class SimulationRunner:
    """
    Base class to run Monte Carlo simulations.

    The main idea of the :class:`SimulationRunner` class is that in order
    to implement a Monte Carlo simulation one would subclass
    :class:`SimulationRunner` and implement the :meth:`_run_simulation`
    method (as well as any of the optional methods). The complete procedure
    is described in the documentation of the :mod:`simulations` module.

    The code below illustrates the minimum pseudo code to implement a
    subclass of :class:`SimulationRunner`.

    .. code-block:: python

       class SomeSimulator(SimulationRunner):
       def __init__(self):
           SimulationRunner.__init__(self)
           # Do whatever you need/want

           # Add the simulation parameters to the `params` attribute.
           self.params.add('par1', par1value)
           ...
           self.params.add('parN', parNvalue)
           # Optionally set some parameter(s) to be unpacked
           self.params.set_unpack_parameter('name_of_a_parameter')

       def _run_simulation(self, current_parameters):
           # Get the simulation parameters from the current_parameters
           # object. If no parameter was marked to be unpacked, then
           # current_parameters will be equivalent to self.params.
           par1 = current_parameters['par1']
           ...
           parN = current_parameters['parN']

           # Do the simulation of one iteration using the parameters
           # par1,...parN from the current_parameters object.
           ...

           # Save the results of this iteration to a SimulationResults
           # object and return it
           simResults = SimulationResults()
           simResults.add_new_result(...)  # Add one each result you want
           simResults.add_new_result(...)
           return simResults

    With that, all there is left to run the simulation is to create a
    SomeSimulator object and call its :meth:`simulate` method.

    Parameters
    ----------
    default_config_file : str
        Name of the config file. This will be parsed with configobj.
    config_spec : list[str]
        Configuration specification to validate the config file.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    save_parsed_file : bool
        If True, the config file will be saved after it is loaded. This is
        useful to add the default parameters to the config file so that
        they can be easily changed later. Note that if the config file does
        not exist at all, then it will be saved regardless of the value of
        `save_parsed_file`.

    See Also
    --------
    .SimulationResults : Class to store simulation results.
    .SimulationParameters : Class to store the simulation parameters.
    .Result : Class to store a single simulation result.
    """
    def __init__(self,
                 default_config_file: Optional[str] = None,
                 config_spec: Optional[str] = None,
                 read_command_line_args: bool = True,
                 save_parsed_file: bool = False) -> None:
        self.rep_max = 1
        # Number of iterations performed by simulation when it finished
        self._runned_reps: List[int] = []

        self._config_filename = None
        # Configobj specification (to validate parameters read from the
        # config file)
        self._configobj_spec = config_spec

        # xxxxx Parse command line arguments (get config filename) xxxxxxxx
        if read_command_line_args is True:  # pragma: no cover
            # Note that the get_common_parser always return the same object
            parser = get_common_parser()

            # This member variable will store all the parsed command line
            # arguments
            self.command_line_args = parser.parse_args()
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            # Get the config filename if it was provided in the command
            # line. If not, use the default value.
            if self.command_line_args.config is None:
                self._config_filename = default_config_file
            else:  # pragma: no cover
                self._config_filename = self.command_line_args.config
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        else:  # pragma: no cover
            # Since we are not parsing command line arguments, the config
            # file will be the provided default_config_file
            self._config_filename = default_config_file
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Read the parameters from the config file xxxxxxxxxxxxx
        if self._config_filename is None:
            self.params = SimulationParameters()
        else:  # pragma: no cover
            if not os.path.isfile(self._config_filename):
                # If the config file does not exist, we will save the file
                # no matter the value of save_parsed_file
                save_parsed_file = True

            self.params = SimulationParameters.load_from_config_file(
                self._config_filename,
                self._configobj_spec,
                save_parsed_file=save_parsed_file)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        self.results = SimulationResults()

        # This variable will be used later to store the progressbar object
        # when it is created in the _get_update_progress_function method
        self._pbar: Optional[ProgressBarBase] = None

        # xxxxxxxxxx update_progress_function_style xxxxxxxxxxxxxxxxxxxxxxx
        # --- When the simulation is performed Serially -------------------
        # Sets the style of the used progressbar. The allowed values are
        # 'text1', 'text2', None, or a callable object.
        # - If it is 'text1' then the ProgressbarText class will be used.
        # - If it is 'text2' then the ProgressbarText2 class will be used.
        # - If it is None, then no progressbar will be used.
        # - If it is a callable, then that callable object must receive two
        #   arguments, the rep_max and the message values, and return a
        #   function that receives a single argument (the custom
        #   parameters).
        # --- When the simulation is performed in parallel ----------------
        # - If it is None then no progressbar will be used
        # - If it is not None then a socket progressbar will be used, which
        #   employs the same style as 'text2'
        self.update_progress_function_style: Optional[str] = 'text2'

        # This can be either 'screen' or 'file'. If it is 'file' then the
        # progressbar will write the progress to a file with appropriated
        # filename
        self.progress_output_type = 'screen'

        # Dictionary with extra arguments that will be passed to the
        # __init__ method of the progressbar class. For instance, when
        # simulating in parallel and update_progress_function_style is not
        # None a progressbar based on ZMQ sockets will be used. Set
        # progressbar_extra_args to "{'port':3456}" in order for the
        # progressbar to use port 3456.
        self.progressbar_extra_args: Dict[str, Any] = {}

        # Additional message printed in the progressbar. The message can
        # contain "{SomeParameterName}" which will be replaced with the
        # parameter value.
        #
        # Note that if the update_progress_function_style is None, then no
        # message will be printed either.
        self.progressbar_message = 'Progress'

        # This variable will be used to store the AsyncMapResult object
        # that will be created in the simulate_in_parallel method. This
        # object is part of ipyparallel framework and is used to get
        # the actual results of performing an asynchronous task in IPython.
        self._async_results: Optional[Any] = None

        # xxxxxxxxxx Configure saving of simulation results xxxxxxxxxxxxxxx
        # If this variable is set to True the saved partial results will be
        # deleted after the simulation is finished.
        self.delete_partial_results_bool = False

        # Folder where the partial results will be saved. Set this to None
        # to save the partial results in the same folder of the final
        # results.
        self.partial_results_folder = 'partial_results'
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Internal variables you should not modify xxxxxxxxxxxxxxxxxx
        # Variable to store the name of the file where the simulation
        # results will be stored.
        self._results_base_filename: Optional[str] = None
        # Variable to store all the names for the partial results. Each
        # name in it will be equivalent to the value of
        # _results_base_filename appended with unpack_i where i will be an
        # integer. These names will be used after the simulation has
        # finished and full results were saved to delete the files with the
        # partial results.
        self._results_base_filename_unpack_list: List[str] = []
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Interval variables for tracking simulation time xxxxxxxxxxx
        # Note that self._elapsed_time is different from the 'elapsed_time'
        # result returned after the simulation has finished. This variable
        # only tracks the CURRENT SIMULATION and does not account the time
        # any loaded partial results required to be simulated.
        self._elapsed_time = 0.0
        self.__tic = 0.0
        self.__toc = 0.0
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def set_results_filename(self, filename: Optional[str] = None) -> None:
        """
        Set the name of the file where the simulation results will be
        saved.

        This must be done before calling the `simulate` of the
        `simulate_in_parallel` methods.

        The `filename` argument is formatted with the simulation
        parameters. That is, suppose there are two parameters Nr=2 and
        Nt=1, then if `filename` is equal to "results_for_{Nr}x{Nt}" the
        actual name of the file used to store the simulation results will
        be "results_for_2x1.pickle".

        The advantage of setting the name of the results file with
        `set_results_filename` instead of manually saving the results after
        the simulation is finished is that partial results will also be
        saved. Therefore the simulation can be stopped and continued later
        from these partial results.

        Parameters
        ----------
        filename : str
            The name of the file (without extension) where the simulation
            results will be stored. If not provided the results will not be
            automatically stored.
        """
        self._results_base_filename = filename

    @property
    def results_base_filename(self) -> Optional[str]:
        """
        Get name of the file where the last simulation results were stored.

        Returns
        -------
        str
            The result base filename.
        """
        if self._results_base_filename is None:
            filename = None
        else:
            # noinspection PyTypeChecker
            filename = self.results.get_filename_with_replaced_params(
                self._results_base_filename)
        return filename

    @property
    def results_filename(self) -> Optional[str]:
        """
        Get name of the file where the last simulation results were stored.

        Returns
        -------
        str
            The result base filename.
        """
        results_base_filename = self.results_base_filename
        if results_base_filename is None:
            results_filename = None
        else:
            ext = os.path.splitext(results_base_filename)[-1]
            if ext == '.pickle':
                results_filename = results_base_filename
            else:
                results_filename = '{0}.pickle'.format(results_base_filename)

        return results_filename

    def _get_progress_output_sink(self, param_variation_index: Union[int, str]
                                  ) -> Any:  # pragma: no cover
        """
        Get the output sink for the progressbars.

        If 'self.progress_output_type' is equal to 'screen' this method
        will simple return sys.stdout.

        Parameters
        ----------
        param_variation_index : int | str
            Int or a string that can be converted to int. If this is
            provided and 'self.progress_output_type' is 'file',
            this will be used in the filename of the progress output file.

        Returns
        -------
        out : sys.stdout or a file object
        """
        out = sys.stdout
        if self.progress_output_type == 'screen':
            pass
        else:
            total_unpacks = self.params.get_num_unpacked_variations()
            num_digits = len(str(total_unpacks))
            unpack_index_str = str(param_variation_index).zfill(num_digits)
            filename = '{0}_progress_{1}_of_{2}.txt'.format(
                self.results_base_filename, unpack_index_str, total_unpacks)
            out = open(filename, 'w')

        return out

    def __delete_partial_results_maybe(self) -> None:
        """
        (maybe) Delete the files containing partial results.

        This method is called inside the simulate method after the full
        results were saved. It will only have an effect if the
        "delete_partial_results_bool" variable is True (thus the 'maybe' in
        the method's name).

        Notes
        -----
        This method will do nothing if self.delete_partial_results_bool is
        not True.
        """
        if self.delete_partial_results_bool is True:
            for name in self._results_base_filename_unpack_list:
                try:
                    os.remove(name)
                except OSError:  # pragma: no cover
                    pass
            self._results_base_filename_unpack_list = []
        else:
            # Do nothing if self.delete_partial_results_bool is not True
            pass

    def __save_partial_results(self, current_rep: int,
                               current_params: SimulationParameters,
                               current_sim_results: SimulationResults,
                               partial_results_filename: str) -> None:
        """
        Save the partial simulation results to a file.

        Parameters
        ----------
        current_rep : int
            Current repetition.
        current_params : SimulationParameters
            The current parameters.
        current_sim_results : SimulationResults
            The partial simulations results object to be saved.
        partial_results_filename : str
            The name of the file to save the partial simulation results.

        Notes
        -----
        The name of the results file where the partial results were saved,
        that is, the value of the `partial_results_filename` argument will
        be automatically added to the list of files to be deleted after the
        simulation is finished if delete_partial_results_bool is
        True. However, remember that in the simulate_in_parallel method
        this method will be run in a different object in an IPython
        engine. Therefore, you will need to manually add, the value of the
        partial_results_filename variable to the list of files to be
        deleted (the __results_base_filename_unpack_list variable).
        """
        # xxxxxxxxxx Save partial results to file xxxxxxxxxxxxxxxxxxxxx
        # First we add the current parameters to the partial simulation
        # results object
        current_sim_results.set_parameters(current_params)
        # Now we can save the partial results to a file.
        current_sim_results.current_rep = current_rep

        # Try to save the partial results
        try:
            filename = current_sim_results.save_to_file(
                partial_results_filename)
        except IOError as e:
            if self.partial_results_folder is not None:

                os.mkdir(self.partial_results_folder)
                # This should not raise IOError again.
                filename = current_sim_results.save_to_file(
                    partial_results_filename)
            else:  # pragma: no cover
                raise e

        self._results_base_filename_unpack_list.append(filename)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def clear(self) -> None:  # pragma: no cover
        """
        Clear the SimulationRunner.

        This will erase any results from previous simulations as well as
        other internal variables. The SimulationRunner object will then be
        as if it was just created, except that the simulation parameters
        will be kept.

        """
        self._elapsed_time = 0.0
        # Number of iterations performed by simulation when it finished
        self._runned_reps = []
        self.results = SimulationResults()

    def __run_simulation_and_track_elapsed_time(
            self,
            current_parameters: SimulationParameters) -> SimulationResults:
        """
        Perform the _run_simulation method and track its execution time.
        This time will be added as a Result to the returned
        :class:`.SimulationResults` object from _run_simulation.

        Parameters
        ----------
        current_parameters : SimulationParameters
            SimulationParameters object with the parameters for the
            simulation. The self.params variable is not used directly. It
            is first unpacked in the simulate function which then calls
            _run_simulation for each combination of unpacked parameters.

        Returns
        -------
        SimulationResults
            The current simulation results.

        Notes
        -----
        This method is called in the `simulate` and `simulate_in_parallel`.
        """
        tic = time()
        current_sim_results = self._run_simulation(current_parameters)
        toc = time()
        elapsed_time_result = Result.create('elapsed_time', Result.SUMTYPE,
                                            toc - tic)
        current_sim_results.add_result(elapsed_time_result)

        return current_sim_results

    def _run_simulation(self, current_parameters: SimulationParameters
                        ) -> SimulationResults:
        """
        Performs one iteration of the simulation.

        This function must be implemented in a subclass. It should take the
        needed parameters from the params class attribute (which was filled
        in the constructor of the derived class) and return the results as
        a :class:`.SimulationResults` object.

        Note that _run_simulation will be called self.rep_max times (or
        less if an early stop criteria is reached, which requires
        reimplementing the _keep_going function in the derived class) and
        the results from multiple repetitions will be merged.

        Parameters
        ----------
        current_parameters : SimulationParameters
            SimulationParameters object with the parameters for the
            simulation. The self.params variable is not used directly. It
            is first unpacked in the simulate function which then calls
            _run_simulation for each combination of unpacked parameters.

        Returns
        -------
        simulation_results : SimulationResults
            A SimulationResults object containing the simulation results of
            the run iteration.

        """
        raise NotImplementedError("'_run_simulation' must be implemented "
                                  "in a subclass of SimulationRunner")

    # pylint: disable=W0613,R0201
    def _keep_going(self, current_params: SimulationParameters,
                    current_sim_results: SimulationResults,
                    current_rep: int) -> bool:
        """
        Check if the simulation should continue or stop.

        This function may be reimplemented in the derived class if a stop
        condition besides the number of iterations is desired. The idea is
        that _run_simulation returns a :class:`.SimulationResults` object, which is
        then passed to _keep_going, which is then in charge of deciding if
        the simulation should stop or not.

        Parameters
        ----------
        current_params : SimulationParameters
            SimulationParameters object with the parameters of the
            simulation.
        current_sim_results : SimulationResults
            SimulationResults object from the last iteration (merged with
            all the previous results)
        current_rep : int
            Number of iterations already run.

        Returns
        -------
        result : bool
            True if the simulation should continue or False otherwise.

        Notes
        -----
        This method should be simple (it will be run many times) and SHOULD
        NOT modify the object.
        """
        # If this function is not reimplemented in a subclass it always
        # returns True. Therefore, the simulation will only stop when the
        # maximum number of allowed iterations is reached.
        return True

    def _get_serial_update_progress_function(
            self, current_params: SimulationParameters
    ) -> UpdateFunction:  # pragma: no cover
        """
        Return a function that should be called to update the
        progressbar for the simulation of the current parameters.

        This method is only called in the 'simulate' method.

        The returned function accepts a single argument, corresponding to
        the number of iterations executed so far.

        The progressbar used to get the returned function depend on the
        value of the self.update_progress_function_style attribute.

        Parameters
        ----------
        current_params : SimulationParameters
            The current combination of simulation parameters. This should
            be used to perform any replacement in the
            self.progressbar_message string that will be written in the
            progressbar. Note that string replacement will also work in
            self.progressbar_message for a 'rep_max' field.

        Returns
        -------
        func : (int) -> []
            A function that accepts a single integer argument and can be
            called to update the progressbar.

        Notes
        -----
        The equivalent of this method which is used in the
        simulate_in_parallel method if the
        _get_parallel_update_progress_function method.
        """
        if 'rep_max' not in current_params.parameters:
            current_params.parameters['rep_max'] = self.rep_max

        # If the progressbar_message has any string replacements in the
        # form {some_param} where 'some_param' is a parameter in
        # current_params then it will be replaced by the current value of
        # 'some_param'.
        message = self.progressbar_message.format(**current_params.parameters)

        # By default, the returned function is a dummy function that does
        # nothing
        def update_progress_func(_: int) -> None:
            pass

        if self.update_progress_function_style is None:
            return update_progress_func

        # If self.progress_output_type is equal to 'screen', this will be
        # sys.stdout, otherwise it will be a file object. Note that
        # self.progress_output_type is used when
        # self.update_progress_function_style is either 'text1' or 'text2'.
        output_progress_sink = self._get_progress_output_sink(
            current_params.unpack_index)

        # If the self.update_progress_function_style attribute matches one
        # of the available styles, then update_progress_func will be
        # appropriately set.
        if self.update_progress_function_style == 'text1':  # pragma: no cover
            # We will use the ProgressbarText class
            # noinspection PyArgumentList
            self._pbar = ProgressbarText(self.rep_max,
                                         '*',
                                         message,
                                         output=output_progress_sink,
                                         **self.progressbar_extra_args)
            update_progress_func = self._pbar.progress
            self._pbar.delete_progress_file_after_completion = True
        elif self.update_progress_function_style == 'text2':
            # We will use the ProgressbarText2 class
            # noinspection PyArgumentList
            self._pbar = ProgressbarText2(self.rep_max,
                                          '*',
                                          message,
                                          output=output_progress_sink,
                                          **self.progressbar_extra_args)
            update_progress_func = self._pbar.progress
            self._pbar.delete_progress_file_after_completion = True
        elif self.update_progress_function_style == 'ipython':
            self._pbar = ProgressBarIPython(self.rep_max, message)
            update_progress_func = self._pbar.progress

        elif callable(self.update_progress_function_style) is True:
            # We will use a custom function to update the progress. Note
            # that we call self.update_progress_function_style to return
            # the actual function that will be used to update the
            # progress. That is, the function stored in
            # self.update_progress_function_style should basically do what
            # _get_update_progress_function is supposed to do.

            # pylint: disable=E1102
            # noinspection PyCallingNonCallable
            update_progress_func = self.update_progress_function_style(
                self.rep_max, self.progressbar_message)  # pragma: no cover

        return update_progress_func

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _get_parallel_update_progress_function(
            self) -> ProxybarData:  # pragma: no cover
        """
        Return a function that should be called to update the
        progressbar for the simulation of the current parameters.

        This method is only called in the 'simulate_in_parallel' method.

        The returned function accepts a single argument, corresponding to
        the number of iterations executed so far.

        # The progressbar used to get the returned function depend on the
        # value of the self.update_progress_function_style attribute.

        Returns
        -------
        list[int,int]
            List with the proxybar client_id, ip and port.

        Notes
        -----
        While the _get_serial_update_progress_function method receives the
        "current parameters" this _get_parallel_update_progress_function
        receives the full parameters. The reason for this is that when the
        simulation is performed in parallel multiple process (for different
        parameters) will update the same progressbar. Therefore, it does
        not make sense to perform replacements in the progressbar message
        based on current parameters.
        """
        if self.update_progress_function_style is not None:
            if self._pbar is None:
                parameters = self.params.parameters
                # If the progressbar_message has any string replacements
                #  in the form {some_param} where 'some_param' is a
                # parameter in 'full_params' then it will be replaced by
                #  the value of 'some_param'.
                message = self.progressbar_message.format(**parameters)

                sleep_time = 1.0
                if self.progress_output_type == 'screen':
                    filename = None
                else:
                    # Lets update the progress every 30 seconds
                    sleep_time = 30.0
                    filename = '{0}_progress.txt'.format(
                        self._results_base_filename)

                self._pbar = ProgressbarZMQServer(
                    message=message,
                    sleep_time=sleep_time,
                    filename=filename,
                    **self.progressbar_extra_args)

            # Note that this will be an object of the ProgressbarZMQClient
            # class, but it behaves like a function.
            proxybar = \
                self._pbar.register_client_and_get_proxy_progressbar(
                    self.rep_max)
            proxybar_data = [proxybar.client_id, proxybar.ip, proxybar.port]
        else:
            return None

        return proxybar_data

    @property
    def elapsed_time(self) -> float:
        """
        Get the simulation elapsed time. Do not set this value.

        Returns
        -------
        float
            The elapsed time.
        """
        return pretty_time(self._elapsed_time)

    @property
    def runned_reps(self) -> List[int]:
        """
        Get method for the runned_reps property.

        Returns
        -------
        list[int]
            The value of the runned_reps property, which stores the
            number of runned repetitions for each unpacked parameters
            combination.
        """
        return self._runned_reps

    # This method is called when the SimulationRunner class is pickled
    def __getstate__(self):  # pragma: no cover
        # We will pickle everything as default, except for the "_pbar"
        # member variable that will not be pickled. The reason is that
        # it may be a ProgressbarZMQServer object, which cannot be
        # pickled (uses ZMQ sockets).
        state = dict(self.__dict__)
        del state['_pbar']
        return state

    # def get_runned_reps_fix_params(
    #         self,
    #         fixed_params_dict=None):  # pragma: no cover
    #     """
    #     Get the number of runned repetitions for a given set of parameters.

    #     You can get a list of the number of repetitions combination of
    #     transmit parameters with the "runned_reps" property. However, if
    #     you have more then one transmit parameter set to be unpacked it
    #     might be difficult knowing which element in the list corresponds to
    #     the simulation for a given set of transmit parameters. By using the
    #     get_runned_reps_fix_params method you will be the number
    #     repetitions for the desired set of transmit parameters.

    #     Parameters
    #     ----------
    #     fixed_params_dict : dictionary
    #     """
    #     if fixed_params_dict is None:
    #         fixed_params_dict = dict()

    #     indexes = self.params.get_pack_indexes(fixed_params_dict)
    #     runned_reps_subset = np.array(self.runned_reps)[indexes]
    #     return runned_reps_subset

    # noinspection PyUnboundLocalVariable
    def _simulate_for_current_params_common(
            self,
            current_params: SimulationParameters,
            update_progress_func: UpdateFunction = lambda value: None
    ) -> Tuple[int, SimulationResults, str]:  # pragma: no cover
        """
        Parameters
        ----------
        current_params : SimulationParameters
            The current parameters
        update_progress_func : (int) -> []
            The function that can be called to update the current progress.

        Returns
        -------
        (int, SimulationResults, str)
            The value of `current_rep`, the current results as a
            SimulationResults object, and the name of the file storing
            partial results.
        """
        # Implement the _on_simulate_current_params_start method in a
        # subclass if you need to run code before the _run_simulation
        # iterations for each combination of simulation parameters.
        self._on_simulate_current_params_start(current_params)

        # First we try to Load the partial results for the current
        # parameters.
        try:
            # Name of the file where the partial results will be saved
            if self._results_base_filename is not None:
                partial_results_filename = get_partial_results_filename(
                    self.results_base_filename, current_params,
                    self.partial_results_folder)
            else:
                # If __results_base_filename is None there is also no
                # partial results to load. Therefore, lets raise an
                # IOError here to go to the except catching
                raise IOError()

            # If loading partial results succeeds, then we will have
            # partial results. If it fails because the file does not
            # exist, this will thrown a IOError exception and we will
            # execute the except block instead.
            current_sim_results = SimulationResults.load_from_file(
                partial_results_filename)
            num_skipped_reps_result = Result.create("num_skipped_reps",
                                                    Result.SUMTYPE, 0)
            current_sim_results.add_result(num_skipped_reps_result)

            # Note that at this point we have successfully loaded the
            # partial results from the file. However, we still need to
            # make sure it matches our current parameters. It it does
            # not match the user likely used a wrong name and we raise
            # an exception to stop the simulation.
            #
            # NOTE: If the type of this exception is changed in the
            # future make sure it is not caught in the except block.
            if not current_params == current_sim_results.params:
                err_msg = ("Partial results loaded from file does not match"
                           " current parameters. \nfile: {0}")
                raise ValueError(err_msg.format(partial_results_filename))

            # The current_rep will be set to the value or run
            # repetitions in the loaded partial results. This means
            # that the "while" statement after this try/except block
            # will have a head start and if current_rep is greater than
            # or equal to rep_max the while loop won't run at all.
            # noinspection PyUnresolvedReferences
            current_rep = current_sim_results.current_rep

        # If loading partial results failed then we will run the FIRST
        # repetition here and the "while" statement after this
        # try/except block will run as usual.
        except IOError:
            # Perform the first iteration of _run_simulation
            current_sim_results = \
                self.__run_simulation_and_track_elapsed_time(
                    current_params)
            # Add the extra 'num_skipped_reps' Result.
            current_sim_results.add_new_result('num_skipped_reps',
                                               Result.SUMTYPE, 0)
            current_rep = 1

        last_tic = time()
        # Run more iterations until one of the stop criteria is
        # reached. Note that if partial results were loaded successfully
        # from file and they already achieve the stop criteria then the
        # while loop below will not run.
        while (self._keep_going(current_params, current_sim_results,
                                current_rep) and current_rep < self.rep_max):
            # xxxxxxxxxx Run one repetition of the simulation xxxxxxxxxxxxx
            try:
                # Run one repetition of the `_run_simulation` and merge the
                # new results. If `_run_simulation` raises a SkipThisOne
                # exception, then we do not increase current_rep or the
                # current progress, since there is no new result to merge.
                current_sim_results.merge_all_results(
                    self.__run_simulation_and_track_elapsed_time(
                        current_params))

                current_rep += 1
                update_progress_func(current_rep)
            except SkipThisOne:
                # Each time a SkipThisOne exception is raised we increase
                # the num_skipped_reps_reps result to indicate that, but we
                # don't increase the current progress or the
                # current_rep. After that, the while loop will continue the
                # simulation.
                current_sim_results['num_skipped_reps'][-1].update(1)
                # TODO: Maybe log that one repetition was skipped
                # print("\nAlready skipped {0} repetitions").format(
                #     current_sim_results['num_skipped_reps'][-1].get_result())
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            toc = time()
            # Save partial results each 500 iterations as well as each 5
            # minutes
            if ((toc - last_tic > 300 or current_rep % 500 == 0)
                    and self._results_base_filename is not None):
                self.__save_partial_results(current_rep, current_params,
                                            current_sim_results,
                                            partial_results_filename)
                last_tic = time()

        # If the while loop ended before rep_max repetitions (because
        # _keep_going returned false) then set the progressbar to full.
        update_progress_func(self.rep_max)

        # Implement the _on_simulate_current_params_finish method in a
        # subclass if you need to run code after all _run_simulation
        # iterations for each combination of simulation parameters
        # finishes.
        self._on_simulate_current_params_finish(current_params,
                                                current_sim_results)

        # xxxxxxxxxx Save partial results to file xxxxxxxxxxxxxxxxxxxxx
        # Save partial results for current parameters after all repetitions
        if self._results_base_filename is not None:
            self.__save_partial_results(current_rep, current_params,
                                        current_sim_results,
                                        partial_results_filename)
        else:
            partial_results_filename = None
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # This function returns a tuple containing the number of
        # iterations run as well as the SimulationResults object.
        return current_rep, current_sim_results, partial_results_filename

    def _simulate_for_current_params_serial(
            self, current_params: SimulationParameters,
            var_print_iter: Iterator[int]
    ) -> Tuple[int, SimulationResults, str]:
        """
        Simulate (serial) for the current parameters.

        Parameters
        ----------
        current_params : SimulationParameters
            The current parameters
        var_print_iter : iterator
            An iterator obtained from calling the
            __get_print_variation_iterator method.

        Returns
        -------
        (int, SimulationResults, str)
            The value of `current_rep`, the current results as a
            SimulationResults object, and the name of the file storing
            partial results.
        """
        # (maybe) Print the current variation. This is something like
        # ------------- Current Variation: 4/84 ------------
        next(var_print_iter)

        update_progress_func = self._get_serial_update_progress_function(
            current_params)

        return self._simulate_for_current_params_common(
            current_params, update_progress_func)

    # This method is run in another process. Therefore, the python coverage
    # program cannot see that it is actually used (it is used when the
    # simulate_in_parallel method is tested). Because of that we add the
    # pragma line here.
    @staticmethod
    def _simulate_for_current_params_parallel(
            obj: "SimulationRunner",
            current_params: SimulationParameters,
            proxybar_data=None
    ) -> Tuple[int, SimulationResults, str]:  # pragma: no cover
        """
        Simulate (parallel) for the current parameters.

        Parameters
        ----------
        obj : SimulationRunner
            The same as the self parameter in regular methods. The reason
            that this method is set to static is to allow it to be pickled.
        current_params : SimulationParameters
            The current parameters
        proxybar_data : (int,str,int) | None
                The elements are the "client_id" (and int), the "ip" (a
                string with an IP address) and the "port". This data should
                be used to create a ProgressbarZMQClient object that can be
                used to update the progressbar (via a ZMQ socket)

        Returns
        -------
        (int, SimulationResults, str)
            The value of `current_rep`, the current results as a
            SimulationResults object, and the name of the file storing
            partial results.
        """
        # xxxxxxxxxx Function to update the progress xxxxxxxxxxxxxxxxxx
        if proxybar_data is None:

            def update_progress_func(_):
                pass
        else:
            client_id, ip, port = proxybar_data  # pylint: disable=W0633
            proxybar = ProgressbarZMQClient(client_id, ip, port)
            update_progress_func = proxybar.progress
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # pylint: disable= W0212
        # noinspection PyProtectedMember
        return obj._simulate_for_current_params_common(current_params,
                                                       update_progress_func)

    def __get_print_variation_iterator(self,
                                       num_variations: int,
                                       start: int = 0) -> Iterator[int]:
        """
        Returns an iterator that prints the current variation each time
        it's "next" method is called.

        Create the var_print_iter Iterator Each time the 'next' method of
        var_print_iter is called it will print something like

        ------------- Current Variation: 4/84 ------------

        which means the variation 4 of 84 variations.

        Parameters
        ----------
        num_variations : int
            The number of different variations.
        start : int (default is 1)
            The index of the first variation.

        Returns
        -------
        iterator
            An iterator that can be called to print the current variation.
        """
        if (self.update_progress_function_style is None
                or self.progress_output_type != 'screen'
                or self.update_progress_function_style == 'ipython'):
            for _ in itertools.repeat(''):
                yield 0
        else:  # pragma: no cover
            variation_pbar = ProgressbarText3(num_variations,
                                              progresschar='-',
                                              message="Current Variation:")

            for i in range(start, num_variations):
                variation_pbar.progress(i + 1)
                print()  # print a new line
                yield i

    def simulate(self, param_variation_index: Optional[int] = None) -> None:
        """
        Performs the full Monte Carlo simulation (serially).

        Implements the general code for every simulation. Any code
        specific to a single simulation iteration must be implemented in
        the _run_simulation method of a subclass of SimulationRunner.

        The main idea behind the SimulationRunner class is that the general
        code in every simulator is implemented in the SimulationRunner
        class, more specifically in the `simulate` method, while the
        specific code of a single iteration is implemented in the
        _run_simulation method in a subclass.

        Parameters
        ----------
        param_variation_index : int, optional
            If not provided, the full simulation will be run. If this is
            provided the simulation will be run only for the parameters
            variation with index given by `param_variation_index`. In that
            case, calling the set_results_filename method before the
            simulate method is required since only the partial results will
            be saved.

        See Also
        --------
        simulate_in_parallel
        """
        if param_variation_index is not None:  # pragma: no cover
            # Maybe even though param_variation_index is a valid integer it
            # was passed as a string. Let's try to convert whatever we have
            # to an integer.
            param_variation_index = int(param_variation_index)

        # xxxxxxxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.__tic = time()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Store rep_max in the results object xxxxxxxxxxxxxxxxxxxxxxx
        self.results.rep_max = self.rep_max
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Store the Simulation parameters in the SimulationResults object.
        # With this, the simulation parameters will be available for
        # someone that has the SimulationResults object (loaded from a
        # file, for instance).
        self.results.set_parameters(self.params)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Implement the _on_simulate_start method in a subclass if you need
        # to run code at the start of the simulate method.
        self._on_simulate_start()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Get the number of variations of the transmit parameters
        num_variations = self.params.get_num_unpacked_variations()

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxx Start of the code unique to the serial version xxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Create the var_print_iter Iterator
        # Each time the 'next' method of var_print_iter is called it will
        # print something like
        # ------------- Current Variation: 4/84 ------------
        # which means the variation 4 of 84 variations.

        # However, this will only be printed if
        # self.progress_output_type is equal to 'screen'
        if param_variation_index is None:
            var_print_iter = self.__get_print_variation_iterator(
                num_variations)
        else:
            var_print_iter = self.__get_print_variation_iterator(
                num_variations, start=param_variation_index)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Here we can either simulate for a single combination of
        # parameters, if param_variation_index was provided, or for all
        # combinations if it is not provided.
        if param_variation_index is not None:
            if self._results_base_filename is None:
                err_msg = ('The results filename must be set before'
                           ' calling the "simulate" method.')
                raise RuntimeError(err_msg)
            param_comb_list = self.params.get_unpacked_params_list()

            if 0 <= param_variation_index < len(param_comb_list):
                current_params = param_comb_list[param_variation_index]
                self._simulate_for_current_params_serial(
                    current_params, var_print_iter)

        # If param_variation_index is None we will run for all parameters
        # combinations
        else:
            # xxxxx FOR UNPACKED PARAMETERS xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            # Loop through all the parameters combinations
            for current_params in self.params.get_unpacked_params_list():
                (current_rep, current_sim_results, _) \
                    = self._simulate_for_current_params_serial(
                        current_params, var_print_iter)

                # Store the number of repetitions actually ran for the
                # current parameters combination
                self._runned_reps.append(current_rep)
                # Lets append the simulation results for the current
                # parameters
                self.results.append_all_results(current_sim_results)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # Implement the _on_simulate_finish method in a subclass if you
            # need to run code at the end of the simulate method.
            self._on_simulate_finish()

            # xxxxxxx Save the number of runned iterations xxxxxxxxxxxxxxxx
            self.results.runned_reps = self._runned_reps
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Update the elapsed time xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            self.__toc = time()
            self._elapsed_time = self.__toc - self.__tic

            # Also save the elapsed time in the SimulationResults object
            self.results.elapsed_time = self._elapsed_time
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Save the results if results_filename is not None xxxxxx
            if self._results_base_filename is not None:
                self.results.save_to_file(self._results_base_filename)
                # Delete the partial results (this will only delete the
                # partial results if self.delete_partial_results_bool is
                # True)
                self.__delete_partial_results_maybe()
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # The unittests for this method only run if an ipython cluster is
    # started with a profile called "tests".
    def simulate_in_parallel(self, view: ParallelView,
                             wait: bool = True) -> None:  # pragma: no cover
        """
        Same as the simulate method, but the different parameters
        configurations are simulated in parallel.

        Parameters
        ----------
        view : LoadBalancedView | DirectView
            A ´view´ of the IPython engines.
            The parallel processing will happen by calling the 'map' method
            of the provided view to simulate in parallel the different
            configurations of transmission parameters.
        wait : bool
            If True then the self.wait_parallel_simulation method will be
            automatically called at the end of simulate_in_parallel. If
            False, the YOU NEED to manually call
            self.wait_parallel_simulation at some point after calling
            simulate_in_parallel.

        Notes
        -----
        There is a limitation regarding the partial simulation results. The
        partial results files will be saved in the folder where the IPython
        engines are running, since the "saving part" is performed in an
        IPython engine. However, the deletion of the files is not performed
        in by the IPython engines, but by the main python
        program. Therefore, unless the Ipython engines are running in the
        same folder where the main python program will be run the partial
        result files won't be automatically deleted after the simulation is
        finished.
        """
        # xxxxxxxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.__tic = time()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Store rep_max in the results object xxxxxxxxxxxxxxxxxxxxxxx
        self.results.rep_max = self.rep_max  # pylint: disable=W0201
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Store the Simulation parameters in the SimulationResults object.
        # With this, the simulation parameters will be available for
        # someone that has the SimulationResults object (loaded from a
        # file, for instance).
        self.results.set_parameters(self.params)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Implement the _on_simulate_start method in a subclass if you need
        # to run code at the start of the simulate method.
        self._on_simulate_start()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Get the number of variations of the transmit parameters
        num_variations = self.params.get_num_unpacked_variations()

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxx Start of the code unique to the parallel version xxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Progressbar for the parallel simulation xxxxxxxxxxxxxx
        if self.update_progress_function_style is not None:  # pragma: no cover
            # Create the proxy progressbars
            proxybar_data_list = []
            for _ in range(num_variations):
                proxybar_data_list.append(
                    self._get_parallel_update_progress_function())
        else:  # self.update_progress_function_style is None
            # Create the dummy update progress functions
            proxybar_data_list = [None] * num_variations
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxx Perform the actual simulation in asynchronously parallel xxxx
        # NOTE: If this fails because of some pickling error, make sure the
        # class of 'self' (that is, the subclass of SimulationRunner that
        # you are trying to run) is pickle-able.
        self._async_results = view.map(
            # simulate_for_current_params,
            SimulationRunner._simulate_for_current_params_parallel,
            # We need to pass the SimulationRunner
            # object to the IPython engine ...
            [self] * num_variations,
            # ... and we also need to pass the
            # simulation parameters for each engine
            self.params.get_unpacked_params_list(),
            proxybar_data_list,
            block=False)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        if self._pbar is not None:  # pragma: no cover
            self._pbar.start_updater()

        if wait is True:
            self.wait_parallel_simulation()

    def wait_parallel_simulation(self) -> None:  # pragma: no cover
        """
        Wait for the parallel simulation to finish and then update the
        self.results variable (as well as other internal variables).
        """
        # Note that at the end of this method we set self._async_results to
        # None. Therefore, if wait_parallel_simulation is called multiple
        # times nothing will happen.
        if self._async_results is not None:
            # Wait for the tasks (running in the IPython engines) to finish
            self._async_results.wait()

            results = self._async_results.get()
            for reps, r, filename in results:
                self._runned_reps.append(reps)
                self.results.append_all_results(r)
                self._results_base_filename_unpack_list.append(filename)

            # Implement the _on_simulate_finish method in a subclass if you
            # need to run code at the end of the simulate method.
            self._on_simulate_finish()

            # xxxxxxx Save the number of runned iterations xxxxxxxxxxxxxxxx
            self.results.runned_reps = self.runned_reps
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Update the elapsed time xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            self.__toc = time()
            self._elapsed_time = self.__toc - self.__tic

            # Also save the elapsed time in the SimulationResults object
            self.results.elapsed_time = self.elapsed_time
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Save the results if results_base_filename is not None x
            if self._results_base_filename is not None:
                self.results.save_to_file(self.results_base_filename)
                # Delete the partial results (this will only delete the
                # partial results if self.delete_partial_results_bool is
                #  True)
                self.__delete_partial_results_maybe()
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # Stop the progressbar updating progress
            if self.update_progress_function_style is not None:
                # pragma: no cover
                self._pbar.stop_updater()

            # Erase the self._async_results object, since we already got
            #  all information we needed from it
            self._async_results = None

    # noinspection PyMethodMayBeStatic
    def _on_simulate_start(self) -> None:
        """This method is called only once, in the beginning of the the
        simulate method.

        """

    # noinspection PyMethodMayBeStatic
    def _on_simulate_finish(self) -> None:
        """This method is called only once at the end of the simulate method.

        """
    def _on_simulate_current_params_start(self,
                                          current_params: SimulationParameters
                                          ) -> None:
        """
        This method is called once for each simulation parameters
        combination before any iteration of _run_simulation is performed
        (for that combination of simulation parameters).

        Parameters
        ----------
        current_params : SimulationParameters
            The current combination of simulation parameters.

        Notes
        -----
        Because this method will run inside the _run_simulation method,
        which is called in a different process when the simulation is
        performed in parallel, it should only modify member variables that
        are only used inside _run_simulation. If any other variable is
        modified these changes WILL NOT be carried back to the original
        process.
        """

    # noinspection PyMethodMayBeStatic
    def _on_simulate_current_params_finish(
            self, current_params: SimulationParameters,
            current_params_sim_results: SimulationResults) -> None:
        """This method is called once for each simulation parameters
        combination after all iterations of _run_simulation are performed
        (for that combination of simulation parameters).

        Parameters
        ----------
        current_params : SimulationParameters
            The current combination of simulation parameters.
        current_params_sim_results : SimulationResults
            SimulationResults object with the results for the finished
            simulation with the parameters in current_params.

        Notes
        -----
        Because this method will run inside the _run_simulation method,
        which is called in a different process when the simulation is
        performed in parallel, it should only modify member variables that
        are only used inside _run_simulation.
        """


# xxxxxxxxxx SimulationRunner - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
