#!/usr/bin/env python
"""Module implementing helper functions for simulators."""

import sys
from typing import Any, List, Optional, Union

from ..progressbar import ProgressbarZMQServer
from .runner import SimulationRunner

try:
    # noinspection PyUnresolvedReferences
    from ipyparallel import Client, LoadBalancedView, DirectView
except ImportError:  # pragma: no cover
    Client = Any
    LoadBalancedView = Any
    DirectView = Any


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Module functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def simulate_do_what_i_mean(
        runner_or_list_of_runners: Union[SimulationRunner,
                                         List[SimulationRunner]],
        folder: Optional[str] = None) -> None:  # pragma: no cover
    """
    This will either call the simulate method or the simulate_in_parallel
    method as appropriated.

    If the 'parameters variation index' was specified in the command
    line, then the 'simulate' method will be called with that index. If
    not, then the simulate method will be called without any index or,
    if there is an ipython cluster running, the simulate_in_parallel
    method will be called.

    Parameters
    ----------
    runner_or_list_of_runners : SimulationRunner | list[SimulationRunner]
        The SimulationRunner object for which either the 'simulate' or the
        'simulate_in_parallel' method will be called. If this is a list,
        then we just call this method individually for each member of the
        list.
    folder : str
        Folder to be added to the python path. This should be the main
        pyphysim folder
    """
    if isinstance(runner_or_list_of_runners, list):
        list_of_runners = runner_or_list_of_runners
        _simulate_do_what_i_mean_multiple_runners(list_of_runners, folder)
    else:
        # xxxxxxxxxx We only have one SimulationRunner obj xxxxxxxxxxxxxxxx
        runner = runner_or_list_of_runners
        _simulate_do_what_i_mean_single_runner(runner, folder, block=True)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def _add_folder_to_ipython_engines_path(
        client: Client, folder: str) -> None:  # pragma: no cover
    """
    Add a folder to sys.path of each ipython engine.

    The list of engines is get as a direct view from 'client'.

    This will also add the folder to the local python path.

    Parameters
    ----------
    client : Client
        The client from which we will get a direct view to access the
        engines.
    folder : str
        The folder to be added to the python path at each engine.
    """
    # Add the folder to the python path of the main application
    sys.path.append(folder)

    # We create a direct view to run coe in all engines
    dview = client.direct_view()
    # Reset the engines so that we don't have variables there from last
    # computations
    dview.execute('%reset')
    dview.execute('import sys')
    # Add the folder to the python path of each engine. We use
    # block=True to ensure that all engines have modified their
    # path to include the folder with the simulator before we
    # continue.
    dview.execute('sys.path.append("{0}")'.format(folder), block=True)


def _simulate_do_what_i_mean_single_runner(
        runner: SimulationRunner,
        folder: Optional[str] = None,
        block: bool = True) -> None:  # pragma: no cover
    """
    This will either call the `simulate` method or the
    `simulate_in_parallel` method as appropriated.

    If the 'parameters variation index' was specified in the command line,
    then the `simulate` method will be called with that index. If not, then
    the `simulate` method will be called without any index or, if there is
    an ipython cluster running, the `simulate_in_parallel` method will be
    called.

    Parameters
    ----------
    runner : SimulationRunner
        The SimulationRunner object for which either the 'simulate' or the
        'simulate_in_parallel' method will be called.
    folder : str, optional
        Folder to be added to the python path. This should be the main
        pyphysim folder
    block : bool, optional
        Passed to the simulate_in_parallel method when the simulation is
        performed in parallel. If this is false, you need to call the
        method 'wait_parallel_simulation' of the runner object at some
        point.
    """
    if runner.command_line_args.index is not None:
        # Perform the simulation (serially) for the desired index
        msg = "Simulation will be run for the parameters variation: {0}"
        print(msg.format(runner.command_line_args.index))
        runner.simulate(runner.command_line_args.index)

    else:
        run_in_parallel = True
        try:
            # If we can get an IPython view that means that the IPython
            # engines are running. In that case we will perform the
            # simulation in parallel
            from ipyparallel import Client
            # cl = Client(profile="ssh")
            # noinspection PyTypeChecker
            cl = Client(profile="default", timeout=2.0)

            if folder is not None:
                _add_folder_to_ipython_engines_path(cl, folder)

            # For the actual simulation we are better using a
            # load_balanced_view
            lview = cl.load_balanced_view()
        except (IOError, ImportError):
            # If we can't get an IPython view then we will perform the
            # simulation serially
            run_in_parallel = False
            lview = None

        if run_in_parallel is True:
            print("Simulation will be run in Parallel")

            # Remove the " - SNR: {SNR}" string in the progressbar message,
            # since when the simulation is performed in parallel we get a
            # single progressbar for all the simulation.
            runner.progressbar_message = 'Elapsed Time: {{elapsed_time}}'
            runner.simulate_in_parallel(lview, wait=block)
        else:
            print("Simulation will be run serially")
            runner.simulate()


def _simulate_do_what_i_mean_multiple_runners(
        list_of_runners: List[SimulationRunner],
        folder: Optional[str] = None) -> None:  # pragma: no cover
    """
    This will either call the `simulate` method or the
    `simulate_in_parallel` method as appropriated.

    If the 'parameters variation index' was specified in the command line,
    then the `simulate` method will be called with that index. If not, then
    the `simulate` method will be called without any index or, if there is
    an ipython cluster running, the `simulate_in_parallel` method will be
    called.

    Parameters
    ----------
    list_of_runners : list[SimulationRunner]
        The `_simulate_do_what_i_mean_single_runner` will be called for
        each object in the list.
    folder : str
        Folder to be added to the python path. This should be the main
        pyphysim folder.
    """
    # If we have a list of SimulationRunner objects, we want two
    # things. First, we want to use the same progressbar for all of
    # them. Second, we want to use 'block=False' for all of them and
    # only later call the wait_parallel_simulation method for each
    # runner.

    # First we will check the progress_output_type variable of the first
    # runner. If it is 'screen' we will create a progressbar that prints
    # the progress to the terminal, otherwise we will create a progressbar
    # the prints the progress to a file.
    if list_of_runners[0].progress_output_type == 'screen':
        # Progress will be printed to the screen
        filename = None
    else:
        # Progress will be printed to this file
        filename = 'multiple_runners_progress.txt'

    # xxxxxxxxxx Create the shared progressbar object xxxxxxxxxxxxxxxxx
    pbar = ProgressbarZMQServer(progresschar='*',
                                message="Elapsed Time: {elapsed_time}",
                                sleep_time=1,
                                filename=filename)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Start the simulation for each runner xxxxxxxxxxxxxxxxx
    add_folder_to_path = True
    for runner in list_of_runners:
        runner._pbar = pbar  # pylint: disable= W0212
        if add_folder_to_path is True:
            _simulate_do_what_i_mean_single_runner(runner, folder, block=False)
            add_folder_to_path = False
        else:
            _simulate_do_what_i_mean_single_runner(runner, None, block=False)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Wait for the simulation of each runner to finish xxxxxxxxxx
    for runner in list_of_runners:
        runner.wait_parallel_simulation()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx Module Functions - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
