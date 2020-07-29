#!/usr/bin/env python

# pylint: disable=E1101
"""
Implement classes to represent the progress of a task.

Use the ProgressbarText class for tasks that do not use multiprocessing,
and the ProgressbarMultiProcessServer class for tasks using
multiprocessing.

Basically, the task code must call the "progress" function to update the
progress bar and pass a number equivalent to the increment in the progress
since the last call. The progressbar must know the maximum value equivalent
to all the progress, which is passed during object creator for
ProgressbarText class.

The ProgressbarMultiProcessServer is similar to ProgressbarText class,
accounts for the progress of multiple processes. For each process you need
to call the register_client_and_get_proxy_progressbar to get a proxy
progressbar, where the maximum value equivalent to all the progress that
will be performed by that process is passed in this proxy creation. Each
process then calls the progress method of the proxy progressbar.

Note that there is also a DummyProgressbar whose progress function does
nothing. This is useful when you want to give the user a choice to show or
not the progressbar such that the task code can always call the progress
method and you only change the progressbar object.
"""

import multiprocessing
import os
import sys
import threading
import time
import warnings
from typing import Any, List, Optional, Tuple, cast, final

from ..util.misc import pretty_time

try:
    from IPython.display import display
    from ipywidgets import FloatProgress, HBox, Label
    _IPYTHON_AVAILABLE = True
except:
    _IPYTHON_AVAILABLE = False

try:
    # noinspection PyUnresolvedReferences
    import zmq
except ImportError:  # pragma: no cover
    # We don't have a fallback for zmq, but only the
    # ProgressbarZMQServer and ProgressbarZMQClient classes require it
    pass

# Type used to store IP address and port number
ClientID = int
IPAddress = str
PortNumber = int

__all__ = [
    'DummyProgressbar', 'ProgressBarBase', 'ProgressbarTextBase',
    'ProgressbarText', 'ProgressbarText2', 'ProgressbarText3',
    'ProgressBarIPython', 'ProgressbarDistributedServerBase',
    'ProgressbarDistributedClientBase', 'ProgressbarMultiProcessServer',
    'ProgressbarMultiProcessClient', 'ProgressbarZMQServer',
    'ProgressbarZMQClient'
]


# If this function is ever used outside this file, then move it to the
# util.misc module.
def center_message(message: str,
                   length: int = 50,
                   fill_char: str = ' ',
                   left: str = '',
                   right: str = '') -> str:
    """
    Return a string with `message` centralized and surrounded by
    `fill_char`.

    Parameters
    ----------
    message: str
        The message to be centered.
    length : int
        Total length of the centered message (original + any fill).
    fill_char : str
        Filling character.
    left : str
        Left part of the filling.
    right : str
       Right part of the filling.

    Returns
    -------
    cent_message : str
        The centralized message.

    Examples
    --------
    >>> print(center_message("Hello World", 50, '-', 'Left', 'Right'))
    Left-------------- Hello World --------------Right
    """
    message_size = len(message)
    left_size = len(left)
    right_size = len(right)
    fill_size = (length - (message_size + 2) - left_size - right_size)
    left_fill_size = fill_size // 2 + (fill_size % 2)
    right_fill_size = (fill_size // 2)

    new_message = "{0}{1} {2} {3}{4}".format(left, fill_char * left_fill_size,
                                             message,
                                             fill_char * right_fill_size,
                                             right)
    return new_message


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx DummyProgressbar - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# noinspection PyMethodMayBeStatic
class DummyProgressbar:  # pragma: no cover
    """
    Dummy progress bar that don't really do anything.

    The idea is that it can be used in place of the :class:`ProgressbarText`
    class, but without actually doing anything.

    See also
    --------
    ProgressbarText
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the DummyProgressbar object.

        This method accepts any argument without errors, but they won't
        matter, since this class does nothing.
        """

    def progress(self, count: int) -> None:
        """This `progress` method has the same signature from the one in the
        :class:`ProgressbarText` class.

        Nothing happens when this method is called.

        Parameters
        ----------
        count : int
            Ignored
        """


# xxxxxxxxxx DummyProgressbar - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class ProgressBarBase:
    """
    Base class for all ProgressBar classes.

    Parameters
    ----------
    finalcount : int
        The total amount that corresponds to 100%. Each time the progress
        method is called with a number that number is added with the
        current amount in the progressbar. When the amount becomes equal to
        `finalcount` the bar will be 100% complete.

    Notes
    -----
    Derived classes should implement :func:`_update_iteration` and
    :func:`_display_current_progress`. Optionally derived class might also
    implement :func:`_perform_initialization` and :func:`_perform_finalizations`
    """
    def __init__(self, finalcount: int):
        """
        Initializes the progressbar object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        """
        self._finalcount = finalcount

        # This will be set to True after the `start` method is called to
        # initialize the progressbar.
        self._initialized = False

        # This will be set to true when the progress reaches 100%. When
        # this is True, any subsequent calls to the `progress` method will
        # be ignored.
        self._finalized = False

        # This variable will store the time when the `start` method was
        # called for the first time (either manually or in the `progress`
        # method. It will be used for tracking the elapsed time.
        self._start_time = 0.0
        # This variable will store the time when the `stop` method was
        # called for the first time (either manually or in the `progress`
        # method. It will be used for tracking the elapsed time.
        self._stop_time = 0.0

    @property
    def finalcount(self):
        return self._finalcount

    @property
    def elapsed_time(self) -> str:
        """
        Get method for the elapsed_time property.

        Returns
        -------
        str
            The elapsed time.
        """
        elapsed_time = 0.0
        if self._initialized is True:
            if self._finalized is False:
                elapsed_time = time.time() - self._start_time
            else:
                elapsed_time = self._stop_time - self._start_time
        return pretty_time(elapsed_time)

    def _count_to_percent(self, count: int) -> float:
        """
        Convert a given count into the equivalent percentage.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount

        Returns
        -------
        percentage : float
            The percentage that `count` is of self.finalcount (between 0
            and 100)
        """
        percentage = (count / float(self._finalcount)) * 100.0
        return percentage

    def _perform_initialization(self) -> None:
        """
        Perform any initializations for the progressbar.

        This method should be implemented in sub-classes if any
        initialization code should be run.
        """

    def _perform_finalizations(self) -> None:  # pragma: nocover
        """
        Perform any finalization (cleanings) after the progressbar stops.

        This method should be implemented in sub-classes if any
        finalization code should be run.
        """

    @final
    def start(self) -> None:
        """
        Start the progressbar.

        This method should be called just before the progressbar is used
        for the first time. Among possible other things, it will store the
        current time so that the elapsed time can be tracked.

        If is automatically called in the `progress` method, if not called
        before.
        """
        if self._initialized is False:
            self._start_time = time.time()
            self._perform_initialization()
            self._initialized = True

    @final
    def stop(self) -> None:
        """
        Stop the progressbar.

        This method is automatically called in the `progress` method when
        the progress reaches 100%. If manually called, any subsequent call
        to the `progress` method will be ignored.
        """
        if self._finalized is False:
            self._stop_time = time.time()

            # When progress reaches 100% we set the internal variable
            # to True so that any subsequent calls to the `progress`
            # method will be ignored.
            self._finalized = True

            self._perform_finalizations()

    # pylint:disable=R0201,W0613
    def _update_iteration(self, count: int) -> None:  # pragma: no cover
        """
        Update the progressbar according with the new `count`.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount
        """
        raise NotImplementedError("Implement this method in a subclass")

    def _display_current_progress(self) -> None:  # pragma: nocover
        """
        Refresh the progress representation.

        This method should be defined in a subclass.
        """
        raise NotImplementedError("Implement this method in a subclass")

    @final
    def progress(self, count: int) -> None:
        """
        Updates the current progress.

        Calling this function will update the the current progress.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount

        Notes
        -----
        How the progressbar is actually represented depends on the
        subclass.  In the subclasses implement the `_update_iteration`
        method to update the current representation of the progressbar and
        the `_update_progress_display` to actually display the current
        progress.
        """
        if self._finalized is False:
            # Start the progressbar. This only have an effect the first
            # time it is called. It initializes the elapsed time tracking
            # and call the _perform_initialization method to perform any
            # initialization.
            self.start()

            # Sanity check. If count is greater then self.finalcount we set
            # it to self.finalcount
            if count > self._finalcount:
                count = self._finalcount

            # Update the progressbar representation. this is up to the
            # subclass. Note that this method should not refresh the
            # progressbar, which is left to the _display_current_progress
            # method.
            self._update_iteration(count)

            # Refresh the progress representation.
            self._display_current_progress()

            # If count is equal to self.finalcount we have reached
            # 100%. In that case, we also write a final newline
            # character.
            if count == self._finalcount:
                self.stop()

    def __call__(self, count: int) -> None:
        """
        Updates the current progress.

        This method is the same as the :meth:`progress`. It is defined so
        that a progressbar object can behave like a function.

        Parameters
        ----------
        count : int
            The new amount of progress.
        """
        # noinspection PyArgumentList,PyTypeChecker
        self.progress(count)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarTextBase - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# The code here and in some of the derived classes is inspired in the code
# located in
# http://nbviewer.ipython.org/url/github.com/ipython/ipython/raw/master/
# /examples/notebooks/Progress%20Bars.ipynb
#
# noinspection PyAbstractClass
class ProgressbarTextBase(ProgressBarBase):  # pylint: disable=R0902,W0223
    """
    Base class for Text progressbars.

    Parameters
    ----------
    finalcount : int
        The total amount that corresponds to 100%. Each time the progress
        method is called with a number that number is added with the
        current amount in the progressbar. When the amount becomes equal to
        `finalcount` the bar will be 100% complete.
    progresschar : str, optional
        The character used to represent progress.
    message : str, optional
        A message to be shown in the top of the progressbar.
    output : File like object
        Object with a 'write' method, which controls where the progress-bar
        will be printed. By default sys.stdout is used, which means that
        the progress will be printed in the standard output.

    Notes
    -----
    Derived classes must implement at least `_update_iteration` and this
    method should update the `prog_bar` member variable with the text
    representation of the progress.
    """
    def __init__(self,
                 finalcount: int,
                 progresschar: str = '*',
                 message: str = '',
                 output: Any = sys.stdout) -> None:
        """
        Initializes the progressbar object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        progresschar : str, optional (default to '*')
            The character used to represent progress.
        message : str, optional
            A message to be shown in the top of the progressbar.
        output : File like object
            Object with a 'write' method, which controls where the
            progress-bar will be printed. By default sys.stdout is used,
            which means that the progress will be printed in the standard
            output.
        """
        super().__init__(finalcount)

        # This will be updated with the progress and should contain the
        # whole string representation of the progressbar.
        self.prog_bar = "Progress Representation"
        self.old_prog_bar = ""

        # character used to represent progress
        self.progresschar = progresschar

        # If output points to a file (and not to stdout) and this is set to
        # True, then the file will be erased after the progress finishes.
        self.delete_progress_file_after_completion = False

        # This should be a multiple of 10. The lower possible value is 40.
        self._width = 50

        # By default, self._output points to sys.stdout so I can use the
        # write/flush methods to display the progress bar.
        self._output = output
        self._message = message  # THIS WILL BE IGNORED

        # If true, an empty line will be printed when the progress finishes
        self._print_empty_line_at_the_end = True

    def __del__(self) -> None:
        """
        Delete the output file if there is any and
        delete_progress_file_after_completion was set to True when the
        progressbar object is deleted.
        """
        # In case the progressbar object is deleted before the progress
        # finishes the `stop` method will not be called and thus the output
        # file (if there is any) would not be deleted even if
        # delete_progress_file_after_completion was set to True. Therefore,
        # we implement the __del__ method here to call the
        # _maybe_delete_output_file method to do that.
        self._maybe_delete_output_file()

    # xxxxxxxxxx width property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Text progressbars have a width property indicating the width (in
    # number of characters) of the full progress.
    @property
    def width(self) -> int:
        """Get method for the width property."""
        return self._width

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _get_percentage_representation(self,
                                       percent: float,
                                       central_message: str = '{percent}%',
                                       left_side: str = '[',
                                       right_side: str = ']') -> str:
        """
        Get the percent representation as a string suitable to the text
        progressbar.

        Parameters
        ----------
        percent : float
            The percentage to be represented.
        central_message : str
            A message that will be in the middle of the percentage bar. If
            there is the label '{percent}' in the central_message it will
            be replaced by the percentage. If there is the label
            '{elapsed_time}' in the central_message it will be replaced by
            the elapsed time. Note that this message should be very small,
            since it hides the progresschars.
        left_side : str
            The left side of the bar.
        right_side : str
            The right side of the bar.

        Returns
        -------
        representation : str
            A string with the representation of the percentage.
        """
        # Remove any fractional part
        percent_done = int(percent)
        elapsed_time = self.elapsed_time

        # Calculates how many characters are spent just for the sides.
        sides_length = len(left_side) + len(right_side)
        # The width should be large enough to contain both the left_side
        # and right_side and still have (reasonable) enough space for the
        # characters representing the progress.
        assert self.width > sides_length + 20

        # Space that will be used bu the characters representing the
        # progress
        all_full = self.width - sides_length

        # Calculates how many characters will be used to represent the
        # `percent_done` value
        num_hashes = int((percent_done / 100.0) * all_full)

        prog_bar = (left_side + self.progresschar * num_hashes + ' ' *
                    (all_full - num_hashes) + right_side)

        # Replace the center of prog_bar with the message
        central_message = central_message.format(percent=percent_done,
                                                 elapsed_time=elapsed_time)
        pct_place = (len(prog_bar) // 2) - (len(str(central_message)) // 2)
        prog_bar = prog_bar[0:pct_place] + central_message + prog_bar[
            pct_place + len(central_message):]

        return prog_bar

    def _maybe_delete_output_file(self) -> None:
        """
        Delete the output file (if there is any) when
        delete_progress_file_after_completion is set to True.
        """
        if self.delete_progress_file_after_completion is True:
            # Try to get the file name associated with the output. If we
            # can get an actual file and
            # delete_progress_file_after_completion is set to True we will
            # delete that file
            try:
                name = self._output.name
                # We will only delete a file if the name does not point to
                # stdout.
                if name != '<stdout>':
                    try:
                        os.remove(name)
                    except OSError:  # Pragma: no cover
                        pass

            except AttributeError:  # pragma: no cover
                # If an attribute error was raised then the output is not a
                # file like object and therefore we don't need to delete
                # any file
                pass

    def _perform_finalizations(self) -> None:
        """
        Perform any finalization (cleanings) after the progressbar stops.
        """
        if self._print_empty_line_at_the_end is True:
            # Print an empty line after the last iteration to be consistent
            # with the ProgressbarText class
            self._output.write("\n")

            # Flush everything to guarantee that at this point everything
            # is written to the output.
            self._output.flush()

            # This will only delete the output file if self._output
            # actually points to a file and if
            # self.delete_progress_file_after_completion is set to True
            self._maybe_delete_output_file()

    def _display_current_progress(self) -> None:
        """
        Refresh the progress representation.

        All text progressbars should implement the `_update_iteration` to
        update the `prog_bar` member variable with the text representation
        of the progressbar.

        This method is responsible to sending this text representation to
        the output.
        """
        # We will only write the progress if it actually changed since
        # the last time. This is specially useful when the output is a
        # file and it will avoid writing many unnecessary equal lines to
        # the file.
        if self.old_prog_bar != self.prog_bar:
            # Save the current prog_bar variable before it is updated in
            # the _update_iteration method.
            self.old_prog_bar = self.prog_bar

            # We simple change the cursor to the beginning of the line and
            # write the string representation of the prog_bar variable.
            self._output.write('\r')
            self._output.write('{0}'.format(self.prog_bar))

            # Flush everything to guarantee that at this point
            # everything is written to the output.
            self._output.flush()

    # The string representation of a text progressbar should display the
    # whole progressbar
    def __str__(self) -> str:
        return str(self.prog_bar)


# xxxxxxxxxx ProgressbarTextBase - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarText - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# http://
# code.activestate.com/recipes/299207-console-text-progress-indicator-class/
#
# CLASS NAME: ProgressbarText
#
# Original Author of the ProgressbarText class:
# Larry Bates (lbates@syscononline.com)
# Written: 12/09/2002
#
# Modified by Darlan Cavalcante Moreira in 10/18/2011
# Released under: GNU GENERAL PUBLIC LICENSE
class ProgressbarText(ProgressbarTextBase):
    """
    Class that prints a representation of the current progress as text.

    You can set the final count for the progressbar, the character that
    will be printed to represent progress and a small message indicating
    what the progress is related to.

    In order to use this class, create an object outsize a loop and inside
    the loop call the `progress` function with the number corresponding to
    the progress (between 0 and finalcount). Each time the `progress`
    function is called a number of characters will be printed to show the
    progress. Note that the number of printed characters correspond is
    equivalent to the progress minus what was already printed.

    See also
    --------
    DummyProgressbar

    Examples
    --------
    >> pb = ProgressbarText(100, 'o', "Hello Simulation")
    >> pb.start()
    ---------------- Hello Simulation ---------------1
        1    2    3    4    5    6    7    8    9    0
    ----0----0----0----0----0----0----0----0----0----0
    >> pb.progress(20)
    oooooooooo
    >> pb.progress(40)
    oooooooooooooooooooo
    >> pb.progress(50)
    ooooooooooooooooooooooooo
    >> pb.progress(100)
    oooooooooooooooooooooooooooooooooooooooooooooooooo
    """
    def __init__(self,
                 finalcount: int,
                 progresschar: str = '*',
                 message: str = '',
                 output: Any = sys.stdout):
        """
        Initializes the ProgressbarText object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        progresschar : str, optional (default to '*')
            The character used to represent progress.
        message : str, optional
            A message to be shown in the top of the progressbar.
        output : File like object
            Object with a 'write' method, which controls where the
            progress-bar will be printed. By default sys.stdout is used,
            which means that the progress will be printed in the standard
            output.
        """
        super().__init__(finalcount, progresschar, message, output)

        # stores how many characters where already printed in a previous
        # call to the `progress` function
        self.progresscharcount = 0

    def __get_initialization_bartitle(self) -> str:
        """
        Get the progressbar title.

        The title is the first line of the progressbar initialization
        message.

        The bar title is something like the line below

        ------------------- % Progress ------------------1\n

        when there is no message.

        Returns
        -------
        str
            The bar title.

        Notes
        -----
        This method is only a helper method called in the
        `_perform_initialization` method.
        """
        if len(self._message) != 0:
            message = self._message
        else:
            message = '% Progress'

        bartitle = center_message(message, self.width + 1, '-', '', '1\n')
        return bartitle

    def __get_initialization_markers(self) -> Tuple[str, str]:
        """
        The initialization markers 'mark' the current progress in the
        progressbar that will appear below it.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the 'two lines' with the progress
            markers. That is, (marker_line1, marker_line2)

        Notes
        -----
        This method is only a helper method called in the
        `_perform_initialization` method.
        """
        steps = self.width // 10  # This division must be exact

        values1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        values2 = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

        common = "{0}{1}\n"

        line1sep = ' ' * (steps - 1)
        line1 = common.format(line1sep, line1sep.join(values1))

        line2sep = '-' * (steps - 1)
        line2 = common.format(line2sep, line2sep.join(values2))

        return line1, line2

    def _perform_initialization(self) -> None:
        bartitle = self.__get_initialization_bartitle()
        marker_line1, marker_line2 = self.__get_initialization_markers()

        self._output.write(bartitle)
        self._output.write(marker_line1)
        self._output.write(marker_line2)

    def _update_iteration(self, count: int) -> None:
        percentage = self._count_to_percent(count)

        # Set the self.prog_bar variable simply as a string containing as
        # many self.progresschar characters as necessary.
        self.prog_bar = self._get_percentage_representation(percentage,
                                                            left_side='',
                                                            right_side='',
                                                            central_message='')

    @ProgressbarTextBase.width.setter
    def width(self, value: int) -> None:
        """
        Set method for the width property.

        Parameters
        ----------
        value : int
        """
        # If value is lower than 40, the width will be set to 40.
        # If value is not a multiple of 10, width will be set to the
        # largest multiple of 10 which is lower then value.
        #
        # Only allow changing widget if the progressbar has not started yet
        if not self._initialized:
            if value < 40:
                value = 40
            self._width = value - (value % 10)


# xxxxxxxxxx ProgressbarText - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarText2 - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarText2(ProgressbarTextBase):
    """
    Class that prints a representation of the current progress as text.

    You can set the final count for the progressbar, the character that
    will be printed to represent progress and a small message indicating
    what the progress is related to.

    In order to use this class, create an object outsize a loop and inside
    the loop call the `progress` function with the number corresponding to
    the progress (between 0 and finalcount). Each time the `progress`
    function is called a number of characters will be printed to show the
    progress. Note that the number of printed characters correspond is
    equivalent to the progress minus what was already printed.

    Parameters
    ----------
    finalcount : int
        The total amount that corresponds to 100%. Each time the progress
        method is called with a number that number is added with the
        current amount in the progressbar. When the amount becomes equal to
        `finalcount` the bar will be 100% complete.
    progresschar : str, optional (default to '*')
        The character used to represent progress.
    message : str, optional
        A message to be shown in the right of the progressbar. If this
        message contains "{elapsed_time}" it will be replaced by the
        elapsed time.
    output : File like object
        Object with a 'write' method, which controls where the progress-bar
        will be printed. By default sys.stdout is used, which means that
        the progress will be printed in the standard output.
    """
    def __init__(self,
                 finalcount: int,
                 progresschar: str = '*',
                 message: str = '',
                 output: Any = sys.stdout) -> None:
        """
        Initializes the progressbar object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        progresschar : str, optional (default to '*')
            The character used to represent progress.
        message : str, optional
            A message to be shown in the right of the progressbar. If this
            message contains "{elapsed_time}" it will be replaced by the
            elapsed time.
        output : File like object
            Object with a 'write' method, which controls where the
            progress-bar will be printed. By default sys.stdout is used,
            which means that the progress will be printed in the standard
            output.
        """
        super().__init__(finalcount, progresschar, message, output)

    @ProgressbarTextBase.width.setter
    def width(self, value: int) -> None:
        """
        Set method for the width property.

        Parameters
        ----------
        value : int
        """
        # If value is lower than 40, the width will be set to 40.
        # If value is not a multiple of 10, width will be set to the
        # largest multiple of 10 which is lower then value.
        if value < 40:
            value = 40
        self._width = value - (value % 10)

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, val):
        self._message = val

    def _update_iteration(self, count: int) -> None:
        """
        Update the self.prog_bar member variable according with the new
        `count`.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount
        """
        # Convert `count` to the equivalent percentage
        percent_count = self._count_to_percent(count)

        # Update the self.prog_bar variable with (only) the current
        # percentage representation. The message is not included and will
        # be appended after this.
        self.prog_bar = self._get_percentage_representation(
            percent_count,
            central_message='{percent}%',
            left_side='[',
            right_side=']')

        # Append the message to the self.prog_bar variable if there is one
        # (or a default message if there is no message set)..
        if len(self._message) != 0:
            message = self._message.format(elapsed_time=self.elapsed_time)
            self.prog_bar += "  {0}".format(message)
        else:
            self.prog_bar += '  %d of %d complete' % (count, self._finalcount)


# xxxxxxxxxx ProgressbarText2 - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarText3 - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarText3(ProgressbarText2):
    """
    Class that prints a representation of the current progress as text.

    You can set the final count for the progressbar, the character that
    will be printed to represent progress and a small message indicating
    what the progress is related to.

    In order to use this class, create an object outsize a loop and inside
    the loop call the `progress` function with the number corresponding to
    the progress (between 0 and finalcount). Each time the `progress`
    function is called a number of characters will be printed to show the
    progress. Note that the number of printed characters correspond is
    equivalent to the progress minus what was already printed.

     Parameters
    ----------
    finalcount : int
        The total amount that corresponds to 100%. Each time the progress
        method is called with a number that number is added with the
        current amount in the progressbar. When the amount becomes equal to
        `finalcount` the bar will be 100% complete.
    progresschar : str, optional (default to '*')
        The character used to represent progress.
    message : str, optional
        A message to be shown in the progressbar.
    output : File like object
        Object with a 'write' method, which controls where the progress-bar
        will be printed. By default sys.stdout is used, which means that
        the progress will be printed in the standard output.
    """
    def __init__(self,
                 finalcount: int,
                 progresschar: str = '*',
                 message: str = '',
                 output: Any = sys.stdout) -> None:
        """
        Initializes the progressbar object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        progresschar : str, optional (default to '*')
            The character used to represent progress.
        message : str, optional
            A message to be shown in the progressbar.
        output : File like object
            Object with a 'write' method, which controls where the
            progress-bar will be printed. By default sys.stdout is used,
            which means that the progress will be printed in the standard
            output.
        """
        super().__init__(finalcount, progresschar, message, output)

        # The ProgressbarText3 class already prints an empty line after
        # each update. Therefore, there is no need to print an empty line
        # after all the progress has been finished.
        self._print_empty_line_at_the_end = False

    def _update_iteration(self, count: int) -> None:
        """
        Update the self.prog_bar member variable according with the new
        `count`.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount
        """
        full_count = "{0}/{1}".format(count, self._finalcount)

        if len(self._message) != 0:
            self.prog_bar = center_message("{0} {1}".format(
                self._message, full_count),
                                           length=self.width,
                                           fill_char=self.progresschar)
        else:
            self.prog_bar = center_message(full_count,
                                           length=self.width,
                                           fill_char=self.progresschar)


# xxxxxxxxxx ProgressbarText3 - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressBarIPython xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressBarIPython(ProgressBarBase):  # pragma: no cover
    """
    Progressbar for IPython notebooks.

    The progressbar will be rendered using IPython widgets.
    """
    def __init__(self, finalcount: int, message: Optional[str] = None) -> None:
        """
        Initializes the progressbar object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        message : str
            A message to display on the right side of the progressbar. This
            is rendered as Latex and thus can contain math.
        """
        if not _IPYTHON_AVAILABLE:
            raise ModuleNotFoundError(
                'To use ProgressBarIPython please install IPython and ipywidgets'
            )

        super().__init__(finalcount)

        # IPython already provide us a nice widget to represent
        # progressbars.
        self.prog_bar = FloatProgress()

        # If `side_message` is provided then we will add the message as a
        # LatexWidget with the message as the value.
        self._message = Label()
        if message is None:
            self._message.visible = False
        else:
            self._message.value = message
            self._message.disabled = True

        # In order to put the float progressbar and the message side by
        # side we use a container.
        self.container_widget = HBox()
        self.container_widget.children = [self.prog_bar, self._message]

    @property
    def message(self) -> str:
        return self._message.value

    @message.setter
    def message(self, value: str):
        self._message.value = value

    def _update_iteration(self, count: int) -> None:
        """
        Update the self.prog_bar member variable according with the new
        `count`.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount
        """
        percentage = self._count_to_percent(count)

        # Update the IPython progressbar widget
        self.prog_bar.value = percentage

    def _display_current_progress(self) -> None:
        """
        Refresh the progress representation.
        """
        # This method is called everytime the `progress` method is
        # called. However, for progressbar using IPython widgets we only
        # need to display the widget once and IPython will take care of
        # re-displaying it whenever the widget changes. Therefore, we don't
        # need to do anything here and we will display the widget in the
        # `_perform_initialization` method instead, since it is called only
        # once.

    def _perform_initialization(self) -> None:
        """
        Perform any initializations for the progressbar.

        This method should be implemented in sub-classes if any
        initialization code should be run.
        """

        # Display the container with the progressbar and the message.
        # If no message was provided the the text widget inside the
        # container will be invisible
        display(self.container_widget)

    def _perform_finalizations(self) -> None:
        """
        Perform any finalization (cleanings) after the progressbar stops.
        """
        self.prog_bar.bar_style = "success"


# xxxxxxxxxx ProgressBarIPython - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarServerBase xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# pylint:disable=R0902
class ProgressbarDistributedServerBase:
    """
    Base class for progressbars for distributed computations.

    In order to track the progress of distributed computations two classes
    are required, one that acts as a central point and is responsible to
    actually show the progress (the server), and other class that acts as a
    proxy (the client) and is responsible to sending the current progress
    to the server. There will be one object of the "server class" and one
    or more objects of the "client class", each one tracking the progress
    of one of the distributed computations.

    This class is a base class for the "server part", while the
    :class:`ProgressbarDistributedClientBase` class is a base class for the
    "client part".

    For a full implementation, see the :class:`ProgressbarMultiProcessServer`
    and :class:`ProgressbarMultiProcessClient` classes.

    Parameters
    ----------
    progresschar : str
        Character used in the progressbar.
    message : str
        Message writen in the progressbar.
    sleep_time : float
        Time between progressbar updates (in seconds).
    filename : str
        If filename is None (default) then progress will be output to
        sys.stdout. If it is not None then the progress will be output
        to a file with name `filename`. This is usually useful for
        debugging and testing purposes.
    style : str
        The progressbar style. It controls which progressbar is used to display
        progress. It can be either 'text1', 'text2', 'text3', or 'ipython'
    """
    def __init__(self,
                 progresschar: str = '*',
                 message: str = '',
                 sleep_time: float = 1.0,
                 filename: Optional[str] = None,
                 style="text2"):
        """
        Initializes the ProgressbarDistributedServerBase object.

        Parameters
        ----------
        progresschar : str
            Character used in the progressbar.
        message : str
            Message writen in the progressbar.
        sleep_time : float
            Time between progressbar updates (in seconds).
        filename : str
            If filename is None (default) then progress will be output to
            sys.stdout. If it is not None then the progress will be output
            to a file with name `filename`. This is usually useful for
            debugging and testing purposes.
        style : str
            The progressbar style. It controls which progressbar is used to
            display progress. It can be either 'text1', 'text2', 'text3', or
            'ipython'
        """
        self._progresschar = progresschar
        self._message = message

        self._sleep_time = sleep_time
        self._last_id = -1

        self._filename = filename

        # self._manager = multiprocessing.Manager()
        # self._client_data_list: List[Any] = self._manager.list()  # pylint: disable=E1101
        self._client_data_list: List[int] = []

        self._style = style

        # total_final_count will be updated each time the register_*
        # function is called.
        #
        # Note that we use a Manager.Value object to store the value
        # instead of using a simple integer because we want modifications
        # to this value to be seem by the other updating process even after
        # start_updater has been called if we are still in the
        # 'start_delay' time.
        # pylint: disable=E1101
        self._total_final_count: int = 0
        # self._total_final_count = self._manager.Value('L', 0)

        # self._update_process will store the process responsible to update
        # the progressbar. It will be created in the first time the
        # start_updater method is called.
        self._update_process: Optional[threading.Thread] = None
        # self._update_process: Optional[multiprocessing.Process] = None

        # The event will be set when the process updating the progressbar
        # is running and unset (clear) when it is stopped.
        #
        # Starts unset. Is is set in the _update_progress function
        self._is_running = False
        # self._is_running = multiprocessing.Event()

        # Each time the start_updater method is called this variable is
        # increased by one and each time the stop_updater method is called
        # it is decreased by one. A started update process will only be
        # stopped in the stop_updater method if this variables reaches 0.
        # This control is useful so that we can share the same progressbar
        # for multiple SimulationRunner objects.
        self._start_updater_count = 0

        # # Used for time tracking
        # self._tic = multiprocessing.Value('f', 0.0)
        # self._toc = multiprocessing.Value('f', 0.0)

    @property
    def finalcount(self) -> int:
        """
        Get method for the total_final_count property.

        Returns
        -------
        int
            The final count.
        """
        return self._total_final_count

    @property
    def is_running(self):
        return self._is_running

    @property
    def num_clients(self):
        """Number of registered clients"""
        return self._last_id + 1

    def _update_client_data_list(self) -> None:
        """
        This method process the communication between the client and the
        server.

        It should gather the information sent by the clients (proxy
        progressbars) and update the member variable self._client_data_list
        accordingly, which will then be automatically represented in the
        progressbar.
        """
        # Implement this method in a derived class.

    def register_client_and_get_proxy_progressbar(
            self, total_count: int) -> "ProgressbarDistributedClientBase":
        """
        Register a new "client" for the progressbar and returns a new proxy
        progressbar that the client can use to update its progress by
        calling the `progress` method of this proxy progressbar.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% for function.

        Returns
        -------
        obj : Object of a class derived from ProgressbarDistributedClientBase
            The proxy progressbar.
        """
        # Implement this method in a derived class
        #
        # Note: The first thing the implementation of this method in a
        # derived class must do is call the _register_client method to
        # register the new client and get its client_id, like the example
        # below
        # >>> client_id = self._register_client(total_count)
        #
        # After that the implementation of
        # register_client_and_get_proxy_progressbar can create the
        # corresponding proxy progressbar passing the client_id and any
        # other required data.

    def _register_client(self, total_count: int) -> ClientID:
        """
        Register a new "client" for the progressbar and return its `client_id`.

        These returned values must be passed to the corresponding proxy
        progressbar.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% progress for the
            function.

        Returns
        -------
        client_id : ClientID
            The client_id. The function whose process is tracked by the
            ProgressbarMultiProcessServer must update the element
            `client_id` with the current
            count.
        """
        # Set self._total_final_count to the value currently stored plus
        # total_count.
        # Remember that the self._total_final_count variable is actually a
        # proxy to the true value (that is, self._total_final_count is a
        # multiprocessing.Manager.Value object). That is way we need the
        # two lines below.
        self._total_final_count += total_count

        # Update the last_id
        self._last_id += 1

        # client_id that will be used by the function
        client_id = self._last_id

        self._client_data_list.append(0)
        return client_id

    # Only called inside `_update_progress`
    def __create_inner_progressbar(self, output):
        if self._style == 'text1':
            return ProgressbarText(self.finalcount,
                                   self._progresschar,
                                   self._message,
                                   output=output)
        if self._style == 'text3':
            return ProgressbarText3(self.finalcount,
                                    self._progresschar,
                                    self._message,
                                    output=output)
        if self._style == 'ipython':
            return ProgressBarIPython(self.finalcount, self._message)

        # Default style
        return ProgressbarText2(self.finalcount,
                                self._progresschar,
                                self._message,
                                output=output)

    # This method will be run in a different process. Because of this the
    # coverage program does not see that this method in run in the test code
    # even though we know it is run (otherwise no output would
    # appear). Therefore, we put the "pragma: no cover" line in it
    def _update_progress(self,
                         filename: Optional[str] = None,
                         start_delay: float = 0.0) -> None:  # pragma: no cover
        """
        Collects the progress from each registered proxy progressbar and
        updates the actual visible progressbar.

        Parameters
        ----------
        filename : str
            Name of a file where the data will be written to. If this is
            None then all progress will be printed in the standard output
            (default)
        start_delay : float, optional
            Delay in seconds before starting the progressbar. During this
            time it is still possible to register new clients and the
            progressbar will only be shown after this delay..
        """
        if start_delay > 0.0:
            time.sleep(start_delay)

        if self.finalcount == 0:
            warnings.warn('No clients registered in the progressbar')

        if filename is None:
            output = sys.stdout
        else:
            output = open(filename, 'w')

        # pbar = ProgressbarText2(self.total_final_count,
        #                         self._progresschar,
        #                         self._message,
        #                         output=output)
        pbar = self.__create_inner_progressbar(output)

        count = 0
        while count < self.finalcount and self.is_running:
            time.sleep(self._sleep_time)
            # Gather information from all client proxybars and update the
            # self._client_data_list member variable
            self._update_client_data_list()

            # Calculates the current total count from the
            # self._client_data_list
            count = sum(self._client_data_list)

            # Maybe a new client was registered. In that case
            # self.total_final_count changed and we need to reflect this in
            # the pbar.finalcount variable.
            pbar._finalcount = self.finalcount

            # Represents the current total count in the progressbars
            pbar.progress(count)

        # If the self.running event was cleared (because the stop_updater
        # method was called) we most likely exited the while loop before
        # the progressbar was full (count is lower then the total final
        # count). If that is the case, let's set the progressbar to full
        # here.
        if count < self.finalcount:
            pbar.progress(self.finalcount)

        # It may exit the while loop in two situations: if count reached
        # the maximum allowed value, in which case the progressbar is full,
        # or if the self.running event was cleared in another
        # process. Since in the first case the event is still set, we clear
        # it here to have some consistence (a cleared event will always
        # mean that the progressbar is not running).
        self._is_running = False
        # self._toc.value = time.time()

        if self._filename is not None:
            output.close()

    def start_updater(self, start_delay: float = 0.0) -> None:
        """
        Start the process that updates the progressbar.

        Parameters
        ----------
        start_delay : float
            Delay in seconds before starting the progressbar. During this
            time it is still possible to register new clients and the
            progressbar will only be shown after this delay..

        Notes
        -----
        If this method is called multiple times then the `stop_updater`
        method must be called the same number of times for the updater
        process to actually stop.
        """
        if self.is_running is False:
            # self._update_process stores the process responsible to update
            # the progressbar. It may be finished anytime by calling the
            # stop_updater method. Also, it is set as a daemon process so
            # that we don't get errors if the program closes before the
            # process updating the progressbar ends (because the user
            # forgot to call the stop_updater method).
            self._update_process = threading.Thread(
                name="ProgressBarUpdater",
                target=self._update_progress,
                args=(self._filename, start_delay))

            self._update_process.daemon = True

            self._is_running = True
            self._update_process.start()

        self._start_updater_count += 1

    def stop_updater(self, timeout: Optional[float] = None) -> None:
        """
        Stop the process updating the progressbar.

        You should always call this function in your main process (the same
        that created the progressbar) after joining all the processes that
        update the progressbar. This guarantees that the progressbar
        updated any pending change and exited clearly.

        Parameters
        ----------
        timeout : float
            The timeout to join for the process to stop.

        Notes
        -----
        If the `start_updater` was called multiple times the process will
        only be stopped when `stop_updater` is called the same number of
        times.
        """
        self._start_updater_count -= 1
        if self._start_updater_count == 0:
            self._is_running = False
            # self._toc.value = time.time()
            assert (self._update_process is not None)
            self._update_process.join(timeout)

    # # TODO: Check if the duration property work correctly
    # @property
    # def duration(self, ):
    #     """Duration of the progress.

    #     Returns
    #     -------
    #     toc_minus_tic : float
    #         The duration passed until the progressbar reaches 100%.
    #     """
    #     # The progressbar is still running, calculate the duration since
    #     # the beginning
    #     if self.running.is_set():
    #         toc = time.time()
    #     else:
    #         toc = self._toc.value

    #     return toc - self._tic.value


class ProgressbarDistributedClientBase(ProgressBarBase):
    """
    Proxy progressbar that behaves like a ProgressbarText object, but is
    actually updating a shared (with other clients) progressbar.

    The basic idea is that this proxy progressbar has the "progress" method
    similar to the standard ProgressbarText class. However, when this
    method is called it will update a value that will be read by a "server
    progressbar" object which is responsible to actually show the current
    progress.

    Parameters
    ----------
    client_id : ClientID
        The client ID.
    """
    def __init__(self, client_id: ClientID, finalcount: int):
        """
        """
        super().__init__(finalcount)
        self.client_id = client_id

    def _update_iteration(self, count: int) -> None:  # pragma: no cover
        """
        Update the progressbar according with the new `count`.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount
        """
        raise NotImplementedError("Implement this method in a subclass")

    def _display_current_progress(self) -> None:  # pragma: nocover
        """
        Refresh the progress representation.

        This method should be defined in a subclass.
        """
        # There is nothing to do in a client progressbar
        pass


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarMultiProcessServer - START xxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarMultiProcessServer(ProgressbarDistributedServerBase):
    """Class that prints a representation of the current progress of
    multiple process as text.

    While the :class:`ProgressbarText` class only tracks the progress of a
    single process, the :class:`ProgressbarMultiProcessServer` class can
    track the joint progress of multiple processes. This may be used, for
    instance, when you parallelize some task using the multiprocessing
    module.

    Using the ProgressbarMultiProcessServer class requires a little more work
    than using the ProgressbarText class, as it is described in the
    following:

     1. First you create an object of the ProgressbarMultiProcessServer class
        as usual. However, differently from the ProgressbarText class you
        don't pass the `finalcount` value to the progressbar yet.
     2. After that, for each process to be tracked, call the
        :meth:`register_client_and_get_proxy_progressbar` method passing
        the number equivalent to full progress for **that process**. This
        function returns a "proxy progressbar" that behaves like a regular
        ProgressbarText. Pass that proxy progressbar as an argument to that
        process so that it can call its "progress" method. Each process
        that calls the "progress" method of the received proxy progressbar
        will actually update the progress of the main
        ProgressbarMultiProcessServer object.
     3. Start all the processes and call the start_updater method of
        ProgressbarMultiProcessServer object so that the bar is updated by
        the different processes.
     4. After joining all the process (all work is finished) call the
        stop_updater method of the ProgressbarMultiProcessServer object.

    Parameters
    ----------
    progresschar : str
        Character used in the progressbar.
    message : str
        Message writen in the progressbar.
    sleep_time : float
        Time between progressbar updates (in seconds).
    filename : str
        If filename is None (default) then progress will be output to
        sys.stdout. If it is not None then the progress will be output
        to a file with name `filename`. This is usually useful for
        debugging and testing purposes.

    Examples
    --------

    .. code-block:: python

       import multiprocessing
       # Create a ProgressbarMultiProcessServer object
       pb = ProgressbarMultiProcessServer(message="some message")
       # Creates a proxy progressbar for one process passing the value
       # corresponding to 100% progress for the first process
       proxybar1 = pb.register_client_and_get_proxy_progressbar(60)
       # Creates a proxy progressbar for another process
       proxybar2 = pb.register_client_and_get_proxy_progressbar(80)
       # Create the first process passing the first proxy progressbar as
       # an argument
       p1 = multiprocessing.Process(target=some_function, args=[proxybar1])
       # Creates another process
       p2 = multiprocessing.Process(target=some_function, args=[proxybar2])
       # Start both processes
       p1.start()
       p2.start()
       # Call the start_updater method of the ProgressbarMultiProcessServer
       pb.start_updater()
       # Joint the process and then call the stop_updater method of the
       # ProgressbarMultiProcessServer
       p1.join()
       p2.join()
       pb.stop_updater()

    """
    def __init__(self,
                 progresschar: str = '*',
                 message: str = '',
                 sleep_time: float = 1.0,
                 filename: Optional[str] = None,
                 style: str = "text2") -> None:
        """
        Initializes the ProgressbarMultiProcessServer object.

        Parameters
        ----------
        progresschar : str
            Character used in the progressbar.
        message : str
            Message writen in the progressbar.
        sleep_time : float
            Time between progressbar updates (in seconds).
        filename : str
            If filename is None (default) then progress will be output to
            sys.stdout. If it is not None then the progress will be output
            to a file with name `filename`. This is usually useful for
            debugging and testing purposes.
        """
        super().__init__(progresschar, message, sleep_time, filename, style)

        self._manager = multiprocessing.Manager()

        # Change self._client_data_list to be a managed list -> We will pass
        # this list to the clients, which means they will directly update their
        # progress and it will be reflected here
        self._client_data_list: List[Any] = self._manager.list()  # pylint: disable=E1101

    def _update_client_data_list(self) -> None:
        """
        This method process the communication between the client and the
        server.
        """
        # Since the clients will directly modify their progress, then we don't
        # need to implement a `_update_client_data_list` method here

    def register_client_and_get_proxy_progressbar(
            self, total_count: int) -> "ProgressbarMultiProcessClient":
        """
        Register a new "client" for the progressbar and returns a new proxy
        progressbar that the client can use to update its progress by
        calling the `progress` method of this proxy progressbar.

        The function whose process is tracked by the
        ProgressbarMultiProcessServer must must call the `progress` method of
        the returned ProgressbarMultiProcessClient object with the current
        count. This is a little less intrusive regarding the tracked
        function.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% for function.

        Returns
        -------
        obj : ProgressbarMultiProcessClient
            The proxy progressbar.
        """
        client_id = self._register_client(total_count)
        return ProgressbarMultiProcessClient(client_id, self._client_data_list,
                                             total_count)


# Used by the ProgressbarMultiProcessServer class
class ProgressbarMultiProcessClient(ProgressbarDistributedClientBase):
    """
    Proxy progressbar that behaves like a ProgressbarText object,
    but is actually updating a ProgressbarMultiProcessServer progressbar.

    The basic idea is that this proxy progressbar has the "progress" method
    similar to the standard ProgressbarText class. However, when this
    method is called it will update a value that will be read by a
    ProgressbarMultiProcessServer object instead.

    Parameters
    ----------
    client_id : ClientID
        The client ID
    client_data_list : list
        The client data list
    """
    def __init__(self, client_id: ClientID, client_data_list: List[Any],
                 finalcount: int) -> None:
        """Initializes the ProgressbarMultiProcessClient object."""
        super().__init__(client_id, finalcount)
        self._client_data_list = client_data_list

    def _update_iteration(self, count: int) -> None:
        """Updates the proxy progress bar.

        Parameters
        ----------
        count : int
            The new amount of progress.
        """
        self._client_data_list[self.client_id] = count


# xxxxxxxxxx ProgressbarMultiProcessServer - END xxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarZMQServer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarZMQServer(ProgressbarDistributedServerBase):
    """
    Distributed "server" progressbar using ZMQ sockets.

    In order to track the progress of distributed computations two classes
    are required, one that acts as a central point and is responsible to
    actually show the progress (the server), and other class that acts as a
    proxy (the client) and is responsible to sending the current progress
    to the server. There will be one object of the "server class" and one
    or more objects of the "client class", each one tracking the progress
    of one of the distributed computations.

    This class acts like the server. It creates a ZMQ socket which expects
    (string) messages in the form "client_id:current_progress", where the
    client_id is the ID of one client progressbar previously registered
    with the register_client_and_get_proxy_progressbar method while the
    current_progress is a "number" with the current progress of that
    client.

    Note that the client proxybar for this class is implemented in the
    ProgressbarZMQClient class.

    Parameters
    ----------
    progresschar : str
        Character used in the progressbar.
    message : str
        Message writen in the progressbar.
    sleep_time : float
        Time between progressbar updates (in seconds).
    filename : str
        If filename is None (default) then progress will be output to
        sys.stdout. If it is not None then the progress will be output to a
        file with name `filename`. This is usually useful for debugging and
        testing purposes.
    ip : IPAddress
        An string representing the address of the server socket.
        Ex: '192.168.0.117', 'localhost', etc.
    port : PortNumber
        The port to bind the socket.
    """
    def __init__(self,
                 progresschar: str = '*',
                 message: str = '',
                 sleep_time: float = 1.0,
                 filename: Optional[str] = None,
                 ip: IPAddress = 'localhost',
                 port: PortNumber = 7396,
                 style: str = "text2") -> None:
        super().__init__(progresschar, message, sleep_time, filename, style)

        # Create a Multiprocessing namespace
        # pylint: disable=E1101
        # self._ns = self._manager.Namespace()

        # We store the IP and port of the socket in the Namespace, since
        # the socket will be created in a different process
        self._ip = ip  # type: ignore
        self._port = port  # type: ignore

        # This will be set to a ZMQ Context in the _update_progress method
        self._zmq_context: Optional[zmq.Context] = None
        # This will be set to a ZMQ Socket in the _update_progress method
        self._zmq_pull_socket: Optional[zmq.sugar.socket.Socket] = None

    def __repr__(self):
        status = '-> updating' if self.is_running else '-> stopped'
        return f"ProgressbarZMQServer(ip={self.ip}, port={self.port}, num_clients={self.num_clients}) {status}"

    @property
    def ip(self) -> IPAddress:
        """
        Get method for the ip property.

        Returns
        -------
        str
            The string representing the address of the server socket.
        """
        return cast(IPAddress, self._ip)  # type: ignore

    @property
    def port(self) -> PortNumber:
        """
        Get method for the port property.

        Returns
        -------
        int
            The port used.
        """
        return cast(PortNumber, self._port)  # type: ignore

    def register_client_and_get_proxy_progressbar(
            self, total_count: int) -> "ProgressbarZMQClient":
        """
        Register a new client progressbar and return a proxy to it.

        Parameters
        ----------
        total_count : int
            The total count for the client we are registering.

        Returns
        -------
        ProgressbarZMQClient
            The proxy progressbar.
        """
        client_id = self._register_client(total_count)
        proxybar = ProgressbarZMQClient(client_id, self.ip, self.port,
                                        total_count)
        return proxybar

    # noinspection PyUnresolvedReferences
    def _update_progress(self,
                         filename: Optional[str] = None,
                         start_delay: float = 0.0) -> None:  # pragma: no cover
        """
        Collects the progress from each registered proxy progressbar and
        updates the actual visible progressbar.

        Parameters
        ----------
        filename : str
            Name of a file where the data will be written to. If this is
            None then all progress will be printed in the standard output
            (default)
        start_delay : float
            Delay in seconds before starting the progressbar. During this
            time it is still possible to register new clients and the
            progressbar will only be shown after this delay.

        Notes
        -----
        We re-implement it here only to create the ZMQ socket. After that we
        call the base class implementation of _update_progress method. Note
        that the _update_progress method in the base class calls the
        _update_client_data_list and we indeed re-implement this method in
        this class and use the socket created here in that implementation.
        """
        # First we create the context and the socket. Then we bind the
        # socket to the respective ip:port.
        self._zmq_context = zmq.Context()
        self._zmq_pull_socket = self._zmq_context.socket(zmq.PULL)
        self._zmq_pull_socket.bind("tcp://*:%s" % self.port)
        ProgressbarDistributedServerBase._update_progress(self, filename)

    # This method is called in the _update_progress, which is run in a
    # different process. Therefore, the python coverage program does not
    # detect that this method is actually run (and it is) and thus we set
    # the "pragma: no cover" comment here.
    def _update_client_data_list(self) -> None:  # pragma: no cover
        """
        This method process the communication between the client and the
        server.

        This method will read the received messages in the socket which
        were sent by the clients (ProgressbarZMQClient objects) and update
        self._client_data_list variable accordingly. The messages are in
        the form "client_id:current_progress", which is parsed by the
        _parse_progress_message method.

        Notes
        -----
        This method is called inside a loop in the _update_progress method.
        """

        pending_messages = True
        while pending_messages is True and self.is_running:
            try:
                # Try to read a message. If this fail we will get a
                # zmq.ZMQError exception and then pending_messages will be
                # set to False so that we exit the while loop.
                assert (self._zmq_pull_socket is not None)
                message = self._zmq_pull_socket.recv_string(flags=zmq.NOBLOCK)

                # If we are here that means that a new message was
                # successfully received from the client.  Let's call the
                # _parse_progress_message method to parse the message and
                # update the self._client_data_list member variable.
                self._parse_progress_message(message)
            except zmq.ZMQError:
                pending_messages = False

    # This method run in a different process and thus the python coverage
    # program does not detect it is run even when it is run. Therefore, we
    # set the "pragma: no cover" comment here.
    def _parse_progress_message(self,
                                message: str) -> None:  # pragma: no cover
        """
        Parse the message sent from the client proxy progressbars.

        The messages sent from the proxy progressbars are in the form
        'client_id:current_count'. We need to set the element of index
        "client_id" in self._client_data_list to the value of
        "current_count". This method will simply parse the message and
        perform this operation.

        Parameters
        ----------
        message : str
            A string in the form 'client_id:current_count'.
        """
        client_id, current_count = map(int, message.split(":"))
        self._client_data_list[client_id] = current_count


# Used by the ProgressbarZMQServer class
class ProgressbarZMQClient(ProgressbarDistributedClientBase):
    """
    Proxy progressbar that behaves like a ProgressbarText object,
    but is actually updating a ProgressbarZMQServer progressbar.

    The basic idea is that this proxy progressbar has the "progress" method
    similar to the standard ProgressbarText class. However, when this
    method is called it will update a value that will be read by a
    ProgressbarZMQServer object instead.

    Parameters
    ----------
    client_id : ClientID
        The client ID.
    ip : IPAddress
        A string representing the IP address of the server.
    port : PortNumber
        The port number used by the server.
    """
    def __init__(self, client_id: ClientID, ip: IPAddress, port: PortNumber,
                 finalcount: int) -> None:
        super().__init__(client_id, finalcount)
        self.ip = ip
        self.port = port

        # ZMQ Variables: These variables will be set the first time the
        # progress method is called.
        self._zmq_context: Optional[zmq.Context] = None
        self._zmq_push_socket: Optional[zmq.sugar.socket.Socket] = None

    def _update_iteration(self, count: int) -> None:
        """Updates the proxy progress bar.

        Parameters
        ----------
        count : int
            The new amount of progress.
        """
        # The message is a string composed of the client ID and the current
        # count
        message = f"{self.client_id}:{count}"
        assert (self._zmq_push_socket is not None)
        self._zmq_push_socket.send_string(message, flags=zmq.NOBLOCK)

    def _perform_initialization(self) -> None:
        """
        Creates the "push socket", connects it to the socket of the main
        progressbar and then updates the progress.

        This function will be called only in the first time the "progress"
        method is called. Subsequent calls to "progress" will actually
        calls the "_progress" method.

        Parameters
        ----------
        count : int
            The new amount of progress.
        """
        self._zmq_context = zmq.Context()
        self._zmq_push_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_push_socket.connect("tcp://{0}:{1}".format(
            self.ip, self.port))

        # The default LINGER value for a ZMQ socket is -1, which means
        # "wait forever". That means that if the message was not received
        # by the server (the main progressbar) the process with the
        # push_socket will hang. Since we don't want that, we set the
        # LINGER option to 0 so that it does not wait for the message to be
        # received.
        self._zmq_push_socket.setsockopt(zmq.LINGER, 0)

        # self._progress_func = ProgressbarZMQClient._progress
        # # noinspection PyArgumentList,PyTypeChecker
        # self._progress_func(self, self.finalcount)

    def __repr__(self):
        return f"ProgressbarZMQClient(client_id={self.client_id}, ip='{self.ip}', port={self.port}, finalcount={self._finalcount})"


# xxxxxxxxxx ProgressbarZMQServer - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
