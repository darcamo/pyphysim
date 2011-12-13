# http://code.activestate.com/recipes/299207-console-text-progress-indicator-class/
# CLASS NAME: ProgressbarText
#
# Author: Larry Bates (lbates@syscononline.com)
#
# Written: 12/09/2002
#
# Modified by Darlan Cavalcante Moreira in 10/18/2011
# Released under: GNU GENERAL PUBLIC LICENSE

"""Module docstring"""

__version__ = "$Revision: $"
# $Source$


class DummyProgressbar():
    """Dummy progress bar that don't really do anything."""

    def __init__(self, ):
        pass

    def progress(self, count):
        pass


class ProgressbarText:
    """Class that prints a representation of the current progress as
    text.

    You can set the final count for the progressbar, the character that
    will be printed to represent progress and a small message indicating
    what the progress is related to.

    In order to use this class, create an object outsize a loop and inside
    the loop call the `progress` function with the number corresponding to
    the progress (between 0 and finalcount). Each time the `progress`
    function is called a number of characters will be printed to show the
    progress. Note that the number of printed characters correspond is
    equivalent to the progress minus what was already printed.

    Example:
    >>> pb = ProgressbarText(100, 'o', "Hello Simulation")
    <BLANKLINE>
    ---------------- Hello Simulation ---------------1
        1    2    3    4    5    6    7    8    9    0
    ----0----0----0----0----0----0----0----0----0----0
    >>> pb.progress(20)
    oooooooooo
    >>> pb.progress(40)
    oooooooooo
    >>> pb.progress(50)
    ooooo
    >>> pb.progress(100)
    ooooooooooooooooooooooooo
    """
    def __init__(self, finalcount, progresschar='*', message=''):
        import sys
        self.finalcount = finalcount
        self.blockcount = 0
        self.block = progresschar
        #
        # Get pointer to sys.stdout so I can use the write/flush
        # methods to display the progress bar.
        #
        self.f = sys.stdout
        #
        # If the final count is zero, don't start the progress gauge
        #
        if not self.finalcount:
            return
        if(len(message) != 0):
            bartitle = '\n{0}\n'.format(center_message(message, 50, '-', '', '1'))
        else:
            bartitle = '\n------------------ % Progress -------------------1\n'

        self.f.write(bartitle)
        self.f.write('    1    2    3    4    5    6    7    8    9    0\n')
        self.f.write('----0----0----0----0----0----0----0----0----0----0\n')
        return

    def progress(self, count):
        #
        # Make sure I don't try to go off the end (e.g. >100%)
        #
        count = min(count, self.finalcount)
        #
        # If finalcount is zero, I'm done
        #
        if self.finalcount:
            percentcomplete = int(round(100 * count / self.finalcount))
            if percentcomplete < 1:
                percentcomplete = 1
        else:
            percentcomplete = 100

        blockcount = int(percentcomplete / 2)
        if blockcount > self.blockcount:
            for i in range(self.blockcount, blockcount):
                self.f.write(self.block)
                self.f.flush()

        if percentcomplete == 100:
            self.f.write("\n")
        self.blockcount = blockcount
        return


def center_message(message, length=50, fill_char=' ', left='', right=''):
    """Return a string with `message` centralized and surrounded by
    fill_char.

    Arguments:
    - `message`: The message to be centered
    - `length`: Total length of the centered message (original + any fill)
    - `fill_char`:
    - `left`:
    - `right`:

    >>> print center_message("Hello Progress", 50, '-', 'Left', 'Right')
    Left------------- Hello Progress ------------Right
    """
    message_size = len(message)
    left_size = len(left)
    right_size = len(right)
    fill_size = (length - (message_size + 2) - left_size - right_size)
    left_fill_size = fill_size // 2 + (fill_size % 2)
    right_fill_size = (fill_size // 2)

    new_message = "{0}{1} {2} {3}{4}".format(left,
                                           fill_char * left_fill_size,
                                           message,
                                           fill_char * right_fill_size,
                                           right)
    return new_message


# xxxxx Perform the doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    import doctest
    doctest.testmod()
