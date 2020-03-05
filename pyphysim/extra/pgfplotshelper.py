#!/usr/bin/env python

# pylint: disable-all
"""
This module contains useful functions to create pgfplots code (latex
code using the pgfplots package) from python data.

One example of tex code for a plot using pgfplots is show below

.. code-block:: latex

    \\begin{tikzpicture}
      \\begin{axis}[axis options]
        \\addplot [plot options]
        plot [options]
        coordinates {
          (0, 0)
          (1, 1)
          (2, 4)
          (3, 9)};
        \\addlegendentry{Legend of the last line};
      \\end{axis}
    \\end{tikzpicture}

"""


def generate_pgfplots_plotline(x, y, errors=None, options=None, legend=None):
    """
    This function generates the code corresponding to the "addplot" command
    in a pgfplots plot for the coordinates given in `x` and `y`.

    If the parameter `errors` is also provided then error bars will be
    added in the `y` direction, while options to the addplot command can be
    passed as a string in the `options` argument.

    Parameters
    ----------
    x : np.ndarray | list[float] | list[int]
        The data for the 'x' axis in the plot.
    y : np.ndarray | list[float] | list[int]
        The data for the 'x' axis in the plot
    errors : np.ndarray | list[float] | list[int], optional
        The error for plotting the errorbars.
    options : str
        pgfplot options for the plot line.
        Ex: "color=red,
        solid,
        mark=square,
        mark options={solid}"
    legend : str
        The legend for the plot line.
    """
    import itertools

    # xxxxxxxxxx Creates the coordinates part of the plot line xxxxxxxxxxxx
    points = zip(x, y)
    num_points = min(len(x), len(y))
    if errors is None:
        points_string = "\n".join([str(p) for p in points])
        plot_line = "plot[]\ncoordinates{{{0}}};".format(points_string)
    else:
        error_points = zip(itertools.repeat(0.0, num_points), errors / 2.0)
        points_and_errors_list = [
            "{0} +- {1}".format(a, b) for a, b in zip(points, error_points)
        ]
        points_string = "\n".join(points_and_errors_list)
        plot_line = ("plot[error bars/.cd, y dir = both, y explicit]\n"
                     "coordinates{{{0}}};").format(points_string)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Get the whole addplot line xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if options is None:
        addplot_line = "\\addplot[]\n{0}".format(plot_line)
    else:
        addplot_line = "\\addplot[{1}]\n{0}".format(plot_line, options)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    if legend is not None:
        legend_line = "\n\\addlegendentry{{{0}}};".format(legend)
        addplot_line += legend_line

    return addplot_line
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
