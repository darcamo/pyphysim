#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module that implements Cell related classes."""

from matplotlib import pylab

from shapes import Coordinate


class Node(Coordinate):
    """Class representing a node in the network.
    """
    def __init__(self, pos):
        """

        Arguments:
        - `pos`: The position of the node (a complex number)
        """
        Coordinate.__init__(self, pos)
        self.plot_marker = '*'
        self.marker_color = 'r'

    def plot_node(self, ax=None):
        """Plot the node using the matplotlib library.

        If an axes 'ax' is specified, then the node is added to that
        axes. Otherwise a new figure and axes are created and the node is
        plotted to that.

        Arguments:
        - `ax`: A matplotlib axes

        """
        stand_alone_plot = False

        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = pylab.axes()
            stand_alone_plot = True

        ax.plot(self.pos.real,
                self.pos.imag,
                marker=self.plot_marker,
                markerfacecolor=self.marker_color,
                markeredgecolor=self.marker_color)

        if stand_alone_plot is True:
            ax.plot()
            pylab.show()


class CellBaseClass(Coordinate):
    """Base class for all cell types.
    """

    def __init__(self, pos, radius, cell_id=None):
        """
        """
        Coordinate.__init__(self, pos)
        self.id = cell_id
        self.radius = radius  # Should be in a Shape class
        self._users = []

    def _get_num_users(self):
        return len(self._users)
    num_users = property(_get_num_users)

    def _get_users(self):
        return self._users
    users = property(_get_users)

    def delete_users(self, ):
        """Delete all users from the cell.
        """
        self._users = []

    def add_user(self, new_user, relative_pos_bool):
        """Adds a new user to the cell.

        Arguments:
        - `new_user`: New user to be added to the cell.
        - `relative_pos_bool`: Indicates if the 'pos' attribute of the user
                               is relative to the center of the cell or not

        """
        TODO: Implement-me


if __name__ == '__main__':
    c = CellBaseClass(0, 1)
    print c
    #n = Node(1+4j)
    #n.plot_node()
