#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module that implements Cell related classes."""

from matplotlib import pylab
from collections import Iterable

from shapes import Coordinate, Shape, Hexagon


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


class CellBaseClass(Node, Shape):
    """Base class for all cell types.
    """

    def __init__(self, pos, radius, cell_id=None):
        """
        """
        Node.__init__(self, pos)
        Shape.__init__(self, pos, radius)
        self.plot_marker = '^'
        self.marker_color = 'b'

        self.id = cell_id
        self._users = []

    def _get_num_users(self):
        return len(self._users)
    num_users = property(_get_num_users)

    def _get_users(self):
        return self._users
    users = property(_get_users)

    def delete_all_users(self, ):
        """Delete all users from the cell.
        """
        self._users = []

    def add_user(self, new_user, relative_pos_bool=True):
        """Adds a new user to the cell.

        Arguments:
        - `new_user`: New user to be added to the cell.
        - `relative_pos_bool`: Indicates if the 'pos' attribute of the user
                               is relative to the center of the cell or not

        """
        # TODO: Implement-me
        if isinstance(new_user, Node):
            if relative_pos_bool is True:
                # If the position of the user is relative to the cell, that
                # means that the real and imaginary parts of new_user.pos
                # are in the [-1, 1] range. We need to convert them to an
                # absolute coordinate.
                new_user.pos = new_user.pos * self.radius + self.pos
            if self.is_point_inside_shape(new_user.pos):
                self._users.append(new_user)
            else:
                raise ValueError("User position is outside the cell -> User not added")
        else:
            raise TypeError("User must be Node object.")

    def add_border_user(self, angles, ratio=None, userColor=None):
        """Adds a user at the border of the cell, located at a specified
        angle (in degrees).

        If the `angles` variable is an iterable, one user will be added for
        each value in `angles`.

        Arguments:
        - `angles`:
        - `ratio`:
        - `userColor`:

        """
        if ratio is None:
            ratio = 1.0
        else:
            if (ratio < 0) or (ratio > 1):
                raise ValueError("ratio must be between 0 and 1")

        if not isinstance(angles, Iterable):
            angles = [angles]

        for angle in angles:
            new_user = Node(self.get_border_point(angle, ratio))
            if userColor is not None:
                new_user.marker_color = userColor

            self.add_user(new_user)

    def _plot_common_part(self, ax):
        """Common code for plotting the classes. Each subclass must implement a
        `plot` method in which it calls the command to plot the class shape
        followed by _plot_common_part.

        Arguments:
        - `ax`: A matplotlib axes
        """
        # If self.id is None, plot a single marker at the center of the
        # cell
        if self.id is None:
            self.plot_node(ax)
        else:
            # If self.id is not None, plot the cell ID at the center of the
            # cell
            pylab.text(self.pos.real,
                       self.pos.imag,
                       '{0}'.format(self.id),
                       color=self.marker_color

            )

        # Now we plot all the users in the cell
        for user in self.users:
            user.plot_node(ax)


class Cell(Hexagon, CellBaseClass):
    """Class representing an hexagon cell.
    """

    def __init__(self, pos, radius, cell_id=None):
        """
        """
        CellBaseClass.__init__(self, pos, radius, cell_id)

    def plot(self, ax=None):
        """Plot the cell using the matplotlib library.

        If an axes 'ax' is specified, then the shape is added to that
        axes. Otherwise a new figure and axes are created and the shape is
        plotted to that.

        Arguments:
        - `ax`: A matplotlib axes
        """

        stand_alone_plot = False

        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = pylab.axes()
            stand_alone_plot = True

        # Plot the shape part
        Hexagon.plot(self, ax)
        # Plot the node part as well as the users in the cell
        self._plot_common_part(ax)

        if stand_alone_plot is True:
            ax.plot()
            pylab.show()


if __name__ == '__main__':
    c = Cell(0, 1, cell_id=3)
    c.fill_face_bool = True
    print c

    n = Node(0.2 + 0.3j)
    #n.plot_node()

    c.add_user(n)
    c.add_border_user([90, 130], 0.7,'b')
    c.plot()
