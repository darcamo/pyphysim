#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module that implements Cell related classes."""

from matplotlib import pylab
from collections import Iterable
import numpy as np
from numpy.random import rand

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

    def __init__(self, pos, radius, cell_id=None, rotation=0):
        """
        """
        Node.__init__(self, pos)
        Shape.__init__(self, pos, radius, rotation)
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

    def add_random_user(self, user_color=None, min_dist_ratio=0):
        """Adds a user randomly located in the cell.

        The variable `userColor` can be any color that the plot command and
        friends can understand. If not specified the default value of the
        node class will be used.

        Arguments:
        - `user_color`:
        - `min_dist_ratio`: Minimum allowed (relative) distance betweem the
                            cell center and the generated random user. The
                            value must be between 0 and 0.7.

        """
        # Creates a new user. Note that this user can be invalid (outside
        # the cell) or not.
        new_user = Node(self.pos + complex(2 * (rand() - 0.5) * self.radius, 2 * (rand() - 0.5) * self.radius))

        while (not self.is_point_inside_shape(new_user.pos) or (self.calc_dist(new_user) < (min_dist_ratio * self.radius))):
            # Create another, since the previous one is not valid
            new_user = Node(self.pos + complex(2 * (rand() - 0.5) * self.radius, 2 * (rand() - 0.5) * self.radius))

        if user_color is not None:
            new_user.marker_color = user_color

        # Finally add the user to the cell
        self.add_user(new_user)

    def add_random_users(self, num_users, user_color=None, min_dist_ratio=0):
        """Add `num_users` users randomly located in the cell.

        Arguments:
        - `num_users`: Number of users to be added to the cell.
        - `user_color`:
        - `min_dist_ratio`: Minimum allowed (relative) distance betweem the
                            cell center and the generated random user. The
                            value must be between 0 and 0.7.
        """
        for k in range(num_users):
            self.add_random_user(user_color, min_dist_ratio)

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
                       color=self.marker_color,
                       horizontalalignment='center',
                       verticalalignment='center')

        # Now we plot all the users in the cell
        for user in self.users:
            user.plot_node(ax)


class Cell(Hexagon, CellBaseClass):
    """Class representing an hexagon cell.
    """

    def __init__(self, pos, radius, cell_id=None, rotation=0):
        """
        """
        CellBaseClass.__init__(self, pos, radius, cell_id, rotation)

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


class Cluster(Shape):
    """Class representing a cluster of Hexagonal cells.

    Valid cluster sizes are given by the formula
       N = i^2+i*j+j^2
    where i and j are interger numbers. The allowed values in the Clusterm
    class are sumarized below with the corresponding values of i and j.
    | i,j |  N |
    |-----+----|
    | 1,0 |  1 |
    | 1,1 |  3 |
    | 2,0 |  4 |
    | 2,1 |  7 |
    | 3,1 | 13 |
    | 3,2 | 19 |

    """
    _ii_and_jj = {1: (1, 0),
                  3: (1, 1),
                  4: (2, 0),
                  7: (2, 1),
                  13: (3, 1),
                  19: (3, 2)}

    # Property to get the Cluster radius. The radius property is inherited
    # from shape. However, since here it represents the cluster radius,
    # which depends on the number of cells and the cell radius, then we
    # disable setting this property.
    radius = property(Shape._get_radius)

    def __init__(self, cell_radius, num_cells, pos=0 + 0j, cluster_id=None):
        """

        Arguments:
        - `cell_radius`: Radius of the cells in the cluster.
        - `num_cells`: Number of cells in the cluster
        - `pos`: Central Position of the Cluster
        - `cluster_id`: ID of the cluster
        """
        Shape.__init__(self, pos, radius=0, rotation=0)

        self.cluster_id = cluster_id
        self._cell_radius = cell_radius

        # Cells in the cluster
        self._cells = []

        cell_positions = Cluster._calc_cell_positions(cell_radius, num_cells)
        # Correct the positions to take into account the grid central
        # position.
        cell_positions[:, 0] = cell_positions[:, 0] + self.pos

        # Finally, create the cells at the specified positions (also
        # rotated)
        for index in range(num_cells):
            cell_id = index + 1
            self._cells.append(Cell(cell_positions[index, 0],
                                    cell_radius,
                                    cell_id,
                                    cell_positions[index, 1]))
        # Calculates the cluster radius.
        #
        # The radius of the cluster is defined as half the distance from one
        # cluster to another. That is, if you plot multiple clusters and one
        # circle positioned in each cluster center with radius equal to the
        # cluster radius, the circles should be tangent to each other.
        self._radius = Cluster._calc_cluster_radius(num_cells, cell_radius)
        # Calculates the cluster external radius.
        self._external_radius = self._calc_cluster_external_radius(cell_radius)

    # Property to get the cluster external radius
    # The cluster class also has a external_radius parameter that
    # corresponds to the radius of the smallest circle that can completely
    # hold the cluster inside of it.
    def _get_external_radius(self):
        return self._external_radius
    external_radius = property(_get_external_radius)

    # Property to get the number of users in the cluster.
    def _get_num_users(self):
        num_users = [cell.num_users for cell in self._cells]
        return np.sum(num_users)
    num_users = property(_get_num_users)

    # property to get the number of _cells in the cluster
    def _get_num_cells(self):
        return len(self._cells)

    num_cells = property(_get_num_cells)

    def get_all_users(self):
        """Return all users in the cluster.
        """
        all_users = []
        for cell in self._cells:
            all_users.extend(cell.users)
        return all_users

    def delete_all_users(self):
        """Delete all users in every cell of the cluster.
        """
        for cell in self._cells:
            cell.delete_all_users()

    @staticmethod
    def _get_ii_and_jj(num_cells):
        """Valid cluster sizes are given by the formula
                N = i^2+i*j+j^2

        where i and j are integer numbers and "N" is the number of cells in
        the cluster. This static function returns the values "i" and "j"
        for a given "N". The values are summarized below.
            | i,j |  N |
            |-----+----|
            | 1,0 |  1 |
            | 1,1 |  3 |
            | 2,0 |  4 |
            | 2,1 |  7 |
            | 3,1 | 13 |
            | 3,2 | 19 |

        NOTE: If num_cells is not in the table above then (0, 0) will be
        returned.

        Arguments:
        - `num_cells`: Number of cells in the cluster.

        """
        return Cluster._ii_and_jj.get(num_cells, (0, 0))

    @staticmethod
    def _calc_cell_positions(cell_radius, num_cells):
        """Helper function used by the ClusterClass.

        The calc_cell_positions method calculates the position (and
        rotation) of the 'num_cells' different cells, each with radius
        equal to 'cell_radius', so that they properly fit in the cluster.

        Arguments:
        - `cell_radius`: Radius of each cell in the cluster.
        - `num_cells`: Number of cells in the cluster.

        """
        # The first column in cell_positions has the cell positions
        # (complex number) and the second column has the cell rotation
        # (only the real part is considered)
        cell_positions = np.zeros([num_cells, 2], dtype=complex)
        cell_height = cell_radius * np.sqrt(3.) / 2.

        # xxxxx Get the positions of cells from 2 to 7 xxxxxxxxxxxxxxxxxxxx
        # angles_first_ring -> 30:60:330
        angles_first_ring = np.linspace(np.pi / 6., 11. * np.pi / 6., 6)
        max_value = min(num_cells, 7)
        for index in range(1, max_value):
            cell_positions[index, 0] = 2 * cell_height * np.exp(1j * angles_first_ring[index - 1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Get the positions of cells from 8 to 13 xxxxxxxxxxxxxxxxxxxx
        if num_cells > 7:
            angles_second_ring_A = np.linspace(0, np.pi * 5. / 3., 6)
            max_value = min(num_cells, 13)
            for index in range(7, max_value):
                cell_positions[index, 0] = 3 * cell_radius * np.exp(1j * angles_second_ring_A[index - 7])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # FIXME: Positions from cells 14 to 19 are not correct
        # xxxxx Get the positions of cells from 14 to 19 xxxxxxxxxxxxxxxxxxxx
        if num_cells > 13:
            angles_second_ring_B = angles_first_ring
            max_value = min(num_cells, 19)
            for index in range(13, max_value):
                cell_positions[index, 0] = 4 * cell_height * np.exp(1j * angles_second_ring_B[index - 13])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # The cell positions calculated up to now do not consider
        # rotation. Lets use the rotate function of the Shape class to
        # rotate the coordinates.
        cell_positions[:, 0] = Shape._rotate(cell_positions[:, 0], -30)
        cell_positions[:, 1] = 30

        # The coordinates of the cells calculated up to now consider the
        # center of the first cell as the origin. However, we want the
        # center of the cluster to be the origin. Therefore, lets calculate
        # the central position of the cluster and then correct all
        # coordinates to move the center of the cluster to the origin.
        central_pos = np.sum(cell_positions, axis=0) / num_cells
        # We correct only the first column, which is the position
        cell_positions[:, 0] = cell_positions[:, 0] - central_pos[0]

        return cell_positions

    @staticmethod
    def _calc_cluster_radius(num_cells, cell_radius):
        """Calculates the "cluster radius" for a cluster with "num_cells"
        cells, each cell with radius equal to "cell_radius". The cluster
        "radius" is equivalent to half the distance between two clusters.

        Arguments:
        - `num_cells`: Number of cells in the cluster.
        - `cell_radius`: Radius of each cell in the cluster.

        """
        cell_height = cell_radius * np.sqrt(3.0) / 2.0
        # In the Rappaport book we have
        # the formula
        #       N = i^2+i*j+j^2
        # where N is the number of cells in a cluster and "i" and "j" are
        # two integer numbers. For each valid value of N we set the "ii" and
        # "jj" variables appropriately.
        (ii, jj) = Cluster._get_ii_and_jj(num_cells)

        # Considering one cluster with center located at the origin, we can
        # calculate the center of another cluster using the "ii" and "jj"
        # variables (see the Rappaport book).
        other_cluster_pos = (cell_height * ((jj * 0.5) + (1j * jj * np.sqrt(3.) / 2.)) + cell_height * ii)

        # Now we can calculate the radius simple getting half the distance
        # from the origin to the center of the other cluster.
        radius = np.abs(other_cluster_pos)
        return radius

    def _calc_cluster_external_radius(self, cell_radius):
        """Calculates the radius of the smallest circle that can completely
        hold the cluster inside of it. This circle should touch only the
        most external vertexes of the cells in the cluster.

        Get the vertex positions of the last cell

        Arguments:
        - `cell_radius`: Radius of each cell in the cluster.

        """
        vertex_positions = self._cells[-1].vertices
        dists = vertex_positions - self.pos
        external_radius = np.max(np.abs(dists))
        return external_radius

    def plot(self, ax=None):
        """Plot the cluster
        """
        stand_alone_plot = False
        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = pylab.axes()
            stand_alone_plot = True

        # self.fill_face_bool = False
        # self.fill_color = 'r'
        # self.fill_opacity = 0.1
        for cell in self._cells:
            if self.fill_face_bool is True:
                cell.fill_face_bool = True
                cell.fill_color = self.fill_color
                cell.fill_opacity = self.fill_opacity
            else:
                cell.fill_face_bool = False
            cell.plot(ax)

        if stand_alone_plot is True:
            ax.plot()
            pylab.show()



if __name__ == '__main__1':
    c = Cell(0, 1, cell_id=3)
    c.fill_face_bool = True
    print c

    n = Node(0.2 + 0.3j)
    #n.plot_node()

    c.add_user(n)
    #c.add_border_user([90, 130], 0.7,'b')
    c.add_random_users(100, 'b', 0.5)

    c.plot()

if __name__ == '__main__':
    from shapes import *
    from matplotlib import pyplot as plt
    ax = pylab.axes()
    cell_radius = 1
    num_cells = 19
    pos = 0 + 0j
    cluster_id = None
    C = Cluster(cell_radius, num_cells, pos, cluster_id)
    #print Cluster._calc_cell_positions(1, 7).round(4)
    # print C.radius
    # print C.external_radius
    C.fill_face_bool = True
    C.plot(ax)

    # Circle ilustrating the cluster radius
    circ = Circle(C.pos, C.radius)
    circ.plot(ax)

    # Circle ilustrating the cluster external radius
    circ2 = Circle(C.pos, C.external_radius)
    circ2.plot(ax)

    ax.plot()
    plt.axis('equal')
    plt.show()
