#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module that implements Cell related classes."""

from matplotlib import pylab, patches
from collections import Iterable
import numpy as np
from numpy.random import rand
import itertools

from shapes import Coordinate, Shape, Hexagon, from_complex_array_to_real_matrix


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Node class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Node(Coordinate):
    """Class representing a node in the network.
    """
    def __init__(self, pos, plot_marker='*', marker_color='r'):
        """

        Arguments:
        - `pos`: The position of the node (a complex number)
        """
        Coordinate.__init__(self, pos)
        self.plot_marker = plot_marker
        self.marker_color = marker_color

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
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Cell classes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class CellBase(Node, Shape):
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

    def add_border_user(self, angles, ratio=None, user_color=None):
        """Adds a user at the border of the cell, located at a specified
        angle (in degrees).

        If the `angles` variable is an iterable, one user will be added for
        each value in `angles`. Also, in that case ration and user_color may
        also be iterables with the same length of `angles` in order to
        specify individual ratio and user_color for each angle.

        Arguments:
        - `angles`: Angles for which users will be added (may be a single
                    number or an iterable)
        - `ratio`: The ration (relative distance from cell center) for
                   which users will be added (may be a single number or an
                   iterable)
        - `user_color`: Color of the user's marker.

        """
        # Assures that angle is an iterable
        if not isinstance(angles, Iterable):
            angles = [angles]

        if isinstance(user_color, str):
            user_color = itertools.repeat(user_color)
        else:
            if not isinstance(user_color, Iterable):
                user_color = itertools.repeat(user_color)
        if not isinstance(ratio, Iterable):
            ratio = CellBase._validate_ratio(ratio)
            ratio = itertools.repeat(ratio)
        else:
            ratio = [CellBase._validate_ratio(i) for i in ratio]

        all_data = zip(angles, ratio, user_color)
        for data in all_data:
            angle, ratio, user_color = data
            new_user = Node(self.get_border_point(angle, ratio))
            if user_color is not None:
                new_user.marker_color = user_color
            self._users.append(new_user)

    def add_random_user(self, user_color=None, min_dist_ratio=0):
        """Adds a user randomly located in the cell.

        The variable `user_color` can be any color that the plot command and
        friends can understand. If not specified the default value of the
        node class will be used.

        Arguments:
        - `user_color`: Color of the user's marker.
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
        self.add_user(new_user, relative_pos_bool=False)

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

    @staticmethod
    def _validate_ratio(ratio):
        """Return `ratio` if is valid, 1.0 if `ratio` is None, or throw an
        exception if it is not valid.

        This is a herper method used in the add_border_user method
        implementation.

        Arguments:
        - `ratio`: An scalar.

        Throws:
        - ValueError: If `ratio` is not between 0 and 1.

        """
        # If ratio is None then it was not specified and we assume it to be
        # equal to one (border of the shape). However, if we set ratio to
        # be exactly 1.0 then the is_point_inside_shape method would
        # return false which is probably not what you want. Therefore, we
        # set it to be a little bit lower then 1.0.
        if (ratio is None) or (ratio == 1.0):
            ratio = 1.0 - 1e-15
        else:
            if (ratio < 0) or (ratio > 1):
                raise ValueError("ratio must be between 0 and 1")
        return ratio


class Cell(Hexagon, CellBase):
    """Class representing an hexagon cell.
    """

    def __init__(self, pos, radius, cell_id=None, rotation=0):
        """
        """
        CellBase.__init__(self, pos, radius, cell_id, rotation)

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
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Cluster Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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

    def _get_cell_radius(self):
        return self._cell_radius
    # Property to get the radius of the cells in the cluster.
    cell_radius = property(_get_cell_radius)

    def get_all_users(self):
        """Return all users in the cluster.
        """
        all_users = []
        for cell in self._cells:
            all_users.extend(cell.users)
        return all_users

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
        """Helper function used by the Cluster class.

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

    def _get_outer_vertexes(self, vertexes, central_pos, distance):
        """Filter out vertexes closer to the shape center them `distance`.

        This is a herper method used in the _get_vertex_positions method.

        Arguments:
        - `vertexes`: A numpy array of vertexes.
        - `central_pos`: Central position of the shape.
        - `distance`: A minimum distance. Any vertex that is closer to the
                      shape center then this distance will be removed.

        """
        # Filter function. Returns True for vertexes which are closer to
        # the shape center then distance.
        f = lambda x: np.abs(x - central_pos) > (distance)
        vertexes = vertexes[f(vertexes)]

        # Remove duplicates

        # Float equality test (used implicitly by 'set' to remove
        # duplicates) is not trustable. We lower the precision to make
        # it more trustable but maybe calculating the cluster vertexes
        # like this is not the best way.
        vertexes = frozenset(vertexes.round(12))
        vertexes = np.array([i for i in vertexes])

        # In order to use these vertices for plotting, we need them to be
        # in order (lowest angle to highest)
        vertexes = vertexes[np.argsort(np.angle(vertexes - self.pos))]
        return vertexes

    def _get_vertex_positions(self):
        """Get the vertex positions of the cluster borders.

        This is only valid for cluster sizes from 1 to 19.

        """
        cell_radius = self._cells[0].radius
        if self.num_cells == 1:
            # If the cluster has a single cell, the cluster vertexes are
            # the same as the ones from this single cell. We don't have to
            # do anything else.
            return self._cells[0].vertices

        if self.num_cells < 4:  # From 2 to 3
            start_index = 0
            distance = 0.2 * cell_radius
        elif self.num_cells < 7:  # From 4 to 6
            start_index = 0
            distance = 1.05 * cell_radius
        elif self.num_cells < 11:  # From 7 to 10
            start_index = 1
            distance = 1.8 * cell_radius
        elif self.num_cells < 14:  # From 11 to 13
            start_index = 4
            distance = 2.15 * cell_radius
        elif self.num_cells < 16:  # From 14 to 15
            start_index = 7
            distance = 2.45 * cell_radius
        elif self.num_cells < 20:  # From 16 to 19
            start_index = 7
            distance = 2.80 * cell_radius
        else:
            # Invalid number of cells per cluster
            return np.array([])

        all_vertexes = np.array([cell.vertices for cell in self._cells[start_index:]]).flatten()
        return self._get_outer_vertexes(all_vertexes, self.pos, distance)
    # Note: The _get_vertex_positions method which should return the shape
    # vertexes without translation and rotation and the vertexes property
    # from the Shape class would add the translation and rotation. However,
    # the _get_vertex_positions method in the Cluster class return vertices's
    # that already contains the translation and rotation. Therefore, we
    # overwrite the property here to return the output of
    # _get_vertex_positions.
    vertices = property(_get_vertex_positions)

    def plot(self, ax=None):
        """Plot the cluster.

        Arguments:
        - `ax`:  A matplotlib axes
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

    def plot_border(self, ax=None):
        """Plot only the border of the Cluster.

        Only workers for cluster sizes that can calculate the cluster
        vertices, such as cluster with 1, 7 or 19 cells.

        Arguments:
        - `ax`: A matplotlib axes

        """
        if len(self.vertices) != 0:
            stand_alone_plot = False
            if (ax is None):
                # This is a stand alone plot. Lets create a new axes.
                ax = pylab.axes()
                stand_alone_plot = True

            polygon_edges = patches.Polygon(
                from_complex_array_to_real_matrix(self.vertices),
                True,
                facecolor='none',  # No face
                alpha=1,
                linewidth=2)
            ax.add_patch(polygon_edges)

            if stand_alone_plot is True:
                ax.plot()
                pylab.show()

    def add_random_users(self, cell_ids, num_users=1, user_color=None, min_dist_ratio=0):
        """Adds one or more users to the Cells with the specified cell IDs (the
        first cell has an ID equal to 1.).

        Note: If `cell_ids` is an iterable then the other atributes
        (num_users, user_color and min_dist_ratio) may also be iterable
        with the same length of cell_ids in order to specifying individual
        values for each cell ID.

        Arguments:
        - `cell_ids`: IDs of the cells in the Cluster for which users will
                      be added. The first cell has an ID equal to 1 and
                      `cell_ids` may be an iterable with the IDs of several
                      cells.
        - `num_users`: Number of users to be added to each cell.
        - `user_color`: Color of the user's marker.
        - `min_dist_ratio`: Minimum allowed (relative) distance betweem the
                            cell center and the generated random user. See
                            Cell.add_random_user method for details.

        """
        if isinstance(cell_ids, Iterable):
            if not isinstance(num_users, Iterable):
                num_users = itertools.repeat(num_users)
            if isinstance(user_color, str):
                user_color = itertools.repeat(user_color)
            else:
                if not isinstance(user_color, Iterable):
                    user_color = itertools.repeat(user_color)
            if not isinstance(min_dist_ratio, Iterable):
                min_dist_ratio = itertools.repeat(min_dist_ratio)

            all_data = zip(cell_ids, num_users, user_color, min_dist_ratio)
            for data in all_data:
                self.add_random_users(*data)
        else:
            for index in range(num_users):
                # Note that here cell_ids will be a single value, as well as user_color and min_dist_ratio
                self._cells[cell_ids - 1].add_random_user(user_color, min_dist_ratio)

    def add_border_users(self, cell_ids, angles, ratios=None, user_color=None):
        """Add users to all the cells indicated by `cell_indexes` at the
        specified angle(s) (in degrees) and ratio (relative distance from
        the center to the border of the cell.

        Arguments:
        - `cell_ids`: IDs of the cells in the Cluster for which users will
                      be added. The first cell has an ID equal to 1 and
                      `cell_ids` may be an iterable with the IDs of several
                      cells.
        - `angles`: Angles (in degrees)
        - `ratios`: Ratios (from 0 to 1)
        - `user_color`: Color of the user's marker.

        """
        for cell_id in cell_ids:
            self._cells[cell_id - 1].add_border_user(angles, ratios, user_color)
        # TODO: Finish the implementation. The current implementation
        # simply pass angles, ratios, user_color to each cell in
        # cell_ids. That is, we can't add users in different locations in
        # each cell.

    def remove_all_users(self, cell_id=None):
        """Remove all users from one or more cells.

        If cell_id is an integer > 0, only the users from the cell whose
        index is `cell_id` will be removed. If cell_id is an iterable, then
        the users of cells pointed by it will be removed. If cell_id is
        `None` or not specified, then the users of all cells will be
        removed.

        Arguments:
        - `cell_id`: ID(s) of the cells from which users will be
                     removed. If equal to None, all the users from all
                     cells will be removed.

        """
        if isinstance(cell_id, Iterable):
            for i in cell_id:
                self._cells[i - 1].users.delete_all_users()
        elif cell_id is None:
            for cell in self._cells:
                cell.delete_all_users()
        else:
            self._cells[cell_id - 1].delete_all_users()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Grid Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Grid(object):
    """Class representing a grid of clusters of cells or a single cluster with
    its surrounding cells.

    Valid cluster sizes are given by the formula
          $N = i^2+i*j+j^2$
    where i and j are integer numbers. The values allowed in the Cluster
    are summarized below with the corresponding values of i and j.
    | i,j |  N |
    |-----+----|
    | 1,0 |  1 |
    | 1,1 |  3 |
    | 2,0 |  4 |
    | 2,1 |  7 |
    | 3,1 | 13 |
    | 3,2 | 19 |

    """
    # Available colors for the clusters. These colos must be understook by
    # the plot library
    _colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def __init__(self):
        """
        """
        self._cell_radius = 0
        self._num_cells = 0

        # A list with the clusters in the grid
        self._clusters = []
        # Surrounding cells in the grid, which are not part of any cluster
        self._surrounding_cells = []

    def _get_num_clusters(self):
        return len(self._clusters)
    # Property to get the number of clusters in the grid
    num_clusters = property(_get_num_clusters)

    def clear(self):
        """Clear everything in the grid.
        """
        self._clusters = []
        self._surrounding_cells = []
        self._cell_radius = 0
        self._num_cells = 0

    def create_clusters(self, num_clusters, num_cells, cell_radius):
        """Create the clusters in the grid.

        Arguments:
        - `num_clusters`: Number of clusters to be created in the grid.
        - `num_cells`: Number of cells per clusters.
        - `cell_radius`: The radius of each cell.
        """
        self.clear()

        if not num_cells in frozenset([2, 3, 7]):
            raise AttributeError("The Grid class does not implement the case of clusters with {0} cells".format(num_cells))

        self._cell_radius = cell_radius
        self._num_cells = num_cells

        options = {2: self._calc_cluster_pos2,
                   3: self._calc_cluster_pos3,
                   7: self._calc_cluster_pos7}

        # Method to calculate the central position of the next cluster
        calc_pos = options[num_cells]

        for index in range(num_clusters):
            central_pos = calc_pos()
            # cell_radius, num_cells, pos=0 + 0j, cluster_id=None
            new_cluster = Cluster(cell_radius,
                              num_cells,
                              central_pos,
                              self.num_clusters + 1)
            new_cluster.fill_face_bool = True
            new_cluster.fill_color = Grid._colors[self.num_clusters]
            new_cluster.fill_opacity = 0.3
            self._clusters.append(new_cluster)

    def _calc_cluster_pos2(self):
        """Calculates the central position of clusters with 2 cells.

        Note that the returned central position will depend on how many
        clusters were already added to the grid.

        """
        cluster_index = self.num_clusters + 1
        if cluster_index == 1:
            return 0 + 0j
        elif cluster_index == 2:
            angle = np.pi / 3.0
            length = np.sqrt(3) * self._cell_radius
            return length * np.exp(1j * angle)
        else:
            RuntimeError("For the two cells per cluster case only two clusters may be used")

    def _calc_cluster_pos3(self):
        """Calculates the central position of clusters with 3 cells.

        Note that the returned central position will depend on how many
        clusters were already added to the grid.

        """
        cluster_index = self.num_clusters + 1
        if cluster_index == 1:
            return 0 + 0j
        else:
            angle = (np.pi / 3.) * (cluster_index - 1) - (np.pi / 6.)
            length = 3 * self._cell_radius
            return length * np.exp(1j * angle)

    def _calc_cluster_pos7(self):
        """Calculates the central position of clusters with 7 cells.

        Note that the returned central position will depend on how many
        clusters were already added to the grid.

        """
        cluster_index = self.num_clusters + 1
        if cluster_index == 1:
            return 0 + 0j
        else:
            angle = np.arctan(np.sqrt(3.) / 5.)
            angle = angle + (np.pi / 3) * (cluster_index - 2)
            length = np.sqrt(21) * self._cell_radius
            return length * np.exp(1j * angle)

    def plot(self, ax=None):
        """Plot the grid of clusters.

        Arguments:
        - `ax`:  A matplotlib axes
        """
        stand_alone_plot = False
        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = pylab.axes()
            stand_alone_plot = True

        for cluster in self._clusters:
            cluster.plot(ax)

        if stand_alone_plot is True:
            ax.plot()
            pylab.show()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx Main methods xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__1':
    c = Cell(3 + 2j, 1, cell_id=3)
    c.fill_face_bool = True

    n = Node(0.2 + 0.3j)
    #n.plot_node()

    c.add_user(n)
    c.add_border_user([0, 30, 60, 90, 120], 0.9, user_color='g')
    #c.add_border_user([90, 130], 0.7,'b')
    #c.add_random_users(100, 'b', 0.5)

    c.plot()

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    ax = pylab.axes()
    cell_radius = 1
    num_cells = 19
    node_pos = 3 + 15j
    cluster_id = None
    C = Cluster(cell_radius, num_cells, node_pos, cluster_id)

    C.add_random_users([3, 10, 13], [5, 10, 20], ['y', 'k', 'g'], [0.7, 0.3, 0.5])
    #C.add_border_users([1, 5, 7], [90, 60], [1, 0.8], ['g', 'k'])

    # vertexes = C.vertices
    # ax.plot(vertexes.real, vertexes.imag, 'o')

    C.fill_face_bool = True
    C.plot(ax)
    C.plot_border(ax)

    # # Circle ilustrating the cluster radius
    # print C.radius
    # circ = Circle(C.node_pos, C.radius)
    # print circ.radius
    # circ.plot(ax)

    # # Circle ilustrating the cluster external radius
    # circ2 = Circle(C.node_pos, C.external_radius)
    # circ2.plot(ax)

    ax.plot()
    plt.axis('equal')
    plt.show()


if __name__ == '__main__1':
    from matplotlib import pyplot as plt
    ax = pylab.axes()
    cell_radius = 1
    num_cells = 7
    num_clusters = 7

    # pos1 = 2 + 8j
    # C1 = Cluster(cell_radius, num_cells, pos1, cluster_id=1)

    # pos2 = 0 + 0j
    # C2 = Cluster(cell_radius, num_cells, pos2, cluster_id=2)

    grid = Grid()
    # grid._clusters = [C1, C2]

    grid.create_clusters(num_clusters, num_cells, cell_radius)

    grid.plot(ax)

    ax.plot()
    plt.axis('equal')
    plt.show()
