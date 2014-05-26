#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module that implements Cell related classes."""

__revision__ = "$Revision$"

try:
    from matplotlib import patches
    from matplotlib import pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

from collections import Iterable
import numpy as np
import itertools
from io import BytesIO
import cmath

from . import shapes

__all__ = ['Node', 'CellBase', 'Cell', 'Cell3Sec', 'Cluster', 'Grid']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Node class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Node(shapes.Coordinate):
    """Class representing a node in the network.
    """
    def __init__(self, pos, plot_marker='*', marker_color='r'):
        """Initializes the Node object.

        Parameters
        ----------
        pos : complex
            The position of the node in the complex grid.
        """
        shapes.Coordinate.__init__(self, pos)
        self.plot_marker = plot_marker
        self.marker_color = marker_color

    def plot_node(self, ax=None):  # pragma: no cover
        """Plot the node using the matplotlib library.

        If an axes 'ax' is specified, then the node is added to that
        axes. Otherwise a new figure and axes are created and the node is
        plotted to that.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the node will be plotted. If not provided, a new
            figure (and axis) will be created.

        """
        stand_alone_plot = False
        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        ax.plot(self.pos.real,
                self.pos.imag,
                marker=self.plot_marker,
                markerfacecolor=self.marker_color,
                markeredgecolor=self.marker_color)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Cell classes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class CellBase(Node, shapes.Shape):  # pylint: disable=W0223
    """Base class for all cell types.
    """

    def __init__(self, pos, radius, cell_id=None, rotation=0):
        """Initializes the CellBase object.

        Parameters
        ----------
        pos : complex
            The central position of the cell in the complex grid.
        radius : float
            The cell radius.
        cell_id : int, optional
            The cell ID. If not provided the cell won't have an ID and its
            plot will shown a symbol in cell center instead of the cell ID.
        rotation : float, optional (default to 0)
            The rotation of the cell (regarding the cell center).
        """
        Node.__init__(self, pos)
        shapes.Shape.__init__(self, pos, radius, rotation)
        self.plot_marker = '^'
        self.marker_color = 'b'

        self.id = cell_id
        self._users = []

        # xxxxx Appearance for plotting xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Set this to a number. If None, default value for Matplotlib will be used.
        self.cell_id_fontsize = None
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def __repr__(self):
        """
        Representation of a CellBase (or derived) object.
        """
        return "{0}(pos={1},radius={2},cell_id={3},rotation={4})".format(
            self.__class__.__name__, self.pos, self.radius, self.id, self.rotation)

    # The _set_pos property inherited from the Coordinate class only
    # changes the self._pos variable. We re-implement it here so that when
    # the position is changed we update the positions of any user already
    # in the cell.
    def _set_pos(self, value):
        """Set method for the pos property."""
        diff = value - self._pos
        self._pos = value
        for user in self._users:
            user.pos += diff
    pos = property(fget=shapes.Coordinate._get_pos, fset=_set_pos)

    def _get_num_users(self):
        """Get method for the num_users property."""
        return len(self._users)
    num_users = property(_get_num_users)

    def _get_users(self):
        """Get method for the users property."""
        return self._users
    users = property(_get_users)

    def delete_all_users(self):
        """Delete all users from the cell.
        """
        self._users = []

    def add_user(self, new_user, relative_pos_bool=True):
        """Adds a new user to the cell.

        Parameters
        ----------
        new_user : An object of the Node class
            The new user to be added to the cell.
        relative_pos_bool : bool, optional (default to True)
            Indicates if the 'pos' attribute of the `new_user` is relative
            to the center of the cell or not.

        Raises
        ------
        ValueError
            If the user position is outside the cell (the user won't be added).

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

        Parameters
        ----------
        angles : float or an iterable of floats
            Angle(s) for which users will be added (may be a single number or
            an iterable).
        ratio : float or an iterable of floats, optional
            The ration (relative distance from cell center) for which users
            will be added (may be a single number or an iterable). If not
            specified the users will be added to the cell's border at the
            angles specified in `angles`.
        user_color : srt or an iterable of srt
            Color of the user's marker.

        Raises
        ------
        ValueError
            If the ratio is invalid (negative or greater than 1).

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

        Parameters
        ----------
        user_color : srt
            Color of the user's marker.
        min_dist_ratio : float
            Minimum allowed (relative) distance between the cell center and
            the generated random user. The value must be between 0 and 0.7.

        """
        # Creates a new user. Note that this user can be invalid (outside
        # the cell) or not.
        new_user = Node(self.pos + complex(2 * (np.random.rand() - 0.5) * self.radius, 2 * (np.random.rand() - 0.5) * self.radius))

        while (not self.is_point_inside_shape(new_user.pos) or (self.calc_dist(new_user) < (min_dist_ratio * self.radius))):
            # Create another, since the previous one is not valid
            new_user = Node(self.pos + complex(2 * (np.random.rand() - 0.5) * self.radius, 2 * (np.random.rand() - 0.5) * self.radius))

        if user_color is not None:
            new_user.marker_color = user_color

        # Finally add the user to the cell
        self.add_user(new_user, relative_pos_bool=False)

    def add_random_users(self, num_users, user_color=None, min_dist_ratio=0):
        """
        Add `num_users` users randomly located in the cell.

        Parameters
        ----------
        num_users : int
            Number of users to be added to the cell.
        user_color : srt
            Color of the user's marker.
        min_dist_ratio : float
            Minimum allowed (relative) distance between the cell center and
            the generated random user. The value must be between 0 and 0.7.
        """
        for _ in range(num_users):
            self.add_random_user(user_color, min_dist_ratio)

    def _plot_common_part(self, ax):  # pragma: no cover
        """Common code for plotting the classes. Each subclass must implement a
        `plot` method in which it calls the command to plot the class shape
        followed by _plot_common_part.

        Parameters
        ----------
        ax : A matplotlib axis
            The axis where the cell will be plotted.
        """
        # If self.id is None, plot a single marker at the center of the
        # cell
        if self.id is None:
            self.plot_node(ax)
        else:
            # If self.id is not None, plot the cell ID at the center of the
            # cell
            plt.text(self.pos.real,
                     self.pos.imag,
                     '{0}'.format(self.id),
                     color=self.marker_color,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=self.cell_id_fontsize)

        # Now we plot all the users in the cell
        for user in self.users:
            user.plot_node(ax)

    def plot_border(self, ax=None):  # pragma: no cover
        """
        Plot the border of the cell.

        If an axes 'ax' is specified, then the shape is added to that
        axis. Otherwise a new figure and axis are created and the shape is
        plotted to that.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the cell will be plotted. If not provided, a new
            figure (and axis) will be created.
        """
        stand_alone_plot = False

        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        # Plot the border of the cell
        shapes.Shape.plot(self, ax)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()
        else:
            ax.autoscale_view(False, True, True)

    @staticmethod
    def _validate_ratio(ratio):
        """Return `ratio` if is valid, 1.0 if `ratio` is None, or throw an
        exception if it is not valid.

        This is a herper method used in the add_border_user method
        implementation.

        Parameters
        ----------
        ratio : float
            The ratio (a number between 0 and 1).

        Returns
        -------
        ratio : float
            The valid ratio. If ratio parameter was 'None' then 1.0 will be
            returned.

        Raises
        ------
        ValueError
            If `ratio` is not between 0 and 1.

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


class Cell(shapes.Hexagon, CellBase):
    """Class representing an hexagon cell.
    """

    def __init__(self, pos, radius, cell_id=None, rotation=0):
        """Initializes the Cell object.

        Parameters
        ----------
        pos : complex
            The central position of the cell in the complex grid.
        radius : float
            The cell radius.
        cell_id : int, optional
            The cell ID. If not provided the cell won't have an ID and its
            plot will shown a symbol in cell center instead of the cell ID.
        rotation : float, optional (default to 0)
            The rotation of the cell (regarding the cell center).
        """
        shapes.Hexagon.__init__(self, pos, radius, rotation)
        CellBase.__init__(self, pos, radius, cell_id, rotation)

    def plot(self, ax=None):  # pragma: no cover
        """
        Plot the cell using the matplotlib library.

        If an axes 'ax' is specified, then the shape is added to that
        axis. Otherwise a new figure and axis are created and the shape is
        plotted to that.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the cell will be plotted. If not provided, a new
            figure (and axis) will be created.
        """

        stand_alone_plot = False

        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        # Plot the shape part
        shapes.Hexagon.plot(self, ax)
        # Plot the node part as well as the users in the cell
        self._plot_common_part(ax)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()
        else:
            ax.autoscale_view(False, True, True)


class Cell3Sec(CellBase):
    """
    Class representing a cell with 3 sectors.

    Each sector corresponds to an hexagon.
    """
    def __init__(self, pos, radius, cell_id=None, rotation=0):
        """
        Initializes the Cell3Sec object.

        Parameters
        ----------
        pos : complex
            The central position of the cell in the complex grid.
        radius : float
            The cell radius. The sector radius will be equal to half the
            cell radius.
        cell_id : int, optional
            The cell ID. If not provided the cell won't have an ID and its
            plot will shown a symbol in cell center instead of the cell ID.
        rotation : float, optional (default to 0)
            The rotation of the cell (regarding the cell center).
        """
        CellBase.__init__(self, pos, radius, cell_id, rotation)

        # secradius = self.secradius
        # h = secradius * (np.sqrt(3) / 2.0)

        sec_positions = self._calc_sectors_positions()

        self._sec1 = Cell(sec_positions[0], self.secradius, cell_id='',
                          rotation=self.rotation-30)
        self._sec2 = Cell(sec_positions[1], self.secradius, cell_id='',
                          rotation=self.rotation-30)
        self._sec3 = Cell(sec_positions[2], self.secradius, cell_id='',
                          rotation=self.rotation-30)

    def _calc_sectors_positions(self):
        """
        Calculates the positions of the sectors with the current rotation,
        center position and radius.
        """
        secradius = self.secradius
        h = secradius * (np.sqrt(3) / 2.0)

        sec_positions = np.empty(3, dtype=complex)
        sec_positions[0] = 0 - h - (0.5j * secradius)
        sec_positions[1] = 0 + h - (0.5j * secradius)
        sec_positions[2] = 0 + (1j * secradius)

        sec_positions = shapes.Shape._rotate(sec_positions, self.rotation)
        sec_positions += self.pos
        return sec_positions

    def _set_radius(self, value):
        """Set method for the radius property"""
        # Overwrite the set property for radius in the Shape parent class
        # so that if radius is changed we update the radius of each sector.
        self._radius = value
        secradius = self.secradius  # self.secradius is updated when self.radius changes
        self._sec1.radius = secradius
        self._sec2.radius = secradius
        self._sec3.radius = secradius

        # When the radius change, we also need to update the position of each sector.
        sec_positions = self._calc_sectors_positions()
        self._sec1.pos = sec_positions[0]
        self._sec2.pos = sec_positions[1]
        self._sec3.pos = sec_positions[2]
    radius = property(fget=shapes.Shape._get_radius, fset=_set_radius)

    def _set_rotation(self, value):
        """Set method for the rotation property."""
        # Overwrite the set property for rotation in the Shape parent class
        # so that if rotation is changed we update the rotation of each
        # sector.
        self._rotation = value

        self._sec1.rotation = value - 30
        self._sec2.rotation = value - 30
        self._sec3.rotation = value - 30
        sec_positions = self._calc_sectors_positions()

        self._sec1.pos = sec_positions[0]
        self._sec2.pos = sec_positions[1]
        self._sec3.pos = sec_positions[2]
    rotation = property(fget=shapes.Shape._get_rotation, fset=_set_rotation)

    def _set_pos(self, value):
        """Set method for the pos property."""

        # Calling the _set_pos method of CellBase class will not only
        # update the position of the cell, but also update the position of
        # any users already in the cell.
        CellBase._set_pos(self, value)

        # Update the sectors' positions
        sec_positions = self._calc_sectors_positions()
        self._sec1.pos = sec_positions[0]
        self._sec2.pos = sec_positions[1]
        self._sec3.pos = sec_positions[2]
    pos = property(fget=shapes.Shape._get_pos, fset=_set_pos)

    def _get_secradius(self):
        """
        Get method for the secradius property.

        The radius of a sector.
        """
        # The value "sqrt(3) * r / 3" was chosen to be the section radius
        # so that the area of the three sectorized cell with radius equal
        # to `secradius` be the same of an hexagonal cell with radius equal
        # to `r`.
        return np.sqrt(3) * self.radius / 3.0
    secradius = property(_get_secradius)

    def _get_vertex_positions(self):
        """
        """
        secradius = self.secradius
        h = secradius * (np.sqrt(3) / 2.0)

        # The three sectors are hexagons. We set their positions to the origin.
        sec1 = shapes.Hexagon(0 - h - (0.5j * secradius), secradius, rotation=30)
        sec2 = shapes.Hexagon(0 + h - (0.5j * secradius), secradius, rotation=30)
        sec3 = shapes.Hexagon(0 + (1j * secradius), secradius, rotation=30)

        # The vertexes of the whole cell correspond to the union of the
        # vertexes of all sectors.
        aux = [sec1.vertices[[0, 1]],
               sec2.vertices[[0, 1, 2, 3]],
               sec3.vertices[[2, 3, 4, 5]],
               sec1.vertices[[4, 5]]]
        all_vertexes = np.hstack(aux)

        return all_vertexes

    def add_random_user_in_sector(self, sector, user_color=None, min_dist_ratio=0):
        """
        Adds a user randomly located in the specified `sector` of the cell.

        Parameters
        ----------
        sector : int
            The sector index. Can only be 1, 2 or 3.
        user_color : srt
            Color of the user's marker.
        min_dist_ratio : float
            Minimum allowed (relative) distance between the cell center and
            the generated random user. The value must be between 0 and 0.7.
        """
        if sector == 1:
            sec = self._sec1
        elif sector == 2:
            sec = self._sec2
        elif sector == 3:
            sec = self._sec3
        else:
            raise RuntimeError('Invalid sector number: {0}'.format(sector))

        sec.add_random_user(user_color, min_dist_ratio)
        self._users.extend(sec.users)
        sec.delete_all_users()

    def add_random_users_in_sector(self, num_users, sector, user_color=None, min_dist_ratio=0):
        """
        Add `num_users` users randomly in the specified `sector` of the cell

        Parameters
        ----------
        num_users : int
            Number of users to be added to the sector.
        sector : int
            The sector index. Can only be 1, 2 or 3.
        user_color : srt
            Color of the user's marker.
        min_dist_ratio : float
            Minimum allowed (relative) distance between the cell center and
            the generated random user. The value must be between 0 and 0.7.
        """
        for _ in range(num_users):
            self.add_random_user_in_sector(sector, user_color, min_dist_ratio)

    def plot(self, ax=None):
        """
        Plot the cell using the matplotlib library.

        If an axes 'ax' is specified, then the shape is added to that
        axis. Otherwise a new figure and axis are created and the shape is
        plotted to that.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the cell will be plotted. If not provided, a new
            figure (and axis) will be created.
        """
        stand_alone_plot = False

        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        # Plot the shape part
        shapes.Shape.plot(self, ax)

        # xxxxxxxxxx Plot the dashed lines (border between sectors xxxxxxxx
        rotation = self.rotation * np.pi / 180
        angle = (np.pi / 6.) + rotation
        p1 = self.pos + (np.cos(angle) + (np.sin(angle) * 1j)) * self.secradius
        angle = angle + 2 * np.pi / 3.
        p2 = self.pos + (np.cos(angle) + (np.sin(angle) * 1j)) * self.secradius
        angle = angle + 2 * np.pi / 3.
        #p3 = self.pos - 1j * self.secradius
        p3 = self.pos + (np.cos(angle) + (np.sin(angle) * 1j)) * self.secradius

        line1 = plt.Line2D([self.pos.real, p1.real], [self.pos.imag, p1.imag],
                           linestyle='dashed', color='black', alpha=0.5)
        line2 = plt.Line2D([self.pos.real, p2.real], [self.pos.imag, p2.imag],
                           linestyle='dashed', color='black', alpha=0.5)
        line3 = plt.Line2D([self.pos.real, p3.real], [self.pos.imag, p3.imag],
                           linestyle='dashed', color='black', alpha=0.5)

        ax.add_line(line1)
        ax.add_line(line2)
        ax.add_line(line3)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Plot the node part as well as the users in the cell
        self._plot_common_part(ax)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()
        else:
            ax.autoscale_view(False, True, True)


class CellWrap(CellBase):
    """Class that wraps another cell.
    """
    def __init__(self, pos, wrapped_cell):
        """
        Initializes the CellWrap object.

        Parameters
        ----------
        pos : complex
            The central position where the wrapped cell will be in the
            complex grid.
        wrapped_cell : A subclass of CellBase.
            The wrapped cell.
        """
        # Except for the _wrapped_cell member variable below, all other
        # member variables are defined in some base class of CellWrap.
        self._wrapped_cell = wrapped_cell

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        radius = wrapped_cell.radius
        rotation = wrapped_cell.rotation
        if wrapped_cell.id is not None:
            cell_id = "Wrap {0}".format(wrapped_cell.id)
        else:
            cell_id = None
        CellBase.__init__(self, pos, radius, cell_id, rotation)

        self.fill_face_bool = True
        self.fill_color = 'gray'
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx pos property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    pos = property(shapes.Shape._get_pos)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx radius property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def _get_radius(self):
        """Get method for the radius property."""
        return self._wrapped_cell.radius
    radius = property(_get_radius)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx rotation property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def _get_rotation(self):
        """Get method for the rotation property."""
        return self._wrapped_cell.rotation
    rotation = property(_get_rotation)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx users property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def _get_users(self):
        """Get method for the users property."""
        wrapped_cell_pos = self._wrapped_cell.pos
        users = [Node(u.pos - wrapped_cell_pos + self.pos) for u in self._wrapped_cell.users]
        return users
    users = property(_get_users)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def __repr__(self):
        """
        Representation of a CellWrap (or derived) object.
        """
        return "{0}(pos={1},cell_id={3})".format(
            self.__class__.__name__, self.pos, self._wrapped_cell)

    def _get_vertex_positions(self):
        return self._wrapped_cell._get_vertex_positions()

    def plot(self, ax=None):  # pragma: no cover
        stand_alone_plot = False

        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        # Plot the shape part
        shapes.Shape.plot(self, ax)
        # Plot the node part as well as the users in the cell
        self._plot_common_part(ax)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()
        else:
            ax.autoscale_view(False, True, True)



# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Cluster Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Cluster(shapes.Shape):
    """Class representing a cluster of Hexagonal cells.

    Valid cluster sizes are given by the formula
    :math:`N = i^2+i*j+j^2`
    where i and j are interger numbers. The allowed values in the Cluster
    class are sumarized below with the corresponding values of i and j.

    .. aafig::
        +-----+----+
        | i,j |  N |
        +-----+----+
        | 1,0 | 01 |
        | 1,1 | 03 |
        | 2,0 | 04 |
        | 2,1 | 07 |
        | 3,1 | 13 |
        | 3,2 | 19 |
        +-----+----+

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
    radius = property(shapes.Shape._get_radius)

    def __init__(self,
                 cell_radius,
                 num_cells,
                 pos=0 + 0j,
                 cluster_id=None,
                 cell_type='simple',
                 rotation=None):
        """Initializes the Cluster object.

        Parameters
        ----------
        cell_radius : float
            Radius of the cells in the cluster.
        num_cells : int
            Number of cells in the cluster.
        pos : complex
            Central Position of the Cluster in the complex grid.
        cluster_id : int
            ID of the cluster.
        cell_type : string (can be either 'simple' or '3sec'
            The type of the cell. If it is 'simple' it means the standard
            hexagon shaped cell. If '3sec' it means a 3 sectorized cell
            composed of 3 hexagons.
        rotation : float
            Rotation of the cluster.
        """
        shapes.Shape.__init__(self, pos, radius=0, rotation=0)

        # xxxxx Store for later reference (in __repr__) xxxxxxxxxxxxxxxxxxx
        self._cell_type = cell_type
        self._rotation = rotation
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        self.cluster_id = cluster_id
        self._cell_radius = cell_radius

        # Cells in the cluster
        self._cells = []

        cell_positions = Cluster._calc_cell_positions(cell_radius,
                                                      num_cells,
                                                      cell_type,
                                                      rotation)
        # Correct the positions to take into account the grid central
        # position.
        cell_positions[:, 0] = cell_positions[:, 0] + self.pos

        if cell_type == 'simple':
            CELLCLASS = Cell
        elif cell_type == '3sec':
            CELLCLASS = Cell3Sec
        else:
            raise RuntimeError('Invalid cell type: {0}'.format(cell_type))

        # Finally, create the cells at the specified positions (also
        # rotated)
        for index in range(num_cells):
            cell_id = index + 1
            self._cells.append(CELLCLASS(cell_positions[index, 0],
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
        self._external_radius = self._calc_cluster_external_radius()

        # xxxxx Plot appearance xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self._cell_id_fontsize = None  # If None, default value will be used
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx pos property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We reimplement the pos property here so that we can disable setting
    # the position
    pos = property(shapes.Coordinate._get_pos)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def __repr__(self):
        """
        Representation of a Cluster (or derived) object.
        """
        return "{0}(cell_radius={1},num_cells={2},pos={3},cluster_id={4},cell_type={5},rotation={6})".format(
            self.__class__.__name__, self._cell_radius, self.num_cells, self.pos,
            self.cluster_id, repr(self._cell_type), self._rotation)

    def _set_cell_id_fontsize(self, value):
        """
        Set method for the cell_id_fontsize property.

        The value of cell_id_fontsize only matters for plotting the
        cluster.

        Parameters
        ----------
        value : None or an int
            The font size used to plot the cell id. If it is None, the
            default value in matplotlib will be used.
        """
        self._cell_id_fontsize = value
        for c in self._cells:
            c.cell_id_fontsize = value

    def _get_cell_id_fontsize(self):
        """
        Get method for the cell_id_fontsize property.

        The value of cell_id_fontsize only matters for plotting the
        cluster.
        """
        return self._cell_id_fontsize
    cell_id_fontsize = property(_get_cell_id_fontsize, _set_cell_id_fontsize)

    # Property to get the cluster external radius
    # The cluster class also has a external_radius parameter that
    # corresponds to the radius of the smallest circle that can completely
    # hold the cluster inside of it.
    def _get_external_radius(self):
        """Get method for the external_radius property."""
        return self._external_radius
    external_radius = property(_get_external_radius)

    # Property to get the number of users in the cluster.
    def _get_num_users(self):
        """Get method for the num_users property."""
        num_users = [cell.num_users for cell in self._cells]
        return np.sum(num_users)
    num_users = property(_get_num_users)

    # property to get the number of _cells in the cluster
    def _get_num_cells(self):
        """Get method for the num_cells property."""
        return len(self._cells)

    num_cells = property(_get_num_cells)

    def _get_cell_radius(self):
        """Get method for the cell_radius property."""
        return self._cell_radius
    # Property to get the radius of the cells in the cluster.
    cell_radius = property(_get_cell_radius)

    @staticmethod
    def _calc_cell_height(radius):
        """
        Calculates the cell height from the cell radius.

        Parameters
        ----------
        radius : float
            The cell Radius.

        Returns
        -------
        height : float
            The cell height.
        """
        return radius * np.sqrt(3.0) / 2.0

    def _get_cell_height(self):
        """Get method for the cell_height property."""
        return self._calc_cell_height(self.cell_radius)
    cell_height = property(_get_cell_height)

    def __iter__(self):
        """Iterator for the cells in the cluster"""
        return iter(self._cells)

    def get_all_users(self):
        """Return all users in the cluster.

        Returns
        -------
        all_users : list
            A list with all users in the cluster.
        """
        all_users = []
        for cell in self._cells:
            all_users.extend(cell.users)
        return all_users

    @staticmethod
    def _get_ii_and_jj(num_cells):
        """Valid cluster sizes are given by the formula
                :math:`N = i^2+i*j+j^2`

        where i and j are integer numbers and "N" is the number of cells in
        the cluster. This static function returns the values "i" and "j"
        for a given "N". The values are summarized below.

        .. aafig::
            +-----+----+
            | i,j |  N |
            +-----+----+
            | 1,0 | 01 |
            | 1,1 | 03 |
            | 2,0 | 04 |
            | 2,1 | 07 |
            | 3,1 | 13 |
            | 3,2 | 19 |
            +-----+----+

        Parameters
        ----------
        num_cells : int
            Number of cells in the cluster.

        Returns
        -------
        ii and jj : tuple of ints
            The ii and jj values corresponding to number of cells
            'num_cells'.

        Notes
        -----
        If `num_cells` is not in the table above then (0, 0) will be
        returned.

        """
        return Cluster._ii_and_jj.get(num_cells, (0, 0))

    @staticmethod
    def _calc_cell_positions(cell_radius, num_cells, cell_type="simple", rotation=None):
        """
        Helper function used by the Cluster class.

        The calc_cell_positions method calculates the position (and
        rotation) of the 'num_cells' different cells, each with radius
        equal to 'cell_radius', so that they properly fit in the cluster.

        Parameters
        ----------
        cell_radius : float
            Radius of each cell in the cluster.
        num_cells : int
            Number of cells in the cluster.
        cell_type : string (can be either 'simple' or '3sec'
            The type of the cell. If it is 'simple' it means the standard
            hexagon shaped cell. If '3sec' it means a 3 sectorized cell
            composed of 3 hexagons.
        rotation : float
            Rotation of the cluster.

        Returns
        -------
        cell_positions : 2D numpy array
            The first column of `cell_positions` has the positions of the
            cells in a cluster with `num_cells` cells with radius
            `cell_radius`. The second column has the rotation of each cell.
        """
        if cell_type == 'simple':
            cell_positions = Cluster._calc_cell_positions_hexagon(
                cell_radius, num_cells, rotation)
        elif cell_type == '3sec':
            cell_positions = Cluster._calc_cell_positions_3sec(cell_radius,
                                                               num_cells,
                                                               rotation)
        else:
            raise RuntimeError('Invalid cell type: {0}'.format(cell_type))

        # xxxxxxx Possibly translate the postions of each cell xxxxxxxxxxxx
        # The coordinates of the cells calculated up to now consider the
        # center of the first cell as the origin. However, we want the
        # center of the cluster to be the origin. Therefore, lets calculate
        # the central position of the cluster and then correct all
        # coordinates to move the center of the cluster to the origin.
        central_pos = np.sum(cell_positions, axis=0) / num_cells
        # We correct only the first column, which is the position
        cell_positions[:, 0] = cell_positions[:, 0] - central_pos[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return cell_positions

    @staticmethod
    def _calc_cell_positions_3sec(cell_radius, num_cells, rotation=None):
        """
        Helper function used by the Cluster class.

        The _calc_cell_positions_3sec method calculates the position (and
        rotation) of the 'num_cells' different cells, each with radius
        equal to 'cell_radius', so that they properly fit in the cluster.

        Parameters
        ----------
        cell_radius : float
            Radius of each cell in the cluster.
        num_cells : int
            Number of cells in the cluster.
        rotation : float
            Rotation of the cluster.

        Returns
        -------
        cell_positions : 2D numpy array
            The first column of `cell_positions` has the positions of the
            cells in a cluster with `num_cells` cells with radius
            `cell_radius`. The second column has the rotation of each cell.
        """
        cell_positions = np.zeros([num_cells, 2], dtype=complex)
        sec_radius = np.sqrt(3) * cell_radius / 3.
        sec_height = sec_radius * np.sqrt(3.) / 2.

        # xxxxx Get the positions of cells from 2 to 7 xxxxxxxxxxxxxxxxxxxx
        # angles_first_ring -> 30:60:330 -> 30,90,150,210,270,330
        angles_first_ring = np.linspace(np.pi / 6., 11. * np.pi / 6., 6)
        max_value = min(num_cells, 7)
        for index in range(1, max_value):
            angle = angles_first_ring[index - 1]
            cell_positions[index, 0] = cmath.rect(3 * sec_radius, angle)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Get the positions of cells from 8 to 13 xxxxxxxxxxxxxxxxxxxx
        if num_cells > 7:
            # angles_second_ring_A -> 0:60:300 -> 0,60,120,180,240,300
            angles_second_ring_A = np.linspace(0, np.pi * 5. / 3., 6)
            max_value = min(num_cells, 13)
            for index in range(7, max_value):
                angle = angles_second_ring_A[index - 7]
                cell_positions[index, 0] = cmath.rect(6 * sec_height, angle)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Get the positions of cells from 14 to 19 xxxxxxxxxxxxxxxxxxxx
        if num_cells > 13:
            # 30:60:330 -> 30,90,150,210,270,330
            angles_second_ring_B = angles_first_ring
            max_value = min(num_cells, 19)
            for index in range(13, max_value):
                angle = angles_second_ring_B[index - 13]
                cell_positions[index, 0] = cmath.rect(6 * sec_radius, angle)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        if rotation is not None:
            # The cell positions calculated up to now do not consider
            # rotation. Lets use the rotate function of the Shape class to
            # rotate the coordinates.
            cell_positions[:, 0] = shapes.Shape._rotate(cell_positions[:, 0], rotation)
            cell_positions[:, 1] = rotation

        return cell_positions

    @staticmethod
    def _calc_cell_positions_hexagon(cell_radius, num_cells,
                                     rotation=None):
        """
        Helper function used by the Cluster class.

        The calc_cell_positions method calculates the position (and
        rotation) of the 'num_cells' different cells, each with radius
        equal to 'cell_radius', so that they properly fit in the cluster.

        Parameters
        ----------
        cell_radius : float
            Radius of each cell in the cluster.
        num_cells : int
            Number of cells in the cluster.
        rotation : float
            Rotation of the cluster.

        Returns
        -------
            The first column of `cell_positions` has the positions of the
            cells in a cluster with `num_cells` cells with radius
            `cell_radius`. The second column has the rotation of each cell.
        """
        # The first column in cell_positions has the cell positions
        # (complex number) and the second column has the cell rotation
        # (only the real part is considered)
        cell_positions = np.zeros([num_cells, 2], dtype=complex)
        cell_height = Cluster._calc_cell_height(cell_radius)

        # xxxxx Get the positions of cells from 2 to 7 xxxxxxxxxxxxxxxxxxxx
        # angles_first_ring -> 30:60:330 -> 30,90,150,210,270,330
        angles_first_ring = np.linspace(np.pi / 6., 11. * np.pi / 6., 6)
        max_value = min(num_cells, 7)
        for index in range(1, max_value):
            angle = angles_first_ring[index - 1]
            cell_positions[index, 0] = cmath.rect(2 * cell_height, angle)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Get the positions of cells from 8 to 19 xxxxxxxxxxxxxxxxxxxx
        # angles -> 0, 30, 60, ..., 330
        angles = np.linspace(0, 11 * np.pi / 6., 12)
        # For angle 0, the distance is 3*cell_radius, for angle 30 the
        # distance is 4*cell_height, for angle 60 the distance is
        # 3*cell_radius, for angle 90 the distance is 4*cell_height and the
        # pattern continues.
        dists = itertools.cycle([3*cell_radius, 4*cell_height])

        # The distance alternates between 3*cell_radius and 4*cell_height.
        for index, a, d in zip(range(7, num_cells), angles, dists):
            cell_positions[index, 0] = cmath.rect(d, a)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        if rotation is not None:
            # The cell positions calculated up to now do not consider
            # rotation. Lets use the rotate function of the Shape class to
            # rotate the coordinates.
            cell_positions[:, 0] = shapes.Shape._rotate(cell_positions[:, 0], +rotation)
            cell_positions[:, 1] = rotation

        return cell_positions

    @staticmethod
    def _calc_cluster_radius(num_cells, cell_radius):
        """
        Calculates the "cluster radius" for a cluster with "num_cells" cells,
        each cell with radius equal to "cell_radius". The cluster "radius"
        is equivalent to half the distance between two clusters when they
        are in a Grid.

        Parameters
        ----------
        num_cells : int
            Number of cells in the cluster.
        cell_radius : float
            Radius of each cell in the cluster.

        Returns
        -------
        cluster_radius : float
            The radius of a cluster with `num_cells` cells with radius
            `cell_radius`.

        Notes
        -----
        The cluster "radius" is equivalent to half the distance between two
        clusters.
        """
        cell_height = Cluster._calc_cell_height(cell_radius)
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

    def _calc_cluster_external_radius(self):
        """Calculates the cluster external radius.

        The cluster external radius is equal to the radius of the smallest
        circle (located at the center of the cluster) that contains the
        cluster. This circle should touch only the most external vertexes
        of the cells in the cluster.

        Get the vertex positions of the last cell.

        Returns
        -------
        external_radius : float
            The cluster external radius.

        """
        vertex_positions = self._cells[-1].vertices
        dists = vertex_positions - self.pos
        external_radius = np.max(np.abs(dists))
        return external_radius

    def _get_outer_vertexes(self, vertexes, central_pos, distance):
        """Filter out vertexes closer to the shape center them `distance`.

        This is a helper method used in the _get_vertex_positions method.

        Parameters
        ----------
        vertexes : numpy array
            The outer vertexes of the cluster.
        central_pos : complex
            Central position of the shape.
        distance : float
            A minimum distance. Any vertex that is closer to the shape
            center then this distance will be removed.

        Returns
        -------
        outer_vertexes : 1D numpy array
            The cluster outer vertexes.
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

        Returns
        -------
        vertex_positions : 1D numpy array
            The vertex positions of the cluster borders.

        Notes
        -----
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

    def plot(self, ax=None):  # pragma: no cover
        """Plot the cluster.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the cluster will be plotted. If not provided, a new
            figure (and axis) will be created.
        """
        stand_alone_plot = False
        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
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
            plt.show()
        else:
            ax.autoscale_view(False, True, True)

    def plot_border(self, ax=None):  # pragma: no cover
        """
        Plot only the border of the Cluster.

        Only work's for cluster sizes that can calculate the cluster
        vertices, such as cluster with 1, 7 or 19 cells.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the cluster will be plotted. If not provided, a new
            figure (and axis) will be created.

        """
        if len(self.vertices) != 0:
            stand_alone_plot = False
            if (ax is None):
                # This is a stand alone plot. Lets create a new axes.
                ax = plt.axes()
                stand_alone_plot = True

            polygon_edges = patches.Polygon(
                shapes.from_complex_array_to_real_matrix(self.vertices),
                True,
                facecolor='none',  # No face
                alpha=1,
                linewidth=2)
            ax.add_patch(polygon_edges)

            if stand_alone_plot is True:
                ax.plot()
                plt.show()
            else:
                ax.autoscale_view(False, True, True)

    def add_random_users(self, cell_ids, num_users=1, user_color=None, min_dist_ratio=0):
        """Adds one or more users to the Cells with the specified cell IDs (the
        first cell has an ID equal to 1.).

        Parameters
        ----------
        cell_ids : int or iterable
            IDs of the cells in the Cluster for which users will be
            added. The first cell has an ID equal to 1 and `cell_ids` may
            be an iterable with the IDs of several cells.
        num_users : int
            Number of users to be added to each cell.
        user_color : str
            Color of the user's marker.
        min_dist_ratio : float
            Minimum allowed (relative) distance between the cell center and
            the generated random user. See Cell.add_random_user method for
            details.

        Notes
        -----
        If `cell_ids` is an iterable then the other attributes (num_users,
        user_color and min_dist_ratio) may also be iterable with the same
        length of cell_ids in order to specifying individual values for
        each cell ID.

        """
        if isinstance(cell_ids, Iterable):
            if not isinstance(num_users, Iterable):
                num_users = itertools.repeat(num_users)
            if isinstance(user_color, str) or user_color is None:
                user_color = itertools.repeat(user_color)
            ## If you ever have problems with user_color, try uncommenting
            ## the code below
            # else:
            #     if not isinstance(user_color, Iterable):
            #         user_color = itertools.repeat(user_color)
            if not isinstance(min_dist_ratio, Iterable):
                min_dist_ratio = itertools.repeat(min_dist_ratio)

            all_data = zip(cell_ids, num_users, user_color, min_dist_ratio)
            for data in all_data:
                self.add_random_users(*data)
        else:
            for _ in range(num_users):
                # Note that here cell_ids will be a single value, as well
                # as user_color and min_dist_ratio
                self._cells[cell_ids - 1].add_random_user(user_color, min_dist_ratio)

    def add_border_users(self, cell_ids, angles, ratios=None, user_color=None):
        """Add users to all the cells indicated by `cell_indexes` at the
        specified angle(s) (in degrees) and ratio (relative distance from
        the center to the border of the cell.

        Parameters
        ----------
        cell_ids : int or Iterable
            IDs of the cells in the Cluster for which users will be
            added. The first cell has an ID equal to 1 and `cell_ids` may
            be an iterable with the IDs of several cells.
        angles : float or iterable
            Angles (in degrees)
        ratios : float or iterable
            Ratios (from 0 to 1)
        user_color : str or iterable of str
            Color of the user's marker.

        Examples
        --------
        >>> cluster = Cluster(cell_radius=1.0, num_cells=3)
        >>> # Add a single user in the angle of 30 degrees with a ration of 0.9
        >>> # to the first cell in the cluster
        >>> cluster.add_border_users(1, 30, 0.9)
        >>>
        >>> # Add 3 users at the angles of 0, 95 and 185 degrees to the second
        >>> # cell of the cluster
        >>> cluster.add_border_users(2, [0, 95, 185], 0.9, 'b')
        >>>
        >>> # Add one user in each cell at the angle of 10 degrees
        >>> cluster.add_border_users([1, 2, 3], 10, 0.9, 'g')
        >>>
        >>> # Add a user in each cell at different angles per cell
        >>> cluster.add_border_users([1, 2, 3], [90, 150, 190], 0.9, 'y')
        >>>
        >>> # Add multiple users to multiple cells at different angles
        >>> cluster.add_border_users([1, 2, 3], [[180, 270], [-30], [60, 120]], 0.9, 'k')

        """
        # If cell_ids is not an iterable, that is, cell_ids is a single
        # number, then we are simply calling the add_border_users method of
        # the specified cell
        if (not isinstance(cell_ids, Iterable)):
            self._cells[cell_ids - 1].add_border_user(
                angles, ratios, user_color)
        else:
            # If angles is not an iterable, then lets repeat the same value
            # for all specified cells by using itertools.repeat to make
            # angles an iterable.
            if not isinstance(angles, Iterable):
                angles = itertools.repeat(angles)

            # If ratios is not an iterable, then lets repeat the same value
            # for all specified cells by using itertools.repeat to make
            # ratios an iterable.
            if not isinstance(ratios, Iterable):
                ratios = itertools.repeat(ratios)

            # If user_color is not an iterable of strings, then lets repeat
            # the same value for all specified cells by using
            # itertools.repeat to make user_color an iterable of strings.
            if isinstance(user_color, str):
                user_color = itertools.repeat(user_color)
            else:
                if not isinstance(user_color, Iterable):
                    user_color = itertools.repeat(user_color)

            all_data = zip(cell_ids, angles, ratios, user_color)

            for data in all_data:
                cell_id, angle, ratio, color = data
                self._cells[cell_id - 1].add_border_user(angle, ratio, color)

    def delete_all_users(self, cell_id=None):
        """Remove all users from one or more cells.

        If cell_id is an integer > 0, only the users from the cell whose
        index is `cell_id` will be removed. If cell_id is an iterable, then
        the users of cells pointed by it will be removed. If cell_id is
        `None` or not specified, then the users of all cells will be
        removed.

        Parameters
        ----------
        cell_id : int or iterable, optional
            ID(s) of the cells from which users will be removed. If equal
            to None, all the users from all cells will be removed.

        """
        if isinstance(cell_id, Iterable):
            for i in cell_id:
                self._cells[i - 1].delete_all_users()
        elif cell_id is None:
            for cell in self._cells:
                cell.delete_all_users()
        else:
            self._cells[cell_id - 1].delete_all_users()

    def calc_dist_all_cells_to_all_users(self):
        """Returns a matrix with the distance from each cell center to each
        user in each cell.

        This matrix is suitable to later calculate the path loss from each
        base station to each mobile station.

        Because usually the base station is the transmitter and the mobile
        station is the receiver the matrix is such that each column
        corresponds to a different base station and each row corresponds to
        a different mobile station.

        Returns
        -------
        all_dists : 2D numpy array
            Distance from each cell center to each user.

        Notes
        -----
        There is no explicit indication from which cell each user
        is. However, in a case, for instance, where there are 3 cells in
        the cluster with 2, 2 and 3 users in each of them, respectively,
        then the first 2 rows correspond to the users in the first cell,
        the following 2 rows correspond to the users in the second cell and
        the last three rows correspond to the users in the third cell.

        """
        all_users = self.get_all_users()

        # We use the ndmin=2 option so that the array has two dimensions
        # instead of just one
        all_users_pos = np.array([x.pos for x in all_users], ndmin=2)
        all_cells_pos = np.array([x.pos for x in self._cells], ndmin=2)

        # Using broadcast we can calculate all distances in one go without
        # any for loop. -> dists[user_index, cell_index]
        dists = np.abs(all_users_pos.T - all_cells_pos)
        return dists

    def create_wrap_around_cells(self):
        """
        This function will create the wrapped cells, as well as the wrapinfo
        data.
        """
        if self.num_cells == 19:
            pass
        else:
            raise RuntimeError("Wrap around not implemented for a cluster with {0} cells.".format(self.num_cells))


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
    :math:`N = i^2+i*j+j^2`
    where i and j are integer numbers. The values allowed in the Cluster
    are summarized below with the corresponding values of i and j.

    .. aafig::
       +-----+----+
       | i,j |  N |
       +-----+----+
       | 1,0 | 01 |
       | 1,1 | 03 |
       | 2,0 | 04 |
       | 2,1 | 07 |
       | 3,1 | 13 |
       | 3,2 | 19 |
       +-----+----+

    """
    # Available colors for the clusters. These colos must be understook by
    # the plot library
    _colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def __init__(self):
        """Initializes the grid object.
        """
        self._cell_radius = 0
        self._num_cells = 0

        # A list with the clusters in the grid
        self._clusters = []
        # Surrounding cells in the grid, which are not part of any cluster
        self._surrounding_cells = []

    def get_cluster_from_index(self, index):
        """
        Return the cluster object with index `index` in the Grid.

        Parameters
        ----------
        index : int
            The index of the desirable cluster.
        """
        return self._clusters[index]

    def _get_num_clusters(self):
        """Get method for the num_clusters property."""
        return len(self._clusters)
    # Property to get the number of clusters in the grid
    num_clusters = property(_get_num_clusters)

    def __iter__(self):
        """Iterator for the clusters in the Grid
        """
        return iter(self._clusters)

    def clear(self):
        """Clear everything in the grid.
        """
        self._clusters = []
        self._surrounding_cells = []
        self._cell_radius = 0
        self._num_cells = 0

    def create_clusters(self, num_clusters, num_cells, cell_radius):
        """Create the clusters in the grid.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to be created in the grid.
        num_cells : int
            Number of cells per clusters.
        cell_radius : float
            The radius of each cell.
        """
        self.clear()

        if not num_cells in frozenset([2, 3, 7]):
            raise ValueError("The Grid class does not implement the case of clusters with {0} cells".format(num_cells))

        self._cell_radius = cell_radius
        self._num_cells = num_cells

        options = {2: self._calc_cluster_pos2,
                   3: self._calc_cluster_pos3,
                   7: self._calc_cluster_pos7}

        # Method to calculate the central position of the next cluster
        calc_pos = options[num_cells]

        for _ in range(num_clusters):
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

        Returns
        -------
        central_pos : complex
            Central position of the next cluster to be added to the Grid.

        Notes
        -----
        The returned central position will depend on how many clusters were
        already added to the grid.

        """
        cluster_index = self.num_clusters + 1
        if cluster_index == 1:
            return 0 + 0j
        elif cluster_index == 2:
            angle = np.pi / 3.0
            length = np.sqrt(3) * self._cell_radius
            return length * np.exp(1j * angle)
        else:
            raise ValueError("For the two cells per cluster case only two clusters may be used")

    def _calc_cluster_pos3(self):
        """Calculates the central position of clusters with 3 cells.

        Returns
        -------
        central_pos : complex
            Central position of the next cluster to be added to the Grid.

        Notes
        -----
        The returned central position will depend on how many clusters were
        already added to the grid.

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

        Returns
        -------
        central_pos : complex
            Central position of the next cluster to be added to the Grid.

        Notes
        -----
        The returned central position will depend on how many clusters were
        already added to the grid.

        """
        cluster_index = self.num_clusters + 1
        if cluster_index == 1:
            return 0 + 0j
        else:
            angle = np.arctan(np.sqrt(3.) / 5.)
            angle = angle + (np.pi / 3) * (cluster_index - 2)
            length = np.sqrt(21) * self._cell_radius
            return length * np.exp(1j * angle)

    def plot(self, ax=None):  # pragma: no cover
        """Plot the grid of clusters.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the grid will be plotted. If not provided, a new
            figure (and axis) will be created.
        """
        stand_alone_plot = False
        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        for cluster in self._clusters:
            cluster.plot(ax)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()

    # This method is the same in the Shape class
    def _repr_some_format_(self, extension='png', axis_option='equal'):
        """
        Return the representation of the shape in the desired format.

        Parameters
        ----------
        extension : str
            The extension of the desired format. This should be something
            that the savefig method in a matplotlib figure can understandm,
            such as 'png', 'svg', stc.
        axis_option : str
            Option to be given to the ax.axis function.
        """
        plt.ioff() # turn off interactive mode
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        output = BytesIO()
        self.plot(ax)
        ax.axis(axis_option)
        fig.savefig(output, format=extension)
        output.seek(0)
        plt.close(fig)
        plt.ion() # turn on interactive mode

        return output.getvalue()

    # This method is the same in the Shape class
    def _repr_png_(self):
        """
        Return the PNG representation of the shape.
        """
        return self._repr_some_format_('png')

    # This method is the same in the Shape class
    def _repr_svg_(self):
        """
        Return the SVG representation of the shape.
        """
        return self._repr_some_format_('svg')

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
