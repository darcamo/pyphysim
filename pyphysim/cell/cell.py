#!/usr/bin/env python
"""Module that implements Cell and Cluster related classes."""

try:
    # noinspection PyUnresolvedReferences
    from matplotlib import patches
    # noinspection PyUnresolvedReferences
    from matplotlib import pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

import cmath
import itertools
import math
from collections.abc import Iterable
from io import BytesIO
from typing import Any, Dict
from typing import Iterable as Iterable_t  # distinguish it from collections.abc.Iterable
from typing import Iterator, List, Optional, Tuple, Type, Union, cast

import numpy as np

from ..cell import shapes

__all__ = [
    'Node', 'AccessPoint', 'CellBase', 'Cell', 'Cell3Sec', 'CellSquare',
    'CellWrap', 'Cluster', 'Grid'
]

IntOrIntIterable = Union[Iterable_t[int], int]
FloatOrFloatIterable = Union[Iterable_t[float], float]
StrOrStrIterable = Union[Iterable_t[str], str]


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Node class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Node(shapes.Coordinate):
    """
    Class representing a node in the network.

    Parameters
    ----------
    pos : complex
        The position of the node in the complex grid.
    plot_marker : str
        The marker to be used in a plot to represent the Node. This
        marker should be something that matplotlib can understand, such
        as '*', for instance.
    marker_color : str
        The color that will be used to plot the marker representing the
        Node. This color should be something that matplotlib can
        understand, such as 'r' for the color red, for instance.
    cell_id : str, int, optional
        The ID of the cell where the Node is located.
    parent_pos : complex
        The position of the cell where the Node is located (if any).
    """
    def __init__(self,
                 pos: complex,
                 plot_marker: str = '*',
                 marker_color: str = 'r',
                 cell_id: Optional[Union[str, int]] = None,
                 parent_pos: Optional[complex] = None) -> None:
        super().__init__(pos)
        self.plot_marker: str = plot_marker
        self.marker_color: str = marker_color
        self.marker_size: int = 6  # Changing this value will affect only the plot

        # ID of the cell where the user is located
        self.cell_id: Optional[Union[str, int]] = cell_id

        self._relative_pos: Optional[complex] = None
        if parent_pos is not None:
            self._relative_pos = pos - parent_pos

    @property
    def relative_pos(self) -> Optional[complex]:
        """
        Get method for the relative_pos property.

        Returns
        -------
        complex | None
            The relative position of the Node regarding its parent
            Node's position.
        """
        return self._relative_pos

    def plot_node(self, ax: Optional[Any] = None) -> None:  # pragma: no cover
        """
        Plot the node using the matplotlib library.

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
        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        ax.plot(self.pos.real,
                self.pos.imag,
                marker=self.plot_marker,
                markerfacecolor=self.marker_color,
                markeredgecolor=self.marker_color,
                markersize=self.marker_size)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx AccessPoint class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class AccessPoint(Node):
    """
    Access point class.

    An access point acts as a transmitter to one of more users.

    Parameters
    ----------
    pos : complex
        The central position of the cell in the complex grid.
    ap_id : int, str, optional
        The AccessPoint ID. If not provided the access point won't have
        an ID and its plot will shown a symbol at the access point
        location instead of the ID.
    """
    def __init__(self,
                 pos: complex,
                 ap_id: Optional[Union[int, str]] = None) -> None:
        super().__init__(pos, plot_marker='^', marker_color='b', cell_id=ap_id)

        # List to store the users associated with this access point
        self._users: List[Node] = []

        # ID of the access point
        self.id: Optional[Union[int, str]] = ap_id

        # xxxxx Appearance for plotting xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Set this to a number. If None, default value for Matplotlib will
        # be used.
        self.id_fontsize: Optional[int] = None
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def __repr__(self) -> str:  # pragma: nocover
        """
        Representation of a AccessPoint object.

        Returns
        -------
        str
            The string representation of the AccessPoint.
        """
        return "{0}(pos={1},ap_id={2})".format(self.__class__.__name__,
                                               self.pos, self.id)

    # The set pos property inherited from the Coordinate class only changes
    # the self._pos variable. We re-implement it here so that when the
    # position is changed we update the positions of any associated
    # user. Notice how we are only changing the 'set' part of the property
    # defined in the Coordinate base class.
    @property
    def pos(self) -> complex:
        """
        Get the AccessPoint position.

        Returns
        -------
        complex
            The AccessPoint position.
        """
        return self._pos

    @pos.setter
    def pos(self, value: complex) -> None:
        """
        Set the AccessPoint position.

        Parameters
        ----------
        value : complex
            The new AccessPoint position.
        """
        diff = value - self._pos
        self._pos = value
        for user in self._users:
            user.pos += diff

    @property
    def num_users(self) -> int:
        """
        Get method for the num_users property.

        Returns
        -------
        int
            The number of users associated with the AccessPoint.
        """
        return len(self._users)

    @property
    def users(self) -> List[Node]:
        """
        Get method for the users property.

        Returns
        -------
        list[Node]
            The users associated with the AccessPoint.
        """
        return self._users

    def delete_all_users(self) -> None:
        """Delete all users from the cell.
        """
        self._users = []

    def add_user(self, new_user: Node, relative_pos_bool: bool = True) -> None:
        """
        Associate a new user with the access point.

        Parameters
        ----------
        new_user : Node
            The new user to be associated with the access point.
        relative_pos_bool : bool
            Indicates it the position of the `new_user` is relative.
        """
        new_user.cell_id = self.id
        self._users.append(new_user)

    # noinspection PyUnresolvedReferences
    def _plot_common_part(self, ax: Any) -> None:  # pragma: no cover
        """
        Common code for plotting the classes. Each subclass must
        implement a `plot` method in which it calls the command to plot
        the class shape followed by _plot_common_part.

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
            # noinspection PyUnresolvedReferences
            plt.text(self.pos.real,
                     self.pos.imag,
                     '{0}'.format(self.id),
                     color=self.marker_color,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=self.id_fontsize)

        # Now we plot all the users in the cell
        for user in self.users:
            user.plot_node(ax)

    def plot(self, ax: Optional[Any] = None) -> None:  # pragma: no cover
        """
        Plot the AccessPoint using the matplotlib library.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the cell will be plotted. If not provided, a new
            figure (and axis) will be created.
        """
        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            _, ax = plt.subplots()

        # Plot the node part as well as the users in the cell
        self._plot_common_part(ax)
        # ax.set_ylim([-1, 1])
        # ax.set_xlim([-1, 1])
        plt.draw()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Cell classes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TODO: maybe refactor this class so that we don't inherit from any
# shape and use composition instead
# noinspection PyAbstractClass
class CellBase(shapes.Shape, AccessPoint):  # pylint: disable=W0223
    """
    Base class for all cell types.

    A cell is an AccessPoint with a predefined shape, where the users
    associated with it are inside the shape.

    Parameters
    ----------
    pos : complex
        The central position of the cell in the complex grid.
    radius : float
        The cell radius.
    cell_id : str, int, optional
        The cell ID. If not provided the cell won't have an ID and its
        plot will shown a symbol in cell center instead of the cell ID.
    rotation : float, optional
        The rotation of the cell (regarding the cell center).
    """
    def __init__(self,
                 pos: complex,
                 radius: float,
                 cell_id: Optional[Union[str, int]] = None,
                 rotation: float = 0.0,
                 **kw) -> None:
        super().__init__(pos=pos,
                         radius=radius,
                         rotation=rotation,
                         ap_id=cell_id,
                         **kw)

    def __repr__(self) -> str:
        """
        Representation of a CellBase object.

        Returns
        -------
        str
            The string representation of the CellBase.
        """
        return "{0}(pos={1},radius={2},cell_id={3},rotation={4})".format(
            self.__class__.__name__, self.pos, self.radius, self.id,
            self.rotation)

    def add_user(self, new_user: Node, relative_pos_bool: bool = True) -> None:
        """
        Adds a new user to the cell.

        Parameters
        ----------
        new_user : Node
            The new user to be added to the cell.
        relative_pos_bool : bool, optional (default to True)
            Indicates if the 'pos' attribute of the `new_user` is relative
            to the center of the cell or not.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the user position is outside the cell (the user won't be
            added).
        """
        if isinstance(new_user, Node):
            if relative_pos_bool is True:
                # If the position of the user is relative to the cell, that
                # means that the real and imaginary parts of new_user.pos
                # are in the [-1, 1] range. We need to convert them to an
                # absolute coordinate.
                new_user.pos = new_user.pos * self.radius + self.pos
            if self.is_point_inside_shape(new_user.pos):
                # pylint: disable=W0212
                new_user.cell_id = self.id
                new_user._relative_pos = new_user.pos - self.pos
                self._users.append(new_user)
            else:
                raise ValueError("User position is outside the cell -> "
                                 "User not added")
        else:
            raise TypeError("User must be Node object.")

    def add_border_user(
            self,
            angles: Union[float, Iterable_t[float]],
            ratio: Optional[Union[float, Iterable_t[float]]] = None,
            user_color: Optional[Union[str, Iterable_t[str]]] = None) -> None:
        """
        Adds a user at the border of the cell, located at a specified
        angle (in degrees).

        If the `angles` variable is an iterable, one user will be added
        for each value in `angles`. Also, in that case ration and
        user_color may also be iterable with the same length of `angles`
        in order to specify individual ratio and user_color for each
        angle.

        Parameters
        ----------
        angles : float | list[float] | np.ndarray
            Angle(s) for which users will be added (may be a single
            number or an iterable).
        ratio : float | list[float] | np.ndarray | None
            The ration (relative distance from cell center) for which users
            will be added (may be a single number or an iterable). If not
            specified the users will be added to the cell's border at the
            angles specified in `angles`.
        user_color : str | list[str], optional
            Color of the user's marker.

        Raises
        ------
        ValueError
            If the ratio is invalid (negative or greater than 1).
        """
        # Assures that angle is an iterable
        if not isinstance(angles, Iterable):
            angles = [angles]

        if user_color is None:
            user_color = itertools.repeat(user_color)  # type: ignore
        elif isinstance(user_color, str):
            user_color = itertools.repeat(user_color)
        if isinstance(ratio, float):
            ratio = CellBase._validate_ratio(ratio)
            ratio = itertools.repeat(ratio)
        elif ratio is None:
            ratio = itertools.repeat(None)  # type: ignore
        else:
            assert (isinstance(ratio, Iterable))
            ratio = [CellBase._validate_ratio(i) for i in ratio]

        assert (isinstance(ratio, Iterable))
        assert (isinstance(user_color, Iterable))
        all_data = zip(angles, ratio, user_color)
        for data in all_data:
            a, r, c = data
            new_user = Node(self.get_border_point(a, r),
                            cell_id=self.id,
                            parent_pos=self.pos)
            if c is not None:
                new_user.marker_color = c
            self._users.append(new_user)

    def add_random_user(self,
                        user_color: Optional[str] = None,
                        min_dist_ratio: float = 0.0) -> None:
        """
        Adds a user randomly located in the cell.

        The variable `user_color` can be any color that the plot command
        and friends can understand. If not specified the default value
        of the node class will be used.

        Parameters
        ----------
        user_color : str, optional
            Color of the user's marker.
        min_dist_ratio : float
            Minimum allowed (relative) distance between the cell center and
            the generated random user. The value must be between 0 and 0.7.

        """
        # Creates a new user. Note that this user can be invalid (outside
        # the cell) or not.
        pos = (self.pos +
               complex(2 * (np.random.random_sample() - 0.5) * self.radius, 2 *
                       (np.random.random_sample() - 0.5) * self.radius))
        new_user = Node(pos, cell_id=self.id, parent_pos=self.pos)

        # noinspection PyPep8
        while not self.is_point_inside_shape(
                new_user.pos) or (self.calc_dist(new_user) <
                                  (min_dist_ratio * self.radius)):

            # Create another, since the previous one is not valid
            pos = (self.pos +
                   complex(2 *
                           (np.random.random_sample() - 0.5) * self.radius, 2 *
                           (np.random.random_sample() - 0.5) * self.radius))
            new_user = Node(pos, cell_id=self.id, parent_pos=self.pos)

        if user_color is not None:
            new_user.marker_color = user_color

        # Finally add the user to the cell
        self.add_user(new_user, relative_pos_bool=False)

    def add_random_users(self,
                         num_users: int,
                         user_color: Optional[str] = None,
                         min_dist_ratio: float = 0.0) -> None:
        """
        Add `num_users` users randomly located in the cell.

        Parameters
        ----------
        num_users : int
            Number of users to be added to the cell.
        user_color : str, optional
            Color of the user's marker.
        min_dist_ratio : float
            Minimum allowed (relative) distance between the cell center and
            the generated random user. The value must be between 0 and 0.7.
        """
        for _ in range(num_users):
            self.add_random_user(user_color, min_dist_ratio)

    def plot_border(self,
                    ax: Optional[Any] = None) -> None:  # pragma: no cover
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

        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            _, ax = plt.subplots(figsize=self.figsize)
            stand_alone_plot = True

        # Plot the border of the cell
        shapes.Shape.plot(self, ax)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()
        else:
            ax.autoscale_view(False, True, True)

    @staticmethod
    def _validate_ratio(ratio: float) -> float:
        """Return `ratio` if is valid, 1.0 if `ratio` is None, or throw an
        exception if it is not valid.

        This is a helper method used in the add_border_user method
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
        if ratio == 1.0:
            ratio = 1.0 - 1e-15
        else:
            if (ratio < 0) or (ratio > 1):
                raise ValueError("ratio must be between 0 and 1")
        return ratio


class Cell(shapes.Hexagon, CellBase):
    """Class representing an hexagon cell.

    Parameters
    ----------
    pos : complex
        The central position of the cell in the complex grid.
    radius : float
        The cell radius.
    cell_id : str, int, optional
        The cell ID. If not provided the cell won't have an ID and its
        plot will shown a symbol in cell center instead of the cell ID.
    rotation : float, optional
        The rotation of the cell (regarding the cell center).
    """

    # noinspection PyCallByClass
    def __init__(self,
                 pos: complex,
                 radius: float,
                 cell_id: Optional[Union[str, int]] = None,
                 rotation: float = 0.0) -> None:
        super().__init__(pos=pos,
                         radius=radius,
                         rotation=rotation,
                         cell_id=cell_id)

    def plot(self, ax: Optional[Any] = None) -> None:  # pragma: no cover
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

        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            _, ax = plt.subplots(figsize=self.figsize)
            stand_alone_plot = True

        # Plot the shape part
        # noinspection PyCallByClass
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

    Parameters
    ----------
    pos : complex
        The central position of the cell in the complex grid.
    radius : float
        The cell radius. The sector radius will be equal to half the
        cell radius.
    cell_id : str, int, optional
        The cell ID. If not provided the cell won't have an ID and its
        plot will shown a symbol in cell center instead of the cell ID.
    rotation : float, optional
        The rotation of the cell (regarding the cell center).
    """
    def __init__(self,
                 pos: complex,
                 radius: float,
                 cell_id: Optional[Union[str, int]] = None,
                 rotation: float = 0.0) -> None:
        super().__init__(pos, radius, cell_id, rotation)

        sec_positions = self._calc_sectors_positions()

        self._sec1 = Cell(sec_positions[0],
                          self.secradius,
                          cell_id=None,
                          rotation=self.rotation - 30)
        self._sec2 = Cell(sec_positions[1],
                          self.secradius,
                          cell_id=None,
                          rotation=self.rotation - 30)
        self._sec3 = Cell(sec_positions[2],
                          self.secradius,
                          cell_id=None,
                          rotation=self.rotation - 30)

    def _calc_sectors_positions(self) -> np.ndarray:
        """
        Calculates the positions of the sectors with the current
        rotation, center position and radius.

        Returns
        -------
        np.ndarray
            The positions of the 3 sectors.
        """
        secradius = self.secradius
        h = secradius * (math.sqrt(3) / 2.0)

        sec_positions: np.ndarray = np.empty(3, dtype=complex)
        sec_positions[0] = 0 - h - (0.5j * secradius)
        sec_positions[1] = 0 + h - (0.5j * secradius)
        sec_positions[2] = 0 + (1j * secradius)

        sec_positions = shapes.Shape.calc_rotated_pos(sec_positions,
                                                      self.rotation)
        sec_positions += self.pos
        return sec_positions

    @property
    def radius(self) -> float:
        """
        Get the radius of the Cell3Sec object.

        Returns
        -------
        float
            The radius of the Cell3Sec object.
        """
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        """
        Set the radius of the Cell3Sec object.

        Parameters
        ----------
        value : float
            The new radius of the Cell3Sec object.
        """
        # Overwrite the set property for radius in the Shape parent class
        # so that if radius is changed we update the radius of each sector.
        self._radius = value

        # self.secradius is updated when self.radius changes
        secradius = self.secradius
        self._sec1.radius = secradius
        self._sec2.radius = secradius
        self._sec3.radius = secradius

        # When the radius change, we also need to update the position of
        # each sector.
        sec_positions = self._calc_sectors_positions()
        self._sec1.pos = sec_positions[0]
        self._sec2.pos = sec_positions[1]
        self._sec3.pos = sec_positions[2]

    @property
    def rotation(self) -> float:
        """
        Get method for the rotation property.

        Returns
        -------
        float
            The shape rotation.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value: float) -> None:
        """
        Set method for the rotation property.

        Parameters
        ----------
        value : float
            The rotation angle (in degrees).
        """
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

    @property
    def pos(self) -> complex:
        """
        Get the Cell3Sec position.

        Returns
        -------
        complex
            The Cell3Sec position.
        """
        return self._pos

    @pos.setter
    def pos(self, value: complex) -> None:
        """
        Set the Cell3Sec position.

        Parameters
        ----------
        value : complex
            The new Cell3Sec position.
        """

        # Calling the "set method" of the "pos" property of the CellBase
        # class will not only update the position of the cell, but also
        # update the position of any users already in the cell.
        CellBase.pos.fset(self, value)  # type: ignore

        # Update the sectors' positions
        sec_positions = self._calc_sectors_positions()
        self._sec1.pos = sec_positions[0]
        self._sec2.pos = sec_positions[1]
        self._sec3.pos = sec_positions[2]

    @property
    def secradius(self) -> float:
        """
        Get method for the secradius property.

        The radius of a sector.

        Returns
        -------
        float
            The radius of one sector of the Cell3Sec object.
        """
        # The value "sqrt(3) * r / 3" was chosen to be the section radius
        # so that the area of the three sectorized cell with radius equal
        # to `secradius` be the same of an hexagonal cell with radius equal
        # to `r`.
        return math.sqrt(3) * self.radius / 3.0

    def _get_vertex_positions(self) -> np.ndarray:
        """
        Calculates the vertex positions ignoring any rotation and
        considering that the shape is at the origin (rotation and
        translation will be added automatically later).

        Returns
        -------
        vertex_positions : np.ndarray
            The positions of the vertexes of the shape.
        """
        secradius = self.secradius
        h = secradius * (math.sqrt(3) / 2.0)

        # The three sectors are hexagons. We set their positions to the
        # origin.
        sec1 = shapes.Hexagon(0 - h - (0.5j * secradius),
                              secradius,
                              rotation=30)
        sec2 = shapes.Hexagon(0 + h - (0.5j * secradius),
                              secradius,
                              rotation=30)
        sec3 = shapes.Hexagon(0 + (1j * secradius), secradius, rotation=30)

        # The vertexes of the whole cell correspond to the union of the
        # vertexes of all sectors.
        aux = [
            sec1.vertices[[0, 1]], sec2.vertices[[0, 1, 2, 3]],
            sec3.vertices[[2, 3, 4, 5]], sec1.vertices[[4, 5]]
        ]
        all_vertexes: np.ndarray = np.hstack(aux)

        return all_vertexes

    def add_random_user_in_sector(self,
                                  sector: int,
                                  user_color: Optional[str] = None,
                                  min_dist_ratio: float = 0.0) -> None:
        """
        Adds a user randomly located in the specified `sector` of the cell.

        Parameters
        ----------
        sector : int
            The sector index. Can only be 1, 2 or 3.
        user_color : str
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

    def add_random_users_in_sector(self,
                                   num_users: int,
                                   sector: int,
                                   user_color: Optional[str] = None,
                                   min_dist_ratio: float = 0.0) -> None:
        """
        Add `num_users` users randomly in the specified `sector` of the
        cell

        Parameters
        ----------
        num_users : int
            Number of users to be added to the sector.
        sector : int
            The sector index. Can only be 1, 2 or 3.
        user_color : str
            Color of the user's marker.
        min_dist_ratio : float
            Minimum allowed (relative) distance between the cell center and
            the generated random user. The value must be between 0 and 0.7.
        """
        for _ in range(num_users):
            self.add_random_user_in_sector(sector, user_color, min_dist_ratio)

    # noinspection PyUnresolvedReferences
    def plot(self, ax: Optional[Any] = None) -> None:  # pragma: no cover
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

        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            _, ax = plt.subplots(figsize=self.figsize)
            stand_alone_plot = True

        # Plot the shape part
        shapes.Shape.plot(self, ax)

        # xxxxxxxxxx Plot the dashed lines (border between sectors xxxxxxxx
        rotation = self.rotation * math.pi / 180
        angle = (math.pi / 6.) + rotation
        p1 = (self.pos + (math.cos(angle) +
                          (math.sin(angle) * 1j)) * self.secradius)
        angle += 2 * math.pi / 3.

        p2 = (self.pos + (math.cos(angle) +
                          (math.sin(angle) * 1j)) * self.secradius)
        angle += 2 * math.pi / 3.

        # p3 = self.pos - 1j * self.secradius
        p3 = (self.pos + (math.cos(angle) +
                          (math.sin(angle) * 1j)) * self.secradius)

        line1 = plt.Line2D([self.pos.real, p1.real], [self.pos.imag, p1.imag],
                           linestyle='dashed',
                           color='black',
                           alpha=0.5)
        line2 = plt.Line2D([self.pos.real, p2.real], [self.pos.imag, p2.imag],
                           linestyle='dashed',
                           color='black',
                           alpha=0.5)
        line3 = plt.Line2D([self.pos.real, p3.real], [self.pos.imag, p3.imag],
                           linestyle='dashed',
                           color='black',
                           alpha=0.5)

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


class CellSquare(shapes.Rectangle, CellBase):
    """
    Class representing a 'square' cell.

    Parameters
    ----------
    pos : complex
        The central position of the cell in the complex grid.
    side_length : float
        The cell side length.
    cell_id : str, int, optional
        The cell ID. If not provided the cell won't have an ID and its
        plot will shown a symbol in cell center instead of the cell ID.
    rotation : float, optional
        The rotation of the cell (regarding the cell center).
    """

    # noinspection PyCallByClass
    def __init__(self,
                 pos: complex,
                 side_length: float,
                 cell_id: Optional[Union[str, int]] = None,
                 rotation: float = 0.0) -> None:
        half_side = side_length / 2.

        first = pos - half_side - 1j * half_side
        second = pos + half_side + 1j * half_side
        radius = math.sqrt(2.0) * side_length / 2.

        # super().__init__(first=first, second=second, rotation=rotation, pos=pos, radius=radius, cell_id=cell_id)
        shapes.Rectangle.__init__(self, first, second, rotation)
        CellBase.__init__(self, pos, radius, cell_id, rotation)

    def plot(self, ax: Optional[Any] = None) -> None:  # pragma: no cover
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

        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            _, ax = plt.subplots(figsize=self.figsize)
            stand_alone_plot = True

        # Plot the shape part
        # noinspection PyCallByClass
        shapes.Rectangle.plot(self, ax)
        # Plot the node part as well as the users in the cell
        self._plot_common_part(ax)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()
        else:
            ax.autoscale_view(False, True, True)

    def add_user(self, new_user: Node, relative_pos_bool: bool = True) -> None:
        """
        Adds a new user to the cell.

        Parameters
        ----------
        new_user : Node
            The new user to be added to the cell.
        relative_pos_bool : bool, optional
            Indicates if the 'pos' attribute of the `new_user` is relative
            to the center of the cell or not.

        Raises
        ------
        ValueError
            If the user position is outside the cell (the user won't be
            added).
        """
        if isinstance(new_user, Node):
            if relative_pos_bool is True:
                # If the position of the user is relative to the cell, that
                # means that the real and imaginary parts of new_user.pos
                # are in the [-1, 1] range. We need to convert them to an
                # absolute coordinate.
                half_side = abs(self._lower_coord.real -
                                self._upper_coord.real) / 2

                new_user.pos = new_user.pos * half_side + self.pos
            if self.is_point_inside_shape(new_user.pos):
                # pylint: disable=W0212
                new_user.cell_id = self.id
                new_user._relative_pos = new_user.pos - self.pos
                self._users.append(new_user)
            else:
                raise ValueError("User position is outside the cell -> "
                                 "User not added")
        else:
            raise TypeError("User must be Node object.")


class CellWrap(CellBase):
    """
    Class that wraps another cell.

    Parameters
    ----------
    pos : complex
        The central position where the wrapped cell will be in the
        complex grid.
    wrapped_cell : T <= CellBase
        The wrapped cell. It must be an object of some subclass of
        CellBase.
    include_users_bool : bool
        Set to True if the users of the original cells should appear in
        the wrapped version.
    """
    def __init__(self,
                 pos: complex,
                 wrapped_cell: CellBase,
                 include_users_bool: bool = False):
        assert isinstance(wrapped_cell, CellBase), \
            'wrapped_cell must be a subclass of CellBase'
        # Except for the _wrapped_cell member variable below, all other
        # member variables are defined in some base class of CellWrap.
        self._wrapped_cell = wrapped_cell

        # If True, users of the wrapped cells will be included as users of
        # the CellWrap object. Otherwise the CellWrap object will have 0
        # users.
        self.include_users_bool: bool = include_users_bool

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        radius = wrapped_cell.radius
        rotation = wrapped_cell.rotation
        cell_id: Optional[str]
        if wrapped_cell.id is not None:
            cell_id = "Wrap {0}".format(wrapped_cell.id)
        else:
            cell_id = None
        super().__init__(pos, radius, cell_id, rotation)

        self.fill_face_bool: bool = True
        self.fill_color: str = 'gray'
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    @property
    def radius(self) -> float:
        """
        Get the radius of the CellWrap object.

        Returns
        -------
        float
            The radius of the CellWrap object.
        """
        return self._wrapped_cell.radius

    @radius.setter
    def radius(self, value: float) -> None:
        """
        Set the radius of the CellWrap object.

        Parameters
        ----------
        value : float
            The new radius of the CellWrap object.
        """
        raise AttributeError("The radius of a CellWrap should not be changed")

    @property
    def rotation(self) -> float:
        """
        Get the rotation of the CellWrap object.

        Returns
        -------
        float
            The rotation of the CellWrap object.
        """
        return self._wrapped_cell.rotation

    @rotation.setter
    def rotation(self, value: float) -> None:
        """
        Set method for the rotation property.

        Parameters
        ----------
        value : float
            The new rotation value.
        """
        raise AttributeError(
            "The rotation of a CellWrap should not be changed")

    @property
    def num_users(self) -> int:
        """
        Get method for the num_users property.

        Returns
        -------
        int
            The number of users associated with the AccessPoint.
        """
        if self.include_users_bool is True:
            return self._wrapped_cell.num_users

        return 0

    @property
    def users(self) -> List[Node]:
        """
        Get method for the users property.

        Returns
        -------
        list
            The users associated with the AccessPoint.
        """
        users: List[Node]
        if self.include_users_bool is True:
            wrapped_cell_pos = self._wrapped_cell.pos
            users = [
                Node(u.pos - wrapped_cell_pos + self.pos,
                     marker_color='g',
                     parent_pos=self.pos) for u in self._wrapped_cell.users
            ]
        else:
            users = []
        return users

    def __repr__(self) -> str:
        """
        Representation of a CellWrap object.

        Returns
        -------
        str
            The string representation of the CellWrap.
        """
        return "{0}(pos={1},cell_id={2})".format(self.__class__.__name__,
                                                 self.pos, self.id)

    def _get_vertex_positions(self) -> np.ndarray:
        """
        Calculates the vertex positions ignoring any rotation and
        considering that the shape is at the origin (rotation and
        translation will be added automatically later).

        Returns
        -------
        vertex_positions : np.ndarray
            The positions of the vertexes of the shape.
        """
        return self._wrapped_cell.vertices_no_trans_no_rotation

    def plot(self, ax: Optional[Any] = None) -> None:  # pragma: no cover
        stand_alone_plot = False

        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            _, ax = plt.subplots(figsize=self.figsize)
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
    """
    Class representing a cluster of Hexagonal cells.

    Valid cluster sizes are given by the formula
    :math:`N = i^2+i*j+j^2`
    where i and j are integer numbers. The allowed values in the Cluster
    class are summarized below with the corresponding values of i and j.

    ====  ===
    i, j   N
    ====  ===
    1,0   01
    1,1   03
    2,0   04
    2,1   07
    3,1   13
    3,2   19
    ====  ===

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
    cell_type : str
        The type of the cell as a string. It can be either 'simple',
        '3sec' or 'square'.
        If it is 'simple' it means the standard hexagon shaped cell. If
        '3sec' it means a 3 sectorized cell composed of 3 hexagons.
    rotation : float
        Rotation of the cluster.
    """
    _ii_and_jj = {
        1: (1, 0),
        3: (1, 1),
        4: (2, 0),
        7: (2, 1),
        13: (3, 1),
        19: (3, 2)
    }

    # Store cell positions in a cluster centered at the origin without any
    # rotation and with a radius equal to one.
    _normalized_cell_positions: Dict[int, np.ndarray] = {}

    def __init__(self,
                 cell_radius: float,
                 num_cells: int,
                 pos: complex = 0 + 0j,
                 cluster_id: Optional[int] = None,
                 cell_type: str = 'simple',
                 rotation: float = 0.0) -> None:
        super().__init__(pos=pos, radius=0, rotation=0)

        # xxxxx Store for later reference (in __repr__) xxxxxxxxxxxxxxxxxxx
        self._cell_type: str = cell_type
        self._rotation: float = rotation
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        self.cluster_id: Optional[int] = cluster_id
        self._cell_radius: float = cell_radius

        # Cells in the cluster
        self._cells: List[CellBase] = []

        # Dictionary to store the wrapped cells (when wrap around is used)
        self._wrapped_cells: Dict[str, CellWrap] = {}

        # Each element is a list where the first element of that list if
        # the position of the corresponding cell. The subsequent elements
        # are the positions of that same cell wrapped somewhere.
        self._cell_pos: List[complex] = []

        # This will be set later as a 2D numpy array with the difference of
        # the coordinates between each pair of cells (possibly considering
        # wrap around)
        self._cell_pos_diffs: Optional[
            Iterable_t[complex]] = None  # np.ndarray of complex numbers

        cell_positions = Cluster._calc_cell_positions(cell_radius, num_cells,
                                                      cell_type, rotation)
        # Correct the positions to take into account the grid central
        # position.
        cell_positions[:, 0] = cell_positions[:, 0] + self.pos

        CELLCLASS: Type[CellBase]
        if cell_type == 'simple':
            CELLCLASS = Cell
        elif cell_type == '3sec':
            CELLCLASS = Cell3Sec
        elif cell_type == 'square':
            CELLCLASS = CellSquare
        else:  # pragma: no cover
            # Note that it the code should never get here, since if the
            # cell type is not valid an exception will be raised in the
            # '_calc_cell_positions' method which is called before this
            # point.
            raise RuntimeError("Invalid cell type: '{0}'".format(cell_type))

        # Finally, create the cells at the specified positions (also
        # rotated)
        for index in range(num_cells):
            cell_id = index + 1
            c = CELLCLASS(cell_positions[index, 0], cell_radius, cell_id,
                          cell_positions[index, 1])
            self._cells.append(c)
            self._cell_pos.append(c.pos)

        # Calculates the cluster radius.
        #
        # The radius of the cluster is defined as half the distance from
        #  one cluster to another. That is, if you plot multiple
        # clusters and one circle positioned in each cluster center with
        #  radius equal to the cluster radius, the circles should be
        # tangent to each other.
        self._radius: float = Cluster._calc_cluster_radius(
            num_cells, cell_radius)
        # Calculates the cluster external radius.
        self._external_radius: float = self._calc_cluster_external_radius()

        # xxxxx Plot appearance xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self._cell_id_fontsize: Optional[int] = None  # If None -> use default
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx radius property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We re-implement the pos setter property here so that we can disable
    # setting the radius of the cluster.
    @property
    def radius(self) -> float:  # pragma: no cover
        """
        Get the radius of the Cluster object.

        Returns
        -------
        float
            The radius of the Cluster object.
        """
        return self._radius

    @radius.setter
    def radius(self, _: Any) -> None:  # pylint: disable=R0201
        """
        Disabled setter for the radius property defined in base class.
        """
        raise AttributeError("can't set attribute")

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx pos property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We re-implement the pos setter property here so that we can disable
    # setting the position of the cluster.
    @property
    def pos(self) -> complex:
        """
        Get the Cluster position.

        Returns
        -------
        complex
            The Cluster position.
        """
        return self._pos

    @pos.setter
    def pos(self, _: Any) -> None:  # pylint: disable=R0201
        """
        Disabled setter for the pos property defined in base class.
        """
        raise AttributeError("can't set attribute")

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx rotation property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def rotation(self) -> float:
        """
        Get method for the rotation property.

        Returns
        -------
        float
            The shape rotation.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, _: Any) -> None:  # pylint: disable=R0201
        """
        Disabled setter for the rotation property defined in base class.
        """
        raise AttributeError("can't set attribute")

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def __repr__(self) -> str:
        """
        Representation of a Cluster object.

        Returns
        -------
        str
            The string representation of the Cluster.
        """
        msg = ("{0}(cell_radius={1},num_cells={2},pos={3},cluster_id={4},"
               "cell_type={5},rotation={6})")
        return msg.format(self.__class__.__name__, self._cell_radius,
                          self.num_cells, self.pos, self.cluster_id,
                          repr(self._cell_type), self._rotation)

    @property
    def cell_id_fontsize(self) -> Optional[int]:
        """
        Get method for the cell_id_fontsize property.

        The value of cell_id_fontsize only matters for plotting the
        cluster.

        Returns
        -------
        int | None
            The font size that should be used for the cell IDs in the
            plot.
        """
        return self._cell_id_fontsize

    @cell_id_fontsize.setter
    def cell_id_fontsize(self, value: Optional[int] = None) -> None:
        """
        Set method for the cell_id_fontsize property.

        The value of cell_id_fontsize only matters for plotting the
        cluster.

        Parameters
        ----------
        value : None | int
            The font size used to plot the cell id. If it is None, the
            default value in matplotlib will be used.
        """
        self._cell_id_fontsize = value
        for c in self._cells:
            c.id_fontsize = value

    # Property to get the cluster external radius
    # The cluster class also has a external_radius parameter that
    # corresponds to the radius of the smallest circle that can completely
    # hold the cluster inside of it.
    @property
    def external_radius(self) -> float:
        """
        Get the external_radius of the Cluster.

        Returns
        -------
        float
            The external_radius of the Cluster.
        """
        return self._external_radius

    @property
    def num_users(self) -> int:
        """
        Get method for the num_users property.

        Returns
        -------
        int
            The number of users in the Cluster.
        """
        num_users = [cell.num_users for cell in self._cells]
        return sum(num_users)

    @property
    def num_cells(self) -> int:
        """
        Get method for the num_cells property.

        Returns
        -------
        int
            Number of cells in the Cluster.
        """
        return len(self._cells)

    @property
    def cell_radius(self) -> float:
        """
        Get method for the cell_radius property.

        Returns
        -------
        float
            The radius of the cells in the Cluster.
        """
        return self._cell_radius

    @staticmethod
    def _calc_cell_height(radius: float) -> float:
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
        return radius * math.sqrt(3.0) / 2.0

    @property
    def cell_height(self) -> float:
        """
        Get method for the cell_height property.

        Returns
        -------
        float
            The height of the cells in the Cluster.
        """
        return self._calc_cell_height(self.cell_radius)

    def __iter__(self) -> Iterator[CellBase]:
        """Iterator for the cells in the cluster"""
        return iter(self._cells)

    def get_cell_by_id(self, cell_id: int) -> CellBase:
        """
        Get the cell in the Cluster with the given `cell_id`.

        Parameters
        ----------
        cell_id : int
            The ID of the desired cell.

        Returns
        -------
        c : Cell
            The desired cell.
        """
        return self._cells[cell_id - 1]

    def get_all_users(self) -> List["Node"]:
        """
        Return all users in the cluster.

        Returns
        -------
        all_users : list[Node]
            A list with all users in the cluster.
        """
        all_users = []
        for cell in self._cells:
            all_users.extend(cell.users)
        return all_users

    @staticmethod
    def _get_ii_and_jj(num_cells: int) -> Tuple[int, int]:
        """
        Valid cluster sizes are given by the formula

                :math:`N = i^2+i*j+j^2`

        where i and j are integer numbers and "N" is the number of cells in
        the cluster. This static function returns the values "i" and "j"
        for a given "N". The values are summarized below.

        ====  ===
        i, j   N
        ====  ===
        1,0   01
        1,1   03
        2,0   04
        2,1   07
        3,1   13
        3,2   19
        ====  ===

        Parameters
        ----------
        num_cells : int
            Number of cells in the cluster.

        Returns
        -------
        ii and jj : (int,int)
            The ii and jj values corresponding to number of cells
            'num_cells'.

        Notes
        -----
        If `num_cells` is not in the table above then (0, 0) will be
        returned.
        """
        return Cluster._ii_and_jj.get(num_cells, (0, 0))

    @staticmethod
    def _calc_cell_positions(cell_radius: float,
                             num_cells: int,
                             cell_type: str = "simple",
                             rotation: Optional[float] = None) -> np.ndarray:
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
        cell_type : str
            The type of the cell. It should be a string with one of the
            possible values: 'simple', '3sec', or 'square'.
            If it is 'simple' it means the standard hexagon shaped cell.
            If '3sec' it means a 3 sectorized cell composed of 3 hexagons.
        rotation : float | None, optional
            Rotation of the cluster.

        Returns
        -------
        cell_positions : np.ndarray
            The first column of `cell_positions` has the positions of the
            cells in a cluster with `num_cells` cells with radius
            `cell_radius`. The second column has the rotation of each cell.
        """
        if cell_type == 'simple':
            cell_positions = Cluster._calc_cell_positions_hexagon(
                cell_radius, num_cells, rotation)
        elif cell_type == '3sec':
            cell_positions = Cluster._calc_cell_positions_3sec(
                cell_radius, num_cells, rotation)
        elif cell_type == 'square':
            cell_positions = Cluster._calc_cell_positions_square(
                cell_radius, num_cells, rotation)
        else:
            raise RuntimeError("Invalid cell type: '{0}'".format(cell_type))

        # xxxxx Possibly translate the positions of each cell xxxxxxxxxxxxx
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
    def _calc_cell_positions_3sec(
            cell_radius: float,
            num_cells: int,
            rotation: Optional[float] = None) -> np.ndarray:
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
        rotation : float | None, optional
            Rotation of the cluster.

        Returns
        -------
        cell_positions : np.ndarray
            The first column of `cell_positions` has the positions of the
            cells in a cluster with `num_cells` cells with radius
            `cell_radius`. The second column has the rotation of each cell.
        """
        # In the end, the position of the cells of the Cell3Sec class in
        # the cluster are exactly the same positions they would get if the
        # were of the Cell (Hexagon shape) class.
        return Cluster._calc_cell_positions_hexagon(cell_radius, num_cells,
                                                    rotation)

    @staticmethod
    def _calc_cell_positions_hexagon(
            cell_radius: float,
            num_cells: int,
            rotation: Optional[float] = None) -> np.ndarray:
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
        rotation : float | None, optional
            Rotation of the cluster.

        Returns
        -------
        np.ndarray
            The first column of `cell_positions` has the positions of the
            cells in a cluster with `num_cells` cells with radius
            `cell_radius`. The second column has the rotation of each cell.
        """
        # Note that the Cluster._normalized_cell_positions dictionary store
        # the positions of the cells for a cluster with radius equal to
        # 1.0. Each key in the dictionary corresponds to a specific number
        # f cells.
        #
        # If Cluster._normalized_cell_positions has no key with the
        # value of 'num_cells' that means we still need to calculate it.
        #  Note, however, that this will be true only in the first time
        # that this method is called and any subsequent call of this
        # method for the same value of num_cells will avoid the
        # calculations in the if block below.
        if num_cells not in Cluster._normalized_cell_positions:
            norm_radius = 1.0
            # The first column in cell_positions has the cell positions
            # (complex number) and the second column has the cell rotation
            # (only the real part is considered)
            cell_positions: np.ndarray = np.zeros([num_cells, 2],
                                                  dtype=complex)
            cell_height = Cluster._calc_cell_height(norm_radius)

            # xxxxx Get the positions of cells from 2 to 7 xxxxxxxxxxxxxxxx
            # angles_first_ring -> 30:60:330 -> 30,90,150,210,270,330
            angles_first_ring = np.linspace(np.pi / 6., 11. * np.pi / 6., 6)
            max_value = min(num_cells, 7)
            for index in range(1, max_value):
                angle = angles_first_ring[index - 1]
                cell_positions[index, 0] = cmath.rect(2 * cell_height, angle)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Get the positions of cells from 8 to 19 xxxxxxxxxxxxxxx
            # angles -> 0, 30, 60, ..., 330
            angles = np.linspace(0, 11 * np.pi / 6., 12)
            # For angle 0, the distance is 3*norm_radius, for angle 30 the
            # distance is 4*cell_height, for angle 60 the distance is
            # 3*norm_radius, for angle 90 the distance is 4*cell_height and
            # the pattern continues.
            dists = itertools.cycle([3 * norm_radius, 4 * cell_height])

            # The distance alternates between 3*norm_radius and
            # 4*cell_height.
            for index, a, d in zip(range(7, num_cells), angles, dists):
                cell_positions[index, 0] = cmath.rect(d, a)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # Store the normalized cell positions for a cluster with
            # 'num_cells' cells for later reference.
            Cluster._normalized_cell_positions[num_cells] = cell_positions
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # At this point we know that Cluster._normalized_cell_positions[
        # num_cells] has the positions of the cells in a cluster with
        # radius equal to 1.0 and with 'num_cells' cells. All we need to
        #  do is multiply that our desired cell_radius and then apply
        # the 'rotation' (if there is any).
        cell_positions = (Cluster._normalized_cell_positions[num_cells] *
                          cell_radius)

        if rotation is not None:
            # The cell positions calculated up to now do not consider
            # rotation. Lets use the rotate function of the Shape class to
            # rotate the coordinates.
            cell_positions[:, 0] = shapes.Shape.calc_rotated_pos(
                cell_positions[:, 0], rotation)
            cell_positions[:, 1] = rotation

        return cell_positions

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _calc_cell_positions_square(
            side_length: float,
            num_cells: int,
            rotation: Optional[float] = None) -> np.ndarray:
        """
        Helper function used by the Cluster class.

        The calc_cell_positions method calculates the position (and
        rotation) of the 'num_cells' different cells, each with side
        equal to 'cell_radius', so that they properly fit in the cluster.

        Parameters
        ----------
        side_length : float
            The side length of each square cell in the cluster.
        num_cells : int
            Number of cells in the cluster.
        rotation : float | None, optional
            Rotation of the cluster.

        Returns
        -------
        cell_positions : np.ndarray
            The first column of `cell_positions` has the positions of the
            cells in a cluster with `num_cells` cells with radius
            `cell_radius`. The second column has the rotation of each cell.
        """
        cell_positions = np.zeros([num_cells, 2], dtype=complex)
        sqrt_num_cells = int(math.sqrt(num_cells))

        if sqrt_num_cells**2 != num_cells:
            raise ValueError("num_cells must be a perfect square number")

        int_positions = np.unravel_index(np.arange(num_cells),
                                         (sqrt_num_cells, sqrt_num_cells))

        cell_positions[:, 0] = (
            side_length *
            (int_positions[1] + 1j * int_positions[0][::-1] - 0.5 - 0.5j))

        if rotation is not None:
            # The cell positions calculated up to now do not consider
            # rotation. Lets use the rotate function of the Shape class to
            # rotate the coordinates.
            cell_positions[:, 0] = shapes.Shape.calc_rotated_pos(
                cell_positions[:, 0], rotation)
            cell_positions[:, 1] = rotation

        return cell_positions

    @staticmethod
    def _calc_cluster_radius(num_cells: int, cell_radius: float) -> float:
        """
        Calculates the "cluster radius" for a cluster with "num_cells"
        cells, each cell with radius equal to "cell_radius". The cluster
        "radius" is equivalent to half the distance between two clusters
        when they are in a Grid.

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
        # where N is the number of cells in a cluster and "i" and "j"
        # are two integer numbers. For each valid value of N we set the
        # "ii" and "jj" variables appropriately.
        (ii, jj) = Cluster._get_ii_and_jj(num_cells)

        # Considering one cluster with center located at the origin, we can
        # calculate the center of another cluster using the "ii" and "jj"
        # variables (see the Rappaport book).
        other_cluster_pos = (cell_height * ((jj * 0.5) +
                                            (1j * jj * math.sqrt(3.) / 2.)) +
                             cell_height * ii)

        # Now we can calculate the radius simple getting half the distance
        # from the origin to the center of the other cluster.
        radius = abs(other_cluster_pos)
        return radius

    def _calc_cluster_external_radius(self) -> float:
        """
        Calculates the cluster external radius.

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
        return cast(float, external_radius)

    def _get_outer_vertexes(self, vertexes: np.ndarray, central_pos: complex,
                            distance: float) -> np.ndarray:
        """
        Filter out vertexes closer to the shape center them `distance`.

        This is a helper method used in the _get_vertex_positions method.

        Parameters
        ----------
        vertexes : np.ndarray
            The outer vertexes of the cluster.
        central_pos : complex
            Central position of the shape.
        distance : float
            A minimum distance. Any vertex that is closer to the shape
            center then this distance will be removed.

        Returns
        -------
        outer_vertexes : np.ndarray
            The cluster outer vertexes.
        """
        def f(x: np.ndarray) -> np.ndarray:
            """
            Filter function. Returns True for vertexes which are closer
            to the shape center than `distance`.

            Parameters
            ----------
            x : np.ndarray
            """
            return np.abs(x - central_pos) > distance

        vertexes = vertexes[f(vertexes)]

        # Remove duplicates

        # Float equality test (used implicitly by 'set' to remove
        # duplicates) is not trustable. We lower the precision to make
        # it more trustable but maybe calculating the cluster vertexes
        # like this is not the best way.
        vertexes = frozenset(vertexes.round(12))
        vertexes = np.fromiter(vertexes, dtype=complex)

        # In order to use these vertices for plotting, we need them to be
        # in order (lowest angle to highest)
        vertexes = vertexes[np.argsort(np.angle(vertexes - self.pos))]
        return vertexes

    def _get_vertex_positions(self) -> np.ndarray:
        """
        Get the vertex positions of the cluster borders.

        Returns
        -------
        vertex_positions : np.ndarray
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

        all_vertexes = np.array(
            [cell.vertices for cell in self._cells[start_index:]]).flatten()
        return self._get_outer_vertexes(all_vertexes, self.pos, distance)

    # Note: The _get_vertex_positions method which should return the
    # shape vertexes without translation and rotation and the vertexes
    # property from the Shape class would add the translation and
    # rotation. However, the _get_vertex_positions method in the Cluster
    #  class return vertices's that already contains the translation and
    #  rotation. Therefore, we overwrite the property here to return the
    #  output of _get_vertex_positions.
    vertices = property(_get_vertex_positions)

    def plot(self, ax: Optional[Any] = None) -> None:  # pragma: no cover
        """
        Plot the cluster.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the cluster will be plotted. If not provided,
            a new figure (and axis) will be created.
        """
        stand_alone_plot = False
        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            _, ax = plt.subplots(figsize=self.figsize)
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

        for wrapped_cell in self._wrapped_cells.values():
            if self.fill_face_bool is True:
                wrapped_cell.fill_face_bool = True
                wrapped_cell.fill_opacity = self.fill_opacity

                # wrapped_cell.fill_color = self.fill_color
                # wrapped_cell.fill_opacity = self.fill_opacity
            else:
                wrapped_cell.fill_face_bool = False
            wrapped_cell.plot(ax)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()
        else:
            ax.autoscale_view(False, True, True)

    def plot_border(self,
                    ax: Optional[Any] = None) -> None:  # pragma: no cover
        """
        Plot only the border of the Cluster.

        Only work's for cluster sizes that can calculate the cluster
        vertices, such as cluster with 1, 7 or 19 cells.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the cluster will be plotted. If not provided,
            a new figure (and axis) will be created.

        """
        if len(self.vertices) != 0:
            stand_alone_plot = False
            if ax is None:
                # This is a stand alone plot. Lets create a new axes.
                _, ax = plt.subplots(figsize=self.figsize)
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

    def add_random_users(self,
                         cell_ids: Optional[IntOrIntIterable] = None,
                         num_users: IntOrIntIterable = 1,
                         user_color: Optional[StrOrStrIterable] = None,
                         min_dist_ratio: FloatOrFloatIterable = 0.0) -> None:
        """
        Adds one or more users to the Cells with the specified cell IDs
        (the first cell has an ID equal to 1.).

        Parameters
        ----------
        cell_ids : int | list[int] | np.ndarray
            IDs of the cells in the Cluster for which users will be
            added. The first cell has an ID equal to 1 and `cell_ids` may
            be an iterable with the IDs of several cells. If not provided,
            all cells will be assumed.
        num_users : int | list[int] | np.ndarray
            Number of users to be added to each cell.
        user_color : str | list[str], optional
            Color of the user's marker.
        min_dist_ratio : float, list[float], optional
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
        if cell_ids is None:
            cell_ids = range(1, self.num_cells + 1)

        if isinstance(cell_ids, Iterable):
            num_users_iterable = num_users if isinstance(
                num_users, Iterable) else itertools.repeat(num_users)

            user_color_iterable = itertools.repeat(user_color) if (isinstance(
                user_color, str) or user_color is None) else user_color

            min_dist_ratio_iterable = min_dist_ratio if isinstance(
                min_dist_ratio, Iterable) else itertools.repeat(min_dist_ratio)

            all_data = zip(cell_ids, num_users_iterable, user_color_iterable,
                           min_dist_ratio_iterable)
            for data in all_data:
                self.add_random_users(*data)
        else:
            assert (isinstance(num_users, int))
            assert (isinstance(min_dist_ratio, float))
            assert (user_color is None or isinstance(user_color, str))
            for _ in range(num_users):
                # Note that here cell_ids will be a single value, as well
                # as user_color and min_dist_ratio
                self.get_cell_by_id(cell_ids).add_random_user(
                    user_color, min_dist_ratio)

    def add_border_users(
            self,
            cell_ids: IntOrIntIterable,
            angles: FloatOrFloatIterable,
            ratios: FloatOrFloatIterable = 1.0,
            user_color: Optional[StrOrStrIterable] = None) -> None:
        """
        Add users to all the cells indicated by `cell_indexes` at the
        specified angle(s) (in degrees) and ratio (relative distance
        from the center to the border of the cell).

        Parameters
        ----------
        cell_ids : int | list[int] | np.ndarray
            IDs of the cells in the Cluster for which users will be
            added. The first cell has an ID equal to 1 and `cell_ids` may
            be an iterable with the IDs of several cells.
        angles : float | list[float] | np.ndarray
            Angles (in degrees)
        ratios : float | list[float]
            Ratios (from 0 to 1)
        user_color : str | list[str]
            Color of the user's marker.

        Examples
        --------
        >>> cluster = Cluster(cell_radius=1.0, num_cells=3)
        >>> # Add a single user in the angle of 30 degrees with a ration of
        >>> #  0.9 to the first cell in the cluster
        >>> cluster.add_border_users(1, 30, 0.9)
        >>>
        >>> # Add 3 users at the angles of 0, 95 and 185 degrees to the
        >>> # second cell of the cluster
        >>> cluster.add_border_users(2, [0, 95, 185], 0.9, 'b')
        >>>
        >>> # Add one user in each cell at the angle of 10 degrees
        >>> cluster.add_border_users([1, 2, 3], 10, 0.9, 'g')
        >>>
        >>> # Add a user in each cell at different angles per cell
        >>> cluster.add_border_users([1, 2, 3], [90, 150, 190], 0.9, 'y')
        >>>
        >>> # Add multiple users to multiple cells at different angles
        >>> cluster.add_border_users(\
                [1, 2, 3], [[180, 270], [-30], [60, 120]], 0.9, 'k')
        """
        # If cell_ids is not an iterable, that is, cell_ids is a single
        # number, then we are simply calling the add_border_users method of
        # the specified cell
        if not isinstance(cell_ids, Iterable):
            self.get_cell_by_id(cell_ids).add_border_user(
                angles, ratios, user_color)
        else:
            # If angles is not an iterable, then lets repeat the same value
            # for all specified cells by using itertools.repeat to make
            # angles an iterable.
            angles_iter = angles if isinstance(
                angles, Iterable) else itertools.repeat(angles)

            # If ratios is not an iterable, then lets repeat the same value
            # for all specified cells by using itertools.repeat to make
            # ratios an iterable.
            ratios_iter = ratios if isinstance(
                ratios, Iterable) else itertools.repeat(ratios)

            # If user_color is not an iterable of strings, then lets repeat
            # the same value for all specified cells by using
            # itertools.repeat to make user_color an iterable of strings.
            user_color_iter = itertools.repeat(user_color) if (isinstance(
                user_color, str) or user_color is None) else user_color

            all_data = zip(cell_ids, angles_iter, ratios_iter, user_color_iter)

            for cell_id, angle, ratio, color in all_data:
                self.get_cell_by_id(cell_id).add_border_user(
                    angle, ratio, color)

    def delete_all_users(self,
                         cell_id: Optional[IntOrIntIterable] = None) -> None:
        """
        Remove all users from one or more cells.

        If cell_id is an integer > 0, only the users from the cell whose
        index is `cell_id` will be removed. If cell_id is an iterable, then
        the users of cells pointed by it will be removed. If cell_id is
        `None` or not specified, then the users of all cells will be
        removed.

        Parameters
        ----------
        cell_id : int | list[int], optional
            ID(s) of the cells from which users will be removed. If equal
            to None, all the users from all cells will be removed.
        """
        if isinstance(cell_id, Iterable):
            for i in cell_id:
                self.get_cell_by_id(i).delete_all_users()
        elif cell_id is None:
            for cell in self._cells:
                cell.delete_all_users()
        else:
            self.get_cell_by_id(cell_id).delete_all_users()

    def create_wrap_around_cells(
            self,  # pragma: no cover
            include_users_bool: bool = False) -> None:
        """
        This function will create the wrapped cells, as well as the
        wrap info data.

        Parameters
        ----------
        include_users_bool : bool
            Set to True if the users of the original cells should appear in
            the wrapped version.
        """
        positions = Cluster._calc_cell_positions(self.cell_radius,
                                                 self.num_cells,
                                                 self._cell_type,
                                                 self.rotation)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # noinspection PyUnresolvedReferences
        def get_pos_from_relative(rel_center_idx: int,
                                  rel_cell_idx: int) -> complex:
            """

            Parameters
            ----------
            rel_center_idx : int
                Index (starting from 1) of the cell that should be
                considered as the center of the 7-Cell cluster.
            rel_cell_idx : int
                Index (starting from 1) of the desired cell in the 7-Cell
                cluster.

            Returns
            -------
            complex
            """
            return cast(complex, (positions[rel_center_idx - 1, 0] +
                                  positions[rel_cell_idx - 1, 0] + self.pos))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # TODO: Maybe implement for other cluster sizes
        if self.num_cells == 19:
            # Reset the variable with the distance between cells, since we
            # will create new (wrapped) cells.
            self._cell_pos_diffs = None

            # In order to explain the sequences in the for loop below
            # let's take as an example the first value of each sequence,
            #  that is, (17, 7, 13). That means that we will create a
            # wrapped cell for cell 13 and it will be located at a
            # position corresponding to the position of cell 7 in a
            # 7-cell cluster centered at the position of the cell 17 in
            # our 19-cell cluster.
            for rel_center, rel_cell, wrapped_id in zip(
                    # Relative centers
                [
                    17, 18, 19, 8, 8, 9, 9, 10, 11, 12, 13, 13, 14, 15, 15, 16,
                    17, 17, 12, 13, 13, 13, 14, 15, 15, 15, 15, 16, 17, 17, 17,
                    18, 19, 19, 19, 8, 9, 9, 9, 10, 11, 11
                ],
                    # Relative positions regarding the relative center
                [
                    7, 7, 7, 7, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 11,
                    11, 12, 13, 13, 13, 14, 15, 16, 16, 16, 17, 18, 18, 18, 19,
                    8, 8, 8, 9, 10, 10, 10, 11
                ],
                    # ID of the wrapped cell
                [
                    13, 12, 11, 15, 14, 13, 17, 16, 15, 19, 18, 17, 9, 8, 19,
                    11, 10, 9, 8, 7, 6, 16, 10, 2, 7, 18, 12, 3, 2, 8, 14, 4,
                    3, 10, 16, 5, 4, 12, 18, 6, 5, 14
                ]):
                pos = get_pos_from_relative(rel_center, rel_cell)
                w = CellWrap(pos, self.get_cell_by_id(wrapped_id),
                             include_users_bool)
                self._wrapped_cells['wrap{0}_{1}:{2}'.format(
                    wrapped_id, rel_center, rel_cell)] \
                    = w
                self._cell_pos[wrapped_id - 1] = np.append(
                    self._cell_pos[wrapped_id - 1], w.pos)
        else:
            msg = ("Wrap around not implemented for a cluster with {0} "
                   "cells.")
            raise RuntimeError(msg.format(self.num_cells))

    def calc_dists_between_cells(self) -> np.ndarray:
        """
        This method calculates the distance between any two cells in the
        cluster possibly considering wrap around.

        If the `create_wrap_around_cells` method was called before this
        one, then when calculating the distance between two cells if the
        distance between a given cell and the wrapped version of another
        cell is smaller then the distance to that other cell it will be
        sued instead.

        For instance, the

        Returns
        -------
        dists : np.ndarray
            A matrix with the distance from each cell to each other cell in
            the cluster.
        """
        if self._cell_pos_diffs is None:
            diffs = np.empty([self.num_cells, self.num_cells], dtype=complex)

            pos = [np.array(p) for p in self._cell_pos]
            for i, c in enumerate(self._cells):
                a = np.abs(c.pos - pos)
                indexes = map(np.argmin, a)

                for j, idx in enumerate(indexes):
                    diffs[i, j] = (c.pos - pos[j][idx])

            self._cell_pos_diffs = diffs

        return self._cell_pos_diffs

    # This method was originally created to calculate the distance between
    # each user and each cell before wrap around was implemented.
    def calc_dist_all_users_to_each_cell_no_wrap_around(self) -> np.ndarray:
        """
        Returns a matrix with the distance from each user to each cell
        center.

        This matrix is suitable to later calculate the path loss from each
        base station to each mobile station.

        Because usually the base station is the transmitter and the mobile
        station is the receiver the matrix is such that each column
        corresponds to a different base station and each row corresponds to
        a different mobile station.

        Returns
        -------
        all_dists : np.ndarray
            Distance from each cell center to each user.

        Notes
        -----
        There is no explicit indication from which cell each user came
        from. However, in a case, for instance, where there are 3 cells in
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

    def calc_dist_all_users_to_each_cell(self) -> np.ndarray:
        """
        Returns a matrix with the distance from each user to each cell
        center.

        This matrix is suitable to later calculate the path loss from each
        base station to each mobile station.

        Because usually the base station is the transmitter and the mobile
        station is the receiver the matrix is such that each column
        corresponds to a different base station and each row corresponds to
        a different mobile station.

        Returns
        -------
        all_dists : np.ndarray
            Distance from each cell center to each user.

        Notes
        -----
        There is no explicit indication from which cell each user came
        from. However, in a case, for instance, where there are 3 cells in
        the cluster with 2, 2 and 3 users in each of them, respectively,
        then the first 2 rows correspond to the users in the first cell,
        the following 2 rows correspond to the users in the second cell and
        the last three rows correspond to the users in the third cell.
        """
        all_users = self.get_all_users()

        # Array with the position of all users in the cluster (no matter
        # which cell they are assigned to)
        all_users_pos = np.array([u.pos for u in all_users])

        # Array with the position of each cell in the cluster
        all_cells_pos = np.array([c.pos for c in self])

        # Calculate the distance from each user to each cell
        all_dists = np.abs(all_users_pos[:, np.newaxis] - all_cells_pos)

        return all_dists


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Grid Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Grid:
    """
    Class representing a grid of clusters of cells or a single cluster with
    its surrounding cells.

    Valid cluster sizes are given by the formula
    :math:`N = i^2+i*j+j^2`
    where i and j are integer numbers. The values allowed in the Cluster
    are summarized below with the corresponding values of i and j.

    ====  ===
    i, j   N
    ====  ===
    1,0   01
    1,1   03
    2,0   04
    2,1   07
    3,1   13
    3,2   19
    ====  ===
    """
    # Available colors for the clusters. These colors must be understood by
    # the plot library
    _colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def __init__(self) -> None:
        self._cell_radius: float = 0.0
        self._num_cells: int = 0

        # A list with the clusters in the grid
        self._clusters: List[Cluster] = []

    def get_cluster_from_index(self, index: int) -> Cluster:
        """
        Return the cluster object with index `index` in the Grid.

        Parameters
        ----------
        index : int
            The index of the desirable cluster.

        Returns
        -------
        Cluster
            The desired cluster in the Grid.
        """
        return self._clusters[index]

    @property
    def num_clusters(self) -> int:
        """
        Get method for the num_clusters property.

        Returns
        -------
        int
            The number of clusters in teh grid.
        """
        return len(self._clusters)

    def __iter__(self) -> Iterator[Cluster]:
        """Iterator for the clusters in the Grid
        """
        return iter(self._clusters)

    def clear(self) -> None:
        """Clear everything in the grid.
        """
        self._clusters = []
        self._cell_radius = 0.0
        self._num_cells = 0

    def create_clusters(self, num_clusters: int, num_cells: int,
                        cell_radius: float) -> None:
        """
        Create the clusters in the grid.

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

        if num_cells not in frozenset([2, 3, 7]):
            msg = ("The Grid class does not implement the case of clusters"
                   " with {0} cells")
            raise ValueError(msg.format(num_cells))

        self._cell_radius = cell_radius
        self._num_cells = num_cells

        options = {
            2: self._calc_cluster_pos2,
            3: self._calc_cluster_pos3,
            7: self._calc_cluster_pos7
        }

        # Method to calculate the central position of the next cluster
        calc_pos = options[num_cells]

        for _ in range(num_clusters):
            central_pos = calc_pos()
            # cell_radius, num_cells, pos=0 + 0j, cluster_id=None
            new_cluster = Cluster(cell_radius, num_cells, central_pos,
                                  self.num_clusters + 1)
            new_cluster.fill_face_bool = True
            new_cluster.fill_color = Grid._colors[self.num_clusters]
            new_cluster.fill_opacity = 0.3
            self._clusters.append(new_cluster)

    def _calc_cluster_pos2(self) -> complex:
        """
        Calculates the central position of clusters with 2 cells.

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

        if cluster_index == 2:
            angle = np.pi / 3.0
            length = math.sqrt(3) * self._cell_radius
            return length * cmath.exp(1j * angle)

        msg = ("For the two cells per cluster case only two clusters"
               " may be used")
        raise ValueError(msg)

    def _calc_cluster_pos3(self) -> complex:
        """
        Calculates the central position of clusters with 3 cells.

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

        angle = (np.pi / 3.) * (cluster_index - 1) - (np.pi / 6.)
        length = 3 * self._cell_radius
        return length * cmath.exp(1j * angle)

    def _calc_cluster_pos7(self) -> complex:
        """
        Calculates the central position of clusters with 7 cells.

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
            return 0.0 + 0.0j

        angle = math.atan(math.sqrt(3.) / 5.)
        angle += (math.pi / 3) * (cluster_index - 2)
        length = math.sqrt(21) * self._cell_radius
        return length * cmath.exp(1j * angle)

    def plot(self, ax: Optional[Any] = None) -> None:  # pragma: no cover
        """
        Plot the grid of clusters.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the grid will be plotted. If not provided, a new
            figure (and axis) will be created.
        """
        stand_alone_plot = False
        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            # noinspection PyUnresolvedReferences
            _, ax = plt.subplots(figsize=(8, 8))
            stand_alone_plot = True

        for cluster in self._clusters:
            cluster.plot(ax)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()

    # This method is the same in the Shape class
    def _repr_some_format_(
            self,
            extension: str = 'png',
            axis_option: str = 'equal') -> Any:  # pragma: no cover
        """
        Return the representation of the shape in the desired format.

        Parameters
        ----------
        extension : str
            The extension of the desired format. This should be something
            that the savefig method in a matplotlib figure can understand,
            such as 'png', 'svg', stc.
        axis_option : str
            Option to be given to the ax.axis function.

        Returns
        -------
        Any
            The representation in the desired format.
        """
        plt.ioff()  # turn off interactive mode
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        output = BytesIO()
        self.plot(ax)
        ax.axis(axis_option)
        fig.savefig(output, format=extension)
        output.seek(0)
        plt.close(fig)
        plt.ion()  # turn on interactive mode

        return output.getvalue()

    # This method is the same in the Shape class
    def _repr_png_(self) -> Any:  # pragma: no cover
        """
        Return the PNG representation of the shape.
        """
        return self._repr_some_format_('png')

    # This method is the same in the Shape class
    def _repr_svg_(self) -> Any:  # pragma: no cover
        """
        Return the SVG representation of the shape.
        """
        return self._repr_some_format_('svg')


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
