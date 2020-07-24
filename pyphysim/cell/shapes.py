#!/usr/bin/env python
"""
Module implementing geometric shapes.

Each shape knows how to plot itself.
"""

try:
    # noinspection PyUnresolvedReferences
    from matplotlib import pyplot as plt
    # noinspection PyUnresolvedReferences
    from matplotlib import patches, path
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

import cmath
import math
from abc import ABCMeta, abstractmethod
from io import BytesIO
from typing import Any, Optional, TypeVar, cast

import numpy as np

__all__ = ['Coordinate', 'Shape', 'Hexagon', 'Rectangle', 'Circle']

ComplexOrArray = TypeVar("ComplexOrArray", np.ndarray, complex)


class Coordinate:
    """
    Base class for a coordinate in a 2D grid.

    A Coordinate object knows its location in the grid (represented as a
    complex number) and how to calculate the distance from it to another
    location.
    """
    def __init__(self, pos: complex):
        """
        Initializes the Coordinate object.

        Parameters
        ----------
        pos : complex
            Coordinate in the complex grid.
        """
        self._pos: complex = pos

    # xxxxxxxxxx pos property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def pos(self) -> complex:
        """
        Get the coordinate position as a complex number.

        Returns
        -------
        complex
            The coordinate position (a complex number).
        """
        return self._pos

    @pos.setter
    def pos(self, value: complex) -> None:
        """
        Set the coordinate position.

        Parameters
        ----------
        value : complex
            The new coordinate position (a complex number).
        """
        self._pos = value

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def calc_dist(self, other: "Coordinate") -> float:
        """
        Calculates the distance to another coordinate.

        Parameters
        ----------
        other : Coordinate
            A different coordinate object.

        Returns
        -------
        dist : float
            Distance from self to the other coordinate.
        """
        dist: float = np.abs(self.pos - other.pos)
        return dist

    def move_by_relative_coordinate(self, rel_pos: complex) -> None:
        """
        Move from the current position to the relative coordinate.

        This is equivalent to moving to a new position given by the current
        position plus `coordinate`.

        Parameters
        ----------
        rel_pos : complex
            Relative coordinate
        """
        self.pos += rel_pos

    def move_by_relative_polar_coordinate(self, radius: float,
                                          angle: float) -> None:
        """
        Move from the current position to the relative coordinate.

        This is equivalent to moving to a new position given by the current
        position plus a the provided coordinate.

        Parameters
        ----------
        radius : float
            Distance of the movement in the direction given by `angle`.
        angle : float
            Angle (in radians) pointing the direction of the movement.
        """
        rel_pos = cmath.rect(radius, angle)
        self.move_by_relative_coordinate(rel_pos)

    def __repr__(self) -> str:  # pragma: no cover
        """
        Representation of a Coordinate object.

        Returns
        -------
        str
            The string representation of the object.
        """
        return "{0}({1})".format(self.__class__.__name__, self.pos)


class Shape(Coordinate):
    """
    Base class for all 2D shapes.

    Each subclass must implement the _get_vertex_positions method.

    Parameters
    ----------
    pos : complex
        Coordinate of the shape in the complex grid.
    radius : float
        Radius of the shape. It must be positive.
    rotation : float
        Rotation of the shape in degrees.
    """
    # The Shape class is an abstract class and all methods marked as
    # 'abstract' must be implemented in a subclass.
    __metaclass__ = ABCMeta

    def __init__(self, pos: complex, radius: float, rotation: float = 0, **kw):
        super().__init__(pos=pos, **kw)

        self._radius = radius
        self._rotation = rotation

        # Properties for the plot representation
        self.fill_face_bool = False
        self.fill_color = 'r'
        self.fill_opacity = 0.1

        # Default figsize passed to matplotlib when the plot or the
        # _repr_some_format_ methods are called. Note that if you passed
        # the 'ax' argument in the plot method this will not be used.
        self.figsize = (8, 8)

    def __repr__(self) -> str:  # pragma: no cover
        """
        Representation of a Shape object.

        Returns
        -------
        str
            The string representation of the object.
        """
        return "{0}(pos={1},radius={2},rotation={3})".format(
            self.__class__.__name__, self.pos, self.radius, self.rotation)

    # xxxxx radius property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def radius(self) -> float:
        """
        Get method for the radius property.

        Returns
        -------
        float
            The Shape radius.
        """
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        """
        Set method for the radius property.

        Parameters
        ----------
        value : float
            The new radius.
        """
        self._radius = value

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx rotation property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
            The new shape rotation.
        """
        self._rotation = value

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    @abstractmethod
    def _get_vertex_positions(self) -> np.ndarray:  # pragma: no cover
        """
        Calculates the vertex positions ignoring any rotation and
        considering that the shape is at the origin (rotation and
        translation will be added automatically later).

        Returns
        -------
        vertex_positions : np.ndarray
            The positions of the vertexes of the shape (as a 1D numpy
            array).

        Notes
        -----
        Not implemented. Must be implemented in a subclass and return a
        one-dimensional numpy array (complex dtype) with the vertex
        positions.
        """
        raise NotImplementedError(
            ('get_vertex_positions still needs to be implemented in the '
             '{0} class'.format(self.__class__.__name__)))

    # xxxxx vertex property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def vertices_no_trans_no_rotation(self) -> np.ndarray:  # pragma: no cover
        """
        Get the shape vertexes without translation and rotation.

        Returns
        -------
        vertex_positions : np.ndarray
            The positions of the vertexes of the shape without any
            translation or rotation (as a 1D numpy array).
        """
        return self._get_vertex_positions()

    @property
    def vertices(self) -> np.ndarray:
        """
        Get method for the vertices property.

        Returns
        -------
        np.ndarray
            The shape vertexes.
        """
        vertex_positions: np.ndarray = self._get_vertex_positions()
        vertex_positions2: np.ndarray = self.pos + Shape.calc_rotated_pos(
            vertex_positions, self.rotation)
        return vertex_positions2

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def is_point_inside_shape(self, point: complex) -> bool:
        """
        Test is a point is inside the shape.

        Parameters
        ----------
        point
            A single complex number.

        Returns
        -------
        inside_or_not
            True if `point` is inside the shape, False otherwise.
        """
        mpl_path = path.Path(from_complex_array_to_real_matrix(self.vertices))

        # This code is used with Matplotlib version 1.2 or higher.
        return cast(bool, mpl_path.contains_point([point.real, point.imag]))

        # xxxxx Code for Matplotlib version 1.1 xxxxxxxxxxxxxxxxxxxxxxxxxxx
        # The code below was used for Matplotlib lower then version
        # 1.2. However, since Matplotlib version 1.2 the pnpoly function is
        # deprecated

        # # pnpoly returns 1 if point is inside the polygon and 0 otherwise
        # import matplotlib.nxutils as mnx
        # return mnx.pnpoly(point.real, point.imag,
        #                   from_complex_array_to_real_matrix(
        #                       self.vertices)) == 1

    # noinspection PyUnresolvedReferences
    def get_border_point(
            self,
            angle: float,
            ratio: Optional[float] = None) -> complex:  # pylint: disable=R0914
        """
        Calculates the coordinate of the point that intercepts the
        border of the shape if we go from the origin with a given angle
        (in degrees).

        Parameters
        ----------
        angle : float
            Angle in degrees.
        ratio : float
            The ratio from the cell center to the border where the desired
            point is located. This MUST be a value between 0 and 1.

        Returns
        -------
        point : complex
            A point in the line between the shape's center and the shape's
            border with the desired angle. If ratio is equal to one the
            point will be in the end of the line (touching the shape's
            border)
        """
        if ratio is None:
            ratio = 1.0

        angle_rad = np.pi * angle / 180.

        # Which point we get if we walk a distance of cell radius in the
        # desired angle direction?
        point = cast(complex, self.pos + self._radius * np.exp(angle_rad * 1j))

        # Calculates the distance of this point to all vertices and finds
        # the closest vertices
        dists = np.abs(self.vertices - point)
        # Get the two closest vertices from point
        closest_vertices = self.vertices[np.argsort(dists)[:2]]

        # The equation of a straight line is given by "y = ax + b". We have
        # two points in this line (the two closest vertices) and we can use
        # them to find 'a' and 'b'. First let's find the different of these
        # two closest vertexes
        diff = closest_vertices[0] - closest_vertices[1]

        # xxxxx Special case for a vertical line xxxxxxxxxxxxxxxxxxxxxxxxxx
        # noinspection PyTypeChecker
        if np.allclose(diff.real, 0.0, atol=1e-15):
            # If the the real part of diff is equal to zero, that means
            # that the straight line is actually a vertical
            # line. Therefore, all we need to do to get the border point is
            # to start from the shape's center and go with the desired
            # angle until the value in the 'x' axis is equivalent to
            # closest_vertices[0].real.
            adjacent_side = closest_vertices[0].real - self.pos.real
            side = np.tan(angle_rad) * adjacent_side
            point = self.pos + adjacent_side + 1j * side
            # Now all that is left to do is apply the ratio, which only
            # means that the returned point is a linear combination between
            # the shape's central position and the point at the border of
            # the shape

            return (1 - ratio) * self.pos + ratio * point
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Calculates the 'a' and 'b' in the line equation "y=ax+b"
        a = diff.imag / diff.real
        b = closest_vertices[1].imag - a * closest_vertices[1].real

        # Note that is we start from self.pos and walk in the direction
        # pointed by the angle by "some step" we should reach the line
        # where the two closest vertexes are. If we can find this "step"
        # then we will get our desired point.
        # That is, for the step "z" we have
        #    self.pos + np.exp(1j * angle_rad) * z = complex(x, a * x + b)
        # Which we can write as the system of equations
        #    self.pos.real + np.exp(1j * angle).real * z = x
        #    self.pos.imag + np.exp(1j * angle).imag * z = a * x + b
        # Lets create some aliases for the constants so that
        #     A + B * z = x
        #     C + D * z = a * x + b
        A = self.pos.real
        B = np.exp(1j * angle_rad).real
        C = self.pos.imag
        D = np.exp(1j * angle_rad).imag
        # Through some algebraic manipulation the correct step "z" is given
        # by
        z = (A * a + b - C) / (D - (a * B))

        # Now we can finally find the desired point at the border of the
        # shape
        point = self.pos + np.exp(1j * angle_rad) * z

        # Now all that is left to do is apply the ratio, which only means
        # that the returned point is a linear combination between the
        # shape's central position and the point at the border of the shape
        return (1 - ratio) * self.pos + ratio * point

    # noinspection PyShadowingNames,PyShadowingNames
    def plot(self, ax: Any = None) -> None:  # pragma: no cover
        """
        Plot the shape using the matplotlib library.

        Parameters
        ----------
        ax : A matplotlib ax, optional
            The ax where the shape will be plotted. If not provided, a new
            figure (and ax) will be created.

        Notes
        -----
        If an axes 'ax' is specified, then the shape is added to that
        axes. Otherwise a new figure and axes are created and the shape is
        plotted to that.
        """
        stand_alone_plot = False

        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            _, ax = plt.subplots(figsize=self.figsize)
            stand_alone_plot = True

        if self.fill_face_bool:
            # Matplotlib does not seem to allow us to set a different alpha
            # value for the face and the edges. Therefore, we will need to
            # plot twice to get that effect.
            polygon_face = patches.Polygon(
                from_complex_array_to_real_matrix(self.vertices),
                True,
                facecolor=self.fill_color,
                edgecolor='none',  # No edges
                alpha=self.fill_opacity)
            ax.add_patch(polygon_face)

        polygon_edges = patches.Polygon(
            from_complex_array_to_real_matrix(self.vertices),
            True,
            facecolor='none',  # No face
            edgecolor="black",
            alpha=1)

        ax.add_patch(polygon_edges)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()

    # noinspection PyShadowingNames
    def _repr_some_format_(
            self,
            extension: str = 'png',
            axis_option: str = 'equal') -> Any:  # pragma: nocover
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
        output
        """
        plt.ioff()  # turn off interactive mode
        fig = plt.figure(figsize=self.figsize)
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

    def _repr_png_(self) -> Any:  # pragma: no cover
        """
        Return the PNG representation of the shape.
        """
        return self._repr_some_format_('png')

    def _repr_svg_(self) -> Any:  # pragma: no cover
        """
        Return the SVG representation of the shape.
        """
        return self._repr_some_format_('svg')

    @staticmethod
    def calc_rotated_pos(cur_pos: ComplexOrArray,
                         angle: float) -> ComplexOrArray:
        """
        Rotate the complex numbers in the `cur_pos` array by `angle` (in
        degrees) around the origin.

        Parameters
        ----------
        cur_pos : complex | np.ndarray
            The complex number(s) to be rotated.
        angle: float
            Angle in degrees to rotate the positions.

        Returns
        -------
        rotated_pos : complex | np.ndarray
            The rotate complex number(s).
        """
        angle_rad = angle * np.pi / 180.
        return cur_pos * np.exp(1j * angle_rad)  # type: ignore


class Hexagon(Shape):
    """
    Hexagon shape class.

    Besides the `pos`, `radius` and `rotation` properties from the Shape
    base class, the Hexagon also has a height property (read-only) from the
    base of the Hexagon to its center.

    Parameters
    ----------
    pos : complex
        Coordinate of the shape in the complex grid.
    radius : float
        Radius of the hexagon. It must be a positive number.
    rotation : float
        Rotation of the hexagon in degrees.
    """
    def __init__(self, pos: complex, radius: float, rotation: float = 0, **kw):
        super().__init__(pos=pos, radius=radius, rotation=rotation, **kw)

    @property
    def height(self) -> float:
        """
        Get method for the height property.

        Returns
        -------
        float
            The height of the Hexagon.
        """
        return self._radius * math.sqrt(3.) / 2.0

    def _get_vertex_positions(self) -> np.ndarray:
        """
        Calculates the vertex positions ignoring any rotation and
        considering that the hexagon is at the origin (rotation and
        translation will be added automatically later).

        Returns
        -------
        vertex_positions : np.ndarray
            The positions of the vertexes of the shape.
        """
        vertex_positions: np.ndarray = np.zeros(6, dtype=complex)
        vertex_positions[0] = complex(-self._radius / 2., -self.height)
        # noinspection PyTypeChecker
        angles = np.linspace(0, 240, 5) * np.pi / 180.

        for k in range(5):
            # noinspection PyUnresolvedReferences
            vertex_positions[k + 1] = (vertex_positions[k] +
                                       self._radius * np.exp(angles[k] * 1j))

        return vertex_positions


class Rectangle(Shape):
    """
    Rectangle shape class.

    The rectangle is initialized from two coordinates as well as from
    the rotation.

    Parameters
    ----------
    first : complex
        First coordinate (without rotation).
    second : complex
        Second coordinate (without rotation).
    rotation : float
        Rotation of the rectangle in degrees.
    """
    def __init__(self,
                 first: complex,
                 second: complex,
                 rotation: float = 0,
                 **kw):
        central_pos = (first + second) / 2
        radius = np.abs(second - central_pos)
        super().__init__(pos=central_pos,
                         radius=radius,
                         rotation=rotation,
                         **kw)
        self._lower_coord = complex(min(first.real, second.real),
                                    min(first.imag, second.imag))
        self._upper_coord = complex(max(first.real, second.real),
                                    max(first.imag, second.imag))

    def __repr__(self) -> str:  # pragma: no cover
        """
        Representation of a Rectangle object.

        Returns
        -------
        str
            The string representation of the Rectangle object.
        """
        return "{0}(A={1},B={2},rotation={3})".format(self.__class__.__name__,
                                                      self._lower_coord,
                                                      self._upper_coord,
                                                      self.rotation)

    def _get_vertex_positions(self) -> np.ndarray:
        """
        Calculates the vertex positions ignoring any rotation and
        considering that the rectangle is at the origin (rotation and
        translation will be added automatically later).

        Returns
        -------
        vertex_positions : np.ndarray
            The positions of the vertexes of the shape.
        """
        vertex_positions: np.ndarray = np.zeros(4, dtype=complex)
        A = self._lower_coord - self.pos
        B = self._upper_coord - self.pos
        vertex_positions[0] = A
        vertex_positions[1] = complex(B.real, A.imag)
        vertex_positions[2] = B
        vertex_positions[3] = complex(A.real, B.imag)
        return vertex_positions

    def _repr_some_format_(
            self,
            extension: str = 'png',
            axis_option: str = 'tight') -> Any:  # pragma: no cover
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

        Notes
        -----
        We only subclass the _repr_some_format_ method from the Shape class
        here so that we can change the axis_option to 'tight' (default in
        the Shape class is 'equal').
        """
        return Shape._repr_some_format_(self,
                                        extension=extension,
                                        axis_option=axis_option)

    def is_point_inside_shape(self, point: complex) -> bool:
        """
        Test is a point is inside the rectangle

        Parameters
        ----------
        point : complex
            A single complex number.

        Returns
        -------
        bool
            True if `point` is inside the rectangle, False otherwise.
        """
        min_x = min(self._lower_coord.real, self._upper_coord.real)
        max_x = max(self._lower_coord.real, self._upper_coord.real)
        min_y = min(self._lower_coord.imag, self._upper_coord.imag)
        max_y = max(self._lower_coord.imag, self._upper_coord.imag)

        point_x = point.real
        point_y = point.imag
        if point_x < min_x:
            return False
        if point_x > max_x:
            return False
        if point_y < min_y:
            return False
        if point_y > max_y:
            return False
        return True


class Circle(Shape):
    """
    Circle shape class.

    A circle is initialized only from a coordinate and a radius.

    Parameters
    ----------
    pos : complex
        Coordinate of the center of the circle.
    radius : float
        Circle's radius.
    """
    def __init__(self, pos: complex, radius: float):
        super().__init__(pos=pos, radius=radius)

    def _get_vertex_positions(self) -> np.ndarray:
        """
        Calculates the vertex positions considering that the circle is
        at the origin (translation will be added automatically later).

        Returns
        -------
        vertex_positions : np.ndarray
            The positions of the vertexes of the shape.

        Notes
        -----
        It does not make much sense to get the vertexes of a circle, since
        a circle 'has' infinite vertexes. However, for consistence with the
        Shape's class interface the _get_vertex_positions is implemented
        such that it returns a subset of the circle vertexes. The number of
        returned vertexes was arbitrarily chosen as 12.
        """
        num_vertexes = 12
        angles: np.ndarray = np.linspace(
            0, (num_vertexes - 1.) / num_vertexes * 2 * np.pi, num_vertexes)

        vertex_positions: np.ndarray = self._radius * np.exp(1j * angles)
        return vertex_positions

    def get_border_point(self,
                         angle: float,
                         ratio: Optional[float] = None) -> complex:
        """
        Calculates the coordinate of the point that intercepts the
        border of the circle if we go from the origin with a given angle
        (in degrees).

        Parameters
        ----------
        angle : float
            Angle in degrees
        ratio : float
            The ratio from the cell center to the border where the desired
            point is located. It must be a value between 0 and 1.

        Returns
        -------
        point : complex
            A point in the line between the circle's center and the
            circle's border with the desired angle. If ratio is equal to
            one the point will be in the end of the line (touching the
            circle's border)
        """
        if ratio is None:
            ratio = 1.0

        angle_rad = np.pi * angle / 180.
        return cast(complex,
                    self.pos + np.exp(1j * angle_rad) * self.radius * ratio)

    def is_point_inside_shape(self, point: complex) -> bool:
        """
        Test is a point is inside the circle

        Parameters
        ----------
        point
            A single complex number.

        Returns
        -------
        inside_or_not
            True if `point` is inside the circle, False otherwise.
        """
        return cast(bool, np.abs(self.pos - point) < self.radius)

    # noinspection PyShadowingNames,PyShadowingNames
    def plot(self, ax: Any = None) -> None:  # pragma: no cover
        """
        Plot the circle using the Matplotlib library.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the shape will be plotted. If not provided,
            a new figure (and axis) will be created.

        Notes
        -----
        If an axes 'ax' is specified, then the shape is added to that
        axes. Otherwise a new figure and axes are created and the shape is
        plotted to that.
        """
        stand_alone_plot = False

        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        if self.fill_face_bool:
            # Matplotlib does not seem to allow us to set a different alpha
            # value for the face and the edges. Therefore, we will need to
            # plot twice to get that effect.
            circle_face = patches.Circle(
                [self.pos.real, self.pos.imag],
                self.radius,
                facecolor=self.fill_color,
                edgecolor='none',  # No edges
                alpha=self.fill_opacity)
            ax.add_patch(circle_face)

        circle_edges = patches.Circle(
            [self.pos.real, self.pos.imag],
            self.radius,
            facecolor='none',  # No face
            alpha=1)
        ax.add_patch(circle_edges)

        if stand_alone_plot is True:
            ax.plot()
            plt.show()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def from_complex_array_to_real_matrix(a: np.ndarray) -> np.ndarray:
    """
    Convert an array of complex numbers to a matrix of real numbers.

    Parameters
    ----------
    a : np.ndarray
        A numpy array of complex numbers with N elements.

    Returns
    -------
    np.ndarray
        The converted array with dimension N x 2. That is, a 2D numpy
        array.

    Notes
    -----
    We use complex number to represent coordinates, where the real part is
    the 'x' coordinate and the imaginary part is the 'y' coordinate pf a
    point.

    However, matplotlib methods need the coordinates to be separated. For
    instance, a vector of complex coordinates must be converted to a matrix
    with two columns, where the two columns are the 'x' and 'y' coordinates
    and each row corresponds to a point.

    The method from_complex_array_to_real_matrix does exactly this
    conversion.

    Examples
    --------
    >>> a = np.array([1+2j, 3-4j, 5+6j])
    >>> print(from_complex_array_to_real_matrix(a))
    [[ 1.  2.]
     [ 3. -4.]
     [ 5.  6.]]
    """
    num_elem = np.size(a)
    a.dtype = a.real.dtype
    a = np.reshape(a, (num_elem, 2))
    return a


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__1':  # pragma: no cover
    ax = plt.axes()
    h = Hexagon(2 + 3j, 2, 30)

    # print "Border Point is: {0}".format(point)
    h.plot(ax)

    POINT1 = h.get_border_point(90, 0.9)
    ax.plot(POINT1.real, POINT1.imag, 'ro')

    POINT2 = h.get_border_point(10, 0.9)
    ax.plot(POINT2.real, POINT2.imag, 'go')

    POINT3 = h.get_border_point(30, 0.9)
    ax.plot(POINT3.real, POINT3.imag, 'bo')

    POINT4 = h.get_border_point(107, 1)
    ax.plot(POINT4.real, POINT4.imag, 'bo')

    ax.plot(h.pos.real, h.pos.imag, 'ro')

    # print h.vertices
    ax.axis('equal')
    plt.show()

if __name__ == '__main__':  # pragma: no cover
    ax = plt.axes()

    h = Hexagon(0, 1)
    h.rotation = 30
    h.fill_face_bool = True

    plt.hold(True)
    r = Rectangle(0, 2 + 1j, 0)
    r.rotation = 15
    r.fill_face_bool = True
    r.fill_color = 'b'

    c = Circle(0.2 + 0.4j, 1)
    c.fill_face_bool = True
    c.fill_color = 'g'

    r.plot(ax)
    h.plot(ax)
    c.plot(ax)

    ax.plot()
    plt.axis('equal')
    plt.show()

if __name__ == '__main__1':  # pragma: no cover
    ax = plt.axes()
    c = Circle(2 + 3j, 2)

    # print "Border Point is: {0}".format(point)
    c.plot(ax)

    for v in c.vertices:
        ax.plot(v.real, v.imag, 'bo')

    ax.plot(c.pos.real, c.pos.imag, 'ro')

    # print c.vertices
    ax.axis('equal')
    plt.show()
