#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module implementing geometric shapes.

Each shape knows how to plot itself.
"""

__revision__ = "$Revision$"

try:
    from matplotlib import pylab
    from matplotlib import patches, path
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

import numpy as np

__all__ = ['Coordinate', 'Shape', 'Hexagon', 'Rectangle', 'Circle']


class Coordinate(object):
    """Base class for a coordinate in a grid.

    A Coordinate object knows its location in the grid (represented as a
    complex number) and how to calculate the distance from it to another
    location.
    """

    def __init__(self, pos):
        """Initializes the Coordinate object.

        Parameters
        ----------
        pos : complex
            Coordinate in the complex grid.
        """
        self.pos = pos

    def calc_dist(self, other):
        """Calculates the distance to another coordinate.

        Parameters
        ----------
        other : a coordinate object
            A different coordinate object.

        Returns
        -------
        dist : float
            Distance from self to the other coordinate.
        """
        dist = np.abs(self.pos - other.pos)
        return dist


class Shape(Coordinate):
    """Base class for all 2D shapes.

    Each subclass must implement the _get_vertex_positions method.
    """
    def __init__(self, pos, radius, rotation=0):
        """Initializes the shape.

        Parameters
        ----------
        pos : complex
            Coordinate of the shape in the complex grid.
        radius : float (positive)
            Radius of the shape.
        rotation : float
            Rotation of the shape in degrees.
        """
        Coordinate.__init__(self, pos)

        self._radius = radius
        self._rotation = rotation

        # Properties for the plot representation
        self.fill_face_bool = False
        self.fill_color = 'r'
        self.fill_opacity = 0.1

    # xxxxx radius property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Property to get the shape radius.
    def _set_radius(self, value):
        """Set method for the radius property"""
        self._radius = value

    # Property to set the shape radius
    def _get_radius(self):
        """Set method for the radius property"""
        return self._radius

    radius = property(_get_radius, _set_radius)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx rotation property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def _set_rotation(self, value):
        """Set method for the rotation property."""
        self._rotation = value

    def _get_rotation(self):
        """Get method for the rotation property."""
        return self._rotation

    rotation = property(_get_rotation, _set_rotation)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _get_vertex_positions(self):  # pragma: no cover
        """Calculates the vertex positions ignoring any rotation and
        considering that the shape is at the origin (rotation and
        translation will be added automatically later).

        Returns
        -------
        vertex_positions : 1D numpy array
            The positions of the vertexes of the shape.

        Notes
        -----
        Not implemented. Must be implemented in a subclass and return a
        one-dimensional numpy array (complex dtype) with the vertex
        positions.

        """
        raise NotImplementedError('get_vertex_positions still needs to be implemented in the {0} class'.format(self.__class__.__name__))

    # xxxxx vertex property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def _get_vertices(self):
        """Get method for the vertices property."""
        vertex_positions = self._get_vertex_positions()
        vertex_positions = self.pos + Shape._rotate(
            vertex_positions, self._rotation)
        return vertex_positions

    vertices = property(_get_vertices)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Shape's Path xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # A Matplotlib "Path" corresponding to the shape
    def _get_path(self):
        """Get method for the path property.

        The `path` property returns a Matplotlib Path for the shape.
        """
        return path.Path(from_complex_array_to_real_matrix(self.vertices))

    path = property(_get_path)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def is_point_inside_shape(self, point):
        """Test is a point is inside the shape.

        Parameters
        ----------
        point : complex
            A single complex number.

        Returns
        -------
        inside_or_not : bool
            True if `point` is inside the shape, False otherwise.
        """
        # This code is used with Matplotlib version 1.2 or higher.
        return self.path.contains_point([point.real, point.imag])

        # xxxxx Code for Matplotlib version 1.1 xxxxxxxxxxxxxxxxxxxxxxxxxxx
        # The code below was used for Matplotlib lower then version
        # 1.2. However, since Matplotlib version 1.2 the pnpoly function is
        # deprecated

        ## pnpoly returns 1 if point is inside the polygon and 0 otherwise
        # import matplotlib.nxutils as mnx
        # return mnx.pnpoly(point.real, point.imag, from_complex_array_to_real_matrix(self.vertices)) == 1

    def get_border_point(self, angle, ratio):  # pylint: disable=R0914
        """Calculates the coordinate of the point that intercepts the
        border of the shape if we go from the origin with a given angle
        (in degrees).

        Parameters
        ----------
        angle : floar
            Angle in degrees.
        ratio : float (between 0 and 1)
            The ratio from the cell center to the border where the desired
            point is located.

        Returns
        -------
        point : complex
            A point in the line between the shape's center and the shape's
            border with the desired angle. If ratio is equal to one the
            point will be in the end of the line (touching the shape's
            border)

        """
        angle_rad = np.pi * angle / 180.

        # Which point we get if we walk a distance of cell radius in the
        # desired angle direction?
        point = self.pos + self._radius * np.exp(angle_rad * 1j)

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
        # Which we can write as the sytem of equations
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

    def plot(self, ax=None):  # pragma: no cover
        """Plot the shape using the matplotlib library.

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

        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = pylab.axes()
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
            alpha=1)

        ax.add_patch(polygon_edges)

        if stand_alone_plot is True:
            ax.plot()
            pylab.show()

    @staticmethod
    def _rotate(cur_pos, angle):
        """Rotate the complex numbers in the `cur_pos` array by `angle` (in
        degrees)

        Parameters
        ----------
        cur_pos : complex or numpy array of complexes
            The complex number(s) to be rotated.
        angle: float
            Angle in degrees to rotate the positions.

        Returns
        -------
        rotated_pos : complex or numpy array of complexes
            The rotate complex number(s).
        """
        angle_rad = angle * np.pi / 180.
        return cur_pos * np.exp(1j * angle_rad)


class Hexagon(Shape):
    """Hexagon shape class.

    Besides the `pos`, `radius` and `rotation` properties from the Shape
    base class, the Hexagon also has a height property (read-only) from the
    base of the Hexagon to its center.
    """

    def __init__(self, pos, radius, rotation=0):
        """Initializes the Hexagon object.

        Parameters
        ----------
        pos : complex
            Coordinate of the shape in the complex grid.
        radius : float (positive number)
            Radius of the hexagon.
        rotation : float
            Rotation of the hexagon in degrees.
        """
        Shape.__init__(self, pos, radius, rotation)

    def _get_height(self):
        """Get method for the height property."""
        return self._radius * np.sqrt(3.) / 2.0
    height = property(_get_height)

    def _get_vertex_positions(self):
        """Calculates the vertex positions ignoring any rotation and considering
        that the hexagon is at the origin (rotation and translation will be
        added automatically later).

        Returns
        -------
        vertex_positions : 1D numpy array
            The positions of the vertexes of the shape.

        """
        vertex_positions = np.zeros(6, dtype=complex)
        vertex_positions[0] = complex(-self._radius / 2., -self.height)
        angles = np.linspace(0, 240, 5) * np.pi / 180.

        for k in range(5):
            vertex_positions[k + 1] = vertex_positions[k] + self._radius * np.exp(angles[k] * 1j)
        return vertex_positions


# TODO: Implement a is_point_inside_shape method. The method from the Shape
# class works, but because a Rectangle is very simple it would be more
# efficient to re-implement this method. Note that you will have to write a
# testcase for this as well in the cell_package_test.py file.
class Rectangle(Shape):
    """Rectangle shape class.
    """

    def __init__(self, A, B, rotation=0):
        """Initializes the Rectangle object.

        The rectangle is initialized from two coordinates as well as from
        the rotation.

        Parameters
        ----------
        A : complex
            First coordinate (without rotation).
        B : complex
            Second coordinate (without rotation).
        rotation : float
            Rotation of the rectangle in degrees.
        """
        central_pos = (A + B) / 2
        radius = np.abs(B - central_pos)
        Shape.__init__(self, central_pos, radius, rotation)
        self._lower_coord = complex(min(A.real, B.real), min(A.imag, B.imag))
        self._upper_coord = complex(max(A.real, B.real), max(A.imag, B.imag))

    def _get_vertex_positions(self):
        """Calculates the vertex positions ignoring any rotation and considering
        that the rectangle is at the origin (rotation and translation will
        be added automatically later).

        Returns
        -------
        vertex_positions : 1D numpy array
            The positions of the vertexes of the shape.

        """
        vertex_positions = np.zeros(4, dtype=complex)
        A = self._lower_coord - self.pos
        B = self._upper_coord - self.pos
        vertex_positions[0] = A
        vertex_positions[1] = complex(B.real, A.imag)
        vertex_positions[2] = B
        vertex_positions[3] = complex(A.real, B.imag)
        return vertex_positions


class Circle(Shape):
    """Circle shape class.
    """

    def __init__(self, pos, radius):
        """Initializes the Circle.

        A circle is initialized only from a coordinate and a radius.

        Parameters
        ----------
        pos : complex
            Coordinate of the center of the circle.
        radius : floar
            Circle's radius.

        """
        Shape.__init__(self, pos, radius)

    def _get_vertex_positions(self):
        """Calculates the vertex positions considering that the circle is at the
        origin (translation will be added automatically later).

        Returns
        -------
        vertex_positions : 1D numpy array
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
        angles = np.linspace(0,
                             (num_vertexes - 1.) / num_vertexes * 2 * np.pi,
                             num_vertexes)
        vertex_positions = self._radius * np.exp(1j * angles)
        return vertex_positions

    def get_border_point(self, angle, ratio):
        """Calculates the coordinate of the point that intercepts the
        border of the circle if we go from the origin with a given angle
        (in degrees).

        Parameters
        ----------
        angle : float
            Angle (in degrees)
        ratio : float (between 0 and 1)
            The ratio from the cell center to the border where the desired
            point is located.

        Returns
        -------
        point : complex
            A point in the line between the circle's center and the circle's
            border with the desired angle. If ratio is equal to one the
            point will be in the end of the line (touching the circle's
            border)
        """
        angle_rad = np.pi * angle / 180.
        return self.pos + np.exp(1j * angle_rad) * self.radius * ratio

    def is_point_inside_shape(self, point):
        """Test is a point is inside the circle

        Parameters
        ----------
        point : complex
            A single complex number.

        Returns
        -------
        inside_or_not : bool
            True if `point` is inside the circle, False otherwise.
        """
        return (np.abs(self.pos - point) < self.radius)

    def plot(self, ax=None):  # pragma: no cover
        """Plot the circle using the Matplotlib library.

        Parameters
        ----------
        ax : A matplotlib axis, optional
            The axis where the shape will be plotted. If not provided, a new
            figure (and axis) will be created.

        Notes
        -----
        If an axes 'ax' is specified, then the shape is added to that
        axes. Otherwise a new figure and axes are created and the shape is
        plotted to that.
        """
        stand_alone_plot = False

        if (ax is None):
            # This is a stand alone plot. Lets create a new axes.
            ax = pylab.axes()
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
            pylab.show()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def from_complex_array_to_real_matrix(a):
    """Convert an array of complex numbers to a matrix of real numbers.

    Parameters
    ----------
    a : 1D numpy array (complex dtype)
        A numpy array of complex numbers with N elements.

    Returns
    -------
    converted_a : 2D numpy array (float dtype)
        The converted array with dimension N x 2.

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
    ax = pylab.axes()
    h = Hexagon(2 + 3j, 2, 30)

    #print "Border Point is: {0}".format(point)
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

    #print h.vertices
    ax.axis('equal')
    pylab.show()

if __name__ == '__main__1':  # pragma: no cover
    from matplotlib import pyplot as plt
    ax = pylab.axes()

    h = Hexagon(0, 1)
    h.rotation = 30
    h.fill_face_bool = True

    pylab.hold(True)
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
    ax = pylab.axes()
    c = Circle(2 + 3j, 2)

    #print "Border Point is: {0}".format(point)
    c.plot(ax)

    for v in c.vertices:
        ax.plot(v.real, v.imag, 'bo')

    ax.plot(c.pos.real, c.pos.imag, 'ro')

    #print c.vertices
    ax.axis('equal')
    pylab.show()
