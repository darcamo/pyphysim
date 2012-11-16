#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module implementing geometric shapes.

Each shape knows how to plot itself.
"""

try:
    from matplotlib import pylab
    from matplotlib import patches
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

import numpy as np


class Coordinate(object):
    """Base class for a coordinate in a grid.

    A Coordinate object knows its location in the grid (represented as a
    complex number) and how to calculate the distance from it to another
    location.
    """

    def __init__(self, pos):
        """
        Arguments:
        - `pos`: Coordinate in the grid (A complex number).
        """
        self.pos = pos

    def calc_dist(self, other):
        """Calculates the distance to another coordinate.

        Arguments:
        - `other`: A different coordinate (a complex number).
        """
        dist = np.abs(self.pos - other.pos)
        return dist


class Shape(Coordinate):
    """Base class for all 2D shapes.

    Each subclass must implement the _get_vertex_positions method.
    """
    def __init__(self, pos, radius, rotation=0):
        """Initializes the shape.

        Arguments:
        - `pos`: Coordinate of the shape in the 2D grid (a complex number).
        - `radius`: Radius of the shape (a positive real number).
        - `rotation`: Rotation of the shape in degrees (a positive real
                      number).
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
        self._radius = value

    # Property to set the shape radius
    def _get_radius(self):
        return self._radius

    radius = property(_get_radius, _set_radius)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx rotation property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def _set_rotation(self, value):
        self._rotation = value

    def _get_rotation(self):
        return self._rotation

    rotation = property(_get_rotation, _set_rotation)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _get_vertex_positions(self):  # pragma: no cover
        """Calculates the vertex positions ignoring any rotation and considering
        that the shape is at the origin (rotation and translation will be
        added automatically later).

        Must return a one-dimensional numpy array (complex dtype) with the
        vertex positions.

        """
        raise NotImplementedError('get_vertex_positions still needs to be implemented in the {0} class'.format(self.__class__.__name__))

    # xxxxx vertex property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def _get_vertices(self):
        vertex_positions = self._get_vertex_positions()
        vertex_positions = self.pos + Shape._rotate(
            vertex_positions, self._rotation)
        return vertex_positions

    vertices = property(_get_vertices)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def is_point_inside_shape(self, point):
        """Tests is a point is inside the shape.

        Arguments:
        - `point`: A single complex number.

        """
        import matplotlib.nxutils as mnx
        # pnpoly returns 1 if point is inside the polygon and 0 otherwise
        return mnx.pnpoly(point.real, point.imag, from_complex_array_to_real_matrix(self.vertices)) == 1

    def get_border_point(self, angle, ratio):
        """Calculates the coordinate of the point that intercepts the
        border of the shape if we go from the origin with a given angle
        (in degrees).

        Arguments:
        - `angle`: Angle (in degrees)
        - `ratio`: Ratio (between 0 and 1)

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
        """Initializes the shape.

        Arguments:
        - `pos`: Coordinate of the shape in the 2D grid (a complex number).
        - `radius`: Radius of the shape (a positive real number).
        - `rotation`: Rotation of the shape in degrees (a positive real
                      number).
        """
        Shape.__init__(self, pos, radius, rotation)

    def _get_height(self):
        return self._radius * np.sqrt(3.) / 2.0

    height = property(_get_height)

    def _get_vertex_positions(self):
        """Calculates the vertex positions ignoring any rotation and considering
        that the shape is at the origin (rotation and translation will be
        added automatically later).

        Must return a one-dimensional numpy array (complex dtype) with the
        vertex positions.

        """
        vertexPositions = np.zeros(6, dtype=complex)
        vertexPositions[0] = complex(-self._radius / 2., -self.height)
        angles = np.linspace(0, 240, 5) * np.pi / 180.

        for k in range(5):
            vertexPositions[k + 1] = vertexPositions[k] + self._radius * np.exp(angles[k] * 1j)
        return vertexPositions


# TODO: Implement a is_point_inside_shape method. The method from the Shape
# class works, but because a Rectangle is very simple it would be more
# efficient to re-implement this method. Note that you will have to write a
# testcase for this as well in the cell_package_test.py file.
class Rectangle(Shape):
    """Rectangle shape class.
    """

    def __init__(self, A, B, rotation=0):
        """Initializes the shape.

        The rectangle is initialized from two coordinates as well as from
        the rotation.

        Arguments:
        - `A`: First coordinate (without rotation).
        - `B`: Second coordinate (without rotation).
        - `rotation`: Rotation of the rectangle in degrees (a positive real
                      number).

        """
        central_pos = (A + B) / 2
        radius = np.abs(B - central_pos)
        Shape.__init__(self, central_pos, radius, rotation)
        self._lower_coord = complex(min(A.real, B.real), min(A.imag, B.imag))
        self._upper_coord = complex(max(A.real, B.real), max(A.imag, B.imag))

    def _get_vertex_positions(self):
        """Calculates the vertex positions ignoring any rotation and considering
        that the shape is at the origin (rotation and translation will be
        added automatically later).

        Must return a one-dimensional numpy array (complex dtype) with the
        vertex positions.

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
        """Initializes the shape.

        A circle is initialized only from a coordinate and a radius.

        Arguments:
        - `pos`: Coordinate of the center of the circle.
        - `radius`: Circle's radius.

        """
        Shape.__init__(self, pos, radius)

    def _get_vertex_positions(self):
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

        Arguments:
        - `angle`: Angle (in degrees)
        - `ratio`: Ratio (between 0 and 1)

        """
        angle_rad = np.pi * angle / 180.
        return self.pos + np.exp(1j * angle_rad) * self.radius * ratio

    def is_point_inside_shape(self, point):
        """Tests is a point is inside the circle

        Arguments:
        - `point`: A single complex number.

        """
        return (np.abs(self.pos - point) < self.radius)

    def plot(self, ax=None):  # pragma: no cover
        """Plot the circle using the Matplotlib library.

        If an axes 'ax' is specified, then the circle is added to that
        axes. Otherwise a new figure and axes are created and the circle is
        plotted to that.

        Arguments:
        - `ax`:  A matplotlib axes

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

    We use complex number to represent coordinates, where the real part is
    the 'x' coordinate and the imaginary part is the 'y' coordinate pf a
    point.

    However, matplotlib methods need the coordinates to be separated. For
    instance, a vector of complex coordinates must be converted to a matrix
    with two columns, where the two columns are the 'x' and 'y' coordinates
    and each row corresponds to a point.

    The method from_complex_array_to_real_matrix does exactly this
    conversion.

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

    point1 = h.get_border_point(90, 0.9)
    ax.plot(point1.real, point1.imag, 'ro')

    point2 = h.get_border_point(10, 0.9)
    ax.plot(point2.real, point2.imag, 'go')

    point3 = h.get_border_point(30, 0.9)
    ax.plot(point3.real, point3.imag, 'bo')

    point4 = h.get_border_point(107, 1)
    ax.plot(point4.real, point4.imag, 'bo')

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
