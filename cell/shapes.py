#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module implementing geometric shapes.

Each shape knows how to plot itself.
"""

import numpy as np
from matplotlib import pylab
from matplotlib import patches


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

    def _get_vertex_positions(self):
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
        return mnx.pnpoly(point.real, point.imag, conv_N_complex_array_to_N_by_2_real_matrix(self.vertices)) == 1

    def get_border_point(self, angle, ratio):
        """Calculates the coordinate of the point that intercepts the
        border of the cell if we go from the origin with a given angle
        (in degrees).

        Arguments:
        - `angle`: Angle (in degrees)
        - `ratio`: Ratio (between 0 and 1)

        """
        # Get the vertices (WITH rotation, but WITHOUT translation)
        vertices_no_trans = Shape._rotate(self._get_vertex_positions(),
                                         self._rotation)
        angle_rad = np.pi * angle / 180.

        # Which point we get if we walk a distance of cell radius in the
        # desired angle direction?
        point = self._radius * np.exp(angle_rad * 1j)

        # Calculates the distance of this point to all vertices and finds
        # the closest vertices
        dists = np.abs(vertices_no_trans - point)
        # Get the two closest vertices from point
        closest_vertices = vertices_no_trans[np.argsort(dists)[:2]]

        # The equation of a straight line is given by "y = ax + b". We have
        # two points in this line (the two closest vertices) and we can use
        # them to find 'a' and 'b'.
        a = closest_vertices[0] - closest_vertices[1]
        a = a.imag / a.real
        b = closest_vertices[1].imag - a * closest_vertices[1].real

        # Now lets find the equation of the straight line from the origin
        # of the shape with the desired angle.
        a2 = np.tan(angle_rad)
        b2 = 0
        # The equation of this line is then given by "y2 = a2 x2 + b2"

        # The point in the border of the shape is the point where these two
        # equations meet. That is, "a*x+b == a2*x+b2"
        x = (b2 - b) / (a - a2)
        y = a * x + b

        point = complex(x, y) * ratio + self.pos
        return point

    def plot(self, ax=None):
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
                conv_N_complex_array_to_N_by_2_real_matrix(self.vertices),
                True,
                facecolor=self.fill_color,
                edgecolor='none',  # No edges
                alpha=self.fill_opacity)
            ax.add_patch(polygon_face)

        polygon_edges = patches.Polygon(
            conv_N_complex_array_to_N_by_2_real_matrix(self.vertices),
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


# TODO: create a doctest for the height property. The other stuff should go
# in the unittests.
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


class Rectangle(Shape):
    """Rectangle shape class.
    """

    def __init__(self, A, B, rotation=0):
        """Initializes the shape.

        The rectangle is initialized from two coordinates as well as from
        the rotation.

        Arguments:
        - `A`: First coordinate.
        - `B`: Second coordinate.
        - `rotation`: Rotation of the rectangle in degrees (a positive real
                      number).

        """
        central_pos = (A + B) / 2
        radius = np.abs(B - central_pos)
        Shape.__init__(self, central_pos, radius, rotation)
        self._lower_coord = A
        self._upper_coord = B

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
        # 180 points from 0 to 2pi
        angles = np.linspace(0, 2 * np.pi, 180)
        vertex_positions = self._radius * np.exp(1j * angles)
        return vertex_positions

    def plot(self, ax=None):
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
# TODO: Create a better name and rename this method (use ropemacs for that)
def conv_N_complex_array_to_N_by_2_real_matrix(a):
    """Convert an array of complex number to a matrix of real numbers. The
    first coloumn of the matrix is the real part while the second column is
    the imaginary part of the original array.

    """
    num_elem = np.size(a)
    a.dtype = a.real.dtype
    a = np.reshape(a, (num_elem, 2))
    return a


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    ax = pylab.axes()
    h = Hexagon(2+3j, 2)
    point = h.get_border_point(90, 1)
    print "Border Point is: {0}".format(point)
    h.plot(ax)
    ax.plot(point.real, point.imag, 'r*')
    pylab.show()

if __name__ == '__main__1':
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

    c = Circle(0.2+0.4j, 1)
    c.fill_face_bool = True
    c.fill_color = 'g'

    r.plot(ax)
    h.plot(ax)
    c.plot(ax)


    ax.plot()
    plt.axis('equal')
    plt.show()
