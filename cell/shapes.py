#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module implementing geometric shapes.

Each shape knows how to plot itself.
"""

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

    def calc_didt(self, other):
        """Calculates the distance to another coordinate.

        Arguments:
        - `other`: A different coordinate (a complex number).
        """
        dist = np.abs(self.pos - other.pos)
        return dist


# TODO: Implement the rest of the Shape class
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
        self.radius = radius
        self.rotation = rotation

        # Properties for the plot representation
        self.fill_face_bool = False
        self.fill_color = 'r'
        self.fill_opacity = 0.1

    def _get_vertex_positions(self):
        """Calculates the vertex positions ignoring any rotation and considering
        that the shape is at the origin (rotation and translation will be
        added automatically later).

        Must return a one-dimensional numpy array (complex dtype) with the
        vertex positions.

        """
        raise NotImplementedError('get_vertex_positions still needs to be implemented in the {0} class'.format(self.__class__.__name__))

    # xxxxx vertex property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def _get_vertexes(self):
        vertex_positions = self._get_vertex_positions()
        num_vertexes = vertex_positions.shape,

        vertex_positions = self.pos + Shape._rotate(vertex_positions, self.rotation)
        return vertex_positions

    vertexes = property(_get_vertexes)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    @staticmethod
    def _rotate(cur_pos, angle):
        """Rotate the complex numbers in the `cur_pos` array by `angle` (in
        degrees)

        """
        angle_rad = angle * np.pi / 180.
        return cur_pos * np.exp(1j * angle_rad)


# TODO: create a doctest for the heght property. The other stuff should go
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
        return self.radius * np.sqrt(3.) / 2.0

    height = property(_get_height)

    def _get_vertex_positions(self):
        """Calculates the vertex positions ignoring any rotation and considering
        that the shape is at the origin (rotation and translation will be
        added automatically later).

        Must return a one-dimensional numpy array (complex dtype) with the
        vertex positions.

        """
        vertexPositions = np.zeros(6, dtype=complex)
        vertexPositions[0] = complex(-self.radius / 2., -self.height)
        angles = np.linspace(0, 240, 5) * np.pi / 180.

        for k in range(5):
            vertexPositions[k + 1] = vertexPositions[k] + \
            self.radius * np.exp(angles[k] * 1j)
        return vertexPositions


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    h = Hexagon(0, 1)
    print h.height
    print h._get_vertex_positions()
