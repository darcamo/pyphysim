#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101
"""
Tests for the modules in the cell package.

Each module has several doctests that we run in addition to the unittests
defined here.
"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os

try:
    parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    sys.path.append(parent_dir)
except NameError:  # pragma: no cover
    sys.path.append('../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import math
import unittest
import doctest
import numpy as np

from pyphysim.cell import shapes, cell


class ConcreteShape(shapes.Shape):
    """
    Concrete version of the Shape class.

    Because the shapes.Shape class is an abstract class, we define a
    concrete version here only for testing purposes.
    """
    def __init__(self, pos, radius, rotation=0):
        """Initialize the shape.
        """
        shapes.Shape.__init__(self, pos, radius, rotation)

    def _get_vertex_positions(self):
        """This method will not be called in our tests.
        """
        pass  # pragma: nocover


# UPDATE THIS CLASS if another module is added to the comm package
# noinspection PyMethodMayBeStatic
class CellDoctestsTestCase(unittest.TestCase):
    """
    Test case that run all the doctests in the modules of the cell package.
    """
    def test_shapes(self):
        """
        Run doctests in the shapes module.
        """
        doctest.testmod(shapes)

    def test_cell(self):
        """
        Run doctests in the cell module.
        """
        doctest.testmod(cell)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SHAPES module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class CoordinateTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.C1 = shapes.Coordinate(0 + 3j)
        self.C2 = shapes.Coordinate(2 - 4j)
        self.C3 = shapes.Coordinate(5 - 0j)

    def test_calc_dist(self):
        # Sanity check
        self.assertEqual(self.C1.calc_dist(self.C2),
                         self.C2.calc_dist(self.C1))

        self.assertAlmostEqual(np.sqrt((2**2) + (7**2)),
                               self.C1.calc_dist(self.C2))
        self.assertAlmostEqual(np.sqrt((5**2) + (3**2)),
                               self.C1.calc_dist(self.C3))
        self.assertAlmostEqual(np.sqrt((3**2) + (4**2)),
                               self.C2.calc_dist(self.C3))

    def test_move_by_relative_coordinate(self):
        self.assertEqual(self.C1.pos, 0 + 3j)
        self.C1.move_by_relative_coordinate(2 - 1.5j)
        self.assertEqual(self.C1.pos, 2 + 1.5j)
        self.C1.move_by_relative_coordinate(-1 + 4j)
        self.assertEqual(self.C1.pos, 1 + 5.5j)

    def test_move_by_relative_polar_coordinate(self):
        self.C1.move_by_relative_polar_coordinate(1, np.pi / 2)
        self.assertAlmostEqual(self.C1.pos, 0 + 4j)

        self.C1.move_by_relative_polar_coordinate(1, np.pi)
        self.assertAlmostEqual(self.C1.pos, -1 + 4j)

        self.C1.move_by_relative_polar_coordinate(3, np.pi / 3)
        self.assertAlmostEqual(self.C1.pos, 0.5 + 6.59807621135j)


class ShapeTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        #         ConcreteShape(pos, radius, rotation)
        self.S1 = ConcreteShape(0 + 0j, 1.5, 0)
        self.S2 = ConcreteShape(2 + 3j, 2, 0)
        self.S3 = ConcreteShape(3 + 5j, 1.2, 30)

    def test_radius_property(self):
        self.assertAlmostEqual(self.S1.radius, 1.5)
        self.assertAlmostEqual(self.S2.radius, 2.0)
        self.assertAlmostEqual(self.S3.radius, 1.2)
        self.S1.radius = 3.3
        self.assertAlmostEqual(self.S1.radius, 3.3)

    def test_rotation_property(self):
        self.assertAlmostEqual(self.S1.rotation, 0)
        self.assertAlmostEqual(self.S3.rotation, 30)
        self.S1.rotation = 25
        self.assertAlmostEqual(self.S1.rotation, 25)

    def test_is_point_inside_shape(self):
        # This method uses the _get_vertex_positions method. Therefore, it
        # can only be tested in subclasses of the Shape class.
        #
        # This method was tested in the Hexagon unittests.
        pass

    def test_get_border_point(self):
        # This method uses the _get_vertex_positions method. Therefore, it
        # can only be tested in subclasses of the Shape class.
        #
        # This method was tested in the Hexagon unittests.
        pass

    def test_rotate(self):
        """Test the static method Shape._rotate
        """
        cur_pos = 1 - 2j
        self.assertAlmostEqual(ConcreteShape.calc_rotated_pos(cur_pos, 90),
                               2 + 1j)
        self.assertAlmostEqual(ConcreteShape.calc_rotated_pos(cur_pos, 180),
                               -1 + 2j)


class HexagonTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.H1 = shapes.Hexagon(0 + 0j, 1.5, 0)
        self.H2 = shapes.Hexagon(2 + 3j, 2, 0)
        self.H3 = shapes.Hexagon(3 + 5j, 1.5, 30)

    def test_height_property(self):
        # The height of an Hexagon is given by sqrt(3) * side_length, where
        # the side of the Hexagon is the radius property of the Shape.
        self.assertAlmostEqual(self.H1.height,
                               np.sqrt(3.0) * self.H1.radius / 2.0)
        self.assertAlmostEqual(self.H2.height,
                               np.sqrt(3.0) * self.H2.radius / 2.0)
        self.assertAlmostEqual(self.H3.height,
                               np.sqrt(3.0) * self.H3.radius / 2.0)

    def test_get_vertex_positions(self):
        # Even though H1 and H2 have different position and rotation, they
        # have the same radius and since _get_vertex_positions does not
        # account translation and rotation then the _get_vertex_positions
        # method should return the same value for both of them.
        np.testing.assert_array_equal(self.H1._get_vertex_positions(),
                                      self.H3._get_vertex_positions())

        # noinspection PyTypeChecker
        aux = 1j * np.pi * np.linspace(0, 300, 6) / 180.
        ":type: np.ndarray"

        expected_vertexes = 1.5 * np.exp(aux)
        ":type: np.ndarray"

        # Lets sort the expected vertexes (regarding the angle of the
        # vertexes)
        expected_vertexes \
            = expected_vertexes[np.argsort(np.angle(expected_vertexes))]

        # The way the obtained_vertexes are calculated the vertexes are
        # already in ascending order regarding their angle.
        obtained_vertexes = self.H1._get_vertex_positions()

        np.testing.assert_array_almost_equal(expected_vertexes,
                                             obtained_vertexes)

        expected_vertexes2 = (2. / 1.5) * expected_vertexes
        ":type: np.ndarray"

        obtained_vertexes2 = self.H2._get_vertex_positions()
        np.testing.assert_array_almost_equal(expected_vertexes2,
                                             obtained_vertexes2)

    def test_get_vertices(self):
        # H1 does not have any translation or rotation. Therefore,
        # _get_vertex_positions should return the same values as the
        # vertices property
        np.testing.assert_array_almost_equal(self.H1._get_vertex_positions(),
                                             self.H1.vertices)

        # H2 has translation.
        np.testing.assert_array_almost_equal(
            self.H2._get_vertex_positions() + complex(2, 3), self.H2.vertices)

        # H3 has rotation and translation.
        np.testing.assert_array_almost_equal(
            shapes.Shape.calc_rotated_pos(self.H3._get_vertex_positions(), 30)
            + 3 + 5j, self.H3.vertices)

    def test_is_point_inside_shape(self, ):
        # If the point is exactly in the shape's border, such as the
        # shape's vertexes, then is_point_inside_shape could return either
        # true or false.
        for v in self.H1.vertices:
            # Because the hexagons center is at the origin, we know that if
            # we multiply a vertex by a number lower then one it is
            # guaranteed be inside the shape and vice-verse.
            self.assertTrue(self.H1.is_point_inside_shape(0.9999999 * v))
            self.assertFalse(self.H1.is_point_inside_shape(1.0000001 * v))

            # This test is very incomplete. If any bugs are found in
            # is_point_inside_shape add tests for them here

    def test_get_border_point(self):
        # Test for an angle of 0 degrees
        point_0_degrees = self.H3.get_border_point(0., 1.)
        point_0_degrees_expected = self.H3.pos + self.H3.height
        self.assertAlmostEqual(point_0_degrees, point_0_degrees_expected)

        # Test for an angle of 30 degrees (that is one of the vertexes)
        point_30_degrees = self.H3.get_border_point(30., 1.)
        point_30_degrees_expected \
            = self.H3.pos + self.H3.radius * np.exp(1j * np.pi / 6.)
        self.assertAlmostEqual(point_30_degrees, point_30_degrees_expected)

        # Test for an angle of 30 degrees, but now with a ratio of 0.8
        point_30_degrees_ratio08 = self.H3.get_border_point(30., 0.8)
        point_30_degrees_expected_ratio08 \
            = self.H3.pos + 0.8 * self.H3.radius * np.exp(1j * np.pi / 6.)
        self.assertAlmostEqual(point_30_degrees_ratio08,
                               point_30_degrees_expected_ratio08)

        # Test for an angle of 180 degrees
        point_180_degrees = self.H3.get_border_point(180., 1.)
        point_180_degrees_expected = self.H3.pos - self.H3.height
        self.assertAlmostEqual(point_180_degrees, point_180_degrees_expected)

        # Test for an angle of 180 degrees but with a ratio of 0.5
        point_180_degrees_ratio05 = self.H3.get_border_point(180, 0.5)
        point_180_degrees_expected_ratio05 \
            = self.H3.pos - (self.H3.height / 2.0)
        self.assertAlmostEqual(point_180_degrees_ratio05,
                               point_180_degrees_expected_ratio05)


class RectangleTestCase(unittest.TestCase):
    def test_get_vertex_positions(self):
        A1 = 0 + 0j
        B1 = 1 + 1j
        R1 = shapes.Rectangle(A1, B1)
        # With these coordinates, the central position of R1 is 0.5 + 0.5j
        self.assertAlmostEqual(R1.pos, 0.5 + 0.5j)

        # The rectangle vertexes are then -0.5-0.5j, 0.5-0.5j, 0.5+0.5j and
        # -0.5+0.5j
        np.testing.assert_array_almost_equal(
            R1._get_vertex_positions(),
            np.array([-0.5 - 0.5j, 0.5 - 0.5j, 0.5 + 0.5j, -0.5 + 0.5j]))

        # Note that though A2 and B2 are different from A1 and B1, they are
        # all coordinates of the same rectangle and therefore R2 should
        # have the same position and vertexes as R1.
        A2 = 1 + 0j
        B2 = 0 + 1j
        R2 = shapes.Rectangle(A2, B2)
        self.assertAlmostEqual(R1.pos, R2.pos)
        np.testing.assert_almost_equal(R1._get_vertex_positions(),
                                       R2._get_vertex_positions())

        # Now lets create a rectangle with rotation and translates to a
        # different position. The _get_vertex_positions method should not
        # be affected by this
        trans_step = 1.4 - 3j
        R3 = shapes.Rectangle(A1 + trans_step, B1 + trans_step, 30)
        np.testing.assert_almost_equal(R1._get_vertex_positions(),
                                       R3._get_vertex_positions())

    def test_is_point_inside_shape(self):
        A1 = -1.2 + -0.000001j
        B1 = 1 + 3.4j
        R1 = shapes.Rectangle(A1, B1)

        self.assertTrue(R1.is_point_inside_shape(0.))
        self.assertTrue(R1.is_point_inside_shape(0.01j))
        self.assertTrue(R1.is_point_inside_shape(0.1 + 0.1j))
        self.assertTrue(R1.is_point_inside_shape(-0.63 + 2.4j))
        self.assertTrue(R1.is_point_inside_shape(-1.2 + 2.4j))
        self.assertTrue(R1.is_point_inside_shape(-0.5 + 3.4j))
        self.assertFalse(R1.is_point_inside_shape(-1.3 + 2.4j))
        self.assertFalse(R1.is_point_inside_shape(1.3 + .4j))
        self.assertFalse(R1.is_point_inside_shape(-0.5 + 3.41j))
        self.assertFalse(R1.is_point_inside_shape(-0.1 + 5j))
        self.assertFalse(R1.is_point_inside_shape(-14 + 1j))
        self.assertFalse(R1.is_point_inside_shape(-10 + 5j))
        self.assertFalse(R1.is_point_inside_shape(0 + -0.1j))


class CircleTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.C1 = shapes.Circle(3 - 5j, 2)

    def test_get_border_point(self):
        C1 = self.C1
        point = C1.get_border_point(90, 1)
        expected_point = C1.pos + 1j * C1.radius
        self.assertAlmostEqual(point, expected_point)

        # from matplotlib import pylab
        # ax = pylab.axes()
        # C1.plot(ax)
        # for angle in np.arange(0, 359, 30):
        #     border_point = C1.get_border_point(angle, 1)
        #     border_point2  = C1.get_border_point(angle, 0.5)
        #     ax.plot(border_point.real, border_point.imag, 'go')
        #     ax.plot(border_point2.real, border_point2.imag, 'bo')
        # pylab.show()

    def test_get_vertex_positions(self):
        num_vertexes = 12
        # The angles need to be in degrees for the get_border_point method
        angles = np.linspace(0, (num_vertexes - 1.) / num_vertexes * 360,
                             num_vertexes)
        ":type: np.ndarray"

        # pylint: disable=E1103
        expected_vertexes = np.array(
            list(map(self.C1.get_border_point, angles, np.ones(
                angles.shape)))) - self.C1.pos
        vertexes = self.C1._get_vertex_positions()
        np.testing.assert_array_almost_equal(expected_vertexes, vertexes)

    def test_is_point_inside_shape(self):
        # Note that the Shape class version of the is_point_inside_shape
        # method will fail in these tests. That is because it uses the
        # shape's vertexes to determine if a point is inside of the shape
        # of not, which is not good for a circle. The circle class version
        # of the is_point_inside_shape compares the distance from the point
        # to the circle center with the circle radius to determine of the
        # point is inside or outside the circle.

        # This point is inside the circle
        point = self.C1.get_border_point(89, 0.99999999)
        self.assertTrue(self.C1.is_point_inside_shape(point))

        # This point is NOT inside the circle
        point2 = self.C1.get_border_point(89, 1.00000001)
        self.assertFalse(self.C1.is_point_inside_shape(point2))


# noinspection PyMethodMayBeStatic
class ShapesModuleMethodsTestCase(unittest.TestCase):
    def test_from_complex_array_to_real_matrix(self, ):
        A = np.random.random_sample(10) + 1j * np.random.random_sample(10)
        B = A.copy()
        B.shape = (B.size, 1)

        expected_value = np.hstack([B.real, B.imag])
        np.testing.assert_array_almost_equal(
            expected_value, shapes.from_complex_array_to_real_matrix(A))


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx CELL module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class NodeTestCase(unittest.TestCase):
    def test_attributes(self):
        n = cell.Node(1 + 3j, plot_marker='v', marker_color='g')
        self.assertEqual(n.pos, 1 + 3j)
        self.assertEqual(n.plot_marker, 'v')
        self.assertEqual(n.marker_color, 'g')


class CellTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.C1 = cell.Cell(pos=2 - 3j, radius=2.5, cell_id=1, rotation=30)
        self.C2 = cell.Cell(pos=0 + 2j, radius=2, cell_id=2, rotation=20)
        self.C3 = cell.Cell(pos=-3 + 5j, radius=1.5, cell_id=3, rotation=70)

    def test_repr(self):
        self.assertEqual(repr(self.C1),
                         'Cell(pos=(2-3j),radius=2.5,cell_id=1,rotation=30)')

    def test_add_user(self):
        # The cell has no users yet
        self.assertEqual(self.C1.num_users, 0)

        # User with the same position as the cell center
        user1 = cell.Node(self.C1.pos, marker_color='b')
        self.C1.add_user(user1, relative_pos_bool=False)
        self.assertEqual(user1.cell_id, self.C1.id)
        self.assertAlmostEqual(user1.relative_pos, user1.pos - self.C1.pos)

        # User (relative to cell center) located at the top of the cell
        user2 = cell.Node(0 + 0.99999j, marker_color='r')
        self.C1.add_user(user2)
        self.assertEqual(user2.cell_id, self.C1.id)
        self.assertAlmostEqual(user2.relative_pos, user2.pos - self.C1.pos)

        # User (relative to cell center) located at some point in the north
        # east part of the cell
        user3 = cell.Node(0.4 + 0.7j, marker_color='g')
        self.C1.add_user(user3)
        self.assertEqual(user3.cell_id, self.C1.id)
        self.assertAlmostEqual(user3.relative_pos, user3.pos - self.C1.pos)

        # We have successfully added 3 users to the cell
        self.assertEqual(self.C1.num_users, 3)

        # This user will fall outside the cell and add_user should raise an
        # exception
        user4 = cell.Node(0.4 + 0.8j)
        self.assertRaises(
            ValueError,
            self.C1.add_user,
            # Args to self.C1.add_user
            user4)

        # This user will also fall outside the cell and add_user should
        # raise an exception
        user5 = cell.Node(0 + 0j)
        self.assertRaises(
            ValueError,
            self.C1.add_user,
            # Args to self.C1.add_user
            user5,
            relative_pos_bool=False)

        # Test if we try to add a 'user' which is not an instance of the
        # NodeClass
        self.assertRaises(TypeError, self.C1.add_user, 0 + 3j)

        # The cell still has only 3 users
        self.assertEqual(self.C1.num_users, 3)

        # Lets get a list with the users added to the cell
        users = self.C1.users
        self.assertEqual(users[0], user1)
        self.assertEqual(users[1], user2)
        self.assertEqual(users[2], user3)

        # Now lets delete all users
        self.C1.delete_all_users()
        self.assertEqual(self.C1.num_users, 0)

    def test_add_border_user(self):
        # xxxxx Test adding a single user
        angles = 30
        ratio = 0.8
        # The get_border_point comes from the shape class and it should be
        # already tested.
        expected_pos = self.C1.get_border_point(angles, ratio)
        self.C1.add_border_user(angles, ratio, user_color='g')
        self.assertAlmostEqual(self.C1.users[0].pos, expected_pos)

        # Test adding a single user without specifying the ration, which
        # should default to 1.
        self.C1.delete_all_users()
        self.C1.add_border_user(angles)
        expected_pos = self.C1.get_border_point(angles, 1)
        self.assertAlmostEqual(self.C1.users[0].pos, expected_pos)

        # Test adding a single user an invalid ratio. This should raise an
        # exception.
        self.C1.delete_all_users()
        self.assertRaises(ValueError, self.C1.add_border_user, angles, 1.1)
        self.assertRaises(ValueError, self.C1.add_border_user, angles, -0.4)

        # xxxxx Test adding multiple users with the same ratio
        angles2 = [30, 45, 60, 90, 120]
        ratio2 = 0.75
        self.C2.add_border_user(angles2, ratio2)
        self.assertEqual(self.C2.num_users, 5)
        for index in range(5):
            expected_pos2 = self.C2.get_border_point(angles2[index], ratio2)
            self.assertAlmostEqual(self.C2.users[index].pos, expected_pos2)

        # xxxxx Test adding multiple users with the different ratios and
        # user colors
        angles3 = [30, 45, 60, 90, 120]
        ratios3 = [0.9, 0.4, 0.6, 0.85, 0.3]
        colors = ['g', 'b', 'k', 'r', 'y']
        self.C3.add_border_user(angles3, ratios3, colors)
        self.assertEqual(self.C3.num_users, 5)
        for index in range(5):
            absolute_pos = self.C3.get_border_point(angles3[index],
                                                    ratios3[index])
            self.assertAlmostEqual(self.C3.users[index].pos, absolute_pos)
            self.assertEqual(self.C3.users[index].marker_color, colors[index])

    def test_add_random_user(self):
        self.C1.add_random_user(user_color='y')
        self.assertEqual(self.C1.num_users, 1)
        self.assertEqual(self.C1.users[0].marker_color, 'y')

        min_dist_ratio = 0.6
        user_color = 'g'
        self.C1.add_random_users(10, user_color, min_dist_ratio)
        self.assertEqual(self.C1.num_users, 11)
        for index in range(1, 11):
            self.assertEqual(self.C1.users[index].marker_color, 'g')
            min_dist = self.C1.radius * min_dist_ratio
            self.assertTrue(self.C1.calc_dist(self.C1.users[index]) > min_dist)

    def test_set_pos(self):
        # When the position of a cell is changed, the position of any user
        # in the cell must be updated.
        self.C1.add_border_user(0, 0.1)
        self.C1.add_border_user(90, 0.4)
        self.C1.add_border_user(270, 0.3)
        users = self.C1.users

        self.assertAlmostEqual(self.C1.pos, 2 - 3j)
        self.assertAlmostEqual(users[0].pos, (2.21650635094611 - 3j))
        self.assertAlmostEqual(users[1].pos, (2 - 2j))
        self.assertAlmostEqual(users[2].pos, (2 - 3.75j))

        # Change the position of the cell ...
        self.C1.pos = 0 - 1j
        # ... and check if the position of the users changed
        self.assertAlmostEqual(users[0].pos, (0.21650635094611 - 1j))
        self.assertAlmostEqual(users[1].pos, (0 - 0j))
        self.assertAlmostEqual(users[2].pos, (0 - 1.75j))


class Cell3SecTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.C1 = cell.Cell3Sec(pos=2 - 3j, radius=2.5, cell_id=1, rotation=0)
        self.C2 = cell.Cell3Sec(pos=-3.5 + 3j,
                                radius=2.5,
                                cell_id=1,
                                rotation=60)
        self.C2.fill_color = 'r'
        self.C2.fill_face_bool = True

        self.C3 = cell.Cell(pos=2 - 3j, radius=1, cell_id=1, rotation=0)
        self.C3.fill_color = 'b'
        self.C3.fill_face_bool = True

    def test_calc_sectors_positions(self):
        expected_sec_positions = np.array(
            [0.75 - 3.72168784j, 3.25 - 3.72168784j, 2.00 - 1.55662433j])

        pos = self.C1._calc_sectors_positions()
        np.testing.assert_array_almost_equal(pos, expected_sec_positions)

        self.C1.rotation = 14
        self.C1.pos = 4 - 1j
        self.C1.radius = 3.2
        pos = self.C1._calc_sectors_positions()
        expected_sec_positions = np.array([
            2.67100471 - 2.28339583j, 5.77595104 - 1.50924577j,
            3.55304425 + 0.7926416j
        ])
        np.testing.assert_array_almost_equal(pos, expected_sec_positions)

    def test_set_pos(self):
        # Add a few users in the cell
        self.C1.add_random_users(5)

        # Whenever the pos property of the Cell3Sec object changes, the
        # position of each individual sector should change
        expected_sec1_pos, expected_sec2_pos, expected_sec3_pos \
            = self.C1._calc_sectors_positions()
        self.assertAlmostEqual(self.C1._sec1.pos, expected_sec1_pos)
        self.assertAlmostEqual(self.C1._sec2.pos, expected_sec2_pos)
        self.assertAlmostEqual(self.C1._sec3.pos, expected_sec3_pos)

        self.C1.pos = 10 - 1j
        expected_sec1_pos, expected_sec2_pos, expected_sec3_pos \
            = self.C1._calc_sectors_positions()

        self.assertAlmostEqual(self.C1._sec1.pos, expected_sec1_pos)
        self.assertAlmostEqual(self.C1._sec2.pos, expected_sec2_pos)
        self.assertAlmostEqual(self.C1._sec3.pos, expected_sec3_pos)

        # self.C1.plot()

    def test_secradius(self):
        expected_secradius = np.sqrt(3) * self.C1.radius / 3.0
        self.assertAlmostEqual(self.C1.secradius, expected_secradius)

        self.C1.radius = 10
        expected_secradius = np.sqrt(3) * self.C1.radius / 3.0
        self.assertAlmostEqual(self.C1.secradius, expected_secradius)

    def test_set_radius(self):
        # Whenever the radius property of the Cell3Sec object changes, the
        # position and radius of each individual sector should change
        expected_sec1_pos, expected_sec2_pos, expected_sec3_pos \
            = self.C1._calc_sectors_positions()

        self.assertAlmostEqual(expected_sec1_pos, self.C1._sec1.pos)
        self.assertAlmostEqual(expected_sec2_pos, self.C1._sec2.pos)
        self.assertAlmostEqual(expected_sec3_pos, self.C1._sec3.pos)

        # Lets change the position, rotation and radius of self.C1 ...
        self.C1.rotation = 14
        self.C1.pos = 4 - 1j
        self.C1.radius = 3.2
        # ... and calculate the new sec positions
        expected_sec1_pos, expected_sec2_pos, expected_sec3_pos \
            = self.C1._calc_sectors_positions()

        # Now lets test if the pos property of each sector really changed
        self.assertAlmostEqual(expected_sec1_pos, self.C1._sec1.pos)
        self.assertAlmostEqual(expected_sec2_pos, self.C1._sec2.pos)
        self.assertAlmostEqual(expected_sec3_pos, self.C1._sec3.pos)

    def test_set_rotation(self):
        # Whenever the rotation property of the Cell3Sec object changes, the
        # position and rotation of each individual sector should change
        expected_sec1_pos, expected_sec2_pos, expected_sec3_pos \
            = self.C1._calc_sectors_positions()

        self.assertAlmostEqual(expected_sec1_pos, self.C1._sec1.pos)
        self.assertAlmostEqual(expected_sec2_pos, self.C1._sec2.pos)
        self.assertAlmostEqual(expected_sec3_pos, self.C1._sec3.pos)

        self.assertAlmostEqual(self.C1._sec1.rotation, -30)
        self.assertAlmostEqual(self.C1._sec2.rotation, -30)
        self.assertAlmostEqual(self.C1._sec3.rotation, -30)

        # Lets change the rotation of self.C1 ...
        self.C1.rotation = 23
        # ... and calculate the new sec positions
        expected_sec1_pos, expected_sec2_pos, expected_sec3_pos \
            = self.C1._calc_sectors_positions()

        self.assertAlmostEqual(expected_sec1_pos, self.C1._sec1.pos)
        self.assertAlmostEqual(expected_sec2_pos, self.C1._sec2.pos)
        self.assertAlmostEqual(expected_sec3_pos, self.C1._sec3.pos)

        expected_sec_rotataion = self.C1.rotation - 30
        self.assertAlmostEqual(self.C1._sec1.rotation, expected_sec_rotataion)
        self.assertAlmostEqual(self.C1._sec2.rotation, expected_sec_rotataion)
        self.assertAlmostEqual(self.C1._sec3.rotation, expected_sec_rotataion)

    def test_get_vertex_positions(self):
        # The _get_vertex_positions method return the vertexes of the cell
        # ignoring the cell position as if the cell was located at the
        # origin.
        vertexes_no_translation = self.C1._get_vertex_positions()

        self.assertEqual(len(vertexes_no_translation), 12)

        expected_vertexes_no_translation = [
            -1.25000000e+00 - 2.16506351e+00j,
            0.00000000e+00 - 1.44337567e+00j, 1.25000000e+00 - 2.16506351e+00j,
            2.50000000e+00 - 1.44337567e+00j, 2.50000000e+00 + 0.00000000e+00j,
            1.25000000e+00 + 7.21687836e-01j, 1.25000000e+00 + 2.16506351e+00j,
            5.55111512e-16 + 2.88675135e+00j,
            -1.25000000e+00 + 2.16506351e+00j,
            -1.25000000e+00 + 7.21687836e-01j,
            -2.50000000e+00 + 7.77156117e-16j,
            -2.50000000e+00 - 1.44337567e+00j
        ]

        np.testing.assert_array_almost_equal(expected_vertexes_no_translation,
                                             vertexes_no_translation)

        vertexes_with_translation = self.C1.vertices

        expected_vertexes_with_translation = np.array(
            expected_vertexes_no_translation) + self.C1.pos

        np.testing.assert_array_almost_equal(
            expected_vertexes_with_translation, vertexes_with_translation)

    def test_add_random_users_in_sector(self):
        self.C1.add_random_user_in_sector(1)
        self.assertEqual(self.C1.num_users, 1)

        self.C1.add_random_user_in_sector(1)
        self.C1.add_random_user_in_sector(2)

        self.assertEqual(self.C1.num_users, 3)

        self.C1.add_random_user_in_sector(2)
        self.C1.add_random_user_in_sector(2)
        self.C1.add_random_user_in_sector(3)
        self.C1.add_random_users_in_sector(4, 3)  # Add 4 users in sector 3

        self.assertEqual(self.C1.num_users, 10)

        for i in range(self.C1.num_users):
            self.assertTrue(self.C1.is_point_inside_shape(
                self.C1.users[i].pos))

        # If we change the position of the cell, the position of the users
        # already in the cell should be updated.
        self.C1.pos = 0
        for i in range(self.C1.num_users):
            self.assertTrue(self.C1.is_point_inside_shape(
                self.C1.users[i].pos))

        # Sector 5 does not exist and a RuntimeError exception should be
        # raised
        with self.assertRaises(RuntimeError):
            self.C1.add_random_users_in_sector(2, 5)


# TODO: finish implementation
class CellSquareTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.C1 = cell.CellSquare(pos=2 - 3j,
                                  side_length=2.5,
                                  cell_id=1,
                                  rotation=0)
        self.C2 = cell.CellSquare(pos=-3.5 + 3j,
                                  side_length=2.5,
                                  cell_id=1,
                                  rotation=60)
        self.C2.fill_color = 'r'
        self.C2.fill_face_bool = True

    # TODO: Add test methods for the CellSquare class
    def test_some_method(self):
        # self.C1.plot()

        pass
        # cluster = cell.Cluster(
        #     cell_radius=1.5, num_cells=9, cell_type='square')
        # cluster.plot()


class CellWrapTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.C = cell.Cell(1 - 1j, 1.0, cell_id=1, rotation=10)
        self.C2 = cell.Cell(-2 + 3j, 2.0, rotation=60)
        self.W = cell.CellWrap(-1 + 0j, self.C)
        self.W2 = cell.CellWrap(0 + 0j, self.C2)

    def test_init(self):
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            cell.CellWrap(pos=0, wrapped_cell='should_be_cell_object')

    def test_pos(self):
        self.assertAlmostEqual(self.W.pos, -1 + 0j)

        self.W.pos = 4 - 2j
        self.assertAlmostEqual(self.W.pos, 4 - 2j)

    def test_radius(self):
        self.assertAlmostEqual(self.W.radius, 1.0)
        self.assertAlmostEqual(self.W2.radius, 2.0)

        # The radius property in a CellWrap object should not be changed
        with self.assertRaises(AttributeError):
            # noinspection PyPropertyAccess
            self.W.radius = 2.0

    def test_rotation(self):
        self.assertAlmostEqual(self.W.rotation, 10.0)
        self.assertAlmostEqual(self.W2.rotation, 60.0)

        # The rotation property in a CellWrap object should not be changed
        with self.assertRaises(AttributeError):
            # noinspection PyPropertyAccess
            self.W.rotation = 30
        self.assertAlmostEqual(self.W.rotation, 10.0)

    def test_get_users(self):
        self.C.add_random_users(5)
        self.assertEqual(self.C.num_users, 5)
        orig_users = self.C.users

        self.assertFalse(self.W.include_users_bool)
        self.assertEqual(self.W.num_users, 0)

        users = self.W.users
        self.assertEqual(users, [])

        self.W.include_users_bool = True
        self.assertEqual(self.W.num_users, 5)

        users = self.W.users

        pos_diff = self.W.pos - self.C.pos
        for u, ou in zip(users, orig_users):
            self.assertAlmostEqual(u.pos, ou.pos + pos_diff)

        # Add two more users in the original cell
        self.C.add_random_users(2)
        self.assertEqual(self.W.num_users, 7)

        for u, ou in zip(users, orig_users):
            self.assertAlmostEqual(u.pos, ou.pos + pos_diff)

    def test_repr(self):
        self.assertEqual(repr(self.W), 'CellWrap(pos=(-1+0j),cell_id=Wrap 1)')
        self.assertEqual(repr(self.W2), 'CellWrap(pos=0j,cell_id=None)')

    def test_get_vertex_positions(self):
        A = self.C._get_vertex_positions()
        B = self.W._get_vertex_positions()
        np.testing.assert_array_almost_equal(A, B)


# TODO: Extend the tests to consider the case of the Cell3Sec class.
# noinspection PyMethodMayBeStatic
class ClusterTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.C1 = cell.Cluster(pos=1 - 2j, cell_radius=1.0, num_cells=3)
        self.C2 = cell.Cluster(pos=-2 + 3j,
                               cell_radius=1.0,
                               num_cells=7,
                               rotation=20)
        self.C3 = cell.Cluster(pos=0 - 1.1j, cell_radius=1.0, num_cells=19)
        self.C4 = cell.Cluster(pos=0 - 1.1j,
                               cell_radius=1.0,
                               num_cells=19,
                               cell_type='3sec')

        # Add two users to the first cell of Cluster1
        self.C1._cells[0].add_random_user()
        self.C1._cells[0].add_random_user()

        # Add three users to the second cell of Cluster1
        self.C1._cells[1].add_random_user()
        self.C1._cells[1].add_random_user()
        self.C1._cells[1].add_random_user()

        # Add five users to the third cell of Cluster1
        self.C1._cells[2].add_random_user()
        self.C1._cells[2].add_random_user()
        self.C1._cells[2].add_random_user()
        self.C1._cells[2].add_random_user()
        self.C1._cells[2].add_random_user()

    def test_init(self):
        self.assertAlmostEqual(self.C1.pos, 1 - 2j)
        self.assertAlmostEqual(self.C2.pos, -2 + 3j)
        self.assertAlmostEqual(self.C3.pos, -1.1j)

        self.assertEqual(self.C1.num_users, 10)
        self.assertEqual(self.C2.num_users, 0)
        self.assertEqual(self.C3.num_users, 0)

        self.assertEqual(type(self.C1._cells[0]), cell.Cell)
        self.assertEqual(type(self.C2._cells[0]), cell.Cell)
        self.assertEqual(type(self.C3._cells[0]), cell.Cell)
        self.assertEqual(type(self.C4._cells[0]), cell.Cell3Sec)

        with self.assertRaises(RuntimeError):
            # cell_type can only be 'simple' or '3sec'
            cell.Cluster(1.0, num_cells=19, cell_type='invalid_type')

    def test_repr(self):
        self.assertEqual(repr(self.C1),
                         ("Cluster(cell_radius=1.0,num_cells=3,pos=(1-2j),"
                          "cluster_id=None,cell_type='simple',rotation=0.0)"))
        self.assertEqual(repr(self.C2),
                         ("Cluster(cell_radius=1.0,num_cells=7,pos=(-2+3j),"
                          "cluster_id=None,cell_type='simple',rotation=20)"))
        self.assertEqual(repr(self.C3),
                         ("Cluster(cell_radius=1.0,num_cells=19,pos=-1.1j,"
                          "cluster_id=None,cell_type='simple',rotation=0.0)"))
        self.assertEqual(repr(self.C4),
                         ("Cluster(cell_radius=1.0,num_cells=19,pos=-1.1j,"
                          "cluster_id=None,cell_type='3sec',rotation=0.0)"))

    def test_cell_id_fontsize_property(self):
        self.assertIsNone(self.C1.cell_id_fontsize)
        for c in self.C1:
            self.assertIsNone(c.id_fontsize)

        self.C1.cell_id_fontsize = 20
        self.assertEqual(self.C1.cell_id_fontsize, 20)
        for c in self.C1:
            self.assertEqual(c.id_fontsize, 20)

            # The effect of the cell_id_fontsize can only be seen when plotting
            # self.C1.plot()

    def test_pos_and_rotation(self):
        self.assertAlmostEqual(self.C2.pos, -2 + 3j)
        self.assertAlmostEqual(self.C2.rotation, 20)

        with self.assertRaises(AttributeError):
            self.C2.pos = 0
        self.assertAlmostEqual(self.C2.pos, -2 + 3j)

        with self.assertRaises(AttributeError):
            self.C2.rotation = 30
        self.assertAlmostEqual(self.C2.rotation, 20.0)

    def test_get_ii_and_jj(self):
        # This test is here simple to indicate if the Cluster._ii_and_jj
        # variable ever changes, in which case it will fail. If this is the
        # case change this test_get_ii_and_jj unittest to reflect the
        # changes and make this test below pass.
        self.assertEqual(sorted(cell.Cluster._ii_and_jj.keys()),
                         [1, 3, 4, 7, 13, 19])

        self.assertEqual(cell.Cluster._get_ii_and_jj(1), (1, 0))
        self.assertEqual(cell.Cluster._get_ii_and_jj(3), (1, 1))
        self.assertEqual(cell.Cluster._get_ii_and_jj(4), (2, 0))
        self.assertEqual(cell.Cluster._get_ii_and_jj(7), (2, 1))
        self.assertEqual(cell.Cluster._get_ii_and_jj(13), (3, 1))
        self.assertEqual(cell.Cluster._get_ii_and_jj(19), (3, 2))

        # Test if we get (0,0) for an invalid key.
        self.assertEqual(cell.Cluster._get_ii_and_jj(30), (0, 0))

    def test_get_cell_by_id(self):
        for cell_id in range(1, self.C1.num_cells + 1):
            c = self.C1.get_cell_by_id(cell_id)
            self.assertEqual(c.id, cell_id)

        for cell_id in range(1, self.C2.num_cells + 1):
            c = self.C2.get_cell_by_id(cell_id)
            self.assertEqual(c.id, cell_id)

        for cell_id in range(1, self.C3.num_cells + 1):
            c = self.C3.get_cell_by_id(cell_id)
            self.assertEqual(c.id, cell_id)

    def test_remove_all_users(self):
        # Remove all users from the second cell
        self.C1.delete_all_users(2)
        self.assertEqual(self.C1._cells[0].num_users, 2)
        self.assertEqual(self.C1._cells[1].num_users, 0)
        self.assertEqual(self.C1._cells[2].num_users, 5)

        # Remove all users from the second and third cells
        self.C1.delete_all_users([2, 3])
        self.assertEqual(self.C1._cells[0].num_users, 2)
        self.assertEqual(self.C1._cells[1].num_users, 0)
        self.assertEqual(self.C1._cells[2].num_users, 0)

        # Remove all users from all cells
        self.C1.delete_all_users()
        self.assertEqual(self.C1._cells[0].num_users, 0)
        self.assertEqual(self.C1._cells[1].num_users, 0)
        self.assertEqual(self.C1._cells[2].num_users, 0)

    def test_get_all_users(self):
        self.assertEqual(len(self.C1.get_all_users()), 10)

        self.C1._cells[0].delete_all_users()
        self.assertEqual(len(self.C1.get_all_users()), 8)

        self.C1.delete_all_users()
        self.assertEqual(len(self.C1.get_all_users()), 0)

    # TODO: implement or remove
    def test_calc_cluster_radius(self):
        pass

    def test_calc_cluster_external_radius(self):
        self.assertAlmostEqual(self.C1._calc_cluster_external_radius(),
                               2.0 * self.C1.cell_radius)

        self.assertAlmostEqual(self.C2._calc_cluster_external_radius(),
                               2.6457513110645903 * self.C2.cell_radius)

        # Note that the external_radius property will implicitly call the
        # _calc_cluster_external_radius method.
        self.assertAlmostEqual(self.C3.external_radius,
                               4.3588989435406731 * self.C3.cell_radius)

    # noinspection PyTypeChecker
    def test_calc_cell_positions_hexagon(self):
        # xxxxxxxxxx Test with a rotation of 0 degrees xxxxxxxxxxxxxxxxxxxx
        positions = cell.Cluster._calc_cell_positions_hexagon(cell_radius=1.0,
                                                              num_cells=19,
                                                              rotation=None)

        expected_positions = np.array([
            0.0 + 0.0j, 1.5 + 8.66025404e-01j, 0.0 + 1.73205081j,
            -1.5 + 8.66025404e-01j, -1.5 - 8.66025404e-01j, 0.0 - 1.73205081j,
            1.5 - 8.66025404e-01j, 3.0 + 0.0j, 3.0 + 1.73205081j,
            1.5 + 2.59807621j, 0.0 + 3.46410162j, -1.5 + 2.59807621j,
            -3.0 + 1.73205081j, -3.0, -3.0 - 1.73205081j, -1.5 - 2.59807621j,
            0.0 - 3.46410162j, 1.5 - 2.59807621j, 3.0 - 1.73205081j
        ])

        np.testing.assert_array_almost_equal(positions[:, 0],
                                             expected_positions)
        np.testing.assert_array_almost_equal(positions[:, 1], 0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now test with a rotation of 30 degrees xxxxxxxxxxxxxxx
        positions2 = cell.Cluster._calc_cell_positions_hexagon(cell_radius=1.0,
                                                               num_cells=19,
                                                               rotation=30)
        expected_positions2 = shapes.Shape.calc_rotated_pos(
            expected_positions, 30)
        np.testing.assert_array_almost_equal(positions2[:, 0],
                                             expected_positions2)
        np.testing.assert_array_almost_equal(positions2[:, 1], 30.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now with a different cell radius and rotation xxxxxxxx
        expected_positions3 = shapes.Shape.calc_rotated_pos(
            expected_positions * 1.5, 48)
        positions3 = cell.Cluster._calc_cell_positions_hexagon(cell_radius=1.5,
                                                               num_cells=19,
                                                               rotation=48)
        np.testing.assert_array_almost_equal(positions3[:, 0],
                                             expected_positions3)
        np.testing.assert_array_almost_equal(positions3[:, 1], 48.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # noinspection PyTypeChecker
    def test_calc_cell_positions_3sec(self):
        # xxxxxxxxxx Test with a rotation of 0 degrees xxxxxxxxxxxxxxxxxxxx
        positions = cell.Cluster._calc_cell_positions_3sec(cell_radius=1.0,
                                                           num_cells=19,
                                                           rotation=None)

        expected_positions = np.array([
            0.0 + 0.0j, 1.50000000 + 8.66025404e-01j, 0.0 + 1.73205081j,
            -1.50000000 + 8.66025404e-01j, -1.50000000 - 8.66025404e-01j,
            0.0 - 1.73205081j, 1.50000000 - 8.66025404e-01j, 3.00000000 + 0.0j,
            3.00000000 + 1.73205081j, 1.50000000 + 2.59807621j,
            0.0 + 3.46410162j, -1.50000000 + 2.59807621j,
            -3.00000000 + 1.73205081j, -3.00000000 + 0.0j,
            -3.00000000 - 1.73205081j, -1.50000000 - 2.59807621j,
            0.0 - 3.46410162j, 1.50000000 - 2.59807621j,
            3.00000000 - 1.73205081j
        ])
        np.testing.assert_array_almost_equal(positions[:, 0],
                                             expected_positions)

        np.testing.assert_array_almost_equal(positions[:, 1], 0.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test with a rotation of 30 degrees xxxxxxxxxxxxxxxxxxx
        positions2 = cell.Cluster._calc_cell_positions_3sec(cell_radius=1.0,
                                                            num_cells=19,
                                                            rotation=30)

        expected_positions2 = shapes.Shape.calc_rotated_pos(
            expected_positions, 30)
        np.testing.assert_array_almost_equal(positions2[:, 0],
                                             expected_positions2)
        np.testing.assert_array_almost_equal(positions2[:, 1], 30.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now with a different cell radius and rotation xxxxxxxx
        expected_positions3 = shapes.Shape.calc_rotated_pos(
            expected_positions * 1.5, 48)
        positions3 = cell.Cluster._calc_cell_positions_3sec(cell_radius=1.5,
                                                            num_cells=19,
                                                            rotation=48)
        np.testing.assert_array_almost_equal(positions3[:, 0],
                                             expected_positions3)
        np.testing.assert_array_almost_equal(positions3[:, 1], 48.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # noinspection PyTypeChecker
    def test_calc_cell_positions_square(self):
        # xxxxxxxxxx Test with a rotation of 0 degrees xxxxxxxxxxxxxxxxxxxx
        positions = cell.Cluster._calc_cell_positions_square(side_length=1.0,
                                                             num_cells=9,
                                                             rotation=None)

        expected_positions = np.array([[-0.5 + 1.5j, 0.0 + 0.j],
                                       [0.5 + 1.5j, 0.0 + 0.j],
                                       [1.5 + 1.5j, 0.0 + 0.j],
                                       [-0.5 + 0.5j, 0.0 + 0.j],
                                       [0.5 + 0.5j, 0.0 + 0.j],
                                       [1.5 + 0.5j, 0.0 + 0.j],
                                       [-0.5 - 0.5j, 0.0 + 0.j],
                                       [0.5 - 0.5j, 0.0 + 0.j],
                                       [1.5 - 0.5j, 0.0 + 0.j]])
        np.testing.assert_almost_equal(positions, expected_positions)

        positions = cell.Cluster._calc_cell_positions_square(side_length=1.5,
                                                             num_cells=9,
                                                             rotation=None)
        np.testing.assert_almost_equal(positions, expected_positions * 1.5)

        # xxxxxxxxxx Test with rotation of 30 degrees xxxxxxxxxxxxxxxxxxxxx
        positions2 = cell.Cluster._calc_cell_positions_square(side_length=1.0,
                                                              num_cells=9,
                                                              rotation=30)

        expected_positions2 = shapes.Shape.calc_rotated_pos(
            expected_positions, 30)
        expected_positions2[:, 1] = 30
        np.testing.assert_array_almost_equal(positions2, expected_positions2)
        np.testing.assert_array_almost_equal(positions2[:, 1], 30.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_get_vertex_positions(self):
        # For a cluster of a single cell, the cluster vertexes are the same
        # as the cell vertexes
        C1 = cell.Cluster(cell_radius=1.0, num_cells=1)
        np.testing.assert_array_almost_equal(C1.vertices,
                                             C1._cells[0].vertices)

        # THIS TEST IS NOT COMPLETE
        #
        # Except for C1, we are only testing the number of vertexes here,
        # but at least it is something.
        C3 = cell.Cluster(cell_radius=1.0, num_cells=3)
        self.assertEqual(len(C3.vertices), 12)

        C6 = cell.Cluster(cell_radius=1.0, num_cells=6)
        self.assertEqual(len(C6.vertices), 18)

        C7 = cell.Cluster(cell_radius=1.0, num_cells=7)
        self.assertEqual(len(C7.vertices), 18)

        C13 = cell.Cluster(cell_radius=1.0, num_cells=13)
        self.assertEqual(len(C13.vertices), 28)

        C15 = cell.Cluster(cell_radius=1.0, num_cells=15)
        self.assertEqual(len(C15.vertices), 25)

        C19 = cell.Cluster(cell_radius=1.0, num_cells=19)
        self.assertEqual(len(C19.vertices), 30)

        Cinvalid = cell.Cluster(cell_radius=1.0, num_cells=20)
        self.assertEqual(len(Cinvalid.vertices), 0)

    def test_add_random_users(self):
        # Test adding 2 users to the third cell in the cluster
        self.C2.add_random_users(3, 2, 'aqua')
        self.assertEqual(self.C2.num_users, 2)
        self.assertEqual(self.C2._cells[2].num_users, 2)

        # Test adding 3 users to 2 different cells
        self.C2.add_random_users([4, 7], 3, 'g')
        # 8 users, since we added 2 users to cell 3 before and now we added
        # 3 users to cells 4 and 7.
        self.assertEqual(self.C2.num_users, 8)
        self.assertEqual(self.C2._cells[2].num_users, 2)
        self.assertEqual(self.C2._cells[3].num_users, 3)
        self.assertEqual(self.C2._cells[6].num_users, 3)

        # Test adding a different number of users to different cells (cells
        # 1 and 5)
        self.C2.add_random_users([1, 5], [2, 5], ['k', 'b'])
        self.assertEqual(self.C2.num_users, 15)
        self.assertEqual(self.C2._cells[2].num_users, 2)
        self.assertEqual(self.C2._cells[3].num_users, 3)
        self.assertEqual(self.C2._cells[6].num_users, 3)
        self.assertEqual(self.C2._cells[0].num_users, 2)
        self.assertEqual(self.C2._cells[4].num_users, 5)

        # Test adding user with no user color (the default should be used)

        self.C2.add_random_users([2, 6], [8, 3], ['k', None])
        self.C2.add_random_users(2, 4, None)
        self.assertEqual(self.C2._cells[1].num_users, 12)
        self.assertEqual(self.C2._cells[5].num_users, 3)

        # If cell id is not provided, then it is assumed we will add to all
        # cells. Here we add one user in each of the 19 cells in self.C3.
        self.C3.add_random_users(num_users=1)
        self.assertEqual(self.C3.num_users, 19)
        for c in self.C3:
            self.assertEqual(c.num_users, 1)

            # self.C3.plot()

    def test_add_border_users(self):
        self.C1.delete_all_users()
        cell_ids = [1, 2, 3]
        angles = [210, 30, 90]

        # There are many possible ways to call the add_border_users method.
        # 1. Single cell_id with a single or multiple angles
        self.C1.add_border_users(1, 0, 0.9)
        self.C1.add_border_users(1, angles, 0.7)
        self.assertEqual(self.C1._cells[0].num_users, 4)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # 2. Multiple cell_ids with one user added per cell. The angle of
        # the added users may be the same for each cell or different for
        # each cell.
        self.C1.delete_all_users()
        # Add a user with the same angle in each cell
        self.C1.add_border_users(cell_ids, 270, 0.9)
        self.assertEqual(self.C1._cells[0].num_users, 1)
        self.assertEqual(self.C1._cells[1].num_users, 1)
        self.assertEqual(self.C1._cells[2].num_users, 1)
        # Add a user with different angles and ratios for each cell
        self.C1.add_border_users(cell_ids, angles, [0.95, 0.8, 0.7], 'b')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # 3. Add multiple users to multiple cells with different angles,
        # ratios, etc.
        self.C1.delete_all_users()
        # Notice how we have to repeat "angles" 3 times, one time for each
        # cell_id. We also set a different color for the users in each
        # cell.
        self.C1.add_border_users(cell_ids, angles, 0.9, ['g', 'b', 'k'])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_dist_all_users_to_each_cell_no_wrap_around(self):
        all_dists = self.C1.calc_dist_all_users_to_each_cell_no_wrap_around()

        self.assertEqual(all_dists.shape, (10, 3))

        nrows, ncols = all_dists.shape
        expected_all_dists = np.zeros([nrows, ncols])

        all_cells = self.C1._cells
        all_users = self.C1.get_all_users()

        for ii in range(nrows):
            for jj in range(ncols):
                expected_all_dists[ii, jj] \
                    = all_cells[jj].calc_dist(all_users[ii])

        np.testing.assert_array_almost_equal(expected_all_dists, all_dists)

    def test_calc_dist_all_cells_to_all_users(self):
        all_dists = self.C1.calc_dist_all_users_to_each_cell()
        self.assertEqual(all_dists.shape, (10, 3))

        nrows, ncols = all_dists.shape
        expected_all_dists = np.zeros([nrows, ncols])

        all_cells = self.C1._cells
        all_users = self.C1.get_all_users()

        for ii in range(nrows):
            for jj in range(ncols):
                expected_all_dists[ii, jj] \
                    = all_cells[jj].calc_dist(all_users[ii])

        np.testing.assert_array_almost_equal(expected_all_dists, all_dists)

    def test_properties(self):
        # Test num_cells property
        self.assertEqual(self.C1.num_cells, 3)
        self.assertEqual(self.C2.num_cells, 7)
        self.assertEqual(self.C3.num_cells, 19)

        # Test num_users property
        self.assertEqual(self.C1.num_users, 10)
        self.assertEqual(self.C2.num_users, 0)

        # Test cell_radius property
        self.assertEqual(self.C1.cell_radius, 1.0)
        self.assertEqual(self.C2.cell_radius, 1.0)
        self.assertEqual(self.C3.cell_radius, 1.0)

        # Test cell height
        self.assertEqual(self.C1.cell_height, math.sqrt(3) / 2.0)
        self.assertEqual(self.C2.cell_height, math.sqrt(3) / 2.0)
        self.assertEqual(self.C3.cell_height, math.sqrt(3) / 2.0)

        with self.assertRaises(AttributeError):
            # noinspection PyPropertyAccess
            self.C1.cell_radius = 3.0

        with self.assertRaises(AttributeError):
            # noinspection PyPropertyAccess
            self.C1.cell_height = 3.0

        with self.assertRaises(AttributeError):
            self.C1.radius = 3.0

    def test_iterator_cells_in_the_cluster(self):
        i = -1  # Initialize the i variable
        for i, c in enumerate(self.C1):
            self.assertTrue(isinstance(c, cell.Cell))
        self.assertEqual(i, 2)

        i = -1  # Initialize the i variable
        for i, c in enumerate(self.C2):
            self.assertTrue(isinstance(c, cell.Cell))
        self.assertEqual(i, 6)

    def test_create_wrap_around_cells(self):
        # It is complicated to test the create_wrap_around_cells method.
        # However, with a simple plot you can easily see if it was done
        # correctly. Therefore, for now we don't implement proper unittests for
        # create_wrap_around_cells.
        #
        # Note: if you ever implement this test, remove the pragma comment
        # from the create_wrap_around_cells method.
        pass


class GridTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_create_clusters(self):
        G1 = cell.Grid()

        # Try to create a grid with invalid number cells per cluster
        with self.assertRaises(ValueError):
            G1.create_clusters(7, 4, 0.5)

        # Try to create a grid of more then 2 clusters of 2 cells per
        # clusters, which is not supported.
        with self.assertRaises(ValueError):
            G1.create_clusters(3, 2, 0.5)

        # xxxxx Test creating a grid with 2 clusters of 2 cells xxxxxxxxxxx
        # Note that for a grid of clusters with 2 cells, only 2 clusters
        # may be created.
        G2 = cell.Grid()
        G2.create_clusters(2, 2, 0.5)
        cluster2_positions = np.array([c.pos for c in G2._clusters])
        np.testing.assert_array_almost_equal(cluster2_positions,
                                             np.array([0, 0.4330127 + 0.75j]))

        # xxxxx Test creating a grid with 7 clusters of 3 cells xxxxxxxxxxx
        G3 = cell.Grid()
        G3.create_clusters(7, 3, 0.5)
        self.assertEqual(G3.num_clusters, 7)

        cluster3_positions = np.array([c.pos for c in G3._clusters])
        np.testing.assert_array_almost_equal(
            cluster3_positions,
            np.array([
                0, 1.29903811 + 0.75j, 0 + 1.5j, -1.29903811 + 0.75j,
                -1.29903811 - 0.75j, 0 - 1.5j, 1.29903811 - 0.75j
            ]))

        # xxxxx Test creating a grid with 7 clusters of 7 cells xxxxxxxxxxx
        G7 = cell.Grid()
        G7.create_clusters(7, 7, 0.5)
        cluster7_positions = np.array([c.pos for c in G7._clusters])
        np.testing.assert_array_almost_equal(
            cluster7_positions,
            np.array([
                0, 2.16506351 + 0.75j, 0.43301270 + 2.25j, -1.73205081 + 1.5j,
                -2.16506351 - 0.75j, -0.43301270 - 2.25j, 1.73205081 - 1.5j
            ]))

    def test_get_cluster_from_index(self):
        G1 = cell.Grid()
        G1.create_clusters(7, 3, 0.5)
        for i in range(7):
            self.assertTrue(G1._clusters[i] is G1.get_cluster_from_index(i))

    def test_iterator_for_clusters(self):
        G1 = cell.Grid()
        G1.create_clusters(2, 2, 0.5)
        i = -1  # Initialize the i variable
        for i, c in enumerate(G1):
            self.assertTrue(isinstance(c, cell.Cluster))
        self.assertEqual(i, 1)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
