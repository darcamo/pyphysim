#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the cell package.

Each module has several doctests that we run in addition to the unittests
defined here.

"""

__revision__ = "$Revision$"

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import numpy as np

from cell import shapes, cell


# def positivy_angle_rad(angle):
#     """Return the positive angle corresponding to `angle`..

#     Arguments:
#     - `angle`: An angle (in radians)

#     """
#     if angle >= 0:
#         return angle
#     else:
#         return 2 * np.pi + angle

# def get_sorted_array_by_angle(x):
#     """Function to return the complex array sorted by the angles
#     ("positivated angles")

#     """
#     angles_rad = np.angle(x)
#     #while any(x < 0):

#     angles_rad = map(positivy_angle_rad, angles_rad)
#     return x[np.argsort(angles_rad)]


# UPDATE THIS CLASS if another module is added to the comm package
class CellDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the cell
    package.
    """

    def test_shapes(self):
        """Run doctests in the shapes module.
        """
        doctest.testmod(shapes)

    def test_cell(self):
        """Run doctests in the cell module.
        """
        doctest.testmod(cell)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SHAPES module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class CoordinateTestCase(unittest.TestCase):
    def test_calc_dist(self):
        C1 = shapes.Coordinate(0 + 3j)
        C2 = shapes.Coordinate(2 - 4j)
        C3 = shapes.Coordinate(5 - 0j)

        # Sanity check
        self.assertEqual(C1.calc_dist(C2), C2.calc_dist(C1))

        self.assertAlmostEqual(np.sqrt((2 ** 2) + (7 ** 2)), C1.calc_dist(C2))
        self.assertAlmostEqual(np.sqrt((5 ** 2) + (3 ** 2)), C1.calc_dist(C3))
        self.assertAlmostEqual(np.sqrt((3 ** 2) + (4 ** 2)), C2.calc_dist(C3))


class ShapeTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        #         shapes.Shape(pos, radius, rotation)
        self.S1 = shapes.Shape(0 + 0j, 1.5, 0)
        self.S2 = shapes.Shape(2 + 3j, 2, 0)
        self.S3 = shapes.Shape(3 + 5j, 1.2, 30)

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
        self.assertAlmostEqual(shapes.Shape._rotate(cur_pos, 90), 2 + 1j)
        self.assertAlmostEqual(shapes.Shape._rotate(cur_pos, 180), -1 + 2j)


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
        # Even though H1 and H2 have different position and rottion, they
        # have the same radius and since _get_vertex_positions does not
        # account translation and rotation then the _get_vertex_positions
        # method should return the same value for both of them.
        np.testing.assert_array_equal(self.H1._get_vertex_positions(),
                                      self.H3._get_vertex_positions())

        expected_vertexes = 1.5 * np.exp(
            1j * np.pi * np.linspace(0, 300, 6) / 180.)
        # Lets sort the expected vertexes (regarding the angle of the
        # vertexes)
        expected_vertexes = expected_vertexes[np.argsort(np.angle(expected_vertexes))]

        # The way the obtained_vertexes are calculated the vertexes are
        # already in ascending order regarding their angle.
        obtained_vertexes = self.H1._get_vertex_positions()

        np.testing.assert_array_almost_equal(expected_vertexes,
                                             obtained_vertexes)

        expected_vertexes2 = (2. / 1.5) * expected_vertexes
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
            self.H2._get_vertex_positions() + complex(2, 3),
            self.H2.vertices)

        # H3 has rotation and translation.
        np.testing.assert_array_almost_equal(
            shapes.Shape._rotate(self.H3._get_vertex_positions(), 30) + 3 + 5j,
            self.H3.vertices)

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

        # This test is very imcomplete. If any bugs are found in
        # is_point_inside_shape add tests for them here

    def test_get_border_point(self):
        # Test for an angle of 0 degrees
        point_0_degrees = self.H3.get_border_point(0., 1.)
        point_0_degrees_expected = self.H3.pos + self.H3.height
        self.assertAlmostEqual(point_0_degrees, point_0_degrees_expected)

        # Test for an angle of 30 degrees (that is one of the vertexes)
        point_30_degrees = self.H3.get_border_point(30., 1.)
        point_30_degrees_expected = self.H3.pos + self.H3.radius * np.exp(1j * np.pi / 6.)
        self.assertAlmostEqual(point_30_degrees, point_30_degrees_expected)

        # Test for an angle of 30 degrees, but now with a ratio of 0.8
        point_30_degrees_ratio08 = self.H3.get_border_point(30., 0.8)
        point_30_degrees_expected_ratio08 = self.H3.pos + 0.8 * self.H3.radius * np.exp(1j * np.pi / 6.)
        self.assertAlmostEqual(point_30_degrees_ratio08, point_30_degrees_expected_ratio08)

        # Test for an angle of 180 degrees
        point_180_degrees = self.H3.get_border_point(180., 1.)
        point_180_degrees_expected = self.H3.pos - self.H3.height
        self.assertAlmostEqual(point_180_degrees, point_180_degrees_expected)

        # Test for an angle of 180 degrees but with a ratio of 0.5
        point_180_degrees_ratio05 = self.H3.get_border_point(180, 0.5)
        point_180_degrees_expected_ratio05 = self.H3.pos - (self.H3.height / 2.0)
        self.assertAlmostEqual(point_180_degrees_ratio05, point_180_degrees_expected_ratio05)


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
        np.testing.assert_almost_equal(R1._get_vertex_positions(), R2._get_vertex_positions())

        # Now lets create a rectangle with rotation and translates to a
        # different position. The _get_vertex_positions method should not
        # be affected by this
        trans_step = 1.4 - 3j
        R3 = shapes.Rectangle(A1 + trans_step, B1 + trans_step, 30)
        np.testing.assert_almost_equal(R1._get_vertex_positions(), R3._get_vertex_positions())


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
        angles = np.linspace(0,
                             (num_vertexes - 1.) / num_vertexes * 360,
                             num_vertexes)
        expected_vertexes = np.array(
            list(map(self.C1.get_border_point,
                     angles, np.ones(angles.shape)))) - self.C1.pos
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


class ShapesModuleMethodsTestCase(unittest.TestCase):
    def test_from_complex_array_to_real_matrix(self, ):
        A = np.random.randn(10) + 1j * np.random.randn(10)
        B = A.copy()
        B.shape = (B.size, 1)

        expected_value = np.hstack([B.real, B.imag])
        np.testing.assert_array_almost_equal(
            expected_value,
            shapes.from_complex_array_to_real_matrix(A))
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
        self.C1 = cell.Cell(2 - 3j, 2.5, 1, 30)
        self.C2 = cell.Cell(0 + 2j, 2, 1, 20)
        self.C3 = cell.Cell(-3 + 5j, 1.5, 1, 70)

    def test_add_user(self):
        # The cell has no users yet
        self.assertEqual(self.C1.num_users, 0)

        # User with the same position as the cell center
        user1 = cell.Node(self.C1.pos, marker_color='b')
        self.C1.add_user(user1, relative_pos_bool=False)

        # User (relative to cell center) located at the top of the cell
        user2 = cell.Node(0 + 0.99999j, marker_color='r')
        self.C1.add_user(user2)

        # User (relative to cell center) located at some point in the north
        # east part of the cell
        user3 = cell.Node(0.4 + 0.7j, marker_color='g')
        self.C1.add_user(user3)

        # We have successfully added 3 users to the cell
        self.assertEqual(self.C1.num_users, 3)

        # This user will fall ouside the cell and add_user should raise an
        # exception
        user4 = cell.Node(0.4 + 0.8j)
        self.assertRaises(ValueError, self.C1.add_user,
                          # Args to self.C1.add_user
                          user4)

        # This user will also fall ouside the cell and add_user should
        # raise an exception
        user5 = cell.Node(0 + 0j)
        self.assertRaises(ValueError, self.C1.add_user,
                          # Args to self.C1.add_user
                          user5, relative_pos_bool=False)

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
        self.assertAlmostEqual(self.C1.users[0].pos,
                               expected_pos)

        # Test adding a single user without specifying the ration, which
        # should default to 1.
        self.C1.delete_all_users()
        self.C1.add_border_user(angles)
        expected_pos = self.C1.get_border_point(angles, 1)
        self.assertAlmostEqual(self.C1.users[0].pos,
                               expected_pos)

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
            absolute_pos = self.C3.get_border_point(
                angles3[index], ratios3[index])
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


class ClusterTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.C1 = cell.Cluster(cell_radius=1.0, num_cells=3)
        self.C2 = cell.Cluster(cell_radius=1.0, num_cells=7)
        self.C3 = cell.Cluster(cell_radius=1.0, num_cells=19)

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

    def test_remove_all_users(self):
        # Remove all users from the second cell
        self.C1.remove_all_users(2)
        self.assertEqual(self.C1._cells[0].num_users, 2)
        self.assertEqual(self.C1._cells[1].num_users, 0)
        self.assertEqual(self.C1._cells[2].num_users, 5)

        # Remove all users from the second and third cells
        self.C1.remove_all_users([2, 3])
        self.assertEqual(self.C1._cells[0].num_users, 2)
        self.assertEqual(self.C1._cells[1].num_users, 0)
        self.assertEqual(self.C1._cells[2].num_users, 0)

        # Remove all users from all cells
        self.C1.remove_all_users()
        self.assertEqual(self.C1._cells[0].num_users, 0)
        self.assertEqual(self.C1._cells[1].num_users, 0)
        self.assertEqual(self.C1._cells[2].num_users, 0)

    def test_get_all_users(self):
        self.assertEqual(len(self.C1.get_all_users()), 10)

        self.C1._cells[0].delete_all_users()
        self.assertEqual(len(self.C1.get_all_users()), 8)

        self.C1.remove_all_users()
        self.assertEqual(len(self.C1.get_all_users()), 0)

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

    # TODO: Implement-me
    def test_calc_cell_positions(self):
        pass

    def test_get_vertex_positions(self):
        # For a cluster of a single cell, the cluster vertexes are the same
        # as the cell vertexes
        C1 = cell.Cluster(cell_radius=1.0, num_cells=1)
        np.testing.assert_array_almost_equal(C1.vertices, C1._cells[0].vertices)

        # THIS TEST IS NOT COMPLETE
        #
        # Except for C1, we are only testing the number of verexes here,
        # but at least it is something.
        C3 = cell.Cluster(cell_radius=1.0, num_cells=3)
        self.assertEqual(len(C3.vertices), 12)

        C6 = cell.Cluster(cell_radius=1.0, num_cells=6)
        self.assertEqual(len(C6.vertices), 18)

        C7 = cell.Cluster(cell_radius=1.0, num_cells=7)
        self.assertEqual(len(C7.vertices), 18)

        C13 = cell.Cluster(cell_radius=1.0, num_cells=13)
        self.assertEqual(len(C13.vertices), 30)

        C15 = cell.Cluster(cell_radius=1.0, num_cells=15)
        self.assertEqual(len(C15.vertices), 30)

        C19 = cell.Cluster(cell_radius=1.0, num_cells=19)
        self.assertEqual(len(C19.vertices), 30)

        Cinvalid = cell.Cluster(cell_radius=1.0, num_cells=20)
        self.assertEqual(len(Cinvalid.vertices), 0)

    # TODO: Implement-me
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

        #self.C2.plot()

    def test_add_border_users(self):
        self.C1.remove_all_users()
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
        self.C1.remove_all_users()
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
        self.C1.remove_all_users()
        # Notice how we have to repeate "angles" 3 times, one time for each
        # cell_id. We also set a different color for the users in each
        # cell.
        self.C1.add_border_users(cell_ids,
                                 [angles, angles, angles],
                                 0.9,
                                 ['g', 'b', 'k'])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_dist_all_cells_to_all_users(self):
        all_dists = self.C1.calc_dist_all_cells_to_all_users()
        self.assertEqual(all_dists.shape, (10, 3))

        nrows, ncols = all_dists.shape
        expected_all_dists = np.zeros([nrows, ncols])

        all_cells = self.C1._cells
        all_users = self.C1.get_all_users()

        for ii in range(nrows):
            for jj in range(ncols):
                expected_all_dists[ii, jj] = all_cells[jj].calc_dist(all_users[ii])

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

    def test_iterator_cells_in_the_cluster(self):
        for i,c in enumerate(self.C1):
            self.assertTrue(isinstance(c, cell.Cell))
        self.assertEqual(i, 2)

        for i,c in enumerate(self.C2):
            self.assertTrue(isinstance(c, cell.Cell))
        self.assertEqual(i, 6)

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
        np.testing.assert_array_almost_equal(
            cluster2_positions,
            np.array([0, 0.4330127 + 0.75j]))

        # xxxxx Test creating a grid with 7 clusters of 3 cells xxxxxxxxxxx
        G3 = cell.Grid()
        G3.create_clusters(7, 3, 0.5)
        self.assertEqual(G3.num_clusters, 7)

        cluster3_positions = np.array([c.pos for c in G3._clusters])
        np.testing.assert_array_almost_equal(
            cluster3_positions,
            np.array([0, 1.29903811 + 0.75j, 0 + 1.5j,
                      -1.29903811 + 0.75j, -1.29903811 - 0.75j,
                      0 - 1.5j, 1.29903811 - 0.75j]))

        # xxxxx Test creating a grid with 7 clusters of 7 cells xxxxxxxxxxx
        G7 = cell.Grid()
        G7.create_clusters(7, 7, 0.5)
        cluster7_positions = np.array([c.pos for c in G7._clusters])
        np.testing.assert_array_almost_equal(
            cluster7_positions,
            np.array([0, 2.16506351 + 0.75j, 0.43301270 + 2.25j,
                      -1.73205081 + 1.5j, -2.16506351 - 0.75j,
                      -0.43301270 - 2.25j, 1.73205081 - 1.5j]))

    def test_iterator_for_clusters(self):
        G1 = cell.Grid()
        G1.create_clusters(2, 2, 0.5)
        for i, c in enumerate(G1):
            self.assertTrue(isinstance(c, cell.Cluster))
        self.assertEqual(i, 1)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
