import unittest
from src.geometryFunctions import * 

class TestGeometryFunctions(unittest.TestCase):
    def setUp(self) -> None:
        #Array generated randomly in IDL
        self.vec = np.array([-0.836854, -0.172280, 0.187117, 1.61544, -0.176774])
    
    def test_polygon_perimeter(self):

        with self.subTest('Case 1: Empty xCoords array'):
            xCoords = []
            yCoords = [1.0, 2.0, 3.0, 4.0, 5.0]

            expected_perimeter = -1.0 # Invalid xCoords
            obtained_perimeter = polygon_perimeter(xCoords, yCoords)

            self.assertEqual(expected_perimeter, obtained_perimeter)

        with self.subTest('Case 2: One element xCoords array'):
            xCoords = [1.0]
            yCoords = [1.0, 2.0, 3.0, 4.0, 5.0]

            expected_perimeter = 0.0
            obtained_perimeter = polygon_perimeter(xCoords, yCoords)

            self.assertEqual(expected_perimeter, obtained_perimeter)

        with self.subTest('Case 3: Empty yCoords array'):
            xCoords = [1.0, 2.0, 3.0, 4.0, 5.0]
            yCoords = []

            expected_perimeter = -1.0 # Invalid yCoords
            obtained_perimeter = polygon_perimeter(xCoords, yCoords)

            self.assertEqual(expected_perimeter, obtained_perimeter)

        with self.subTest('Case 4: One element yCoords array'):
            xCoords = [1.0, 2.0, 3.0, 4.0, 5.0]
            yCoords = [1.0]

            expected_perimeter = 0.0
            obtained_perimeter = polygon_perimeter(xCoords, yCoords)

            self.assertEqual(expected_perimeter, obtained_perimeter)

        with self.subTest('Case 5: x=y line | Empty xyFactor'):
            xCoords = [1.0, 2.0, 3.0, 4.0, 5.0]
            yCoords = [1.0, 2.0, 3.0, 4.0, 5.0]
            xy_factor = []

            expected_perimeter = 11.3137
            obtained_perimeter = polygon_perimeter(xCoords, yCoords, xyFactor=xy_factor)

            self.assertAlmostEqual(obtained_perimeter, expected_perimeter, places=4)

        with self.subTest('Case 6: x=y line | xyFactor = [0.5]'):
            xCoords = [1.0, 2.0, 3.0, 4.0, 5.0]
            yCoords = [1.0, 2.0, 3.0, 4.0, 5.0]
            xy_factor = [0.5]

            expected_perimeter = 5.65685
            obtained_perimeter = polygon_perimeter(xCoords, yCoords, xyFactor=xy_factor)

            self.assertAlmostEqual(obtained_perimeter, expected_perimeter, places=5)

        with self.subTest('Case 6: x=y line | xyFactor = [0.1, 0.1]'):
            xCoords = [1.0, 2.0, 3.0, 4.0, 5.0]
            yCoords = [1.0, 2.0, 3.0, 4.0, 5.0]
            xy_factor = [0.1, 0.1]

            expected_perimeter = 1.13137
            obtained_perimeter = polygon_perimeter(xCoords, yCoords, xyFactor=xy_factor)

            self.assertAlmostEqual(obtained_perimeter, expected_perimeter, places=5)


    def test_calcNorm_L1ForVector(self):        

        expected_norm = 2.98847 # Value obtained from IDL
        obtained_norm = calcNorm_L1ForVector(self.vec)

        self.assertAlmostEqual(expected_norm, obtained_norm, places=4)

    def test_calcNorm_L2ForVector(self):

        expected_norm = 1.84551 # Value obtained from IDL
        obtained_norm = calcNorm_L2ForVector(self.vec)

        self.assertAlmostEqual(expected_norm, obtained_norm, places=5)

    def test_calcNorm_LInfiniteForVector(self):
        
        expected_norm = 1.61544 # Value obtained from IDL
        obtained_norm = calcNorm_LInfiniteForVector(self.vec)
        
        self.assertAlmostEqual(expected_norm, obtained_norm, places=5)
    
    def test_polygon_line_sample(self):

        with self.subTest('Case 0: Number of segments less than 1'):
            xCoords = np.array([])
            yCoords = np.array([])

            self.assertIsNone(polygon_line_sample(xCoords, yCoords))
            self.assertIsNone(polygon_line_sample(xCoords, yCoords, f_close_output=True))
            self.assertIsNone(polygon_line_sample(xCoords, yCoords, f_close_output=False))

        with self.subTest('Case 1: Square | Array squeeze/flatten'):

            xCoords = np.array([[0.0, 1.0, 1.0, 0.0]])
            yCoords = np.array([[0.0, 0.0, 1.0, 1.0]])

            expected_x_sample = np.array([0.00, 0.25, 0.50, 0.75, 
                                    1.00, 1.00, 1.00, 1.00, 
                                    1.00, 0.75, 0.50, 0.25, 
                                    0.00, 0.00, 0.00, 0.00])

            expected_y_sample = np.array([0.00, 0.00, 0.00, 0.00, 
                                    0.00, 0.25, 0.50, 0.75, 
                                    1.00, 1.00, 1.00, 1.00, 
                                    1.00, 0.75, 0.50, 0.25])
            
            obtained_x_sample, obtained_y_sample = polygon_line_sample(xCoords, yCoords, f_close_output=True)

            self.assertTrue(np.allclose(expected_x_sample, obtained_x_sample))
            self.assertTrue(np.allclose(expected_y_sample, obtained_y_sample))

        with self.subTest('Case 2: Sine wave'):
            xCoords = (np.arange(21)/20.0) * 2.0 * np.pi
            yCoords = np.sin(xCoords)

            # Expected values obtained from IDL
            expected_x_sample = [0.000000, 0.314159, 0.628319, 0.942478, 1.25664,
                            1.57080,  1.88496,  2.19911,  2.51327,  2.82743,
                            3.14159,  3.45575,  3.76991,  4.08407,  4.39823,
                            4.71239,  5.02655,  5.34071,  5.65487,  5.96903,  
                            6.28319]
            expected_y_sample = [ 0.000000, 0.309017,  0.587785,  0.809017,  0.951057,
                            1.00000,  0.951056,  0.809017,  0.587785,  0.309017,
                            -8.94070e-008, -0.309017, -0.587786, -0.809017, 
                            -0.951056,  -1.00000, -0.951056, -0.809017, -0.587785,
                            -0.309017, 1.78814e-007]
            
            obtained_x_sample, obtained_y_sample = polygon_line_sample(xCoords, yCoords)

            self.assertTrue(np.allclose(expected_x_sample, obtained_x_sample))
            self.assertTrue(np.allclose(expected_y_sample, obtained_y_sample, atol=1e-6))

    def test_curv_d(self):

        # reasonable params
        # points: min -> 4; max -> 500
        # radius: min -> 1; max -> 10

        min_circunference_points = 4 
        max_circunference_points = 500 
        min_circunference_radius = 1
        max_circunference_radius = 10

        for i in range(min_circunference_points, max_circunference_points + 1):

            for j in range(min_circunference_radius, max_circunference_radius + 1):

                polygon_radius = j

                n_pts = i

                d_theta = 2 * np.pi / n_pts

                x = np.array([polygon_radius * np.cos(i * d_theta) for i in range(n_pts)])
                y = np.array([polygon_radius * np.sin(i * d_theta) for i in range(n_pts)])

                _a = np.cos(np.pi / n_pts) / polygon_radius

                expected_curv = np.full(len(x), _a)

                obtainded_curv = get_curv_d(x, y)

                with self.subTest(f'Case {(i * max_circunference_radius) + j}: Number of points -> {i}; Radius -> {j}'):
                    self.assertTrue(np.allclose(expected_curv, obtainded_curv), 
                        msg=f"ex: {expected_curv}\n ob: {obtainded_curv}")

if __name__ == '__main__':
    unittest.main() # pragma: no cover
