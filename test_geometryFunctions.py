import unittest
from geometryFunctions import * 

class TestActiveContour(unittest.TestCase):
    def setUp(self) -> None:
        #Array generated randomly in IDL
        self.vec = np.array([-0.836854, -0.172280, 0.187117, 1.61544, -0.176774])
    
    def test_polygon_perimeter(self):
        
        x = []
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(polygon_perimeter(x, y), -1.0)
        
        x = [1.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(polygon_perimeter(x, y), 0.0)

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = []
        self.assertEqual(polygon_perimeter(x, y), -1.0)
        
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0]
        self.assertEqual(polygon_perimeter(x, y), 0.0)

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        xy_factor = []

        expected_res = 11.3137

        self.assertAlmostEqual(polygon_perimeter(x, y, xyFactor=xy_factor), expected_res, 4)
        
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        xy_factor = [0.5]

        expected_res = 5.65685

        self.assertAlmostEqual(polygon_perimeter(x, y, xyFactor=xy_factor), expected_res, 5)
        
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        xy_factor = [0.1, 0.1]

        expected_res = 1.13137

        self.assertAlmostEqual(polygon_perimeter(x, y, xyFactor=xy_factor), expected_res, 5)


    def test_calcNorm_L1ForVector(self):
        #Using the same value of self.vec, it returns 1.84551
        self.assertAlmostEqual(calcNorm_L1ForVector(self.vec),2.98847, 4)

    def test_calcNorm_L2ForVector(self):
        self.assertAlmostEqual(calcNorm_L2ForVector(self.vec),1.84551, 5)

    def test_calcNorm_LInfiniteForVector(self):
        self.assertAlmostEqual(calcNorm_LInfiniteForVector(self.vec),1.61544, 5)
    
    def test_polygon_line_sample(self):

        x = [-1.0, -1.0, 1.0, 1.0]
        y = [-1.0, 1.0, -1.0, 1.0]

        expected_x_out = [-1.00000, -1.00000, -1.00000, -0.333333, 
                           0.333333, 1.00000,  1.00000,  1.00000]
        expected_y_out = [-1.00000,  0.000000,  1.00000,  0.333333,
                          -0.333333, -1.00000,  0.000000,  1.00000]
        
        obtained_x_out, obtained_y_out = polygon_line_sample(x, y)

        self.assertTrue(np.allclose(expected_x_out, obtained_x_out))
        self.assertTrue(np.allclose(expected_y_out, obtained_y_out))


        x = (np.arange(21)/20.0) * 2.0 * np.pi
        y = np.sin(x)

        expected_x_out = [0.000000, 0.314159, 0.628319, 0.942478, 1.25664,
                          1.57080,  1.88496,  2.19911,  2.51327,  2.82743,
                          3.14159,  3.45575,  3.76991,  4.08407,  4.39823,
                          4.71239,  5.02655,  5.34071,  5.65487,  5.96903,  
                          6.28319]
        expected_y_out = [ 0.000000, 0.309017,  0.587785,  0.809017,  0.951057,
                           1.00000,  0.951056,  0.809017,  0.587785,  0.309017,
                          -8.94070e-008, -0.309017, -0.587786, -0.809017, 
                          -0.951056,  -1.00000, -0.951056, -0.809017, -0.587785,
                          -0.309017, 1.78814e-007]
        
        obtained_x_out, obtained_y_out = polygon_line_sample(x, y)

        self.assertTrue(np.allclose(expected_x_out, obtained_x_out))
        self.assertTrue(np.allclose(expected_y_out, obtained_y_out, atol=1e-6))

if __name__ == '__main__':
    unittest.main() # pragma: no cover
