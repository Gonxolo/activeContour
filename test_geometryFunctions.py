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

if __name__ == '__main__':
    unittest.main()
