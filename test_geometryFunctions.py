import unittest
from geometryFunctions import * 

class TestGeometryFunctions(unittest.TestCase):
    def setUp(self) -> None:
        #Array generated randomly in IDL
        self.vec = np.array([-0.836854, -0.172280, 0.187117, 1.61544, -0.176774])
    
    def test_polygon_perimeter(self):
        pass

    def test_calcNorm_L1ForVector(self):
        #Using the same value of self.vec, it returns 1.84551
        self.assertAlmostEqual(calcNorm_L1ForVector(self.vec),2.98847, 4)

    def test_calcNorm_L2ForVector(self):
        self.assertAlmostEqual(calcNorm_L2ForVector(self.vec),1.84551, 5)

    def test_calcNorm_LInfiniteForVector(self):
        self.assertAlmostEqual(calcNorm_LInfiniteForVector(self.vec),1.61544, 5)

if __name__ == '__main__':
    unittest.main()
