import unittest
from ActiveContour import *

class TestActiveContour(unittest.TestCase):
    def setUp(self) -> None:
        image = [[1,2,3],[4,5,6]]
        x = [1,2,3]
        y = [4,5,6]
        self.activeContour = ActiveContour(image, x, y)
        #self.activeContour1 = ActiveContour(image)

    #parab hacer varios asserts en una fun test se usa self.subTest():self.assertBKJa

    def test_get_x_coords(self):
        self.assertTrue(np.array_equal(self.activeContour.get_x_coords(), np.array([1,2,3])))
        #self.assertEquals(self.activeContour1.get_x_coords(), -1)

    def test_get_y_coords(self):
        self.assertTrue(np.array_equal(self.activeContour.get_y_coords(), np.array([4,5,6])))
        #self.assertEquals(self.activeContour1.get_y_coords(), -1)

    def test_get_GGVF(self):
        pass

    def test_laplacian(self):
        pass
    
    def test_calcGGVF(self):
        pass

    def test_plotGVF(self):
        pass

    def test_getCoords(self):
        self.assertTrue(np.array_equal(self.activeContour.getCoords(), np.array([[1,2,3], [4,5,6]])))
        self.assertTrue(np.array_equal(self.activeContour.getCoords(xyRes = np.array([2.,3.])), np.array([[2,4,6], [12,15,18]])))

    def test_setContour(self):
        pass

    def test_getPerimeter(self):
        #in IDL, given the same paramenters it returns 5.6568542
        self.assertAlmostEqual(self.activeContour.getPerimeter(), 5.6568542, 6)
        #in IDL, given the same paramenters it returns 14.422205
        self.assertAlmostEqual(self.activeContour.getPerimeter(xyRes=np.array([2.,3.])), 14.422205, 6)
        #arrays generated randomly in IDL
        self.activeContour.x = [-0.836854, -0.172280, 0.187117, 1.61544, -0.176774]
        self.activeContour.y = [0.961467, 0.484601, -0.984221, -0.446593, 0.524246]
        #in IDL, using the paramenters x and y from above, it returns 6.68629
        self.assertAlmostEqual(self.activeContour.getPerimeter(), 6.68629,5)

    def test_getDistance(self):
        #in IDL, using the same paramenters it returns [1.4142136, 1.4142136, 2.8284271]
        self.assertTrue(np.allclose(self.activeContour.getDistance(), np.array([1.4142136, 1.4142136, 2.8284271])))
        #in IDL, using the same paramenters it returns [3.6055513, 3.6055513, 7.2111026]
        self.assertTrue(np.allclose(self.activeContour.getDistance(xyRes=np.array([2.,3.])), np.array([3.6055513, 3.6055513, 7.2111026])))
        #arrays generated randomly in IDL
        self.activeContour.x = [-0.836854, -0.172280, 0.187117, 1.61544, -0.176774]
        self.activeContour.y = [0.961467, 0.484601, -0.984221, -0.446593, 0.524246]
        #in IDL, using the paramenters x and y from above, it returns [0.817961, 1.51215, 1.52615, 2.03827, 0.791750]
        self.assertTrue(np.allclose(self.activeContour.getDistance(), np.array([0.817961, 1.51215, 1.52615, 2.03827, 0.791750])))


    def test_arcSample(self):
        pass

    def test_adjustContour(self):
        pass

    def test_gradient(self):
        pass

    def test_edgeMap(self):
        pass

if __name__ == '__main__':
    unittest.main()
