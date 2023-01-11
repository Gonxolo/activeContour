import unittest
from ActiveContour import *

class TestActiveContour(unittest.TestCase):
    def setUp(self) -> None:
        image = [[1,2,3],[4,5,6]]
        x = [1,2,3]
        y = [4,5,6]
        self.activeContour = ActiveContour(image, x, y)
        self.activeContour1 = ActiveContour(image)

    #parab hacer varios asserts en una fun test se usa self.subTest():self.assertBKJa

    def test_get_x_coords(self):
        self.assertTrue(np.array_equal(self.activeContour.get_x_coords(), np.array([1,2,3])))
        self.assertEquals(self.activeContour1.get_x_coords(), -1)

    def test_get_y_coords(self):
        self.assertTrue(np.array_equal(self.activeContour.get_y_coords(), np.array([1,2,3])))
        self.assertEquals(self.activeContour1.get_y_coords(), -1)

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
        pass

    def test_setContour(self):
        pass

    def test_getPerimeter(self):
        pass

    def test_getDistance(self):
        pass

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
