import unittest
from ActiveContour import ActiveContour

class TestActiveContour(unittest.TestCase):
    def setUp(self) -> None:
        self.activeContour = ActiveContour([[1,2,3],[4,5,6]], [1,2,3], [4,5,6])

    #parab hacer varios asserts en una fun test se usa self.subTest():self.assertBKJa
    def test_laplacian():
        pass
    
    def test_calcGGVF():
        pass

    def test_plotGVF():
        pass

    def test_getCoords():
        pass

    

if __name__ == '__main__':
    unittest.main()
