import unittest
import numpy as np
from ActiveContour import ActiveContour

class TestActiveContour(unittest.TestCase):
    def setUp(self) -> None:
        self.image = [[1,2,3],
                 [4,5,6]]
        self.x = [1,2,3]
        self.y = [4,5,6]
        self.alpha_min_value = 0.001
        self.beta_min_value = 0.001
        self.gamma_min_value = 0.1
        self.kappa_min_value = 0.0
        self.mu_min_value = 0.001
        self.mu_max_value = 0.25
        self.activeContour = ActiveContour(self.image, self.x, self.y)

    def test_init(self):
        """
        Checks that ActiveContour is properly initialized when only image, x
        and y parameters are given
        """
        
        # activeContour's image, x and y must be the same as the arrays given 
        # on setUp
        self.assertTrue(np.array_equal(self.activeContour.image, self.image))
        self.assertTrue(np.array_equal(self.activeContour.x, self.x))
        self.assertTrue(np.array_equal(self.activeContour.y, self.y))

        # alpha, beta, gamma, kappa, mu, gvf_iterations and iterations must be 
        # in their respective ranges when initialized (smaller than the maximum
        # larger than the minimum)
        self.assertTrue(self.activeContour.alpha >= self.alpha_min_value)
        self.assertTrue(self.activeContour.beta >= self.beta_min_value)
        self.assertTrue(self.activeContour.gamma >= self.gamma_min_value)
        self.assertTrue(self.activeContour.kappa >= self.kappa_min_value)
        self.assertTrue(self.mu_max_value >= self.activeContour.mu >= self.mu_min_value)
        self.assertTrue(self.activeContour.gvf_iterations >= 1)
        self.assertTrue(self.activeContour.iterations >= 1)

        # [u, v] are initialized as None and the value is assigned in edgeMap()
        self.assertIsNone(self.activeContour.u)
        self.assertIsNone(self.activeContour.v)

        # npts must be initialized with a positive integer
        self.assertTrue(self.activeContour.npts >= 0)

    #parab hacer varios asserts en una fun test se usa self.subTest():self.assertBKJa
    def test_laplacian(self):
        
        """
        Basic laplacian test comparing the convolution array obtained in IDL vs.
        the one calculated by laplacian in Python
        """

        ex_image = np.identity(5)

        expected_convolution = np.array([
            [-0.50000021,   0.33333319,   0.24999990,   0.00000000,   0.00000000],
            [ 0.33333319,  -0.83333340,   0.16666660,   0.24999990,   0.00000000],
            [ 0.24999990,   0.16666660,  -0.83333340,   0.16666660,   0.24999990],
            [ 0.00000000,   0.24999990,   0.16666660,  -0.83333340,   0.33333319],
            [ 0.00000000,   0.00000000,   0.24999990,   0.33333319,  -0.50000021]
        ])

        calculated_convolution = self.activeContour.laplacian(ex_image)

        self.assertTrue(np.allclose(expected_convolution, calculated_convolution))

    def test_gradient(self):
        
        pass

    def test_edgeMap(self):
        pass
    
    def test_calcGGVF():
        pass

    def test_plotGVF():
        pass

    def test_getCoords():
        pass

    

if __name__ == '__main__':
    unittest.main()
