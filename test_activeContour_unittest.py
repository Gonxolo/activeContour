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

        test_matrix = [
            [-0.836854,  -0.172280,   0.187117,   1.61544,   -0.176774],
            [ 0.653145,  -0.546364,   0.194146,   0.925709,   1.20432],
            [ 1.53055,   -1.35556,    0.0514889,  1.02018,   -1.22616],
            [ 0.708497,   0.871673,  -0.789721,   0.332079,   0.205603],
            [-0.169367,  -0.318417,  -0.295643,   0.522291,  -2.23105]
        ]

        expected_matrix_convolution = [
            [ 0.66591765,      0.012273326,   0.052208400,    -1.0948680,      0.56363349],
            [-0.59603944,      0.75719726,    0.054236547,    -0.46943980,    -0.92206375],
            [-1.2139334,       1.7500138,     0.019178236,    -0.98239035,     1.2502566],
            [-0.45590434,     -0.91020440,    0.92644553,     -0.35022358,    -0.74269571],
            [ 0.33935901,      0.17016366,    0.11910681,     -1.0048016,      1.3236182]
        ]

        obtained_matrix_convolution = self.activeContour.laplacian(test_matrix)

        self.assertTrue(np.allclose(expected_matrix_convolution, obtained_matrix_convolution))

    def test_gradient(self):
        
        test_matrix_size = 5 # square matrix (n x n)
        test_matrix = np.arange(test_matrix_size * test_matrix_size) \
                        .reshape([test_matrix_size, test_matrix_size])
        
        test_matrix = [
            [-0.836854,  -0.172280,   0.187117,   1.61544,   -0.176774],
            [ 0.653145,  -0.546364,   0.194146,   0.925709,   1.20432],
            [ 1.53055,   -1.35556,    0.0514889,  1.02018,   -1.22616],
            [ 0.708497,   0.871673,  -0.789721,   0.332079,   0.205603],
            [-0.169367,  -0.318417,  -0.295643,   0.522291,  -2.23105]
        ]

        expected_gradient_0 = [
            [ 0.511986,    0.511986,   0.893860,  -0.181946,  -0.181946],
            [-0.229499,   -0.229499,   0.736036,   0.505087,   0.505087],
            [-0.739533,   -0.739533,   1.18787,   -0.638825,  -0.638825],
            [-0.749109,   -0.749109,  -0.269797,   0.497662,   0.497662],
            [-0.0631376,  -0.0631376,  0.420354,  -0.967704,  -0.967704]
        ]

        expected_gradient_1 = [
            [ 1.18370,   -0.591640,  -0.0678143,  -0.297631,  -0.524693],
            [ 1.18370,   -0.591640,  -0.0678143,  -0.297631,  -0.524693],
            [ 0.0276763,  0.709018,  -0.491933,   -0.296815,  -0.499359],
            [-0.849961,   0.518571,  -0.173566,   -0.248944,  -0.502445],
            [-0.849961,   0.518571,  -0.173566,   -0.248944,  -0.502445],
        ]

        obtained_gradient_0 = self.activeContour.gradient(test_matrix, 0)
        obtained_gradient_1 = self.activeContour.gradient(test_matrix, 1)
        
        self.assertTrue(np.allclose(expected_gradient_0, obtained_gradient_0, atol=1e-07))
        self.assertTrue(np.allclose(expected_gradient_1, obtained_gradient_1, atol=1e-07))

        expected_error = -1
        obtained_error = self.activeContour.gradient(test_matrix, 2)
        self.assertEqual(expected_error, obtained_error)


    def test_edgeMap(self):

        test_matrix = [
            [-0.836854,  -0.172280,   0.187117,   1.61544,   -0.176774],
            [ 0.653145,  -0.546364,   0.194146,   0.925709,   1.20432],
            [ 1.53055,   -1.35556,    0.0514889,  1.02018,   -1.22616],
            [ 0.708497,   0.871673,  -0.789721,   0.332079,   0.205603],
            [-0.169367,  -0.318417,  -0.295643,   0.522291,  -2.23105]
        ]

        # edgeMap operates over the image parameter in activeContour
        self.activeContour.image = test_matrix

        self.activeContour.edgeMap()

        expected_u = [
            [-0.20294272,     -0.20294272,   -0.22375031,   -0.17602059,     -0.17602059],
            [-0.24078942,     -0.24078942,   -0.024944522,  -0.0056031423,   -0.0056031423],
            [ 0.28159152,      0.28159152,   -0.16518826,   -0.24506052,     -0.24506052],
            [-0.41912083,     -0.41912083,   -0.18301274,    0.19939879,      0.19939879],
            [-0.20514606,     -0.20514606,    0.24606335,    0.32800322,      0.32800322]
        ]
        expected_v = [
            [-0.28364461,     0.12493585,     0.20088962,     0.18349790,    0.13184969],
            [-0.28364461,     0.12493585,     0.20088962,     0.18349790,    0.13184969],
            [-0.037562439,    0.14268830,    -0.21589385,    -0.015379923,  -0.010891916],
            [ 0.057928947,   -0.25911694,    -0.42880863,     0.15213467,    0.14425511],
            [ 0.057928947,   -0.25911694,    -0.42880863,     0.15213467,    0.14425511]
        ]

        obtained_u = self.activeContour.u
        obtained_v = self.activeContour.v
        
        self.assertTrue(np.allclose(expected_u, obtained_u))
        self.assertTrue(np.allclose(expected_v, obtained_v))

    
    def test_calcGGVF(self):

        test_matrix = [
            [-0.836854,  -0.172280,   0.187117,   1.61544,   -0.176774],
            [ 0.653145,  -0.546364,   0.194146,   0.925709,   1.20432],
            [ 1.53055,   -1.35556,    0.0514889,  1.02018,   -1.22616],
            [ 0.708497,   0.871673,  -0.789721,   0.332079,   0.205603],
            [-0.169367,  -0.318417,  -0.295643,   0.522291,  -2.23105]
        ]

        # edgeMap operates over the image parameter in activeContour
        self.activeContour.image = test_matrix

        self.activeContour.gvf_iterations = 30
        self.activeContour.mu = 0.25

        self.activeContour.calcGGVF()

        expected_u = [
            [-0.19888015,     -0.19051523,     -0.21328954,     -0.17007296,     -0.16678564],
            [-0.22533313,     -0.21492307,    -0.069503441,    -0.059182901,    -0.061994284],
            [ 0.16040234,      0.20230814,     -0.14182237,     -0.16331192,     -0.16201230],
            [-0.37913884,     -0.39737244,     -0.16747069,      0.15865931,      0.17054295],
            [-0.19488824,     -0.18525131,      0.23100745,      0.30214295,      0.31290017]
        ]
        expected_v = [
            [-0.26007422,     0.090908136,   0.17828062,    0.17093918,    0.13137188],
            [-0.25864705,     0.085847438,   0.12721763,    0.13828413,    0.11447035],
            [-0.051232137,    0.10007335,   -0.17521568,    0.0036853191,  0.018308106],
            [ 0.036271498,   -0.24825132,   -0.39794455,    0.10567059,    0.12099263],
            [ 0.0060995944,  -0.24035929,   -0.40910713,    0.12275513,    0.13339623]
        ]        

        obtained_u = self.activeContour.u
        obtained_v = self.activeContour.v

        self.assertTrue(np.allclose(expected_u, obtained_u))
        self.assertTrue(np.allclose(expected_v, obtained_v))

    def test_setContour(self):

        test_matrix = [
            [-0.836854,  -0.172280,   0.187117,   1.61544,   -0.176774],
            [ 0.653145,  -0.546364,   0.194146,   0.925709,   1.20432]
        ]

        self.activeContour.setContour(test_matrix[0], test_matrix[1])

        expected_x = test_matrix[0]
        expected_y = test_matrix[1]
        expected_npts = len(expected_x)
        
        obtained_x = self.activeContour.x
        obtained_y = self.activeContour.y
        obtained_npts = self.activeContour.npts

        self.assertTrue(np.array_equal(expected_x, obtained_x))
        self.assertTrue(np.array_equal(expected_y, obtained_y))
        self.assertTrue(expected_npts, obtained_npts)

    def test_plotGVF(self):
        pass

    def test_getCoords(self):
        pass

    

if __name__ == '__main__':
    unittest.main() # pragma: no cover
