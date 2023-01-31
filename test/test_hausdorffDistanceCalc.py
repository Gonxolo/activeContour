import unittest
from hausdorffDistanceCalc import * 

class TestHausdorffDistanceCalc(unittest.TestCase):
    def setUp(self) -> None:
        #Array generated randomly in IDL
        self.x1 = np.array([-0.836854, -0.172280, 0.187117, 1.61544, -0.176774])
        self.y1 = np.array([0.961467, 0.484601, -0.984221, -0.446593, 0.524246])
        self.x2 = np.array([0.00923687, 0.782694, 0.242533, -0.5265676, 0.466242])
        self.y2 = np.array([0.282261, -0.834615, -0.484865, -0.173802, -1.94928])
    
    def hausdorffDistanceFor2Dpoints(self):
        #en IDL 1.0849839
        self.assertAlmostEqual(hausdorffDistanceFor2Dpoints(self.x1, self.y1, self.x2, self.y2))

if __name__ == '__main__':
    unittest.main()