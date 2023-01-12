import unittest
from hausdorffDistanceCalc import * 

class TestHausdorffDistanceCalc(unittest.TestCase):
    def setUp(self) -> None:
        #Array generated randomly in IDL
        self.vec = np.array([-0.836854, -0.172280, 0.187117, 1.61544, -0.176774])
    
    def hausdorffDistanceFor2Dpoints(self):
        pass

if __name__ == '__main__':
    unittest.main()