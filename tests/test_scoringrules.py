import unittest
import gpmp as gp
import gpmp.num as gnp    

class TestScoringRules(unittest.TestCase):
    def test_tcrps_01(self):
        mu = 4.0
        sigma = gnp.sqrt(2)
        a = -gnp.inf
        b = 3.0
        z = 3.5
        x = gnp.to_scalar(gp.misc.scoringrules.tcrps_gaussian(mu, sigma, z, a, b))
        self.assertAlmostEqual(x, 0.02661950759116)

    def test_tcrps_02(self):
        mu = 1.86
        sigma = gnp.sqrt(0.8)
        a = 1.5
        b = 2.3
        z = 1.64
        x = gnp.to_scalar(gp.misc.scoringrules.tcrps_gaussian(mu, sigma, z, a, b))
        self.assertAlmostEqual(x, 0.159017709237)
        
    def test_tcrps_03(self):
        mu = -1.7
        sigma = gnp.sqrt(0.1)
        a = -1.0
        b = gnp.inf
        z = -1.5
        x = gnp.to_scalar(gp.misc.scoringrules.tcrps_gaussian(mu, sigma, z, a, b))
        self.assertAlmostEqual(x, 1.0475838916335078e-05)

    def test_tcrps_04(self):
        mu = 0.0 
        sigma = 1.0
        a = -1.0
        b = 1.0
        z = [1.64, 0.0, 0.5, -1.5]
        x = gp.misc.scoringrules.tcrps_gaussian(mu, sigma, z, a, b)
        x_expected = gnp.array([0.5879712039755658, 0.21922482360305862, 0.3169333776028054, 0.5879712039755658])
        self.assertAlmostEqual(gnp.to_scalar(gnp.norm(x - x_expected)), 0.)
    
        
if __name__ == "__main__":
    unittest.main()
