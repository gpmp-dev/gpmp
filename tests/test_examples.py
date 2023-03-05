import unittest
from examples import (
    gpmp_example01_materncov,
    gpmp_example02_1d_interpolation,
    gpmp_example03_2d,
    gpmp_example04_nd,
    gpmp_example05_1d_custom_kernel,
    gpmp_example06_1d_regression,
    gpmp_example10_sample_paths,
    gpmp_example11_sample_paths_noisy_obs
    )
    

class TestExamples(unittest.TestCase):
    def test_01(self):
        gpmp_example01_materncov.main()

    def test_02(self):
        gpmp_example02_1d_interpolation.main()

    def test_03(self):
        gpmp_example03_2d.main()

    def test_04(self):
        gpmp_example04_nd.main()

    def test_05(self):
        gpmp_example05_1d_custom_kernel.main()

    def test_06(self):
        gpmp_example06_1d_regression.main()

    def test_10(self):
        gpmp_example10_sample_paths.main()

    def test_11(self):
        gpmp_example11_sample_paths_noisy_obs.main()

if __name__ == "__main__":
    unittest.main()
