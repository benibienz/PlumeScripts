from scipy.io import loadmat
import numpy as np

TEST_VARS = loadmat('/Users/beni/PycharmProjects/PlumeScripts/PlumeScripts/testfile.mat')


def test_var(var, var_str, dec=2):
    """
    Test against var from .mat file. Function to be inserted into scripts for debugging purposes only.
    :param var: variable to test
    :param var_str: string of variable name in .mat file
    :param dec: decimal precision
    :return: Raises an exception even if it passes to stop the code.
    """
    test_var = TEST_VARS[var_str]
    if isinstance(var, (float, int)):
        test_var = test_var[0]
    print(test_var, var)
    np.testing.assert_array_almost_equal(test_var, var, decimal=dec)
    raise Exception('Test passed')

