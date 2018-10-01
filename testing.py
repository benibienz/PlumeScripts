from scipy.io import loadmat
import numpy as np
import time

TEST_VARS = loadmat('/Users/beni/PycharmProjects/PlumeScripts/PlumeScripts/testfile.mat')


def test_var(var, var_str, dec=2, stop=True, printing=True):
    """
    Test against var from .mat file. Function to be inserted into scripts for debugging purposes only.
    :param var: variable to test
    :param var_str: string of variable name in .mat file
    :param dec: decimal precision
    :param stop: Raises an exception even if it passes to stop the code.
    :param printing: print vars
    """
    # tv = TEST_VARS
    test_var = TEST_VARS[var_str]
    if isinstance(var, (float, int)) or test_var.size == 1:
        test_var = test_var[0]
    if printing:
        print('test: {}\nvar: {}'.format(test_var, var))
    time.sleep(0.1)
    np.testing.assert_array_almost_equal(test_var, var, decimal=dec)
    if stop:
        raise Exception('Test passed')


def test_vars(vars, vars_str):
    """
    Test a group of vars
    :param vars: list of vars
    :param vars_str: list of var strings
    :return:
    """
    failed_tests = 0
    for i in range(len(vars)):
        print(vars_str[i])
        try:
            test_var(vars[i], vars_str[i], stop=False, printing=False)
        except Exception as e:
            failed_tests += 1
            print(e)

    print('Failed {}/{} tests'.format(failed_tests, i + 1))