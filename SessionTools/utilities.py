import sys
import os

import numpy as np

class suppress_output:
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr
            
# @njit
def cart2pol(x, y):
    '''
    cartesian to polar coordinates
    :param x:
    :param y:
    :return: rho: length of vector
             phi: angle of vector
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    return rho, phi

# @njit
def pol2cart(rho, phi):
    '''
    polar to cartesian coordinates

    :param rho: vector length
    :param phi: angle
    :return:
    '''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def validate_dict_keys(input_dict, required_keys):
    for key in required_keys:
        if key not in input_dict:
            raise ValueError(f"Missing required key: {key}")