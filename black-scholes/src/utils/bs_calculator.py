# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:24:46 2020

@author: schne
"""
import sys
# to specify argument types and outputs in functions
from typing import Tuple

module_path = '..\\'
if module_path not in sys.path:
    sys.path.insert(1, module_path)
    

import math
from utils.logger import get_log_object
import numpy as np
from timeit import default_timer as timer

#log_calculator = get_log_object()


def phi(x: float) -> np.ndarray:
    return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi)


def pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return phi((x-mu)/sigma) / sigma


def Phi(z: float) -> float:

    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0
    
    total = 0.0
    term = z
    i = 3
    while total != total + term:
        total += term
        term *= z*z / float(i)
        i += 2
    return 0.5 + total*phi(z)


def cdf(z, mu: float = 0.0, sigma: float = 1.0) -> float:
    return Phi((z-mu)/sigma)


def callPriceRatio(bs_variables: np.ndarray) -> np.ndarray:
    #log_calculator.info('===============================================================')
    
    start_time = timer()
    bs_variables = np.asarray(bs_variables)
    
    sox, r, sigma, t = bs_variables[:, 0], bs_variables[:, 1], bs_variables[:, 2], bs_variables[:, 3]
    
    assert np.all(sox >= 0), "sox must be a value equal to or greater than zero"
    assert np.all(sigma >= 0), "sigma must be a value equal to or greater than zero"
    assert np.all(t > 0), "t must be a value greater than zero"
        
    d1 = (np.log(sox) + (r + sigma * sigma / 2.0) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    cdf_v = np.vectorize(cdf)
    
    price_ratio = sox * cdf_v(d1) - np.exp(-r * t) * cdf_v(d2)
    
    end_time = timer()
    
    #log_calculator.info('Output of call price ratio calculations =\n%s', str(price_ratio))
    #log_calculator.info('Time spanned calculating call price ratios = %f seconds', end_time - start_time)
    #log_calculator.info('===============================================================')
    
    return price_ratio


def putPriceRatio(bs_variables: np.ndarray) -> float:
    sox, r, sigma, t = bs_variables
    call = callPriceRatio(bs_variables)
    return call - sox + math.exp(-r*t)


def optionPriceRatios(bs_variables: np.ndarray) -> Tuple[np.ndarray, float]:
    call_ratio = callPriceRatio(bs_variables)
    put_ratio = putPriceRatio(bs_variables)

    return call_ratio, put_ratio
