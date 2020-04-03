# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:39:07 2020

@author: schne
"""

import sys
module_path = '..\\src'
if module_path not in sys.path:
    sys.path.insert(1, module_path)

import unittest
import numpy as np

from utils.bs_calculator import callPriceRatio as cPR

class test_black_scholes_calcs(unittest.TestCase):
    
    def test_call_price(self):
        self.assertEqual(round(cPR(np.array(1.1, 0.01, 0.1, 1.5)), 5), 0.12766)
        self.assertEqual(round(cPR(np.array(0.9, 0.01, 0.1, 1)), 5), 0.00858)
        self.assertEqual(round(cPR(np.array(0.8, 0.01, 0.99, 1)), 5), 0.25173)
        self.assertEqual(round(cPR(np.array(8, 0.07, 0.40, 2)), 4), 7.1307)
        
if __name__ == '__main__':
    unittest.main()