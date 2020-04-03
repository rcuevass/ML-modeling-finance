# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:39:02 2020

@author: schne
"""

#import numpy as np
import sys

# to specify Typle type as part of outputs of functions
from typing import Tuple

module_path = '..\\'
if module_path not in sys.path:
    sys.path.insert(1, module_path)

import copy
import numpy as np
from timeit import default_timer as timer
from utils.logger import get_log_object

log_normalizer = get_log_object()


def normalize_data(options_lhs: np.ndarray, cp_ratios: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    # starting time
    start_time = timer()

    log_normalizer.info('===============================================================')
    log_normalizer.info('Scaling data values between 0 and 1')
    cp_ratios_aux = []
    for value in cp_ratios:
        cp_ratios_aux.append([value])
    data = copy.deepcopy(options_lhs)
    data = np.append(data, copy.deepcopy(cp_ratios_aux), 1)
    
    for i in range(0, len(data[0, :])):
        subset = data[:, i]
        subset -= min(data[:, i])
        subset /= (max(data[:, i]) - min(data[:, i]))

    # ending time
    end_time = timer()
    
    log_normalizer.info('Output normalized data = %s\n', str(data))
    log_normalizer.info('Time spanned normalizing data = %f seconds', end_time - start_time)
    log_normalizer.info('========================++++++=======================================')

    # returning, respectively volatility and implied volatility
    return data[:, [0, 1, 3, 4]], data[:, 2]
