from pyDOE import *
import numpy as np
from utils.logger import get_log_object
from timeit import default_timer as timer

log_sampler = get_log_object()


def latin_hypercube_sampling(number_sample_points: int,
                             dict_range_for_each_variable: dict) -> np.ndarray:

    """
    :param number_sample_points: int - number of samples to be generated
    :param dict_range_for_each_variable: dictionary - contains as keys name of variables and as values a tuple corresponding
                                    to min and max for each variable
                                    e.g. range_for_each_variable = {var_1: (var_1_min, var_1_max),
                                                                    var_2: (var_2_min, var_2_max),
                                                                    ....,
                                                                    var_n: (var_n_min, var_n_max)}
    :return: sampled_lhs: np.array - returns an array of shape (number_sample_points, number_features)
                                     containing the sample points from LHS
    """
    log_sampler.info('sampling %d points', number_sample_points)
    start_time = timer()
    
    # get number of variables from dictionary
    number_variables = len(dict_range_for_each_variable)

    # get lhs from uniform distribution, between 0 and 1
    sampled_lhs = lhs(n=number_variables, samples=number_sample_points)

    # get list of keys
    list_keys = dict_range_for_each_variable.keys()

    # auxiliary index to loop over matrix for random samples
    index_counter = 0

    for key_name in list_keys:
        
        # get min and max value for corresponding key
        min_val_key_name = dict_range_for_each_variable[key_name][0]
        max_val_key_name = dict_range_for_each_variable[key_name][1]

        # rescale the random samples to be in interval [val_min, val_max] as opposed to [0, 1]
        # to map interval [0,1] to [x_min, x_max]  the following operation needs to be performed
        # r -> (x_max - x_min)*r + x_min. This operation is done column-wise
        sampled_lhs[:, index_counter] *= (max_val_key_name - min_val_key_name)
        sampled_lhs[:, index_counter] += min_val_key_name

        """
        log_sampler.info('===============================================================')
        log_sampler.info('%s range = %s' % (key_name, dict_range_for_each_variable[key_name]))
        log_sampler.info('Min value of %s=%f' % (key_name, sampled_lhs[:, index_counter].min()))
        log_sampler.info('Max value of %s=%f'% (key_name, sampled_lhs[:, index_counter].max()))
        log_sampler.info('===============================================================')
        """

        # update aux index to loop over array
        index_counter += 1
        
    end_time = timer()
    
    #log_sampler.info('Output of sample = \n%s', str(sampled_lhs))
    #log_sampler.info('Time spanned to generate lhs sample = %f seconds ', end_time-start_time)
    #log_sampler.info('Shape of LHS generated = %s', str(sampled_lhs.shape))
    
    return sampled_lhs
