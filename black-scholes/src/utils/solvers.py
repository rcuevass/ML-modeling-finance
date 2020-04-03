# to enforce type on function's arguments
from typing import Tuple
# for vectors and arrays
import numpy as np
# to find roots from Brent's method
from scipy import optimize
# custom utilities to use Black-Scholes
from utils import bs_calculator
# custom function to log information generated
from utils.logger import get_log_object
# to handle dataframes
import pandas as pd
# for plotting
import matplotlib.pyplot as plt

# instantiate log
log = get_log_object()


def sf(sigma: float, s_value: float, sox: float, r: float, tau: float) -> float:
    """
    Auxiliary function used to find implicit volatility via Brent's method
    This is the function to be used as parameter of the Brent's solver
    :param sigma: float - volatility
    :param s_value: float - computed  from Black-Scholes
    :param sox: float - stock price
    :param r: float - risk free rate
    :param tau: float - time of maturity
    :return: delta : float - difference between actual s_value and estimated from guess sigma
    """

    # format arguments of Black-Scholes equation as vector
    input_vector = np.array([sox, r, sigma, tau])
    # reshape vector to make it compliant with callPriceRatio function
    input_vector = np.reshape(input_vector, (1, 4))
    # compute s_value from given argument; approximate value of S from guessed sigma
    s_approx_value = bs_calculator.callPriceRatio(input_vector)
    # difference between actual s_value
    delta = s_value - s_approx_value[0]
    # return how off is the actual S from the estimated one from guessed sigma
    return delta


def solve_for_volatility(s_value: float, sox: float, r: float, tau: float,
                         sigma_range: Tuple[float, float] = (0.001, 4.0),
                         max_number_attempts: int = 1):
    """
    Function that allows to find implicit volatility via Brent's method
    :param s_value: float - input value for call price, S
    :param sox: float - stock price
    :param r: float - risk free rate
    :param tau: float - time of maturity
    :param sigma_range: duple of floats - defines range within the root will be looked for
    :param max_number_attempts: int - maximum number of attempts when looking for a root
                                      if the first attempt fails
    :return: sigma_root_value: float - volatility computed from implicit Black-Scholes function
    """

    # initialize output
    sigma_root_value = None

    # get min and max values of sigmas from intervals
    sigma_min = sigma_range[0]
    sigma_max = sigma_range[1]

    # for given values of s_value, sox, r and tau, define function where
    # sigma will be argument. This is the function that will be used as argument
    # in Brent's root finder
    def delta_sigma(sigma_arg: float) -> float:
        """
        Function that computes call price from given s_value, sox, r, tau and sigma = sigma_arg
        s_value, sox, r, tau are parameters for the function
        sigma = sigma_arg is the only argument function; argument to be used in Brent's root finder
        :param sigma_arg: float - volatility, sigma; only argument of this function
        :return: call_price: float
        """
        call_price = sf(sigma_arg, s_value, sox, r, tau)
        return call_price

    # compute function of which root is to be found.
    # it is computed on the ends of the interval to assess feasibility of secant method
    delta_sigma_min = delta_sigma(sigma_min)
    delta_sigma_max = delta_sigma(sigma_max)

    # set counter to zero
    # this counter will keep track of the number of times an attempt is made to
    # find a root via Brent's method
    count_failed_attempts = 0

    # if the secant does not cross the X-axis to look for a root...
    if delta_sigma_max*delta_sigma_min >= 0:

        # record values on the log for investigation purposes
        log.info('delta sigma min = %f', delta_sigma_min)
        log.info('delta sigma max = %f', delta_sigma_max)

        # we inform the user that we  will try a very naive/brute force approach
        # to find a feasible secant
        log.info('finding root by brute force...')

        # we initialize auxiliary end of interval where root will be looked for
        # initialized with the default range defined in function
        delta_sigma_min_i = delta_sigma_min
        delta_sigma_max_i = delta_sigma_max

        # initialize auxiliary ends of the interval; they will be used within the
        # the while loop below...
        # sigma_min_fixed_i = sigma_min
        # sigma_max_fixed_i = sigma_max

        # try the following while the maximum number of attempts have not been reached and
        # the points corresponding to the end of the interval are on opposite sides of the X-axis
        while (count_failed_attempts < max_number_attempts) & (delta_sigma_min_i*delta_sigma_max_i >= 0):

            # update number of failed attempts
            count_failed_attempts += 1

            # update the minimum value in the interval with a random number
            sigma_min_fixed_i = np.random.uniform(low=0.1, high=0.25, size=1)[0]
            # update the minimum value in the interval with a random number between the minimum value we
            # just updated and 5
            sigma_max_fixed_i = np.random.uniform(low=sigma_min_fixed_i, high=5, size=1)[0]

            # compute the value of the function at the recently updated ends of the interval
            delta_sigma_min_i = delta_sigma(sigma_min_fixed_i)
            delta_sigma_max_i = delta_sigma(sigma_max_fixed_i)

            # inform the user...
            log.info('sigma min = %f in attempt = %i', sigma_min_fixed_i, count_failed_attempts)
            log.info('sigma max = %f in attempt = %i', sigma_max_fixed_i, count_failed_attempts)
            log.info('delta sigma min = %f in attempt = %i', delta_sigma_min_i, count_failed_attempts)
            log.info('delta sigma max = %f in attempt = %i', delta_sigma_max_i, count_failed_attempts)
            log.info('####################################################################################')

        '''
        # if we beat the while loop, compute the root
        if count_failed_attempts < max_number_attempts:

            log.info('We were able to find a root in attempt = %i', count_failed_attempts)
            sigma_root_value = optimize.brentq(f=delta_sigma, a=sigma_min_fixed_i, b=sigma_max_fixed_i,
                                               xtol=2e-15, rtol=8.881784197001252e-16,
                                               maxiter=200)
                                               
        '''

        log.info('Attempt=%i = for vector=%s', count_failed_attempts,
                 str((s_value, sox, r, tau)))

        log.info('Estimated volatility=%s', str(sigma_root_value))
        log.info('************************************************************')

    else:
        # find volatility from Brent's root finder
        sigma_root_value = optimize.brentq(f=delta_sigma, a=sigma_min, b=sigma_max,
                                           xtol=2e-15, rtol=8.881784197001252e-16,
                                           maxiter=200)

    return sigma_root_value


def solve_for_volatility_vectorized(arg_array: np.ndarray,
                                    call_price_vector: np.ndarray,
                                    remove_na: bool = True,
                                    generate_plot: bool = True) -> dict:

    # get number of data points
    num_data_pts = arg_array.shape[0]

    # subset for stock price and rate ratio
    sx_rate_array = arg_array[:, 0:2]

    # subset for volatility
    volatility_array = arg_array[:, 2]

    # subset for time of maturity (tau) and reshape to concatenate
    time_maturity_array = np.reshape(arg_array[:, 3], (num_data_pts, 1))

    # reshape call price ratio to properly concatenate
    call_price_array = np.reshape(call_price_vector, (num_data_pts, 1))

    # concatenate sx_rate and time of maturity ...
    sx_rate_maturity_array = np.concatenate((sx_rate_array, time_maturity_array), axis=1)

    # ... and concatenate it to call price
    call_price_sx_rate_maturity_array = np.concatenate((call_price_array, sx_rate_maturity_array), axis=1)

    # find roots
    vector_roots = np.array([solve_for_volatility(s_value=x[0],
                                                  sox=x[1],
                                                  r=x[2],
                                                  tau=x[3]) if solve_for_volatility(s_value=x[0], sox=x[1],
                                                                                    r=x[2], tau=x[3]) is not None
                             else -10 for x in call_price_sx_rate_maturity_array])

    # reshaping vectors into array; need to concatenate to rest of data before turning
    # into data frame
    log.info('reshaping actual and estimated volatilities...')
    vector_roots_array = np.reshape(vector_roots, (num_data_pts, 1))
    volatility_array_reshaped = np.reshape(volatility_array, (num_data_pts, 1))

    # concatenate all data, including actual and computed volatility
    # ... and concatenate it to call price
    log.info('concatenating actual and estimated volatilities...')
    volatility_vector_roots_array = np.concatenate((volatility_array_reshaped, vector_roots_array), axis=1)

    log.info('concatenating all data to turn into dataframe...')
    call_price_sx_rate_maturity_volatility_vector_roots_array =\
        np.concatenate((call_price_sx_rate_maturity_array, volatility_vector_roots_array), axis=1)

    log.info('creating data frame...')
    df = pd.DataFrame(data=call_price_sx_rate_maturity_volatility_vector_roots_array,
                      index=range(num_data_pts),
                      columns=['CALL_PRICE_RATIO', 'STOCK_PRICE', 'RISK_RATE', 'TIME_MATURITY',
                               'VOLATILITY', 'ESTIMATED_VOLATILITY'])
    df = df.replace(-10, 'NaN')

    volatility_array = volatility_array[(vector_roots >= 0) & (~np.isnan(vector_roots))]
    vector_roots = vector_roots[(vector_roots >= 0) & (~np.isnan(vector_roots))]

    log.info('saving data frame, as is, to csv file. It may include NaN on target...')
    df.to_csv('../data/output/exact_and_estimated_volatilities.csv')

    if remove_na:
        log.info('Removing nas from dataframe')
        num_na = sum(df['ESTIMATED_VOLATILITY'] == 'NaN')
        log.info('Number of nas removed=%i', num_na)
        df = df[df['ESTIMATED_VOLATILITY'] != 'NaN']

    log.info('saving data frame, without NaN on target, to csv file...')
    df.to_csv('../data/output/exact_and_estimated_volatilities_prepared.csv')

    if generate_plot:
        actual_values = df['VOLATILITY'].values
        estimated_values = df['ESTIMATED_VOLATILITY'].values
        delta_values = estimated_values - actual_values
        line_start = actual_values.min()
        line_end = actual_values.max()
        plot_title_string = 'Brents estimated vs. actual volatilities - ' +\
                            str(num_data_pts) + ' points'
        plt.title(plot_title_string)
        plt.scatter(x=actual_values, y=estimated_values)
        plt.plot([line_start, line_end],
                 [line_start, line_end],
                 'k-', color='r')
        plt.xlabel('Actual')
        plt.ylabel('Estimated')
        # save the recently populated array of images to file
        plt.savefig('../plots/volatility/brents_method/brents_estimated.png')
        # clears plot to avoid overlap with future plots coming from future calls
        plt.clf()

    dict_df_sigma_roots = dict()
    dict_df_sigma_roots['vector_roots'] = vector_roots
    dict_df_sigma_roots['data_frame'] = df

    delta_volatility = vector_roots - volatility_array
    mse_volatility = np.sqrt(np.dot(delta_volatility, delta_volatility)/df.shape[0])

    log.info('MSE volatility = %s', str(mse_volatility))

    return dict_df_sigma_roots
