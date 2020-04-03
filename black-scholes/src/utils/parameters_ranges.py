
def get_dictionary_ranges(type_range: str) -> dict:
    """
    Function to obtain dictionary of ranges to generate data for Black-Scholes
    :param type_range: string to select ranges.
        options are: narrow, wide

    :return: dict_range: dictionary with ranges
        of each parameter
    """

    # if selected range is 'narrow'
    if type_range == 'narrow':
        stock_price_range = (0.5, 1.5)
        risk_free_rate_r_range = (0.03, 0.08)
        volatility_sigma_range = (0.02, 0.9)
        time_maturity_tau_range = (0.3, 0.95)

    # if selected range is 'wide'
    elif type_range == 'wide':

        stock_price_range = (0.4, 1.6)
        risk_free_rate_r_range = (0.02, 0.1)
        volatility_sigma_range = (0.01, 1.0)
        time_maturity_tau_range = (0.2, 1.1)

    # if selected range is 'realistic_like'
    elif type_range == 'realistic_like':

        stock_price_range = (0.5, 2.5)
        risk_free_rate_r_range = (0.01, 0.10)
        volatility_sigma_range = (0.01, 2.5)
        time_maturity_tau_range = (0.1, 3.0)

    # if any other option; no range is returned - to-be reconsidered
    else:
        stock_price_range = None
        risk_free_rate_r_range = None
        volatility_sigma_range = None
        time_maturity_tau_range = None

    # build dictionary to be returned
    dict_range = {'stock_price_s0_k': stock_price_range, 'risk_free_rate_r': risk_free_rate_r_range,
                  'volatility_sigma': volatility_sigma_range, 'time_maturity_tau': time_maturity_tau_range}

    return dict_range
