# customized module for logging
from utils.logger import get_log_object
# module for Latin-Hypercube-Sampling
from utils.sampler import latin_hypercube_sampling
# another implementation of Latin-Hypercube-Sampling
from utils.bs_calculator import callPriceRatio as cPR
# customized function to normalize data
from utils.normalizer import normalize_data
# custom utility function to get ranges of parameters to generate data
from utils.parameters_ranges import get_dictionary_ranges
# custom function to compute volatility
from utils.solvers import solve_for_volatility_vectorized
# to keep track of time for computations
from timeit import default_timer as timer
# to split data into training and test
from utils.split_process_data import clean_split_df
# to use TF2 and all customized functions associated with it
from utils.dl_models_definition import regression_model_01, regression_model_02,\
    regression_model_03, regression_model_04, regression_model_05
from utils.train_evaluate_model import evaluate_regression_model

# instantiate log object
log = get_log_object()


if __name__ == '__main__':

    # start timer
    start_time = timer()

    # Latin Hypercube Sampling with multi-dimensional uniformity
    log.info('Testing custom module for LHS...')

    # Generate dictionary with ranges of parameters
    type_artificial_data = 'narrow'
    dict_ranges = get_dictionary_ranges(type_range=type_artificial_data)

    log.info('Latin Hypercube Sampling with multi-dimensional uniformity...')
    number_data_points = 10000
    number_epochs = 60
    options_lhs = latin_hypercube_sampling(number_sample_points=number_data_points,
                                           dict_range_for_each_variable=dict_ranges)

    log.info('Calculating call price ratios for each option LHS...')

    call_price_ratios = cPR(options_lhs)

    dict_sigma_roots_data_frame =\
        solve_for_volatility_vectorized(arg_array=options_lhs, call_price_vector=call_price_ratios)

    log.info('keys of dictionary=%s', dict_sigma_roots_data_frame.keys())

    # check final time
    end_time = timer()
    log.info('Time taken for whole process (second)=%s', end_time-start_time)

    log.info('Final number of computed volatilities=%i', dict_sigma_roots_data_frame['vector_roots'].shape[0])

    # get data frame
    df = dict_sigma_roots_data_frame['data_frame']

    # to test split
    dict_datasets = clean_split_df(data_frame=df,
                                   list_features=['CALL_PRICE_RATIO', 'STOCK_PRICE',
                                                  'RISK_RATE', 'TIME_MATURITY'],
                                   target_name='VOLATILITY',
                                   include_validation_set=False)

    # features and targets
    x_train = dict_datasets['train']['features']
    y_train = dict_datasets['train']['target']
    x_test = dict_datasets['test']['features']
    y_test = dict_datasets['test']['target']

    input_shape = x_train.shape[1:]

    dictionary_models = dict({'model 01': regression_model_01(input_shape_=input_shape),
                              'model 02': regression_model_02(input_shape_=input_shape),
                              'model 03': regression_model_05(input_shape_=input_shape)})

    # TF models
    for model_name in dictionary_models.keys():
        model_x = dictionary_models[model_name]
        name_model = model_name + ' ' + type_artificial_data + ' ' + str(number_data_points) + ' ' + 'points'
        evaluate_regression_model(model_=model_x,
                                  model_name=name_model,
                                  train_set=[x_train, y_train],
                                  test_set=[x_train, y_train],
                                  num_epochs=number_epochs,
                                  location_plot='../plots/volatility/deep_learning/')
