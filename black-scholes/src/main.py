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


def generate_synthetic_df(num_data_points: int,
                          type_artificial_data: str = 'realistic_like'):

    # Generate dictionary with ranges of parameters
    dictionary_ranges = get_dictionary_ranges(type_range=type_artificial_data)

    log.info('Latin Hypercube Sampling with multi-dimensional uniformity...')
    #number_epochs = 60
    options_from_lhs = latin_hypercube_sampling(number_sample_points=num_data_points,
                                                dict_range_for_each_variable=dictionary_ranges)

    log.info('Calculating call price ratios for each option LHS...')

    call_price_ratio_values = cPR(options_from_lhs)

    dictionary_sigma_roots_data_frame =\
        solve_for_volatility_vectorized(arg_array=options_from_lhs,
                                        call_price_vector=call_price_ratio_values)

    log.info('keys of dictionary=%s', dictionary_sigma_roots_data_frame.keys())

    log.info('Final number of computed volatilities=%i',
             dictionary_sigma_roots_data_frame['vector_roots'].shape[0])

    # get data frame
    df_volatilities = dictionary_sigma_roots_data_frame['data_frame']

    return df_volatilities


def generate_train_test_data(data_frame, list_features, target_name='VOLATILITY',
                             include_validation_set=False):
    # to test split
    dictionary_datasets = clean_split_df(data_frame=data_frame,
                                         list_features=list_features,
                                         target_name=target_name,
                                         include_validation_set=include_validation_set)

    # features and targets
    x_train_array = dictionary_datasets['train']['features']
    y_train_array = dictionary_datasets['train']['target']
    x_test_array = dictionary_datasets['test']['features']
    y_test_array = dictionary_datasets['test']['target']

    # dictionary with data
    dictionary_data = dict({'train_features': x_train_array,
                            'train_target': y_train_array,
                            'test_features': x_test_array,
                            'test_target': y_test_array})

    return dictionary_data


if __name__ == '__main__':

    number_data_points = 10000
    type_artificial_data = 'realistic_like'
    number_epochs = 60

    log.info('Executing code for %s, with %i number of points and %i epochs',
             type_artificial_data, number_data_points, number_epochs)

    # start timer
    start_time = timer()

    log.info('generate data frame with selected number of points')
    df = generate_synthetic_df(num_data_points=number_data_points,
                               type_artificial_data=type_artificial_data)

    # check final time
    end_time = timer()
    log.info('Time taken to generate data (second)=%s', end_time - start_time)

    log.info('generate dictionary of datasets')
    dict_datasets = generate_train_test_data(data_frame=df,
                                             list_features=['CALL_PRICE_RATIO',
                                                            'STOCK_PRICE',
                                                            'RISK_RATE',
                                                            'TIME_MATURITY'])

    log.info('collecting training and test set from dictionary of data')
    x_train = dict_datasets['train_features']
    y_train = dict_datasets['train_target']
    x_test = dict_datasets['test_features']
    y_test = dict_datasets['test_target']

    input_shape = x_train.shape[1:]

    log.info('creating dictionary of models')
    dictionary_models = dict({'model 01': regression_model_01(input_shape_=input_shape),
                              'model 02': regression_model_02(input_shape_=input_shape),
                              'model 03': regression_model_05(input_shape_=input_shape)})

    log.info('Looping over DL models...')
    for model_name in dictionary_models.keys():
        log.info('***************************************************')
        log.info('Training model %s ', model_name)
        model_x = dictionary_models[model_name]
        name_model = model_name + ' ' + type_artificial_data + ' ' + str(number_data_points) + ' ' + 'points'
        log.info('About to evaluate model %s ', name_model)
        start_time = timer()
        evaluate_regression_model(model_=model_x,
                                  model_name=name_model,
                                  train_set=[x_train, y_train],
                                  test_set=[x_train, y_train],
                                  num_epochs=number_epochs,
                                  location_plot='../plots/volatility/deep_learning/')
        end_time = timer()
        log.info('Model %s evaluated', name_model)
        log.info('Time to evaluate model %s took (second)=%s', name_model, end_time - start_time)
