# to split data into training and test
from sklearn.model_selection import train_test_split
# to standardize data
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.logger import get_log_object

log = get_log_object()


def clean_split_df(data_frame, list_features: list, target_name: str,
                   fraction_test: float = 0.30, ignore_missing: bool = True,
                   scale_features: bool = True, include_validation_set: bool = True) -> dict:

    """
    Function to do process dataframe and split into training, validation and test
    :param data_frame: dataframe containing all data
    :param list_features: list - contains the names of features used to train model
    :param target_name: string  - captures the name of target variable
    :param fraction_test: float - fraction of records used for test
    :param ignore_missing: boolean - If True, removes records with missing values in target,
                                     keeps them otherwise
    :param scale_features: boolean - If True, does scaling on features, otherwise it does not
    :param include_validation_set: boolean - If Tue, does extra split to generate validation set;
                                         generates training and test otherwise
    :return: dict_data_sets - dictionary containing datasets
                               keys - train, validation and test
                               values - tuples (x, y) for each set: training, validation and test
    """

    log.info('Initializing dictionary for datasets')
    dict_data_sets = dict()

    if ignore_missing:
        log.info('Removing records with missing values on target')
        data_frame = data_frame[data_frame[target_name].notna()]

    log.info('Extracting features and values from data frame...')
    x_features = data_frame[list_features].values
    y_target = data_frame[target_name].values

    if include_validation_set:
        log.info('Splitting data into training, validation and test...')
        x_train_val, x_test, y_train_val, y_test =\
            train_test_split(x_features, y_target, test_size=fraction_test, random_state=2020)

        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                          test_size=0.25,
                                                          random_state=2020)

        if scale_features:
            log.info('Scaling features and populating dictionary with datasets...')
            scale_ = StandardScaler()
            x_train = scale_.fit_transform(x_train)
            x_val = scale_.transform(x_train)
            x_test = scale_.transform(x_train)
            dict_data_sets['train'] = dict({'features': x_train, 'target': y_train})
            dict_data_sets['validation'] = dict({'features': x_val, 'target': y_val})
            dict_data_sets['test'] = dict({'features': x_test, 'target': y_test})

        else:
            log.info('Populating dictionary with datasets...')
            dict_data_sets['train'] = dict({'features': x_train, 'target': y_train})
            dict_data_sets['validation'] = dict({'features': x_val, 'target': y_val})
            dict_data_sets['test'] = dict({'features': x_test, 'target': y_test})

    else:
        log.info('Splitting data into training and test...')
        x_train, x_test, y_train, y_test =\
            train_test_split(x_features, y_target, test_size=fraction_test, random_state=2020)

        if scale_features:
            log.info('Scaling features and populating dictionary with datasets...')
            scale_ = StandardScaler()
            x_train = scale_.fit_transform(x_train)
            x_test = scale_.transform(x_train)
            dict_data_sets['train'] = dict({'features': x_train, 'target': y_train})
            dict_data_sets['test'] = dict({'features': x_test, 'target': y_test})

        else:
            log.info('Populating dictionary with datasets...')
            dict_data_sets['train'] = dict({'features': x_train, 'target': y_train})
            dict_data_sets['test'] = dict({'features': x_test, 'target': y_test})

    return dict_data_sets


'''
    epochs_ = 32

    for model_ in dictionary_models.keys():
        log.info('Evaluating model=%s', model_)
        model_name = model_
        model_object = dictionary_models[model_name]

        if model_name == 'model_03':
            utils.evaluate_regression_model_deep_wide(model_=model_object,
                                                      model_name=model_name,
                                                      train_set=[X_train, y_train],
                                                      test_set=[X_valid, y_valid],
                                                      num_epochs=epochs_,
                                                      location_plot='plots/california_housing/')
        else:
            utils.evaluate_regression_model(model_=model_object,
                                            model_name=model_name,
                                            train_set=[X_train, y_train],
                                            test_set=[X_valid, y_valid],
                                            num_epochs=epochs_,
                                            location_plot='plots/california_housing/')

        log.info('Evaluation of model has been completed=%s', model_)

'''