from typing import Tuple

import tensorflow as tf
from utils.logger import get_log_object

''' Module where DL/TF models are defined'''

# instantiate log
log = get_log_object()


def get_tf_version():
    log.info('TensorFlow version=%s', tf.__version__)


def regression_model_01(input_shape_) -> tf.keras.models.Sequential:
    model_ = tf.keras.models.Sequential([tf.keras.layers.Dense(30,
                                                               activation="relu",
                                                               input_shape=input_shape_),
                                         tf.keras.layers.Dense(1)
                                         ])

    return model_


def regression_model_02(input_shape_) -> tf.keras.models.Sequential:
    model_ = tf.keras.models.Sequential([tf.keras.layers.Dense(100,
                                                               activation="relu",
                                                               input_shape=input_shape_),
                                         # hidden layer with relu activation and 100 hidden units,
                                         tf.keras.layers.Dense(50, activation='relu'),
                                         # hidden layer with relu activation function and 50 hidden units
                                         tf.keras.layers.Dense(25, activation='relu'),
                                         # hidden layer with relu activation function and 100 hidden units
                                         tf.keras.layers.Dense(50, activation='relu'),
                                         tf.keras.layers.Dense(1),
                                         ])
    return model_


def regression_model_03(input_shape_) -> tf.keras.models.Sequential:
    model_ = tf.keras.models.Sequential([tf.keras.layers.Dense(100,
                                                               activation="sigmoid",
                                                               input_shape=input_shape_),
                                         # hidden layer with relu activation and 100 hidden units,
                                         tf.keras.layers.Dense(50, activation='relu'),
                                         # hidden layer with relu activation function and 50 hidden units
                                         tf.keras.layers.Dense(25, activation='relu'),
                                         # hidden layer with relu activation function and 100 hidden units
                                         tf.keras.layers.Dense(50, activation='relu'),
                                         tf.keras.layers.Dense(1),
                                         ])
    return model_


def regression_model_04(input_shape_) -> tf.keras.models.Sequential:
    model_ = tf.keras.models.Sequential([tf.keras.layers.Dense(100,
                                                               activation="relu",
                                                               input_shape=input_shape_),
                                         # hidden layer with relu activation and 100 hidden units,
                                         tf.keras.layers.Dense(50, activation='relu'),
                                         # hidden layer with relu activation function and 50 hidden units
                                         tf.keras.layers.Dense(25, activation='relu'),
                                         # hidden layer with relu activation function and 100 hidden units
                                         tf.keras.layers.Dense(50, activation='relu'),
                                         tf.keras.layers.Dense(1),
                                         ])
    return model_


def regression_model_05(input_shape_) -> tf.keras.models.Sequential:
    model_ = tf.keras.models.Sequential([tf.keras.layers.Dense(100,
                                                               activation="relu",
                                                               input_shape=input_shape_),
                                         # hidden layer with relu activation and 100 hidden units,
                                         tf.keras.layers.Dense(60, activation='relu'),
                                         # hidden layer with relu activation function and 50 hidden units
                                         tf.keras.layers.Dense(30, activation='relu'),
                                         tf.keras.layers.Dropout(0.2),
                                         # hidden layer with relu activation function and 30 hidden units
                                         tf.keras.layers.Dense(15, activation='relu'),
                                         # hidden layer with relu activation function and 50 hidden units
                                         tf.keras.layers.Dense(30, activation='relu'),
                                         tf.keras.layers.Dropout(0.2),
                                         # hidden layer with relu activation function and 100 hidden units
                                         tf.keras.layers.Dense(100, activation='relu'),
                                         tf.keras.layers.Dense(1),
                                         ])
    return model_
