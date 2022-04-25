import numpy as np
import tensorflow as tf
CHARACTERISTIC_STD = 75.0
CHARACTERISTIC_MEAN = 530.0
def center_and_reduce(dataset):

    dataset = dataset.map(lambda x, y: (x-tf.math.reduce_mean(x), y))
    dataset = dataset.map(lambda x, y: (x/tf.math.reduce_std(x), y))
    return dataset
    
def standard_normalization(dataset):
    dataset = dataset.map(lambda x, y: ((x-CHARACTERISTIC_MEAN)/CHARACTERISTIC_STD, y))
    return dataset

