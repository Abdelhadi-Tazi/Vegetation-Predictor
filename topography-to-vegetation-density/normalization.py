import numpy as np
import tensorflow as tf
CHARACTERISTIC_STD = 75.0
CHARACTERISTIC_MEAN = 530.0

def center_and_reduce_image(image):
    return (image-tf.math.reduce_mean(image))/tf.math.reduce_std(image)
    
def center_and_reduce(dataset):

    dataset = dataset.map(lambda x, y: (center_and_reduce_image(x), y))
    return dataset
   

def standard_normalization_image(image):
    return ((image-CHARACTERISTIC_MEAN)/CHARACTERISTIC_STD)

def standard_normalization(dataset):

    dataset = dataset.map(lambda x, y: (standard_normalization_image(x), y))
    return dataset



