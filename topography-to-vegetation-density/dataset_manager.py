import os
from unittest import skip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
def load_images(dir, max_images):
    img_paths = [os.path.join(dir,filename)for filename in os.listdir(dir)][:max_images]
    img_list = [np.load(path) for path in img_paths]
    i = 0
    return img_list

def load_dataset(input_dir, output_dir, desired_shape, max_images):
    input_img_list = load_images(input_dir, max_images)
    output_img_list = load_images(output_dir, max_images)
    input_img_list = [np.reshape(cv2.resize(img, desired_shape), (desired_shape[0], desired_shape[1], 1)) for img in input_img_list]
    output_img_list = [np.reshape(cv2.resize(np.uint8(img), desired_shape), (desired_shape[0], desired_shape[1], 1)) for img in output_img_list]
    print(len(output_img_list))

    input_tensor = tf.constant(input_img_list)
    output_tensor = tf.constant(output_img_list)
    return tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))

def split_data(dataset, split_size):
    split_size = int(split_size)
    train_dataset = dataset.take(split_size)
    validation_dataset = dataset.skip(split_size)
    return train_dataset, validation_dataset

def print_dataset_info(dataset):
    
    print("Number of images :" + dataset.cardinality())
    
