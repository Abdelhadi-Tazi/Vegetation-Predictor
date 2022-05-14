
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import dataset_manager as dm
import display_manager as disp
import normalization as norm
import numpy as np
import cv2
import labeling


MODEL_DIRECTORY = './models/model5'
INPUT_IMAGE = './dbALPES_test/topo/topo1_1.npy' #path to the image to predict
SAVING_DIRECTORY = "./generated_cds/"
NORMALIZATION_TYPE = 'standard' #'standard' or 'car_self' or 'car_dataset' #must match the normalization of the training dataset
INPUT_SHAPE = (64,64)
OUTPUT_SHAPE = (256, 256)
DATASET_SIZE = 152

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#load the trained model
print("Loading model")
model = tf.keras.models.load_model(MODEL_DIRECTORY)

#load the image
print("Loading image")
image = np.load(INPUT_IMAGE)
image = np.reshape(cv2.resize(image, INPUT_SHAPE), (INPUT_SHAPE[0], INPUT_SHAPE[1], 1))

#Normalize the image
print("Normalizing image")
if(NORMALIZATION_TYPE == 'standard'):
    image = norm.standard_normalization_image(image)
elif(NORMALIZATION_TYPE == 'car_self'):
    image = norm.center_and_reduce_image(image)

#Predict
print("Predicting")
prediction = disp.create_mask(model.predict(tf.reshape(image, (1,INPUT_SHAPE[0],INPUT_SHAPE[1], 1))))
prediction = prediction / labeling.L
image = cv2.resize(image, OUTPUT_SHAPE)
#display
#print("Displaying")
#plt.imshow(prediction)
#plt.show()
#save
print("Saving")
#get the name of the image
image_name = INPUT_IMAGE.split('/')[-1]
#remove the extension
image_name = image_name.split('.')[0]

#save the image
np.save(SAVING_DIRECTORY + image_name + "_prediction.npy", prediction)


    
