
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import dataset_manager as dm
import display_manager as disp
import normalization as norm
import numpy as np
import labeling
import unet

MODEL_DIRECTORY = './models/h5/model6.h5'
INPUT_DIRECTORY = './dbALPES_test/topo' 
OUTPUT_DIRECTORY = './dbALPES_test/dens_seuil'
NORMALIZATION_TYPE = 'standard' #'standard' or 'car_self' or 'car_dataset' #must match the normalization of the training dataset
IMAGES_SHAPE = (64,64)
DATASET_SIZE = 152

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#load the trained model
print("Loading model")
model = tf.keras.models.load_model(MODEL_DIRECTORY)
#load the test dataset
print("Loading test dataset")
test_set = dm.load_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY, IMAGES_SHAPE, DATASET_SIZE)
disp.display_head(test_set)

#Normalize the test dataset
print("Normalizing test dataset")
if(NORMALIZATION_TYPE == 'standard'):
    test_set = norm.standard_normalization(test_set)
elif(NORMALIZATION_TYPE == 'car_self'):
    test_set = norm.center_and_reduce(test_set)

#Get all predictions from the model
print("Predicting")
predictions = [disp.create_mask(model.predict(tf.reshape(x, (1,IMAGES_SHAPE[0],IMAGES_SHAPE[1], 1)))) for x, y in test_set]

#Flatten and concatenate all predictions
print("Flattening predictions")
flat_predictions = [tf.reshape(x, (IMAGES_SHAPE[0]*IMAGES_SHAPE[1])) for x in predictions]
flat_predictions = tf.concat(flat_predictions, axis=0)
print(flat_predictions)
#Flatten and concatenate all true results
true_res = [y for x, y in test_set]
print("Flattening true results")
flat_true = [tf.reshape(y, (IMAGES_SHAPE[0]*IMAGES_SHAPE[1])) for y in true_res]
flat_true = tf.concat(flat_true, axis=0)

#Calculate confusion matrix
print("Calculating confusion matrix")
confusion_matrix = np.array(tf.math.confusion_matrix(flat_true, flat_predictions))

#Calculate Accuracy, precision, recall and F1-score from the confusion matrix
print("Calculating classification metrics")
accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
f1 = 2 * precision * recall / (precision + recall)
mean_precision = np.mean([p for p in precision if p == p])
mean_recall = np.mean([r for r in recall if r == r])
mean_f1 = np.mean([f for f in f1 if f == f])

print("Delabeling predctions")
delabeled_flat_predictions = np.array(labeling.invert_labels(np.array(flat_predictions)))
print("Delabeling true results")
delabeled_flat_true = np.array(labeling.invert_labels(np.array(flat_true)))

#Calculate mse for thresholds
print("Calculating regression metrics")

rss = np.sum(np.square(delabeled_flat_predictions - delabeled_flat_true))

mean = np.mean(delabeled_flat_true)

ess = np.sum(np.square(delabeled_flat_predictions - mean))

tss = np.sum(np.square(delabeled_flat_true - mean))

#Calculate r2
r2 = ess/tss

print("#####################TEST RESULTS##########################")
print("")

print("Total number of predictions: {}".format(len(predictions)))
print("")

print("Confusion matrix :")
print(confusion_matrix)
print("")

plt.imshow(confusion_matrix)

#print metrics
print("CLASSIFICATION METRICS :")

print("Precision : {}".format(precision))
print("Recall : {}".format(recall))
print("F1-score : {}".format(f1))
print("Mean precision : {}".format(mean_precision))
print("Mean recall : {}".format(mean_recall))
print("Mean F1-score : {}".format(mean_f1))
print("Accuracy : {}".format(accuracy))
print("")

print("REGRESSION METRICS :")
print("SQRT MSE : {}".format(np.sqrt(rss/len(delabeled_flat_predictions))))
print("R2-coefficient : {}".format(r2))
disp.display_predictions(model, test_set)