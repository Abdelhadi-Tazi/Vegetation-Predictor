
from cProfile import label
import unet 
import display_manager as disp
import dataset_manager as dm
import normalization as norm
import tensorflow as tf
import os
import labeling

MODEL_LOADING_DIRECTORY = './models_weights/model4_weights.ckpt'
MODEL_SAVING_DIRECTORY = './models/model7'
INPUT_DIRECTORY = './dbALPES1/augmented/topo' 
OUTPUT_DIRECTORY = './dbALPES1/augmented/dens_seuil'
DATASET_SIZE = 647 
TRAIN_RATIO = 0.9 #ratio of the dataset that is used for training
IMAGES_SHAPE = (64,64)
NORMALIZATION_TYPE = 'standard' #'standard' or 'car_self' (center and reduce according to own mean and std)
EPOCHS = 500  
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
DROPOUT_PROB = 0.4
L2_REGULARIZATION = 0.0005
N_CLASSES = labeling.L

LOAD_MODEL = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("Loading dataset")
#load the dataset
dataset = dm.load_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY, IMAGES_SHAPE, DATASET_SIZE)
disp.display_head(dataset)

dataset = dataset.shuffle(DATASET_SIZE)

#dm.print_dataset_info(dataset)

#split in two components : training and validation dataset
print("Splitting dataset")
training_dataset, validation_dataset = dm.split_data(dataset, TRAIN_RATIO*DATASET_SIZE)



#apply normalization
print("Normalizing dataset with {}".format(NORMALIZATION_TYPE))
if(NORMALIZATION_TYPE == 'standard'):
    training_dataset = norm.standard_normalization(training_dataset)
    validation_dataset = norm.standard_normalization(validation_dataset)
elif(NORMALIZATION_TYPE == 'car_self'):
    training_dataset = norm.center_and_reduce(training_dataset)
    validation_dataset = norm.center_and_reduce(validation_dataset)


#display head of the training and validation dataset
print("Displaying head of the training dataset")
training_dataset = training_dataset.cache()
validation_dataset = validation_dataset.cache()

#disp.display_head(training_dataset)
#disp.display_head(validation_dataset)




#create a unet model
model = unet.unet_model(input_size = (IMAGES_SHAPE[0], IMAGES_SHAPE[1], 1), dropout_probability = DROPOUT_PROB, regularization=L2_REGULARIZATION, n_classes = N_CLASSES)
if LOAD_MODEL:
    model.load_weights(MODEL_LOADING_DIRECTORY)
model.summary()


#compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             metrics=["accuracy"])

#train the model
model_history = model.fit(training_dataset.batch(BATCH_SIZE),
                         epochs=EPOCHS,
                         use_multiprocessing='True',
                         validation_data=validation_dataset.batch(BATCH_SIZE)
                        )

#print model data
disp.plot_history(model_history)

#display predictions
disp.display_predictions(model, validation_dataset)
disp.display_predictions(model, training_dataset)
#save model
model.save(MODEL_SAVING_DIRECTORY)
