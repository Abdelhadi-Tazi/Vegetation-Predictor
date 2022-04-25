import matplotlib.pyplot as plt
import tensorflow as tf
def plot_history(model_history):
    """
    :param model: a keras model
    :param history: a keras history object
    """
    history = model_history.history
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 1, 1)
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.subplot(2, 1, 2)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.ylim([0, 5])
    plt.legend(loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def display_examples(display_list, title_list):

    plt.figure(figsize=(15, 15))
    #title = ['topo', 'vege brute', 'vege lissée + seuil']
    n_rows = len(display_list[0])
    n_cols = len(display_list)  # = 1, 2 ou 3
    for line in range(n_rows):
        for col in range(n_cols):
            plt.subplot(n_rows, n_cols, col+1 + n_cols*line)
            plt.title(title_list[col])
            plt.imshow(display_list[col][line])
            plt.axis('off')
    plt.show()

def display_head(dataset):
    images = [image for image, mask in dataset.take(5)]
    masks = [mask for image, mask in dataset.take(5)]
    display_list = [images, masks]
    display_examples(display_list, ["Topography", "Vegetation"])

def display_5_random(dataset, size):
    d = dataset.shuffle(size).take(5)
    images = [image for image, mask in d]
    masks = [mask for image, mask in d]
    display_list = [images, masks]
    display_examples(display_list, ["Topography", "Vegetation"])


def display_predictions(model, dataset):
    images = [image for image, mask in dataset.take(5)]
    masks = [mask for image, mask in dataset.take(5)]
    predictions = [create_mask(model.predict(image[tf.newaxis, ...])) for image in images]
    display_examples([images, masks, predictions], ["Topography", "True Vegetation", "Prediction"])

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]  #ajoute un axe
    return pred_mask[0]

def display_full(dataset, dataset_size):
    for i in range(dataset_size//5):
        display_head(dataset.skip(5*i))

def display_all_predictions(model, dataset, dataset_size):
    for i in range(dataset_size//5):
        display_predictions(model, dataset.skip(5*i))
