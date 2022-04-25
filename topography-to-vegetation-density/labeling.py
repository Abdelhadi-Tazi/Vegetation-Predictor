
import numpy as np
L = 32
SEUILS = [i/L for i in range(L+1)]

def find_label(pixel_value):
    def dichotomy(i, j):
        if pixel_value < SEUILS[i+1]:
            return i
        m = (i+j)//2
        if pixel_value >= SEUILS[m]:
            return dichotomy(m, j)
        else:
            return dichotomy(i, m)
    #return dichotomy(0, len(SEUILS) - 1)

    return int(pixel_value*L)

def apply_labels(image):
    labels = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            labels[i, j] = find_label(image[i, j])

    return labels

def invert_labels(labels):
    return [SEUILS[l] for l in labels]

