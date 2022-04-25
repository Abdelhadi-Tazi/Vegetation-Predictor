"""
    Procède à la data augmentation à partir des chemins de la topo et des "masks" associés.

    data augmentation :
        -4 rotations de 90 degrés
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def display(display_list):
    """
    :param display_list: une liste de 3 listes [input_list, target_list, predict_list]
    :return: affichage
    """
    plt.figure(figsize=(15, 15))
    title = ['topo', 'vege brute', 'vege lissée + seuil', 'autre']
    n_rows = len(display_list[0])
    n_cols = len(display_list)  # = 1, 2 ou 3
    for line in range(n_rows):
        for col in range(n_cols):
            plt.subplot(n_rows, n_cols, col+1 + n_cols*line)
            plt.title(title[col])
            plt.imshow(display_list[col][line])
            plt.axis('off')
    plt.show()

def good_num(i):
    if i <10 :
        return "000"+ str(i)
    if i <100 :
        return "00"+ str(i)
    if i >=100 and i <1000:
        return "0" + str(i)
    if i>=1000 :
        return str(i)


input_dir = './dbALPES1/topo/'
target_dir = './dbALPES1/dens/'
input_img_paths = [os.path.join(input_dir,filename)for filename in os.listdir(input_dir)]
target_img_paths = [os.path.join(target_dir, filename) for filename in os.listdir(target_dir)]

print("Number of images : ", len(target_img_paths))
print("The input images' paths are:", input_img_paths)
print("The target images' paths are:", target_img_paths)

N = len(target_img_paths)
compteur = 0
for i in range(N):
    topo = np.load(input_img_paths[i])
    mask = np.load(target_img_paths[i])
    for j in range(4):
        topo_modif = np.rot90(topo)
        mask_modif = np.rot90(mask)
        string = "./dbALPES1/augmented/topo/" + good_num(compteur) + ".npy"         #renseigner ici le dossier où seront enregistrés les topos
        with open(string, "wb") as f:
            np.save(f, topo_modif)
        string = "./dbALPES1/augmented/dens/" + good_num(compteur) + ".npy"         #renseigner ici le dossier où seront enregistrés les masks
        with open(string, "wb") as f:
            np.save(f, mask_modif)
        print(compteur)         #pour suivre
        compteur += 1
        topo = topo_modif
        mask = mask_modif
print("La taille du dataset augmenté est de :", compteur)

