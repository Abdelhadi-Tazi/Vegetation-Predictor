"""
    Script pour traiter la densité brut (application moyenne + application seuil)
"""
import numpy as np
import os
from scipy import signal
import labeling 

def good_num(i):
    if i <10 :
        return "000"+ str(i)
    if i <100 :
        return "00"+ str(i)
    if i >=100 and i <1000:
        return "0" + str(i)
    if i>=1000 :
        return str(i)

target_dir = 'dbAUDE1/dens/'  #chemin vers la densité brute
target_img_paths = [os.path.join(target_dir, filename) for filename in os.listdir(target_dir)]

print("Number of images : ", len(target_img_paths))
print("The target images' paths are:", target_img_paths)

N = len(target_img_paths)
kernel_size = 10           #longueur, en pixels, du carré sur lequel on moyenne
kernel = np.ones((kernel_size,kernel_size))
compteur = 0
for i in range(N):
    vege_brute = np.load(target_img_paths[i])
    vege_moy = signal.convolve2d(vege_brute,
                                  kernel,
                                  mode='same')
    vege_moy = vege_moy / (kernel_size**2)
    mask = labeling.apply_labels(vege_moy)
    string = "./dbAUDE1/dens_seuil/" + good_num(compteur) + ".npy"
    with open(string, "wb") as f:
        np.save(f, mask)
    print(compteur)
    compteur += 1
print("Le nombre de végétations brutes traitées est de :", compteur)

