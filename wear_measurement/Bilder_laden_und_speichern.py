##############################################
######### Bilder_laden_und_speichern #########
##############################################

##########FK Version: 10.01.2022##############

import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

#Speed up:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dateipfade in dem die Masken und Augmented Bilder liegen
#####image_directory = "Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Database/Semantic_Segmentation/Images_Masks_Subdir/Images/"
#####mask_directory = "Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Database/Semantic_Segmentation/Images_Masks_Subdir/Masks/"
image_directory = "/home/anton/Programming/HIWI_wbk_German/wear_measurement/Additional_Database/Images/"
mask_directory = "/home/anton/Programming/HIWI_wbk_German/wear_measurement/Additional_Database/Masks/"


# Bestimmung der Bildergröße
SIZE = 512
image_dataset = []
mask_dataset = []

# WICHTIG: sorted(), damit Bilder mit Masken übereinstimmen
images = sorted(os.listdir(image_directory))
i=1

for i, image_name in enumerate(images):
    if (image_name.split('.')[1] == 'tif'):
        print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#Iteration durch allle gespeicherten Bilder und speichern als Numpy Array

masks = sorted(os.listdir(mask_directory))
i=1
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        print(mask_directory + image_name)
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))


image_dataset = np.array(image_dataset)
# Normalisierung der Masken
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

# Speichern als .npy-Datei
np.save('image_dataset_data_aug_5_color', image_dataset)
np.save('mask_dataset_data_aug_5_color', mask_dataset)

# Überprüfung der Größe der Bilder
print(image_dataset.shape)
print(mask_dataset.shape)

# Unterteilung in test Dataset und train Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

# Überprüfung, ob Bilder zu den Masken passen
import random
image_number = random.randint(0, len(X_train))
print(image_number)
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(X_train[image_number])
plt.subplot(132)
plt.imshow(y_train[image_number])
plt.subplot(133)
plt.imshow(X_train[image_number])
plt.imshow(y_train[image_number], alpha=0.5)
plt.show()

plt.show()