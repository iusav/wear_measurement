# https://youtu.be/csFGTLT6_WQ
# https://github.com/bnsreenu/python_for_microscopists/blob/master/204_train_simple_unet_for_mitochondria.py
# """
# Author: Dr. Sreenivas Bhattiprolu
# Training and testing for semantic segmentation (Unet) of mitochondria
# Uses standard Unet framework with no tricks!
# To annotate images and generate labels, you can use APEER (for free):
# www.apeer.com
# """

##############################################
####### Training_Semantic_Segmentation #######
##############################################

##########FK Version: 10.01.2022##############

from U_Net_Model import simple_unet_model
import os
import numpy as np
from matplotlib import pyplot as plt

#Speed up:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Definieren der Bildergröße
SIZE = 512
image_dataset = []
mask_dataset = []

# Laden der Bilder als numpy Array
image_dataset = np.load('image_dataset.npy')
mask_dataset = np.load('mask_dataset.npy')

# Überprüfung der Größe der Datensätze
print(image_dataset.shape)
print(mask_dataset.shape)

# Aufteilung in train dataset und test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

# Überprüfung, ob Bilder zu Masken passen zur Sicherheit
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(X_train[image_number]) #cmap='gray'
plt.subplot(132)
plt.imshow(y_train[image_number]) #cmap='gray'
plt.subplot(133)
plt.imshow(X_train[image_number]) #cmap='gray'
plt.subplot(133)
plt.imshow(y_train[image_number], alpha=0.5)
plt.show()

###############################################################
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

print(IMG_HEIGHT)
print(IMG_WIDTH)
print(IMG_CHANNELS)

# U-Net Modell laden
def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

# Definieren der Parameter und training des Programms
history = model.fit(X_train, y_train,
                    batch_size = 16,
                    verbose=1,
                    epochs=270,
                    validation_data=(X_test, y_test),
                    shuffle=False)

# Speichern des Ergebnisses
model.save('Verschleiss_270_Epochs_Color')

############################################################

# Modell Auswerten
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


# Verlauf von training und validation Accuracy werden als Plot dargestellt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##################################

#IOU Bestimmen
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)


