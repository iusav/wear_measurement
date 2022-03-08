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

import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from U_Net_Model import simple_unet_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definieren der paths
image_directory = 'Datenbasis/Images/'
mask_directory = 'Datenbasis/Masks/'


# Definieren der Bildergröße
SIZE = 512

# Laden der Bilder als numpy Array
def get_data(image_directory, mask_directory):
    image_dataset = []
    mask_dataset = []
    for img_name in img_name_list:
        name = img_name.split('_')[-1].split('.')[0]

        img_path = image_directory+'Image_'+name+'.tif'
        mask_path = mask_directory+'Image_'+name+'.tif'

        img = cv2.imread(img_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        resized_img = resized_img.astype(np.float32) / 255.


        mask = cv2.imread(mask_path); mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        th, th_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY);
        resized_mask = cv2.resize(th_mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask.reshape(resized_mask.shape[0],resized_mask.shape[1],1)
        resized_mask = resized_mask.astype(np.float32) / 255.

        image_dataset.append(resized_img)
        mask_dataset.append(resized_mask)

    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)
    
    return image_dataset, mask_dataset


if __name__ == '__main__':
    img_name_list = sorted(os.listdir(image_directory))
    print('Dataset size: ',len(img_name_list))

    X, y = get_data(image_directory, mask_directory)
    
    # Überprüfung der Größe der Datensätze
    print('Image dataset size: ',X.shape)
    print('Mask dataset size: ',y.shape)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

    # Überprüfung, ob Bilder zu Masken passen zur Sicherheit
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

    IMG_HEIGHT = X.shape[1]
    IMG_WIDTH  = X.shape[2]
    IMG_CHANNELS = X.shape[3]

    # U-Net Modell laden
    model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    gen = ImageDataGenerator(rotation_range=25, width_shift_range=10, height_shift_range=10, zoom_range=0.3, horizontal_flip=True)

    batch_size=16
    history = model.fit_generator(gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True), epochs=50,
                        
                        steps_per_epoch=X_train.shape[0]//batch_size, # number of images comprising of one epoch
                        validation_data=(X_valid, y_valid), # data for validation
                        validation_steps=X_valid.shape[0]//batch_size
                    
                    )

    # Speichern des Ergebnisses
    model.save('Verschleiss_270_Epochs_Color')

    ############################################################

    # Modell Auswerten
    _, acc = model.evaluate(X_valid, y_valid)
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
    y_pred=model.predict(X_valid)
    y_pred_thresholded = y_pred > 0.5

    intersection = np.logical_and(y_valid, y_pred_thresholded)
    union = np.logical_or(y_valid, y_pred_thresholded)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU socre is: ", iou_score)