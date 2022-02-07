##############################################
###### Auswertung_Semantic_Segmentation ######
##############################################

##########FK Version: 10.01.2022##############

import tensorflow as tf
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


#Speed up:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Bildergröße
SIZE=512

# Trainiertes Modell laden
model = tf.keras.models.load_model('Verschleiss_270_Epochs_Color')

# Bild zur Auwertung laden
#####test_img_other = cv2.imread("Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Database/Semantic_Segmentation/Test/Vergleich_Programm_7_2.jpg")
test_img_other = cv2.imread("/home/anton/Programming/HIWI_wbk_German/wear_measurement/wear_measurement/Testbilder/Validaion53.jpg")
test_img_other = Image.fromarray(test_img_other)
test_img_other = test_img_other.resize((SIZE, SIZE))
test_img_other_input =np.array(test_img_other)

# Um IoU-Score zu bestimmen muss eine Vorgefertigte Maske hochgleaden werden, mit der das Ergebnis verglichen werden kann
# Auskommentiert, weil es keine Masken für die Testbilder gibt
"""label_img_other = cv2.imread("Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Database/Semantic_Segmentation/Test/Vergleich_Programm_7_2_Mask.tiff", 0)
label_img_other = Image.fromarray(label_img_other)
label_img_other = label_img_other.resize((SIZE, SIZE))
label_img_other_norm = np.expand_dims(normalize(np.array(label_img_other), axis=1),2)"""


print(test_img_other_input.shape)

# Auswertung mit model.predict()
prediction_other = (model.predict(np.expand_dims(test_img_other_input, axis=0))[0,:,:,0] > 0.2).astype(np.uint8)


"""# IoU Bestimmen
y_pred  = prediction_other
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(label_img_other, y_pred_thresholded)
union = np.logical_or(label_img_other, y_pred_thresholded)
iou_score1 = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score1)"""

# Ergebnisse Darstellen
plt.figure(figsize=(16, 8))
plt.subplot(1, 3, 1)
plt.title('External Image')
plt.imshow(test_img_other)
"""plt.subplot(1, 3, 2)
plt.title('IoU score: ' + str(round(iou_score1, 3)))
plt.imshow(label_img_other)"""
"""plt.imshow(prediction_other, alpha=0.4)"""
plt.subplot(1, 3, 3)
plt.title('Result')
plt.imshow(test_img_other)
plt.imshow(prediction_other, alpha=0.4)
plt.show()
