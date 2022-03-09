##############################################
########## Segmentation evaluation ###########
##############################################

##########AI Version: 09.03.2022##############

import tensorflow as tf
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


#Speed up:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Image size
SIZE=512

# Load trained model
model = tf.keras.models.load_model('Verschleiss_270_Epochs_Color')

# Load image for evaluation
test_img_other = cv2.imread("Testbilder/Validaion33.jpg")
test_img_other = Image.fromarray(test_img_other)
test_img_other = test_img_other.resize((SIZE, SIZE))
test_img_other_input =np.array(test_img_other)

print(test_img_other_input.shape)

# Evaluation with model.predict()
prediction_other = (model.predict(np.expand_dims(test_img_other_input, axis=0))[0,:,:,0] > 0.2).astype(np.uint8)


"""# IoU Bestimmen
y_pred  = prediction_other
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(label_img_other, y_pred_thresholded)
union = np.logical_or(label_img_other, y_pred_thresholded)
iou_score1 = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score1)"""

# Display results
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
