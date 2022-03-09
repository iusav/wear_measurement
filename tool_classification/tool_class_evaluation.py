##############################################
########### Tool class evaluation ############
##############################################

##########FK Version: 10.01.2022##############

#Notwendigen Packages importieren:
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
#Speed up:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Bild zum Auswerten entweder mit Link aus Internet verwenden oder Dateipfadangeben:
r = str()
###filepath = input('Enter Filepath: ')
filepath = "Test_Images/Test_012.jpg"
if filepath.startswith('http'):
    filepath = tf.keras.utils.get_file(origin=filepath)
###elif filepath.startswith('Z:'):
###    filepath = r+filepath
else:

    filepath = filepath

#Bilder größe definieren:
batch_size = 32
img_height = 512
img_width = 512

#Klassen, denen die Bilder zugeordnet werden können:
class_names = ['Schaftfräser', 'Wendeschneidplatte']

#In CNN_classification sind die Daten aus dem trainierten Modell gespeichert und werden mit tf.keras.load_model geladen:
loaded_model = tf.keras.models.load_model('CNN_classification_new_Data')

#Vorhersage des Programms:
img = tf.keras.utils.load_img(filepath, target_size=(img_height, img_width))

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = loaded_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

#Ausgabe des Ergebnisses:
print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.title("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)), fontsize=18)
plt.show()