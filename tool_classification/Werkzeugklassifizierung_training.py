###################################################
##### Image Classification Tools - Train Code #####
###################################################

##########FK Version: 08.11.2021



#import matplotlib.pyplot as plt
#import numpy as np
import os
#import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import pathlib
import matplotlib.pyplot as plt

#Speed up:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Anzeigen der verwendeten Version von Tensorflow:
#print(tf.__version__)

#data_dir ist der Dateipfad, in dem die Bilder bereit gestellt werden. Im Ordner sind weitere Ordner mit den Klassen
#Wendeschneidplatte und Schaftfräser
#####data_dir = r"Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Database/Tools"
data_dir = '/home/anton/Programming/HIWI_wbk_German/wear_measurement/tool_classification/Datenbasis'
data_dir = pathlib.Path(data_dir)

#Anzahl der Bilder ausgeben lassen:
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Anzahl der Bilder:',image_count)

#Beispiel eines Inhalts aus der Datenbasis:
wendeplatten = list(data_dir.glob('Wendeschneidplatte/*'))

#Definition Parameter für 'loader'.
#Batch_size: Anzahl der Stichproben, mit dem das Netzwerk trainiert wird. Wenn die Anzahl der Bilder 445 ist und die Batch_size 25,
# dann nimmt der Algorithmus die ersten 25 Bilder und trainiert das Netzwerk. Als nächstes werden die Bilder 26 bis 51 verwendet, bis
# alle Bilder trainiert sind.
# img:height, img_width: Parameter, um die Größe der Bilder anzupassen und zu vereinheitlichen.:
batch_size = 32
img_height = 512
img_width = 512


# data_dir: Link zum Ordner, wo die Bilder gespeichert sind. Unterordner werden als Klassen erkannt.
# labels='inferred': Label werden aus der Ordnerstruktur erstellt.
# validation_split: Anteil der Bilder für die Validierung.
# subset: Definiert, ob Datenset für das Training oder die Validierung ist.
# seed: Wenn man jedes mal das gleiche Trainingset und Validierungsset haben möchte Seed='Anfangszahl'.
# image_size: Größe der Bilder anpassen, wenn diese nicht in der Größe sind.
# batch_size: Anzahl der Stichproben
#Validation split, 80% training
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#Validation split: 20% validation:
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#Klassennamen aus der Ordnerstruktur:
class_names = train_ds.class_names
print(class_names)

#Visualisierung der Daten

#Die ersten 9 Bilter vom Trainingset werden geplottet:
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
  plt.show()

#Größe der Bilder:
for image_batch, labels_batch in train_ds:
  print('Image-Batch-shape:',image_batch.shape)
  print('Labels-Batch-shape:',labels_batch.shape)
  break

#Beschleunigung des Algorithmus durch gepuffertes Prefetching. gepuffertes Prefetching sorgt dafür, dass Daten von der
#Festplatte abgerufen werden können, ohne dass sie blockiert wird.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#RGB-Kanal-Werte normal zwischen [0, 255]. Dies ist für ein neuronales Netzwerk nicht ideal. Deshalb werden die Eingabewerte
# standardisiert, damit diese einen Bereich zwischen [0, 1] haben.
normalization_layer = tf.keras.layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]


#Model erstellen:
#Das Modell besteht aus drei Faltungsblöcken mit jeweils einer Max-Pooling-Schicht. Es gibt eine fully-connected Schicht
#mit 128 Einheiten, die durch eine ReLU-Funktion aktiviert wird.
num_classes = len(class_names)

def create_model_1():
  model = Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
  ])


#Modell kompilieren
#optimizer='adam': Optimizier, der den Adam-Algorithmus implementiert
#loss: Loss function SparseCategoricalCrossentropy: Berechnet den Crossentropy Verlust zwisachen den Labels und den Vohersagen
#metrics='accuracy': berechnet, wie oft Vorhersagen gleich Labels sind.
  model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  return model

model = create_model_1()

#Checkpoints=Kontrollpunkte. Checkpoints erfassen den exakten Wert aller Parameter, die von einem Modell verwendet werden
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
#ModelCeckpoint()=Callback, um das Keras Modell oder Modell-weights in einer Bestimmten Häufigkeit zu speichern
#filepath: Dateipfad, wo die Datei gespeichert werden soll.
#save_weights_only: Nur die Weights werden gespeichert, wenn 'True'
#verbose:
  #0: Es wird nichts dargestellt,
  #1: [======================>...] wird dargestellt
  #2: Nur 'Epoch 1/10' wird dargestellt
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


#Model trainieren:

#Epochs: Ein Zyklus durch ein komplettes training Datenset
epochs=15

#train_ds: Trainingsdatenset
#validation_data: Validierungsdatenset
#epochs: Anzahl der Epochs
#callbacks: s.o.
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cp_callback])

#Darstellung des Modells. Visualisierung der Layer
model.summary()

#Visualisierung der Trainingsergebnisse. Es fällt auf, dass Trainingsgenauigkeit und Validierungsgenauigkeit stark
#voneinander abweichen. Ursache: zu wenig Trainingsbeipiele. Dabei lernt das Modell manchmal von Rauschen oder ungewollten
#Details der Trainingsbeispiele. --> Overfitting. Lösung: Data augmentation = Datenerweiterung

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Data augmentation:

#Es werden zusätzliche Layer ins Modell eingebunden.
#RandomFlip: dreht zufällig Bilder um
#RandomRodation: dreht zufällig Bilder
#RadomZoom: Zoomt Bilder zufällig
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),])

#Darstellung der Bilder nach Datenerweiterung:
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
  plt.show()

#Dropout: Zusätzliche Technik, um Overfitting zu reduzieren. Während des Trainingsvorgangs wird eine Anzahl von Ausgabeeinheiten
#aus dem Layer zufällig ausgelassen. Der Eingabewert, z.B. 0.2 bedeutet, dass zufällig bspw. 20% der Ausgabeeinheiten aus dem
#Layer fallen.
def create_model_2():
  model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])

#Ab hier: Wiederholen der Schritte aus erstem Trainingssatz, nur mit anderem Modell.

#Modell kompilieren:
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = create_model_2()

#Checkpoint:
checkpoint_path = 'training_2/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

#Modell trainieren:
epochs = 50
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cp_callback])

#Übersicht neues Modell:
model.summary()


#Visualisierung der Ergebnisse: kein/wenig Overfitting!
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Modell Speichern
model.save('CNN_classification_new_Data')