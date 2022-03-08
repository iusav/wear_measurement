# *Wear measurement*

### Requirements for the important packages
------
------
* keras: Version 2.8.0
* numpy: Version 1.19.5 
* pillow: Version 8.4.0
* python: Version 3.8.10
* matplotlib: Version 3.5.1
* scikit-learn: Version 1.0.2
* tensorflow-gpu: Version 2.4.0 
or
* tensorflow: Version 2.4.0 
* opencv-python: Version 4.5.4.58


### Explanation
------
------
##### Camera implementation script
In this script the camera is calibrated and the data set for image analysis is created.

##### Tool classification script
In the script tool classification the neural network is trained to classify the tool. In the end, the trained model is obtained and the tools can be classified.

##### Wear measurement script
In the Wear measurement script, the recorded data set can be extended for training. Further the network can be trained. At the end of the training, the pre-trained model for Semantic Segmentation is obtained. After that the wear of the parts can be measured.

### Usage
------
------
##### Setup
For comfortable work install ConEmu https://conemu.github.io/ for Powershell. You can two tabs in one with hotkeys open 
-   Strg + Shift + e
    or
-   Strg + Shift + o

1. Create a virtual environment
```sh
python -m venv env_wear_measurement
```

2. Activate the virtual environment
```sh
./env_wear_measurement/Scripts/activate
```

3. Install python packages
```sh
pip install keras==2.8.0
pip install numpy==1.19.5 
pip install pillow==8.4.0
pip install matplotlib==3.5.1
pip install scikit-learn==1.0.2
pip install tensorflow-gpu==2.4.0 
pip install opencv-python==4.5.4.58
```

4. Clone the project from https://github.com/iusav/wear_measurement
```sh
git clone https://github.com/iusav/wear_measurement.git
```
If you want to clone a specific branch run, e.g.    
```sh
git clone https://github.com/iusav/wear_measurement.git -b fabian_kohnle
```

5. Install VS Code for development
6. Open the "wear_measurement" project on VS Code

### Python Script
------
------
##### **Camera implementation script**

##### *Kalibrierung.py*
* Access camera
* Crop the window to 960x960 pixels
* When starting the program, the window with the camera section opens
* Place the "Calibration Target" of the Dino-Lite microscope on the wear surface.
* Take a picture with the "k" key
* Mark distance (keep mouse button pressed)
* Output the number of pixels between the distance
* To close all windows press "Esc" for at least 1 sec.
```sh
$ python Kalibrierung.py
```

##### *Verschleißmessung_mit_Kamera.py*
```sh
$ python Verschleißmessung_mit_Kamera.py
```

##### *Werkzeugklassifizierung_mit_Kamera.py*
* Enter value from calibration.py in `calibration= ...`.
* Load stored (trained) model
```
model = tf.keras.models.load_model('Name')
```
* Image section opens
* Capture image with "k
* Evaluation appears (wear surface is displayed in the console)
* When closing the evaluation a new window appears where it is possible to measure the wear mark width manually, if wanted
* Closing the window leads to a new start of the measurements (end of the wear measurement by holding down the Esc key)
```sh
$ python Werkzeugklassifizierung_mit_Kamera.py
```

##### **Tool classification script**
##### *Werkzeugklassifizierung_training.py*
* Zugriff auf den Überordner (z.B. "Tools")
```
data_dir = r"C:/.../.../Tools"
```
* Programm weiter ausführen (Training)
```sh
$ python Werkzeugklassifizierung_training.py
```
* Modell speichern
```
model.save('Modellname')
```

##### *Werkzeugklassifizierung_auswertung.py*
* Dateipfad des Bildes eingeben (Enter Filepath:)
* gespeichertes Modell laden
```
loaded_model = tf.keras.models.load_model('Modellname')
```
* Auswertung der Prediction (Plot)
```sh
$ python Werkzeugklassifizierung_auswertung.py
```

##### **Wear measurement script**
##### *Data_Augmentation_Images.py*
* Laden der Bilder aus dem Ordner, in dem die Originalbilder liegen
```
folder = "C:/.../.../Images"
```
* Festlegen des Ordners, indem die erweiterten Bilder gespeichert werden sollen
```
copy_to_path = "C:/.../.../Augmented_Images"
```
* Data Augmentation (Drehen um 10°, Drehen um 180°, Drehen um -10°, Gaussian Blur, Zoom um 20%, Kontrast erhöhen um 40%, Helligkeit erhöhen umd 20%)
```sh
$ python Data_Augmentation_Images.py
```

##### *Data_Augmentation_Masks.py*
* Selbes vorgehen wie bei Data_Augmentation_Masks.py
* Data Augmentation ohne Gaussian Blur, Kontrasterhöhung und Helligkeitserhöhung
```sh
$ python Data_Augmentation_Masks.py
```

##### *Bilder_laden_und_speichern.py*
* Programm lädt die Bilder aus den Ordnern in richtiger Reihenfolge und speichert diese als numpy-Array ab.

```
np.save('image_dataset', image_dataset)
np.save('mask_dataset', mask_dataset)
```

* Überprüfung, ob die Bilder zu den Masken passen (Plot)
```sh
$ python Bilder_laden_und_speichern.py
```

##### *U_Net_Model.py*
* Definieren des U-Net Modells

##### *Training_Semantic_Segmentation.py*
* Laden der Erweiterten Datenbasis als Numpy Arrayimage_dataset = np.load('image_dataset.npy')
  mask_dataset = np.load('mask_dataset.npy')
* U-Net Modell laden
```
def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
```
* Modell Trainieren
```sh
$ python Training_Semantic_Segmentation.py
```
* Ergebnisse des Modells speichern
```
model.save('Name')
```
* Auswerten des Trainingsverlaufes (Accuracy/Loss)
* IoU-Score berechnen

##### *Auswertung_Semantic_Segmentation.py*
* Gespeichertes Modell laden
```
model = tf.keras.models.load_model('Name')
```
```sh
$ python Auswertung_Semantic_Segmentation.py
```
* Auszuwertendes Bild laden
```
test_img_other = cv2.imread("C:/.../.../Verschleissbild")
```
