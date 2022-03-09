######################################################
############# Camera tool classification #############
######################################################

##########FK Version: 10.01.2022######################


import cv2
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

#Speed up:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

calibration = 157

# Definieren eines Fensters dür die Kamera
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Werkzeuglassifizierung
def classification(file):
    #Klassen, denen die Bilder zugeordnet werden können:
    class_names = ['Schaftfräser', 'Wendeschneidplatte']

    #In CNN_classification sind die Daten aus dem trainierten Modell gespeichert und werden mit tf.keras.load_model geladen:
    loaded_model = tf.keras.models.load_model('CNN_classification')

    #Vorhersage des Programms:
    img = tf.keras.utils.load_img(file, target_size=(512, 512))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    #Ausgabe des Ergebnisses:
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    plt.show()


if __name__ == '__main__':
    
    print("Please hold Esc key for few seconds to close the window")

    while True:
        
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        
        # Capture the video frame
        # by frame
        # crop for making a squared frame
        ret, frame = vid.read()
        RGB = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        crop = RGB[0:960, 160:1120]

      
        # Display the resulting frame
        #cv2.imshow('frame', RGB)
        cv2.imshow('frame', crop) 
         
        # the 'k' button is set as the
        # picture taking button you may use any
        # desired button of your choice
        
        if cv2.waitKey(1) & 0xFF == ord('k'):
            #Speicherort:
            file = "Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Code/Kameraimplementierung/Test.jpg"
            cv2.imwrite(file, crop)
            classification(file)

            
            if cv2.waitKey(1) == 27:
                print("Escape hit, closing...")
                break
        
        # the 'esc' button is set as the
        # quitting button you may use any
        # desired button of your choice
        elif cv2.waitKey(1) == 27:
            print("Escape hit, closing...")
            break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()