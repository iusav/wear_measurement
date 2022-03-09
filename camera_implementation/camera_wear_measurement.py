################################################
############ Camera wear measurement ###########
################################################

##########FK Version: 10.01.2022################


import cv2
import math
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from datetime import datetime

#Speed up:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Value of the number of Pixels between two points on the Calibrarion Target
calibration = 176.34

# Load Model
model = tf.keras.models.load_model('Verschleiss_270_Epochs_Color')

# define a video capture object
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

class DrawLineWidget(object):
    def __init__(self):
        self.original_image = cv2.imread(file)
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
  
        # org
        org = (22, 55)
  
        # fontScale
        fontScale = 1
           
        # Blue color in BGR
        color = (11, 252, 27)
          
        # Line thickness of 2 px
        thickness = 2
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            # Coordinates of two points of the line
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)

            #calibration of first coin
            distance = math.sqrt(((self.image_coordinates[0][0] - self.image_coordinates[1][0]) ** 2) + ((self.image_coordinates[0][1] - self.image_coordinates[1][1]) ** 2))
            print(distance)
            #measurement for second object
            # 1mm is the Distance on the Calibration Target between two Points
            size = 1/calibration*distance
            print("Size of the item is {:.4f} mm".format(size))
            self.clone = cv2.putText(self.clone, 'Size = {:.4f} mm'.format(size), org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        if event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    
def segmentation(file):

    test_img_other = cv2.imread(file)
    test_img_other = Image.fromarray(test_img_other)
    test_img_other = test_img_other.resize((512, 512))
    test_img_other_input =np.array(test_img_other)

    print(test_img_other_input.shape)

    # Evaluate the Model
    prediction_other = (model.predict(np.expand_dims(test_img_other_input, axis=0))[0,:,:,0] > 0.2).astype(np.uint8)

    # Visualise Results
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title('External Image')
    plt.imshow(test_img_other) 
    plt.subplot(1, 2, 2)
    plt.title('Result')
    plt.imshow(test_img_other)
    plt.imshow(prediction_other, alpha=0.5)
    plt.show(block=False)
    plt.show()
    
    plt.figure()
    plt.imshow(test_img_other)
    plt.imshow(prediction_other, alpha=0.5)

    # Messen des Verschleißes
    number_of_white_pix = np.sum(prediction_other == 1)
    number_of_black_pix = np.sum(prediction_other == 0)
    print('Anzahl weißer Pixel:', number_of_white_pix)
    print('Anzahl schwarzer Pixel:', number_of_black_pix)

    Pixelgröße = float(1/calibration*960/512)  #mm
    Pixelfläche = Pixelgröße**2 #mm^2
    Verschleißfläche = number_of_white_pix*Pixelfläche  #mm^2

    print('Die Größe des Verschleißes beträgt', Verschleißfläche, 'mm^2')

if __name__ == '__main__':
    
    print("Please hold Esc key for few seconds to close the window")
    
    
    while True:
        
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        
        # Capture the video frame
        # by frame
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
            file = "Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Code/Kameraimplementierung/Test.jpg"
            cv2.imwrite(file, crop)
            segmentation(file)
            draw_line_widget = DrawLineWidget()
            cv2.imshow('image', draw_line_widget.show_image())
            
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
        
        
        
