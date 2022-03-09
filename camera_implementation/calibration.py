################################################
################# Calibration ##################
################################################

##########FK Version: 10.01.2022################

import cv2
import math
from datetime import datetime


# define a video capture object
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

class DrawLineWidget(object):
    def __init__(self):
        global file
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
            #346.3697 is the length of ruler of 1mm on the picture (mag=126.0)
            #size = 1/346.3697*distance
            #print("Size of the item is {:.4f} mm".format(size))
            #self.clone = cv2.putText(self.clone, 'Size = {:.4f} mm'.format(size), org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        if event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone
    

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
            file = "Z:/Elements/Fabian/KIT/Bachelor_Arbeit/Code/Kameraimplementierung/Test_calibration.jpg".format(dt_string)
            cv2.imwrite(file, crop)
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
        
        
        

