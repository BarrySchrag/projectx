import sys
import cv2
import keras
from keras import backend
print ("Keras Version:",keras.__version__)
print ("Keras Backend:",backend._BACKEND)
print ("OpenCV Version:",cv2.__version__)
print ("Python Version:",sys.version_info)
print ("done")

