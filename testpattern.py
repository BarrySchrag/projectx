# A class which creates and updates a test pattern
# Update function acts on the test patterns in the chosen manner
# Scenarios for the TestPatternGenerator:
# It will manage updates to 1 or more of images, through morphological translations
# https://en.wikipedia.org/wiki/Transformation_matrix#/media/File:2D_affine_transformation_matrix.svg

# Translation by vector: scale, direction, step, offsetx, offsety
# pattern_generator = TestPatternGenerator(
# 			('Image1.jpg', 'translate', 1.0, 0.0, 1, 250, 250),
# 			('Image2.jpg', 'translate', 0.2,-90.0, 1))
# output_image = pattern_generator.Update()
#
# Scale about origin by
# Rotation by angle:  action, scale, direction, step, (x,y)
# pattern_generator = TestPatternGenerator(
# 			('Image1.jpg', 'translate',	1.0, 0.0, 1),
# 			('Image2.jpg', 'rotate', 	1.5, -1.0, 1))

import time
import threading
import sys
import os
from math import sqrt, acos, pi
import cv2
import numpy as np
import imutils
from perpetualTimer import *

def createTransforms(offset):
    R = np.float32 ( [[1, 0, offset], [0, 1, 0]] )
    L = np.float32 ( [[1, 0, -offset], [0, 1, 0]] )
    U = np.float32 ( [[1, 0, 0], [0, 1, -offset]] )
    D = np.float32 ( [[1, 0, 0], [0, 1, -offset]] )
    return R, L, U, D

class TestPatternGenerator():

    transform = None

    def __init__(self, image, width, height, scale, direction, msec_step, offsetx, offsety):

        self.center = None
        self.width = width
        self.height = height
        self.bg = None
        self.msec_step = msec_step
        self.offsetx = offsetx
        self.offsety = offsety
        self.rlock = threading.RLock()
        self.fps = -1.0

        if image is None:
            self.bg = cv2.imread ( 'background_image.jpg', 0 )
        else:
            self.bg = cv2.imread(image, 0)

        # resize
        self.bg = imutils.resize(self.bg, self.width)
        self.height,self.width = self.bg.shape

        self.transform = createTransforms(1)

        self.timer = perpetualTimer(self.msec_step, self.update)
        # cv2.imshow ( "Test Pattern", self.bg )
        # cv2.moveWindow ( "Test Pattern", self.width, self.height )
        # cv2.waitKey ( 1 )

    def start(self):
        self.timer.start()

    def cancel(self):
        self.timer.cancel()

    def getImage(self):
        return self.bg.copy(), self.fps
    # def showWindowHelper(self):
    #     cv2.imshow ( "Test Pattern", self.bg )
    #     cv2.waitKey ( 1 )

    def update(self):
        with self.rlock:
            self.timer = cv2.getTickCount ()
            # extract the right most column
            bgrightcol = self.bg[0:self.height, self.width - 1:self.width]  # [y:y+h, x:x+w]

            # warp the image right
            self.bg = cv2.warpAffine ( self.bg, self.transform[0], (self.width, self.height) )

            # copy the column to the left side
            self.bg[0:self.height, 0:1] = bgrightcol

            self.fps = cv2.getTickFrequency () / (cv2.getTickCount () - self.timer)

if __name__ == "__main__":

    tpg = TestPatternGenerator('./media/testpattern1.jpg',640, 480,1.0,0.0,1.0
                               ,0,0)
    tpg.start()

    while True:
        img, fps = tpg.getImage()
        print(fps)
        cv2.imshow ( "Test Pattern", img )
        key = cv2.waitKey ( 1 )

        if key == ord ( "q" ) or key == 27:
            break

    tpg.cancel()
    cv2.destroyAllWindows ()
    sys.exit ()