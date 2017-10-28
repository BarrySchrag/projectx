# python demo0.py -a ./media/0146.bmp -b ./media/0147.bmp
# python demo0.py -a ./media/file1.jpg -b ./media/file2.jpg

# Demonstrates finding correlating points in 2 seperate images which are similar but not identical
# DONE Utilize a modern binary descriptor and a matcher, to find coorelated matching points in 2 images sorted by relevance.

import argparse
import cv2
import matplotlib.pyplot as plt
import imutils 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-a", "--imagea", required=True, help="Path to the image1")
ap.add_argument("-b", "--imageb", required=True, help="Path to the image2")
args = vars(ap.parse_args())

img1 = cv2.imread(args["imagea"],1) # queryImage
img2 = cv2.imread(args["imageb"],1) # trainImage

imutils.resize(img1, width=400)
imutils.resize(img2, width=400)

# Initiate
key_descriptors = cv2.ORB_create()
#key_descriptors = cv2.BRISK_create()
#key_descriptors = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT)

# find the keypoints and descriptors
kp1, des1 = key_descriptors.detectAndCompute(img1,None)
kp2, des2 = key_descriptors.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:32], None, flags=2)
plt.axis("off")
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)),plt.show()


