# python demo4.py --video ./media/20171014_180142.mp4 --width 340
# python demo4.py --video ./media/00001-Dup15rVd2eU.mp4 --width 640

# Demonstrates capturing a region which has motion relevancy based on analysis of change in a dynamic scene
#
# From a video stream, pick out one or more region candidates containing significant change  DONE
#   Take each a frameN and frameN+1, apply intra-frame differencing
#   The largest value of contrast difference between frames is held in a first in last out que length ~ 3
#       In the que
#           The mean distance between each x,y point is determined
#           The mean location in x,y is determined
#           If the std deviation of the distances between points is lower than a threshold ~ 40 - execute the next step
# Tag and track regions separately, not switching between them
#   Detect the external contour which encompasses the region of change
#   Capture a bounding box of this contour, and show it on screen
# See toto.txt

# Goto Start:
import argparse
import cv2
import imutils
import sys
import numpy as np
import time
from collections import deque
import copy
from skimage.exposure import rescale_intensity
from geometry import *

def drawRectagleOnImage(image, bbox, color, thickness=1, linetype=1):
    p1 = (int (bbox[0]), int (bbox[1]))
    p2 = (int (bbox[0] + bbox[2] ), int(bbox[1] + bbox[3]))
    cv2.rectangle (image, p1, p2, color, thickness, linetype)

def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = int((kW - 1) / 2)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros ( (iH, iW), dtype="float32" )

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange ( pad, iH + pad ):
        for x in np.arange ( pad, iW + pad ):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum ()

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

            # rescale the output image to be in the range [0, 255]
            output = rescale_intensity ( output, in_range=(0, 255) )
            output = (output * 255).astype ( "uint8" )

            # return the output image
            return output

# Function to get each a,b of a series
# Use: for a, b in pairwise(que):
def pairwise(it):
    it = iter(it)
    while True:
        yield next(it), next(it)


def distance(v1, v2):
    return sum([(x-y) **2 for (x, y) in zip(v1, v2)])**0.5


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split ( '.' )

width = 0
height = 0
channels = 0
min_area = 0
points_to_detect = 0
min_radius = 0
stdev = 0
count_of_concordant_points = 0
output_to_gray = False
frame_counter_start_frame = 0

# construct a sharpening filter
sharpen = np.array((
	[1, 1, 1],
	[1, -7, 1],
	[1, 1, 1]), dtype="int")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-w", "--width", type=int, default=720, help="width to resize the image")
ap.add_argument("-p", "--points", type=int, default=32, help="points to detect in the image")
ap.add_argument("-a", "--min_area", type=int, default=1000, help="minimum area size")
ap.add_argument("-r", "--min_radius", type=int, default=2000, help="minimum radius size")
ap.add_argument("-t", "--feature_tracker_type", default="ALL",
                help="feature_tracker type one of: ALL, NONE")
ap.add_argument("-d", "--stdev_min", type=int, default=38,
                help="minimum distance between x,y points to be a valid region of interest")
ap.add_argument("-c", "--count_of_concordant_points", type=int, default=3,
                help="minimum count of concordant x,y points to be considered valid region of interest")
ap.add_argument("-f", "--frame_counter_start_frame", type=int, default=0,
                help="delay for what frame to begin with")

args = vars(ap.parse_args())

if args.get("min_area", True):
    min_area = args["min_area"]
if args.get("points", True):
    points_to_detect = args["points"]
if args.get("min_radius", True):
    min_radius = args["min_radius"]
if args.get("stdev_min", True):
    stdev_min = args["stdev_min"]
if args.get("count_of_concordant_points", True):
    count_of_concordant_points = args["count_of_concordant_points"]
if args.get("frame_counter_start_frame", True):
    frame_counter_start_frame = args["frame_counter_start_frame"]

# initialize Tracker
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = args.get("feature_tracker_type")

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 25;
params.maxThreshold = 255;

# Filter by Area.
params.filterByArea = True
params.minArea = 500

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create (params)

# initiate detector
# keypoint_detector = cv2.ORB_create()
# keypoint_detector = cv2.BRISK_create()
keypoint_detector = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# settings
calcOpticFlow = False
camera = None
previousFrame = None
frame_counter = 0
frame_count_max = 0
tracked_area = 0
kernel = np.ones((5, 5), np.uint8)
imCrop = None
bbox = None
que = deque()
previousXY = None
captures = 0
captures_max = 18
fps = 0
scalar_stdev_in_xy = 1
capture_window_x = 0
captures_to_file = 0
captures_to_file_max = 0 # 20
capture_start = 0
key = -1
trackers = cv2.MultiTracker_create()
track_found_bbox = []
frames_between_image_captures_for_tracking = 4
while True:
    # start timer
    timer = cv2.getTickCount()

    # if the last frame is reached, reset the capture and the frame_counter
    if frame_counter >= frame_count_max:
        print("Loop frame_counter:", frame_counter)
        previousFrame = None
        frame_counter = 0
        capture_window_x = 0
        captures_to_file = 0

        ok2 = False
        while captures > 0:
            trackers = cv2.MultiTracker_create()
            cv2.destroyWindow(str(captures))
            captures -=1

        # if a video path was not supplied, grab the reference to the webcam
        if not args.get("video", False):
            camera = cv2.VideoCapture(0)
        # otherwise, grab a reference to the video file
        else:
            #video_file = "./media/00001-Dup15rVd2eU.mp4"
            #---video ./media/20171014_180142.mp4
            video_file = args["video"]
            camera = cv2.VideoCapture(video_file)

        frame_count_max = camera.get(cv2.CAP_PROP_FRAME_COUNT)

    # grab the current frame
    (grabbed, frame) = camera.read()
    frame_counter += 1

    # command line paramater to delay the start of processing
    if frame_counter < frame_counter_start_frame:
        continue

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
    if args.get("width", True):
        width = args["width"]
        img = imutils.resize(frame, width)
        hsv = np.zeros_like ( img )
        hsv[..., 1] = 255
    else:
        img = frame
    img0 = img.copy()

    height, width, channels = img0.shape

    # Detect blobs.
    keypoints = None # detector.detect (img0)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #im_with_keypoints = cv2.drawKeypoints (img0, keypoints, np.array ( [] ), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)  # or 7,9,11,13,15,17,21

    # if the first frame is None, initialize it
    if previousFrame is None:
        previousFrame = gray
        continue

    if calcOpticFlow == True:
        #flow = cv2.calcOpticalFlowFarneback ( previousFrame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0 )
        flow = cv2.calcOpticalFlowFarneback ( previousFrame, thresh, None, 0.4, 1, 12, 2, 8, 1.2, 0 )

        mag, ang = cv2.cartToPolar ( flow[..., 0], flow[..., 1] )
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize ( mag, None, 0, 255, cv2.NORM_MINMAX )
        rgb = cv2.cvtColor ( hsv, cv2.COLOR_HSV2BGR )

    # compute the absolute difference between the current and last
    frameDelta = cv2.absdiff(previousFrame, gray)

    # determine min/max value location
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(frameDelta, None)

    # take the threshold of the absolute difference
    thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]

    # draw marker at max location
    cv2.drawMarker(img0, maxLoc, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    # limit the size of the queue
    if len(que) > count_of_concordant_points:
        que.popleft()

    # add the latest location to the que
    que.append(maxLoc)

    mean_x = 0
    mean_y = 0
    x = 0
    y = 0
    vector_result = (0, 0)
    num_pts = len(que)
    for xy in list(que):
        if previousXY is None:
            previousXY = xy
            continue
        # the mean distance between each x,y point is calculated
        mean_x = abs(xy[0] - previousXY[0])
        mean_y = abs(xy[1] - previousXY[1])

        # the mean location in x,y is determined
        x += xy[0]
        y += xy[1]

        # calculate vector
        # vector_result ...

        cv2.drawMarker(img0, xy, (128, 128, 128), cv2.MARKER_CROSS, 10, 1)

    # find the mean of the x, y positions
    x /= num_pts
    y /= num_pts

    # find the mean of the distances between the points
    mean_x /= num_pts
    mean_y /= num_pts

    # find the std deviation of the distances between each point
    list_x = [x[0] for x in que]
    stdev_x = np.std(list_x)
    list_y = [y[1] for y in que]
    stdev_y = np.std(list_y)

    # find the stdev in x,y
    stdev_in_xy = (stdev_x + stdev_y) / 2


    # if the std deviation is small
    if stdev_in_xy < stdev_min:
        # draw the mean center of the x,y points
        cv2.circle(img0, (int(x), int(y)), int(stdev_in_xy * scalar_stdev_in_xy), (255, 0, 0), 2)

        # close the holes
        #thresh = cv2.dilate(thresh, kernel, iterations=1)

        # roi = cv2.circle(np.zeros(thresh.shape, thresh.dtype),
        #                 (int(x), int(y)), int(stdev_in_xy * scalar_stdev_in_xy), 255, -1 )
        # thresh = cv2.bitwise_and(thresh, thresh, mask=roi)

        # find contours - returns image, contours, hierarchy
        (_, contours, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE, None, None)
    else:
        thresh = np.zeros(thresh.shape, thresh.dtype)
        contours.clear

    if len(contours) > 0 and captures < captures_max:
        # find the largest contour
        c = max(contours, key=cv2.contourArea)
        dist = cv2.pointPolygonTest(c, maxLoc, True)
        print("Dist:",dist)
        if dist > 5:
            bbox = cv2.boundingRect(c)
            proposed_object_bbox = Rect ( int ( bbox[0] ), int ( bbox[1] ), int ( bbox[2] ), int ( bbox[3] ) )
            # We have not yet tracked anything, so add it
            if trackers.getObjects() == ():
                imCropOrig = img[int ( bbox[1] ):int ( bbox[1] + bbox[3] ),
                             int ( bbox[0] ):int ( bbox[0] + bbox[2] )]
                captures += 1

                # show the region we are tracking
                cv2.imshow ( str ( captures ), imCropOrig )
                cv2.moveWindow ( str ( captures ), capture_window_x, (height + 10) * 2 )
                capture_window_x += int ( imCropOrig.shape[1] ) + 10

                # initialize a new feature tracker
                trackers.add ( cv2.TrackerMIL_create (), img, bbox )
            else:
                # Loop through all existing bounding boxes to see if this point is already in a region
                for i in np.arange(0, captures):
                    bx = trackers.getObjects()[i]
                    tracked_object_bbox = Rect( int(bx[0]), int(bx[1] ), int(bx[2] ), int(bx[3] ) )
                    # Is the point inside the box OR the tracked objects' bounding box overlaps
                    # Do not make a new track if we are already tracking it
                    if tracked_object_bbox.is_point_inside_rect(Point(maxLoc[0],maxLoc[1])) == True or proposed_object_bbox.overlaps_with(tracked_object_bbox) == True:
                        continue
                    else:
                        imCropOrig = img[int ( bbox[1] ):int ( bbox[1] + bbox[3] ),
                                     int ( bbox[0] ):int ( bbox[0] + bbox[2] )]
                        captures += 1

                        # show the region we are tracking
                        cv2.imshow ( str ( captures ), imCropOrig )
                        cv2.moveWindow ( str ( captures ), capture_window_x, (height + 10) * 2 )
                        capture_window_x += int ( imCropOrig.shape[1] ) + 10

                        # initialize a new feature tracker
                        trackers.add(cv2.TrackerMIL_create(), img, bbox)

    # send the next image to a feature_tracker to find region in the new image
    trackers.update(img)

    # Draw bounding box if tracking success
    for i in np.arange(0, captures):
        drawRectagleOnImage (img0,  trackers.getObjects()[i], (0, 255, 0) )

    cv2.imshow("Thresh", thresh)
    cv2.moveWindow("Thresh", 0, height)

    cv2.imshow("Frame Delta", frameDelta)
    cv2.moveWindow("Frame Delta", width, 0)

    cv2.imshow("Frame", img0)
    cv2.moveWindow("Frame", 0, 0)

    if calcOpticFlow == True:
        cv2.imshow ("Motion", rgb )
        cv2.moveWindow ("Motion", width, height)
    # show blobs
    # cv2.imshow ("Keypoints", im_with_keypoints )
    # cv2.moveWindow ("Keypoints", width, height)

    previousFrame = gray

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q") or key == 27:
        break

    # slow down the process to observe change
    # time.sleep(.25 - (1.0 / cv2.getTickFrequency() / (cv2.getTickCount() - timer)))

    # output result
    print (
        'fps:{0:.1f}, frame#{1}, num_pts:{2}, stdev_x:{3:.1f}, stdev_y:{4:.1f}, stdev_in_xy:{5:.1f}, vector_result:{6}'.format (
            fps, frame_counter, num_pts, stdev_x, stdev_y, stdev_in_xy, vector_result ) )

    # calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

# cleanup the camera and close any open windows
print("Frame_counter:", frame_counter)
print("Frame_count_max:", frame_count_max)
camera.release()
cv2.destroyAllWindows()
sys.exit()
