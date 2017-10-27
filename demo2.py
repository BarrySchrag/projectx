# python demo2.py --video ./media/20171014_180142.mp4 --width 340
# Demonstrates constraining the region of attention based on analysis of change in a dynamic scene
#
# Takes each a frameN and frameN+1
# Applies intra-frame differencing
# The largest value of contrast difference between frames is held in a first in last out que length ~ 3
# The mean distance between each x,y point is determined
# The mean location in x,y is determined
# If the std deviation of the distances between points is lower than a threshold ~ 40
#   Capture the image at the x,y location + radius a multiple of the standard deviation ~2
#   Track the captured area, but exclude it from further processing
#       For each captured track:
#           Perform a correlated point search in the next frame
#               the vectors between the correlated points on an overlayed frame are extracted
#               the vectors magnitude are placed into a histogram relevant to the distance between each other
#               the vectors direction are placed into a histogram relevent to the angle between each other

import argparse
import cv2
import imutils
import sys
import math
import numpy as np
from collections import deque

# Function to get each a,b of a series
# Use: for a, b in pairwise(que):
def pairwise(it):
    it = iter(it)
    while True:
        yield next(it), next(it)

def distance(v1,v2):
    return sum([(x-y)**2 for (x,y) in zip(v1,v2)])**(0.5)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-w", "--width", type=int, default=480, help="width to resize the image")
ap.add_argument("-p", "--points", type=int, default=32, help="points to detect in the image")
ap.add_argument("-a", "--min_area", type=int, default=1000, help="minimum area size")
ap.add_argument("-r", "--min_radius", type=int, default=2000, help="minimum radius size")
ap.add_argument("-t", "--feature_tracker_type", default="KCF",
                help="feature_tracker type one of: BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN")
ap.add_argument("-d", "--stdev_min", type=int, default=30,
                help="minimum distance between x,y points to be a valid region of interest")
ap.add_argument("-c", "--count_of_concordant_points", type=int, default=3,
                help="minimum count of concordant x,y points to be considered valid region of interest")

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

# Initialize Tracker
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = args.get("feature_tracker_type")
if tracker_type not in tracker_types:
    sys.exit()

# Initiate detector
#keypoint_detector = cv2.ORB_create()
#keypoint_detector = cv2.BRISK_create()
keypoint_detector = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
previousFrame = None
frame_counter = 0
frame_count_max = camera.get(cv2.CAP_PROP_FRAME_COUNT)
tracked_area = min_area
kernel = np.ones((5,5),np.uint8)
imCrop = None
que = deque()
previousXY = None

while True:
    # Start timer
    timer = cv2.getTickCount()

    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter >= frame_count_max:
        print("Loop frame_counter:", frame_counter)
        previousFrame = None
        frame_counter = 0
        camera = cv2.VideoCapture(args["video"])
        frame_count_max = camera.get(cv2.CAP_PROP_FRAME_COUNT)

    # grab the current frame
    (grabbed, frame) = camera.read()
    frame_counter += 1

    # if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    if args.get("width", True):
        width = args["width"]
        img0 = imutils.resize(frame, width)
    else:
        img0 = frame

    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0 ) # or 15..17

    # if the first frame is None, initialize it
    if previousFrame is None:
        previousFrame = gray
        continue

    # compute the absolute difference between the current and last
    frameDelta = cv2.absdiff(previousFrame, gray)

    # determine min/max value location
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(frameDelta,None)

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
    num_pts = len(que)
    for xy in list(que):
        if previousXY is None:
            previousXY = xy
            continue
        # The mean distance between each x,y point is calculated
        mean_x = abs(xy[0] - previousXY[0])
        mean_y = abs(xy[1] - previousXY[1])

        # The mean location in x,y is determined
        x += xy[0]
        y += xy[1]
        cv2.drawMarker(img0, xy, (128, 128, 128), cv2.MARKER_CROSS, 10, 1)

    # find the mean of the x, y positions
    x = x / num_pts
    y = y / num_pts

    # find the mean of the distances between the points
    mean_x = mean_x / num_pts
    mean_y = mean_y / num_pts

    # find the std deviation of the distances between each point
    list_x = [x[0] for x in que]
    stdev_x = np.std(list_x)
    list_y = [y[1] for y in que]
    stdev_y = np.std(list_y)

    # find the stdev in x,y
    stdev_in_xy = stdev_x + stdev_y / 2

    print("between pont stdev: num_pts,x,y,mean ", num_pts, stdev_x, stdev_y, stdev_in_xy)

    # draw the mean center of the x,y points with 1 std dev diameter
    if stdev_in_xy < stdev_min:
        cv2.circle(img0, (int(x),int(y)), int(stdev_in_xy*2), (255, 0, 0), 2)

    cv2.imshow("Frame Delta", frameDelta)
    cv2.moveWindow("Frame Delta", 0, 0)

    cv2.imshow("Frame", img0)
    cv2.moveWindow("Frame",width, 0)

    previousFrame = gray

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

# cleanup the camera and close any open windows
print("Frame_counter:", frame_counter)
print("Frame_count_max:",frame_count_max)
camera.release()
cv2.destroyAllWindows()
sys.exit()

