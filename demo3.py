# python demo3.py --video ./media/20171014_180142.mp4 --width 340
# Demonstrates capturing a region which has motion relevancy based on analysis of change in a dynamic scene
#
# DONE Takes each a frameN and frameN+1
# DONE Applies intra-frame differencing
# DONE The largest value of contrast difference between frames is held in a first in last out que length ~ 3
# DONE The mean distance between each x,y point is determined
# DONE The mean location in x,y is determined
# DONE If the std deviation of the distances between points is lower than a threshold ~ 40
# DONE   Capture the image at the x,y location + radius a multiple of the standard deviation ~2
# TODO Determine the extent of the region to track
#        Evaluate blob detection in the region to determine if this satisfies the requirement
# TODO Capture frame 1 blue circle, find it in frame 2 and get vectors of N common points.
#       Exclude the capture from further processing
#       For each captured track:
#           Perform a correlated point search in the next frame
#               the vectors between the correlated points on an overlayed frame are extracted
#               the vectors magnitude are placed into a histogram relevant to the distance between each other
#               the vectors direction are placed into a histogram relevent to the angle between each other

import argparse
import cv2
import imutils
import sys
import numpy as np
import time
from collections import deque


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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-w", "--width", type=int, default=720, help="width to resize the image")
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

# initialize Tracker
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = args.get("feature_tracker_type")
if tracker_type not in tracker_types:
    sys.exit()

# initiate detector
# keypoint_detector = cv2.ORB_create()
# keypoint_detector = cv2.BRISK_create()
keypoint_detector = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# initialize the first frame in the video stream
camera = None
previousFrame = None
frame_counter = 0
frame_count_max = 0
tracked_area = 0
kernel = np.ones((5, 5), np.uint8)
imCrop = None
que = deque()
previousXY = None
captures = 0
fps = 0

while True:
    # start timer
    timer = cv2.getTickCount()

    # if the last frame is reached, reset the capture and the frame_counter
    if frame_counter >= frame_count_max:
        print("Loop frame_counter:", frame_counter)
        previousFrame = None
        frame_counter = 0

        # if a video path was not supplied, grab the reference to the webcam
        if not args.get("video", False):
            camera = cv2.VideoCapture(0)
        # otherwise, grab a reference to the video file
        else:
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

    height, width, channels = img0.shape

    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # or 15..17

    # if the first frame is None, initialize it
    if previousFrame is None:
        previousFrame = gray
        continue

    # compute the absolute difference between the current and last
    frameDelta = cv2.absdiff(previousFrame, gray)

    # determine min/max value location
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(frameDelta, None)

    # take the threshold of the absolute difference
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

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
    stdev_in_xy = stdev_x + stdev_y / 2

    # draw the mean center of the x,y points with some std dev diameter
    if stdev_in_xy < stdev_min:
        cv2.circle(img0, (int(x), int(y)), int(stdev_in_xy*2), (255, 0, 0), 2)
        # cv2.drawMarker(img0, (int(vector_result[0]),int(vector_result[1])), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)

    # close the holes
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    #roi = thresh[int(x - 20):int(x + 20), int(y - 20):int(y + 20)]
    roi = cv2.circle(img0, (int(x), int(y)), int(stdev_in_xy*4), (255, 0, 0), 2)
    mask = np.zeros (roi.shape, np.uint8)
    grayMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, b_mask = cv2.threshold(gray, 127, 255, 0)
    thresh = cv2.bitwise_and(thresh, thresh, mask=b_mask)

    # find contours - returns image, contours, hierarchy
    (_, contours, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                                None, None)
    if len(contours) > 0:
        # find the largest contour in the mask, then compute
        c = max(contours, key=cv2.contourArea)

        if cv2.pointPolygonTest(c, maxLoc, False) > 0:
            bbox = cv2.boundingRect(c)
            # Extract the bounding box from the original image
            imCrop = img0[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

            cv2.imshow("Captured", imCrop)
            cv2.moveWindow("Captured", height, width)
    #     # initialize feature_tracker once for each attention point with first frame and bounding box
    #     if int(minor_ver) < 3:
    #         feature_tracker = cv2.Tracker_create(tracker_type)
    #     else:
    #         if tracker_type == 'BOOSTING':
    #             feature_tracker = cv2.TrackerBoosting_create()
    #         if tracker_type == 'MIL':
    #             feature_tracker = cv2.TrackerMIL_create()
    #         if tracker_type == 'KCF':
    #             feature_tracker = cv2.TrackerKCF_create()
    #         if tracker_type == 'TLD':
    #             feature_tracker = cv2.TrackerTLD_create()
    #         if tracker_type == 'MEDIANFLOW':
    #             feature_tracker = cv2.TrackerMedianFlow_create()
    #         if tracker_type == 'GOTURN':
    #             feature_tracker = cv2.TrackerGOTURN_create()
    #
    #     # Is the point of greatest change inside the largest polygon?
    #     if cv2.pointPolygonTest(c, maxLoc, False) > 0 and tracked_area < 8000:
    #         bbox = cv2.boundingRect(c)
    #         ok = feature_tracker.init(img0, bbox)
    #         captures += 1
    #
    #         # Limit the number of windows drawn
    #         if captures < 20:
    #             # Extract the bounding box from the original image
    #             imCrop = img0[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
    #             print("Captures, tracked_area:", captures, tracked_area)
    #             cv2.imshow(str(captures), imCrop)
    #             cv2.moveWindow(str(captures), captures * 96, 500)
    #
    # if feature_tracker:
    #     # send the next image to the feature_tracker to find the new bounding box
    #     ok, bbox = feature_tracker.update(img0)
    #
    # # Calculate Frames per second (FPS)
    # # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    #
    # # Draw bounding box
    # if ok:
    #     # Tracking success
    #     p1 = (int(bbox[0]), int(bbox[1]))
    #     p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    #     cv2.rectangle(img0, p1, p2, (255, 0, 0), 2, 1)

    cv2.imshow("Thresh", thresh)
    cv2.moveWindow("Thresh", 0, height)

    cv2.imshow("Frame Delta", frameDelta)
    cv2.moveWindow("Frame Delta", 0, 0)

    cv2.imshow("Frame", img0)
    cv2.moveWindow("Frame", width, 0)

    previousFrame = gray

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    # slow down the process to observe change
    # time.sleep(.25 - (1.0 / cv2.getTickFrequency() / (cv2.getTickCount() - timer)))

    # output result
    print("fps, num_pts,x,y,mean, vector_result ", fps, num_pts,
          stdev_x, stdev_y, stdev_in_xy, vector_result)

    # calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

# cleanup the camera and close any open windows
print("Frame_counter:", frame_counter)
print("Frame_count_max:", frame_count_max)
camera.release()
cv2.destroyAllWindows()
sys.exit()
