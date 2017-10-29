# python demo3.py --video ./media/20171014_180142.mp4 --width 340
# Demonstrates capturing a region which has motion relevancy based on analysis of change in a dynamic scene
#
# DONE Takes each a frameN and frameN+1
# DONE Applies intra-frame differencing
# DONE The largest value of contrast difference between frames is held in a first in last out que length ~ 3
# DONE The mean distance between each x,y point is determined
# DONE The mean location in x,y is determined
# DONE If the std deviation of the distances between points is lower than a threshold ~ 40
# DONE   Capture the image at the x,y location + radius a multiple of the standard deviation ~4
# TODO Determine the extent of the region to track
#      If the blob of motion representing the region of interest overlaps the point of greatest change then capture the real image 100x100
#      Using the resulting capture, find frame1 in frame2
#      Get vectors of N points which correlate as common points.
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
import copy
from skimage.exposure import rescale_intensity


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
ap.add_argument("-t", "--feature_tracker_type", default="KCF",
                help="feature_tracker type one of: BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN")
ap.add_argument("-d", "--stdev_min", type=int, default=30,
                help="minimum distance between x,y points to be a valid region of interest")
ap.add_argument("-c", "--count_of_concordant_points", type=int, default=3,
                help="minimum count of concordant x,y points to be considered valid region of interest")
ap.add_argument("-f", "--frame_counter_start_frame", type=int, default=220,
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
bbox = None
que = deque()
previousXY = None
captures = 0
captures_max = 30
fps = 0
scalar_stdev_in_xy = 5
capture_window_x = 0
captures_to_file = 0
captures_to_file_max = 10
capture_start = 21

while True:
    # start timer
    timer = cv2.getTickCount()

    # if the last frame is reached, reset the capture and the frame_counter
    if frame_counter >= frame_count_max:
        print("Loop frame_counter:", frame_counter)
        previousFrame = None
        frame_counter = 0
        captures = 0
        capture_window_x = 0
        captures_to_file = 0

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
    else:
        img = frame
    img0 = img.copy()

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
    stdev_in_xy = (stdev_x + stdev_y) / 2


    # if the std deviation is small
    if stdev_in_xy < stdev_min:
        # draw the mean center of the x,y points
        cv2.circle(img0, (int(x), int(y)), int(stdev_in_xy * scalar_stdev_in_xy),
                   (255, 0, 0), 2)
        # cv2.drawMarker(img0, (int(vector_result[0]),int(vector_result[1])), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)

        # close the holes
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        # roi = thresh[int(x - 20):int(x + 20), int(y - 20):int(y + 20)]
        roi = cv2.circle(np.zeros(thresh.shape, thresh.dtype),
                         (int(x), int(y)), int(stdev_in_xy * scalar_stdev_in_xy),
                         255, -1 )
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi)

        # find contours - returns image, contours, hierarchy
        (_, contours, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE, None, None)
    else:
        thresh = np.zeros(thresh.shape, thresh.dtype)

    if len(contours) > 0:
        # find the largest contour in the mask, then compute
        c = max(contours, key=cv2.contourArea)

        if cv2.pointPolygonTest(c, maxLoc, False) > 0:
            bbox = cv2.boundingRect(c)
            # Extract the bounding box from the original image
            imCrop = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            captures += 1
            if captures < captures_max:
                cv2.imshow(str(captures), imCrop)
                cv2.moveWindow(str(captures), capture_window_x, (height+10)*2)
                capture_window_x += int(imCrop.shape[1])+10

    if captures > capture_start and captures_to_file < captures_to_file_max:
        imCrop = img[int ( bbox[1] ):int ( bbox[1] + 100 ), int ( bbox[0] ):int ( bbox[0] + 100 )]
        if output_to_gray:
            grayCrop = cv2.cvtColor ( imCrop, cv2.COLOR_BGR2GRAY )
            grayCrop = cv2.filter2D ( grayCrop, -1, sharpen )
            cv2.imwrite ( "./media/frame-" + str(frame_counter) + ".jpg", grayCrop )
            cv2.imshow ( str ( captures_to_file ), grayCrop )

        else:
            cv2.imwrite ( "./media/frame-" + str ( frame_counter ) + ".jpg", imCrop )
            cv2.imshow ( str ( captures_to_file ), imCrop )

        cv2.moveWindow ( "Capture:" + str ( captures_to_file ), 100*captures_to_file, ((height + 10) * 2 ) + 50)
        captures_to_file += 1
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
    print("fps, frame#, num_pts,x,y,mean, vector_result ", fps, frame_counter, num_pts,
          stdev_x, stdev_y, stdev_in_xy, vector_result)

    # calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

# cleanup the camera and close any open windows
print("Frame_counter:", frame_counter)
print("Frame_count_max:", frame_count_max)
camera.release()
cv2.destroyAllWindows()
sys.exit()
