# python sprint8.py --video ./media/HandWave1/20171014_180142.mp4 --width 640 --start 60
# python sprint8.py --video ./media/BallDrop1/BallDrop1.mp4 --width 640 -- start 45
# python sprint8.py --video ./ava/clips/26V9UzqSguo/925.mp4 --width 400 #2x scale1 and 10
# python sprint8.py --video ./ava/clips/8nO5FFbIAog/1765.mp4 --width 400 #scale 2, GREAT CAPTURE
# Demonstrates capturing a region which has motion relevancy based on analysis of change in a dynamic scene
# From a video stream, pick out one or more region candidates containing significant change  DONE
#   Take each a frameN and frameN+1, apply intra-frame differencing
#   The largest value of contrast difference between frames is held in a first in last out que length ~ 3
#       In the que
#           The mean distance between each x,y point is determined
#           The mean location in x,y is determined
#           If the std deviation of the distances between points is lower than a threshold ~ 40
#               - execute the next step
# Tag and track regions separately, not switching between them                               DONE
#   Detect the external contour which encompasses the region of change                       DONE
#   Capture a bqounding box of this contour, and show it on screen                            DONE
# Extract the motion vectors contained in the matched region in the bounding box between frames     DONE
#   Keep the mean vectors related to the object being tracked, and the background vectors exclusive DONE
#   Displays the mean vectors as a histogram
#   Detects when the current frame vectors are significantly different from the previous set
#       Changes the color of the box surrounding the region of interest upon every cut.
# DONE: Investigate the data AND write a data extractor for the AVA dataset
#   action_classification/AVA/extract_clips_frames_dset.py  see sprint1.py comments section.
# TODO:  Choose an action class, so we can build a training set
# TODO:  Begin to export the 36 values per time step
# TODO: (vector_angle, vector_magnitude,intra_sample_timestep,viewer_pitch, viewer_roll)
# TODO:  Find an LSTM sample dataset to pattern after.
# TODO Sprint 9 as of Wed Dec 6th - Brian can meet only the 11th, 18th (out 12-17)
# TODO: choose the set of action class instances including the non class instances
# TODO: film one of each

import traceback
import argparse
import cv2
import imutils
import sys
import numpy as np
import math
import time
from collections import deque
import copy
from skimage.exposure import rescale_intensity
from sklearn.preprocessing import normalize
from geometry import *
import logging
from datetime import *
from testpattern import *
from comparehistograms import *
import matplotlib
import random
# Plot a dictionary of figures.
# Use:
# figures = {}
# figures['im'+str(0)] = np.random.randn(100, 100)
# figures['im'+str(1)] = np.random.randn(100, 300)
# plot_images(figures,4)
def plot_images(figures, ncols=1, edgewidth=8, edgeheight=55):
    row=0
    col=0
    colcount = 0
    maxheight = 0
    for ind,title in zip(range(len(figures)), sorted(set(figures))):
        if type(figures[title]) is type ( None ):
            continue
        height, width = figures[title].shape[:2]
        if maxheight < height:
            maxheight = height

        cv2.imshow(title, figures[title])
        cv2.moveWindow(title, col, row)
        col += width+edgewidth
        colcount+=1
        if colcount % ncols == 0:
            col = 0
            row = maxheight + edgeheight

# def refreshScreen(thresh, f0, f1, img0, flow, glitch, width, height):
#     cv2.imshow ( "Thresh", thresh )
#     cv2.moveWindow ( "Thresh", 0, height )
#
#     if type ( f0 ) is not type ( None ):
#         cv2.imshow ( "Angle Histogram", f0 )
#         cv2.moveWindow ( "Angle Histogram", width, 0 )
#     if type ( f1 ) is not type ( None ):
#         cv2.imshow ( "Dir. Histogram", f1 )
#         cv2.moveWindow ( "Dir. Histogram", int(width*2), 0 )
#
#     cv2.imshow ( "Frame", img0 )
#     cv2.moveWindow ( "Frame", 0, 0 )
#
#     if type ( flow ) is not type ( None ):
#         cv2.imshow ( "FB Flow", flow )
#         cv2.moveWindow ( "FB Flow", width, height )
#     if type ( glitch ) is not type ( None ):
#         cv2.imshow ( "FB Flow Glitch", glitch )
#         cv2.moveWindow ( "FB Flow Glitch", width, height )
#     cv2.waitKey ( 1 ) & 0xFF


# In Numpy, axes are defined for arrays with more than one dimension.
# A 2-dimensional array has two corresponding axes:
# the first running vertically downwards across rows (axis 0),
# and the second running horizontally across columns (axis 1).
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])
#
# x.sum(axis=1)
# array([ 6, 22, 38])
def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    rad = math.atan2 ( yDiff, xDiff )
    if rad < 0:
            rad = 2 * math.pi + rad
    return rad


# ddof=0 (default, interprete data as population)
# ddof=1 (interprete it as samples, i.e. estimate true variance)
def std_dev(v1):
    return np.std ( v1, axis=0 )


def var(v1):
    return np.var ( v1, axis=0 )


#def distance(v1, v2):
    # math.sqrt ( (_x2 - x1) * (_x2 - x1) + (_y2 - y1) * (_y2 - y1) )
#    return np.sqrt ( np.sum ( (v1 - v2) ** 2 ) )

def sumColumn(m):
    return [sum ( col ) for col in zip ( *m )]

def meanColumn(m):
    return [np.mean ( col ) for col in zip ( *m )]

def draw_flow(img, arrows, flow, step=16):
    # global arrows
    arrows.clear ()
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape ( 2, -1 ).astype ( int )
    fx, fy = flow[y, x].T
    lines = np.vstack ( [x, y, x + fx, y + fy] ).T.reshape ( -1, 2, 2 )
    lines = np.int32 ( lines + 0.5 )
    vis = cv2.cvtColor ( img, cv2.COLOR_GRAY2BGR )
    cv2.polylines ( vis, lines, 0, (0, 255, 0) )
    for (x1, y1), (_x2, _y2) in lines:
        rad = angle_between ( (x1, y1), (_x2, _y2) )
        dist = distance ( (x1, y1), (_x2, _y2) )
        arrows.append ( [x1, y1, _x2, _y2, rad, dist] )
        cv2.circle ( vis, (x1, y1), 1, (0, 255, 0), -1 )
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2 ( fy, fx ) + np.pi
    v = np.sqrt ( fx * fx + fy * fy )
    hsv = np.zeros ( (h, w, 3), np.uint8 )
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum ( v * 4, 255 )
    bgr = cv2.cvtColor ( hsv, cv2.COLOR_HSV2BGR )
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange ( w )
    flow[:, :, 1] += np.arange ( h )[:, np.newaxis]
    res = cv2.remap ( img, flow, None, cv2.INTER_LINEAR )
    return res


def drawRectagleOnImage(image, bbox, color, thickness=1, linetype=1):
    # x,y,w,h
    p1 = (int ( bbox[0] ), int ( bbox[1] ))
    p2 = (int ( bbox[0] + bbox[2] ), int ( bbox[1] + bbox[3] ))
    cv2.rectangle ( image, p1, p2, color, thickness, linetype )


def drawRectagleOnImage2(image, rect, color, thickness=1, linetype=1):
    p1 = (int ( rect.l_top.x ), int ( rect.l_top.y ))
    p2 = (int ( rect.r_top.x ), int ( rect.l_bot.y ))
    cv2.rectangle ( image, p1, p2, color, thickness, linetype )


def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = int ( (kW - 1) / 2 )
    image = cv2.copyMakeBorder ( image, pad, pad, pad, pad, cv2.BORDER_REPLICATE )
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
    it = iter ( it )
    while True:
        yield next ( it ), next ( it )


def distance(v1, v2):
    return sum ( [(x - y) ** 2 for (x, y) in zip ( v1, v2 )] ) ** 0.5


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split ( '.' )

font = cv2.FONT_HERSHEY_SIMPLEX
width = 0
height = 0
channels = 0
min_area = 0
#points_to_detect = 0
min_radius = 0
stdev = 0
count_of_concordant_points = 0
output_to_gray = False
frame_counter_start_frame = 0  # 150
show_optic_flow_fb = True
show_optic_flow_hsv = False
show_optic_flow_glitch = False
source = None
previousGray = None
previousFrameDelta1 = None
image_angle_hist_shaped = None
previous_image_angle_hist_shaped = None
frameDeltaTemp = None
image_hist_angle = None
image_hist_angle_accumulated = None
image_hist_dist = None
frame_counter = 0
frame_count_max = 0
tracked_area = 0
kernel = np.ones ( (5, 5), np.uint8 )
imCrop = None
bbox = None
que = deque ()
previousXY = None
captures = 0
captures_max = 1
fps = 0
scalar_stdev_in_xy = 1
capture_window_x = 0
captures_to_file = 0
captures_to_file_max = 0  # 20
capture_start = 0
key = -1
trackers = cv2.MultiTracker_create ()
track_found_bbox = []
frames_between_image_captures_for_tracking = 0
imageDerivative = 1
dist_to_polygon = 0
image_thresh = None
image_flow = None
image_curr_glitch = None
optic_flow_window_size=6
arrows = []
vectors_tracked = []
vectors_tracked_for_hist = []
angle_hist_shaped_accumulator = []
vectors_untracked = []
angle_hist_values = ""
dist_hist_values = ""

hist_height_angle = 360
hist_width_angle = 360
nbins_angle = 16
bin_width_angle = hist_width_angle/nbins_angle
action_cut_correlation_value = 1.0001
action_cut_chisquared_value = 0.0000

hist_height_dist = 16
hist_width_dist = 16
nbins_dist = 16
bin_width_dist= hist_width_dist/nbins_dist
figures = {}

histogram_comparrison_result = None
highlight_color = 255
# construct a sharpening filter
sharpen = np.array ( (
    [1, 1, 1],
    [1, -7, 1],
    [1, 1, 1]), dtype="int" )

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser ()
ap.add_argument ( "-v", "--video", help="path to the (optional) video file" )
ap.add_argument ( "-s", "--start", type=int, default=0, help="start frame index to begin processing the video file" )
ap.add_argument ( "-p", "--testpattern", help="path to an (optional) background test pattern file" )
ap.add_argument ( "-w", "--width", type=int, default=400, help="width to resize the image" )
#ap.add_argument ( "-p", "--points", type=int, default=32, help="points to detect in the image" )
ap.add_argument ( "-a", "--min_area", type=int, default=1000, help="minimum area size" )
ap.add_argument ( "-r", "--min_radius", type=int, default=2000, help="minimum radius size" )
ap.add_argument ( "-t", "--feature_tracker_type", default="ALL",
                  help="feature_tracker type one of: ALL, NONE" )
ap.add_argument ( "-d", "--stdev_min", type=int, default=26,
                  help="minimum distance between x,y points to be a valid region of interest" )
ap.add_argument ( "-c", "--count_of_concordant_points", type=int, default=2,
                  help="minimum count of concordant x,y points to be considered valid region of interest" )

args = vars ( ap.parse_args () )

if args.get ( "min_area", True ):
    min_area = args["min_area"]
#if args.get ( "points", True ):
#    points_to_detect = args["points"]
if args.get ( "min_radius", True ):
    min_radius = args["min_radius"]
if args.get ( "stdev_min", True ):
    stdev_min = args["stdev_min"]
if args.get ( "count_of_concordant_points", True ):
    count_of_concordant_points = args["count_of_concordant_points"]
if args.get ( "start", True ):
    frame_counter_start_frame = args["start"]

# initialize Tracker
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = args.get ( "feature_tracker_type" )

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params ()

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
detector = cv2.SimpleBlobDetector_create ( params )

# initiate detector
# keypoint_detector = cv2.ORB_create()
# keypoint_detector = cv2.BRISK_create()
keypoint_detector = cv2.AKAZE_create ( cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT )

# create BFMatcher object
bf = cv2.BFMatcher ( cv2.NORM_HAMMING2, crossCheck=True )

# setup logging
file = './log/{}.csv'.format ( datetime.now ().strftime ( '%H%M%S-%Y%m%d' ) )
logging.basicConfig ( filename=file,
                      format='%(asctime)s,%(levelname)s,%(message)s',
                      filemode='w', level=logging.INFO )

logger = logging.getLogger ( 'info' )

try:
    while True:
        # start timer
        timer = cv2.getTickCount ()

        # if the last frame is reached, reset the capture and the frame_counter
        if frame_counter >= frame_count_max:
            print ( "Loop frame_counter:", frame_counter )
            previousGray = None
            frame_counter = 0
            capture_window_x = 0
            captures_to_file = 0

            ok2 = False
            while captures > 0:
                cv2.destroyWindow ( str ( captures ) )
                captures -= 1
            if captures == 0:
                trackers = cv2.MultiTracker_create ()

            if args.get ( "testpattern", False):
                background_pattern = args["testpattern"]
                source = TestPatternGenerator ( background_pattern, 640, 480, 1.0, 0.0, 1 / 34, 0, 0 )
                source.start ()
            else:
                # if a video path was not supplied, grab the reference to the webcam
                if not args.get ( "video", False ):
                    source = cv2.VideoCapture ( 0 )
                # otherwise, grab a reference to the video file
                else:
                    video_file = args["video"]

                    video_file = './media/BallDrop1/BallDrop1.mp4' # tiny ball drop
                    frame_counter_start_frame = 45

                    #video_file = './media/Bouncing_Ball_Reference-Tk2v1UaTgmk.mp4' # 700 area giant bball

                    #video_file = './media/HandWave1/20171014_180142.mp4' # Kat
                    #frame_counter_start_frame = 70

                    #video_file = './media/Miracle_1-zC5Fh2tTS1U.mp4'
                    #frame_counter_start_frame =38900

                    # video_file = './ava/clips/26V9UzqSguo/925.mp4'
                    #video_file = './ava/clips/8nO5FFbIAog/1765.mp4'

                    #video_file = './ava/clips/8nO5FFbIAog/949.mp4'

                    source = cv2.VideoCapture ( video_file )

            frame_count_max = source.get ( cv2.CAP_PROP_FRAME_COUNT )

        # grab the current frame
        (grabbed, frame) = source.read ()
        frame_counter += 1

        # command line paramater to delay the start of processing
        if frame_counter < frame_counter_start_frame:
            if frame_counter %1000 == 0:
                print('Frame:{}'.format(frame_counter))
            continue

        # if we are viewing a video and we did not grab a
        # frame, then we have reached the end of the video
        if args.get ( "video" ) and not grabbed:
            break
        if args.get ( "width", True ):
            width = args["width"]
            frame = imutils.resize ( frame, width )
        #else:
        #    img = frame

        img = frame
        img0 = frame.copy()

        if len(img0.shape) == 2:
            height, width = img0.shape
            channels = 1
        else:
            height, width, channels = img0.shape

        if channels == 3:
            gray = cv2.cvtColor ( img0, cv2.COLOR_BGR2GRAY )
        else:
            gray = img0

        # Detect blobs.
        keypoints = None  # detector.detect (img0)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        # im_with_keypoints = cv2.drawKeypoints (img0, keypoints, np.array ( [] ), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

        gray = cv2.GaussianBlur ( gray, (7, 7), 0 )  # or 3,5,7,9,11,13,15,17,21
        # close the holes
        # gray = cv2.dilate(gray, kernel, iterations=1)

        # if the first frame is None, initialize it
        if previousGray is None:
            previousGray = gray
            continue

        # compute the absolute difference between the current and last
        frameDelta1 = cv2.absdiff ( previousGray, gray )
        if previousFrameDelta1 is None:
            previousFrameDelta1 = frameDelta1
            continue

        # this calculates the 2nd derivative of the grayscale image
        frameDelta2 = cv2.absdiff ( previousFrameDelta1, frameDelta1 )

        # determine min/max value location
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc ( frameDelta1, None )
        (minVal2, maxVal2, minLoc2, maxLoc2) = cv2.minMaxLoc ( frameDelta2, None )

        # take the threshold of the absolute difference
        if imageDerivative == 2:
            image_thresh = cv2.threshold ( frameDelta2, 10, 255, cv2.THRESH_BINARY )[1]
            maxLoc = maxLoc2
            frameDeltaTemp = frameDelta2
        else:
            image_thresh = cv2.threshold ( frameDelta1, 10, 255, cv2.THRESH_BINARY )[1]
            frameDeltaTemp = frameDelta1

        # determine optic flow
        if show_optic_flow_fb == True:
            image_flow = cv2.calcOpticalFlowFarneback ( previousGray, gray, None,
                                                        #image_thresh,
                                                        0.5, 2, optic_flow_window_size, 2, 5, 1.1,
                                                        0 )  # cv2.OPTFLOW_FARNEBACK_GAUSSIAN )
            # arrows will be cleared
            image_flow = draw_flow ( gray, arrows, image_flow, optic_flow_window_size )

            if show_optic_flow_hsv == True:
                image_flow = draw_hsv ( image_flow )
            if show_optic_flow_glitch == True:
                image_cur_glitch = warp_flow ( image_cur_glitch, image_flow )
                # https://stackoverflow.com/questions/41760437/opencv-how-to-apply-a-filter-on-vectors-obtained-through-calcopticalflowfarneba

        # draw marker at max location
        cv2.drawMarker ( img0, maxLoc, (0, 0, 255), cv2.MARKER_CROSS, 20, 1 )
        cv2.drawMarker ( img0, maxLoc2, (0, 0, 255), cv2.MARKER_DIAMOND, 20, 1 )

        # limit the size of the queue
        if len ( que ) > count_of_concordant_points:
            que.popleft ()

        # add the latest location to the que
        que.append ( maxLoc )

        mean_x = 0
        mean_y = 0
        x = 0
        y = 0
        vector_result = (0, 0)
        num_pts = len ( que )
        for xy in list ( que ):
            if previousXY is None:
                previousXY = xy
                continue
            # the mean distance between each x,y point is calculated
            mean_x = abs ( xy[0] - previousXY[0] )
            mean_y = abs ( xy[1] - previousXY[1] )

            # the mean location in x,y is determined
            x += xy[0]
            y += xy[1]

            cv2.drawMarker ( img0, xy, (128, 128, 128), cv2.MARKER_SQUARE, 10, 1 )

        # find the mean of the x, y positions
        x /= num_pts
        y /= num_pts

        # find the mean of the distances between the points
        mean_x /= num_pts
        mean_y /= num_pts

        # find the std deviation of the distances between each point
        list_x = [x[0] for x in que]
        stdev_x = np.std ( list_x )
        list_y = [y[1] for y in que]
        stdev_y = np.std ( list_y )

        # find the stdev in x,y
        stdev_in_xy = (stdev_x + stdev_y) / 2

        # if the std deviation is small
        if stdev_in_xy < stdev_min:
            # draw the mean center of the x,y points
            cv2.circle ( img0, (int ( x ), int ( y )), int ( stdev_in_xy * scalar_stdev_in_xy ), (255, 0, 0), 2 )

            # close the holes
            #image_thresh = cv2.dilate(image_thresh, kernel, iterations=1)

            # roi = cv2.circle(np.zeros(image_thresh.shape, image_thresh.dtype),
            #                 (int(x), int(y)), int(stdev_in_xy * scalar_stdev_in_xy), 255, -1 )
            # image_thresh = cv2.bitwise_and(image_thresh, image_thresh, mask=roi)

            # find contours - returns image, contours, hierarchy
            (_, contours, hierarchy) = cv2.findContours ( image_thresh, cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_SIMPLE, None, None )
            # else:
            # image_thresh = np.zeros( image_thresh.shape, image_thresh.dtype )
            # contours.clear

        if len ( contours ) > 0 and captures < captures_max:
            # find the largest contour
            c = max ( contours, key=cv2.contourArea )
            dist_to_polygon = cv2.pointPolygonTest ( c, maxLoc, True )
            dist_to_polygon2 = cv2.pointPolygonTest ( c, maxLoc2, True )
            dist_between_dt2 = distance(maxLoc,maxLoc2)
            if frame_counter == 10:
                 area = cv2.contourArea(c)
            area = cv2.contourArea(c)
            if area > 520 and dist_between_dt2 < 2 and dist_to_polygon > -1: #area > 800 and
                #area > 1000 and  dist_between_dt2 < 5:#  and dist_to_polygon > 3:
                # and blue circle contains the min and max loc..
                bboxt = cv2.boundingRect ( c )  # x,y,w,h

                # img.shape returns a tuple of number of rows Y, columns X and channels (if image is color):
                # slice startY:endY, startX: endX
                # if the box is too big (edge of screen) then continue
                # x + w > screen width?
                if bboxt[0] + bboxt[2] > width or  bboxt[2] + bboxt[3] > height:
                    continue
                bbox = bboxt

                tempw = 0
                temph = 0
                # make square
                # if bboxt[2] < bboxt[3]:
                #     tempw = bboxt[3]
                # else:
                #     tempw = bboxt[2]
                # if bboxt[3] < bboxt[2]:
                #     temph = bboxt[2]
                # else:
                #     temph = bboxt[3]
                # bbox = (bboxt[0],bboxt[1],tempw,temph)
                proposed_object_bbox = Rect ( int ( bbox[0] ), int ( bbox[1] ),
                                              int ( bbox[2] ), int ( bbox[3] ) )  # x, y, width, height):
                #drawRectagleOnImage ( img0, bbox, (128, 128, 128) )
                #drawRectagleOnImage2 ( img0, proposed_object_bbox, (0, 0, 128) )

                # We have not yet tracked anything, so add it
                proposed_box_overlaps = False
                if trackers.getObjects () == ():
                    proposed_box_overlaps == False
                else:
                    # Loop through all existing bounding boxes to see if this point is already in a region
                    for i in np.arange ( 0, captures ):

                        bx = trackers.getObjects ()[i]
                        tracked_object_bbox = Rect ( int ( bx[0] ), int ( bx[1] ), int ( bx[2] ), int ( bx[3] ) )

                        # Is the point inside the box OR the tracked objects' bounding box overlaps
                        # Do not make a new track if we are already tracking it
                        if (tracked_object_bbox.is_point_inside_rect ( Point ( maxLoc[0], maxLoc[1] ) ) == True
                            or proposed_object_bbox.overlaps_with ( tracked_object_bbox ) == True):
                            proposed_box_overlaps = True

                # create a new tracked region only if we are not already tracking it
                if proposed_box_overlaps == False:
                    # img.shape returns a tuple of number of rows Y, columns X and channels (if image is color):
                    # slice startY:endY, startX: endX
                    image_CropOrig = img[int ( bbox[1] ):int ( bbox[1] + bbox[3]),
                                     int ( bbox[0] ):int ( bbox[0] + bbox[2] )]

                    # initialize a new feature tracker where bbox defines the region to track
                    trackers.add ( cv2.TrackerMIL_create (), img, (bbox[0],bbox[1], bbox[2],bbox[3]) )
                    #cv2.putText ( image_CropOrig, str ( captures ), (1, 20), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA )

                    # show the region we are tracking
                    cv2.imshow ( str ( captures ), image_CropOrig )
                    cv2.moveWindow ( str ( captures ), capture_window_x, (height + 10) * 2 )
                    capture_window_x += int ( image_CropOrig.shape[1] ) + 10

                    captures += 1

        # send the next image to a feature_tracker to find region in the new image
        trackers.update ( img )

        vectors_tracked.clear ()
        vectors_tracked_for_hist.clear ()
        vectors_untracked.clear ()

        # Draw bounding box if tracking success
        for i in np.arange ( 0, captures ):
            bx = trackers.getObjects ()[i]

            drawRectagleOnImage ( img0, bx, highlight_color )


            #cv2.putText ( img0, str ( i ), (int ( bx[0] + 1 ), int ( bx[1] + 20 )), font, 0.8, (0, 255, 0), 1,
            # cv2.LINE_AA )

            tracked_object_bbox = Rect ( int ( bx[0] ), int ( bx[1] ), int ( bx[2] ), int ( bx[3] ) )

            # extract optic flow only from the tracked region
            for x1, y1, _x2, _y2, rad, dist in arrows:
                # Is the arrow origin inside the tracked object box? add it to our vector list
                if tracked_object_bbox.is_point_inside_rect ( Point ( x1, y1 ) ) == True:
                    #if dist > 3:
                    vectors_tracked.append ( [i, x1, y1, _x2, _y2, rad, dist] )
                    vectors_tracked_for_hist.append([360-math.degrees(rad),dist])
                else:
                    vectors_untracked.append ( [i, x1, y1, _x2, _y2, rad, dist] )
            if 1 == 1:
                # Angle begin histogram - get the angles into a list
                angle_hist = [x[0] for x in vectors_tracked_for_hist]
                angle_hist_shaped = np.array(angle_hist).astype(np.float32)
                #np.save('./media/angle_hist_shaped-{}.npy'.format(frame_counter), angle_hist_shaped)

                image_angle_hist_shaped = cv2.calcHist( [angle_hist_shaped], [0], None, [nbins_angle], [0, hist_width_angle] )
                cv2.normalize ( image_angle_hist_shaped, image_angle_hist_shaped, 0, hist_height_angle, cv2.NORM_MINMAX )

                # Accumulator
                angle_hist_shaped_accumulator = np.append(angle_hist_shaped_accumulator,
                    angle_hist_shaped ).astype( np.float32)

                image_angle_hist_shaped_accumulator = cv2.calcHist( [angle_hist_shaped_accumulator], [0], None, [nbins_angle], [0, hist_width_angle] )
                cv2.normalize ( image_angle_hist_shaped_accumulator, image_angle_hist_shaped_accumulator, 0, hist_height_angle, cv2.NORM_MINMAX )

                if previous_image_angle_hist_shaped is not None:
                    # Compare the per-frame and accumulated histograms
                    H = comparehistograms( image_angle_hist_shaped, previous_image_angle_hist_shaped)
                    histogram_comparrison_result = H.compare('All')
                    print('Frame:' +
                          str(frame_counter) +
                          ' ' +
                              str(
                                list(map((lambda x: x[0][:3] + ':{:.3f}'.format(x[1])), histogram_comparrison_result))
                              )
                          )

                    if (histogram_comparrison_result[0][1] <= action_cut_correlation_value) or (
                              histogram_comparrison_result[0][1] <= 1.0001 and  histogram_comparrison_result[1][1] ==
                                action_cut_chisquared_value):
                       angle_hist_shaped_accumulator = []
                       highlight_color = np.random.randint(0,255,(3)).tolist()
                       print("CUT Frame:{}-", frame_counter)

                # Distance
                dist_hist = [x[1] for x in vectors_tracked_for_hist]
                dist_hist_shaped = np.array(dist_hist).astype(np.float32)
                image_hist_dist = np.zeros((hist_height_dist, hist_width_dist), dtype=np.float32)
                hist_item_dist = cv2.calcHist ( [dist_hist_shaped], [0], None, [nbins_dist], [0, hist_width_dist] )
                #cv2.normalize ( hist_item_dist, hist_item_dist, 0, hist_height_dist, cv2.NORM_MINMAX )

                angle_hist_values = ','.join ( map ( str, image_angle_hist_shaped.flatten () ) )
                dist_hist_values = ','.join ( map ( str, hist_item_dist.flatten () ) )

                # Angle - Loop through each bin and plot the rectangle in 255 white
                image_hist_angle = np.zeros((hist_height_angle, hist_width_angle), dtype=np.float32)
                for x, y in enumerate ( image_angle_hist_shaped ):
                    cv2.rectangle ( image_hist_angle, (int ( x * bin_width_angle ), int ( y )),
                                    (int ( x * bin_width_angle + bin_width_angle - 1 ), int ( hist_height_angle)),
                                    255, -1 )
                image_hist_angle = np.flipud ( image_hist_angle )

                # Accumlator - Loop through each bin and plot the rectangle in 255 white
                image_hist_angle_accumulated = np.zeros((hist_height_angle, hist_width_angle), dtype=np.float32)
                for x, y in enumerate ( image_angle_hist_shaped_accumulator):
                    cv2.rectangle ( image_hist_angle_accumulated,  (int ( x * bin_width_angle ), int ( y )),
                                    (int ( x * bin_width_angle + bin_width_angle - 1 ), int ( hist_height_angle)),
                                    255, -1 )
                image_hist_angle_accumulated = np.flipud ( image_hist_angle_accumulated )


                # Dist - Loop through each bin and plot the rectangle in 255 white
                for x, y in enumerate ( hist_item_dist ):
                    cv2.rectangle ( image_hist_dist, (int ( x * bin_width_dist ), int ( y )),
                                    (int ( x * bin_width_dist + bin_width_dist - 1 ), int ( hist_height_dist )),
                                    255, -1 )
                # Dist - Flip upside down
                image_hist_dist = np.flipud ( image_hist_dist )
                # end histogram

            if len(vectors_tracked) > 0:
                tracked_data = meanColumn ( vectors_tracked )

                tracked_rad = angle_between ( (tracked_data[1], tracked_data[2]), (tracked_data[3], tracked_data[4]) )
                tracked_dist = distance ( (tracked_data[1], tracked_data[2]), (tracked_data[3], tracked_data[4]) )

                tracked_mean_start_x = np.mean ( tracked_data[1] )
                tracked_mean_start_y = np.mean ( tracked_data[2] )
                tracked_mean_end_x = np.mean ( tracked_data[3] )
                tracked_mean_end_y = np.mean ( tracked_data[4] )
                tracked_mean_dist = distance ( (tracked_mean_start_x, tracked_mean_start_y),
                                               (tracked_mean_end_x, tracked_mean_end_y))

                cv2.arrowedLine ( img0, (int ( tracked_mean_start_x ), int ( tracked_mean_start_y )),
                                  (int ( tracked_mean_end_x ), int ( tracked_mean_end_y )),
                                  (255, 255, 255), 1, 8, 0, 4)
                cv2.putText ( img0, '{:.1f}'.format(math.degrees(tracked_rad )),
                              (int ( tracked_mean_start_x + 2 ),  int ( tracked_mean_start_y + 20 )),
                              font, 0.8, (222, 222, 222), 1,cv2.LINE_AA )

                tracked_std_dev_rad = std_dev ( vectors_tracked )[5]
                tracked_std_dev_dist = std_dev ( vectors_tracked )[6]
                tracked_var_rad = var ( vectors_tracked )[5]
                tracked_var_dist = var ( vectors_tracked )[6]

            untracked_data = meanColumn ( vectors_untracked )
            untracked_rad = angle_between ( (untracked_data[1], untracked_data[2]), (untracked_data[3], untracked_data[4]) )
            untracked_dist = distance ( (untracked_data[1], untracked_data[2]), (untracked_data[3], untracked_data[4]) )
            untracked_std_dev_rad = std_dev ( vectors_untracked )[5]
            untracked_std_dev_dist = std_dev ( vectors_untracked )[6]
            untracked_var_rad = var ( vectors_untracked )[5]
            untracked_var_dist = var ( vectors_untracked )[6]

            if len ( vectors_tracked ) > 0:
                image_data = [tracked_data[0], 'tracked', tracked_dist, tracked_rad, math.degrees ( tracked_rad ),
                              tracked_std_dev_dist, tracked_var_rad, tracked_mean_start_x, tracked_mean_start_y,
                              tracked_mean_end_x, tracked_mean_end_y]
                # log all data
                logger.info ( ','.join ( map ( str, image_data ) ) +',5151,'+angle_hist_values+',5152,'+dist_hist_values+
                              ','+ str(frame_counter))

                #print ( 'Tracked   item:{}, dist:{:.1f} rad:{:.1f}, deg:{:.1f}, std_dist:{:.1f}, var_angle:{
                # :.1f}'.format (
                 #   tracked_data[0], tracked_dist, tracked_rad, math.degrees ( tracked_rad ),
                 #   tracked_std_dev_dist, tracked_var_rad ) )
                #print( 'Tracked Hist Angles:{}',format(angle_hist_values))
                image_data = [tracked_data[0], 'untracked', untracked_dist, untracked_rad, math.degrees ( untracked_rad ),
                              untracked_std_dev_dist, untracked_var_rad]
                logger.info ( ','.join ( map ( str, image_data ) ) )

            #print ( 'UnTracked item:{}, dist:{:.1f} rad:{:.1f}, deg:{:.1f}, std_dist:{:.1f}, var_angle:{:.1f}'.format (
            #    untracked_data[0], untracked_dist, untracked_rad, math.degrees ( untracked_rad ),
            #    untracked_std_dev_dist, untracked_var_rad ) )

        # https://www.youtube.com/watch?v=WcmHxBtz3EY  # complete pan up from static camera outdoors dist:0.7 @359.9 deg
        # https://www.youtube.com/watch?v=eBL6vu9NQtw  # complete pan right static camera indoors
        # https://www.youtube.com/watch?v=PoHJsmeHYHo  #  complete pan left from static camera indoors
        # https://www.youtube.com/watch?v=oMumnZtboU4  # complete pan up from static camera indoors
        # https://www.youtube.com/watch?v=7-eowYn1gBE  # complete pan down from static camera indoors
        # https://www.youtube.com/watch?v=yCaZW8pGijE # zoom, rotate, pan


        # If no tracks, then average the flow field
        if captures == 0:
            untracked_data = meanColumn ( arrows )
            untracked_rad = angle_between ( (untracked_data[0], untracked_data[1]), (untracked_data[2], untracked_data[3]) )
            untracked_dist = distance ( (untracked_data[0], untracked_data[1]), (untracked_data[2], untracked_data[3]) )
            untracked_std_dev_rad = std_dev ( arrows )[4]
            untracked_std_dev_dist = std_dev ( arrows )[5]
            untracked_var_rad = var ( arrows )[4]
            untracked_var_dist = var ( arrows )[5]
            image_data = [-1, 'untracked', untracked_dist, untracked_rad, math.degrees ( untracked_rad ),
                          untracked_std_dev_dist,
                          untracked_var_rad]
            logger.info ( ','.join ( map ( str, image_data ) ) )
            print ( 'Bg frame:{} dist:{:.1f} rad:{:.1f}, deg:{:.1f}, std_dist:{:.1f}, var_angle:{:.1f}'.format (
                frame_counter, untracked_dist, untracked_rad, math.degrees ( untracked_rad ), untracked_std_dev_dist,
                untracked_var_rad ) )

        figures['original-'+str(0)] = img0
        figures['flow-'+str(1)] = image_flow
        figures['dist-'+str(2)] = image_hist_dist
        figures['angle-'+str(3)] = image_hist_angle
        figures['accumulated-'+str(4)] = image_hist_angle_accumulated

        plot_images(figures,3)
        # Beware, this writes images, not the original data!
        #cv2.imwrite('./media/hist_images_180bins-{}.jpg'.format(frame_counter),image_hist_angle)

        # save the previous frames to calculate derivatives
        previousGray = gray
        previousFrameDelta1 = frameDelta1
        previous_image_angle_hist_shaped = image_angle_hist_shaped

        key = cv2.waitKey ( 1 )
        # if the 'q' key is pressed, stop the loop
        if key == ord ( "q" ) or key == 27:
            break
        if key == ord ( '1' ):
            show_optic_flow_fb = not show_optic_flow_fb
            print ( 'FB optic flow visualization is', ['off', 'on'][show_optic_flow_fb] )
        if key == ord ( '2' ):
            show_optic_flow_hsv = not show_optic_flow_hsv
            print ( 'HSV optic flow visualization is', ['off', 'on'][show_optic_flow_hsv] )
        if key == ord ( '3' ):
            show_optic_flow_glitch = not show_optic_flow_glitch
            if show_optic_flow_glitch:
                image_cur_glitch = img.copy ()
            print ( 'glitch is', ['off', 'on'][show_optic_flow_glitch] )

        # slow down the process to observe change
        # time.sleep(.25 - (1.0 / cv2.getTickFrequency() / (cv2.getTickCount() - timer)))

        # output result
        #print (
        #    'fps:{0:.1f}, frame#{1}, num_pts:{2}, stdev_x:{3:.1f}, stdev_y:{4:.1f}, stdev_in_xy:{5:.1f},
            # distance:{6} vector_result:{7}'.format (
          #      fps, frame_counter, num_pts, stdev_x, stdev_y, stdev_in_xy, dist_to_polygon, vector_result ) )

        # calculate Frames per second (FPS)
        fps = cv2.getTickFrequency () / (cv2.getTickCount () - timer)

    logger.close ()
    # cleanup the source and close any open windows
    print ( "Frame_counter:", frame_counter )
    print ( "Frame_count_max:", frame_count_max )
    source.release ()
    cv2.destroyAllWindows ()
    sys.exit ()

except Exception as inst:
    print(str(inst))
    if source is not None:
        source.release ()
    cv2.destroyAllWindows ()

    exc_type, exc_obj, exc_tb = sys.exc_info ()
    fname = os.path.split ( exc_tb.tb_frame.f_code.co_filename )[1]
    print(exc_type, fname, exc_tb.tb_lineno , exc_obj)
    #desired_trace = traceback.format_exc ( sys.exc_info () )
    #print(desired_trace)
    del (exc_type, exc_obj, exc_tb)
    sys.exit ()

