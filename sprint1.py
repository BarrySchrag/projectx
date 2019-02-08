# python sprint1.py --video ./media/20171114_080046.mp4 --width 340
# python sprint1.py --video ./ava/clips/-5KQ66BBWC4/904.mp4
# python sprint1.py --video ./ava/clips/zC5Fh2tTS1U/1147.mp4  1x occluded
# python sprint1.py --video ./ava/clips/zC5Fh2tTS1U/1675.mp4  None
# python sprint1.py --video ./ava/clips/26V9UzqSguo/925.mp4 2x scale1 and 10
# python sprint1.py --video ./ava/clips/26V9UzqSguo/943.mp4  2x
# python sprint1.py --video ./ava/clips/26V9UzqSguo/1321.mp4 Police Wave
# python sprint1.py --video ./ava/clips/26V9UzqSguo/1456.mp4 None, maybe in car
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/949.mp4 3x
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/955.mp4 2x
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/961.mp4 1x scale 2
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/964.mp4 1x
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/1462.mp4 1x scale 2
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/1510.mp4 1x hand expansion scale 1
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/1567.mp4 1x scale 2
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/1753.mp4  2x A scale 2, B scale 8
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/1759.mp4 scale 2
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/1762.mp4 scale 2 same as previous
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/1765.mp4 scale 2, GREAT CAPTURE
# python sprint1.py --video ./ava/clips/8nO5FFbIAog/1768.mp4 scale 2, same as previous


# Demonstrates constraining the region of attention based on analysis of change in a dynamic scene
#
# DONE Takes each a frame and frame+1
# DONE Applied intra-frame differencing and blob detection to find the contour region of the largest blob
# DONE The largest blob' bounding box is used as a que to capture the image area of the blob and send it to a feature tracker
# DONE The capture sent to the feature tracker is output in the small window #1-16
# DONE The feature tracker is represented by the blue bounding box,
# DONE   and tracks the region until the blob detector finds a larger region of change.
# DONE A max point of change found via the intra-frame difference method, and is represented by the red +
# DONE When the max point of change is inside the countour region of the largest intra-frame change
# DONE   then re-initialize the feature tracker to track this new region
#

import argparse
import datetime
import cv2
import imutils
import sys
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-w", "--width", type=int, default=720, help="width to resize the image")
ap.add_argument("-p", "--points", type=int, default=32, help="points to detect in the image")
ap.add_argument("-a", "--min_area", type=int, default=1000, help="minimum area size")
ap.add_argument("-r", "--min_radius", type=int, default=2000, help="minimum radius size")
ap.add_argument("-t", "--feature_tracker_type", default="KCF", help="feature_tracker type one of: BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN")

args = vars(ap.parse_args())

if args.get("min_area", True):
    min_area = args["min_area"]
if args.get("points", True):
    points_to_detect = args["points"]
if args.get("min_radius", True):
    min_radius = args["min_radius"]

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
captures = 0
imCrop = None

while True:
    # Start timer
    timer = cv2.getTickCount()

    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter >= frame_count_max:
        print("Loop frame_counter:", frame_counter)
        previousFrame = None
        frame_counter = 0
        captures = 0
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
    gray = cv2.GaussianBlur(gray, (21, 21), 0 )
    #gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # if the first frame is None, initialize it
    if previousFrame is None:
        previousFrame = gray
        continue

    # compute the absolute difference between the current and last
    frameDelta = cv2.absdiff(previousFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(frameDelta,None)

    # close up tiny holes
    #image_thresh = cv2.erode(image_thresh,None, iterations=3)
    #image_thresh = cv2.erode(image_thresh, kernel, iterations=1)
    # dilate the thresholded image to fill in holes, then find contours
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    # returns image, contours, hierarchy
    (_, cnts, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        # find the largest contour in the mask, then compute
        c = max(cnts, key=cv2.contourArea)

        if cv2.contourArea(c) < min_area:
            continue

        # Track the largest changing contour area only if it is greater in size than the last one
        if cv2.contourArea(c) > tracked_area:
            tracked_area = cv2.contourArea(c)

            # Initialize feature_tracker once for each attention point with first frame and bounding box
            if int(minor_ver) < 3:
                feature_tracker = cv2.Tracker_create(tracker_type)
            else:
                if tracker_type == 'BOOSTING':
                    feature_tracker = cv2.TrackerBoosting_create()
                if tracker_type == 'MIL':
                    feature_tracker = cv2.TrackerMIL_create()
                if tracker_type == 'KCF':
                    feature_tracker = cv2.TrackerKCF_create()
                if tracker_type == 'TLD':
                    feature_tracker = cv2.TrackerTLD_create()
                if tracker_type == 'MEDIANFLOW':
                    feature_tracker = cv2.TrackerMedianFlow_create()
                if tracker_type == 'GOTURN':
                    feature_tracker = cv2.TrackerGOTURN_create()

            # Is the point of greatest change inside the largest polygon?
            if cv2.pointPolygonTest(c, maxLoc, False) > 0 and tracked_area < 8000:
                bbox = cv2.boundingRect(c)
                ok = feature_tracker.init(img0, bbox)
                captures += 1

                # Limit the number of windows drawn
                if captures < 20:
                    # Extract the bounding box from the original image
                    imCrop = img0[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                    print("Captures, tracked_area:", captures, tracked_area)
                    cv2.imshow(str(captures), imCrop)
                    cv2.moveWindow(str(captures), captures*96, 500);

        if feature_tracker:
            # send the next image to the feature_tracker to find the new bounding box
            ok, bbox = feature_tracker.update(img0)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img0, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(img0, "Tracking failure detected", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            tracked_area = min_area

        # Display feature_tracker type on frame
        cv2.putText(img0, tracker_type + " Tracker", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(img0, "FPS : " + str(int(fps)), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # loop over all the contours > min_area and draw boxes
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # draw the bbox for the contour
        cv2.rectangle(img0, (x, y), (x + w, y + h), (192, 192, 192), 1)
    # get the largest and draw a circle if ? radius
    # if len(contours) > 0:
    #     # find the largest contour in the mask, then use it to compute
    #     # the minimum enclosing circle and centroid
    #     c = max(contours, key=cv2.contourArea)
    #     ((x, y), radius) = cv2.minEnclosingCircle(c)
    #     M = cv2.moments(c)
    #     #(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #
    #     # only draw if the radius meets a minimum size
    #     if radius > min_radius:
    #         cv2.circle(img0, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    # draw the text and timestamp on the frame
    #cv2.putText(img0, "{}".format(text), (10, 20),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.putText(img0, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
    #            (10, img0.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # display
    #cv2.imshow("Grey", gray)
    cv2.drawMarker(img0, maxLoc, (0, 0, 255), cv2.MARKER_CROSS, 100, 3);

    cv2.imshow("Thresh", thresh)
    cv2.moveWindow("Thresh", 0, 0)
    #cv2.imshow("Frame Delta", frameDelta1)
    cv2.imshow("Frame", img0)
    #h, w, ch = img0.shape
    cv2.moveWindow("Frame",width, 0)

    previousFrame = gray

    # # find the keypoints
    # kp1, des1 = keypoint_detector.detectAndCompute(img1, None)
    # kp2, des2 = keypoint_detector.detectAndCompute(img2, None)
    #
    # # Match descriptors.
    # matches = bf.match(des1,des2)
    #
    # # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:-x.distance)
    #
    # # Draw matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:64], None, flags=2)
    # plt.axis("off")
    # plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)),plt.show()

    # L=81, R=83
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
print("Captures:", captures)
print("Frame_counter:", frame_counter)
print("Frame_count_max:",frame_count_max)
camera.release()
cv2.destroyAllWindows()
sys.exit()





