# python demo1.py --video ./media/20171014_180142.mp4 --width 720

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
ap.add_argument("-w", "--width", type=int, default=640, help="width to resize the image")
ap.add_argument("-p", "--points", type=int, default=32, help="points to detect in the image")
ap.add_argument("-a", "--min_area", type=int, default=1000, help="minimum area size")
ap.add_argument("-r", "--min_radius", type=int, default=2000, help="minimum radius size")
ap.add_argument("-t", "--tracker_type", default="KCF", help="tracker type one of BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN")

args = vars(ap.parse_args())

if args.get("min_area", True):
    min_area = args["min_area"]
if args.get("points", True):
    points_to_detect = args["points"]
if args.get("min_radius", True):
    min_radius = args["min_radius"]

# Initialize Tracker
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = args.get("tracker_type")
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
    #thresh = cv2.erode(thresh,None, iterations=3)
    #thresh = cv2.erode(thresh, kernel, iterations=1)
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

            # Initialize tracker once for each attention point with first frame and bounding box
            if int(minor_ver) < 3:
                tracker = cv2.Tracker_create(tracker_type)
            else:
                if tracker_type == 'BOOSTING':
                    tracker = cv2.TrackerBoosting_create()
                if tracker_type == 'MIL':
                    tracker = cv2.TrackerMIL_create()
                if tracker_type == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                if tracker_type == 'TLD':
                    tracker = cv2.TrackerTLD_create()
                if tracker_type == 'MEDIANFLOW':
                    tracker = cv2.TrackerMedianFlow_create()
                if tracker_type == 'GOTURN':
                    tracker = cv2.TrackerGOTURN_create()

            bbox = cv2.boundingRect(c)

            if cv2.pointPolygonTest(c, maxLoc, False) > 0 and tracked_area < 8000:
                ok = tracker.init(img0, bbox)
                captures += 1

                imCrop = img0[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                print ("Captures, tracked_area:",captures, tracked_area)

                if captures < 20:
                    cv2.imshow(str(captures), imCrop)
                    cv2.moveWindow(str(captures), captures*96, 500);

        if tracker:
            # Update tracker
            ok, bbox = tracker.update(img0)

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

        # Display tracker type on frame
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
    # if len(cnts) > 0:
    #     # find the largest contour in the mask, then use it to compute
    #     # the minimum enclosing circle and centroid
    #     c = max(cnts, key=cv2.contourArea)
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
    #cv2.imshow("Frame Delta", frameDelta)
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





