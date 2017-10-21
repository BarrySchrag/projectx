# python demo1.py --video ./media/20171014_180142.mp4 --width 720

import argparse
import datetime
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-w", "--width", type=int, default=640, help="width to resize the image")
ap.add_argument("-p", "--points", type=int, default=32, help="points to detect in the image")
ap.add_argument("-a", "--min_area", type=int, default=2000, help="minimum area size")
ap.add_argument("-r", "--min_radius", type=int, default=20, help="minimum radius size")
args = vars(ap.parse_args())

if args.get("min_area", True):
    min_area = args["min_area"]
if args.get("points", True):
    points_to_detect = args["points"]
if args.get("min_radius", True):
    min_radius = args["min_radius"]
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

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    text = "No Change"
    frame_counter += 1

    # if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    if args.get("width", True):
        img0 = imutils.resize(frame, width=args["width"])
    else:
        img0 = frame

    #img1 = cv2.dnn.blobFromImage(img0,	0.007843, (300, 300), 127.5)
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if previousFrame is None:
        previousFrame = gray
        continue

    # compute the absolute difference between the current and last
    frameDelta = cv2.absdiff(previousFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=0)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours > min_area and draw boxes
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
                continue

        # compute the bounding box for the contour, draw it
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Change"

    # get the largest and draw a circle if ? radius
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute
        # the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        #(cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only draw if the radius meets a minimum size
        if radius > min_radius:
            cv2.circle(img0, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    # draw the text and timestamp on the frame
    cv2.putText(img0, "Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img0, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, img0.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # display
    #cv2.imshow("Grey", gray)
    cv2.imshow("Thresh", thresh)
    #cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Frame", img0)

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

    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == frame_count_max:
        frame_counter = 0
        # camera = cv2.set(cv2.CAP_PROP_POS_FRAMES, N)
        camera = cv2.VideoCapture(args["video"])
        previousFrame = None

    # L=81, R=83
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()





