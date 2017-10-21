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
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

if args.get("points", True):
    points_to_detect=args["points"]

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

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    text = "No Change"
    frame_counter += 1

    # If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == frame_count_max:
        print("resetting frame_counter from:", frame_count_max)
        frame_counter = 0
        #camera = cv2.set(cv2.CAP_PROP_POS_FRAMES, N)
        camera = cv2.VideoCapture(args["video"])

    print("frame_counter:", frame_counter)
    print("frame_count_max:", frame_count_max)

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

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(previousFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
                continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Change"

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # display
    cv2.imshow("Frame", img0)
    cv2.imshow("Grey", gray)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)

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
    print("Key:",key)
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()





