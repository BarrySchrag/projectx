# python demo0.py -a ./media/0146.bmp -b ./media/0147.bmp
# python demo0.py -a ./media/file1.jpg -b ./media/file2.jpg

# Demonstrates finding correlating points in 2 seperate images which are similar but not identical
# DONE Utilize a modern binary descriptor and a matcher, to find coorelated matching points in 2 images sorted by relevance.

import argparse
import cv2
import matplotlib.pyplot as plt
import imutils
import sys

width_resize = 400
matches_requested = 32
crossCheckValue=True
sort_matches = True
use_ratio=True
ratio_test_value=0.77 # 0.7
vertical_offset = 0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-a", "--imagea", required=True, help="Path to the image1")
ap.add_argument("-b", "--imageb", required=True, help="Path to the image2")
ap.add_argument("-y", "--vertical_offset", required=True, type=int, help="Offset Y axis for output")
args = vars(ap.parse_args())

img1 = cv2.imread(args["imagea"],1) # queryImage
img2 = cv2.imread(args["imageb"],1) # trainImage

if args.get("vertical_offset", True):
    vertical_offset = args["vertical_offset"]

try:
    #if img1.shape[0] > width_resize:
    imutils.resize(img1, width=width_resize)
    imutils.resize(img2, width=width_resize)

    # Initiate
    key_descriptorsOrb = cv2.ORB_create()
    key_descriptorsBrisk = cv2.BRISK_create()
    key_descriptorsAkaze = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT)

    # ORB find the keypoints and descriptors
    kp1_orb, des1_orb = key_descriptorsOrb.detectAndCompute(img1,None)
    kp2_orb, des2_orb = key_descriptorsOrb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck=crossCheckValue)

    # Match descriptors.
    matches_orb = bf_orb.match(des1_orb,des2_orb)

    # Sort them in the order of their distance.
    if sort_matches:
        matches_orb = sorted(matches_orb, key = lambda x:x.distance)

    # Draw matches.
    img_orb = cv2.drawMatches(img1,kp1_orb,img2,kp2_orb,matches_orb[:matches_requested], None, flags=2)
    plt.axis("off")
    #plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB)),plt.show()
    cv2.imshow ( "img_orb", img_orb )
    cv2.moveWindow ( "img_orb", img_orb.shape[1]*0, vertical_offset)
except Exception as e:
    print("ORB Error: " + str(e))

try:
    # Brisk the keypoints and descriptors
    kp1_brisk, des1_brisk = key_descriptorsOrb.detectAndCompute(img1,None)
    kp2_brisk, des2_brisk = key_descriptorsOrb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf_brisk = cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck=crossCheckValue)

    # Match descriptors.
    matches_brisk = bf_brisk.match(des1_brisk,des2_brisk)

    # Sort them in the order of their distance.
    if sort_matches:
        matches_brisk = sorted(matches_brisk, key = lambda x:x.distance)

    # Draw matches.
    img_brisk = cv2.drawMatches(img1,kp1_brisk,img2,kp2_brisk,matches_brisk[:matches_requested], None, flags=2)
    plt.axis("off")
    #plt.imshow(cv2.cvtColor(img_brisk, cv2.COLOR_BGR2RGB)),plt.show()
    cv2.imshow ( "img_brisk", img_brisk )
    cv2.moveWindow ( "img_brisk", img_brisk.shape[1]*1, vertical_offset)
except Exception as e:
    print("BRISK Error: " + str(e))

try:
    # Brisk the keypoints and descriptors
    kp1_Akaze, des1_Akaze = key_descriptorsOrb.detectAndCompute(img1, None)
    kp2_Akaze, des2_Akaze = key_descriptorsOrb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf_Akaze = cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck=crossCheckValue)

    # Match descriptors.
    matches_Akaze = bf_Akaze.match(des1_Akaze,des2_Akaze)

    # Sort them in the order of their distance.
    if sort_matches:
        matches_Akaze = sorted(matches_Akaze, key = lambda x:x.distance)

    # Draw matches.
    img_Akaze = cv2.drawMatches(img1,kp1_Akaze,img2,kp2_Akaze,matches_Akaze[:matches_requested], None, flags=2)
    plt.axis("off")
    #plt.imshow(cv2.cvtColor(img_Akaze, cv2.COLOR_BGR2RGB)),plt.show()
    cv2.imshow ( "img_Akaze", img_Akaze )
    cv2.moveWindow ( "img_Akaze", img_Akaze.shape[1]*2, vertical_offset)
except Exception as e:
    print("AKAZE Error: " + str(e))

try:
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(des1,des2,k=2)
    #
    # # Need to draw only good matches, so create a mask
    # matchesMask = [[0,0] for i in range(len(matches))]
    # # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     if use_ratio:
    #         if m.distance < ratio_test_value*n.distance:
    #             matchesMask[i]=[1,0]
    #     else:
    #         matchesMask[i]=[1,0]
    #
    # draw_params = dict(matchColor = (0,255,0),
    #                    singlePointColor = (255,0,0),
    #                    matchesMask = matchesMask,
    #                    flags = 0)
    # img_sift = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, **draw_params)
    bf = cv2.BFMatcher ()
    matches = bf.knnMatch ( des1, des2, k=2 )
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append ( [m] )
    # cv2.drawMatchesKnn expects list of lists as matches.
    img_sift = cv2.drawMatchesKnn ( img1, kp1, img2, kp2, good, None)

    #plt.imshow(img3,),plt.show()
    cv2.imshow ( "SIFT", img_sift )
    cv2.moveWindow ( "SIFT", img_sift.shape[1]*3, vertical_offset)

except Exception as e:
    print("SIFT Error: " + str(e))

try:
    #SURF
    surf = cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors
    kp1_surf, des1_surf = sift.detectAndCompute(img1,None)
    kp2_surf, des2_surf = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1_surf,des2_surf,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if use_ratio:
            if m.distance < ratio_test_value*n.distance:
                 matchesMask[i]=[1,0]
        else:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    img_surf = cv2.drawMatchesKnn(img1,kp1_surf,img2,kp2_surf,matches,None,**draw_params)
    #plt.imshow(img_surf,),plt.show()
    cv2.imshow ( "SURF", img_surf )
    cv2.moveWindow ( "SURF", img_surf.shape[1]*4, vertical_offset)
except Exception as e:
    print("SURF Error: " + str(e))


while True:
    key = cv2.waitKey ( 1 ) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord ( "q" ):
        break
# # create BFMatcher object
# bf_surf= cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck=crossCheckValue)
#
# # Match descriptors.
# matches_surf = bf_surf.match(des1_surf,des2_surf)
#
# # Sort them in the order of their distance.
# if sort_matches:
#     matches_surf = sorted(matches_surf, key = lambda x:x.distance)
#
# # Draw matches.
# img_surf = cv2.drawMatches(img1,kp1_surf,img2,kp2_surf,matches_surf[:matches_requested], None, flags=2)
# plt.axis("off")
# plt.imshow(cv2.cvtColor(img_surf, cv2.COLOR_BGR2RGB)),plt.show()