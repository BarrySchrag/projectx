# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
import traceback
import sys
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

# use compareHist with HistCompMethods, EMD, compare w/wo equalization (equalizeHist)
# Derivative of the histogram over time

# Fisher transform, the inverse hyperbolic tangent functionf(r) = arctanh(r)
#   also written as (1/2)log( (1+r)/(1-r) )
#   arctanh is a normalizing and variance-stabilizing transformation
#   the results "back transformed" by using the inverse transformation, which is r = tanh(z).
#   numpy.arctanh

# Box-Cox power transformation
# scipy.stats.boxcox(x, lmbda=None, alpha=None)

# Pearson R  scipy.stats.pearsonr(x, y)

# # initialize OpenCV methods for histogram comparison
# # Note that the OpenCV implementation of Chi-Squares only takes the squared difference
# #   of each individual bin, divided by the bin count for the first histogram.
# HISTOGRAM_METHODS = (
#     ("Correlation", cv2.HISTCMP_CORREL), # Correlation  Hk=1/N∑Hk(J)
#     ("Chi-Squared", cv2.HISTCMP_CHISQR), #Chi-Square
#     ("Intersection", cv2.HISTCMP_INTERSECT),  # Intersection
#     ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA), # OpenCV computes Hellinger distance,
#        which is related to Bhattacharyya coefficient
#     ("Hellinger", cv2.HISTCMP_HELLINGER), # Hellinger distance
#     ("Chi-Squared Alternative", cv2.HISTCMP_CHISQR_ALT), #Alternative Chi-Square
#     ("Kullback-Leibler divergence", cv2.HISTCMP_KL_DIV) #Kullback-Leibler divergence
# )
#
# # This algorith takes the squared difference of each bin count,
# # divided by the sum of the bin count values, implying that large differences
# # in the bins should contribute less weight.
# def chi2_distance(histA, histB, eps=1e-10):
#     # compute the chi-squared distance
#     d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
#                       for (a, b) in zip(histA, histB)])
#     # return the chi-squared distance
#     return d
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
#                 help="Path to the directory of images")
# args = vars(ap.parse_args())
#
# # initialize the index dictionary to store the image name
# # and corresponding histograms and the images dictionary
# # to store the images themselves
# index = {}
# images = {}
#
# # loop over the image paths
# for imagePath in glob.glob(args["dataset"] + "/*.png"):
#     # extract the image filename (assumed to be unique) and
#     # load the image, updating the images dictionary
#     filename = imagePath[imagePath.rfind("/") + 1:]
#     image = cv2.imread(imagePath)
#     images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # extract a 3D RGB color histogram from the image,
#     # using 8 bins per channel, normalize, and update
#     # the index
#     hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
#                         [0, 256, 0, 256, 0, 256])
#     hist = cv2.normalize(hist).flatten()
#     index[filename] = hist
#
# # METHOD #1: UTILIZING OPENCV
# # loop over the comparison methods
# for (methodName, method) in HISTOGRAM_METHODS:
#     # initialize the results dictionary and the sort
#     # direction
#     results = {}
#     reverse = False
#
#
#     # Some similarity functions a LARGER value indicates higher similarity
#     #  Correlation and Intersection
#     # For others, a SMALLER value indicates higher similarity
#     #   Chi-Squared and Hellinger
#     # if we are using the correlation or intersection
#     # method, then sort the results in reverse order
#     if methodName in ("Correlation", "Intersection"):
#         reverse = True
#
#     for (k, hist) in index.items():
#         # compute the distance between the two histograms
#         # using the method and update the results dictionary
#         d = cv2.compareHist(index["doge.png"], hist, method)
#         results[k] = d
#
#         # sort the results
#     results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
#
#     # show the query image
#     fig = plt.figure("Query")
#     ax = fig.add_subplot(1, 1, 1)
#     ax.imshow(images["doge.png"])
#     plt.axis("off")
#
#     # initialize the results figure
#     fig = plt.figure("Results: %s" % (methodName))
#     fig.suptitle(methodName, fontsize=20)
#
#     # loop over the results
#     for (i, (v, k)) in enumerate(results):
#         # show the result
#         ax = fig.add_subplot(1, len(images), i + 1)
#         ax.set_title("%s: %.2f" % (k, v))
#         plt.imshow(images[k])
#         plt.axis("off")
#
# # show the OpenCV methods
# plt.show()
#
# # METHOD #2: UTILIZING SCIPY
# # initialize the scipy methods to compaute distances
# SCIPY_METHODS = (
#     ("Euclidean", dist.euclidean),
#     ("Manhattan", dist.cityblock),
#     ("Chebysev", dist.chebyshev))
#
# # loop over the comparison methods
# for (methodName, method) in SCIPY_METHODS:
#     # initialize the dictionary dictionary
#     results = {}
#
#     # loop over the index
#     for (k, hist) in index.items():
#         # compute the distance between the two histograms
#         # using the method and update the results dictionary
#         d = method(index["doge.png"], hist)
#         results[k] = d
#
#     # sort the results
#     results = sorted([(v, k) for (k, v) in results.items()])
#
#     # show the query image
#     fig = plt.figure("Query")
#     ax = fig.add_subplot(1, 1, 1)
#     ax.imshow(images["doge.png"])
#     plt.axis("off")
#
#     # initialize the results figure
#     fig = plt.figure("Results: %s" % (methodName))
#     fig.suptitle(methodName, fontsize=20)
#
#     # loop over the results
#     for (i, (v, k)) in enumerate(results):
#         # show the result
#         ax = fig.add_subplot(1, len(images), i + 1)
#         ax.set_title("%s: %.2f" % (k, v))
#         plt.imshow(images[k])
#         plt.axis("off")
#
# # show the SciPy methods
# plt.show()
#
# # initialize the results dictionary
# results = {}
#
# # loop over the index
# for (k, hist) in index.items():
#     # compute the distance between the two histograms
#     # using the custom chi-squared method, then update
#     # the results dictionary
#     d = chi2_distance(index["doge.png"], hist)
#     results[k] = d
#
# # sort the results
# results = sorted([(v, k) for (k, v) in results.items()])
#
# # show the query image
# fig = plt.figure("Query")
# ax = fig.add_subplot(1, 1, 1)
# ax.imshow(images["doge.png"])
# plt.axis("off")
#
# # initialize the results figure
# fig = plt.figure("Results: Custom Chi-Squared")
# fig.suptitle("Custom Chi-Squared", fontsize=20)
#
# # loop over the results
# for (i, (v, k)) in enumerate(results):
#     # show the result
#     ax = fig.add_subplot(1, len(images), i + 1)
#     ax.set_title("%s: %.2f" % (k, v))
#     plt.imshow(images[k])
#     plt.axis("off")
#
# # show the custom method
# plt.show()
class comparehistograms():

    H1 = None
    H2 = None

    # initialize OpenCV methods for histogram comparison
    # Note that the OpenCV implementation of Chi-Squares only takes the squared difference
    #   of each individual bin, divided by the bin count for the first histogram.
    HISTOGRAM_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL,0),  # Correlation  Hk=1/N∑Hk(J)
        ("Chi-Squared", cv2.HISTCMP_CHISQR,0),  # Chi-Square
        ("Intersection", cv2.HISTCMP_INTERSECT,0),  # Intersection
        ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA,0),
        # OpenCV computes Hellinger distance, which is related to Bhattacharyya coefficient
        ("Hellinger", cv2.HISTCMP_HELLINGER,0),  # Hellinger distance
        ("Chi-Squared Alternative", cv2.HISTCMP_CHISQR_ALT,0),  # Alternative Chi-Square
        ("Kullback-Leibler Divergence", cv2.HISTCMP_KL_DIV,0),  # Kullback-Leibler divergence
        ("Euclidean", 'euclidean',1),
        ("Manhattan", 'cityblock',1),
        ("Chebysev", 'chebyshev',1)
    )
    def __init__(self, H1, H2):
        self.H1, self.H2 = H1, H2

    @staticmethod
    def arrayToHistogram(array, do_normalize, hist_height, hist_width, nbins):

        r = list(filter((lambda x: x > hist_height), array))
        if len(r) > 0:
            raise Exception('Data values must be less than the intended histogram height.' + str(r) )

        # Change type
        data_shaped = np.array(array).astype(np.float32)
        hist_out = np.zeros((hist_height, hist_width), dtype=np.float32)

        # Calculate and normalise the histogram
        H1 = cv2.calcHist([data_shaped], [0], None, [nbins], [0, hist_width])
        if do_normalize == True:
            cv2.normalize(H1, H1, hist_height, cv2.NORM_MINMAX)

        return H1.flatten()

    @staticmethod
    def histogramToImage(H, flip,  hist_height, hist_width, nbins):
        bin_width = hist_width / nbins

        # create an empty image for the histogram
        image = np.zeros((hist_height, hist_width), dtype=np.float32)

        # Loop through each bin and plot the rectangle in 255 white
        for x, y in enumerate(H):
            cv2.rectangle(image, (int(x * bin_width), int(y)),
                          (int(x * bin_width + bin_width), int(hist_height)),
                          255, -1)
        # Flip upside down
        if flip == True:
            image = np.flipud(image)

        return image

    # This algorith takes the squared difference of each bin count,
    # divided by the sum of the bin count values, implying that large differences
    # in the bins should contribute less weight.
    def chi2_distance(self, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(self.H1, self.H2)])
        # return the chi-squared distance
        return d

    def filter_value(self, someList, value):
        for x, y, z in someList:
            if x == value:
                yield x, y,z

    def compare(self, method):
        # initialize the results dictionary
        results = []
        # sort direction
        reverse = False

        # which method will we use to compare the histograms?
        method_requested = list( self.filter_value( self.HISTOGRAM_METHODS, method ) )

        # test with all methods
        if len(method_requested) == 0:
            # loop over the comparison methods
            for (methodName, method, library_type) in self.HISTOGRAM_METHODS:

                # Some similarity functions a LARGER value indicates higher similarity
                #  Correlation and Intersection
                # For others, a SMALLER value indicates higher similarity
                #   Chi-Squared and Hellinger
                # if we are using the correlation or intersection
                # method, then sort the results in reverse order
                if methodName in ("Correlation", "Intersection"):
                    reverse = True
                else:
                    reverse = False

                if library_type == 0:
                    # compute the distance between the two histograms
                    # using the method and update the results dictionary
                    value = cv2.compareHist(self.H1, self.H2, method)
                    #print(methodName + ":" + str(d))
                    if reverse == True:
                        value = 1.0/(value+1e-10)
                    results.append((methodName,value))
                else:
                    value = getattr(dist, method)(self.H1,self.H2)
                    results.append((methodName,value))
                # sort the results each run
                #results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
        else:
            (method_name, method, library_type) = method_requested[0]
            if library_type == 0:
                value = cv2.compareHist(self.H1, self.H2, method)
                results.append(method_name,value)
            else:
                value = getattr(dist, method)(self.H1,self.H2)
                results.append((method_name,value))

        return results

def histogramOutput(unique_id, data_new, data_old, comparrison_type, do_normalize, hist_height, hist_width, nbins):
    H_old = comparehistograms.arrayToHistogram(data_old, do_normalize, hist_height, hist_width, nbins)
    H_new = comparehistograms.arrayToHistogram(data_new, do_normalize, hist_height, hist_width, nbins)

    result = comparehistograms(H_old, H_new).compare(comparrison_type)
    print( str(unique_id) + ' Output histogram values: '+ ','.join(map(str, H_new)))
    for (methodName, value) in result:
        print('\t' + methodName + ': {:.2f}'.format(value))

    image_new = comparehistograms.histogramToImage(H_new, True, hist_height, hist_width, nbins)
    image_old = comparehistograms.histogramToImage(H_old, True, hist_height, hist_width, nbins)
    if type(image_old) is not type(None):
        cv2.imshow(str(unique_id), image_old)
        cv2.moveWindow(str(unique_id), hist_width*unique_id+10, 0)
    if type(image_new) is not type(None):
        cv2.imshow(str(unique_id+10), image_new)
        cv2.moveWindow(str(unique_id+10), hist_width*unique_id+10, hist_height +25)


def shiftLeft(seq, n):
    n = n % len(seq)
    # left: from nth element to end, add the piece from 0 to n to the right side
    return seq[n:] + seq[:n]

def shiftRight(seq, n):
    n = n % len(seq)
    # right: back from the end to the nth element, put that on the left front of the list -n items
    return seq[-n:] + seq[:-n]

def data_from_file():
    hist_height = 360
    hist_width = 360
    nbins = 16
    do_normalize = True
    comparrison_type = 'All'
    # d1 = np.load('./media/angle_hist_shaped-69.npy') #69 fastest ball drop rate
    # d2 = np.load('./media/angle_hist_shaped-70.npy') #70 ball on the floor
    # histogramOutput(0, d1, d2, comparrison_type, do_normalize, hist_height, hist_width, nbins)
    #
    # d1 = np.load('./media/angle_hist_shaped-70.npy') #70 ball on floor
    # d2 = np.load('./media/angle_hist_shaped-71.npy') #71 ball begins to rise
    # histogramOutput(1, d1, d2, comparrison_type, do_normalize, hist_height, hist_width, nbins)
    #
    # d1 = np.load('./media/angle_hist_shaped-82.npy') # ball at arc of apogee 1
    # d2 = np.load('./media/angle_hist_shaped-83.npy') # ball begins to drop almost no motion
    # histogramOutput(2, d1, d2, comparrison_type, do_normalize, hist_height, hist_width, nbins)
    #
    # d1 = np.load('./media/angle_hist_shaped-83.npy') # ball begins to drop almost no motion
    # d2 = np.load('./media/angle_hist_shaped-84.npy') # no motion detected, 1 item
    # histogramOutput(3, d1, d2, comparrison_type, do_normalize, hist_height, hist_width, nbins)
    #
    # d1 = np.load('./media/angle_hist_shaped-84.npy') # no motion detected 1 item
    # d2 = np.load('./media/angle_hist_shaped-85.npy') # begins to drop
    # histogramOutput(4, d1, d2, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    d1 = np.load('./media/angle_hist_shaped-113.npy')
    d2 = np.load('./media/angle_hist_shaped-114.npy')
    histogramOutput(0, d1, d2, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    d1 = np.load('./media/angle_hist_shaped-114.npy')
    d2 = np.load('./media/angle_hist_shaped-115.npy')
    histogramOutput(1, d1, d2, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    d1 = np.load('./media/angle_hist_shaped-115.npy')
    d2 = np.load('./media/angle_hist_shaped-116.npy')
    histogramOutput(2, d1, d2, comparrison_type, do_normalize, hist_height, hist_width, nbins)



if __name__ == "__main__":
    data_from_file()
    cv2.waitKey(0)
    sys.exit()
    # example #:  How to use to compare data via histograms via one or all of:
    # Correlation,Chi-Squared,Intersection,Bhattacharyya,Hellinger,Chi-Squared Alternative,
    # Kullback-Leibler Divergence,Euclidean,Manhattan,Chebysev

    hist_height = 100
    hist_width = 100
    nbins = 10
    do_normalize = False
    comparrison_type = 'All' # or specify one of 10 types 'Correlation'...

    # Gas mileage for year 2000 cars from http://www.shodor.org/interactivate/activities/Histogram/
    original_data = [49, 49, 45, 45, 41, 38, 38, 38, 40, 37, 37, 34, 35, 36, 35,
                     38, 38, 32, 32, 32, 37, 31, 32, 31, 32, 30, 30, 32, 30, 30,
                     29, 28, 29, 29, 29, 30, 28, 27, 29, 30, 28, 27, 28, 27, 27,
                     29, 29, 29, 26, 27, 25, 25, 25, 25, 25, 25, 25, 26, 26, 27]
    histogramOutput(1, original_data, original_data, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    # Phase shift - What does a 1/4 phase shift right do?  15 values moved from the right side to the left
    # Should result in the same histogram, because the values have not changed
    data = shiftLeft(original_data,15)
    histogramOutput(2, data, original_data, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    data = shiftRight(original_data, 15)
    histogramOutput(3, data, original_data, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    # Scale
    scale = 2
    data = list(map((lambda x: x * scale), original_data))
    histogramOutput(4, data, original_data, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    # Scale .5
    scale =.5
    data = list(map((lambda x: x * scale), original_data))
    histogramOutput(5, data, original_data, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    # Example #2 take some data and decide directly how similar it is.
    hist_height = 100
    hist_width = 100
    nbins = 10
    do_normalize = True
    comparrison_type = 'All'

    data = original_data + original_data
    histogramOutput(6, data, original_data, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    data = original_data + original_data + original_data
    histogramOutput(7, data, original_data, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    data = original_data + original_data + original_data + original_data
    histogramOutput(8, data, original_data, comparrison_type, do_normalize, hist_height, hist_width, nbins)

    cv2.waitKey(0)
    sys.exit()
