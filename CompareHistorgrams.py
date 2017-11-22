# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
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

OPENCV_METHODS = (
    cv2.HISTCMP_CORREL, # Correlation  Hk=1/N∑Hk(J)
    cv2.HISTCMP_CHISQR, #Chi-Square
    cv2.HISTCMP_INTERSECT,  # Intersection
    cv2.HISTCMP_BHATTACHARYYA, # OpenCV computes Hellinger distance, which is related to Bhattacharyya coefficient
    cv2.HISTCMP_HELLINGER, # Hellinger distance
    cv2.HISTCMP_CHISQR_ALT, #Alternative Chi-Square
    cv2.HISTCMP_KL_DIV #Kullback-Leibler divergence
)

# finds intersection of histograms
def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

# This algorith takes the squared difference of each bin count,
# divided by the sum of the bin count values, implying that large differences
# in the bins should contribute less weight.
def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to the directory of images")
args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

# loop over the image paths
for imagePath in glob.glob(args["dataset"] + "/*.png"):
    # extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()
    index[filename] = hist

# METHOD #1: UTILIZING OPENCV
# initialize OpenCV methods for histogram comparison
# Note that the OpenCV implementation of Chi-Squares only takes the squared difference
#   of each individual bin, divided by the bin count for the first histogram.
OPENCV_METHODS = (
    ("Correlation", cv2.CV_COMP_CORREL),
    ("Chi-Squared", cv2.CV_COMP_CHISQR),
    ("Intersection", cv2.CV_COMP_INTERSECT),
    ("Bhattacharyya", cv2.Hu COMP_BHATTACHARYYA),
    ("Hellinger", cv2.CV_BHATTACHARYYA))

# loop over the comparison methods
for (methodName, method) in OPENCV_METHODS:
    # initialize the results dictionary and the sort
    # direction
    results = {}
    reverse = False


    # Some similarity functions a LARGER value indicates higher similarity
    #  Correlation and Intersection
    # For others, a SMALLER value indicates higher similarity
    #   Chi-Squared and Hellinger
    # if we are using the correlation or intersection
    # method, then sort the results in reverse order
    if methodName in ("Correlation", "Intersection"):
        reverse = True

    for (k, hist) in index.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        d = cv2.compareHist(index["doge.png"], hist, method)
        results[k] = d

        # sort the results
    results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)

    # show the query image
    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(images["doge.png"])
    plt.axis("off")

    # initialize the results figure
    fig = plt.figure("Results: %s" % (methodName))
    fig.suptitle(methodName, fontsize=20)

    # loop over the results
    for (i, (v, k)) in enumerate(results):
        # show the result
        ax = fig.add_subplot(1, len(images), i + 1)
        ax.set_title("%s: %.2f" % (k, v))
        plt.imshow(images[k])
        plt.axis("off")

# show the OpenCV methods
plt.show()

# METHOD #2: UTILIZING SCIPY
# initialize the scipy methods to compaute distances
SCIPY_METHODS = (
    ("Euclidean", dist.euclidean),
    ("Manhattan", dist.cityblock),
    ("Chebysev", dist.chebyshev))

# loop over the comparison methods
for (methodName, method) in SCIPY_METHODS:
    # initialize the dictionary dictionary
    results = {}

    # loop over the index
    for (k, hist) in index.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        d = method(index["doge.png"], hist)
        results[k] = d

    # sort the results
    results = sorted([(v, k) for (k, v) in results.items()])

    # show the query image
    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(images["doge.png"])
    plt.axis("off")

    # initialize the results figure
    fig = plt.figure("Results: %s" % (methodName))
    fig.suptitle(methodName, fontsize=20)

    # loop over the results
    for (i, (v, k)) in enumerate(results):
        # show the result
        ax = fig.add_subplot(1, len(images), i + 1)
        ax.set_title("%s: %.2f" % (k, v))
        plt.imshow(images[k])
        plt.axis("off")

# show the SciPy methods
plt.show()

# initialize the results dictionary
results = {}

# loop over the index
for (k, hist) in index.items():
    # compute the distance between the two histograms
    # using the custom chi-squared method, then update
    # the results dictionary
    d = chi2_distance(index["doge.png"], hist)
    results[k] = d

# sort the results
results = sorted([(v, k) for (k, v) in results.items()])

# show the query image
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images["doge.png"])
plt.axis("off")

# initialize the results figure
fig = plt.figure("Results: Custom Chi-Squared")
fig.suptitle("Custom Chi-Squared", fontsize=20)

# loop over the results
for (i, (v, k)) in enumerate(results):
    # show the result
    ax = fig.add_subplot(1, len(images), i + 1)
    ax.set_title("%s: %.2f" % (k, v))
    plt.imshow(images[k])
    plt.axis("off")

# show the custom method
plt.show()

if __name__ == "__main__":

    hist_height = 100
    hist_width = 100
    nbins = 10
    bin_width = hist_width / nbins

    # Gas mileage for year 2000 cars from http://www.shodor.org/interactivate/activities/Histogram/
    data = [49,49,45,45,41,38,38,38,40,37,37,34,35,36,35,38,38,32,32,32,37,31,32,31,32,30,30,32,30,30,29,28,29,29,29,30,28,27,29,30,28,27,28,27,27,29,29,29,26,27,25,25,25,25,25,25,25,26,26,27]

    # Change type
    data_shaped = np.array(data).astype(np.float32)
    hist_out = np.zeros((hist_height, hist_width), dtype=np.float32 )

    # Calculate and normalise the histogram
    H1 = cv2.calcHist([data_shaped], [0], None, [nbins], [0, hist_width])
    cv2.normalize(H1, H1, hist_height, cv2.NORM_MINMAX)

    hist_values = ','.join(map(str, H1.flatten()))
    print(hist_values)

    # Angle - Loop through each bin and plot the rectangle in 255 white
    for x, y in enumerate(H1):
        cv2.rectangle(hist_image, (int(x * bin_width), int(y)),
                      (int(x * bin_width + bin_width - 1), int(hist_height)),
                      255, -1)
    # Show
    if type(hist_image) is not type(None):
        cv2.imshow("Histogram", hist_image)
        cv2.moveWindow("Histogram", 0, 0)

    compare = CompareHistograms(H1,H2,MethodOfComparrison)
    compare.result
    sys.exit ()