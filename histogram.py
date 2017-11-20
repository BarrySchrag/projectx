import cv2
import numpy as np

histname = "hist"
#Create window to display image
cv2.namedWindow(histname)

hist_height = 256 #np.iinfo(np.int32).max #256
hist_width = 360
nbins = 16
bin_width = hist_width/nbins

#Create array for the bins
#bins = np.arange(nbins,dtype=np.int32).reshape(nbins,1)

while True:

    #img = cv2.imread("./media/testpattern2.jpg",0)
    img = np.random.randint(0,255, size=(1, 360)).astype(np.uint8)#uint8)
    img = np.array([0.0, 0.0, 0.0, 0.0, 255.96375653207352, 284.03624346792645, 270.0, 274.3987053549955,
                 278.7461622625552,275.19442890773485, 278.7461622625552, 278.7461622625552]).astype(np.float32)

    #print(img.shape)
    #Create an empty image for the histogram
    h = np.zeros((hist_height,hist_width),dtype=np.float32)

    #Create array for the bins
    #bins = np.arange(nbins,dtype=np.int32).reshape(nbins,1)

     #Calculate and normalise the histogram
    hist_item = cv2.calcHist([img],[0],None,[nbins],[0,hist_width])
    cv2.normalize(hist_item,hist_item,hist_height,cv2.NORM_MINMAX)
    #hist=np.int32(np.around(hist_item))
    #pts = np.column_stack((bins,hist))
    print(','.join ( map ( str, hist_item.flatten())))

    #Loop through each bin and plot the rectangle in white
    for x,y in enumerate(hist_item):
        cv2.rectangle(h,(int(x*bin_width),int(y)),
                     (int(x*bin_width + bin_width-1),int(hist_height)),
                      255,1)
    #Show the histogram
    cv2.imshow(histname,h)

    key = cv2.waitKey(1)
    if key == ord ( "q" ) or key == 27:
        break