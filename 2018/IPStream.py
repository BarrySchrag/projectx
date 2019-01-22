# Using Android IP Webcam video .jpg stream in Python2 OpenCV3
# IP Webcam App -> https://play.google.com/store/apps/details?id=com.pas.webcam

import urllib.request
import cv2
import numpy as np
import time

import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

def cuda_call(function_name, a, b):
    multiply_them = mod.get_function(function_name)

    dest = np.zeros_like(a)
    multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400, 1, 1), grid=(1, 1))
    return dest - a * b

def print_shape(img):
    # get dimensions of image
    dimensions = img.shape

    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    #channels = img.shape[2]

    print('Image Dimension    : ', dimensions)
    print('Image Height       : ', height)
    print('Image Width        : ', width)
    #print('Number of Channels : ', channels)
    print('----------------END')


def CalcOpticFlow(prvs, next):
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

url = 'http://192.168.1.15:8080/shot.jpg'

imgResp = urllib.request.urlopen(url)
imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
img = cv2.imdecode(imgNp, -1)
img = cv2.transpose(img)
vertical_img = img.copy()
vector_img = img.copy()
prvs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(img)
hsv[...,1] = 255
fps=0
frame_count=0
while True:
    timer = cv2.getTickCount()
    # Use urllib to get the image and convert into a cv2 usable format
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    vertical_img = cv2.transpose(img)
    #print_shape(vertical_img)
    next = cv2.cvtColor(vertical_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('IPWebcamVertical', vertical_img)
    #print_shape(prvs)
    #print_shape(next)

    vector_img = CalcOpticFlow(prvs, next)
    cv2.imshow('IPWebcamVectors', vector_img)

    # put the image on screen
    #cv2.imshow('IPWebcamOriginal', img)


    # To give the processor some less stress
    # time.sleep(0.1)
    #a = np.random.randn(400).astype(np.float32)
    #b = np.random.randn(400).astype(np.float32)

    #cuda_result = cuda_call('multiply_them', a+frame_count, b)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prvs = next

    # output fps
    if(frame_count%10 == 0):
        print ('fps:{0:.1f}'.format(fps))
        #print (cuda_result)

    # calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    frame_count+=1


# https://www.facebook.com/mrlunk