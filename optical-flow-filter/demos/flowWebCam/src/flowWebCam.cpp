/**
 * \file flowWebCam.cpp
 * \brief Optical flow demo using OpenCV VideoCapture to compute flow from webcam.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <flowfilter/gpu/flowfilter.h>
#include <flowfilter/gpu/display.h>

using namespace std;
using namespace cv;
using namespace flowfilter;
using namespace flowfilter::gpu;


void wrapCVMat(Mat& cvMat, image_t& img) {

    img.height = cvMat.rows;
    img.width = cvMat.cols;
    img.depth = cvMat.channels();
    img.pitch = cvMat.cols*cvMat.elemSize();
    img.itemSize = cvMat.elemSize1();
    img.data = cvMat.ptr();
}

void drawOptFlowMapF(const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
    for (int y = 0; y < cflowmap.rows; y += step)
        for (int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at< Point2f>(y, x);
            line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                color);
            circle(cflowmap, Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), 1, color, -1);
        }
}
void displayF(Mat flow)
{
    //extraxt x and y channels
    Mat xy[2]; //X,Y
    split(flow, xy);

    //calculate angle and magnitude
    Mat magnitude, angle;
    cartToPolar(xy[0], xy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    //build hsv image
    Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    merge(_hsv, 3, hsv);

    //convert to BGR and show
    Mat bgr;//CV_32FC3 matrix
    cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    imshow("optical flow", bgr);
    //imwrite("c://resultOfOF.jpg", bgr);
    //waitKey(0);
}

/**
 * MODE OF USE
 * ./flowWebCam <cameraIndex>
 *
 * where <cameraIndex> is an integer indicating the camera used
 * to capture images. Defaults to 0;
 *
 */
// int mainOriginal(int argc, char** argv) {

//     int cameraIndex = 0;

//     // if user provides camera index
//     if(argc > 1) {
//         cameraIndex = atoi(argv[1]);
//     }

//     VideoCapture cap(cameraIndex); // open the default camera
//     if(!cap.isOpened()){
//         return -1;
//     }
    
//     Mat frame;

//     //  capture a frame to get image width and height
//     cap >> frame;
//     int width = frame.cols;
//     int height = frame.rows;
//     cout << "frame shape: [" << height << ", " << width << "]" << endl;

//     Mat frameGray(height, width, CV_8UC1);
//     Mat fcolor(height, width, CV_8UC4);

//     // structs used to wrap cv::Mat images and transfer to flowfilter
//     image_t hostImageGray;
//     image_t hostFlowColor;

//     wrapCVMat(frameGray, hostImageGray);
//     wrapCVMat(fcolor, hostFlowColor);

//     //#################################
//     // Filter parameters
//     //#################################
//     float maxflow = 40.0f;
//     //float maxflow = 250.0f;
//     //vector<float> gamma = {500.0f, 50.0f, 5.0f};
//     vector<float> gamma = {50.0f, 5.0f};
//     //vector<int> smoothIterations = {2, 8, 20};
//     vector<int> smoothIterations = {2, 16};

//     //#################################
//     // Filter creation with
//     // 3 pyramid levels
//     //#################################
//     PyramidalFlowFilter filter(height, width, 2);
//     filter.setMaxFlow(maxflow);
//     filter.setGamma(gamma);
//     filter.setSmoothIterations(smoothIterations);

//     //#################################
//     // To access optical flow
//     // on the host
//     //#################################
//     Mat flowHost(height, width, CV_32FC2);
//     image_t flowHostWrapper;
//     wrapCVMat(flowHost, flowHostWrapper);

//     // Color encoder connected to optical flow buffer in the GPU
//     FlowToColor flowColor(filter.getFlow(), maxflow);

//     cv::Mat frameBGR, frameHSV, frameLAB, frameRGB, frameMerged;
//     cv::Mat magnitude, angle, hsv[3];
//     double mag_max, angle_max;
//     vector<Mat> spl;

//     namedWindow( "image", cv::WINDOW_AUTOSIZE);
//     namedWindow( "optical flowBGR", cv::WINDOW_AUTOSIZE);
//     namedWindow( "optical flowRGB", cv::WINDOW_AUTOSIZE);

//     // Capture loop
//     for(;;) {

//         // capture a new frame from the camera
//         // and convert it to gray scale (uint8)
//         cap >> frame;
//         cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        
//         // transfer image to flow filter and compute
//         filter.loadImage(hostImageGray);
//         filter.compute();

//  //       cout << "elapsed time: " << filter.elapsedTime() << " ms" << endl;

//         // transfer the optical flow from GPU to
//         // host memory allocated by flowHost cvMat.
//         // After this, optical flow values
//         // can be accessed using OpenCV pixel
//         // access methods.
//         filter.downloadFlow(flowHostWrapper);

//         // computes color encoding (RGBA) and download it to host
//         flowColor.compute();
//         flowColor.downloadColorFlow(hostFlowColor);
        
//         // cvtColor(fcolor, fcolor, COLOR_RGBA2BGRA);
    
//         cvtColor(fcolor, frameBGR, cv::COLOR_RGBA2BGR);
//         bitwise_not(frameBGR,frameBGR);

//         imshow("image", frameGray);
//         imshow("optical flowBGR", frameBGR);

//         // HSV TEST: zero the S channel and then reconvert to RGB, display and view
//         // In OpenCV, value range for 'hue', 'saturation' and 'value' are respectively 0-179, 0-255 and 0-255.
//         // cvtColor(frameBGR, frameHSV, cv::COLOR_BGR2HSV);
//         split(flowHost,spl);

//         // calculate angle and magnitude 
//         cv::cartToPolar(spl[0], spl[1], magnitude, angle);

//         // Set the Hue 
//         hsv[0] = angle * 180.0 / CV_PI / 2.0 ;
//         hsv[0].convertTo(hsv[0], CV_8U);
//         cv::minMaxLoc(hsv[0], 0, &angle_max);

//         cv::minMaxLoc(magnitude, 0, &mag_max);
//         magnitude.convertTo(magnitude, -1, 1.0 / mag_max);
//         // enable full range
//         normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);

//         hsv[1] = magnitude;
//         hsv[1].convertTo(hsv[1], CV_8U);
        
//         Scalar tempVal = mean( hsv[1] );
//         float myMAtMean = tempVal.val[0];
//         cout<<myMAtMean<<endl;

//         // To visualize that only the vectors exist in HSV, enable these lines to view
//         //hsv[2] = Mat::ones(angle.rows, angle.cols, CV_8U);//magnitude; // white screen 255.
//         //bitwise_not(hsv[2],hsv[2]);
        
//         // else set the V plane to the greyscale image
//         hsv[2] = frameGray;
        
//         cv::merge(hsv,3,frameMerged);
//         cvtColor(frameMerged, frameRGB, cv::COLOR_HSV2BGR);
// // Enable to draw visualize the flowmap
// //drawOptFlowMapF(flowHost, frameRGB, 8, Scalar(0,255,0));
//         imshow("optical flowRGB", frameRGB);

//         if(waitKey(1) >= 0) break;
//     }

//     // the camera will be deinitialized automatically in VideoCapture destructor
//     return 0;
// }
int main(int argc, char** argv) {
    cout  << "Starting Processing. " << endl;

    if (argc != 2)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }
    const string source      = argv[1];           // the source file name
    const bool askOutputType = 0;  // If false it will use the inputs codec type
    
    cout  << "Attempting Open: " << source << endl;
    VideoCapture inputVideo(source);              // Open input
    if (!inputVideo.isOpened())
    {
        cout  << "Could not open the input video: " << source << endl;
        return -1;
    }
    cout  << "Opened file: " << source << endl;
    string::size_type pAt = source.find_last_of('.');                  // Find extension point
    const string NAME = source.substr(0, pAt) + ".avi";   // Form the new name with container
    cout  << "New File: " << NAME << endl;
    int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
    Size S = Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter outputVideo;                                        // Open the output
    if (askOutputType)
        outputVideo.open(NAME, ex=-1, inputVideo.get(CAP_PROP_FPS), S, true);
    else
        outputVideo.open(NAME, ex, inputVideo.get(CAP_PROP_FPS), S, true);
    cout  << "fourcc:: " << EXT << endl;
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << NAME << endl;
        return -1;
    }
    cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
         << " of nr#: " << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;
    cout << "Input codec type: " << EXT << endl;

    
    Mat frame;

    //  capture a frame to get image width and height
    inputVideo >> frame;
    int width = frame.cols;
    int height = frame.rows;
    cout << "frame shape: [" << height << ", " << width << "]" << endl;

    Mat frameGray(height, width, CV_8UC1);
    Mat fcolor(height, width, CV_8UC4);

    // structs used to wrap cv::Mat images and transfer to flowfilter
    image_t hostImageGray;
    image_t hostFlowColor;

    wrapCVMat(frameGray, hostImageGray);
    wrapCVMat(fcolor, hostFlowColor);

    //#################################
    // Filter parameters
    //#################################
    float maxflow = 40.0f;
    //float maxflow = 250.0f;
    //vector<float> gamma = {500.0f, 50.0f, 5.0f};
    vector<float> gamma = {50.0f, 5.0f};
    //vector<int> smoothIterations = {2, 8, 20};
    vector<int> smoothIterations = {2, 16};

    //#################################
    // Filter creation with
    // 3 pyramid levels
    //#################################
    PyramidalFlowFilter filter(height, width, 2);
    filter.setMaxFlow(maxflow);
    filter.setGamma(gamma);
    filter.setSmoothIterations(smoothIterations);

    //#################################
    // To access optical flow
    // on the host
    //#################################
    Mat flowHost(height, width, CV_32FC2);
    image_t flowHostWrapper;
    wrapCVMat(flowHost, flowHostWrapper);

    // Color encoder connected to optical flow buffer in the GPU
    FlowToColor flowColor(filter.getFlow(), maxflow);

    cv::Mat frameBGR, frameHSV, frameLAB, frameRGB, frameMerged;
    cv::Mat magnitude, angle, hsv[3];
    double mag_max, angle_max;
    vector<Mat> spl;

    namedWindow( "image", cv::WINDOW_AUTOSIZE);
    namedWindow( "optical flowBGR", cv::WINDOW_AUTOSIZE);
    namedWindow( "optical flowRGB", cv::WINDOW_AUTOSIZE);

    // Capture loop
    for(;;) {

        // capture a new frame from the camera
        // and convert it to gray scale (uint8)
        inputVideo >> frame;
        if (frame.empty()) break; // check if at end
        
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        
        // transfer image to flow filter and compute
        filter.loadImage(hostImageGray);
        filter.compute();

        cout << "elapsed time: " << filter.elapsedTime() << " ms" << endl;

        // transfer the optical flow from GPU to
        // host memory allocated by flowHost cvMat.
        // After this, optical flow values
        // can be accessed using OpenCV pixel
        // access methods.
        filter.downloadFlow(flowHostWrapper);

        // computes color encoding (RGBA) and download it to host
        flowColor.compute();
        flowColor.downloadColorFlow(hostFlowColor);
        
        // cvtColor(fcolor, fcolor, COLOR_RGBA2BGRA);
    
        cvtColor(fcolor, frameBGR, cv::COLOR_RGBA2BGR);
        bitwise_not(frameBGR,frameBGR);

        //imshow("image", frameGray);
        //imshow("optical flowBGR", frameBGR);

        // HSV TEST: zero the S channel and then reconvert to RGB, display and view
        // In OpenCV, value range for 'hue', 'saturation' and 'value' are respectively 0-179, 0-255 and 0-255.
        // cvtColor(frameBGR, frameHSV, cv::COLOR_BGR2HSV);
        split(flowHost,spl);

        // calculate angle and magnitude 
        cv::cartToPolar(spl[0], spl[1], magnitude, angle);

        // Set the Hue 
        hsv[0] = angle * 180.0 / CV_PI / 2.0 ;
        hsv[0].convertTo(hsv[0], CV_8U);
        cv::minMaxLoc(hsv[0], 0, &angle_max);

        cv::minMaxLoc(magnitude, 0, &mag_max);
        magnitude.convertTo(magnitude, -1, 1.0 / mag_max);
        // enable full range
        normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);

        hsv[1] = magnitude;
        hsv[1].convertTo(hsv[1], CV_8U);
        
        //Scalar tempVal = mean( hsv[1] );
        //float myMAtMean = tempVal.val[0];
        //cout<<myMAtMean<<endl;

        // To visualize that only the vectors exist in HSV, enable these lines to view
        //hsv[2] = Mat::ones(angle.rows, angle.cols, CV_8U);//magnitude; // white screen 255.
        //bitwise_not(hsv[2],hsv[2]);
        
        // else set the V plane to the greyscale image
        hsv[2] = frameGray;
        
        cv::merge(hsv,3,frameMerged);
        cvtColor(frameMerged, frameRGB, cv::COLOR_HSV2BGR);
// Enable to draw visualize the flowmap
//drawOptFlowMapF(flowHost, frameRGB, 8, Scalar(0,255,0));
        //imshow("optical flowRGB", frameRGB);

       // if(waitKey(1) >= 0) break;
       // outputVideo.write(res); //save or
       outputVideo << frameRGB;
    }
    cout << "Finished writing" << endl;
    return 0;
}
