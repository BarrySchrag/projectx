/**
 * \file flowWebCam.cpp
 * \brief Optical flow demo using OpenCV VideoCapture to compute flow from webcam.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 TODO
 (py3cv4) barry@LenovoY70:~/PycharmProjects/projectx/optical-flow-filter/demos/flowWebCam/build$ make
(py3cv4) barry@LenovoY70:~/PycharmProjects/projectx/optical-flow-filter/demos/flowWebCam/build ./flowWebCam -i 0 -f RGB -v
 */

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <termio.h>
#include <unistd.h>

#include <flowfilter/gpu/flowfilter.h>
#include <flowfilter/gpu/display.h>

using namespace std;
using namespace cv;
using namespace flowfilter;
using namespace flowfilter::gpu;

bool kbhit(void)
{
    struct termios original;
    tcgetattr(STDIN_FILENO, &original);
    struct termios term;
    memcpy(&term, &original, sizeof(term));
    term.c_lflag &= ~ICANON;
    tcsetattr(STDIN_FILENO, TCSANOW, &term);
    int characters_buffered = 0;
    ioctl(STDIN_FILENO, FIONREAD, &characters_buffered);
    tcsetattr(STDIN_FILENO, TCSANOW, &original);
    bool pressed = (characters_buffered != 0);
    return pressed;
}

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

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
/* 
Parameters indicate 
 -i = Input: One of Camera# | Url | file name
 -f = Format: One of RGBDegrees, RGBRadians, HSVDegree, HSVRadians, or XYG
 -v = view the processing in a window, 
 -o = file output
 -d = drawing vectors at N pixel spacing
 -t = display timing in the UI 
 -r = resize the input to conform to a percentage of the original size
 -c = fourcc codec to apply to the output file

 Arguments for only viewing an input, as RGB, with no file output
 ./flowWebCam -i http://192.168.1.5:8080/video -f RGB -v

  Arguments for converting an input, creating an output file in HSV Degree oriented format, resizing 33%
 ./flowWebCam -i inputFile.mp4 -o HSVDegrees.avi -f HSVDegrees -r 33

 Arguments for only converting an input, creating an output file in HSV Radians format, resizing 50%
 ./flowWebCam -i inputFile.mp4 -o HSVRadians.avi -f HSVRadians -r 50 

Arguments for only viewing an input, as RGB, where color is desaturated and the displayed color is
 the degree angle and magnitude of the motion vector.
 ./flowWebCam -i http://192.168.1.5:8080/video -f RGB -v -d 8 -t 

*/

int main(int argc, char** argv) {
    cout  << "Starting Processing args: " <<  argc << endl;
    InputParser input(argc, argv);
    
    if (argc < 4)
    {
        cout << "Not enough parameters.  -i <SourceStream> -o <DestinationFile> -f <RGB|HSV|XYG> -d <draw vectors with integer speration> -v -r <resize percent>" << endl;
        return -1;
    }
    int cameraIndex = 0;
    
    int flowmapSeperation = 0;
    if(input.cmdOptionExists("-d")){
        cout  << "Processing with Flowmap. " << endl;
        flowmapSeperation = atoi(input.getCmdOption("-d").c_str());
    }
    bool visualOutput = false;
    if(input.cmdOptionExists("-v")){
        cout  << "Detected visualize. " << endl;
        visualOutput=true;
    }
    bool timeOutput = false;
    if(input.cmdOptionExists("-t")){
        cout  << "Detected time display option. "  << endl;
        timeOutput=true;
    }
    const std::string &source = input.getCmdOption("-i");
    if (source.empty()){
         cout  << "-i paramater <input stream> is required. Camera #, or URl, or file are supported." << endl;
         return -1;
    }
    const std::string &outputFile = input.getCmdOption("-o");
    if(!outputFile.empty()){
        cout  << "Detected output file requested: " << outputFile << endl;
    }
    const std::string &outputType = input.getCmdOption("-f");
    if (outputType.empty()){
         cout  << "-f format paramater <RGB|HSVDegrees|HSVRadians> is required. " << endl;
         return -1;
    }
    // Lossless codecs 
    // FFMPEG FFV1  - Looks acceptable @ 4.9MB
    // Huffman HFYU - No good for viewing
    // Lagarith LAGS - Cant find a version
    // DIB @ 17MB
    string fourcc = "FFV1";
    const std::string &codecCode = input.getCmdOption("-c");
    if (codecCode.empty()){
        cout  << "-c codec is set as " << fourcc << endl;
    }

    bool outputTypeRGBDegrees = false;
    if(strcmp(outputType.c_str(), "RGBDegrees") == 0){  // returns true if match
        outputTypeRGBDegrees=true;
    }
     bool outputTypeRGBRadians = false;
    if(strcmp(outputType.c_str(), "RGBRadians") == 0){  // returns true if match
        outputTypeRGBRadians=true;
    }
    bool outputTypeHSVDegrees = false;
    if(strcmp(outputType.c_str(), "HSVDegrees") == 0){  // returns true if match
        outputTypeHSVDegrees=true;
    }
    bool outputTypeHSVRadians = false;
    if(strcmp(outputType.c_str(), "HSVRadians") == 0){  // returns true if match
        outputTypeHSVRadians=true;
    }
    
    cout  << "Attempting Open: " << source << endl;
    VideoCapture inputVideo;

     // if user provides camera index
    if(source.length()==1)
    {
        cameraIndex = atoi(source.c_str());
        inputVideo.open(cameraIndex); 
    } else
    {
        inputVideo.open(source);  
    }
    // Open input
    if (!inputVideo.isOpened())
    {
        cout  << "Could not open the input video: " << source << endl;
        return -1;
    }
    cout  << "Opened source file: " << source << endl;

    Mat frameOriginal,frame;
    //  capture a frame to get image width and height
    inputVideo >> frameOriginal;

    double resizePercentage = 100.0;
    bool resizeImage = false;
     if(input.cmdOptionExists("-r")){
        cout  << "Processing input with resize percentage: " << resizePercentage << endl;
        string s = input.getCmdOption("-r");
        resizePercentage = atof(s.c_str())/100.0;
        resizeImage = true;
        cv::resize(frameOriginal, frame, cv::Size(), resizePercentage, resizePercentage);
    } else 
    {
        frame = frameOriginal;
    }

    Size S = frame.size(); 
    VideoWriter outputVideo;
    char const *c = codecCode.data();
    int fourCC = VideoWriter::fourcc(c[0],c[1],c[2],c[3]);  // Open the output
   
    double fps = inputVideo.get(CAP_PROP_FPS);
    if(!outputFile.empty())
    {
        cout  << "New File: " << outputFile.c_str() << endl;
        outputVideo.open(outputFile.c_str(), fourCC, fps, S, true);
        
        if (!outputVideo.isOpened())
        {
            cout  << "Could not open the output video for write: " << outputFile.c_str() << endl;
            return -1;
        }
    }
    cout << "Input video frame resolution: Width=" << S.width << "  Height=" << S.height
         << " of nr#: " << inputVideo.get(CAP_PROP_FRAME_COUNT) << " frames. "<< endl;
  
    bool is_colored = false;
    if (frame.channels() == 3) {
        is_colored = true;
        cout  << "Video is 3 channel: " << endl;
    } else
    {
        cout  << "Video is 1 channel: " << endl;
    }
    int width = frame.cols;
    int height = frame.rows;
    cout << "frame shape: [" << height << ", " << width << "]" << endl;
 
    Mat frameGray(height, width, CV_8UC1);
    Mat fcolor(height, width, CV_8UC4);

    // structs used to wrap cv::Mat images and transfer to flowfilter
    image_t hostImageGray;
    //image_t hostFlowColor;

    wrapCVMat(frameGray, hostImageGray);
    //wrapCVMat(fcolor, hostFlowColor);

    //#################################
    // Filter parameters
    //#################################
   
    //float maxflow = 250.0f;
    float maxflow = 16.0f;//40
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
    cv::Mat frameBGR, frameHSV, frameLAB, frameRGB, frameRGBResized, frameMerged;
    cv::Mat magnitude, angle, threePlaneMat[3];
    double mag_max;
    vector<Mat> spl;
    if(visualOutput) namedWindow( "optical flow", cv::WINDOW_AUTOSIZE);

    // Capture loop
    for(;;) {
        if(timeOutput ) cout << "elapsed time: " << filter.elapsedTime() << " ms" << endl;

        // capture a new frame and convert to gray scale (uint8)
        inputVideo >> frameOriginal;
         if (frameOriginal.empty()) break; // check if at end
        if (resizeImage)   cv::resize(frameOriginal, frame, cv::Size(), resizePercentage, resizePercentage);
    
        if (is_colored == true) {
            cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        } else
        {
            frameGray = frame;
        }
        
        // Transfer image to flow filter and compute
        filter.loadImage(hostImageGray);
        filter.compute();

        // transfer the optical flow from GPU to host memory allocated by flowHost cvMat.
        // After this, optical flow values can be accessed using OpenCV pixel
        // access methods.
        filter.downloadFlow(flowHostWrapper);

        split(flowHost,spl);

        if(outputTypeRGBDegrees || outputTypeHSVDegrees)
        {
            cvtColor(fcolor, frameBGR, cv::COLOR_RGBA2BGR);
            bitwise_not(frameBGR,frameBGR);

            // Calculate angle and magnitude, last param true = degrees, false = radians
            cv::cartToPolar(spl[0], spl[1], magnitude, angle, true);

            // In OpenCV, value range for 'hue', 'saturation' and 'value' are respectively 0-179, 0-255 and 0-255.
            // Set the Hue 
            threePlaneMat[0] = angle; // * 180.0 / CV_PI / 2; // suspicious radian conversion
            threePlaneMat[0].convertTo(threePlaneMat[0], CV_8U);
    
            // Find the max magnitude
            cv::minMaxLoc(magnitude, 0, &mag_max);
            magnitude.convertTo(magnitude, -1, 1.0 / mag_max); 
            normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);

            threePlaneMat[1] = magnitude;
            threePlaneMat[1].convertTo(threePlaneMat[1], CV_8U);

            threePlaneMat[2] = frameGray;
             // merge the wrapper for export
            cv::merge(threePlaneMat, 3, frameMerged);
        }
        if(outputTypeRGBRadians || outputTypeHSVRadians)
        {
            cvtColor(fcolor, frameBGR, cv::COLOR_RGBA2BGR);
            bitwise_not(frameBGR,frameBGR);

            // Calculate angle and magnitude, last param true = degrees, false = radians
            cv::cartToPolar(spl[0], spl[1], magnitude, angle, false);

            // In OpenCV, value range for 'hue', 'saturation' and 'value' are respectively 0-179, 0-255 and 0-255.
            // Set the Hue 
            threePlaneMat[0] = angle * 180.0 / CV_PI / 2; // suspicious radian conversion
            threePlaneMat[0].convertTo(threePlaneMat[0], CV_8U);
    
            // Find the max magnitude
            cv::minMaxLoc(magnitude, 0, &mag_max);
            magnitude.convertTo(magnitude, -1, 1.0 / mag_max); 
            normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);

            threePlaneMat[1] = magnitude;
            threePlaneMat[1].convertTo(threePlaneMat[1], CV_8U);

            threePlaneMat[2] = frameGray;
             // merge the wrapper for export
            cv::merge(threePlaneMat, 3, frameMerged);
        }
               
        // If a UI is requested, update it,exiting if a key is pressed.
        if(visualOutput) if(waitKey(1) >= 0) break;

       
        // Converts from HSV color space of angle and magnitude, to RGB
        if(outputTypeRGBDegrees | outputTypeRGBRadians)
        {
            cvtColor(frameMerged, frameRGB, cv::COLOR_HSV2RGB);
            if(flowmapSeperation > 0) drawOptFlowMapF(flowHost, frameRGB, flowmapSeperation, Scalar(0,255,0));
        
            if(!outputFile.empty()) {
                outputVideo << frameRGB;
            } 
            if(visualOutput) imshow("optical flow", frameRGB);
        } 
        // Keeps fame in RGB format
        if(outputTypeHSVDegrees | outputTypeHSVRadians)
        {
            if(flowmapSeperation > 0) drawOptFlowMapF(flowHost, frameRGB, flowmapSeperation, Scalar(0,255,0));

            if(!outputFile.empty()) {
                outputVideo << frameMerged;
            }
            if(visualOutput) imshow("optical flow", frameMerged);
        }
        // detect user exit in the case of no opencv UI
        if( kbhit() ) {
            getchar();
            break;
        }
    }
   
    if(!outputFile.empty())
    {
        if(outputTypeRGBDegrees | outputTypeRGBRadians) {
            cout  << "Wrote RGB: " << fps << " fps" << endl;
        } 
        if(outputTypeHSVDegrees | outputTypeHSVRadians){
            cout  << "Wrote HSV: " << fps << " fps" <<  endl;
        }
    }
    
    cout << "Exiting." << endl;
    return 0;
}

// Mat3b M;
// Mat3b hsv;
// cv::Mat frameMerged;
// vector<Mat> spl;
// void on_trackbarH(int val, void*)
// {
//     split(hsv,spl);
//     spl[0].setTo(val);//0=180
//     merge(spl, hsv);
//     // hsv.setTo(Scalar(hue, 255, 255));
//     cvtColor(hsv, M, COLOR_HSV2BGR);
//     imshow("HSV", M);
//     cout << "H" << val << endl;
// }
// void on_trackbarS(int val, void*)
// {
//     split(hsv,spl);
//     spl[1].setTo(val);
//     merge(spl, hsv);
//      //hsv.setTo(Scalar(hue, 255, 255));
//     cvtColor(hsv, M, COLOR_HSV2BGR);
//     imshow("HSV", M);
//     cout << "S" << val << endl;
// }
// void on_trackbarV(int val, void*)
// {
//     split(hsv,spl);
//     spl[2].setTo(val);
//     merge(spl, hsv);
//     cvtColor(hsv, M, COLOR_HSV2BGR);
//     imshow("HSV", M);
//     cout << "V" << val << endl;
// }

// int main(int argc, char** argv)
// {
//     // Init images
//     M = Mat3b(100, 300, Vec3b(0,0,0));
//     cvtColor(M, hsv, COLOR_BGR2HSV);

//     /// Initialize values
//     int sliderH = 0;
//     int sliderS = 255;
//     int sliderV = 255;

//     /// Create Windows
//     namedWindow("HSV", 1);

//     createTrackbar("H", "HSV", &sliderH, 180, on_trackbarH);
//     createTrackbar("S", "HSV", &sliderS, 360, on_trackbarS);
//     createTrackbar("V", "HSV", &sliderV, 360, on_trackbarV);

//     // Show some stuff
//     on_trackbarH(sliderH, NULL); 
//     on_trackbarH(sliderS, NULL); 
//     on_trackbarH(sliderV, NULL); 


//     /// Wait until user press some key
//     waitKey(0);
//     return 0;
// }