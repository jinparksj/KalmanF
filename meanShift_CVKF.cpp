//
// Created by jin on 1/31/19.
//

#include "onMouse.h"

using namespace std;
using namespace cv;
#define DIST_TH 0.4 //threshold for histogram matching

Rect selection;
bool bLButtonDown = false;

typedef enum
{
    INIT,
    CALC_HIST,
    TRACKING
} STATUS;

STATUS trackingMode = INIT;


void onMouse(int mevent, int x, int y, int flags, void* param)
{
    static Point origin;
    Mat *pMat = (Mat *)param;
    Mat image = Mat(*pMat);
    if (bLButtonDown)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = selection.x + abs(x - origin.x);
        selection.height = selection.y + abs(y - origin.y);

        selection.x = MAX(selection.x, 0);
        selection.y = MAX(selection.y, 0);
        selection.width = MIN(selection.width, image.cols);
        selection.height = MIN(selection.height, image.rows);

        selection.width -= selection.x;
        selection.height -= selection.y;
    }

    switch(mevent)
    {
        case EVENT_LBUTTONDOWN:
            origin = Point(x,y);
            selection = Rect(x, y, 0, 0);
            bLButtonDown = true;
            break;
        case EVENT_LBUTTONUP:
            bLButtonDown = false;
            if (selection.width > 0 && selection.height > 0)
                trackingMode = CALC_HIST;
            break;
    }
}


int main()
{
    VideoCapture inputVideo(0);

    if(!inputVideo.isOpened())
    {
        cout << "Can not open inputVideo!!!"<< endl;
        return 0;
    }

    Size size = Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH), (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));
    int fps = (int)(inputVideo.get(CAP_PROP_FPS));
    if (fps <= 0) fps = 24; //for camera

    Mat dstImage;
    namedWindow("dstImage");
    setMouseCallback("dstImage", onMouse, (void *)&dstImage);

    int histSize = 8;
    float valueRange[] = {0, 180}; // hue's maximum is 180.
    const float* ranges[] = {valueRange};
    int channels = 0;
    Mat hist, backProject;

    int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
    bool isColor = true;
    VideoWriter outputVideo("trackingRect.avi", fourcc, fps, size, isColor);
    if (!outputVideo.isOpened())
    {
        cout << "Cannot open outputVideo!!!" << endl;
        return 0;
    }

    if (fourcc != -1)
    {
        //for waiting for ready the camera
        imshow("dstImage", NULL);
        waitKey(100);//not working because of no window
    }

    TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 2);

    Rect trackWindow;
    int delay = 1000 / fps;
    Mat frame, hImage, hsvImage, mask;

    /////////Kalman Filter///////////

    Point2f ptPredicted;
    Point2f ptEstimated;
    Point2f ptMeasured;

    //state vector x = [ x_k, y_k, vx_k, vy_k ]^T
    KalmanFilter KF(4, 2, 0);
    Mat measurement(2, 1, CV_32F);

    float dt = 1.0;
    //Transition matrix A describes model parameters at k-1 and k
    const float A[] = {1, 0, dt, 0,
                       0, 1, 0, dt,
                       0, 0, 1, 0,
                       0, 0, 0, 1};

    memcpy(KF.transitionMatrix.data, A, sizeof(A));
    cout << "KF.transitionMatrix = " << KF.transitionMatrix << endl;

    //Initialize Kalman Parameters
    double Q = 1e-5; //process noise cov
    double R = 0.0001; //estimate of measurement variance
    const float H[] = {1, 0, 0, 0,
                       0, 1, 0, 0};

    memcpy(KF.measurementMatrix.data, H, sizeof(H));
    cout << "KF.measurementMatrix = " << KF.measurementMatrix << endl;

    setIdentity(KF.processNoiseCov, Scalar::all(Q));
    KF.processNoiseCov.at<float>(2, 2) = 0;
    KF.processNoiseCov.at<float>(3, 3) = 0;
    cout << "KF.processNoiseCov = " << KF.processNoiseCov << endl;

    setIdentity(KF.measurementNoiseCov, Scalar::all(R));
    cout << "KF.measurementNoiseCov = " << KF.measurementNoiseCov << endl;

    Mat hist1, hist2; // for histogram matching
    for (;;)
    {
        inputVideo >> frame;
        if(frame.empty())
            break;
        cvtColor(frame, hsvImage, COLOR_BGR2HSV);
        frame.copyTo(dstImage);
        hImage.create(hsvImage.size(), CV_8U);

        if(bLButtonDown && 0 < selection.width && 0 < selection.height)
        {
            Mat dstROI = dstImage(selection);
            bitwise_xor(dstROI, Scalar::all(255), dstROI);
        }
        if (trackingMode) // CALC_HIST or TRACKING
        {
            //create mask image
            int vmin = 50, vmax = 256, smin = 50;
            inRange(hsvImage, Scalar(0, smin, MIN(vmin, vmax)), Scalar(180, 256, MAX(vmin, vmax)), mask);
            //imshow("mask", mask);

            int ch[] = {0, 0};
            hImage.create(hsvImage.size(), CV_8U);
            mixChannels(&hsvImage, 1, &hImage, 1, ch, 1);
//            imshow("hImage", hImage);
            if(trackingMode == CALC_HIST)
            {
                Mat hImageROI(hImage, selection), maskROI(mask, selection);
                calcHist(&hImageROI, 1, &channels, maskROI, hist, 1, &histSize, ranges);
                hist.copyTo(hist1);
                normalize(hist1, hist1, 1.0); //for matching
                normalize(hist, hist, 0, 255, NORM_MINMAX); //for backprojection
                trackWindow = selection;
                trackingMode = TRACKING;

                //initialize the state vector (position and velocity)
                ptMeasured = Point2f(trackWindow.x  + trackWindow.width / 2.0, trackWindow.y + trackWindow.height / 2.0);
                KF.statePost.at<float>(0, 0) = ptMeasured.x;
                KF.statePost.at<float>(1, 0) = ptMeasured.y;
                KF.statePost.at<float>(2, 0) = 0;
                KF.statePost.at<float>(3, 0) = 0;

                setIdentity(KF.errorCovPost, Scalar::all(1));

            }
            Mat prediction = KF.predict(); //predict
            ptPredicted.x = prediction.at<float>(0, 0);
            ptPredicted.y = prediction.at<float>(1, 0);

            // TRACKING
            calcBackProject(&hImage, 1, &channels, hist, backProject, ranges);
            backProject &= mask;
            //bitwist_and(backProject, mask, backProject);
            //imshow("backProject", backProject);

            meanShift(backProject, trackWindow, criteria);
            Point pt1 = Point2f(trackWindow.x, trackWindow.y);
            Point pt2 = Point2f(pt1.x + trackWindow.width, pt1.y + trackWindow.height);
            rectangle(dstImage, pt1, pt2, Scalar(0, 0, 255), 2);


        }
    }



}





