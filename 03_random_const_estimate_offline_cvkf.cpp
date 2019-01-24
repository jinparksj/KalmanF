//
// Created by jin on 1/21/19.
//

#include <opencv2/opencv.hpp>
#include <time.h>

using namespace cv;
using namespace std;

int main()
{
    //Greg Welch and Gary Bishop, "An Introduction to the Kalman Filter", 2006
    //Estimating a Random Constant: Offline with KalmanFilter class in OpenCV

    theRNG().state = time(NULL);
    int t, count = 500;
    double x = -0.37727; // truth value

    ///////////// Kalman Filter /////////////
    double Q = 1e-5; //process variance
    double R = 0.0001; //estimation of measurement variance

    Scalar stddevR = Scalar::all(sqrt(R));
    vector<float> measurement_k(count); //the noisy measurement, z_k
    randn(measurement_k, Scalar::all(x), stddevR);

    vector<float> state_k(count); // state, x_k
    vector<float> postP(count); // the posteriori error covariance, P

    KalmanFilter KF(1, 1, 0);
    Mat measurement(1, 1, CV_32F);

    setIdentity(KF.transitionMatrix); //A = 1
    setIdentity(KF.measurementMatrix); // H = 1
    setIdentity(KF.processNoiseCov, Scalar::all(Q));
    setIdentity(KF.measurementNoiseCov, Scalar::all(R));

    //initial guesses
    setIdentity(KF.statePost, Scalar::all(0)); // x_k(0) = 0
    setIdentity(KF.errorCovPost, Scalar::all(1)); // P_k(0) = 1

    state_k[0] = KF.statePost.at<float>(0);
    postP[0] = KF.errorCovPost.at<float>(0);

    for (t = 1; t < count; t++)
    {
        Mat prediction = KF.predict(); // predict

        measurement.at<float>(0) = measurement_k[t];

        //state_k[t] = KF.statePost.at<float>(0);
        //postP[t] = KF.errorCovPost.at<float>(0);

        Mat estimate = KF.correct(measurement); //update
        // state_k[t] = estimate.at<float>(0);
        state_k[t] = KF.statePost.at<float>(0);
        postP[t] = KF.errorCovPost.at<float>(0);
    }

    // drawing values
    Mat dstImage(512, 512, CV_8UC3, Scalar::all(255));
    Size size = dstImage.size();
    namedWindow("dstImage");

    double minVal, maxVal;
    minMaxLoc(measurement_k, &minVal, &maxVal);
    double scale = size.height / (maxVal - minVal);
    cout << "measurement_k : ";
    cout << " minVal= " << minVal;
    cout << ", maxVal= " << maxVal << endl;

    // drawing the truth value, x = -0.37727
    Point pt1, pt2;
    pt1.x = 0;
    pt1.y = size.height - cvRound(scale * x - scale * minVal);

    pt2.x = size.width;
    pt2.y = size.height - cvRound(scale * x - scale * minVal);
    line(dstImage, pt1, pt2, Scalar(255, 0, 0), 2);

    //drawing the noisy measurements, measurement_k
    int step = size.width / count;
    for (t = 0; t < count; t++)
    {
        pt1.x = t * step;
        pt1.y = size.height - cvRound(scale * measurement_k[t] - scale * minVal);
        circle(dstImage, pt1, 3, Scalar(0, 255, 0), 2);
    }

    //drawing the filter estimate, state_k
    pt1.x = 0;
    pt1.y = size.height - cvRound(scale * state_k[0] - scale * minVal);
    for (t = 1; t < count; t++)
    {
        pt2.x = t * step;
        pt2.y = size.height - cvRound(scale * state_k[t] - scale * minVal);
        line(dstImage, pt1, pt2, Scalar(0, 0, 255), 2);
        pt1 = pt2;
    }
    imshow("dstImage", dstImage);

    //drawing the error covariance, postP
    Mat PImage(size.height, size.width, CV_8UC3, Scalar::all(255));
    size = PImage.size();
    namedWindow("PImage");

    minMaxLoc(postP, &minVal, &maxVal);
    scale = size.height / (maxVal - minVal);
    cout << "error covariance, postP: ";
    cout << " minVal = " << minVal;
    cout << ", maxVal = " << maxVal << endl;

    pt1.x = 0;
    pt1.y = size.height - cvRound(scale * postP[0] - scale * minVal);

    step = size.width / count;
    for (t = 1; t < count; t++)
    {
        pt2.x = t * step;
        pt2.y = size.height - cvRound(scale * postP[t] - scale * minVal);
        line(PImage, pt1, pt2, Scalar(0, 0, 255), 2);
        pt1 = pt2;
    }
    imshow("PImage", PImage);
    waitKey(0);











}