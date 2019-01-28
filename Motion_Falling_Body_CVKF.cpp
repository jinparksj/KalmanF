//
// Created by jin on 1/25/19.
//

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    // Digital and Kalman filtering by S.M.Bozic
    // in Wiley in New Towk, pp. 130
    // Motion of falling body

    int t = 0, count = 7;

    ////////// Kalman Filter //////////
    KalmanFilter KF(2, 1, 1); // dynamParams, measureParams, controlParams
    Mat measurement(1, 1, CV_32F);

    float g = 1.0; //We assume g = 1.0 for simplicity
    Mat controlB(1, 1, CV_32F, -g);
    cout << "controlB = " << controlB << endl;

    //initialize Kalman parameters
    setIdentity(KF.measurementMatrix); //H = {1, 0}
    cout << "KF.measurementMatrix = " << KF.measurementMatrix << endl;
    KF.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1); //A
    KF.controlMatrix =(Mat_<float>(2, 1) << 0.5, 1.0);
    cout << "KF.transitionMatrix = " << KF.transitionMatrix << endl;
    cout << "KF.controlMatrix = " << KF.controlMatrix << endl;

    double Q = 0;
    double R = 1.0; //estimate of measurement variance

    setIdentity(KF.processNoiseCov, Scalar::all(Q));
    setIdentity(KF.measurementNoiseCov, Scalar::all(R));

    //initial value of the state vector (position and velocity)
    KF.statePost.at<float>(0, 0) = 95.0;
    KF.statePost.at<float>(1, 0) = 1.0;

    //initial errors
    setIdentity(KF.errorCovPost); // P(0, 0) = 10, P(1, 1) = 1
    KF.errorCovPost.at<float>(0, 0) = 10.0;
    cout << "KF.errorCovPost = " << KF.errorCovPost << endl;

    //Measurements in the text book of S.M.Bozic
    //z[0] = 0 is a dummy one and it is not used
    float z[7] = {0, 100.0, 97.9, 94.4, 92.7, 87.3, 82.1};

    printf("t = %d: statePost = (%f, %f) : errorCovPost = (%f, %f) \n",
            t, KF.statePost.at<float>(0, 0), KF.statePost.at<float>(1, 0),
                    KF.errorCovPost.at<float>(0, 0), KF.errorCovPost.at<float>(1, 0));

    for (t = 1; t < count ; t ++)
    {
        Mat prediction = KF.predict(controlB); //predict
        measurement.at<float>(0) = z[t];

        Mat estimate = KF.correct(measurement); // update
        printf("t = %d: statePost = (%f, %f) : errorCovPost = (%f, %f) \n",
               t, KF.statePost.at<float>(0, 0), KF.statePost.at<float>(1, 0), //estimate.at.<float>(0, 0) or (1, 0)
               KF.errorCovPost.at<float>(0, 0), KF.errorCovPost.at<float>(1, 0));
    }
    return 0;
}