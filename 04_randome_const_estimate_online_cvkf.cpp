//
// Created by jin on 1/24/19.
//

#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;
using namespace cv;

int main()
{
    //Greg Welch and Gary Bishop, "An Introduction to the Kalman Filter", 2006
    //Estimating a Random Constant: Online with KalmanFilter class in OpenCV

    theRNG().state = time(NULL);
    int t = 0, count = 100;
    double x = -0.37727; // truth value


    ////////// Kalman Filter //////////

    double Q = 1e-5; // process variance
    double R = 0.00001; // estimate of measurement variance
    Scalar stddevR = Scalar::all(sqrt(R));

    //for drawing
    vector<float> state_k(count); //state, x_k
    vector<float> postP(count); //the posteriori error covariance, P
    vector<float> measurement_k(count); //the noisy measurements, z_k

    KalmanFilter KF(1, 1, 0);
    Mat measurement(1, 1, CV_32F);

    setIdentity(KF.transitionMatrix); //A = 1
    setIdentity(KF.measurementMatrix); //H = 1
    setIdentity(KF.processNoiseCov, Scalar::all(Q));
    setIdentity(KF.measurementNoiseCov, Scalar::all(R));






}