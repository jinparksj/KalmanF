//
// Created by jin on 1/19/19.
//

#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;
using namespace cv;

int main()
{
    //Greg Welch and Gary Bishop, "An Introduction to the Kalman Filter", 2006
    //Estimating a Random Constant: Offline

    theRNG().state = time(NULL);
    //random number - normal/Gaussian distribution
    //default random generator with state of time -> every execution has different random number

    int t, count = 100;
    double x = -0.377; //truth value

    ///////// Kalman Filter /////////
    double Q = 1e-5; //process variance
    double R = 0.0001; //estimate of measurement variance

    Scalar stddevR = Scalar::all(0.1);
    vector<float> measurement_k(count);



}