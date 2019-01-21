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



}