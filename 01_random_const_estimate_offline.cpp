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

    int t, count = 500;
    double x = -0.37727; //truth value

    ///////// Kalman Filter /////////
    double Q = 1e-5; //process variance
    double R = 1; //estimate of measurement variance

    Scalar stddevR = Scalar::all(sqrt(R));
    vector<float> measurement_k(count); // the noisy measurements, z_k
    randn(measurement_k, Scalar::all(x), stddevR);

    vector<float> state_k(count); // state, x_k
    vector<float> predict_k(count); //predict, x'_k
    vector<float> postP(count); // the posteriori error covariance P
    vector<float> preP(count); // the priori error covariance P' or M
    vector<float> K(count); // kalman gain, K

    //initial guesses
    state_k[0] = 0.0;
    postP[0] = 1.0;
    for (t = 1; t < count; t++)
    {
        //predict
        predict_k[t] = state_k[t-1];
        preP[t] = postP[t-1] + Q;

        //update
        K[t] = preP[t] / (preP[t] + R);
        state_k[t] = predict_k[t] + K[t] * (measurement_k[t] - predict_k[t]);
        postP[t] = (1 - K[t]) * preP[t];
    }

    //drawing values
    Mat dstImage(512, 512, CV_8UC3, Scalar::all(255));
    Size size = dstImage.size();
    namedWindow("dstImage");

    double minVal, maxVal;
    minMaxLoc(measurement_k, &minVal, &maxVal);
    double scale = size.height / (maxVal - minVal);
    cout << "measurement_k : ";
    cout << "minVal = " << minVal;
    cout << ", maxVal = " << maxVal << endl;

    //drawing the truth value, x = -0.37727
    Point pt1, pt2;
    pt1.x = 0;
    pt1.y = size.height - cvRound(scale * x - scale * minVal);

    pt2.x = size.width;
    pt2.y = size.height - cvRound(scale * x - scale * minVal);
    line(dstImage, pt1, pt2, Scalar(255, 0, 0), 2); // thickness=2, color: Scalar(255, 0, 0) = red
    //drawing the noisy measurements, measurement_k
    int step = size.width / count;

    for (t = 0; t < count; t++)
    {
        pt1.x = t * step;
        pt1.y = size.height - cvRound(scale * measurement_k[t] - scale * minVal);
        circle(dstImage, pt1, 3, Scalar(0, 255, 0), 2); // radius = 3, thickness = 2
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

    // drawing the error covariance, postP
    Mat PImage(size.height, size.width, CV_8UC3, Scalar::all(255));
    size = PImage.size();
    namedWindow("PImage");

    minMaxLoc(postP, &minVal, &maxVal);
    scale = size.height / (maxVal - minVal);
    cout << "error covariance, P: ";
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