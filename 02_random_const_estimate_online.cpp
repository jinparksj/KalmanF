//
// Created by jin on 1/21/19.
//

#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;
using namespace cv;

int main(){
    //Greg Welch and Gary Bishop, "An Introduction to the Kalman Filter", 2006
    //Estimating a Random Constant: online

    theRNG().state = time(NULL);
    int count = 500;
    double x = -0.37727; // truth value

    ////////// Kalman filter //////////

    double Q = 1e-5;
    double R = 1.0; //or 0.0001

    Scalar stddevR = Scalar::all(sqrt(R));

    vector<float> measurement_k(count); // the noisy measurements, 500 measurement_ks with vector<float> type
    vector<float> state_k(count); // state, x_k
    float predict_k; // predict, x'_k
    float postP; //the posteriori error covariance, P
    float preP; // the priori error covariance, P' or M
    float K; // Kalman gain, K

    // drawing values
    namedWindow("dstImage");
    Mat dstImage(512, 512, CV_8UC3, Scalar::all(255));
    Size size = dstImage.size();// size.width = 512, size.height = 512, cv::Size
    int step = size.width / count;

    double minVal = x - stddevR.val[0] * 3;
    double maxVal = x + stddevR.val[0] * 3;
    double scale = size.height / (maxVal - minVal);

    //initial Guesses

    state_k[0] = 0.0;
    postP = 1.0;
    //theRNG().state = time(NULL);
    int t = 1;
    for (;;)
    {
        int t1 = (t - 1 + count) % count; // t-1

        //predict
        predict_k = state_k[t1];
        preP = postP + Q;

        //generate a measurement z_k
        //measurement_k[t] = rng.gaussian(stddevR) + x;
        Mat measurement(1, 1, CV_32F);
        randn(measurement, Scalar::all(x), stddevR); // measurement ~ N(Scalar::all(x), stddevR), Gaussian normal distribution
        measurement_k[t] = measurement.at<float>(0); // save to draw

        //update
        K = preP / (preP + R);
        state_k[t] = predict_k + K * (measurement_k[t] - predict_k);
        postP = (1 - K) * preP;

        //drawing the truth value, x = -0.37727
        Point pt1, pt2;
        pt1.x = 0;
        pt1.y = size.height - cvRound(scale*x - scale*minVal);
        pt2.x = size.width;
        pt2.y = size.height - cvRound(scale*x - scale*minVal);

        line(dstImage, pt1, pt2, Scalar(255, 0, 0), 2);

        for (int k = count - 1; k>0 ; k--)
        {
            int k1 = (t + k) % count;
            int k2 = (t + k + 1) % count;

            //drawing the noisy measurements, observations, measurement_k
            pt1.x = k * step;
            pt1.y = size.height - cvRound(scale * measurement_k[k1] - scale * minVal);

            pt2.x = (k+1) * step;
            pt2.y = size.height - cvRound(scale * measurement_k[k2] - scale * minVal);

            line(dstImage, pt1, pt2, Scalar(0, 255, 0), 2);

            //drawing the filter estimate, state_k
            pt1.x = k*step;
            pt1.y = size.height - cvRound(scale * state_k[k1] - scale * minVal);

            pt2.x = (k+1) * step;
            pt2.y = size.height - cvRound(scale * state_k[k2] - scale * minVal);

            line(dstImage, pt1, pt2, Scalar(0, 0, 255), 2);
        }
        imshow("dstImgage", dstImage);
        int ckey = waitKey(30);
        if (ckey == 27) break;
        t = (t + 1) % count;
        dstImage = Scalar::all(255);
    }
    return 0;








}