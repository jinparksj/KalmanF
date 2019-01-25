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

    randn(measurement, Scalar::all(x), stddevR);
    measurement_k[0] = measurement.at<float>(0);

    //initial guesses
    setIdentity(KF.statePost, Scalar::all(0)); //x_hat(0) = 0
    setIdentity(KF.errorCovPost, Scalar::all(1)); //P(0) = 1
    state_k[0] = KF.statePost.at<float>(0);
    postP[0] = KF.errorCovPost.at<float>(0);


    //drawing values
    namedWindow("dstImage");
    Mat dstImage(512, 512, CV_8UC3, Scalar::all(255));
    Size size = dstImage.size();
    int step = size.width / count;
    double minVal = x - stddevR.val[0] * 3;
    double maxVal = x + stddevR.val[0] * 3;
    double scale = size.height / (maxVal - minVal);

    for ( ; ; )
    {
        Mat prediction = KF.predict(); //predict

        //generate a measurement
        randn(measurement, Scalar::all(x), stddevR);
        measurement_k[t] = measurement.at<float>(0);

        Mat estimate = KF.correct(measurement); //update
        state_k[t] = estimate.at<float>(0);
        postP[t] = KF.errorCovPost.at<float>(0);

        //drawing the truth value, x = -0.37727
        Point pt1, pt2;
        pt1.x = 0;
        pt1.y = size.height - cvRound(scale * x - scale * minVal);
        pt2.x = size.width;
        pt2.y = size.height - cvRound(scale * x - scale * minVal);
        line(dstImage, pt1, pt2, Scalar(255, 0, 0), 2);

        for (int k = count - 1; k > 0 ; k--)
        {
            int k1 = (t + k) % count;
            int k2 = (t + k + 1) % count;

            //drawing the noisy measurements, measurement_k
            pt1.x = k * step;
            pt1.y = size.height - cvRound(scale * measurement_k[k1] - scale * minVal);

            pt2.x = (k + 1) * step;
            pt2.y = size.height - cvRound(scale * measurement_k[k2] - scale * minVal);

            line(dstImage, pt1, pt2, Scalar(0, 255, 0), 2);

            //drawing the filter estimate, state_k
            pt1.x = k * step;
            pt1.y = size.height - cvRound(scale * state_k[k1] - scale * minVal);

            pt2.x = (k + 1) * step;
            pt2.y = size.height - cvRound(scale * state_k[k2] - scale * minVal);
            line(dstImage, pt1, pt2, Scalar(0, 0, 255), 2);

        }

        imshow("dstImage", dstImage);
        int ckey = waitKey(30);
        if (ckey == 27) break; //ESC key
        t = (t + 1) % count;
        dstImage = Scalar::all(255);
    }

    return 0;






}