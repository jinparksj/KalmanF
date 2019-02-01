//
// Created by jin on 1/31/19.
//

#include "onMouse.h"
#include <opencv2/opencv.hpp>

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
