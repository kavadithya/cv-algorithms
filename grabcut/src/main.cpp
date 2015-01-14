#include "grabcut.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

enum RectState {UN_SET, IN_PROCESS, SET};

Rect rect;
RectState rectState = UN_SET;
Mat img;
const Scalar GREEN = Scalar(0, 255, 0);

void show_image(const Mat &img) {
    Mat image = img.clone();
    if(rectState == IN_PROCESS || rectState == SET)
        rectangle(image, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
    imshow("image", image);
}

static void on_mouse(int event, int x, int y, int flags, void* param) {
    switch (event) {
        case EVENT_LBUTTONDOWN:
            if(rectState == UN_SET) {
                rectState = IN_PROCESS;
                rect = Rect(x, y, 1, 1 );
            }
            break;
        case EVENT_LBUTTONUP:
            if(rectState == IN_PROCESS) {
                rect = Rect(Point(rect.x, rect.y), Point(x,y));
                rectState = SET;
                show_image(img);
            }
            break;
        case EVENT_MOUSEMOVE:
            if(rectState == IN_PROCESS) {
                rect = Rect(Point(rect.x, rect.y), Point(x,y));
                show_image(img);
            }
            break;
    }
}

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]);
    int iterCount = atoi(argv[2]);
    img = input.clone();
    namedWindow("image", WINDOW_AUTOSIZE);
    setMouseCallback("image", on_mouse, 0);
    show_image(img);
    //Rect rect(450, 350, input.cols - 450, input.rows);
    waitKey(0);
    while (rectState != SET) ;
    Mat output;
    double t1 = getTickCount();
    grabCut(input, rect, output, iterCount);
    double t2 = getTickCount();
    printf("My grabCut : %g ms\n", (t2 - t1) / getTickFrequency() * 1000);
    if (argc > 3) {
        double t1 = getTickCount();
        Mat output2, model1, model2;
        cv::grabCut(input, output2, rect, model1, model2, iterCount, GC_INIT_WITH_RECT);
        double t2 = getTickCount();
        printf("OpenCV grabCut : %g ms\n", (t2 - t1) / getTickFrequency() * 1000);
    }
    compare(output, GC_PR_FGD, output, CMP_EQ);
    Mat fgd(input.size(), CV_8UC3, Scalar(255,255,255));
    input.copyTo(fgd, output);
    show_image(fgd);
    imwrite("output.png", fgd);
    waitKey(0); 
    return 0;
}
