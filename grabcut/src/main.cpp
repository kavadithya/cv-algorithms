#include "grabcut.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]);
    int iterCount = atoi(argv[2]); 
    Rect rect(450, 350, input.cols - 450, input.rows);
    Mat output;
    grabCut(input, rect, output, iterCount);
    compare(output, GC_PR_FGD, output, CMP_EQ);
    Mat fgd(input.size(), CV_8UC3, Scalar(255,255,255));
    input.copyTo(fgd, output);
    imshow("input", input);
    imshow("output", fgd);
    waitKey(0); 
    return 0;
}
