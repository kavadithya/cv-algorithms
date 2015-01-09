#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "dehaze.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    Mat input = imread(argv[1]);
    Mat output;
    dehaze(input, output);
    imshow("output", output);
    waitKey(0);    
}
