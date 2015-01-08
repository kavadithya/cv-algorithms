#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cstdlib>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    Mat mat = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    mat.convertTo(mat, CV_32FC3, 1.0 / 255);
    
    waitKey(0);    
}
