#ifndef DEHAZE_H
#define DEHAZE_H

#include <opencv2/core/core.hpp>

void dehaze(const cv::Mat &input, cv::Mat &semi_output, cv::Mat &output);

#endif
