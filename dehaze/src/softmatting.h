#ifndef SOFTMATTING_H
#define SOFTMATTING_H

#include <opencv2/core/core.hpp>

void soft_matting(const cv::Mat &input, const cv::Mat &transmission, cv::Mat &refinedTransmission);

#endif
