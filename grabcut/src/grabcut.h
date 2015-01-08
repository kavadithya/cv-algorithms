#ifndef GRABCUT_H
#define GRABCUT_H

#include <opencv2/core/core.hpp>

void grabCut(const cv::Mat &input, cv::Rect &rect, cv::Mat &output, int iterCount = 10); 
#endif
