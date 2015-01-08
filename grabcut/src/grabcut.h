#ifndef GRABCUT_H
#define GRABCUT_H

#include "gmm.h"

void grabCut(const cv::Mat &input, cv::Rect &rect, cv::Mat &output, int iterCount = 10); 
#endif
