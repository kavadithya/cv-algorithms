#include "dehaze.h"
#include "softmatting.h"
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const int patchSize = 15;
const double aerialPerspectiveFactor = 0.95;
const double minTransmission = 0.1;

static void calc_dark_channel(const Mat &input, Mat &darkChannel) {
	darkChannel.create(input.rows, input.cols, CV_64FC1);
	for (int i = 0; i < input.rows; ++i)
		for (int j = 0; j < input.cols; ++j) {
			double min_d = numeric_limits<double>::max();
			for (int x = i - patchSize / 2; x <= i + patchSize / 2; ++x)
				for (int y = j - patchSize / 2; y <= j + patchSize / 2; ++y)
					if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
						Vec3d color = input.at<Vec3b>(x, y);
						for (int k = 0; k < 3; ++k)
							min_d = min(min_d, color[k]);
					}
			darkChannel.at<double>(i, j) = min_d;
		}
}

static Vec3d estimate_atmospheric_light(const Mat &input, const Mat &darkChannel) {
	Vec3d ret;
	vector<pair<double, Vec3d>> pxls;
	for (int i = 0; i < input.rows; ++i)
		for (int j = 0; j < input.cols; ++j)
			pxls.push_back(make_pair(darkChannel.at<double>(i, j), (Vec3d)input.at<Vec3b>(i, j)));
	sort(pxls.begin(), pxls.end(), [](const pair<double, Vec3d> &a, const pair<double, Vec3d> &b) {
		return a.first > b.first;
	});

	int pxlCount = max(1.0, input.rows * input.cols * 0.001);
	double max_i = -numeric_limits<double>::max();
	for (int i = 0; i < pxlCount; ++i) {
		double w = sqrt(pxls[i].second.dot(pxls[i].second));
		if (w > max_i) {
			max_i = w;
			ret = pxls[i].second;
		}
	}
	return ret;
}

void estimate_transmission(const Mat &input, const Vec3d &light, Mat &transmission) {
	transmission.create(input.rows, input.cols, CV_64FC1);
	for (int i = 0; i < input.rows; ++i)
		for (int j = 0; j < input.cols; ++j) {
			double min_t = numeric_limits<double>::max();
			for (int x = i - patchSize / 2; x <= i + patchSize / 2; ++x)
				for (int y = j - patchSize / 2; y <= j + patchSize / 2; ++y)
					if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
						Vec3d color = input.at<Vec3b>(x, y);
						for (int k = 0; k < 3; ++k)
							min_t = min(min_t, aerialPerspectiveFactor * color[k] / light[k]);
					}
			transmission.at<double>(i, j) = 1.0 - min_t;
		}
}

void recover_image(const Mat &input, Mat &transmission, const Vec3d &light, Mat &output) {
	output = input.clone();
	for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            transmission.at<double>(i, j) = max(transmission.at<double>(i, j), minTransmission);
    for (int i = 0; i < input.rows; ++i)
		for (int j = 0; j < input.cols; ++j) {
			Vec3d color = input.at<Vec3b>(i, j);
			output.at<Vec3b>(i, j) = (color - light) / transmission.at<double>(i, j) + light;
		}
}

void dehaze(const Mat &input, Mat &semi_output, Mat &output) {
	Mat darkChannel;
	calc_dark_channel(input, darkChannel);

	Vec3d light = estimate_atmospheric_light(input, darkChannel);
	
	Mat transmission;
	estimate_transmission(input, light, transmission);
	
    // semi_output (no soft matting)
    recover_image(input, transmission, light, semi_output);

    Mat refinedTransmission;
    soft_matting(input, transmission, refinedTransmission);
    // GaussianBlur(transmission, transmission, Size(41, 41), 20);

	recover_image(input, refinedTransmission, light, output);
}
