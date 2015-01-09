#include "dehaze.h"
#include <algorithm>

using namespace std;
using namespace cv;

const int patchSize = 15;
const double aerialPerspectiveFactor = 0.95;

static void calc_dark_channel(const Mat &input, Mat &J) {
	J.create(input.rows, input.cols, CV_64FC1);
	for (int i = 0; i < input.rows; ++i)
		for (int j = 0; j < input.cols; ++j) {
			J.at<double>(i, j) = numeric_limits<double>::max();
			for (int x = i - patchSize / 2; x <= i + patchSize / 2; ++x)
				for (int y = j - patchSize / 2; y <= j + patchSize / 2; ++y)
					if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
						Vec3d color = input.at<Vec3b>(x, y);
						for (int k = 0; k < 3; ++k)
							J.at<double>(i, j) = min(J.at<double>(i, j), color[k]);
					}
		}
}

static Vec3d estimate_atmospheric_light(const Mat &input, const Mat &J) {
	Vec3d ret;
	vector<pair<double, Vec3d>> pxls;
	for (int i = 0; i < input.rows; ++i)
		for (int j = 0; j < input.cols; ++j)
			pxls.push_back(make_pair(J.at<double>(i, j), (Vec3d)input.at<Vec3b>(i, j)));
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

void dehaze(const Mat &input, Mat &output) {
	Mat J;
	calc_dark_channel(input, J);

}
