#include "gmm.h"
#include <cstring>

using namespace std;
using namespace cv;

GMM::GMM() {
    mean.resize(3 * K); // 3 (mean)
    cov.resize(9 * K); // 3 * 3 (covariance)
    weight.resize(K); // 1 (component weight)
    det.resize(K); // the determinant
    inv_cov.resize(9 * K); // the inverse of covariance
}

double GMM::calc_prob (const Vec3d color) const {
    double ret = 0;
    for (int i = 0; i < K; ++i) 
        ret += weight[i] * calc_prob(i, color);
    return ret;
}

double GMM::calc_prob(int index, const Vec3d color) const {
    double ret = 0;
    if (weight[index] > 0) {
        CV_Assert(det[index] > numeric_limits<double>::epsilon());        
        // diff = color - mean
        Vec3d diff = color;
        for (int i = 0; i < 3; ++i)
            diff[i] -= mean[3 * index + i];
        // diff(T) * inv_cov * diff
        double mult = 0;
        for (int i = 0; i < 3; ++i)
            mult += diff[i] * (diff[0] * inv_cov[9 * index + i] + diff[1] * inv_cov[9 * index + 3 + i] + diff[2] * inv_cov[9 * index + 6 + i]);
        ret = 1.0 / sqrt(det[index]) * exp(-0.5 * mult);
    }
    return ret;
}

int GMM::which(const Vec3d color) const {
    int ret = 0;
    double max_p = 0;
    for (int i = 0; i < K; ++i) {
        double p = calc_prob(i, color);
        if (p > max_p) {
            max_p = p;
            ret = i;
        }
    }
    return ret;
}

void GMM::calc_det_and_inv(int index) {
    if (weight[index] > 0) {
        // calculate the determinant
        double *c = &cov[9 * index];        
        det[index] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        // calculate the inverse of covariance
        CV_Assert(det[index] > numeric_limits<double>::epsilon());
        double *ic = &inv_cov[9 * index];
        ic[0] = (c[4] * c[8] - c[5] * c[7]) / det[index];
        ic[1] = -(c[1] * c[8] - c[2] * c[7]) / det[index];
        ic[2] = (c[1] * c[5] - c[2] * c[4]) / det[index];
        ic[3] = -(c[3] * c[8] - c[5] * c[6]) / det[index];
        ic[4] = (c[0] * c[8] - c[2] * c[6]) / det[index];
        ic[5] = -(c[0] * c[5] - c[2] * c[3]) / det[index];
        ic[6] = (c[3] * c[7] - c[4] * c[6]) / det[index];
        ic[7] = -(c[0] * c[7] - c[1] * c[6]) / det[index];
        ic[8] = (c[0] * c[4] - c[1] * c[3]) / det[index];
    }
}

void GMM::init_learning() {
    sum.clear();
    sum.resize(3 * K, 0); // the sum of RGB
    product.clear();
    product.resize(9 * K, 0); // the sum of product of RGB
    sample.clear();
    sample.resize(K, 0); // the number of samples for each component
    total_sample = 0; // the number of all the samples
}

void GMM::add_sample(int index, const Vec3d color) {
    for (int i = 0; i < 3; ++i)
        sum[3 * index + i] += color[i];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            product[9 * index + 3 * i + j] += color[i] * color[j];
    ++sample[index];
    ++total_sample;
}

void GMM::end_learning() {
    for (int i = 0; i < K; ++i) {
        if (sample[i] == 0) {
            weight[i] = 0;   
        } else {
            weight[i] = double(sample[i]) / total_sample;
            for (int j = 0; j < 3; ++j)
                mean[3 * i + j] = sum[3 * i + j] / sample[i];
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    cov[9 * i + 3 * j + k] = product[9 * i + 3 * j + k] / sample[i] - mean[3 * i + j] * mean[3 * i + k];
            double *c = &cov[9 * i];
            double d = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
            if (d <= numeric_limits<double>::epsilon()) {
                c[0] += 0.01;
                c[4] += 0.01;
                c[8] += 0.01;
            }
            calc_det_and_inv(i);
        }
    }
}
