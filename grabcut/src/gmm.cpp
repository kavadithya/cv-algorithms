#include "gmm.h"
#include <cstring>

using namespace cv;

GMM::GMM() {
    mean.resize(3 * K); // 3 (mean)
    cov.resize(9 * K); // 3 * 3 (covariance)
    weight.resize(K); // 1 (component weight)
    det.resize(K); // determinant
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
        det[index] = c[0] * (c[4]*c[8] - c[5]*c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        // calculate the inverse of covariance
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


