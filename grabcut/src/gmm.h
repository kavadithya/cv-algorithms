#ifndef GMM_H
#define GMM_H

#include <opencv2/core/core.hpp>
#include <vector>

/*
 GMM - Gaussian Mixture Model 
 */

class GMM {
public:
    const static int K = 5; // the number of components
    
    GMM();
    double calc_prob(const cv::Vec3d color) const;
    double calc_prob(int index, const cv::Vec3d color) const;
    int which(const cv::Vec3d color) const;

private:
    void calc_det_and_inv(int index);
    std::vector<double> mean, cov, weight;
    std::vector<double> det, inv_cov;
};

#endif
