#include <opencv2/imgproc/imgproc.hpp>
#include "include/Eigen/Sparse"
#include "include/Eigen/LU"
#include <unordered_map>
#include <cassert>
#include <iostream>

using namespace std;
using namespace cv;
using namespace Eigen;

const double lambda = 0.0001;
const int windowSize = 3;
const double epsilon = 0.0001;

struct Window {
    int size;
    Vector3d mean;
    Matrix3d cov;
    Window() : size(0) {
        mean << 0, 0, 0;
        cov << 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }
};

static Vector3d Vec3b2Vector3d(Vec3b color) {
    double a = color[0], b = color[1], c = color[2];
    return Vector3d(a, b, c);
}

void soft_matting(const Mat &input, const Mat &transmission, Mat &refinedTransmission) {
    refinedTransmission = transmission.clone();
    int totalPxl = input.rows * input.cols;

    // calulate all the windows
    printf("calulate all the windows ...\n");
    vector<Window> windows;
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j) {
            Window window;
            for (int x = i - windowSize / 2; x <= i + windowSize / 2; ++x)
                for (int y = j - windowSize / 2; y <= j + windowSize / 2; ++y)
                    if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
                        ++window.size;
                        window.mean += Vec3b2Vector3d(input.at<Vec3b>(x, y));
                    }
            window.mean /= window.size;
            for (int x = i - windowSize / 2; x <= i + windowSize / 2; ++x)
                for (int y = j - windowSize / 2; y <= j + windowSize / 2; ++y)
                    if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
                        Vector3d diff = Vec3b2Vector3d(input.at<Vec3b>(x, y)) - window.mean;
                        window.cov += diff * diff.transpose();
                    }
            window.cov /= window.size - 1;
            windows.push_back(window);       
        }
    
    // calculate Matting Laplacian matrix L
    printf("calculate Matting Laplacian matrix L ...\n");
    Matrix3d U3;
    U3 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    vector<Triplet<double>> listL;
    for (int i = 0; i < totalPxl; ++i) {
        int row = i / input.cols, col = i % input.cols;
        Vector3d Ii = Vec3b2Vector3d(input.at<Vec3b>(row, col));
        unordered_map<int, double> ele;
        for (int x = row - windowSize / 2; x <= row + windowSize / 2; ++x)
            for (int y = col - windowSize / 2; y <= col + windowSize / 2; ++y)
                if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
                    for (int jx = x - windowSize / 2; jx <= x + windowSize / 2; ++jx)
                        for (int jy = y - windowSize / 2; jy <= y + windowSize / 2; ++jy)
                            if (jx >= 0 && jx < input.rows && jy >= 0 && jy < input.cols) {
                                int j = jx * input.cols + jy;
                                Vector3d Ij = Vec3b2Vector3d(input.at<Vec3b>(jx, jy));
                                Vector3d left = Ii - windows[x * input.cols + y].mean;
                                Matrix3d middle = windows[x * input.cols + y].cov + epsilon / windows[x * input.cols + y].size * U3;
                                Vector3d right = Ij - windows[x * input.cols + y].mean;
                                ele[j] += (i == j) - 1.0 / windows[x * input.cols + y].size * (1.0 + left.transpose() * middle.inverse() * right);
                            }
                }
        for (auto & p : ele)
            listL.push_back(Triplet<double>(i, p.first, p.second));
    }
    SparseMatrix<double> L(totalPxl, totalPxl);
    L.setFromTriplets(listL.begin(), listL.end());

    // set the identity matrix U
    printf("set the identity matrix U ...\n");
    vector<Triplet<double>> listU;
    for (int i = 0; i < totalPxl; ++i)
        listU.push_back(Triplet<double>(i, i, 1.0));
    SparseMatrix<double> U(totalPxl, totalPxl);
    U.setFromTriplets(listU.begin(), listU.end());

    // set the vector t'
    printf("set the vector t' ...\n");
    VectorXd tt(VectorXd::Zero(totalPxl));
    for (int i = 0; i < totalPxl; ++i) {
        tt[i] = transmission.at<double>(i / input.cols, i % input.cols);
    }

    // solve the sparse linear system (L + lambda * U) * t = lambda * t'
    printf("solve the sparse linear system (L + lambda * U) * t = lambda * t' ...\n");  
    L += lambda * U;
    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(L);
    if (solver.info() != Success) {
        printf("decomposition failed\n");
        return;
    }
    tt *= lambda;
    VectorXd t = solver.solve(tt);
    if (solver.info() != Success) {
        printf("solving failed\n");
        return;
    }

    // set the answer back to refinedTransmission
    printf("set the answer back to refinedTransmission ...\n");
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j) {
            if (isnan(t[i * input.cols + j]))
                continue;
            refinedTransmission.at<double>(i, j) = t[i * input.cols + j];
        }
}
