#include <opencv2/imgproc/imgproc.hpp>
#include "include/Eigen/Sparse"
#include "include/Eigen/LU"
#include <unordered_map>

using namespace std;
using namespace cv;
using namespace Eigen;

const double lambda = 0.0001;
const int windowSize = 3;
const double epsilon = 0.0001;

struct Window {
    int size;
    Vector3f mean;
    Matrix3f cov;
    Window() : size(0) {
        mean << 0, 0, 0;
        cov << 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }
};

static Vector3f Vec3b2Vector3f(const Vec3b &color) {
    double a = color[0], b = color[1], c = color[2];
    return Vector3f(a, b, c);
}

void soft_matting(const Mat &input, const Mat &transmission, Mat &refinedTransmission) {
    refinedTransmission = transmission.clone();
    int totalPxl = input.rows * input.cols;

    // calulate all the windows
    printf("calulate all the windows ...\n");
    vector<Window> windows;
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; i < input.cols; ++i) {
            Window window;
            for (int x = i - windowSize / 2; x <= i + windowSize / 2; ++x)
                for (int y = j - windowSize / 2; y <= j + windowSize / 2; ++y)
                    if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
                        ++window.size;
                        window.mean += Vec3b2Vector3f(input.at<Vec3b>(x, y));
                    }
            window.mean /= window.size;
            for (int x = i - windowSize / 2; x <= i + windowSize / 2; ++x)
                for (int y = j - windowSize / 2; y <= j + windowSize / 2; ++y)
                    if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
                        Vector3f diff = Vec3b2Vector3f(input.at<Vec3b>(x, y)) - window.mean;
                        window.cov += diff * diff.transpose();
                    }
            window.cov /= window.size;
            windows.push_back(window);       
        }
    
    // calculate Matting Laplacian matrix L
    printf("calculate Matting Laplacian matrix L ...\n");
    Matrix3f U3;
    U3 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    vector<Triplet<double>> listL;
    for (int i = 0; i < totalPxl; ++i) {
        int row = i / input.rows, col = i % input.rows;
        Vector3f Ii = Vec3b2Vector3f(input.at<Vec3b>(row, col));
        unordered_map<int, double> ele;
        for (int x = row - windowSize / 2; x <= row + windowSize / 2; ++x)
            for (int y = col - windowSize / 2; y <= col + windowSize / 2; ++y)
                if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
                    for (int jx = x - windowSize / 2; jx <= x + windowSize / 2; ++jx)
                        for (int jy = y - windowSize / 2; jy <= y + windowSize / 2; ++jy)
                            if (jx >= 0 && jx < input.rows && jy >= 0 && jy < input.cols) {
                                int j = jx * input.rows + jy;
                                Vector3f Ij = Vec3b2Vector3f(input.at<Vec3b>(jx, jy));
                                Vector3f left = Ii - windows[x * input.rows + y].mean;
                                Matrix3f middle = windows[x * input.rows + y].cov + epsilon / windows[x * input.rows + y].size * U3;
                                Vector3f right = Ij - windows[x * input.rows + y].mean;
                                ele[j] += (i == j) - 1.0 / windows[x * input.rows + y].size * (1.0 + left.transpose() * middle.inverse() * right);
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
    for (int i = 0; i < totalPxl; ++i)
        tt[i] = transmission.at<double>(i / input.rows, i % input.rows);

    // solve the sparse linear system (L + lambda * U) * t = lambda * t'
    printf("solve the sparse linear system (L + lambda * U) * t = lambda * t' ...\n");  
    L += lambda * U;
    BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>> solver;
    solver.compute(L);
    if (solver.info() != Success) {
        printf("decomposition failed\n");
        return;
    }
    tt *= lambda;
    VectorXd t = solver.solve(tt);
    //printf("%f\n", t[0]);
    if (solver.info() != Success) {
        printf("solving failed\n");
        return;
    }

    // set the answer back to refinedTransmission
    printf("set the answer back to refinedTransmission ...\n");
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            refinedTransmission.at<double>(i, j) = t[i * input.rows + j];
}