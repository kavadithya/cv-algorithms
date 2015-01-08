#include "grabcut.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "include/opencv/gcgraph.hpp"
#include "include/gmm/gmm.h"

using namespace std;
using namespace cv;

void init_mask(Rect &rect, Size inputSize, Mat &mask) {
    mask.create(inputSize, CV_8UC1);
    mask.setTo(GC_BGD);

    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(inputSize.width - rect.x, rect.width);
    rect.height = min(inputSize.height - rect.y, rect.height);

    (mask(rect)).setTo(Scalar(GC_PR_FGD));
}

void init_GMM(const Mat &input, const Mat &mask, GMM &bgdGMM, GMM &fgdGMM) {
    // use K-means to initilize the GMMs
    const int iterCount = 10;
    vector<Vec3d> bgdSample, fgdSample;
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            if (mask.at<uchar>(i, j) == GC_BGD)
                bgdSample.push_back((Vec3d)input.at<Vec3b>(i, j));
            else
                fgdSample.push_back((Vec3d)input.at<Vec3b>(i, j));
    // for bgdGMM
    vector<int> bgdLabel(bgdSample.size());
    kmeans(Mat(bgdSample.size(), 3, CV_32FC1, &bgdSample[0][0]), GMM::K, bgdLabel, TermCriteria(CV_TERMCRIT_ITER, iterCount, 0.0), 0, KMEANS_PP_CENTERS);
    bgdGMM.init_learning();
    for (int i = 0; i < bgdSample.size(); ++i)
        bgdGMM.add_sample(bgdLabel[i], bgdSample[i]);
    bgdGMM.end_learning();

    // for fgdGMM
    vector<int> fgdLabel(fgdSample.size());
    kmeans(Mat(fgdSample.size(), 3, CV_32FC1, &fgdSample[0][0]), GMM::K, fgdLabel, TermCriteria(CV_TERMCRIT_ITER, iterCount, 0.0), 0, KMEANS_PP_CENTERS);
    fgdGMM.init_learning();
    for (int i = 0; i < fgdSample.size(); ++i)
        fgdGMM.add_sample(fgdLabel[i], fgdSample[i]);
    fgdGMM.end_learning();
}

double calc_beta(const Mat &input) {
    double ret = 0;
    int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            for (int k = 0; k < 4; ++k) {
                int x = i + dx[k], y = j + dy[k];
                if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
                    Vec3d diff = (Vec3d)input.at<Vec3b>(i, j) - (Vec3d)input.at<Vec3b>(x, y);
                    ret += diff.dot(diff);
                }
            }
    if (ret <= numeric_limits<double>::epsilon())  
        ret = 0;
    else 
        ret = 1.0 / (2 * ret / (4 * input.rows * input.cols - 3 * input.rows - 3 * input.cols + 2));
    return ret;
}

void calc_weight(const Mat &input, Mat &left, Mat &upleft, Mat &up, Mat &upright, double beta, double gamma) {
    left.create(input.rows, input.cols, CV_64FC1);
    upleft.create(input.rows, input.cols, CV_64FC1);
    up.create(input.rows, input.cols, CV_64FC1);
    upright.create(input.rows, input.cols, CV_64FC1);
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j) {
            if (j > 0) {
                Vec3d diff = (Vec3d)input.at<Vec3b>(i, j) - (Vec3d)input.at<Vec3b>(i, j - 1);
                left.at<double>(i, j) = gamma * exp(-beta * diff.dot(diff));
            } else {
                left.at<double>(i, j) = 0;
            }
            if (i > 0 && j > 0) {
                Vec3d diff = (Vec3d)input.at<Vec3b>(i, j) - (Vec3d)input.at<Vec3b>(i - 1, j - 1);
                upleft.at<double>(i, j) = gamma / sqrt(2.0) * exp(-beta * diff.dot(diff));
            } else {
                upleft.at<double>(i, j) = 0;
            }
            if (i > 0) {
                Vec3d diff = (Vec3d)input.at<Vec3b>(i, j) - (Vec3d)input.at<Vec3b>(i - 1, j);
                up.at<double>(i, j) = gamma * exp(-beta * diff.dot(diff));
            } else {
                up.at<double>(i, j) = 0;
            }
            if (j + 1 < input.cols && i > 0) {
                Vec3d diff = (Vec3d)input.at<Vec3b>(i, j) - (Vec3d)input.at<Vec3b>(i - 1, j + 1);
                upright.at<double>(i, j) = gamma / sqrt(2.0) * exp(-beta * diff.dot(diff));
            } else {
                upright.at<double>(i, j) = 0;
            }
        }
} 

void assign_and_learn(const Mat &input, const Mat &mask, GMM& bgdGMM, GMM& fgdGMM) {
    // assign components and learn
    bgdGMM.init_learning();
    fgdGMM.init_learning();
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            if (mask.at<uchar>(i, j) == GC_BGD || mask.at<uchar>(i, j) == GC_PR_BGD)
                bgdGMM.add_sample(bgdGMM.which((Vec3d)input.at<Vec3b>(i, j)), (Vec3d)input.at<Vec3b>(i, j));
            else
                fgdGMM.add_sample(fgdGMM.which((Vec3d)input.at<Vec3b>(i, j)), (Vec3d)input.at<Vec3b>(i, j));
    bgdGMM.end_learning();
    fgdGMM.end_learning();
}

void construct_graph(const Mat &input, Mat &mask, const GMM &bgdGMM, const GMM &fgdGMM, double lambda, const Mat &left, const Mat &upleft, const Mat &up, const Mat &upright, GCGraph<double> &graph) {
    int vCount = input.rows * input.cols;
    int eCount = 2 * (4 * vCount - 3 * (input.cols + input.rows) + 2);
    graph.create(vCount, eCount);
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j) {
            int index = graph.addVtx();
            // calculate the first term of Gibbs Energy
            double fromSource, toSink;
            if (mask.at<uchar>(i, j) == GC_PR_BGD || mask.at<uchar>(i, j) == GC_PR_FGD) {
                fromSource = -log(bgdGMM.calc_prob((Vec3d)input.at<Vec3b>(i, j)));
                toSink = -log(fgdGMM.calc_prob((Vec3d)input.at<Vec3b>(i, j)));
            } else {
                fromSource = 0;
                toSink = lambda;
            }
            graph.addTermWeights(index, fromSource, toSink);
            
            // calculate the second term of Gibbs Energy
            if (j > 0)
                graph.addEdges(index, index - 1, left.at<double>(i, j), left.at<double>(i, j));
            if (i > 0 && j > 0)
                graph.addEdges(index, index - input.cols - 1, upleft.at<double>(i, j), upleft.at<double>(i, j));
            if (i > 0)
                graph.addEdges(index, index - input.cols, up.at<double>(i, j), up.at<double>(i, j));
            if (j + 1 < input.rows && i > 0)
                graph.addEdges(index, index - input.cols + 1, upright.at<double>(i, j), upright.at<double>(i, j));
        }
}

void estimate(const Mat &input, Mat &mask, const GMM &bgdGMM, const GMM &fgdGMM, double lambda, const Mat &left, const Mat &upleft, const Mat &up, const Mat &upright) {
    GCGraph<double> graph;
    construct_graph(input, mask, bgdGMM, fgdGMM, lambda, left, upleft, up, upright, graph);
    graph.maxFlow();
    for (int i = 0; i < mask.rows; ++i)
        for (int j = 0; j < mask.cols; ++j)
            if (mask.at<uchar>(i, j) == GC_PR_BGD || mask.at<uchar>(i, j) == GC_PR_FGD) {
                if (graph.inSourceSegment(i * mask.cols + j))
                    mask.at<uchar>(i, j) = GC_PR_FGD;
                else
                    mask.at<uchar>(i, j) = GC_PR_BGD;
            }
}

void grabCut(const Mat &input, Rect &rect, Mat &output, int iterCount) {
    // init mask with rect
    Mat &mask = output;
    init_mask(rect, input.size(), mask);
    
    // init GMM
    GMM bgdGMM, fgdGMM;
    init_GMM(input, mask, bgdGMM, fgdGMM);

    // set parameters
    const double gamma = 50;
    const double lambda = 9 * gamma;
    const double beta = calc_beta(input);

    // calc weights
    Mat left, upleft, up, upright;
    calc_weight(input, left, upleft, up, upright, beta, gamma);

    // iterative computation
    for (int i = 0; i < iterCount; ++i) {
        // step 1 and step 2
        assign_and_learn(input, mask, bgdGMM, fgdGMM);
        // step 3
        estimate(input, mask, bgdGMM, fgdGMM, lambda, left, upleft, up, upright);
    }
}
