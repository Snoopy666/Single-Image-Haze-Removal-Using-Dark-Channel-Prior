#include "dehaze.h"
#include <algorithm>


bool operator> (const struct Pixel & a, const struct Pixel & b) {
    return a.intensity > b.intensity;
}


void DeHaze::loadImage(const char * path) {
    src.release();
    src = cv::imread(path);
    assert(src.data);
    cout << "Imgage successfully loaded from " << path << endl;
    cout << "image size : " << src.rows << " " << src.cols << " " << src.depth() << endl;
    //cout << src.type() << endl;
    //cout << src.channels() << endl;
    tran.release();
    dark.release();
    dst.release();
    //dst.create(src.rows, src.cols, src.type());
    src.copyTo(dst);
    //cv::imshow("dst", dst);
    //cv::waitKey(0);
    tran.create(src.rows, src.cols, CV_32FC1);
    dark.create(src.rows, src.cols, CV_8UC1);
    gtran.create(src.rows, src.cols, CV_32FC1);
    //cout << src << endl;
    cout<<"load done"<<endl;

}


///Single Image Haze Removal Using Dark Channel Prior
void DeHaze::getDarkChannelPrior() {
    //cout << src.channels() << " " << dark.channels() << endl;
    //cout << src.rows << " " << src.cols << endl;
    //cout << dark.rows << " " << dark.cols << endl;
    cout << "Starting to retrieve the dark channel prior of the image..." << endl;
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            uchar pixel = 255;
            Region roi(max(0, i - Radius), max(0, j - Radius), min(i + Radius, src.rows - 1), min(j + Radius, src.cols - 1));
            for (int h = roi.rmin; h <= roi.rmax; ++h) {
                for (int w = roi.cmin; w <= roi.cmax; ++w) {
                    cv::Vec3b vbc = src.at<cv::Vec3b>(h, w);
                    pixel = min(pixel, vbc[0]);
                    pixel = min(pixel, vbc[1]);
                    pixel = min(pixel, vbc[2]);
                }
            }
            dark.at<uchar>(i, j) = pixel;
        }
    }
    cout<<"dark channel done"<<endl;
    //showImage("dark", dark);
}
void DeHaze::getTransmission() {
    cout << "Starting to compute the transmission matrix..." << endl;
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            float pixel = 255.0;
            Region roi(max(0, i - Radius), max(0, j - Radius), min(i + Radius, src.rows - 1), min(j + Radius, src.cols - 1));
            for (int h = roi.rmin; h <= roi.rmax; ++h) {
                for (int w = roi.cmin; w <= roi.cmax; ++w) {
                    cv::Vec3b vbc = src.at<cv::Vec3b>(h, w);
                    pixel = min(pixel, float(int(vbc[0])) / Alight[0]);
                    pixel = min(pixel, float(int(vbc[1])) / Alight[1]);
                    pixel = min(pixel, float(int(vbc[2])) / Alight[2]);
                }
            }
            //cout << pixel << endl;
            tran.at<float>(i, j) = 1.0 - 0.95 * pixel;
            //cout << (1.0 - 0.95 * pixel) << " : " << tran.at<float>(i, j) << endl;
        }
    }
    //showImage("tran", tran);
    cout<<"get transmission done"<<endl;
}

void DeHaze::softMatting() {

}

static cv::Mat boxfilter(const cv::Mat &I, int r)
{
    cv::Mat result;
    cv::blur(I, result, cv::Size(r, r));
    return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
    if (mat.depth() == depth)
        return mat;
    cv::Mat result;
    mat.convertTo(result, depth);
    return result;
}

cv::Mat guidedFilter(const cv::Mat &origI, const cv::Mat &p, int R, double eps) {
    cv::Mat I;
    std::vector<cv::Mat> Ichannels;
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);
    int r = 2*R+1;

    cv::split(I, Ichannels);

    cv::Mat mean_I_r = boxfilter(Ichannels[0], r);
    cv::Mat mean_I_g = boxfilter(Ichannels[1], r);
    cv::Mat mean_I_b = boxfilter(Ichannels[2], r);

    cv::Mat mean_p = boxfilter(p, r);
    cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(p), r);
    cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(p), r);
    cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(p), r);

    cv::Mat var_I_rr = boxfilter(Ichannels[0].mul(Ichannels[0]), r) - mean_I_r.mul(mean_I_r) + eps;
    cv::Mat var_I_rg = boxfilter(Ichannels[0].mul(Ichannels[1]), r) - mean_I_r.mul(mean_I_g);
    cv::Mat var_I_rb = boxfilter(Ichannels[0].mul(Ichannels[2]), r) - mean_I_r.mul(mean_I_b);
    cv::Mat var_I_gg = boxfilter(Ichannels[1].mul(Ichannels[1]), r) - mean_I_g.mul(mean_I_g) + eps;
    cv::Mat var_I_gb = boxfilter(Ichannels[1].mul(Ichannels[2]), r) - mean_I_g.mul(mean_I_b);
    cv::Mat var_I_bb = boxfilter(Ichannels[2].mul(Ichannels[2]), r) - mean_I_b.mul(mean_I_b) + eps;
    cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
    cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
    cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

    cv::Mat invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
    cv::Mat invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
    cv::Mat invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
    cv::Mat invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
    cv::Mat invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
    cv::Mat invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);

    cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);

    invrr /= covDet;
    invrg /= covDet;
    invrb /= covDet;
    invgg /= covDet;
    invgb /= covDet;
    invbb /= covDet;

    cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
    cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
    cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);
    cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b);

    return (boxfilter(a_r, r).mul(Ichannels[0]) + boxfilter(a_g, r).mul(Ichannels[1]) + boxfilter(a_b, r).mul(Ichannels[2])
    + boxfilter(b, r));
}

void DeHaze::gFilter()
{
    gtran = guidedFilter(src, tran, 20, 0.001);
    cout<< "guidedfilter done"<<endl;

}

void DeHaze::recoverSceneRadiace(){
    cout << "g channel : " << tran.channels() << endl;
    cout << "Starting to dehaze the image..." << endl;
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            cv::Vec3b & dv = dst.at<cv::Vec3b>(i, j);
            //cv::Vec3b & sv = src.at<cv::Vec3b>(i, j);
            float t = std::max(gtran.at<float>(i, j), 0.1f);
            cv::Vec3f tmp;
            for (int k = 0; k < 3; k++) {
                tmp[k] = (float(dv[k]) - float(Alight[k])) / t;
            }
            //dst.at<cv::Vec3b>(i, j) = (src.at<cv::Vec3b>(i, j) - Alight) / std::max(tran.at<float>(i, j), 0.1f) + Alight;

            cv::Vec3b res(uchar(tmp[0]) + Alight[0], uchar(tmp[1]) + Alight[1], uchar(tmp[2]) + Alight[2]);
            dv = res;
            //cout << dv << endl;
        }
    }
    //cout << dst << endl;
    cout<<"recover done";
}
void DeHaze::getAtmosphericLight(){
    cout << "Starting to compute the atmospheric light intensity..." << endl;
    std::priority_queue<Pixel, std::vector<Pixel>, std::greater<Pixel>> pq;
    int num = src.rows * src.cols * 0.001;
    for (int i = 0; i < dark.rows; ++i) {
        for (int j = 0; j < dark.cols; ++j) {
            Pixel p(dark.at<uchar>(i, j), i, j);
            pq.push(p);
            if (pq.size() > num) {
                pq.pop();
            }
        }
    }
    unsigned int A[3] = {};
    while (!pq.empty()) {
        Pixel tmp = pq.top();
        //printf("%d (%d %d)\n", tmp.intensity, tmp.i, tmp.j);
        cv::Vec3b vcb = src.at<cv::Vec3b>(tmp.i, tmp.j);
        A[0] += vcb[0], A[1] += vcb[1], A[2] += vcb[2];
        //printf("%d %d %d\n", vcb[0], vcb[1], vcb[2]);
        pq.pop();
    }
    //cout << int(A[0]) << " " << int(A[1]) << " " << int(A[2]) << endl;
    Alight[0] = A[0] / num;
    Alight[1] = A[1] / num;
    Alight[2] = A[2] / num;
    cout << "atmosphere light:" << int(Alight[0]) << " " << int(Alight[1]) << " " << int(Alight[2]) << endl;
    //cout << num << " " << pq.size() << endl;
    //cout<<"at";
}


////
void DeHaze::showImage(const char * wname, MatType mt) {
    switch (mt) {
    case DeHaze::SRC:
        cv::imshow(wname, src);
        break;
    case DeHaze::DST:
        cv::imshow(wname, dst);
        break;
    case DeHaze::DARK:
        cv::imshow(wname, dark);
        break;
    case DeHaze::TRAN:
        cv::imshow(wname, tran);
        break;
    case DeHaze::GTRAN:
        cv::imshow(wname, gtran);
        break;
    }
    //cv::waitKey();
    if (cv::waitKey() == 5) {
        cv::destroyAllWindows();
    }
}


void DeHaze::saveImage(const char * wname, MatType mt) {
    switch (mt) {
    case DeHaze::SRC:
        cv::imwrite(wname, src);
        break;
    case DeHaze::DST:
        cv::imwrite(wname, dst);
        break;
    case DeHaze::DARK:
        cv::imwrite(wname, dark);
        break;
    case DeHaze::TRAN:
        cv::imwrite(wname, tran);
        break;
    case DeHaze::GTRAN:
        cv::imwrite(wname, gtran);
        break;
    }
    if (cv::waitKey() == 5) {
        cv::destroyAllWindows();
    }

}
void DeHaze::getImageName(string & fp) {

}
