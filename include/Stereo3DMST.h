#ifndef __STEREO_3DMST_H__
#define __STEREO_3DMST_H__

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" void stereo3dmst(std::string left_name, std::string right_name, cv::Mat & leftImg, cv::Mat & rightImg, cv::Mat & leftDisp, cv::Mat & rightDisp, std::string data_cost, int Dmax);

void startTimer();

double getTimer();

#endif
