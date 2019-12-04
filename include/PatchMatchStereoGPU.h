#ifndef __PATCHMATCHSTEREO_GPU_H__
#define __PATCHMATCHSTEREO_GPU_H__
#define USE_PCL 0	

#if USE_PCL
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#endif

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


extern "C" void PatchMatchStereoGPU(const cv::Mat& leftImg, const cv::Mat& rightImg, int winRadius, int Dmin, int Dmax, int iteration, float scale, bool showLeftDisp, cv::Mat& leftDisp, cv::Mat& rightDisp);

extern "C" void PatchMatchStereoNL2TGV(const cv::Mat& leftImg, const cv::Mat& rightImg, int winRadius, int Dmin, int Dmax, int iteration, float scale, bool showLeftDisp, cv::Mat& leftDisp, cv::Mat& rightDisp, std::string& data_cost, std::string& smoothess_prior, std::string& left_name, std::string& right_name);

extern "C" void costVolumeStereoPlusVariationalDenoise(const cv::Mat& left_img, const cv::Mat& right_img, const int min_disp=0, const int max_disp=60);

void StartTimer();
double GetTimer();
#endif
