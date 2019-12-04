/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/viz.hpp"

#include <stdio.h>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include "StereoMatching.h"
//#include "PatchMatchStereoGPU.h"
#include "pm.h"

using namespace cv;
using namespace pcl;


float pre_x = 0, pre_y = 0, pre_z = 0, Dist = 0;
void pp_callback(const pcl::visualization::PointPickingEvent& event)
{
	float x, y, z;
	event.getPoint(x, y, z);
	Dist = sqrt(pow(x - pre_x, 2) + pow(y - pre_y, 2) + pow(z - pre_z, 2));
	//	Eigen::Vector3f dir(pre_x-x, pre_y-y, pre_z-z);
	//	dir.normalize();	
	pre_x = x;
	pre_y = y;
	pre_z = z;
	std::cout << "x:" << x << " y:" << y << " z:" << z << " distance:" << Dist/*<<" nx:"<<dir(0)<<" ny:"<<dir(1)<<" nz:"<<dir(2)*/ << std::endl;

}

int main(int argc, char** argv)
{

	cout<<"input image id "<<argv[1]<<endl;

    std::string intrinsic_filename = "";

    float scale;

    scale = 1;

    intrinsic_filename = "/home/lietang/cam_stereo_pheno.yml";

    int color_mode = -1;
	std::string img_id(argv[1]);
    Mat img1 = imread("0000"+img_id+"_191400042.jpg", color_mode);
    Mat img2 = imread("0000"+img_id+"_191400039.jpg", color_mode);

	std::vector<cv::Mat> stereoPair;

	stereoPair.push_back(img1);
	stereoPair.push_back(img2);


    if (img1.empty() || img2.empty())
    {
        printf("Command-line parameter error: could not load the first input image file\n");
        return -1;
    }

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    if( !intrinsic_filename.empty() )
    {
        std::cout<<"Reading intrinsic."<<std::endl;
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            return -1;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;
        fs["R1"] >> R1;
        fs["R2"] >> R2;
        fs["P1"] >> P1;
        fs["P2"] >> P2;
        fs["Q"] >> Q;
        std::cout<<"Rectify"<<std::endl;
        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;

/*
		Mat showimg;
		resize(img1, showimg, Size(img1.cols/2, img1.rows/2));
		imshow("disp", showimg);
		cv::waitKey(0); 
*/

	}

    std::cout<<"Getting disparity map."<<std::endl;

    Mat disp, xyz, disp_right;

	StereoMatching sm;
    int64 t = getTickCount();
	//SGBM
	sm.SGBMStereo(img1, img2, 16, true, disp);
	//PMS
	//PatchMatchStereoGPU(img1, img2, 5, 0, 100, 10, 3.0, true, disp, disp_right);


/* 	//PMS from github
	pm::PatchMatch patch_match(0.9f, 10.0f, 10.0f, 2.0f); //alpha, gamma, tau_c, tau_g
	patch_match.set(img1, img2);
	patch_match.process(3);
	patch_match.postProcess();

	disp = patch_match.getLeftDisparityMap();
	cv::Mat1f disp2 = patch_match.getRightDisparityMap();

	cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX);
	cv::normalize(disp2, disp2, 0, 255, cv::NORM_MINMAX);

	cv::imwrite("left_disparity.png", disp);
	cv::imwrite("right_disparity.png", disp2);
*/

	//get time
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

	//sm.BMStereo(img1, img2, disp);

    reprojectImageTo3D(disp, xyz, Q, true);

   
//visualization 

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZRGB>);

	for (int y = 0; y < xyz.rows; y++)
	{
		        for (int x = 0; x < xyz.cols; x++)
		        {
		            Point3f pointOcv = xyz.at<Point3f>(y, x);

		                //Insert info into point cloud structure
		                pcl::PointXYZRGB point;
						if(fabs(pointOcv.z)< 0.8f && fabs(pointOcv.z)>0.002f){
				            point.x = -pointOcv.x;  // xyz points transformed in Z upwards system
				            point.y =  pointOcv.z;
				            point.z =  pointOcv.y;

							cv::Vec3b intensity = img1.at<cv::Vec3b>(y,x); //BGR 
							uint32_t rgb = (static_cast<uint32_t>(intensity[2]) << 16 | static_cast<uint32_t>(intensity[1]) << 8 | static_cast<uint32_t>(intensity[0])); 

				            point.rgb = *reinterpret_cast<float*>(&rgb);
				            cloud_xyz->points.push_back (point);  // pushback actual point 
						}

		         }
	}

	cloud_xyz->width = 1;
	cloud_xyz->height = cloud_xyz->points.size();

	std::cout<<"cloud size: "<<cloud_xyz->size()<<std::endl;

	pcl::visualization::PCLVisualizer viewer("ICP demo");

	viewer.registerPointPickingCallback(&pp_callback);

	viewer.addPointCloud(cloud_xyz);

	viewer.spin();

    pcl::io::savePCDFileASCII("test.pcd", *cloud_xyz);

    std::cout<<"Saving done"<<std::endl;

    return 0;
}




