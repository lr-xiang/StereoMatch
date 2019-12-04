#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/viz.hpp"
#include "Stereo3DMST.h"
#include <stdio.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
using namespace boost::filesystem;

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

int main(int argc, char** argv)
{
    FileStorage fs("/home/lietang/Projects/CameraCalibration/output/Stereo_cam_para.xml", FileStorage::READ);
    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    //M1 *= scale;
    //M2 *= scale;

    Mat R, T, R1, P1, R2, P2, Q;
    fs["R"] >> R;
    fs["T"] >> T;
    fs["R1"] >> R1;
    fs["R2"] >> R2;
    fs["P1"] >> P1;
    fs["P2"] >> P2;
    fs["Q"] >> Q;
    fs.release();

    std::string path_str = "/home/lietang/FieldData/PhenoSensor/Calibrate/lidar_calibration/image_set2/";

    directory_iterator itr(path_str);


    for(; itr!=directory_iterator(); ++itr)
    {
        std::string file_ext = itr->path().extension().c_str();
        if(file_ext != ".pcd" && file_ext != "pcd")
            continue;

        std::string file_stem = itr->path().stem().c_str();

        std::cout<<"Image path + stem: "<<(path_str + file_stem)<<std::endl;

        cv::Mat img_left = imread(path_str + file_stem + "_r.bmp");
        cv::Mat img_right = imread(path_str + file_stem + "_l.bmp");

        cv::Size img_size = img_left.size();

        std::cout<<"Rectify"<<std::endl;
        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, img_size);

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img_left, img1r, map11, map12, INTER_LINEAR);
        remap(img_right, img2r, map21, map22, INTER_LINEAR);

        img_left = img1r;
        img_right = img2r;

        cv::imwrite("img1r.png",img_left);
        cv::imwrite("img2r.png", img_right);

        std::cout<<"Draw canvas"<<std::endl;

        Mat canvas;
        double sf;
        int w, h;
        bool isVerticalStereo = false;
        if( !isVerticalStereo )
        {
            //sf = 900./MAX(s.imageSize.width, s.imageSize.height);
            sf=1;
            w = cvRound(img_size.width*sf);
            h = cvRound(img_size.height*sf);
            canvas.create(h, w*2, CV_8UC3);
        }
        else
        {
            //sf = 450./MAX(s.imageSize.width, s.imageSize.height);
            sf=1;
            w = cvRound(img_size.width*sf);
            h = cvRound(img_size.height*sf);
            canvas.create(h*2, w, CV_8UC3);
        }

        for( int k = 0; k < 2; k++ )
        {
            // cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            std::cout<<"Put images"<<std::endl;
            if(k==0)
                resize(img_left, canvasPart,
                    canvasPart.size(), 0, 0, INTER_AREA);
            else
                resize(img_right, canvasPart,
                       canvasPart.size(), 0, 0, INTER_AREA);
        }
        std::cout<<"draw lines"<<std::endl;
        if( !isVerticalStereo )
        {
            for( int j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
            line(canvas, Point(canvas.cols/2, 0), Point(canvas.cols/2, canvas.rows-1), Scalar(0, 255, 0), 1, 8);
        }
        else
        {
            for( int j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
            line(canvas, Point(0, canvas.rows/2), Point(canvas.cols/2, canvas.rows-1), Scalar(0, 255, 0), 1, 8);
        }
        imshow("rectified", canvas);
        cv::waitKey(50);

        cv::Mat dispMat_left, dispMat_right, disp_l8;

        startTimer();

        stereo3dmst("img1r.png", "img2r.png", img_left, img_right, dispMat_left, dispMat_right, "MCCNN_acrt", 100);

        double time_processing = getTimer();
        std::cout<<"Timer:"<<time_processing<<std::endl;
        dispMat_left.convertTo(disp_l8, CV_8U, 255/(100));

        cv::imshow("disp_lefft", disp_l8);
        cv::waitKey(1000);

        Mat xyz;

        for(int i = 0; i < img_size.area(); i++)
        {
            if(dispMat_left.at<float>(i) < 10)
                dispMat_left.at<float>(i) = 10;
        }
        reprojectImageTo3D(dispMat_left, xyz, Q, true);

        string cloud_filename = file_stem+".pcd";

        std::cout<<"Saving cloud:"<< cloud_filename << " ..."<<std::endl;

        pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;

        for(int i = 0; i < dispMat_left.rows * dispMat_right.cols; i++)
        {
            pcl::PointXYZRGB point;
            cv::Vec3f coor_3d = xyz.at<cv::Vec3f>(i);
            point.x = coor_3d[0];
            point.y = coor_3d[1];
            point.z = coor_3d[2];
            cv::Vec3b rgb_vec = img_left.at<cv::Vec3b>(i);
            uint32_t rgb_int = rgb_vec[2] * 0x10000 + rgb_vec[1] * 0x100 + rgb_vec[0];
            point.rgb = *reinterpret_cast<float*>(&rgb_int);
            pcl_cloud.push_back(point);
        }

        pcl::io::savePCDFileASCII(cloud_filename.c_str(), pcl_cloud);

        std::cout<<"Saving done"<<std::endl;

    }
    return 0;
}
