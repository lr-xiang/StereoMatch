#include "StereoMatching.h"


bool StereoMatching::checkStereoPairValid(std::vector<cv::Mat>& stereoPair)
{
	if (stereoPair.size() != 2)
	{
		std::cout << "Not a stereo pair" << std::endl;
		return false;
	}

	if (stereoPair[0].data == NULL || stereoPair[1].data == NULL)
	{
		std::cout << "Image data null" << std::endl;
		return false;
	}

	if (stereoPair[0].rows != stereoPair[1].rows || stereoPair[0].cols != stereoPair[1].cols)
	{
		std::cout << "Image size different" << std::endl;
		return false;
	}

	return true;
}

bool StereoMatching::censusStereo(std::vector<cv::Mat>& stereoPair, cv::Mat& disparity)
{
	if (!checkStereoPairValid(stereoPair))
		return false;


	// rectangle window 
	int winRW = 4;
	int winRH = winRW * 2;
	int cols = stereoPair[0].cols;
	int rows = stereoPair[0].rows;
	int dMax = 50;
	
	float epsilon = 0.0045f;

	// normalize intensity
	cv::Mat left, right;

	stereoPair[0].convertTo(left, CV_32FC1, 1./255.);

	stereoPair[1].convertTo(right, CV_32FC1, 1./255.);

	// disparity of right image
	cv::Mat disp = cv::Mat::zeros(rows, cols, CV_16U);

#pragma omp parallel for
	for (int y = winRH; y < rows - winRH; y++)
	{
		unsigned short *p_disp = disp.ptr<unsigned short>(y);
		float *p_left = left.ptr<float>(y);
		float *p_right = right.ptr<float>(y);

		for (int x = winRW; x < cols - winRW - dMax; x++)
		{

			int minCost = 1000000;
			int bestDisp = -1;

			for (int d = 0; d <= dMax; d++)
			{
				int cost = 0;

				for (int h = -winRH; h <= winRH; h++)
				{
					for (int w = -winRW; w <= winRW; w++)
					{
						float lDiff = *(p_left + h*cols + x + w + d) - p_left[x];
						float rDiff = *(p_right + h*cols + x + w) - p_right[x];
						
						uchar lCensus, rCensus;

						if (lDiff > epsilon)
							lCensus = 2;
						else if (lDiff <-epsilon)
							lCensus = 0;
						else
							lCensus = 1;
						
						if (rDiff > epsilon)
							rCensus = 2;
						else if (rDiff <-epsilon)
							rCensus = 0;
						else
							rCensus = 1;

						if (lCensus != rCensus)
							cost++;
					}
				}

				if (cost < minCost)
				{
					minCost = cost;
					bestDisp = d;
				}
			}

			if (bestDisp != -1)
				p_disp[x] = (ushort)bestDisp;
		}
	}

	cv::Mat disp8u;
	disp.convertTo(disp8u, CV_8U);
	cv::imshow("disp", disp8u*3);
	cv::waitKey(0);
	return true;
}

bool StereoMatching::BMStereo(cv::Mat& imgl, cv::Mat& imgr, cv::Mat& disparity)
{
	//if (!checkStereoPairValid(stereoPair))
		//return false;
	std::cout<<"BM Stereo\n";

	int numDisp = 16 * 4;
	int blockSize = 21;
	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(numDisp, blockSize);
	bm->setPreFilterCap(31);
	bm->setBlockSize(blockSize);
	bm->setMinDisparity(0);
	bm->setNumDisparities(numDisp);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(1);
	
	cv::Mat disp, disp8;
  	bm->compute(imgl, imgr, disp);

	disp.convertTo(disp8, CV_8U, 1./16);

	imshow("disp", disp8);

	cv::waitKey(0);

	return true;

}

bool StereoMatching::SGBMStereo(cv::Mat& imgl, cv::Mat& imgr, int numDisp_x16, bool display, cv::Mat& disparity)
{

	std::cout<<"SGBM Stereo\n";

	int numDisp = 16 * numDisp_x16;
	int blockSize = 5;

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);

	sgbm->setPreFilterCap(63);

	sgbm->setBlockSize(blockSize);

	int cn = imgl.channels();

	sgbm->setP1(8 * cn*blockSize*blockSize);
	sgbm->setP2(32 * cn*blockSize*blockSize);
	sgbm->setMinDisparity(16);
	sgbm->setNumDisparities(numDisp);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(cv::StereoSGBM::MODE_HH);
    //sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

	cv::Mat disp, disp8, disp32f;
	sgbm->compute(imgl, imgr, disp);

	if (display)
	{
		disp.convertTo(disp8, CV_8U, 1./16);
		resize(disp8, disp8, Size(disp8.cols/2, disp8.rows/2));
		imshow("disp", disp8);
		cv::waitKey(0);
	}

	if (disp.type() == CV_16S)
		disp.convertTo(disp32f, CV_32F, 1./16);

	disparity = disp32f;

	return true;

}

bool StereoMatching::scaleStereoPairQMatrix(std::vector<cv::Mat>& stereoPair, const cv::Mat& Q, double scale, std::vector<cv::Mat>& outStereoPair, cv::Mat& outQ)
{
	cv::Mat left, right;
	cv::resize(stereoPair[0], left, cv::Size(), scale, scale, CV_INTER_AREA);
	cv::resize(stereoPair[1], right, cv::Size(), scale, scale, CV_INTER_AREA);
	outStereoPair.push_back(left);
	outStereoPair.push_back(right);

	outQ = Q.clone();

	for (int i = 0; i < 3; i++)
		outQ.at<double>(i, 3) *= scale;

	return true;
}
