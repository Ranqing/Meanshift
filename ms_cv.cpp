#include "common.h"
#include "basic.h"
#include "ms_cv.h"

#include "msImageProcessor.h"
#include <opencv2/core/core.hpp>

/**
 * @param src Image to segment
 * @param labels_dst	cv::Mat where the (int) labels will be written in
 * @param segment_dst	cv::Mat the segmentation results
 * @param density_dst	cv::Mat number of pixels in each region
 * @return Number of different labels
 */

int mean_shift_segmentation(const cv::Mat& src, cv::Mat& labels_dst, cv::Mat& segment_dst, cv::Mat& density_dst, int spatial_variance, float color_variance, int minsize)
{
	msImageProcessor proc;
	proc.DefineImage(src.data, (src.channels() == 3 ? COLOR : GRAYSCALE), src.rows, src.cols);
	proc.Segment(spatial_variance,color_variance, minsize, MED_SPEEDUP);//HIGH_SPEEDUP, MED_SPEEDUP, NO_SPEEDUP; high: set speedupThreshold, otherwise the algorithm uses it uninitialized!

	int regions_count = proc.GetRegionsCnt();

	//标签结果
	labels_dst = cv::Mat(src.size(), CV_32SC1);
	proc.GetRegionsLabels(labels_dst.data);

	//每个区域像素数目
	density_dst = cv::Mat(regions_count, 1, CV_32SC1);
	proc.GetRegionsDensitys(density_dst.data);

	//meanshift分割图像
	segment_dst = cv::Mat(src.size(), CV_8UC3);
	proc.GetResults(segment_dst.data);	

	//test
	imshow("segmentation", segment_dst);
	waitKey(0);
	destroyWindow("segmentation");

	return regions_count;
} 

int main(int argc, char * argv[])
{
	if (argc != 3)
	{
		cout << "Usage: meanshift-cv.exe folder image-name" << endl;
		return -1;
	}
	
	string folder = string(argv[1]);
	string imname = string(argv[2]);
	string fn  =  folder + imname + ".png";
	Mat src = imread(fn);
	if (src.data == NULL)
	{
		cout << "failed to open " << fn << endl;
		return 1;
	}
	cout << "load input image done." << endl;

	//Meanshift参数
	int spatial_variance = 7;
	int color_variance = 9;
	int minsize = 500;

	Mat labels;
	Mat segments;
	Mat densitys;

	int regioncnt = mean_shift_segmentation(src, labels, segments, densitys, spatial_variance, color_variance, minsize);
	cout << "mean-shift done:  " << regioncnt << " regions." << endl;

	string savefn;

	//save labels
	savefn = folder + "labels_" + imname + ".txt";
	saveMat(labels, savefn);

	savefn = folder + "densitys_" + imname + ".txt";
	saveMat(densitys, savefn);

	//save segments
	savefn = folder + "meanshift_" + imname + ".jpg";
	imwrite(savefn, segments);

	return 1;
}