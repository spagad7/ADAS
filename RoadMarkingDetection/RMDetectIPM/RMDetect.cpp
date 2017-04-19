#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main() {

	int numImages = 1433;

	for(int i=1; i<=numImages; i++)
	{
		string filename;

		// Generate filename
		if(i<=9)
			filename = "../../RoadMarkings/roadmark_000" + std::to_string(i) + ".jpg";
		else if(i>=10 && i<=99)
			filename = "../../RoadMarkings/roadmark_00" + std::to_string(i) + ".jpg";
		else if(i>=100 && i<=999)
			filename = "../../RoadMarkings/roadmark_0" + std::to_string(i) + ".jpg";
		else
			filename = "../../RoadMarkings/roadmark_" + std::to_string(i) + ".jpg";
			
		// Read source image
		Mat img_src = imread( filename, CV_LOAD_IMAGE_COLOR );

		// Specify points for estimating homography
		// Corners of road
		vector<Point2f> pts_src;
/*
		pts_src.push_back(Point2f(350, 300));
		pts_src.push_back(Point2f(450, 300));
		pts_src.push_back(Point2f(800, 420));
		pts_src.push_back(Point2f(1, 420));
*/

		pts_src.push_back(Point2f(340, 309));
		pts_src.push_back(Point2f(445, 309));
		pts_src.push_back(Point2f(574, 410));
		pts_src.push_back(Point2f(73, 398));

		// Corners of destination image
		vector<Point2f> pts_dst;
		pts_dst.push_back(Point2f(1,1));
		pts_dst.push_back(Point2f(800,1));
		pts_dst.push_back(Point2f(800,600));
		pts_dst.push_back(Point2f(1,600));

		// Calculate homography
		Mat H = findHomography(pts_src, pts_dst);

		// Create black destination image
		Mat img_dst(600, 800, CV_8UC3, Scalar(0,0,0));

		// Warp image
		warpPerspective(img_src, img_dst, H, img_dst.size());

		// Display images
		imshow("Source Image:", img_src);
		imshow("Destination Image:", img_dst);
		waitKey(50);
	}
	
}


