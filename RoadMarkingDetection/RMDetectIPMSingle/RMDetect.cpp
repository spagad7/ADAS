#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Function Declarations
Mat calculateHomography(void);
void findMatch(Mat, Mat);
vector<Mat> loadTemplates(void);


int main() 
{

	string filename;
	vector<Mat> templateVector;

	filename = "../img.jpg";

	// Read source image
	Mat img_src = imread( filename, CV_LOAD_IMAGE_COLOR );

	// Calculate Homography
	Mat H = calculateHomography();
	
	// Create black destination image
	Mat img_warp(600, 800, CV_8UC3, Scalar(0,0,0));
	//Mat img_warp(900, 1200, CV_8UC3, Scalar(0,0,0));

	// Warp image
	warpPerspective(img_src, img_warp, H, img_warp.size());

	// Load templates
	//Mat img_temp = imread("../Template/template2.png", CV_LOAD_IMAGE_COLOR);
	templateVector = loadTemplates();

	findMatch(img_warp, templateVector);
	
	return 0;
}


// Function to load templates
vector<Mat> loadTemplates(void)
{
	vector<Mat> templateVector;
	int numTemplates = 2;
	string filename;

	for(int i=0; i<numTemplates; i++)
	{
		filename = "../Templates/template" + std::to_string(i) + ".png";
		Mat img_temp = imread(filename, CV_LOAD_IMAGE_COLOR);
		templateVector.push_back(img_temp.clone());
	}

	return(templateVector);
}


// Function to calculate Homography
Mat calculateHomography(void)
{
	vector<Point2f> pts_src, pts_dst;

	// Corners in source image
/*
	pts_src.push_back(Point2f(288, 299));
	pts_src.push_back(Point2f(500, 299));
	pts_src.push_back(Point2f(740, 383));
	pts_src.push_back(Point2f(4, 383));
*/
/*	
	pts_src.push_back(Point2f(340, 309));
	pts_src.push_back(Point2f(445, 309));
	pts_src.push_back(Point2f(574, 410));
	pts_src.push_back(Point2f(73, 398));
*/

	pts_src.push_back(Point2f(300, 300));
	pts_src.push_back(Point2f(500, 300));
	pts_src.push_back(Point2f(800, 420));
	pts_src.push_back(Point2f(1, 420));


	// Corners in destination image
	pts_dst.push_back(Point2f(1,1));
	pts_dst.push_back(Point2f(800,1));
	pts_dst.push_back(Point2f(800,600));
	pts_dst.push_back(Point2f(1,600));

	// Calculate homography
	Mat H = findHomography(pts_src, pts_dst);

	return(H);
}


// Function to detect road markings using template matching
void findMatch(Mat img_src, Mat img_temp)
{
	Mat img_src_gray, img_temp_gray, img_src_edge, img_temp_edge, img_result, img_temp_resize;
	//int rows, cols;
	//double scale = 1.5;

	//rows = round(104);
	//cols = round(306);

	//resize(img_temp, img_temp, Size(cols, rows), 0,0, INTER_LINEAR);

/*
	// Convert both the image and template to gray scale
	cvtColor(img_src, img_src_gray, COLOR_BGR2GRAY);
	cvtColor(img_temp, img_temp_gray, COLOR_BGR2GRAY);

	// Get edge image of img_dst and img_temp
	Canny(img_src_gray, img_src_edge, 50, 200);
	Canny(img_temp_gray, img_temp_edge, 50, 200);
*/

	// Template matching
	matchTemplate(img_src, img_temp, img_result, CV_TM_CCOEFF);

	// Find location of bounding rectangle
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(img_result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	rectangle(img_src, maxLoc, Point(maxLoc.x + img_temp.cols, maxLoc.y + img_temp.rows), Scalar(0, 255, 0), 2, 8, 0);
	rectangle(img_result, maxLoc, Point( maxLoc.x + img_temp.cols , maxLoc.y + img_temp.rows ), Scalar(0, 255, 0), 2, 8, 0 );

	// Display images
	imshow("Source Image", img_src);
	//imshow("Template Image", img_temp);
	//imshow("Result Image", img_result);
	waitKey(0);

}