#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <math.h>


using namespace cv;
using namespace std;

// Function Declarations
Mat calculateHomography(void);
Mat calculateInvHomography(void);
vector<Mat> loadTemplates(void);
void findLaneMarker(Mat img);
void findMatch(Mat, vector<Mat>, vector<Point> &);
Mat drawPolygon(Mat , Mat, vector<Point> &);
double calcSSD(Mat, Mat);


int main() 
{

	string filename;
	vector<Mat> templates;
	vector<Point> matchPoints;
	int numImages = 1433;
	//int numImages = 1;

	// Load templates
	templates = loadTemplates();

	// Calculate Homography
	Mat H = calculateHomography();
	//cout << "H = " << H << endl << endl;

	// Calculate Inverse Homography
	Mat I = calculateInvHomography();
	//cout << "I = " << I << endl << endl;

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

		// Set region of interest
		Mat img_roi = img_src(Rect(285, 300, 230, 90));
		//imshow("roi", img_roi);
		String roifilename = "img_roi_" + std::to_string(i) + ".jpg";
		imwrite(roifilename, img_roi);
		//waitKey(0);
		
		// Create black destination image
		Mat img_warp(300, 200, CV_8UC3, Scalar(0,0,0));
		//Mat img_warp(900, 1200, CV_8UC3, Scalar(0,0,0));

		// Warp image
		warpPerspective(img_roi, img_warp, H, img_warp.size());
		//imshow("Warped image", img_warp);
		String warpfilename = "img_warp_" + std::to_string(i) + ".jpg";
		imwrite(warpfilename, img_warp);


		// Draw ROI on img
		line(img_src, Point(375, 300), Point(445, 300), Scalar(0, 0, 255), 2, 8);
		line(img_src, Point(445, 300), Point(515, 390), Scalar(0, 0, 255), 2, 8);
		line(img_src, Point(515, 390), Point(285, 390), Scalar(0, 0, 255), 2, 8);
		line(img_src, Point(285, 390), Point(375, 300), Scalar(0, 0, 255), 2, 8);

		//imshow("Source Image", img_src);

		// Detect lane markers
		//findLaneMarker(img_warp);

		// Detect road markings
		findMatch(img_warp, templates, matchPoints);

		// Draw polygon for matched region
		Mat img_final = drawPolygon(img_src, I, matchPoints);

		imshow("Output", img_final);

		matchPoints.clear();

		waitKey(1);
	}
	
	return 0;
	
}



// Function to calculate Homography
Mat calculateHomography(void)
{
	vector<Point2f> pts_src, pts_dst;

	// Corners in source image
	pts_src.push_back(Point2f(90, 1));
	pts_src.push_back(Point2f(160, 1));
	pts_src.push_back(Point2f(230, 90));
	pts_src.push_back(Point2f(1, 90));

	// Corners in destination image
	pts_dst.push_back(Point2f(1,1));
	pts_dst.push_back(Point2f(200,1));
	pts_dst.push_back(Point2f(200,300));
	pts_dst.push_back(Point2f(1,300));

	// Calculate homography
	// Note: findHomography is better than getPerspectiveTranform. 
	// It uses RANSAC
	Mat H = findHomography(pts_src, pts_dst);

	return(H);
}

Mat calculateInvHomography(void)
{
	vector<Point2f> pts_src, pts_dst;

	// Corners in source image
	pts_src.push_back(Point2f(8, 137));
	pts_src.push_back(Point2f(171, 139));
	pts_src.push_back(Point2f(172, 274));
	pts_src.push_back(Point2f(2, 273));
//	pts_src.push_back(Point2f(39, 136));
//	pts_src.push_back(Point2f(144, 135));
//	pts_src.push_back(Point2f(140, 278));
//	pts_src.push_back(Point2f(30, 278));

	// Corners in destination image
	pts_dst.push_back(Point2f(76,20));
	pts_dst.push_back(Point2f(160,20));
	pts_dst.push_back(Point2f(187,69));
	pts_dst.push_back(Point2f(24,69));
//	pts_dst.push_back(Point2f(91,192));
//	pts_dst.push_back(Point2f(144,19));
//	pts_dst.push_back(Point2f(156,71));
//	pts_dst.push_back(Point2f(48,72));

	// Calculate homography
	// Note: findHomography is better than getPerspectiveTranform. 
	// It uses RANSAC
	Mat I = findHomography(pts_src, pts_dst);

	return(I);	
}



// Function to load templates
vector<Mat> loadTemplates(void)
{
	int numTemplates = 23;
	string filename;
	vector<Mat> templateVector;
	

	for(int i=1; i<=numTemplates; i++)
	{
		filename = "../Templates/temp" + std::to_string(i) + ".png";
		Mat img_temp = imread(filename, CV_LOAD_IMAGE_COLOR);
		templateVector.push_back(img_temp.clone());
	}

	return(templateVector);
}



// Function to detect road markings using template matching
void findMatch(Mat img_src, vector<Mat> templates, vector<Point> &matchPoints)
{
	Mat img_src_gray, img_src_edge, img_temp_gray, img_temp_edge;
	unsigned int numTemplates = templates.size();
	vector<double> maxVal(numTemplates), minVal(numTemplates), ssd(numTemplates);
	vector<Point> minLoc(numTemplates), maxLoc(numTemplates);
	int cannyLowThresh = 50;
	int cannyHighThresh = 150;
	double matchThresh = 0.2;

	//resize(img_temp, img_temp, Size(cols, rows), 0,0, INTER_LINEAR);

	// Get edge image of img_dst and img_temp
	cvtColor(img_src, img_src_gray, CV_BGR2GRAY);
	blur(img_src_gray, img_src_edge, Size(3,3) );
	Canny(img_src_edge, img_src_edge, cannyLowThresh, cannyHighThresh);
	
	
	//#pragma omp parallel for
	for(unsigned int i=0; i<templates.size(); i++)
	{
		cvtColor(templates[i], img_temp_gray, CV_BGR2GRAY);
		blur(img_temp_gray, img_temp_edge, Size(3,3) );
		Canny(img_temp_edge, img_temp_edge, cannyLowThresh, cannyHighThresh);
		//imshow("Template Edge", img_temp_edge);
		//waitKey(0);

		Mat img_result;
		//matchTemplate(img_src, templates[i], img_result, CV_TM_CCORR_NORMED);
		matchTemplate(img_src_edge, img_temp_edge, img_result, CV_TM_CCORR_NORMED);
		//normalize(img_result, img_result, 0, 1, NORM_MINMAX, -1, Mat());
		minMaxLoc(img_result, &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i], Mat());
		Mat img_src_cropped = img_src(Rect(maxLoc[i].x, maxLoc[i].y, templates[i].cols, templates[i].rows));
		//ssd[i] = calcSSD(img_src_cropped, templates[i]);
		//cout << "SSD: " << ssd[i] << endl;
		//cout << "maxVal: " << maxVal[i] << endl;
	}

	//int bestIndex = distance(ssd.begin(), min_element(ssd.begin(), ssd.end()));
	int bestIndex = distance(maxVal.begin(), max_element(maxVal.begin(), maxVal.end()));
	//cout << "maxVal: " << maxVal[bestIndex] << endl;

	if(maxVal[bestIndex] >= matchThresh)
	{
		rectangle(img_src, maxLoc[bestIndex], 
			Point(maxLoc[bestIndex].x + templates[bestIndex].cols, maxLoc[bestIndex].y + templates[bestIndex].rows), 
			Scalar(0, 255, 0), 2, 8, 0);

		// Top Left
		matchPoints.push_back(maxLoc[bestIndex]);
		// Top Right
		matchPoints.push_back(Point(maxLoc[bestIndex].x + templates[bestIndex].cols, maxLoc[bestIndex].y));
		// Bottom Right
		matchPoints.push_back(Point(maxLoc[bestIndex].x + templates[bestIndex].cols, maxLoc[bestIndex].y + templates[bestIndex].rows));
		// Bottom Left
		matchPoints.push_back(Point(maxLoc[bestIndex].x, maxLoc[bestIndex].y + templates[bestIndex].rows));

	}
	//imshow("Warped image", img_src);

	// Display images
	//imshow("Detected Roadmarking", img_src);

}



Mat drawPolygon(Mat img, Mat I, vector<Point> &warpedPoints)
{
	if(!warpedPoints.empty())
	{
		vector<Point> origPoints(warpedPoints.size());

		//cout << "warpedPoints" << warpedPoints << endl;

		for(unsigned int i=0; i<warpedPoints.size(); i++)
		{
			//cout << "Warped Point = " << warpedPoints[i] << endl;
			int data[] = {warpedPoints[i].x, warpedPoints[i].y, 1};
			Mat V(3, 1, CV_32S, data);
			V.convertTo(V, CV_64F);
			//cout << "V vector = " << V << endl;
			//cout << "Hinv = " << H.inv() << endl;
			Mat prod = I * V;
			//cout << "Prod = " << prod << endl;
			//prod.convertTo(prod, CV_32S);
			//cout << "Prod CV_32S = " << prod << endl;
			//cout << "prod.at<double>(0) = " << prod.at<double>(0) << endl;
			//cout << "prod.at<double>(1) = " << prod.at<double>(1) << endl;
			//cout << "prod.at<double>(2) = " << prod.at<double>(2) << endl;
			if(prod.at<double>(2) != 0)
			{
				origPoints[i].x = ( prod.at<double>(0) / prod.at<double>(2) ) + 285;
				origPoints[i].y = ( prod.at<double>(1) / prod.at<double>(2) ) + 300;
			}
			else
			{
				origPoints[i].x = 0;
				origPoints[i].y = 0;
			}
			
		}


		//cout << "OrigPoints" << origPoints << endl << endl;


		// Draw Lines. Points are in order: TopLeft, TopRight, BottomRight, BottomLeft
		line(img, origPoints[0], origPoints[1], Scalar(0, 255, 0), 2, 8);
		line(img, origPoints[1], origPoints[2], Scalar(0, 255, 0), 2, 8);
		line(img, origPoints[2], origPoints[3], Scalar(0, 255, 0), 2, 8);
		line(img, origPoints[3], origPoints[0], Scalar(0, 255, 0), 2, 8);

		origPoints.clear();
	}
	
	return(img);
}



/*
double calcSSD(Mat img, Mat temp)
{
	double ssd = 0;

	// Convert the images to gray scale
	//cvtColor(img, img, CV_BGR2GRAY);
	//cvtColor(temp, temp, CV_BGR2GRAY);

	//cout << img.at<int>(1,1) <<endl;

	// Computer SSD
	for(int row=0; row<temp.rows ; row++)
	{
		for(int col=0; col<temp.cols; col++)
		{
			ssd += pow((img.at<char>(row,col) - temp.at<char>(row,col)),2);
		}
	}

	return ssd;
}
*/


/*
void findLaneMarker(Mat img)
{
	vector<Vec4i> lines;
	double rho = 3;
	double theta = CV_PI/180;
	int threshold = 90;
	double minLineLength = 40;
	double maxLineGap = 100;
	Mat img_gray, img_edge;

	cvtColor(img, img_gray, CV_BGR2GRAY);
	Canny(img_gray, img_edge, 50, 200);

	HoughLinesP(img_edge, lines, rho, theta, threshold, minLineLength, maxLineGap);

	#pragma omp parallel for
	for(size_t i=0; i<lines.size(); i++)
	{
		int dx = lines[i][0] - lines[i][2];
		int dy = lines[i][1] - lines[i][3];
		double angle = atan2(dy, dx) * 180/CV_PI;

		// Only select lines within specified range of angle
		if(fabs(angle) > 75 && fabs(angle) < 105)
		{
			line( img, Point(lines[i][0], lines[i][1]),
        	Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );

        	//cout << max(lines[i][0], lines[i][3]) << endl;
		}
		
	}
	//imshow("Lines", img);
	//waitKey(0);
}
*/
