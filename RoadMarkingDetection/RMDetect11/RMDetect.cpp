#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <errno.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <math.h>


using namespace cv;
using namespace std;

// Function Declarations
Mat calculateHomography(void);
void loadImages(string, vector<Mat> &, bool);
void getFilesInDir(string, vector<string> &, bool);
int findMatch(Mat , vector<Mat>, vector<Point> &);
Mat drawTextPolygon(Mat , Mat, vector<Point> &, int);
//void findLaneMarker(Mat img);
//double calcSSD(Mat, Mat);

map<int, string> template_names;

int main() 
{

	vector<Mat> templates;
	vector<Mat> images;
	vector<Point> matchPoints;
	int imgCount = 0;

	// Load templates
	string tempDirName = "../Templates";
	loadImages(tempDirName, templates, true);

	// Read video
	VideoCapture cap("../Video/Cycle.mp4");
	if(!cap.isOpened())
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	// Calculate Homography
	Mat H = calculateHomography();

	// Calculate Inverse Homography
	Mat I = H.inv();


	while(1)
	{
		Mat img_src;
		bool bSuccess = cap.read(img_src);

		//string filename1 = "../Output/img" + to_string(imgCount) + ".jpg";
		//imwrite(filename1, img_src);

		// Set region of interest
		//Mat img_roi = img_src(Rect(285, 300, 230, 90));
		
		// Create black destination image
		Mat img_warp(300, 200, CV_8UC3, Scalar(0,0,0));

		// Warp image
		warpPerspective(img_src, img_warp, H, img_warp.size());

		//string filename2 = "../OutputWarp/imgWarp" + to_string(imgCount) + ".jpg";
		//imwrite(filename2, img_warp);
		//imshow("Warped Img", img_warp);

		// Draw ROI on img

		line(img_src, Point(604, 170), Point(694, 164), Scalar(0, 0, 255), 2, 8);
		line(img_src, Point(694, 164), Point(1154, 667), Scalar(0, 0, 255), 2, 8);
		line(img_src, Point(1154, 667), Point(260, 667), Scalar(0, 0, 255), 2, 8);
		line(img_src, Point(260, 667), Point(604, 170), Scalar(0, 0, 255), 2, 8);


		// Detect lane markers
		//findLaneMarker(img_warp);

		// Detect road markings
		int matchIndex = findMatch(img_warp, templates, matchPoints);

		// Draw polygon for matched region
		Mat img_final = drawTextPolygon(img_src, I, matchPoints, matchIndex);

		
		string filename = "../Output/img" + to_string(imgCount) + ".jpg";
		imwrite(filename, img_src);
		imshow("Output", img_src);

		waitKey(1);

		imgCount++;

		matchPoints.clear();

	}

	return 0;
	
}



// Function to load images and templates
void loadImages(string dirName, vector<Mat> &imgVector, bool flag)
{
	vector<string> files;

	getFilesInDir(dirName, files, flag);

	for(unsigned int i=0; i<files.size(); i++)
	{
		Mat img_tmp = imread(files[i], CV_LOAD_IMAGE_COLOR);
		imgVector.push_back(img_tmp.clone());
	}

}



// Function to get filenames
void getFilesInDir(string dirName, vector<string> &files, bool flag)
{
	DIR *dp;
	struct dirent *dirp;

	if((dp = opendir(dirName.c_str())) == NULL)
	{
		cout << "Error(" << errno << ") opening" << dirName << endl;
	}

	while ((dirp = readdir(dp)) != NULL)
	{
		string fileName = dirName + "/" + string(dirp->d_name);
		if(string(dirp->d_name) != "." && string(dirp->d_name) != "..")
			files.push_back(fileName);
	}

	closedir(dp);

	sort(files.begin(), files.end());

	// Save template names for labelling detected templates
	if(flag == true)
	{
		for(unsigned int i=0; i<files.size(); i++)
		{
			template_names[i] = files[i];
		}
	}	
}



// Function to calculate Homography
Mat calculateHomography(void)
{
	vector<Point2f> pts_src, pts_dst;

	// Corners in source image
	pts_src.push_back(Point2f(604, 170));
	pts_src.push_back(Point2f(694, 164));
	pts_src.push_back(Point2f(1154, 667));
	pts_src.push_back(Point2f(260, 667));

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



// Function to detect road markings using template matching
int findMatch(Mat img_src, vector<Mat> templates, vector<Point> &matchPoints)
{
	Mat img_src_gray, img_src_edge, img_src_thresh;
	unsigned int numTemplates = templates.size();
	vector<double> maxVal(numTemplates), minVal(numTemplates), ssd(numTemplates);
	vector<Point> minLoc(numTemplates), maxLoc(numTemplates);
	int cannyLowThresh = 50;
	int cannyHighThresh = 100;
	double matchThresh = 0.45;

	//resize(img_temp, img_temp, Size(cols, rows), 0,0, INTER_LINEAR);

	// Get edge image of img_dst and img_temp
	cvtColor(img_src, img_src_gray, CV_BGR2GRAY);
	//threshold(img_src_gray, img_src_thresh, 150, 255, THRESH_BINARY);
	blur(img_src_gray, img_src_edge, Size(5,5) );
	Canny(img_src_edge, img_src_edge, cannyLowThresh, cannyHighThresh);
	
	
	#pragma omp parallel for
	for(unsigned int i=0; i<templates.size(); i++)
	{
		Mat img_temp_gray, img_temp_edge, img_temp_thresh;
		cvtColor(templates[i], img_temp_gray, CV_BGR2GRAY);
		//threshold(img_temp_gray, img_temp_thresh, 150, 255, THRESH_BINARY);
		blur(img_temp_gray, img_temp_edge, Size(5,5) );
		Canny(img_temp_edge, img_temp_edge, cannyLowThresh, cannyHighThresh);

		Mat img_result;
		matchTemplate(img_src_edge, img_temp_edge, img_result, CV_TM_CCORR_NORMED);
		//normalize(img_result, img_result, 0, 1, NORM_MINMAX, -1, Mat());
		minMaxLoc(img_result, &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i], Mat());
		//Mat img_src_cropped = img_src(Rect(maxLoc[i].x, maxLoc[i].y, templates[i].cols, templates[i].rows));
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

		// Display template name
		//cout << template_names[bestIndex] << endl;

		return bestIndex;
	}
	else
		return -1;
}



Mat drawTextPolygon(Mat img, Mat I, vector<Point> &warpedPoints, int matchIndex)
{
	if(!warpedPoints.empty())
	{
		vector<Point> origPoints(warpedPoints.size());

		for(unsigned int i=0; i<warpedPoints.size(); i++)
		{
			int data[] = {warpedPoints[i].x, warpedPoints[i].y, 1};
			Mat V(3, 1, CV_32S, data);
			V.convertTo(V, CV_64F);
			Mat prod = I * V;

			if(prod.at<double>(2) != 0)
			{
				origPoints[i].x = ( prod.at<double>(0) / prod.at<double>(2) );
				origPoints[i].y = ( prod.at<double>(1) / prod.at<double>(2) );
			}
			else
			{
				origPoints[i].x = 0;
				origPoints[i].y = 0;
			}		
		}

		// Draw Lines. Points are in order: TopLeft, TopRight, BottomRight, BottomLeft
		line(img, origPoints[0], origPoints[1], Scalar(0, 255, 0), 2, 8);
		line(img, origPoints[1], origPoints[2], Scalar(0, 255, 0), 2, 8);
		line(img, origPoints[2], origPoints[3], Scalar(0, 255, 0), 2, 8);
		line(img, origPoints[3], origPoints[0], Scalar(0, 255, 0), 2, 8);


		// Get template name
		if(matchIndex != -1)
		{
			string tempName;
			const char *tmp = template_names[matchIndex].c_str();

			for(unsigned int i=13; i<template_names[matchIndex].size(); i++)
			{
				if(tmp[i] == '.' || isdigit(tmp[i]))
					break;
				else if(tmp[i] == '_')
					tempName.push_back(' ');
				else
					tempName.push_back(tmp[i]);
			}

			putText(img, 
				tempName, 
				cvPoint(origPoints[0].x, origPoints[0].y - 10),
    			FONT_HERSHEY_COMPLEX_SMALL, 
    			1.5, 
    			cvScalar(0,255,255), 
    			1, 
    			CV_AA);

		}

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
}
*/
