#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <math.h>


using namespace cv;
using namespace std;

// Function Declarations
Mat calculateHomography(void);
void findLaneMarker(Mat img);
void findMatch(Mat, vector<Mat>);
vector<Mat> loadTemplates(void);
double calcSSD(Mat, Mat);


int main() 
{

	string filename;
	vector<Mat> templates;
	int numImages = 1433;

	// Load templates
	templates = loadTemplates();

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

		// Calculate Homography
		Mat H = calculateHomography();
		
		// Create black destination image
		Mat img_warp(600, 800, CV_8UC3, Scalar(0,0,0));
		//Mat img_warp(900, 1200, CV_8UC3, Scalar(0,0,0));

		// Warp image
		warpPerspective(img_src, img_warp, H, img_warp.size());

		// Detect lane markers
		findLaneMarker(img_warp);

		// Detect road markings
		findMatch(img_warp, templates);

	}
	
	return 0;
	
}



// Function to calculate Homography
Mat calculateHomography(void)
{
	vector<Point2f> pts_src, pts_dst;

	// Corners in source image
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



// Function to load templates
vector<Mat> loadTemplates(void)
{
	int numTemplates = 3;
	string filename;
	vector<Mat> templateVector;
	

	for(int i=0; i<numTemplates; i++)
	{
		filename = "../Templates/template" + std::to_string(i) + ".png";
		Mat img_temp = imread(filename, CV_LOAD_IMAGE_COLOR);
		templateVector.push_back(img_temp.clone());
	}

	return(templateVector);
}


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




// Function to detect road markings using template matching
void findMatch(Mat img_src, vector<Mat> templates)
{
	Mat img_src_gray, img_src_edge, img_temp_gray, img_temp_edge;
	unsigned int numTemplates = templates.size();
	vector<double> maxVal(numTemplates), minVal(numTemplates), ssd(numTemplates);
	vector<Point> minLoc(numTemplates), maxLoc(numTemplates);
	//int rows, cols;
	//double scale = 1.5;

	//rows = round(104);
	//cols = round(306);

	//resize(img_temp, img_temp, Size(cols, rows), 0,0, INTER_LINEAR);


	// Get edge image of img_dst and img_temp
	cvtColor(img_src, img_src_gray, CV_BGR2GRAY);
	//Canny(img_src_gray, img_src_edge, 50, 200);
	
	#pragma omp parallel for
	for(unsigned int i=0; i<templates.size(); i++)
	{
		cvtColor(templates[i], templates[i], CV_BGR2GRAY);
		//Canny(templates[i], templates[i], 50, 200);
		Mat img_result;
		matchTemplate(img_src_gray, templates[i], img_result, CV_TM_CCORR_NORMED);
		//normalize(img_result, img_result, 0, 1, NORM_MINMAX, -1, Mat());
		minMaxLoc(img_result, &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i], Mat());
		Mat img_src_cropped = img_src(Rect(maxLoc[i].x, maxLoc[i].y, templates[i].cols, templates[i].rows));
		ssd[i] = calcSSD(img_src_cropped, templates[i]);
		//cout << "SSD: " << ssd[i] << endl;
		//cout << "maxVal: " << maxVal[i] << endl;
	}


	int bestIndex = distance(ssd.begin(), min_element(ssd.begin(), ssd.end()));
	//int bestIndex = distance(maxVal.begin(), max_element(maxVal.begin(), maxVal.end()));
	//cout << bestIndex << endl;

	rectangle(img_src, maxLoc[bestIndex], 
		Point(maxLoc[bestIndex].x + templates[bestIndex].cols, maxLoc[bestIndex].y + templates[bestIndex].rows), 
		Scalar(0, 255, 0), 2, 8, 0);

	// Display images
	imshow("Source Image", img_src);
	waitKey(1);

}


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