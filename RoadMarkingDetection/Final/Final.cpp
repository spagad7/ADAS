


#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/objdetect/objdetect.hpp> 
#include <typeinfo>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>
#include <errno.h>
#include <dirent.h>
#include <math.h>

using namespace cv;
using namespace std;

// Function Declarations for Roadmarking Detection
Mat backgroundsub(Mat myblur);  // Function for background subtraction 
Mat closingop(Mat sub); // Function for closing operation
Mat calculateHomography(void);
void loadImages(string, vector<Mat> &, bool);
void getFilesInDir(string, vector<string> &, bool);
int findMatch(Mat , vector<Mat>, vector<Point> &);
Mat drawTextPolygon(Mat , Mat, vector<Point> &, int);

// Global Variables
map<int, string> template_names; 


// Main Function
int main(int argc, char* argv[])
{
	// Init
    VideoCapture cap("../Test.avi"); // open the video file for reading

    if ( !cap.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the video file" << endl;
         return -1;
    }

    int c = 0;
    
    //Declaring all the Matrix variables
    Mat prevImg;
   
    // Generating the ROI for main frame
    //Rect Rec(200, 200, 600, 300);
    Rect Rec(400, 400, 1000, 550);

    // Parameters for detection of vehicles     
    vector<vector<Point> > contours;
    double area, ar;
    int count=0;
    vector<vector<Point> > contours_poly(150);
    vector<Rect> boundRect(150);
    vector<Rect> detectRect(150);
    vector<Rect> trackRect(150);
    Rect formed_rect, draw;
    
    // Parameters for Lucas kanade optical flow tracking 
    int win_size = 3;
    int maxCorners = 100;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 5;
    double kt = 0.04;
    int detecting = 0;
    int length = 15;
    //Focal length
    double focal = 568.996;
    double distance;
    
    // Variables for Road Marking Detection
    vector<Mat> templates;
	vector<Mat> images;
	vector<Point> matchPoints;
	int imgCount = 0;
	Mat img_final;

	// Load templates
	string tempDirName = "../Templates";
	loadImages(tempDirName, templates, true);
	
	// Calculate Homography
	Mat H = calculateHomography();

	// Calculate Inverse Homography
	Mat I = H.inv();

    
    while(1)
    {
        
	Mat frame,C, gframe,myblur, kernel,sobely,bin, otsu, sub, closer, vehicle_ROI; // Declaring the Matrices 
        bool bSuccess = cap.read(frame); // read a new frame from video
        
        //cout<<"Frame size"<<frame.size()<<endl;
        //Creating the Kernel
        
        Mat k = (Mat_<int>(1,10) << 255,255,255,255,255,255,255,255,255,255);
        Mat kernel_new = (Mat_<int>(3,3) << 1,1,1,1,1,1,1,1,1);

		//Creating Inverse Camera matrix
		Mat K_inv = (Mat_<double>(3,3) << 0.00175, 0, -1.1304, 0, 0.00175, -0.84, 0, 0, 1);
        
    	// Create black destination image
		Mat img_warp(300, 200, CV_8UC3, Scalar(0,0,0));

		// Warp image
		warpPerspective(frame, img_warp, H, img_warp.size());
		
		// Detect road markings
		int matchIndex = findMatch(img_warp, templates, matchPoints);
    
    
    // Drawing the ROI on the original frame 
    //rectangle(frame, Rec, Scalar(0,0,255), 0, 8, 0);
    Mat image_roi = frame(Rec);

    // Converting RGB to GRAY
    cvtColor(image_roi, gframe, cv::COLOR_RGB2GRAY );

    waitKey(1);
		
    // Blurring the image
    
    blur(gframe, myblur, Size(3,3));
    
    
    
    

//-------------Vehicle Detection Begins------------------------------------


if(detecting%length == 0)
{

   //Background Subtraction 
    sub = backgroundsub(myblur);

    //Closing operation to enhance the shadows 
    
    closer = closingop(sub);
    

   
    //Finding the contours for detection 

    findContours(closer,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    
         
    
        for(size_t i = 0; i < contours.size(); i++ )
		{
			approxPolyDP( Mat(contours[i]), contours_poly[i], 10, true );   //Fitting the polygon around detected contours 
			area = contourArea(contours[i], false);
			detectRect[i].height = 0;
            		detectRect[i].y = 0;
			detectRect[i].x = 0;  
			detectRect[i].width = 0;
            		if(area > 150 && area < 500)
            		{
    				boundRect[i] = boundingRect( Mat(contours_poly[i]) );
				
				vehicle_ROI = myblur(boundRect[i]);
            			ar = vehicle_ROI.cols/vehicle_ROI.rows;   // Creating ratio of width to height
            			if(ar >= 0.3 && ar <= 10)
            			{
            				
            				detectRect[i].height = boundRect[i].height + 50; // Adding extra height for the box to fit over cars
            				detectRect[i].y = boundRect[i].y - 50;           //Adjusting co-ordinates to fit the boxes 
					detectRect[i].x = boundRect[i].x;  
					detectRect[i].width = boundRect[i].width;
            				
            				//Drawing the rectanglular boxes over te detected vehicles 
            				
            				//rectangle( image_roi, detectRect[i].tl(), detectRect[i].br(), Scalar(255,0,0), 2, 8, 0 ); 
					//rectangle( image_roi, trackRect[i].tl(), trackRect[i].br(), Scalar(0,0,255), 2, 8, 0 );
            				count=count+1;
            			}
            			
                		
            		}
            
		}

}




// Tracking operation begins :

 stringstream ss;
        ss << count;
        string s = ss.str(); 
        
        // Creating key point vectors to store the corner values 
        
        vector<KeyPoint> img_corners;
        
        vector<KeyPoint> prevImg_corners;
        
        bool non_max_suppression = true;
	vector<Point2f> diff(150);
	
for(int j = 0; j < contours.size(); j++)
{
	int valid_rect = 0;
	if(detecting%length == 0)
	{
		trackRect[j].height = 0;
        	trackRect[j].y = 0;
		trackRect[j].x = 0;  
		trackRect[j].width = 0;
	}
	if(detectRect[j].x >= 0 && detectRect[j].y >= 0 && detectRect[j].height > 0 && detectRect[j].width > 0)
	{        
	
	// Carrying out FAST feature detection between 2 consecutive frames 
	
	
	FAST(myblur(detectRect[j]), img_corners, 5, non_max_suppression);
	
	vector<Point2f> points_img_corners, points_prevImg_corners;
	KeyPoint::convert(img_corners,points_img_corners);
	KeyPoint::convert(prevImg_corners,points_prevImg_corners);
	
	
	formed_rect = boundingRect(Mat(points_img_corners) );
	//formed_rect.x = formed_rect.x + detectRect[j].x;
	//formed_rect.y = formed_rect.x + detectRect[j].y;
	//formed_rect.width = formed_rect.width;
	//formed_rect.height = formed_rect.height;
	//formed_rect.tl() = detectRect[j].tl() + formed_rect.tl();
	//rectangle( image_roi, detectRect[j].tl() + formed_rect.tl(), detectRect[j].tl() + formed_rect.br(), Scalar(0,255,255), 2, 8, 0 );
	//cout<<"left bottom of shadow "<<(detectRect[j].x + formed_rect.x)<<endl;
	//cout<<"right bottom of shadow "<<(detectRect[j].x + formed_rect.x + formed_rect.width)<<endl;

	if((detectRect[j].x + formed_rect.x) >= 240 && (detectRect[j].x + formed_rect.x + formed_rect.width) <= 880)	
	{
		distance = 2.5*focal/formed_rect.width;
		//cout<<"Estimated distance "<<distance<<endl;
		ostringstream os;
		os << distance;
		string text = os.str();
		//Printing distance on image	
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontScale = 0.5;
		int thickness = 1;
		int baseline=0;
		Size textSize = getTextSize(text, fontFace,
	                            fontScale, thickness, &baseline);
		baseline += thickness;
		// center the text
		Point textOrg((detectRect[j].x + formed_rect.x - textSize.width),
	              (detectRect[j].y + formed_rect.y + formed_rect.height + textSize.height));
		rectangle(image_roi, textOrg + Point(0, baseline),
	          textOrg + Point(textSize.width, -textSize.height),
	          Scalar(0,255,255));
		// ... and the baseline first
		line(image_roi, textOrg + Point(0, thickness),
	     	textOrg + Point(textSize.width, thickness),
	     	Scalar(0, 0, 255));

		// then put the text itself
		putText(image_roi, text, textOrg, fontFace, fontScale,
	        	Scalar::all(0), thickness, 8);
		
	}	 
   if(norm(points_img_corners)>0)
   {
	
        vector<uchar> features_found;
        features_found.reserve(maxCorners);
        vector<float> feature_errors;
        feature_errors.reserve(maxCorners);
        
        if(c>0)
        {
        
        //Carrying out Lucas Kanade Optical flow method 
        

        calcOpticalFlowPyrLK( myblur(detectRect[j]), prevImg(detectRect[j]), points_img_corners, points_prevImg_corners, features_found, feature_errors ,
                             Size( win_size, win_size ), 3,
                             cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0, kt);
	
	for( int i=0; i < features_found.size(); i++ ){
	    
            Point2f p0( ceil( detectRect[j].x+points_img_corners[i].x ), ceil( detectRect[j].y+points_img_corners[i].y ) );
            Point2f p1( ceil( detectRect[j].x+points_prevImg_corners[i].x ), ceil( detectRect[j].y+points_prevImg_corners[i].y ) );
	    diff[j] = diff[j] + p0 - p1;

	   
            //line( image_roi, p0, p1, CV_RGB(0,255,0), 3 );
	    //circle(image_roi, p0, 1, Scalar(255,255,0), 2, 8, 0);
	    valid_rect++;
        }
	diff[j].x = diff[j].x/valid_rect;
	diff[j].y = diff[j].y/valid_rect;
        
        }
        

        
   }

	}

}

prevImg = myblur;

prevImg_corners = img_corners;

c++;

//Using tracking upon the detected objects 



if(detecting%length == 0)
{	
	//cout<<"Updating trackrect while detecting"<<endl;
	
	for(int j = 0; j < contours.size(); j++)
	{
		
		trackRect[j].x = detectRect[j].x + diff[j].x;
		trackRect[j].y = detectRect[j].y + diff[j].y;
		trackRect[j].width = detectRect[j].width;
		trackRect[j].height = detectRect[j].height;
		draw.x = trackRect[j].x + Rec.x;
		draw.y = trackRect[j].y + Rec.y;
		draw.width = trackRect[j].width;
		draw.height = trackRect[j].height;
		rectangle( frame, draw.tl(), draw.br(), Scalar(255,0,0), 2, 8, 0 );
		rectangle( frame, draw.tl(), draw.br(), Scalar(0,0,255), 2, 8, 0 );
		//cout<<"Vehicle x = "<<(2*trackRect[j].x + trackRect[j].width)/2<<endl;
		//cout<<"Vehicle y = "<<(2*trackRect[j].y + trackRect[j].height)/2<<endl;
		
		// Draw ROI on img
		line(frame, Point(843, 663), Point(1032, 663), Scalar(0, 0, 255), 2, 8);
		line(frame, Point(1032, 663), Point(1648, 1030), Scalar(0, 0, 255), 2, 8);
		line(frame, Point(1648, 1030), Point(274, 1030), Scalar(0, 0, 255), 2, 8);
		line(frame, Point(274, 1030), Point(843, 663), Scalar(0, 0, 255), 2, 8);

		// Draw polygon for matched region
		img_final = drawTextPolygon(frame, I, matchPoints, matchIndex);
		
	}
}
else
{
	for(int j = 0; j < contours.size(); j++)
	{	
		
		trackRect[j].x = trackRect[j].x + diff[j].x;
		trackRect[j].y = trackRect[j].y + diff[j].y;
		trackRect[j].width = trackRect[j].width;
		trackRect[j].height = trackRect[j].height;
		draw.x = trackRect[j].x + Rec.x;
		draw.y = trackRect[j].y + Rec.y;
		draw.width = trackRect[j].width;
		draw.height = trackRect[j].height;
		rectangle( frame, draw.tl(), draw.br(), Scalar(255,0,0), 2, 8, 0 );
		rectangle( frame, draw.tl(), draw.br(), Scalar(0,0,255), 2, 8, 0 );
		//cout<<"Vehicle x = "<<(2*trackRect[j].x + trackRect[j].width)/2<<endl;
		//cout<<"Vehicle y = "<<(2*trackRect[j].y + trackRect[j].height)/2<<endl;
		
		// Draw ROI on img
		line(frame, Point(843, 663), Point(1032, 663), Scalar(0, 0, 255), 2, 8);
		line(frame, Point(1032, 663), Point(1648, 1030), Scalar(0, 0, 255), 2, 8);
		line(frame, Point(1648, 1030), Point(274, 1030), Scalar(0, 0, 255), 2, 8);
		line(frame, Point(274, 1030), Point(843, 663), Scalar(0, 0, 255), 2, 8);

		// Draw polygon for matched region
		img_final = drawTextPolygon(frame, I, matchPoints, matchIndex);
		
	}
	
}






	


//Displaying the video output 


        if (!bSuccess) //if not success, break loop
     {
                        cout << "Cannot read the frame from video file" << endl;
                       break;
        }
	
        //imshow("Video Output",frame); //show the frame in "Video Output" window
        imshow("Video Output",img_final);
	
	
        if(waitKey(1) == 27) //bwait for 'esc' key press for 10 ms. If 'esc' key is pressed, break loop
       {
                cout << "Parking into the garage " << endl; 
                break;
       }
 detecting++;
 	//waitKey(0);
 	
 		matchPoints.clear();
    }//While ends
    

    return 0;

}





//Background Subtraction function 



Mat backgroundsub(Mat myblur)
{
    Mat sobely, otsu, bin, sub;
    
    
    // Sobel edge detector 
	Sobel(myblur, sobely, CV_8U, 0,1);
	
    // Otsu's thresholding 
	threshold(sobely, otsu, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);


    //Creating 5 samples of the road area 

	threshold(myblur, bin ,151,255, CV_THRESH_BINARY);



    // Subtracting the images 

	
	sub = otsu - bin;
	
    	return sub;


}




    //Closing function 
    
    
    
    Mat closingop(Mat sub)
{
    Mat closer, more_closer;
    //Parameters for closing operation 
    int morph_size = 7;

    Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
	
	//Carrying out the closing operation 

    morphologyEx(sub, closer, MORPH_CLOSE, element);
   

    return closer;

}



///////////////////////////
// Roadmarking Detection //
///////////////////////////

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
	pts_src.push_back(Point2f(843, 663));
	pts_src.push_back(Point2f(1032, 663));
	pts_src.push_back(Point2f(1648, 1030));
	pts_src.push_back(Point2f(274, 1030));

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
	double matchThresh = 0.50;

	//resize(img_temp, img_temp, Size(cols, rows), 0,0, INTER_LINEAR);

	// Get edge image of img_dst and img_temp
	cvtColor(img_src, img_src_gray, CV_BGR2GRAY);
	//threshold(img_src_gray, img_src_thresh, 150, 255, THRESH_BINARY);
	blur(img_src_gray, img_src_edge, Size(5,5) );
	Canny(img_src_edge, img_src_edge, cannyLowThresh, cannyHighThresh);
	
	
	//#pragma omp parallel for
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
    			1.0, 
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







