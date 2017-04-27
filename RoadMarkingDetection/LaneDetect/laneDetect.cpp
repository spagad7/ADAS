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
#include<opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

Mat Canny_Edge(Mat img,float low,float high)
{
	Mat gblur,can_edg;
	GaussianBlur(img,gblur,Size(3,3),1,0);
	Mat sob_op;
	//Sobel(gblur,sob_op,CV_8U,0,0);
	//threshold(sob_op,can_edg,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
	Canny(gblur,can_edg,low,high);
	return can_edg;
}

Mat reg_of_interest(Mat img)
{
	int no_rows=img.rows;
	int no_cols=img.cols;
	//int vertices[4][2]={{100,no_rows},{200,150},{300,165},{no_cols,no_rows}};	

	Point vertex[1][4];
	vertex[0][0]=Point(274, 1030);
	vertex[0][1]=Point(843,663);
	vertex[0][2]=Point(1032,663);
	vertex[0][3]=Point(1648,1030);
	const Point* vertices[1] = { vertex[0] };
	int npt[] = { 4 };

/*
	line(frame, Point(843, 663), Point(1032, 663), Scalar(0, 0, 255), 2, 8);
		line(frame, Point(1032, 663), Point(1648, 1030), Scalar(0, 0, 255), 2, 8);
		line(frame, Point(1648, 1030), Point(274, 1030), Scalar(0, 0, 255), 2, 8);
		line(frame, Point(274, 1030), Point(843, 663), Scalar(0, 0, 255), 2, 8);
*/

	// Doing it for grayscale image now

	int ig_mc=255;
	
	Mat mask=img*0;
	fillPoly( mask, vertices, npt , 1, Scalar( 255 ), 8 );
	
	Mat masked_img;
	
	bitwise_and(img,mask,masked_img);
	return masked_img;
	
}

Mat hough_lines(Mat img,Mat c_img, int rho, float theta, int thresh,int min_line_length, int max_line_gap)
{
	vector<Vec4i> lines;
	HoughLinesP(img,lines,rho,theta,thresh,min_line_length,max_line_gap);
	Mat d_img=c_img*0;
	float x1,x2,y1,y2,slope,ml,mr;
	int xrq,yrq,xlq,ylq,xrp,yrp=0,xlp,ylp=0,ymax;
	int xlt,ylt,xrt,yrt,ytemp;

	for( size_t i = 0; i < lines.size(); i++ )
	{
		x1=lines[i][0];
		x2=lines[i][2];
		y1=lines[i][1];
		y2=lines[i][3];

		
		slope=(y2-y1)/(x2-x1);
		if(slope < 0)
			ylt=min(y1,y2);
		else
			yrt=min(y1,y2);
	}

	for( size_t i = 0; i < lines.size(); i++ )
	{
		x1=lines[i][0];
		x2=lines[i][2];
		y1=lines[i][1];
		y2=lines[i][3];

		
		slope=(y2-y1)/(x2-x1);
		if(slope < 0)
		{
			ytemp=min(y1,y2);
			if(ylt <= ytemp)
			{
				ylt=ytemp;
				xlt=max(x1,x2);
			}
		}
		else
		{
			ytemp=min(y1,y2);
			if(yrt <= ytemp)
			{
				yrt=ytemp;
				xrt=min(x1,x2);
			}
		}
	}
	
	for( size_t i = 0; i < lines.size(); i++ )
	{
		
	    x1=lines[i][0];
		x2=lines[i][2];
		y1=lines[i][1];
		y2=lines[i][3];

		
		slope=(y2-y1)/(x2-x1);

		if (slope>0.3 || slope<-0.3)
		{
			if(slope<0 )
			{
				if(max(y1,y2)>ylp)
				{
					ylp=max(y1,y2);
					xlp=min(x1,x2);
					ml=slope;
				}
			}
			else
			{
				if(max(y1,y2)>yrp)
				{
					yrp=max(y1,y2);
					xrp=max(x1,x2);
					mr=slope;
				}
			}					
				
		//line( d_img, Point(lines[i][0], lines[i][1]),Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
		line(d_img,Point(x1,y1),Point(x2,y2),Scalar(255, 255, 0), 10, 8 );
		
    	}
    } 

	ymax=d_img.rows;
	ylq=ymax;
	yrq=ymax;
	xlq=xlp+(1/ml)*(ylq-ylp);
	xrq=xrp+(1/mr)*(yrq-yrp);

	int ynew=0.7*ylq,xln,xrn;
	
	xln=xlq+(1/ml)*(ynew-ylq);
	xrn=xrq+(1/mr)*(ynew-yrq);

	Point vertex[1][4];
	vertex[0][0]=Point(xln,ynew);
	vertex[0][1]=Point(xrn,ynew);
	vertex[0][2]=Point(xrq,yrq);
	vertex[0][3]=Point(xlq,ylq);
	const Point* vertices[1] = { vertex[0] };
	int npt[] = { 4 };

	//line(d_img,Point(xlp,ylp),Point(xlq,ylq),Scalar(0,0,255), 9, 8 );
	//line(d_img,Point(xrp,yrp),Point(xrq,yrq),Scalar(0,0,255), 9, 8 );

	//fillPoly( d_img, vertices, npt , 1, Scalar( 255,0,255 ), 8 );
 
	return d_img;
}
Mat weighted_img(Mat img,Mat init_img)
{
	float alph=0.8,bet=1,gam=0;
	Mat w_img;
	addWeighted(init_img,alph,img,bet,gam,w_img);
	return w_img;
}


int main(int argc, char* argv[])
{
	VideoCapture cap("../Test.avi");
	//VideoCapture cap("/home/raj/Github/Computer-Vision-5561-Project/lane_detect_1_video/highway.mkv");

	if ( !cap.isOpened() )  // if not success, exit program
    	{
        	 cout << "Cannot open the video file" << endl;
        	 return -1;
    	}

	namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

        while(1)
      	{	
		    Mat frame;
		    cap >> frame; // get a new frame from camera  
		    

		    Mat img;
		    img=frame;
		
		    Mat gray;	
			cvtColor(img,gray,CV_BGR2GRAY);

			// Canny Edge
	
			Mat can_edg=Canny_Edge(gray,100,150);

			// Finding region of interest
	
	
			Mat img_roi=reg_of_interest(can_edg); 
	
			// Hough lines
			int rho=3;
			float theta=3.1416/180;
			int thresh=85;
			int min_line_length=40;
			int max_line_gap=100;

			Mat hough_img=hough_lines(img_roi,img,rho,theta,thresh,min_line_length,max_line_gap);

			Mat w_img=weighted_img(hough_img,img);


			  
		    imshow("Video", w_img);
		    if(waitKey(30) == 27) 
		    break;
		
			

	}
}

