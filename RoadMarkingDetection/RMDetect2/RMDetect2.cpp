#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat findRoadMarking(Mat image)
{
	Mat smoothImage, imageBlurHSV, mask;
	GaussianBlur(image, smoothImage, Size(7,7), 0, 0);
	cvtColor(smoothImage, imageBlurHSV, COLOR_BGR2HSV);
	
	inRange(imageBlurHSV, Scalar(0, 0, 215), Scalar(100,255, 255), mask);
	
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	
	return mask;
}

int main() {

	Mat image, imageroi, imagerm, imagerm2;
	string filename;
	
	// Generate filename
	filename = "../roadmark3.jpg";
	
	// Read image	
	image = imread( filename, CV_LOAD_IMAGE_COLOR );
	
	// Convert color from BGR to RGB
	cvtColor(image, image, COLOR_BGR2RGB);
	
	// Set Region of interest
	//imageroi = image(Rect(200,300,400,100));
	
	imagerm = findRoadMarking(image);
	
	//cvtColor(imagerm, imagerm2, COLOR_RGB2BGR);
	
	imshow("RoadMarking Detection", imagerm);
	waitKey(0);
	
	return 0;
}


