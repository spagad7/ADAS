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
	
	inRange(imageBlurHSV, Scalar(0, 0, 215), Scalar(100, 255, 255), mask);
	
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
	
	return mask;
}

int main() {

	for(int i=1; i<=1443; i++)
	{
		Mat image, imageroi, imagerm, imagerm2;
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
			
		// Read image	
		image = imread( filename, CV_LOAD_IMAGE_COLOR );
		
		// Convert color from BGR to RGB
		cvtColor(image, image, COLOR_BGR2RGB);
		
		// Set Region of interest
		imageroi = image(Rect(0,250,800,300));
		
		imagerm = findRoadMarking(imageroi);
		
		//cvtColor(imagerm, imagerm2, COLOR_RGB2BGR);
		
		imshow("RoadMarking Detection", imagerm);
		waitKey(25);
	}
	return 0;
}


