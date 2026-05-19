// driver.cpp

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

typedef struct ParkingSpot {

} ParkingSpot;

vector<ParkingSpot> getParkingSpotLocations(Mat img) {
	return {};
}

int main() {
	Mat img = imread("test3.jpg");
	Mat gray, blurred, edges;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blurred, Size(5, 5), 0);
	Canny(blurred, edges, 50, 150);
	imshow("Original", img);
	imshow("Edges", edges);
	waitKey(0);
	
	return 0;
}