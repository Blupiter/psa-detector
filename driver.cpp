// driver.cpp

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// parameters

const string IMAGE_PATH = "test1.jpg";

// Canny edge detection
const int CANNY_LOW_THRESHOLD = 100;
const int CANNY_HIGH_THRESHOLD = 175;
const int GAUSSIAN_BLUR_SIZE = 5;

// Hough line detection
const int HOUGH_RHO = 1;
const double HOUGH_THETA = CV_PI / 180;
const int HOUGH_THRESHOLD = 35;
const int HOUGH_MIN_LINE_LENGTH = 35;
const int HOUGH_MAX_LINE_GAP = 20;

// Vertical line filtering
const double MIN_VERTICAL_ANGLE = 85.0;
const double MAX_VERTICAL_ANGLE = 95.0;
const double MIN_VERTICAL_LINE_LENGTH_FACTOR = 0.42;

// X-position merging
const int MERGE_DISTANCE = 12;
const double MIN_DIVIDER_SPACING_FACTOR = 0.06;

// Drawing
const Scalar RED = Scalar(0, 0, 255);
const int LINE_THICKNESS = 2;

typedef struct ParkingSpot {

} ParkingSpot;

Mat getEdges(const Mat& img) {
    Mat gray, blurred, edges;

    cvtColor(img, gray, COLOR_BGR2GRAY);

    GaussianBlur(
        gray,
        blurred,
        Size(GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE),
        0
    );

    Canny(
        blurred,
        edges,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD
    );

    return edges;
}

vector<Vec4i> getHoughLines(const Mat& edges) {
    vector<Vec4i> lines;

    HoughLinesP(
        edges,
        lines,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        HOUGH_MIN_LINE_LENGTH,
        HOUGH_MAX_LINE_GAP
    );

    return lines;
}

bool isVerticalLine(const Vec4i& lineSegment, int imageHeight) {
    int x1 = lineSegment[0];
    int y1 = lineSegment[1];
    int x2 = lineSegment[2];
    int y2 = lineSegment[3];

    double dx = x2 - x1;
    double dy = y2 - y1;

    double length = sqrt(dx * dx + dy * dy);
    double angle = atan2(dy, dx) * 180.0 / CV_PI;
    angle = fabs(angle);

    bool angleIsVertical =
        angle > MIN_VERTICAL_ANGLE &&
        angle < MAX_VERTICAL_ANGLE;

    bool lineIsLongEnough =
        length > (imageHeight * MIN_VERTICAL_LINE_LENGTH_FACTOR);

    return angleIsVertical && lineIsLongEnough;
}

vector<int> getVerticalXPositions(
    const vector<Vec4i>& lines,
    Mat& verticalLineDisplay
) {
    vector<int> xPositions;

    for (const Vec4i& l : lines) {
        if (isVerticalLine(l, verticalLineDisplay.rows)) {
            int x1 = l[0];
            int y1 = l[1];
            int x2 = l[2];
            int y2 = l[3];

            line(
                verticalLineDisplay,
                Point(x1, y1),
                Point(x2, y2),
                RED,
                LINE_THICKNESS
            );

            int xAvg = (x1 + x2) / 2;

            cout << "vertical line x = " << xAvg << endl;

            xPositions.push_back(xAvg);
        }
    }

    return xPositions;
}

vector<int> mergeXPositions(vector<int> xPositions) {
    sort(xPositions.begin(), xPositions.end());

    vector<int> mergedX;

    for (int x : xPositions) {
        if (mergedX.empty() || abs(x - mergedX.back()) > MERGE_DISTANCE) {
            mergedX.push_back(x);
        }
        else {
            mergedX.back() = (mergedX.back() + x) / 2;
        }
    }

    return mergedX;
}

vector<int> filterCloseXPositions(const vector<int>& mergedX, int imageWidth) {
    vector<int> filteredX;

    for (int x : mergedX) {
        if (filteredX.empty() || abs(x - filteredX.back()) >= (MIN_DIVIDER_SPACING_FACTOR * imageWidth)) {
            filteredX.push_back(x);
        }
    }

    return filteredX;
}

Mat drawMergedVerticalLines(const Mat& img, const vector<int>& filteredX) {
    Mat mergedLineDisplay = img.clone();

    cout << endl << "Final vertical divider x positions:" << endl;

    for (int x : filteredX) {
        cout << "divider x = " << x << endl;

        line(
            mergedLineDisplay,
            Point(x, 0),
            Point(x, img.rows),
            RED,
            LINE_THICKNESS
        );
    }

    return mergedLineDisplay;
}

vector<ParkingSpot> getParkingSpotLocations(Mat img) {
    return {};
}

int main() {
    Mat img = imread(IMAGE_PATH);

    if (img.empty()) {
        cout << "Could not load image: " << IMAGE_PATH << endl;
        return -1;
    }

    Mat edges = getEdges(img);

    imshow("Original", img);
    imshow("Edges", edges);
    waitKey(0);

    vector<Vec4i> lines = getHoughLines(edges);

    Mat verticalLineDisplay = img.clone();

    vector<int> xPositions =
        getVerticalXPositions(lines, verticalLineDisplay);

    vector<int> mergedX =
        mergeXPositions(xPositions);

    vector<int> filteredX =
        filterCloseXPositions(mergedX, img.cols);

    Mat mergedLineDisplay =
        drawMergedVerticalLines(img, filteredX);

    imshow("Original", img);
    imshow("Edges", edges);
    imshow("Vertical Lines", verticalLineDisplay);
    imshow("Merged Vertical Lines", mergedLineDisplay);

    waitKey(0);

    return 0;
}