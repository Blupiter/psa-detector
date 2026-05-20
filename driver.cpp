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

// --------------------------------------------------
// Parameters
// --------------------------------------------------

const string IMAGE_PATH = "test1.jpg";

// Canny edge detection
const int CANNY_LOW_THRESHOLD = 100;
const int CANNY_HIGH_THRESHOLD = 175;
const int GAUSSIAN_BLUR_SIZE = 5;

// Shared Hough settings
const int HOUGH_RHO = 1;
const double HOUGH_THETA = CV_PI / 180;

// Vertical Hough settings
const int VERTICAL_HOUGH_THRESHOLD = 15;
const int VERTICAL_HOUGH_MIN_LINE_LENGTH = 35;
const int VERTICAL_HOUGH_MAX_LINE_GAP = 50;

// Horizontal Hough settings
// These can be permissive because final filtering is based on connected coverage.
const int HORIZONTAL_HOUGH_THRESHOLD = 20;
const int HORIZONTAL_HOUGH_MIN_LINE_LENGTH = 20;
const int HORIZONTAL_HOUGH_MAX_LINE_GAP = 15;

// Vertical line filtering
const double MIN_VERTICAL_ANGLE = 85.0;
const double MAX_VERTICAL_ANGLE = 95.0;
const double MIN_VERTICAL_LINE_LENGTH_FACTOR = 0.42;

// Horizontal line filtering
const double MAX_HORIZONTAL_ANGLE = 3.0;
const double MIN_HORIZONTAL_SEGMENT_LENGTH_FACTOR = 0.05;

// Horizontal connected coverage filtering
const int HORIZONTAL_INTERVAL_GAP = 25;
const double MIN_HORIZONTAL_CONNECTED_COVERAGE_FACTOR = 0.65;

// Position merging
const int MERGE_DISTANCE = 12;
const int HORIZONTAL_MERGE_DISTANCE = 12;
const double MIN_DIVIDER_SPACING_FACTOR = 0.06;

// Drawing
const Scalar RED = Scalar(0, 0, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar GREEN = Scalar(0, 255, 0);
const int LINE_THICKNESS = 2;

// --------------------------------------------------
// Structs
// --------------------------------------------------

typedef struct ParkingSpot {

} ParkingSpot;

typedef struct HoughParams {
    int threshold;
    int minLineLength;
    int maxLineGap;
} HoughParams;

const HoughParams VERTICAL_HOUGH_PARAMS = {
    VERTICAL_HOUGH_THRESHOLD,
    VERTICAL_HOUGH_MIN_LINE_LENGTH,
    VERTICAL_HOUGH_MAX_LINE_GAP
};

const HoughParams HORIZONTAL_HOUGH_PARAMS = {
    HORIZONTAL_HOUGH_THRESHOLD,
    HORIZONTAL_HOUGH_MIN_LINE_LENGTH,
    HORIZONTAL_HOUGH_MAX_LINE_GAP
};

typedef struct VerticalLineEvidence {
    int x;
    int topY;
    int bottomY;
} VerticalLineEvidence;

typedef struct MergedDivider {
    int x;
    int topY;
    int bottomY;
} MergedDivider;

typedef struct HorizontalLineEvidence {
    int y;
    int leftX;
    int rightX;
} HorizontalLineEvidence;

typedef struct Interval {
    int leftX;
    int rightX;
} Interval;

typedef struct MergedHorizontalDivider {
    int y;
    int leftX;
    int rightX;
    int connectedCoverage;
} MergedHorizontalDivider;

// --------------------------------------------------
// Utility functions
// --------------------------------------------------

int medianValue(vector<int> values) {
    if (values.empty()) {
        return 0;
    }

    sort(values.begin(), values.end());

    int middle = static_cast<int>(values.size()) / 2;

    if (values.size() % 2 == 1) {
        return values[middle];
    }

    return (values[middle - 1] + values[middle]) / 2;
}

double getLineLength(const Vec4i& lineSegment) {
    int x1 = lineSegment[0];
    int y1 = lineSegment[1];
    int x2 = lineSegment[2];
    int y2 = lineSegment[3];

    double dx = x2 - x1;
    double dy = y2 - y1;

    return sqrt(dx * dx + dy * dy);
}

double getLineAngleDegrees(const Vec4i& lineSegment) {
    int x1 = lineSegment[0];
    int y1 = lineSegment[1];
    int x2 = lineSegment[2];
    int y2 = lineSegment[3];

    double dx = x2 - x1;
    double dy = y2 - y1;

    double angle = atan2(dy, dx) * 180.0 / CV_PI;

    return fabs(angle);
}

// --------------------------------------------------
// Image preprocessing
// --------------------------------------------------

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

vector<Vec4i> getHoughLines(const Mat& edges, HoughParams params) {
    vector<Vec4i> lines;

    HoughLinesP(
        edges,
        lines,
        HOUGH_RHO,
        HOUGH_THETA,
        params.threshold,
        params.minLineLength,
        params.maxLineGap
    );

    return lines;
}

// --------------------------------------------------
// Vertical divider detection
// --------------------------------------------------

bool isVerticalLine(const Vec4i& lineSegment, int imageHeight) {
    double length = getLineLength(lineSegment);
    double angle = getLineAngleDegrees(lineSegment);

    bool angleIsVertical =
        angle > MIN_VERTICAL_ANGLE &&
        angle < MAX_VERTICAL_ANGLE;

    bool lineIsLongEnough =
        length > (imageHeight * MIN_VERTICAL_LINE_LENGTH_FACTOR);

    return angleIsVertical && lineIsLongEnough;
}

vector<VerticalLineEvidence> getVerticalLineEvidence(
    const vector<Vec4i>& lines,
    Mat& verticalLineDisplay
) {
    vector<VerticalLineEvidence> evidenceList;

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
            int topY = min(y1, y2);
            int bottomY = max(y1, y2);

            VerticalLineEvidence evidence;
            evidence.x = xAvg;
            evidence.topY = topY;
            evidence.bottomY = bottomY;

            evidenceList.push_back(evidence);
        }
    }

    return evidenceList;
}

vector<MergedDivider> mergeVerticalEvidence(vector<VerticalLineEvidence> evidenceList) {
    sort(
        evidenceList.begin(),
        evidenceList.end(),
        [](const VerticalLineEvidence& a, const VerticalLineEvidence& b) {
            return a.x < b.x;
        }
    );

    vector<MergedDivider> mergedDividers;

    vector<int> currentXs;
    vector<int> currentTopYs;
    vector<int> currentBottomYs;

    for (const VerticalLineEvidence& evidence : evidenceList) {
        if (currentXs.empty()) {
            currentXs.push_back(evidence.x);
            currentTopYs.push_back(evidence.topY);
            currentBottomYs.push_back(evidence.bottomY);
        }
        else {
            int currentMedianX = medianValue(currentXs);

            if (abs(evidence.x - currentMedianX) <= MERGE_DISTANCE) {
                currentXs.push_back(evidence.x);
                currentTopYs.push_back(evidence.topY);
                currentBottomYs.push_back(evidence.bottomY);
            }
            else {
                MergedDivider divider;

                divider.x = medianValue(currentXs);
                divider.topY = *min_element(currentTopYs.begin(), currentTopYs.end());
                divider.bottomY = *max_element(currentBottomYs.begin(), currentBottomYs.end());

                mergedDividers.push_back(divider);

                currentXs.clear();
                currentTopYs.clear();
                currentBottomYs.clear();

                currentXs.push_back(evidence.x);
                currentTopYs.push_back(evidence.topY);
                currentBottomYs.push_back(evidence.bottomY);
            }
        }
    }

    if (!currentXs.empty()) {
        MergedDivider divider;

        divider.x = medianValue(currentXs);
        divider.topY = *min_element(currentTopYs.begin(), currentTopYs.end());
        divider.bottomY = *max_element(currentBottomYs.begin(), currentBottomYs.end());

        mergedDividers.push_back(divider);
    }

    return mergedDividers;
}

vector<MergedDivider> filterCloseDividers(
    const vector<MergedDivider>& mergedDividers,
    int imageWidth
) {
    vector<MergedDivider> filteredDividers;

    int minDividerSpacing =
        static_cast<int>(MIN_DIVIDER_SPACING_FACTOR * imageWidth);

    for (const MergedDivider& divider : mergedDividers) {
        if (filteredDividers.empty()) {
            filteredDividers.push_back(divider);
        }
        else {
            MergedDivider& lastDivider = filteredDividers.back();

            int distance = abs(divider.x - lastDivider.x);

            if (distance >= minDividerSpacing) {
                filteredDividers.push_back(divider);
            }
            else {
                int currentHeight = divider.bottomY - divider.topY;
                int lastHeight = lastDivider.bottomY - lastDivider.topY;

                if (currentHeight > lastHeight) {
                    lastDivider = divider;
                }
            }
        }
    }

    return filteredDividers;
}

Mat drawMergedVerticalLines(
    const Mat& img,
    const vector<MergedDivider>& filteredDividers
) {
    Mat display = img.clone();

    cout << endl << "Final vertical dividers:" << endl;

    for (const MergedDivider& divider : filteredDividers) {
        cout << "vertical divider x = " << divider.x
            << ", topY = " << divider.topY
            << ", bottomY = " << divider.bottomY
            << endl;

        line(
            display,
            Point(divider.x, divider.topY),
            Point(divider.x, divider.bottomY),
            RED,
            LINE_THICKNESS
        );

        circle(display, Point(divider.x, divider.topY), 4, BLUE, FILLED);
        circle(display, Point(divider.x, divider.bottomY), 4, BLUE, FILLED);
    }

    return display;
}

// --------------------------------------------------
// Horizontal divider detection
// --------------------------------------------------

bool isHorizontalLine(const Vec4i& lineSegment, int imageWidth) {
    double length = getLineLength(lineSegment);
    double angle = getLineAngleDegrees(lineSegment);

    bool angleIsHorizontal =
        angle < MAX_HORIZONTAL_ANGLE ||
        angle > 180.0 - MAX_HORIZONTAL_ANGLE;

    bool lineIsLongEnough =
        length > (imageWidth * MIN_HORIZONTAL_SEGMENT_LENGTH_FACTOR);

    return angleIsHorizontal && lineIsLongEnough;
}

vector<HorizontalLineEvidence> getHorizontalLineEvidence(
    const vector<Vec4i>& lines,
    Mat& horizontalLineDisplay
) {
    vector<HorizontalLineEvidence> evidenceList;

    for (const Vec4i& l : lines) {
        if (isHorizontalLine(l, horizontalLineDisplay.cols)) {
            int x1 = l[0];
            int y1 = l[1];
            int x2 = l[2];
            int y2 = l[3];

            line(
                horizontalLineDisplay,
                Point(x1, y1),
                Point(x2, y2),
                BLUE,
                LINE_THICKNESS
            );

            int yAvg = (y1 + y2) / 2;
            int leftX = min(x1, x2);
            int rightX = max(x1, x2);

            HorizontalLineEvidence evidence;
            evidence.y = yAvg;
            evidence.leftX = leftX;
            evidence.rightX = rightX;

            evidenceList.push_back(evidence);
        }
    }

    return evidenceList;
}

vector<Interval> mergeIntervals(vector<Interval> intervals) {
    if (intervals.empty()) {
        return {};
    }

    sort(
        intervals.begin(),
        intervals.end(),
        [](const Interval& a, const Interval& b) {
            return a.leftX < b.leftX;
        }
    );

    vector<Interval> mergedIntervals;
    mergedIntervals.push_back(intervals[0]);

    for (int i = 1; i < static_cast<int>(intervals.size()); i++) {
        Interval& last = mergedIntervals.back();
        Interval current = intervals[i];

        if (current.leftX <= last.rightX + HORIZONTAL_INTERVAL_GAP) {
            last.rightX = max(last.rightX, current.rightX);
        }
        else {
            mergedIntervals.push_back(current);
        }
    }

    return mergedIntervals;
}

MergedHorizontalDivider buildHorizontalDividerFromGroup(
    const vector<int>& ys,
    const vector<Interval>& intervals
) {
    vector<Interval> mergedIntervals = mergeIntervals(intervals);

    Interval bestInterval = mergedIntervals[0];
    int bestCoverage = bestInterval.rightX - bestInterval.leftX;

    for (const Interval& interval : mergedIntervals) {
        int coverage = interval.rightX - interval.leftX;

        if (coverage > bestCoverage) {
            bestCoverage = coverage;
            bestInterval = interval;
        }
    }

    MergedHorizontalDivider divider;
    divider.y = medianValue(ys);
    divider.leftX = bestInterval.leftX;
    divider.rightX = bestInterval.rightX;
    divider.connectedCoverage = bestCoverage;

    return divider;
}

vector<MergedHorizontalDivider> mergeHorizontalEvidence(
    vector<HorizontalLineEvidence> evidenceList,
    int imageWidth
) {
    sort(
        evidenceList.begin(),
        evidenceList.end(),
        [](const HorizontalLineEvidence& a, const HorizontalLineEvidence& b) {
            return a.y < b.y;
        }
    );

    vector<MergedHorizontalDivider> mergedDividers;

    vector<int> currentYs;
    vector<Interval> currentIntervals;

    int minConnectedCoverage =
        static_cast<int>(imageWidth * MIN_HORIZONTAL_CONNECTED_COVERAGE_FACTOR);

    for (const HorizontalLineEvidence& evidence : evidenceList) {
        Interval interval;
        interval.leftX = evidence.leftX;
        interval.rightX = evidence.rightX;

        if (currentYs.empty()) {
            currentYs.push_back(evidence.y);
            currentIntervals.push_back(interval);
        }
        else {
            int currentMedianY = medianValue(currentYs);

            if (abs(evidence.y - currentMedianY) <= HORIZONTAL_MERGE_DISTANCE) {
                currentYs.push_back(evidence.y);
                currentIntervals.push_back(interval);
            }
            else {
                MergedHorizontalDivider divider =
                    buildHorizontalDividerFromGroup(currentYs, currentIntervals);

                if (divider.connectedCoverage >= minConnectedCoverage) {
                    mergedDividers.push_back(divider);
                }

                currentYs.clear();
                currentIntervals.clear();

                currentYs.push_back(evidence.y);
                currentIntervals.push_back(interval);
            }
        }
    }

    if (!currentYs.empty()) {
        MergedHorizontalDivider divider =
            buildHorizontalDividerFromGroup(currentYs, currentIntervals);

        if (divider.connectedCoverage >= minConnectedCoverage) {
            mergedDividers.push_back(divider);
        }
    }

    return mergedDividers;
}

Mat drawMergedHorizontalLines(
    const Mat& img,
    const vector<MergedHorizontalDivider>& horizontalDividers
) {
    Mat display = img.clone();

    cout << endl << "Final horizontal dividers:" << endl;

    for (const MergedHorizontalDivider& divider : horizontalDividers) {
        cout << "horizontal divider y = " << divider.y
            << ", leftX = " << divider.leftX
            << ", rightX = " << divider.rightX
            << ", connectedCoverage = " << divider.connectedCoverage
            << endl;

        line(
            display,
            Point(divider.leftX, divider.y),
            Point(divider.rightX, divider.y),
            BLUE,
            LINE_THICKNESS
        );

        circle(display, Point(divider.leftX, divider.y), 4, GREEN, FILLED);
        circle(display, Point(divider.rightX, divider.y), 4, GREEN, FILLED);
    }

    return display;
}

// --------------------------------------------------
// Placeholder
// --------------------------------------------------

vector<ParkingSpot> getParkingSpotLocations(Mat img) {
    return {};
}

// --------------------------------------------------
// Main
// --------------------------------------------------

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

    vector<Vec4i> verticalLines =
        getHoughLines(edges, VERTICAL_HOUGH_PARAMS);

    vector<Vec4i> horizontalLines =
        getHoughLines(edges, HORIZONTAL_HOUGH_PARAMS);

    // -----------------------------
    // Vertical divider detection
    // -----------------------------

    Mat verticalLineDisplay = img.clone();

    vector<VerticalLineEvidence> verticalEvidence =
        getVerticalLineEvidence(verticalLines, verticalLineDisplay);

    vector<MergedDivider> mergedVerticalDividers =
        mergeVerticalEvidence(verticalEvidence);

    vector<MergedDivider> filteredVerticalDividers =
        filterCloseDividers(mergedVerticalDividers, img.cols);

    Mat mergedVerticalDisplay =
        drawMergedVerticalLines(img, filteredVerticalDividers);

    // -----------------------------
    // Horizontal divider detection
    // -----------------------------

    Mat horizontalLineDisplay = img.clone();

    vector<HorizontalLineEvidence> horizontalEvidence =
        getHorizontalLineEvidence(horizontalLines, horizontalLineDisplay);

    vector<MergedHorizontalDivider> mergedHorizontalDividers =
        mergeHorizontalEvidence(horizontalEvidence, img.cols);

    Mat mergedHorizontalDisplay =
        drawMergedHorizontalLines(img, mergedHorizontalDividers);

    // -----------------------------
    // Combined debug display
    // -----------------------------

    Mat combinedDisplay = img.clone();

    for (const MergedDivider& divider : filteredVerticalDividers) {
        line(
            combinedDisplay,
            Point(divider.x, divider.topY),
            Point(divider.x, divider.bottomY),
            RED,
            LINE_THICKNESS
        );
    }

    for (const MergedHorizontalDivider& divider : mergedHorizontalDividers) {
        line(
            combinedDisplay,
            Point(divider.leftX, divider.y),
            Point(divider.rightX, divider.y),
            BLUE,
            LINE_THICKNESS
        );
    }

    // -----------------------------
    // Show results
    // -----------------------------

    imshow("Original", img);
    imshow("Edges", edges);
    imshow("Vertical Lines", verticalLineDisplay);
    imshow("Merged Vertical Lines", mergedVerticalDisplay);
    imshow("Horizontal Lines", horizontalLineDisplay);
    imshow("Merged Horizontal Lines", mergedHorizontalDisplay);
    imshow("Combined Dividers", combinedDisplay);

    waitKey(0);

    return 0;
}