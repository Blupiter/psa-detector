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

// Parameters

const string IMAGE_PATH = "test2.jpg";

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
const int HORIZONTAL_INTERVAL_GAP = 55;
const double MIN_HORIZONTAL_CONNECTED_COVERAGE_FACTOR = 0.65;

// Position merging
const int MERGE_DISTANCE = 12;
const int HORIZONTAL_MERGE_DISTANCE = 12;
const double MIN_DIVIDER_SPACING_FACTOR = 0.06;

// Occupied detection
const double OCCUPIED_EDGE_DENSITY_THRESHOLD = 0.001;
const double PARKING_LOCATION_BLUR = 13; // use between 9 and 13

// Drawing
const Scalar RED = Scalar(0, 0, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar GREEN = Scalar(0, 255, 0);
const int LINE_THICKNESS = 2;

// Structs

typedef struct ParkingSpot {
    int id;
    Rect box;
    string rowName;
    bool occupied;
    double occupiedScore;
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

// A real/raw vertical line segment found by Hough.
typedef struct ActualVerticalLine {
    int x;
    int topY;
    int bottomY;
} ActualVerticalLine;

// A final/merged vertical parking divider.
typedef struct FinalVerticalLine {
    int x;
    int topY;
    int bottomY;
} FinalVerticalLine;

// A real/raw horizontal line segment found by Hough.
typedef struct ActualHorizontalLine {
    int y;
    int leftX;
    int rightX;
} ActualHorizontalLine;

typedef struct Interval {
    int leftX;
    int rightX;
} Interval;

// A final/merged horizontal parking-row divider.
typedef struct FinalHorizontalLine {
    int y;
    int leftX;
    int rightX;
    int connectedCoverage;
} FinalHorizontalLine;

// Utility functions

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

// Image preprocessing

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

// Vertical divider detection

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

vector<ActualVerticalLine> getActualVerticalLines(
    const vector<Vec4i>& lines,
    Mat& verticalLineDisplay
) {
    vector<ActualVerticalLine> actualVerticalLines;

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

            ActualVerticalLine actualLine;
            actualLine.x = xAvg;
            actualLine.topY = topY;
            actualLine.bottomY = bottomY;

            actualVerticalLines.push_back(actualLine);
        }
    }

    return actualVerticalLines;
}

vector<FinalVerticalLine> mergeActualVerticalLines(vector<ActualVerticalLine> actualVerticalLines) {
    sort(
        actualVerticalLines.begin(),
        actualVerticalLines.end(),
        [](const ActualVerticalLine& a, const ActualVerticalLine& b) {
            return a.x < b.x;
        }
    );

    vector<FinalVerticalLine> finalVerticalLines;

    vector<int> currentXs;
    vector<int> currentTopYs;
    vector<int> currentBottomYs;

    for (const ActualVerticalLine& actualLine : actualVerticalLines) {
        if (currentXs.empty()) {
            currentXs.push_back(actualLine.x);
            currentTopYs.push_back(actualLine.topY);
            currentBottomYs.push_back(actualLine.bottomY);
        }
        else {
            int currentMedianX = medianValue(currentXs);

            if (abs(actualLine.x - currentMedianX) <= MERGE_DISTANCE) {
                currentXs.push_back(actualLine.x);
                currentTopYs.push_back(actualLine.topY);
                currentBottomYs.push_back(actualLine.bottomY);
            }
            else {
                FinalVerticalLine finalLine;

                // Median x keeps one weird line from pulling the divider sideways.
                finalLine.x = medianValue(currentXs);

                // Min/max y makes the divider span the full detected height.
                finalLine.topY = *min_element(currentTopYs.begin(), currentTopYs.end());
                finalLine.bottomY = *max_element(currentBottomYs.begin(), currentBottomYs.end());

                finalVerticalLines.push_back(finalLine);

                currentXs.clear();
                currentTopYs.clear();
                currentBottomYs.clear();

                currentXs.push_back(actualLine.x);
                currentTopYs.push_back(actualLine.topY);
                currentBottomYs.push_back(actualLine.bottomY);
            }
        }
    }

    if (!currentXs.empty()) {
        FinalVerticalLine finalLine;

        finalLine.x = medianValue(currentXs);
        finalLine.topY = *min_element(currentTopYs.begin(), currentTopYs.end());
        finalLine.bottomY = *max_element(currentBottomYs.begin(), currentBottomYs.end());

        finalVerticalLines.push_back(finalLine);
    }

    return finalVerticalLines;
}

vector<FinalVerticalLine> filterCloseVerticalLines(
    const vector<FinalVerticalLine>& finalVerticalLines,
    int imageWidth
) {
    vector<FinalVerticalLine> filteredVerticalLines;

    int minDividerSpacing =
        static_cast<int>(MIN_DIVIDER_SPACING_FACTOR * imageWidth);

    for (const FinalVerticalLine& lineCandidate : finalVerticalLines) {
        if (filteredVerticalLines.empty()) {
            filteredVerticalLines.push_back(lineCandidate);
        }
        else {
            FinalVerticalLine& lastLine = filteredVerticalLines.back();

            int distance = abs(lineCandidate.x - lastLine.x);

            if (distance >= minDividerSpacing) {
                filteredVerticalLines.push_back(lineCandidate);
            }
            else {
                int currentHeight = lineCandidate.bottomY - lineCandidate.topY;
                int lastHeight = lastLine.bottomY - lastLine.topY;

                if (currentHeight > lastHeight) {
                    lastLine = lineCandidate;
                }
            }
        }
    }

    return filteredVerticalLines;
}

Mat drawFinalVerticalLines(
    const Mat& img,
    const vector<FinalVerticalLine>& finalVerticalLines
) {
    Mat display = img.clone();

    cout << endl << "Final vertical lines:" << endl;

    for (const FinalVerticalLine& finalLine : finalVerticalLines) {
        cout << "vertical line x = " << finalLine.x
            << ", topY = " << finalLine.topY
            << ", bottomY = " << finalLine.bottomY
            << endl;

        line(
            display,
            Point(finalLine.x, finalLine.topY),
            Point(finalLine.x, finalLine.bottomY),
            RED,
            LINE_THICKNESS
        );

        circle(display, Point(finalLine.x, finalLine.topY), 4, BLUE, FILLED);
        circle(display, Point(finalLine.x, finalLine.bottomY), 4, BLUE, FILLED);
    }

    return display;
}

// Horizontal divider detection

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

vector<ActualHorizontalLine> getActualHorizontalLines(
    const vector<Vec4i>& lines,
    Mat& horizontalLineDisplay
) {
    vector<ActualHorizontalLine> actualHorizontalLines;

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

            ActualHorizontalLine actualLine;
            actualLine.y = yAvg;
            actualLine.leftX = leftX;
            actualLine.rightX = rightX;

            actualHorizontalLines.push_back(actualLine);
        }
    }

    return actualHorizontalLines;
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

FinalHorizontalLine buildFinalHorizontalLineFromGroup(
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

    FinalHorizontalLine finalLine;
    finalLine.y = medianValue(ys);
    finalLine.leftX = bestInterval.leftX;
    finalLine.rightX = bestInterval.rightX;
    finalLine.connectedCoverage = bestCoverage;

    return finalLine;
}

vector<FinalHorizontalLine> mergeActualHorizontalLines(
    vector<ActualHorizontalLine> actualHorizontalLines,
    int imageWidth
) {
    sort(
        actualHorizontalLines.begin(),
        actualHorizontalLines.end(),
        [](const ActualHorizontalLine& a, const ActualHorizontalLine& b) {
            return a.y < b.y;
        }
    );

    vector<FinalHorizontalLine> finalHorizontalLines;

    vector<int> currentYs;
    vector<Interval> currentIntervals;

    int minConnectedCoverage =
        static_cast<int>(imageWidth * MIN_HORIZONTAL_CONNECTED_COVERAGE_FACTOR);

    for (const ActualHorizontalLine& actualLine : actualHorizontalLines) {
        Interval interval;
        interval.leftX = actualLine.leftX;
        interval.rightX = actualLine.rightX;

        if (currentYs.empty()) {
            currentYs.push_back(actualLine.y);
            currentIntervals.push_back(interval);
        }
        else {
            int currentMedianY = medianValue(currentYs);

            if (abs(actualLine.y - currentMedianY) <= HORIZONTAL_MERGE_DISTANCE) {
                currentYs.push_back(actualLine.y);
                currentIntervals.push_back(interval);
            }
            else {
                FinalHorizontalLine finalLine =
                    buildFinalHorizontalLineFromGroup(currentYs, currentIntervals);

                if (finalLine.connectedCoverage >= minConnectedCoverage) {
                    finalHorizontalLines.push_back(finalLine);
                }

                currentYs.clear();
                currentIntervals.clear();

                currentYs.push_back(actualLine.y);
                currentIntervals.push_back(interval);
            }
        }
    }

    if (!currentYs.empty()) {
        FinalHorizontalLine finalLine =
            buildFinalHorizontalLineFromGroup(currentYs, currentIntervals);

        if (finalLine.connectedCoverage >= minConnectedCoverage) {
            finalHorizontalLines.push_back(finalLine);
        }
    }

    return finalHorizontalLines;
}

Mat drawFinalHorizontalLines(
    const Mat& img,
    const vector<FinalHorizontalLine>& finalHorizontalLines
) {
    Mat display = img.clone();

    cout << endl << "Final horizontal lines:" << endl;

    for (const FinalHorizontalLine& finalLine : finalHorizontalLines) {
        cout << "horizontal line y = " << finalLine.y
            << ", leftX = " << finalLine.leftX
            << ", rightX = " << finalLine.rightX
            << ", connectedCoverage = " << finalLine.connectedCoverage
            << endl;

        line(
            display,
            Point(finalLine.leftX, finalLine.y),
            Point(finalLine.rightX, finalLine.y),
            BLUE,
            LINE_THICKNESS
        );

        circle(display, Point(finalLine.leftX, finalLine.y), 4, GREEN, FILLED);
        circle(display, Point(finalLine.rightX, finalLine.y), 4, GREEN, FILLED);
    }

    return display;
}

vector<ParkingSpot> getParkingSpotLocations(
    const vector<FinalVerticalLine>& finalVerticalLines,
    const vector<FinalHorizontalLine>& finalHorizontalLines,
    const Mat& img
) {
    vector<ParkingSpot> spots;

    if (finalVerticalLines.size() < 2) {
        return spots;
    }

    vector<FinalVerticalLine> verticalLines = finalVerticalLines;

    sort(
        verticalLines.begin(),
        verticalLines.end(),
        [](const FinalVerticalLine& a, const FinalVerticalLine& b) {
            return a.x < b.x;
        }
    );

    // If we have no horizontal divider, we cannot confidently split top/bottom rows yet.
    if (finalHorizontalLines.empty()) {
        cout << "No horizontal divider found, so no parking spots generated yet." << endl;
        return spots;
    }

    // Pick the strongest horizontal divider.
    // Right now that means the one with the largest connected coverage.
    FinalHorizontalLine rowDivider = finalHorizontalLines[0];

    for (const FinalHorizontalLine& line : finalHorizontalLines) {
        if (line.connectedCoverage > rowDivider.connectedCoverage) {
            rowDivider = line;
        }
    }

    int dividerY = rowDivider.y;

    // Use only vertical dividers that cross the horizontal row divider.
    vector<FinalVerticalLine> usableVerticalLines;

    for (const FinalVerticalLine& verticalLine : verticalLines) {
        bool xInsideHorizontalSpan =
            verticalLine.x >= rowDivider.leftX &&
            verticalLine.x <= rowDivider.rightX;

        bool crossesDividerY =
            verticalLine.topY <= dividerY &&
            verticalLine.bottomY >= dividerY;

        if (xInsideHorizontalSpan && crossesDividerY) {
            usableVerticalLines.push_back(verticalLine);
        }
    }

    if (usableVerticalLines.size() < 2) {
        cout << "Not enough usable vertical dividers to generate spots." << endl;
        return spots;
    }

    vector<int> topYs;
    vector<int> bottomYs;

    for (const FinalVerticalLine& verticalLine : usableVerticalLines) {
        topYs.push_back(verticalLine.topY);
        bottomYs.push_back(verticalLine.bottomY);
    }

    // Use median so one weird divider endpoint does not distort the whole row.
    int rowTopY = medianValue(topYs);
    int rowBottomY = medianValue(bottomYs);

    int insetX = 4;
    int insetY = 4;

    int id = 0;

    for (int i = 0; i < static_cast<int>(usableVerticalLines.size()) - 1; i++) {
        int leftX = usableVerticalLines[i].x;
        int rightX = usableVerticalLines[i + 1].x;

        int width = rightX - leftX;

        if (width <= 0) {
            continue;
        }

        // Top row spot
        Rect topSpotBox(
            leftX + insetX,
            rowTopY + insetY,
            width - 2 * insetX,
            dividerY - rowTopY - 2 * insetY
        );

        if (topSpotBox.width > 10 && topSpotBox.height > 10) {
            ParkingSpot spot;
            spot.id = id++;
            spot.box = topSpotBox;
            spot.rowName = "top";
            spot.occupied = false;
            spot.occupiedScore = 0.0;

            spots.push_back(spot);
        }

        // Bottom row spot
        Rect bottomSpotBox(
            leftX + insetX,
            dividerY + insetY,
            width - 2 * insetX,
            rowBottomY - dividerY - 2 * insetY
        );

        if (bottomSpotBox.width > 10 && bottomSpotBox.height > 10) {
            ParkingSpot spot;
            spot.id = id++;
            spot.box = bottomSpotBox;
            spot.rowName = "bottom";
            spot.occupied = false;
            spot.occupiedScore = 0.0;

            spots.push_back(spot);
        }
    }

    return spots;
}

Mat drawParkingSpots(const Mat& img, const vector<ParkingSpot>& parkingSpots) {
    Mat display = img.clone();

    for (const ParkingSpot& spot : parkingSpots) {
        Scalar color = spot.occupied ? RED : GREEN;
        string label = spot.occupied ? "Occ" : "Empty";

        rectangle(
            display,
            spot.box,
            color,
            LINE_THICKNESS
        );

        putText(
            display,
            label + " " + to_string(spot.id),
            Point(spot.box.x + 5, spot.box.y + 20),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        );
    }

    return display;
}

double computeEdgeDensityInSpot(const Mat& img, const Rect& spotBox) {
    Rect safeBox = spotBox & Rect(0, 0, img.cols, img.rows);

    if (safeBox.width <= 0 || safeBox.height <= 0) {
        return 0.0;
    }

    Mat roi = img(safeBox);

    Mat gray, blurred, edges;

    cvtColor(roi, gray, COLOR_BGR2GRAY);

    GaussianBlur(
        gray,
        blurred,
        Size(PARKING_LOCATION_BLUR, PARKING_LOCATION_BLUR),
        0
    );

    Canny(
        blurred,
        edges,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD
    );

    double edgePixels = countNonZero(edges);
    double totalPixels = edges.rows * edges.cols;

    return edgePixels / totalPixels;
}

void classifyParkingSpots(Mat img, vector<ParkingSpot>& parkingSpots) {
    for (ParkingSpot& spot : parkingSpots) {
        double score = computeEdgeDensityInSpot(img, spot.box);

        spot.occupiedScore = score;
        spot.occupied = score > OCCUPIED_EDGE_DENSITY_THRESHOLD;

        cout << "spot " << spot.id
            << ", row = " << spot.rowName
            << ", score = " << spot.occupiedScore
            << ", occupied = " << spot.occupied
            << endl;
    }
}

// Main

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

    vector<Vec4i> verticalHoughLines =
        getHoughLines(edges, VERTICAL_HOUGH_PARAMS);

    vector<Vec4i> horizontalHoughLines =
        getHoughLines(edges, HORIZONTAL_HOUGH_PARAMS);

    // Vertical divider detection

    Mat actualVerticalLineDisplay = img.clone();

    vector<ActualVerticalLine> actualVerticalLines =
        getActualVerticalLines(verticalHoughLines, actualVerticalLineDisplay);

    vector<FinalVerticalLine> mergedVerticalLines =
        mergeActualVerticalLines(actualVerticalLines);

    vector<FinalVerticalLine> finalVerticalLines =
        filterCloseVerticalLines(mergedVerticalLines, img.cols);

    Mat finalVerticalLineDisplay =
        drawFinalVerticalLines(img, finalVerticalLines);

    // Horizontal divider detection

    Mat actualHorizontalLineDisplay = img.clone();

    vector<ActualHorizontalLine> actualHorizontalLines =
        getActualHorizontalLines(horizontalHoughLines, actualHorizontalLineDisplay);

    vector<FinalHorizontalLine> finalHorizontalLines =
        mergeActualHorizontalLines(actualHorizontalLines, img.cols);

    Mat finalHorizontalLineDisplay =
        drawFinalHorizontalLines(img, finalHorizontalLines);

    vector<ParkingSpot> parkingSpots =
        getParkingSpotLocations(finalVerticalLines, finalHorizontalLines, img);

    classifyParkingSpots(img, parkingSpots);

    Mat parkingSpotDisplay =
        drawParkingSpots(img, parkingSpots);

    // Combined debug display

    Mat combinedDisplay = img.clone();

    for (const FinalVerticalLine& finalLine : finalVerticalLines) {
        line(
            combinedDisplay,
            Point(finalLine.x, finalLine.topY),
            Point(finalLine.x, finalLine.bottomY),
            RED,
            LINE_THICKNESS
        );
    }

    for (const FinalHorizontalLine& finalLine : finalHorizontalLines) {
        line(
            combinedDisplay,
            Point(finalLine.leftX, finalLine.y),
            Point(finalLine.rightX, finalLine.y),
            BLUE,
            LINE_THICKNESS
        );
    }

    // Show results

    imshow("Original", img);
    imshow("Edges", edges);
    waitKey(0);
    imshow("Actual Vertical Lines", actualVerticalLineDisplay);
    imshow("Final Vertical Lines", finalVerticalLineDisplay);
    waitKey(0);
    imshow("Actual Horizontal Lines", actualHorizontalLineDisplay);
    imshow("Final Horizontal Lines", finalHorizontalLineDisplay);
    waitKey(0);
    imshow("Combined Dividers", combinedDisplay);
    waitKey(0);
    imshow("Parking Spots", parkingSpotDisplay);

    waitKey(0);

    return 0;
}