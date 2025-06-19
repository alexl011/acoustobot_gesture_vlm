#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

class BackgroundRemover {
public:
    void removeBackground(Mat& input, const Mat& background) {
        int thresholdOffset = 10;
        
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                uchar framePixel = input.at<uchar>(i, j);
                uchar bgPixel = background.at<uchar>(i, j);
                
                if (framePixel >= bgPixel - thresholdOffset && framePixel <= bgPixel + thresholdOffset)
                    input.at<uchar>(i, j) = 0;
                else
                    input.at<uchar>(i, j) = 255;
            }
        }
    }
};

class HandDetector {
private:
    BackgroundRemover bgRemover;
    Mat backgroundFrame;
    bool backgroundSet = false;
    bool skinSampled = false;
    Scalar lowerSkin, upperSkin;
    
    void performOpening(Mat& binaryImage, int kernelShape, Point kernelSize) {
        Mat structuringElement = getStructuringElement(kernelShape, kernelSize);
        morphologyEx(binaryImage, binaryImage, MORPH_OPEN, structuringElement);
    }
    
    void sampleSkinColor(const Mat& roi) {
        Mat hsv;
        cvtColor(roi, hsv, COLOR_BGR2HSV);
        
        // Sample skin color from the ROI
        Scalar meanColor = mean(hsv);
        lowerSkin = Scalar(std::max(0.0, meanColor[0] - 20), std::max(20.0, meanColor[1] - 50), std::max(70.0, meanColor[2] - 50));
        upperSkin = Scalar(std::min(20.0, meanColor[0] + 20), 255, 255);
        
        skinSampled = true;
        cout << "Skin color sampled. Lower: " << lowerSkin << " Upper: " << upperSkin << endl;
    }
    
public:
    void removeFace(Mat& frame) {
        CascadeClassifier faceCascade;
        if (faceCascade.load("/usr/local/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")) {
            vector<Rect> faces;
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
            
            for (const Rect& face : faces) {
                rectangle(frame, face, Scalar(0, 0, 0), -1);
            }
        }
    }
    
    int detectFingers(const Mat& roi, Mat& output) {
        if (!backgroundSet || !skinSampled) {
            return -1;
        }

        // Always work on a color copy for drawing
        Mat roi_color;
        if (roi.channels() == 3) {
            roi_color = roi.clone();
        } else {
            cvtColor(roi, roi_color, COLOR_GRAY2BGR);
        }

        // For processing, use grayscale
        Mat gray, foreground, skinMask, handMask;
        cvtColor(roi, gray, COLOR_BGR2GRAY);

        // Background removal (make sure backgroundFrame is also grayscale)
        foreground = gray.clone();
        Mat roiBackground = backgroundFrame(Rect(0, 0, roi.cols, roi.rows));
        bgRemover.removeBackground(foreground, roiBackground);
        imshow("Foreground", foreground);

        // Skin color detection
        Mat hsv;
        cvtColor(roi_color, hsv, COLOR_BGR2HSV);
        inRange(hsv, lowerSkin, upperSkin, skinMask);
        imshow("SkinMask", skinMask);

        // Morphological operations
        performOpening(skinMask, MORPH_ELLIPSE, Point(3, 3));
        dilate(skinMask, skinMask, Mat(), Point(-1, -1), 3);

        handMask = skinMask.clone();
        imshow("HandMask", handMask);

        // Find contours
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(handMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            roi_color.copyTo(output);
            return -1;
        }

        // Find largest contour (hand)
        size_t maxIndex = 0;
        double maxArea = 0;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIndex = i;
            }
        }

        if (maxArea < 1000) { // Minimum hand area
            roi_color.copyTo(output);
            return -1;
        }

        vector<Point> handContour = contours[maxIndex];

        // Convex hull
        vector<int> hullIndices;
        convexHull(handContour, hullIndices, false, false);

        vector<Point> hullPoints;
        for (int idx : hullIndices) {
            hullPoints.push_back(handContour[idx]);
        }

        // Convexity defects
        vector<Vec4i> defects;
        if (hullIndices.size() > 3) {
            convexityDefects(handContour, hullIndices, defects);
        }

        // Count fingers
        int fingerCount = 0;
        Point center = Point(0, 0);
        for (const Point& p : handContour) {
            center += p;
        }
        center.x /= handContour.size();
        center.y /= handContour.size();

        for (const Vec4i& defect : defects) {
            Point start = handContour[defect[0]];
            Point end = handContour[defect[1]];
            Point far = handContour[defect[2]];
            float depth = defect[3] / 256.0f;

            // Calculate angle
            float a = norm(start - end);
            float b = norm(start - far);
            float c = norm(end - far);
            float angle = acos((b * b + c * c - a * a) / (2 * b * c + 1e-5)) * 180 / CV_PI;

            // Filter defects based on angle and depth
            if (angle <= 90 && depth > 20) {
                // Check if this is a valid finger
                if (far.y < center.y && depth > 30) {
                    fingerCount++;
                    circle(roi_color, far, 5, Scalar(0, 0, 255), -1);
                }
            }
        }

        // Draw contours on the color ROI
        drawContours(roi_color, vector<vector<Point>>{hullPoints}, 0, Scalar(0, 255, 255), 2);
        drawContours(roi_color, contours, maxIndex, Scalar(255, 255, 0), 2);

        // Copy the drawn ROI back to output
        roi_color.copyTo(output);
        return fingerCount;
    }
    
    void setBackground(const Mat& frame) {
        cvtColor(frame, backgroundFrame, COLOR_BGR2GRAY);
        backgroundSet = true;
        cout << "Background set" << endl;
    }
    
    void sampleSkin(const Mat& roi) {
        sampleSkinColor(roi);
    }
    
    bool isReady() const {
        return backgroundSet && skinSampled;
    }
};

// Returns the number of fingers detected in the ROI
int countFingers(const Mat& roi) {
    Mat hsv, mask;
    // 1. Convert to HSV
    cvtColor(roi, hsv, COLOR_BGR2HSV);

    // 2. Skin color range (tune as needed)
    Scalar lower_skin(0, 30, 60);
    Scalar upper_skin(20, 150, 255);
    inRange(hsv, lower_skin, upper_skin, mask);

    // 3. Morphological operations
    erode(mask, mask, Mat(), Point(-1, -1), 2);
    dilate(mask, mask, Mat(), Point(-1, -1), 2);
    GaussianBlur(mask, mask, Size(5, 5), 0);

    // 4. Find contours
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return 0;

    // 5. Find the largest contour (hand)
    int maxIdx = 0;
    double maxArea = 0;
    for (int i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }
    vector<Point> handContour = contours[maxIdx];

    // 6. Convex hull and defects
    vector<int> hull;
    convexHull(handContour, hull, false, false);

    if (hull.size() < 3) return 0;

    vector<Vec4i> defects;
    convexityDefects(handContour, hull, defects);

    // 7. Count fingers using defects
    int fingerCount = 0;
    for (size_t i = 0; i < defects.size(); ++i) {
        Point s = handContour[defects[i][0]];
        Point e = handContour[defects[i][1]];
        Point f = handContour[defects[i][2]];
        float depth = defects[i][3] / 256.0;

        // Calculate the angle
        float a = norm(s - f);
        float b = norm(e - f);
        float c = norm(s - e);
        float angle = acos((a*a + b*b - c*c) / (2*a*b + 1e-5)) * 180 / CV_PI;

        // Only count as a finger if angle < 90 and depth > 20
        if (angle < 90 && depth > 20) {
            fingerCount++;
        }
    }

    // Usually, fingerCount = number of gaps between fingers, so add 1
    return min(fingerCount + 1, 5);
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Cannot open camera" << endl;
        return -1;
    }
    
    HandDetector detector;
    Mat frame, output;
    
    cout << "Instructions:" << endl;
    cout << "1. Press 'B' to set background (without hand in frame)" << endl;
    cout << "2. Press 'S' to sample skin color (with hand in frame)" << endl;
    cout << "3. Press 'Q' to quit" << endl;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        flip(frame, frame, 1); // Mirror image
        output = frame.clone();

        // Remove face before hand detection
        detector.removeFace(frame);
        
        // Draw ROI
        Rect roiBox(250, 50, 400, 400);
        rectangle(output, roiBox, Scalar(0, 255, 0), 2);
        
        // Show instructions
        if (!detector.isReady()) {
            putText(output, "Press 'B' to set background", Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            putText(output, "Press 'S' to sample skin", Point(50, 80), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }
        
        // Detect fingers if ready
        if (detector.isReady()) {
            Mat roi_frame = frame(roiBox); // for processing
            Mat roi_output = output(roiBox); // for drawing
            int fingerCount = detector.detectFingers(roi_frame, roi_output);
            
            string gesture;
            if (fingerCount == 0) {
                gesture = "Fist (Haptics)";
            } else if (fingerCount == 1) {
                gesture = "Thumbs Up (Levitation)";
            } else if (fingerCount >= 4) {
                gesture = "Open Palm (Audio)";
            } else {
                gesture = "Unknown";
            }
            
            putText(output, "Fingers: " + to_string(fingerCount), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            putText(output, "Gesture: " + gesture, Point(50, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        
        imshow("Handy Gesture Recognition", output);
        // The debug windows are shown from detectFingers
        
        char key = waitKey(1);
        if (key == 'q' || key == 'Q') {
            break;
        } else if (key == 'b' || key == 'B') {
            detector.setBackground(frame);
        } else if (key == 's' || key == 'S') {
            Mat roi = frame(roiBox);
            detector.sampleSkin(roi);
        }
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}
