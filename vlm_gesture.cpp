#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Cannot open camera" << endl;
        return -1;
    }

    Mat frame, roi, hsv, mask, blur;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        flip(frame, frame, 1); // mirror image

        // Define Region of Interest (ROI)
        Rect roi_box(350, 100, 300, 300);
        roi = frame(roi_box);
        rectangle(frame, roi_box, Scalar(0, 255, 0), 2);

        // Convert ROI to HSV
        cvtColor(roi, hsv, COLOR_BGR2HSV);

        // Skin color range in HSV
        Scalar lower_skin(0, 20, 70);
        Scalar upper_skin(20, 255, 255);

        // Mask skin color
        inRange(hsv, lower_skin, upper_skin, mask);
        dilate(mask, mask, Mat(), Point(-1, -1), 2);
        GaussianBlur(mask, blur, Size(5, 5), 0);

        // Find contours
        findContours(blur, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            size_t max_index = 0;
            double max_area = 0;

            // Find largest contour
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = contourArea(contours[i]);
                if (area > max_area) {
                    max_area = area;
                    max_index = i;
                }
            }

            vector<Point> hand_contour = contours[max_index];
            vector<int> hull_indices;
            vector<Vec4i> defects;

            convexHull(hand_contour, hull_indices, false, false);

            vector<Point> hull_points;
            for (int idx : hull_indices)
                hull_points.push_back(hand_contour[idx]);

            convexHull(hand_contour, hull_points);
            drawContours(roi, vector<vector<Point>>{hull_points}, 0, Scalar(0, 255, 255), 2);
            drawContours(roi, contours, max_index, Scalar(255, 255, 0), 2);

            if (hull_indices.size() > 3) {
                convexityDefects(hand_contour, hull_indices, defects);
                int count_defects = 0;

                for (size_t i = 0; i < defects.size(); ++i) {
                    Point s = hand_contour[defects[i][0]];
                    Point e = hand_contour[defects[i][1]];
                    Point f = hand_contour[defects[i][2]];
                    float depth = defects[i][3] / 256.0;

                    float a = norm(Mat(s), Mat(e));
                    float b = norm(Mat(s), Mat(f));
                    float c = norm(Mat(e), Mat(f));
                    float angle = acos((b * b + c * c - a * a) / (2 * b * c + 1e-5)) * 180 / CV_PI;

                    if (angle <= 90 && depth > 20) {
                        count_defects++;
                        circle(roi, f, 5, Scalar(0, 0, 255), -1);
                    }
                }

                string gesture;
                if (count_defects == 0) {
                    gesture = "Fist (Haptics)";
                } else if (count_defects == 1) {
                    gesture = "Thumbs Up (Levitation)";
                } else if (count_defects >= 4) {
                    gesture = "Open Palm (Audio)";
                } else {
                    gesture = "Unknown";
                }

                putText(frame, "Gesture: " + gesture, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                cout << "Gesture detected: " << gesture << endl;
            }
        }

        imshow("Gesture Recognition", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
