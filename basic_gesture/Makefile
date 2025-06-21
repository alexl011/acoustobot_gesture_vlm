CXX = g++
CXXFLAGS = -std=c++11 -I/usr/local/opt/opencv/include/opencv4
LDFLAGS = -L/usr/local/opt/opencv/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_gapi -lopencv_objdetect

TARGET = vlm_gesture
SOURCE = vlm_gesture.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean 