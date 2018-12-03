#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "classifier.h"

using namespace cv;

int main(int argc, char** argv )
{
    // Sort of programm states
    enum
    {
        NEUTRAL = 0,
        RECORD  = 1,
        PREDICT = 2
    };

    theRNG().state = static_cast<uint64>(getTickCount()); // random generator for user id :: ToDo

    // Create facenet object
    FacenetClassifier classifier("/home/evv/CLionProjects/FacenetClassifier/openface.nn4.small2.v1.t7"); // Remove hardcode path :: ToDo

    cv::CascadeClassifier cascade("/home/evv/CLionProjects/FacenetClassifier/haarcascade_frontalface_alt.xml"); // Remove hardcode path :: ToDo
    if (cascade.empty()) {
        cerr << "Could not load cascade: ";
        return -1;
    }

    VideoCapture cap;
    cap.open(0);
    if (! cap.isOpened()) {
        cerr << "Could not open capture: " << endl;
        return -2;
    }

    String imgpath = "/home/evv/CLionProjects/FacenetClassifier/lfw40_crop"; // Remove hardcode :: ToDo


    utils::fs::createDirectory(imgpath);
    unsigned long n = classifier.train(imgpath);
    cout << "train: " << n << " classes." << endl;

    namedWindow("Face Recognition");
    cout << "press 'r'     evv to record new persons" << endl;
    cout << "      'space'  to stop recording (then input a name on the console)" << endl;
    cout << "      'p'      to predict" << endl;
    cout << "      'esc'    to quit" << endl;

    vector<Mat> images;
    String caption = "";
    int frameNo = 0;
    int state = NEUTRAL;
    Scalar color[3] = {
            Scalar(30,100,30),
            Scalar(10,10,160),
            Scalar(160,100,10),
    };

    while(true) {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;


        Mat gray;
        cvtColor(frame, gray, COLOR_RGB2GRAY);

        std::vector<cv::Rect> faces;
        cascade.detectMultiScale(gray, faces, 1.2, 3,
                                 CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH  ,
                                 Size(40, 40), Size(300,300));

        if (faces.size() > 0) {
            Rect roi = faces[0];
            if ((state == RECORD ) && (frameNo % 3 == 0)) {
                Mat m;
                resize(frame(roi), m, Size(FIXED_FACE,FIXED_FACE));
                images.push_back(m);
                cout << ".";
            }
            if (state == PREDICT) {
                caption = classifier.predict(frame(roi));
                if (caption != "") {
                    putText(frame, caption, Point(roi.x, roi.y+roi.width+13),
                            FONT_HERSHEY_PLAIN, 1.3, color[state], 1, LINE_AA);
                }
            }
            rectangle(frame, roi, color[state]);
        }
        for(int i=6,sc=6; i>1; i--,sc+=2) // status led
            circle(frame, Point(10,10), i, color[state]*(float(sc)/10), -1, LINE_AA);

        imshow("Face Recognition", frame);
        int k = waitKey(30);
        if (k == 27 ) break;
        if (k == 'p') state = PREDICT;
        if (k == 'r' && state != RECORD ) {
            images.clear();
            state = RECORD ;
        }
        if (k==' ') {
            if ((state == RECORD ) && (!images.empty())) {
                // ask for a name, and write the images to that folder:
                cout << endl << "please enter a name (leave empty to ignore) :" << endl;
                string n; cin >> n;
                if ((!n.empty()) && (images.size() > 0)) {
                    String folder(imgpath + String("/") + String(n));
                    utils::fs::createDirectory(folder);
                    for (const auto &image : images) {
                        imwrite(format("%s/%6d.jpg", folder.c_str(), theRNG().uniform(0,100000)), image);
                    }
                    classifier.train(imgpath);
                }
            }
            state = NEUTRAL;
        }
        frameNo++;
    }
    return 0;
}