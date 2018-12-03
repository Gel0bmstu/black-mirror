//
// Created by evv on 25.11.18.
//

#ifndef FACENETCLASSIFIER_CLASSIFIER_H
#define FACENETCLASSIFIER_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>

const int  FIXED_FACE = 96;

using namespace std;
using namespace cv;






class FacenetClassifier {
public:

    cv::dnn::Net net;
    Ptr<ml::SVM> svm;
    vector<String> names;

    FacenetClassifier(const String &pathToTorchnet) : svm(ml::SVM::create())
    {
        svm->setKernel(ml::SVM::LINEAR);
        try {
            net = dnn::readNetFromTorch(pathToTorchnet);
        } catch(Exception &e) {
            cerr << "Download it from:  ";
            cerr << "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7" << endl;
        }
    }

    Mat process(const Mat &image)
    {
        Mat inputBlob = dnn::blobFromImage(image, 1./255, Size(FIXED_FACE, FIXED_FACE), Scalar(), true, false);
        net.setInput(inputBlob);
        return net.forward();
    }

    unsigned long train(const String &imgdir)
    {
        names.clear();
        Mat features, labels;

        vector<String> vec_cls;
        utils::fs::glob(imgdir,"",vec_cls,false,true);
        if (vec_cls.empty())
            return 0;

        for (size_t label=0; label<vec_cls.size(); label++) {
            vector<String> vec_person;
            glob(vec_cls[label], vec_person, false);
            if (vec_person.empty())
                return 0;
            String name = vec_cls[label].substr(imgdir.size()+1);
            cout  << name << " " << vec_person.size() << " images." << endl;
            names.push_back(name);
            for (const auto &i : vec_person) {
                Mat img=imread(i);
                if (img.empty()) continue;
                features.push_back(process(img));
                labels.push_back(int(label));
            }
        }
        svm->train(features, 0, labels);
        //svm->save("faces.xml.gz");
        return vec_cls.size();
    }

    String predict(const Mat & img)
    {
        if (!svm->isTrained()) return names.empty() ? names[0] : "";
        Mat feature = process(img);
        float id = svm->predict(feature);
        return names[int(id)];
    }

private:

};
#endif //FACENETCLASSIFIER_CLASSIFIER_H
