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

/**
 * Facenet classifier with train/predict methids
 */
class FacenetClassifier {
public:
    /** Deep neural network instance */
    cv::dnn::Net net;
    /** SVM( support vector machines) instance */
    Ptr<ml::SVM> svm;
    /** Vector of classes (faces) string */
    vector<String> names;

    FacenetClassifier(const String &pathToTorchnet, const String &pathToImages) :
    svm(ml::SVM::create()),
    _imgs_path(pathToImages),
    _facenet_path(pathToTorchnet)
    {
        svm->setKernel(ml::SVM::LINEAR);
        try {
            net = dnn::readNetFromTorch(pathToTorchnet);
        } catch(Exception &e) {
            cerr << "Download it from:  ";
            cerr << "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7" << endl;
        }
        train(pathToImages);
    }

    /**
     * Convert image to blob ( binary large object )
     * @param image recieeve image
     * @return blob for first output of specified layer
     */
    Mat process(const Mat &image)
    {
        Mat inputBlob = dnn::blobFromImage(image, 1./255, Size(FIXED_FACE, FIXED_FACE), Scalar(), true, false);
        net.setInput(inputBlob);
        return net.forward();
    }


    /**
     * Train FaceNet on images.
     * @param imgdir path to folders with labeled images
     * @return  number of classes
     */
    unsigned long train(const String &imgdir)
    {
        names.clear();
        Mat features,labels;

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

        return vec_cls.size();
    }

    /**
     * Function to retrain.
     */
    void retrain(){
        train(_imgs_path);
    }


    /**
     * Face reckognition on img
     * @param img to find face
     * @return label ( face id/login/name )
     */
    String predict(const Mat & img)
    {
        if (!svm->isTrained()) return names.empty() ? names[0] : "";
        Mat feature = process(img);
        Mat results;
        svm->predict(feature, results, ml::StatModel::RAW_OUTPUT);
        float confidence =  -results.at<float>(0);
        if (abs(confidence) > 0.9) { // Тут можно тюнить параметр SVM ( чем меньше параметр confidence тем слабее предсказание )
            float id = svm->predict(feature);
            return names[int(id)];
        } else {
            return "null"; // ToDo :: Может быть удобнее возвращать не "null" стрингой, покумекаете.
        }
    }
private:
    String _imgs_path;
    String _facenet_path;

};
#endif //FACENETCLASSIFIER_CLASSIFIER_H
