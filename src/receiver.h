//
// Created by evv on 14.12.18.
//

#ifndef FACENETCLASSIFIER_RECEIVER_H
#define FACENETCLASSIFIER_RECEIVER_H
#include "opencv2/opencv.hpp"
#include "iostream"
#include "sys/types.h"
#include <thread>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <vector>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include "boost/bind.hpp"
#include <boost/system/error_code.hpp>

#include "classifier.h"

/**
 * Recieve --> preprocessing --> recognition
 * ( maybe better split logic)
 */
class ImageHandler
{
public:
    /** Facenet Classifier instance ( trained on create )*/
    FacenetClassifier cls = FacenetClassifier(
            "/home/evv/CLionProjects/FacenetClassifier/openface.nn4.small2.v1.t7" // path to facenet ToDo:: Уберите пжлст хардкод пути
            ,"/home/evv/CLionProjects/FacenetClassifier/faces"); // path to image folders ToDo:: Уберите пжлст хардкод пути

    /** Haar Cascade */
    cv::CascadeClassifier csc = cv::CascadeClassifier("/home/evv/CLionProjects/FacenetClassifier/haarcascade_frontalface_alt.xml"); // ToDo:: Уберите пжлст хардкод пути


    /**
     * Function to handle ( recieve - processing - recognition) img
     * @param debug if True - imshow
     */
    void handle(bool debug = true)
    {
        try
        {
            boost::asio::io_service service;
            boost::asio::ip::tcp::acceptor acceptor(service,
                    boost::asio::ip::tcp::endpoint(
                            boost::asio::ip::tcp::v4(), _port)
                            );
            boost::asio::ip::tcp::socket socket(service);

            acceptor.accept(socket);

            boost::system::error_code error;

            while (true)
            {
                boost::array< char, 16 > header{};
                std::size_t length = boost::asio::read(
                        socket,
                        boost::asio::buffer(header),
                        boost::asio::transfer_all(),
                        error);

                // ToDo :: Вот это выглядит бессмысленно, надо поправить.
                if (length != 16)
                    continue;

                std::vector<uchar> body(
                        static_cast<unsigned long>(std::atoi((std::string(header.begin(), header.end())).c_str())));
                std::size_t lengthbody = boost::asio::read(socket, boost::asio::buffer(body), boost::asio::transfer_all(), error);
                cv::Mat frame = cv::imdecode(cv::Mat(body), cv::IMREAD_COLOR);


                //ToDo ::
                // Внимательно прочитайте пжлст, первый раз классификатор обучается при создание экземпляра класса,
                // lля того чтобы обучить его повтоно ( например после того как добавилась новая папка с фотографиями
                // вызовите функцию retrain следующим образом cls.retrain();
                // ( вызывать ее надо после того как мы получили новые изображения для обучения )

                // Create vector of faces
                std::vector<cv::Rect> faces;
                // Detect faces on img
                csc.detectMultiScale(frame, faces, 1.2, 3,
                                 CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH  ,
                                 Size(40, 40), Size(300,300));
                if (!faces.empty()) {
                    Rect roi = faces[0];
                    std::string predicted = cls.predict(frame(roi)); // ToDo :: Predicted - это значение которое надо отправить на малину. Это название класса ( то есть название папки по сути )
                    rectangle(frame, roi, cv::Scalar(160,100,10)); // Draw rectangle
                    if (!predicted.empty()) {
                        // Add label(name/label/id) under rectangle
                        putText(frame, predicted, Point(roi.x, roi.y+roi.width+13),
                            FONT_HERSHEY_PLAIN, 1.3, Scalar(160,100,10), 1, LINE_AA);
                    }
                }

                if (debug) {
                    cv::imshow("Face Recognition", frame);
                    cv::waitKey(10);
                }
            }
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
    /**
     * Set port to connect
     * @param port port value in range [0, 65535]
     */
    void set_port(unsigned short port) {
        if (port > 65535) {
            throw std::invalid_argument("Port must be in range [0,65535]");
        }
        _port = port;
    }


private:
    unsigned short _port = 5353;
};
#endif //FACENETCLASSIFIER_RECEIVER_H
