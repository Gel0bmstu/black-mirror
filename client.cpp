#include "opencv2/opencv.hpp"
#include "iostream"
#include "sys/types.h"
#include <thread>
#include <string>

#include <vector>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include "boost/bind.hpp"

using namespace boost::asio;

using namespace std;
using namespace cv;

class Client
{
private:

public:
    void sync()
    {
        boost::asio::io_service service;

        ip::tcp::endpoint end_point(boost::asio::ip::address::from_string("127.0.0.1"), 5353);
        ip::tcp::socket socket(service);

        boost::system::error_code ignored_error;

        socket.connect(end_point);

        cout << "Connection: OK" << endl;

        VideoCapture cap(0);

        // Check if camera opened successfully
        if(!cap.isOpened()){
          cout << "Error opening video stream or file" << endl;
          return;
        }

        Mat frame, grayFrame;

        while (true)
        {
            cap >> frame;

            cvtColor(frame,grayFrame,COLOR_RGB2GRAY);

            // If the frame is empty, break immediately
            if (grayFrame.empty())
              break;

            // jpeg compression
            vector<uchar> buff;//buffer for coding

            vector<int> param = vector<int>(2);

            param[0]=CV_IMWRITE_JPEG_QUALITY;
            param[1]=95;//default(95) 0-100

            //imshow("jpgopencvclient", grayFrame);

            imencode(".jpg",grayFrame,buff,param);
//            cout<<"coded file size(jpg)"<<buff.size()<<endl;//fit buff size automatically.

            string headlength(std::to_string(buff.size()));
            headlength.resize(16);
            std::size_t length = boost::asio::write(socket, boost::asio::buffer(headlength), boost::asio::transfer_all(), ignored_error);
//            cout << "length : "<<length<<endl;
            std::size_t lengthbody = boost::asio::write(socket, boost::asio::buffer(string(buff.begin(),buff.end())), boost::asio::transfer_all(), ignored_error);
//            cout << "lengthbody : "<<lengthbody<<endl;
            cout<<"send image finished"<<endl;
        }
        socket.close();
        cap.release();

        // Closes all the frames
        destroyAllWindows();
    }
};

int main(){
 Client client;

 client.sync();

 return 0;
}