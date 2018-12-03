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

class Server
{
private:

public:
    void run()
    {
        try
        {
            io_service service;
            ip::tcp::acceptor acceptor(service, ip::tcp::endpoint(ip::tcp::v4(), 5353));
            ip::tcp::socket socket(service);

            acceptor.accept(socket);

            boost::system::error_code error;

            while (true)
            {
                boost::array< char, 16 > header;
                std::size_t length = boost::asio::read(socket, boost::asio::buffer(header), boost::asio::transfer_all(), error);
                //cout<<"length : "<< length<<endl;

                if(length != 16)
                    continue;

                //cout << "length data : " << string(header.begin(), header.end()) << endl;

                std::vector<uchar> body(atoi((string(header.begin(), header.end())).c_str()));
                std::size_t lengthbody = boost::asio::read(socket, boost::asio::buffer(body), boost::asio::transfer_all(), error);

                //cout << "lengthbody : "<< lengthbody <<endl;

                Mat frame = imdecode(Mat(body),CV_LOAD_IMAGE_COLOR);

                imshow("jpgopencvserver", frame);

                waitKey(10);
            }

            socket.close();

        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
};

int main()
{
    Server server;

    server.run();

    return 0;
}