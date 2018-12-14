#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "classifier.h"
#include "receiver.h"

using namespace std;

using namespace cv;

int main(int argc, char** argv )
{
    ImageHandler reciever;
    reciever.handle();
    return 0;
}