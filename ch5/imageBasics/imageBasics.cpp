#include <iostream>
#include <chrono>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main ( int argc, char** argv ){
    // read the image specified by argv[1]
    cv::Mat image;
    image = cv::imread ( argv[1] ); //cv::imread function reads the image under the specified path
    // Determine whether the image file is read correctly
    if ( image.data == nullptr ) //The data does not exist, it may be that the file does not exist
    {
        cerr<<"file"<<argv[1]<<" does not exist."<<endl;
        return 0;
    }
    
    // The file is read smoothly, first output some basic information
    cout<<" image width is "<<image.cols<<", the height is "<<image.rows<<", and the number of channels is "<<image.channels()<<endl;
    cv::imshow ( "image", image ); // Display images with cv::imshow
    cv::waitKey ( 0 ); // Pause the program and wait for a key input
    // Determine the type of image
    if ( image.type() != CV_8UC1 && image.type() != CV_8UC3 )
    {
        // Image type does not meet the requirements
        cout<<" Please enter a color map or grayscale image."<<endl;
        return 0;
    }

    // Traverse the image, please note that the following traversal method can also be used for random pixel access
    // Use std::chrono to time the algorithm
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for ( size_t y=0; y<image.rows; y++ )
    {   // Get the line pointer of the image with cv::Mat::ptr
        unsigned char* row_ptr = image.ptr<unsigned char> ( y ); // row_ptr is the head pointer of the yth line
        for ( size_t x=0; x<image.cols; x++ )
        {   // access the pixel at x, y
            unsigned char* data_ptr = &row_ptr[ x*image.channels() ]; // data_ptr points to the pixel data to be accessed
            // output each channel of the pixel, if there is only one channel in grayscale
            for ( int c = 0; c != image.channels(); c++ )
            {
                unsigned char data = data_ptr[c]; // data is the value of the cth channel of I(x,y)
            }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<" traverses the image time: "<<time_used.count()<<" seconds."<<endl;

    // About a copy of cv::Mat
    // Direct assignment does not copy data
    cv::Mat image_another = image;
    // Modify image_another will cause image to change
    image_another ( cv::Rect ( 0,0,100,100 ) ).setTo ( 0 ); // Zero the block with the top left corner of 100*100
    cv::imshow ( "image", image );
    cv::waitKey ( 0 );
    
    // Use the clone function to copy data
    cv::Mat image_clone = image.clone();
    image_clone ( cv::Rect ( 0,0,100,100 ) ).setTo ( 255 );
    cv::imshow ( "image", image );
    cv::imshow ( "image_clone", image_clone );
    cv::waitKey ( 0 );

    // There are many basic operations on the image, such as cutting, rotating, scaling, etc., limited by the length of the introduction, please refer to the OpenCV official documentation to query the calling method of each function.
    cv::destroyAllWindows();
    return 0;
}
