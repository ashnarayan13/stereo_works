#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>
#include <math.h>
#include <sstream>

using namespace message_filters;
using namespace sensor_msgs;
using namespace cv;
using namespace cv::ximgproc;
using namespace std;


//counter for storing images;
static long long int number = 0;
std::string NumberToString ( long long int Number )
  {
     std::ostringstream ss;
     ss << Number;
     return ss.str();
  }

bool enableVisualization = 0 ;

//replace with the sgm code!
void compute_stereo(Mat& imL, Mat& imR)
{
    Mat img1, img2;
    img1 = imL;
    img2 = imR;

    //enter camera intrinsics
    double cm1[3][3] = {{1.1501150677096175e+03, 0., 6.4480016865536629e+02}, {0,1.1530880630873185e+03, 2.4197762381320196e+01}, {0,0,1}};
    double cm2[3][3] = {{1.1505120168785099e+03, 0., 6.3578158887660538e+02}, {0,1.1511036983012948e+03, -3.6358074121825226e+00}, {0,0,1}};
    double d1[1][5] = {{-2.1893597042538027e-01, 2.2858406371589762e-01,8.1594742871778396e-04, 4.6419558504298848e-04,-1.3323578036490979e-01}};
    double d2[1][5] = {{-2.0902762537785144e-01, 2.0460757206538363e-01,7.1263330760824829e-04, 2.0061718325421339e-03,-1.1828316735697730e-01}};
    Mat CM1 (3,3, CV_64FC1, cm1);
    Mat CM2 (3,3, CV_64FC1, cm2);
    Mat D1(1,5, CV_64FC1, d1);
    Mat D2(1,5, CV_64FC1, d2);
    double r[3][3] = {{9.9947346941681314e-01, 3.2072925187972827e-02,-4.9103362212282661e-03},{-3.1767152847893811e-02,9.9808136060879526e-01, 5.3145513500525370e-02},{6.6054471353088739e-03, -5.2961543361033862e-02,9.9857470576465135e-01}};
    double t[3][4] = {{-2.7023322216827433e-01},{7.0929404961777688e-03},{-1.1203861117350783e-03}};
    Mat R (3,3, CV_64FC1, r);
    Mat T (3,1, CV_64FC1, t);
    Mat R1, R2, T1, T2, Q, P1, P2;
    stereoRectify(CM1, D1,CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
    Mat map11, map12, map21, map22;
    Size img_size = img1.size();

    //if you have the distortion and the camera coefficients can use this to undistort
    double lm1[3][3] = {{1144.594841, 0.000000, 643.335413},{ 0.000000, 1145.764848, 0.887165},{ 0.000000, 0.000000, 1.000000}};
    double ld1[1][5] = {{-0.213842, 0.141430, 0.000813, 0.001011, 0.000000}};
    double lr1[3][3] = {{0.999220, 0.011047, -0.037914},{ -0.010362, 0.999780, 0.018214},{ 0.038106, -0.017807, 0.999115}};
    double lp1[3][4] = {{1123.767906, 0.000000, 695.535263, 0.000000},{ 0.000000, 1123.767906, -13.858699, 0.000000}, {0.000000, 0.000000, 1.000000, 0.000000}};
    Mat LM1 (3, 3, CV_64FC1, lm1);
    Mat LD1 (1, 5, CV_64FC1, ld1);
    Mat LR1 (3, 3, CV_64FC1, lr1);
    Mat LP1 (3, 4, CV_64FC1, lp1);

    double rm1[3][3] = {{1149.366381, 0.000000, 632.345931},{ 0.000000, 1148.921265, -30.555935},{ 0.000000, 0.000000, 1.000000}};
    double rd1[1][5] = {{-0.194019, 0.112212, 0.000032, 0.001191, 0.000000}};
    double rr1[3][3] = {{0.998656, -0.038797, -0.034370},{0.038172, 0.999097, -0.018677},{0.035063, 0.017340, 0.9992355}};
    double rp1[3][4] = {{1123.767906, 0.000000, 695.535263, -302.45151},{0.000000, 1123.767906, -13.858699, 0.000000}, {0.000000, 0.000000, 1.000000, 0.000000}};
    Mat RM1 (3, 3, CV_64FC1, rm1);
    Mat RD1 (1, 5, CV_64FC1, rd1);
    Mat RR1 (3, 3, CV_64FC1, rr1);
    Mat RP1 (3, 4, CV_64FC1, rp1);

    initUndistortRectifyMap(CM1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(CM2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);
    img1 = img1r;
    img2 = img2r;

    //store images
    std::string t1, t2, t3, tf1, tf2;
    std::ostringstream s1, s2;
    std::string temp1;
    temp1 = NumberToString(number);
    t1 = "/home/ashwath/tufast/Cam_Data/left/";
    t2 = "/home/ashwath/tufast/Cam_Data/right/";
    t3 = ".png";
    tf1 = t1 + temp1 + t3;
    tf2 = t2 + temp1 + t3;
    number++;
    imwrite(tf1, img1);
    imwrite(tf2, img2);
}

void callback(const ImageConstPtr& left, const ImageConstPtr& right) {
  // conversion to rosmsgs::Image to cv::Mat using cv_bridge

  cv_bridge::CvImagePtr cv_left;
  try
    {
      cv_left = cv_bridge::toCvCopy(left);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

  cv_bridge::CvImagePtr cv_right;
  try
      {
        cv_right = cv_bridge::toCvCopy(right);
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
  compute_stereo(cv_left->image,cv_right->image);
  ROS_INFO("Entered callback");
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "stereo_node");
	ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  ROS_INFO("Init");
  //left and right rectified images subscriber
	message_filters::Subscriber<Image> left_sub(nh, "/camera/left/image_raw", 1);
	message_filters::Subscriber<Image> right_sub(nh, "/camera/right/image_raw", 1);

  //time syncronizer to publish 2 images in the same callback function
	typedef sync_policies::ApproximateTime<Image, Image> MySyncPolicy;
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), left_sub, right_sub);
	//TimeSynchronizer<Image, Image> sync(left_sub, right_sub, 50);

  //call calback each time a new message arrives
  sync.registerCallback(boost::bind(&callback, _1, _2));

	ros::spin();
	return 0;
}
