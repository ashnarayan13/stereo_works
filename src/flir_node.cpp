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
#include "opencv2/cudastereo.hpp"
#include <iostream>
#include <string>
#include <math.h>
#include <sstream>
#include <ctime>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

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

bool enableVisualization = 1 ;
// uncomment to visualize results
static const std::string OPENCV_WINDOW = "Image window";


//publishers for image and pointcloud
image_transport::Publisher pub, lpub, rpub;
ros::Publisher pcpub;

//parameters for stereo matching and filtering
double vis_mult = 5.0;
int wsize = 7;
int max_disp = 256;//16 * 10
double lambda = 8000.0;
double sigma = 1.5;


//Some object instatiation that can be done only once
Mat left_for_matcher,right_for_matcher;
Mat left_disp, right_disp;
Mat filtered_disp;
Rect ROI ;
Ptr<DisparityWLSFilter> wls_filter;
Mat filtered_disp_vis;
Mat limg, rimg;

//replace with the sgm code!
void compute_stereo(Mat& imL, Mat& imR)
{
    int start_s = clock();
    Mat mask = imread("/home/ashwath/catkin_ws/mask.png", imL.type());
    Mat img1, img2;
    cuda::GpuMat d_left, d_right;
    //img1 = cv::Mat::zeros(imL.size(), imL.type());
    //img2 = cv::Mat::zeros(imR.size(), imR.type());
    //imL.copyTo(img1, mask);
    //imR.copyTo(img2, mask);
    img1 = imL;
    img2 = imR;
    /*double cm1[3][3] = {{1.1501150677096175e+03, 0., 6.4480016865536629e+02}, {0,1.1530880630873185e+03, 2.4197762381320196e+01}, {0,0,1}};
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

    initUndistortRectifyMap(CM1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(CM2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);
    img1 = img1r;
    img2 = img2r;*/
    d_left.upload(img1);
    d_right.upload(img2);
    //Ptr<StereoSGBM> leftSBM = StereoSGBM::create( 0, 144, 9, 36, 288, 1, 10, 10, 100, 32, StereoSGBM::MODE_SGBM );
    Mat disp, disp8;
    //leftSBM->compute( img1, img2, disp );
    Ptr<cuda::StereoBM> bm;
    bm->compute(d_left, d_right, disp);
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    // IMPORTANT FOR TUFAST
    Mat points, points1;
    double baseline = 2.6994964132128269e-01;//std::sqrt(pow((coord1[0]-coord2[0]),2)+pow((coord1[1]-coord2[1]),2)+pow((coord1[2]-coord2[2]),2));
    filtered_disp_vis = disp8;
    /*
    std::string t1, t2, t3, tf1, tf2;
    std::ostringstream s1, s2;
    std::string temp1;
    temp1 = NumberToString(number);
    t1 = "/home/ashwath/tufast/Cam_Data/pointcloud/";
    t3 = ".xyz";
    tf1 = t1 + temp1 + t3;
    number++;
    std::cout<<"Q: "<<Q<<endl;
    reprojectImageTo3D(disp, points, Q, true); //to get the reconstruction
    cvtColor(points, points1, CV_BGR2GRAY);

    ofstream point_cloud_file;
    point_cloud_file.open("point_cloud.xyz");
    for(int i = 0; i < points.rows; i++) {
        for(int j = 0; j < points.cols; j++) {
            if(points.at<Vec3f>(i,j)[2] < 40*baseline) {
                point_cloud_file << points.at<Vec3f>(i,j)[0] << " " << points.at<Vec3f>(i,j)[1] << " " << points.at<Vec3f>(i,j)[2] << endl;
            }
        }
    }
    point_cloud_file.close();*/
    int stop_s = clock();
    double total;
    total=(stop_s-start_s)/double(CLOCKS_PER_SEC)*1000;
    total = 1000/total;
    ROS_INFO("time: %lf",total);
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
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8",filtered_disp_vis).toImageMsg();
  pub.publish(msg);

}

int main(int argc, char **argv) {

  ros::init(argc, argv, "stereo_node");
	ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  ROS_INFO("Init");
  //pointcloud publisher
  pcpub = nh.advertise<sensor_msgs::PointCloud2> ("/results/depth/pointcloud", 1);
  // depth image publisher
  pub = it.advertise("/results/depth/image", 1);
  //left and right rectified images subscriber
	message_filters::Subscriber<Image> left_sub(nh, "/camera/left/image_rect", 1);
	message_filters::Subscriber<Image> right_sub(nh, "/camera/right/image_rect", 1);

  //time syncronizer to publish 2 images in the same callback function
	typedef sync_policies::ApproximateTime<Image, Image> MySyncPolicy;
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(1), left_sub, right_sub);
	//TimeSynchronizer<Image, Image> sync(left_sub, right_sub, 50);

  //call calback each time a new message arrives
  sync.registerCallback(boost::bind(&callback, _1, _2));

	ros::spin();
	return 0;
}
