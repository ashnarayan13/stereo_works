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
image_transport::Publisher pub;
ros::Publisher pcpub;
image_transport::Publisher left_pcpub;
image_transport::Publisher right_pcpub;

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

//replace with the sgm code!
void compute_stereo(Mat& imL, Mat& imR)
{

  //image rectification
  double cm1[3][3] = {{1159,0,624.53}, {0,1154.4,10.111}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double cm2[3][3] = {{1162.6,0,608.43}, {0,1166.7,-12.178}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double d1[1][5] = {{ -0.163717115196273,0.0934397325136809, 0.000532967508272809,-0.00505118594885863, -0.0291492303856468}};
    double d2[1][5] = {{-0.231854971220349,0.205222935379801, 0.00322565624843538, -0.00221715658714623, -0.00249104165918136}};
    Mat CM1 (3,3, CV_64FC1, cm1);
    Mat CM2 (3,3, CV_64FC1, cm2);
    Mat D1(1,5, CV_64FC1, d1);
    Mat D2(1,5, CV_64FC1, d2);
    double r[3][3] = {{0.998950095962653,-0.0456520744550679,-0.00382019294735637},{0.0455241143701689,0.998551099348858,-0.0286924554543245},{0.00512452798000697,0.0284884202288501,0.999580986776883}};
    double t[3][4] = {{-26.9622986611212},{0.857861563076584},{0.710471337477876}};
    Mat R (3,3, CV_64FC1, r);
    Mat T (3,1, CV_64FC1, t);
    Mat R1, R2, T1, T2, Q, P1, P2;

    stereoRectify(CM1, D1,CM2,  D2, imL.size(), R, T, R1, R2, P1, P2, Q);
    Mat map11, map12, map21, map22;
    Size img_size = imL.size();
    initUndistortRectifyMap(CM1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(CM2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    Mat img1r, img2r;
    remap(imL, img1r, map11, map12, INTER_LINEAR);
    remap(imR, img2r, map21, map22, INTER_LINEAR);
    //imL = img1r;
    //imR = img2r;
    //save as images
   //storing images
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
   imwrite(tf1, imL);
   imwrite(tf2, imR);
   //imwrite(s2, imR);
  //end of rectification
  Mat Limage(imL);
  //confidence map
  Mat conf_map = Mat(imL.rows,imL.cols,CV_8U); //convert to single channel grayscale
  conf_map = Scalar(255); //scale the image

  // downsample images to speed up results
  max_disp/=2;
  if(max_disp%16 != 0) max_disp += 16-(max_disp%16);
  resize(imL, left_for_matcher,Size(),0.5,0.5);
  resize(imR, right_for_matcher,Size(),0.5,0.5);

  //compute disparity
  //replace with normal sgm disparity requests
  Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp, wsize);
  left_matcher->setP1(10*wsize*wsize);//24*wsize*wsize
  left_matcher->setP2(96*wsize*wsize);//96*wsize*wsize
  left_matcher->setPreFilterCap(63);
  left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
  wls_filter = createDisparityWLSFilter(left_matcher);
  Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

  left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
  right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);

  //filter
  wls_filter->setLambda(lambda);
  wls_filter->setSigmaColor(sigma);
  wls_filter->filter(left_disp,imL,filtered_disp,right_disp);
  conf_map = wls_filter->getConfidenceMap();
  ROI = wls_filter->getROI();

  //visualization
  getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);


  //PointCloud Generation======================================================
	/*
  // Q matrix (guess until we can do the correct calib process)
  double w = imR.cols;
  double  h = imR.rows;
  double f = 843.947693;
  double cx = 508.062911;
  double cx1 = 526.242457;
  double cy = 385.070250;
  double Tx = -120.00;
  Mat Q = Mat(4,4, CV_64F, double(0));
  Q.at<double>(0,0) = 1.0;
  Q.at<double>(0,3) = -cx;
  Q.at<double>(1,1) = 1.0;
  Q.at<double>(1,3) = -cy;
  Q.at<double>(2,3) = f;
  Q.at<double>(3,2) = -1.0/ Tx;
  Q.at<double>(3,3) = ( cx - cx1)/ Tx;



  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new  pcl::PointCloud<pcl::PointXYZRGB>());
  Mat xyz;
  reprojectImageTo3D(filtered_disp, xyz, Q, true);
  pointcloud->width = static_cast<uint32_t>(filtered_disp.cols);
  pointcloud->height = static_cast<uint32_t>(filtered_disp.rows);
  pointcloud->is_dense = true;
  pcl::PointXYZRGB point;
  for (int i = 0; i < filtered_disp.rows; ++i)
  {
    uchar* rgb_ptr = Limage.ptr<uchar>(i);
    uchar* filtered_disp_ptr = filtered_disp.ptr<uchar>(i);
    double* xyz_ptr = xyz.ptr<double>(i);

    for (int j = 0; j < filtered_disp.cols; ++j)
    {

      uchar d = filtered_disp_ptr[j];
      //if (d == 0) continue;
      Point3f p = xyz.at<Point3f>(i, j);

      double radius = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
      if(radius < 20*100)
      {
        point.z = p.z/100.0;
        point.x = p.x/100.0;
        point.y = p.y/100.0;
        point.b = rgb_ptr[ j];
        point.g = rgb_ptr[ j];
        point.r = rgb_ptr[ j];
        pointcloud->points.push_back(point);
      }

      else
      {
        point.z = 0.0;
        point.x = 0.0;
        point.y = 0.0;

        point.b = rgb_ptr[3 * j];
        point.g = rgb_ptr[3 * j];
        point.r = rgb_ptr[3 * j];
        pointcloud->points.push_back(point);
      }
    }
  }


  // voxel grid filter
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new  pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud (pointcloud);
  sor.setLeafSize (0.01, 0.01, 0.01);
  sor.filter (*cloud_filtered);


  //outliner removal filter
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered2(new  pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor1;
  sor1.setInputCloud (cloud_filtered);
  sor1.setMeanK (100);
  sor1.setStddevMulThresh (0.001);
  sor1.filter (*cloud_filtered2);
  if(enableVisualization)
  {
    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    cv::imshow(OPENCV_WINDOW, filtered_disp_vis);
    viewer.showCloud(cloud_filtered2);
    cv::waitKey(3);
  }

   // Convert to ROS data type
   sensor_msgs::PointCloud2 pointcloud_msg;
   pcl:: toROSMsg(*cloud_filtered2,pointcloud_msg);
   pointcloud_msg.header.frame_id = "stereo_frame";

   // Publishes pointcloud message
   pcpub.publish(pointcloud_msg);*/


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

//save as images
   //storing images
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
   imwrite(tf1, cv_left->image);
   imwrite(tf2, cv_right->image);
   //imwrite(s2, imR);*/
   compute_stereo(cv_left->image,cv_right->image);
   ROS_INFO("Entered callback");
   sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8",filtered_disp_vis).toImageMsg();
   sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(std_msgs::Header(), "mono8",left_disp).toImageMsg();
   sensor_msgs::ImagePtr msg3 = cv_bridge::CvImage(std_msgs::Header(), "mono8",right_disp).toImageMsg();
   pub.publish(msg);
   left_pcpub.publish(msg2);
   right_pcpub.publish(msg3);

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
  //left_pcpub = it.advertise("/results/depth/leftimage", 1);
  //right_pcpub = it.advertise("/results/depth/rightimage", 1);
  //left and right rectified images subscriber
	message_filters::Subscriber<Image> left_sub(nh, "/camera/left/image_raw", 1);
	message_filters::Subscriber<Image> right_sub(nh, "/camera/right/image_raw", 1);

  //time syncronizer to publish 2 images in the same callback function
	typedef sync_policies::ApproximateTime<Image, Image> MySyncPolicy;
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(1), left_sub, right_sub);
	//TimeSynchronizer<Image, Image> sync(left_sub, right_sub, 50);

  //call calback each time a new message arrives
  sync.registerCallback(boost::bind(&callback, _1, _2));

	ros::spin();
	return 0;
}
