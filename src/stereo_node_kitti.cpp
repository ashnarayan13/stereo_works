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
#include <ctime>

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
  int start_s=clock();
    cv::Rect roi;
    roi.x = 0;
    roi.y = 180;
    roi.width = imL.size().width - (0*2);
    roi.height = imL.size().height - (180);
    Mat img1, img2;
    img1 = imL(roi);
    img2 = imR(roi);
    /*double cm1[3][3] = {{9.597910e+02, 0.000000e+00, 6.960217e+02}, {0.000000e+00, 9.569251e+02, 2.241806e+02}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double cm2[3][3] = {{9.037596e+02, 0.000000e+00, 6.957519e+02 }, {0.000000e+00, 9.019653e+02, 2.242509e+02}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double d1[1][5] = {{ -3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02}};
    double d2[1][5] = {{-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02}};

    Mat CM1 (3,3, CV_64FC1, cm1);
    Mat CM2 (3,3, CV_64FC1, cm2);
    Mat D1(1,5, CV_64FC1, d1);
    Mat D2(1,5, CV_64FC1, d2);
    double r[3][3] = {{9.995599e-01, 1.699522e-02, -2.431313e-02},{-1.704422e-02, 9.998531e-01, -1.809756e-03 },{2.427880e-02, 2.223358e-03, 9.997028e-01}};
    double t[3][4] = {{ -4.731050e-01}, {5.551470e-03}, {-5.250882e-03}};

    Mat R (3,3, CV_64FC1, r);
    Mat T (3,1, CV_64FC1, t);

    Mat R1, R2, T1, T2, Q, P1, P2;
    ROS_INFO("HERE!");

    stereoRectify(CM1, CM2, D1, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
    Mat map11, map12, map21, map22;
    Size img_size = img1.size();

    initUndistortRectifyMap(CM1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(CM2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);
    //img1 = img1r; 
    //img2 = img2r;*/
    Ptr<StereoSGBM> leftSBM = StereoSGBM::create( 0, 144, 3, 36, 288, 1, 10, 10, 100, 32, StereoSGBM::MODE_SGBM );
    Mat disp, disp8;
    leftSBM->compute( img1, img2, disp );
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    // IMPORTANT FOR TUFAST
    Mat points, points1;
    double coord1[] = {5.956621e-02, 2.900141e-04, 2.577209e-03};
    double coord2[] = {-4.731050e-01, 5.551470e-03, -5.250882e-03};
    double baseline = 0.54;//td::sqrt(pow((coord1[0]-coord2[0]),2)+pow((coord1[1]-coord2[1]),2)+pow((coord1[2]-coord2[2]),2));
    ROS_INFO("Baseline %lf", baseline);
    filtered_disp_vis = disp8;
    
    //POINT CLOUD!
  /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new  pcl::PointCloud<pcl::PointXYZRGB>());
  Mat xyz, Limage;
  Limage = imL;
  reprojectImageTo3D(disp8, xyz, Q, true);
  pointcloud->width = static_cast<uint32_t>(disp8.cols);
  pointcloud->height = static_cast<uint32_t>(disp8.rows);
  pointcloud->is_dense = true;
  pcl::PointXYZRGB point;
  for (int i = 0; i < disp8.rows; ++i)
  {
    uchar* rgb_ptr = Limage.ptr<uchar>(i);
    uchar* filtered_disp_ptr = disp8.ptr<uchar>(i);
    double* xyz_ptr = xyz.ptr<double>(i);

    for (int j = 0; j < disp8.cols; ++j)
    {

      Point3f p = xyz.at<Point3f>(i, j);
      if(p.z < 20*baseline)
      {
      point.z = p.z/100.0;
      point.x = p.x/100.0;
      point.y = p.y/100.0;
      point.b = rgb_ptr[ j];
      point.g = rgb_ptr[ j];
      point.r = rgb_ptr[ j];
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
    double total;
    int stop_s = clock();
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
