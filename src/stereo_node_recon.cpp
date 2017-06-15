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
    Mat img1, img2;
    img1 = imL; 
    img2 = imR;
    Ptr<StereoSGBM> leftSBM = StereoSGBM::create( 0, 144, 3, 36, 288, 1, 10, 10, 100, 32, StereoSGBM::MODE_SGBM );
    Mat disp, disp8;
    leftSBM->compute( img1, img2, disp );
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    // IMPORTANT FOR TUFAST
    Mat points, points1;
    double coord1[] = {5.956621e-02, 2.900141e-04, 2.577209e-03};
    double coord2[] = {-4.731050e-01, 5.551470e-03, -5.250882e-03};
    double baseline = std::sqrt(pow((coord1[0]-coord2[0]),2)+pow((coord1[1]-coord2[1]),2)+pow((coord1[2]-coord2[2]),2));
    filtered_disp_vis = disp8;
    /*reprojectImageTo3D(disp, points, Q, true); //to get the reconstruction
    cvtColor(points, points1, CV_BGR2GRAY);

    ofstream point_cloud_file;
    int start_s=clock();
    point_cloud_file.open ("point_cloud.xyz");
    
    //uchar blue = intensity.val[0];
    //uchar green = intensity.val[1];
    //uchar red = intensity.val[2];
    for(int i = 0; i < points.rows; i++) {
        for(int j = 0; j < points.cols; j++) {
            if(points.at<Vec3f>(i,j)[2] < 40*baseline) {
                point_cloud_file << points.at<Vec3f>(i,j)[0] << " " << points.at<Vec3f>(i,j)[1] << " " << points.at<Vec3f>(i,j)[2] << endl;
            }
        }
    }
    point_cloud_file.close();*/
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
