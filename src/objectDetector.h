#include <ros/ros.h>

// Include messages
#include "nav_msgs/Path.h"
#include "std_msgs/Header.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Twist.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/PoseStamped.h"

class ObjectDetector {
    ros::NodeHandle n;

    ros::Subscriber laser_sub;
    ros::Subscriber odom_sub;

    ros::Publisher traj_pub;

    nav_msgs::Path traj_msg;
public:
    ObjectDetector();
    int fillObjects();
    void laser_callback(const sensor_msgs::LaserScan::ConstPtr& msg);
    void odom_callback(const nav_msgs::Odometry::ConstPtr& msg);

    void trajectory_publisher(std_msgs::Header header, geometry_msgs::PoseStamped pose);
};