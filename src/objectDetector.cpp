#include "objectDetector.h"

ObjectDetector::ObjectDetector() {
    laser_sub = n.subscribe("/scan", 1, &ObjectDetector::laser_callback, this);
    odom_sub = n.subscribe("/odom", 1, &ObjectDetector::odom_callback, this);
    // Init publishers
    traj_pub = n.advertise<nav_msgs::Path>("trajectory", 1000);
}

void ObjectDetector::laser_callback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    // Find the clusters
    // TODO
}

void ObjectDetector::odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header = msg->header;
    pose_msg.pose = msg->pose.pose;

    trajectory_publisher(msg->header, pose_msg);
}

void ObjectDetector::trajectory_publisher(std_msgs::Header header, geometry_msgs::PoseStamped pose) {
    traj_msg.header = header;
    traj_msg.poses.push_back(pose);

    traj_pub.publish(traj_msg);
}

/*********************************
              Main
**********************************/
int main(int argc, char *argv[]) {
    ros::init(argc, argv, "object-detector");

    ROS_INFO_STREAM("Starting up the object detection module.");
    ObjectDetector o;

    ros::Rate rate(100);
    while(ros::ok()) {
        // TODO
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}