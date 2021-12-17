#!/usr/bin/env python3
import rospy

from objectDetector import ObjectDetector

if __name__ == '__main__':
    rospy.init_node('object-detector', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    det = ObjectDetector()

    while not rospy.is_shutdown():
        rospy.loginfo("Inside Loop")
        rospy.spin()
        rate.sleep()