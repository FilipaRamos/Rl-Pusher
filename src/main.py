#!/usr/bin/env python3
import rospy

from objectDetector import ObjectDetector
from src.deepPusher import DeepPusher

if __name__ == '__main__':
    rospy.init_node('object-detector', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    counter = 0
    pusher = DeepPusher()
    det = ObjectDetector()

    while not rospy.is_shutdown():
        rospy.spin()
        pusher.state_manager(counter)
        counter += 1
        rate.sleep()