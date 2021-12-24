#!/usr/bin/env python3
import rospy

from objectDetector import ObjectDetector
from deepPusher import DeepPusher

if __name__ == '__main__':
    rospy.init_node('object_detector', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    counter = 0

    det = ObjectDetector()
    pusher = DeepPusher(det)
    #rospy.spin()
    while not rospy.is_shutdown():
        pusher.registry_manager(counter)
        counter += 1
        rate.sleep()