#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

class Navigator():
    def __init__(self):
        self.mov_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

    def move_forward(self, vel):
        msg = Twist()
        msg.linear.x = vel
        msg.linear.z = 0.0
        self.mov_pub.publish(msg)

    def move_left(self, vel):
        msg = Twist()
        msg.linear.x = vel / 6
        msg.linear.z = vel
        self.mov_pub.publish(msg)

    def move_right(self, vel):
        msg = Twist()
        msg.linear.x = vel / 6
        msg.linear.z = -vel
        self.mov_pub.publish(msg)