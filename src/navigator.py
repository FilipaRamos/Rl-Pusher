#!/usr/bin/env python3
import rospy

from geometry_msgs.msg import Twist

class Navigator():
    def __init__(self):
        self.msg = Twist()
        self.mov_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

    '''
        Actuator messages for carrying actions
    '''
    def clean_msg(self):
        self.msg.linear.x = 0
        self.msg.linear.y = 0
        self.msg.linear.z = 0

        self.msg.angular.x = 0
        self.msg.angular.y = 0
        self.msg.angular.z = 0

    def stop(self):
        self.clean_msg()
        self.mov_pub.publish(self.msg)

    def move_forward(self, v):
        self.clean_msg()
        self.msg.linear.x = v
        self.msg.linear.z = 0.0
        self.mov_pub.publish(self.msg)

    def turn(self, w):
        self.clean_msg()
        self.msg.linear.x = w / 3
        self.msg.angular.z = w
        self.mov_pub.publish(self.msg)

    def move_left(self, vel):
        self.clean_msg()
        self.msg.linear.x = vel / 3
        self.msg.angular.z = vel
        self.mov_pub.publish(self.msg)

    def move_right(self, vel):
        self.clean_msg()
        self.msg.linear.x = vel / 3
        self.msg.angular.z = -vel
        self.mov_pub.publish(self.msg)