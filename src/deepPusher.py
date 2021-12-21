#!/usr/bin/env python3
import rospy

class DeepPusher:
    def __init__(self, objectDetector):
        self.state = 0
        self.frequency = 10
        self.objDet = objectDetector

    def state_manager(self, counter):
        if counter > 0 and counter % self.frequency:
            self.state = 1
            self.register_cylinder()
    
    def register_cylinder(self):
        self.cylinder = self.objDet.find_cylinder()
        self.state = 2