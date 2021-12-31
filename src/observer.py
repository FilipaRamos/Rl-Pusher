#!/usr/bin/env python3
import time
import rospy
import numpy as np

from observerUtils import Utils, TargetCylinder

class Observer():
    ''' Class that encapsulates the robot's observation of the world '''
    def __init__(self):
        self.ut = Utils()
        self.start_ns = time.time_ns()

        # Registry control and target cylinder object
        self.register = True
        self.cyl = TargetCylinder([], [0, 0, 0])

    def observe(self, force_ob_cyl=False):
        cur_ns = time.time_ns()
        if (cur_ns - self.start_ns) % 10 == 0 and self.register:
            try:
                self.register_cylinder()
                self.register = False
            except Exception as e:
                print("[ EXCEPTION ] Raising " + type(e) + " due to " + e.args)
                return self.observe()
        elif force_ob_cyl:
            self.register = True
            pass
        elif not self.register and self.cyl.current_pos != np.asarray([0, 0, 0]):
            return self.cyl.get_layout_dict()
            
    def register_cylinder(self):
        points, cluster_size = self.det.find_cylinder(plot=True)
        if points.size > 3:
            # TODO: nr img points?
            if points.size >= cluster_size - 1:
                self.cyl.set_points(points)
                #print("[ LOG] Registered cylinder", self.cylinder.points)
            else:
                #print("[ LOG] Not registering the cylinder as there were less points than expected for the cluster size.")
                raise Exception('register_cylinder', 'less points than cluster size')
        else: 
            #print("[ LOG] Not registering the cylinder as there were not enough points for assessment.")
            raise Exception('register_cylinder', 'not enough points for assessment')

    def track_cylinder(self):
        # TODO
        print("Necessary... ?")
