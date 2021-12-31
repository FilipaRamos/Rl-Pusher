#!/usr/bin/env python3
import os
import sys
import gym
import rospy
import random
import signal
import subprocess

from rosgraph_msgs.msg import Clock

class MainEnv(gym.Env):
    '''My Main Custom Env class following gym interface'''
    metadata = {'render.modes': ['human']}

    def __init__(self, pkg_path, launch_file):
        self.last_clock_msg = Clock()

        # Set port variables
        random_number = random.randint(10000, 15000)
        self.port = str(random_number)
        self.port_gazebo = str(random_number + 1)

        os.environ["ROS_MASTER_URI"] = "http://localhost:" + self.port
        os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + self.port_gazebo

        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))
        path_launch = os.path.join(pkg_path, 'launch')
        path = os.path.join(path_launch, launch_file)
        if not os.path.exists(path):
            raise IOError("File " + path + " does not exist")

        self._roslaunch = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"),
                "-p", self.port, path])

        self.gazeboc_pid = 0
        rospy.init_node('gym', anonymous=True)

    def _render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
        else:
            self.gzclient_pid = 0

    def _close(self):

        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if (gzclient_count or gzserver_count or roscore_count or rosmaster_count >0):
            os.wait()