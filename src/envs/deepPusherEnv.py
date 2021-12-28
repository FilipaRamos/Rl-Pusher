import gym
import json
import rospy
import numpy as np

from gym import utils, spaces
from gym.utils import seeding

from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan

from navigator import Navigator
from envs.mainEnv import MainEnv

class DeepPusherEnv(MainEnv):
    def __init__(self):
        MainEnv.__init__(self, "deepPusher-v0.launch")

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Only allow front, left, right, back movement
        self.action_space = spaces.Discrete(3)
        self.reward_range(-np.inf, np.inf)
        self._seed()
        # Load sensor config
        self.lidar = self.load_lidar_config()['lidar']
        self.navigator = Navigator()

    def load_lidar_config(self, config="./config/lidar.config"):
        data = None
        with open(config) as j_file:
            data = json.load(j_file)        
        return data

    def discretize_observation(self, data, new_ranges):
        d_ranges = []
        t_range = 0.2
        done = False
        mod = len(data.ranges) / new_ranges

        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if data.ranges[i] == float('Inf') or np.isinf(data.ranges[i]):
                    d_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    d_ranges.append(0)
                else:
                    d_ranges.append(int(data.ranges[i]))

            if (t_range > data.ranges[i] > 0):
                done = True

        return d_ranges, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("[LOG] /gazebo/unpause_physics service call failed")

        if action == 0:
            self.navigator.move_forward(0.3)
        elif action == 1:
            self.navigator.move_left(0.3)
        elif action == 2:
            self.navigator.move_right(0.3)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # TODO get cylinder to move towards to
            except:
                pass

        rospy.wait_for_message('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("[LOG] /gazebo/pause_physics service call failed")

        state, done = self.discretize_observation(data, 5)

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):
        # Restart environment state and return initial observation
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except(rospy.ServiceException) as e:
            print("[LOG] /gazebo/reset_simulation service call failed")

        # Unpause and observe
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except(rospy.ServiceException) as e:
            print("[LOG] /gazebo/unpause_physics service call failed")

        # Get laser reading
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # TODO: Reset cylinder
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except(rospy.ServiceException) as e:
            print("[LOG] /gazebo/pause_physics service call failed")

        state = self.discretize_observation(data, 5)
        return state