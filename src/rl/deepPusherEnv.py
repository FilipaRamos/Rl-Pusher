import gym
import json
import rospy
import numpy as np

from gym import utils, spaces
from gym.utils import seeding

from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from mainEnv import MainEnv

class deepPusherEnv(MainEnv):
    def __init__(self):
        MainEnv.__init__(self, "deep-rl-pusher.launch")

        self.mov_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Only allow front, left, right movement
        self.action_space = spaces.Discrete(3)
        self.reward_range(-np.inf, np.inf)
        self._seed()
        # Load sensor config
        self.lidar = self.load_lidar_config()['lidar']

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
            
