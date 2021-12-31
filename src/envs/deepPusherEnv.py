#!/usr/bin/env python3
import gym
import json
import rospy
import numpy as np

from gym import utils, spaces
from gym.utils import seeding

from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan

from observer import Observer
from navigator import Navigator
from envs.mainEnv import MainEnv

class DeepPusherEnv(MainEnv):
    def __init__(self):
        MainEnv.__init__(self, "deepPusher-v0.launch")

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Load configs
        sim_cfg = self.load_config("../config/sim.config")
        self.sim = sim_cfg['sim']
        self.obs = sim_cfg['obs']

        self.lidar = self.load_config("../config/lidar.config")['lidar']
        self.actions = self.load_config("../config/actions.config")['actions']

        rewards_cfg = self.load_config("../config/rewards.config")['rewards']
        self.obs_idx_r, self.obs_idx_p, self.rewards, self.penalties = rewards_cfg['obs_index_r'], rewards_cfg['obs_index_p'], \
                                                                       rewards_cfg['rewards'], rewards_cfg['penalties']

        # Class that handles robot observations
        self.observer = Observer()
        # Class that handles all navigation
        self.navigator = Navigator()

        # Actions are loaded from config
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = (-np.inf, np.inf)
        self._seed()

    def load_config(self, config):
        data = None
        with open(config) as file:
            data = json.load(file)
        return data

    def robot_pos(self):
        return self.navigator.get_robot_pos()

    def dist_goal(self):
        pos = [self.goal['pos']['x'], self.goal['pos']['y'], self.goal['pos']['z']]
        return self.dist_xy(pos)

    def dist_target_cyl_goal(self, c_pos):
        g_pos = [self.goal['pos']['x'], self.goal['pos']['y'], self.goal['pos']['z']]
        return np.sqrt(np.sum(np.square(c_pos - g_pos)))

    def dist_xy(self, pos):
        robot_pos, _ = self.robot_pos()
        pos = np.asarray(pos)

        if pos.shape == (3,):
            pos = pos[:2]
        return np.sqrt(np.sum(np.square(pos - robot_pos[:2])))

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

    def observe(self):
        # TODO
        # if self.observe_goal_pos
        # if self.observe_cyl_pos
        print("TODO")

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("[LOG] /gazebo/unpause_physics service call failed")

        if action == self.actions['none']:
            self.navigator.do_nothing()
        elif action == self.actions['push_forward']:
            self.navigator.push_forward(0.3)
        elif action == self.actions['push_left']:
            self.navigator.push_left(0.3)
        elif action == self.actions['push_right']:
            self.navigator.push_right(0.3)
        elif action == self.actions['move_forward']:
            self.navigator.move_forward(0.3)
        elif action == self.actions['move_left']:
            self.navigator.move_left(0.3)
        elif action == self.actions['move_right']:
            self.navigator.move_right(0.3)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # if (self.reg_cyl is not None):
                # TODO get cylinder to move towards to
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("[LOG] /gazebo/pause_physics service call failed")

        state, done = self.discretize_observation(data, 5)

        # if self.at_goal():

        if not done:
            reward = self.reward()

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

    def reward(self):
        reward = 0.0

        dist_goal = self.dist_goal()
        dist_cyl = self.observer.cyl.dist_to()
        
        '''gate_dist_box_reward = (self.last_dist_box > self.box_null_dist * self.box_size)
        reward += (self.last_dist_box - dist_box) * self.reward_box_dist * gate_dist_box_reward
        self.last_dist_box = dist_box
        
        dist_box_goal = self.dist_box_goal()
        reward += (self.last_box_goal - dist_box_goal) * self.reward_box_goal
        self.last_box_goal = dist_box_g
'''