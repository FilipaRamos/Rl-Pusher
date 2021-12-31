#!/usr/bin/env python3
import gym
import json
import rospy
import rospkg
import numpy as np

from gym import utils, spaces
from gym.utils import seeding

from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState

from observer import Observer
from navigator import Navigator
from envs.mainEnv import MainEnv

class DeepPusherEnv(MainEnv):
    def __init__(self):
        # Get ros package path
        ros_ = rospkg.RosPack()
        ros_path = ros_.get_path('deep-rl-pusher')
        MainEnv.__init__(self, ros_path, "deepPusher-v0.launch")

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Load configs
        sim_cfg = self.load_config(ros_path + "/config/sim.config")
        self.sim = sim_cfg['sim']['world']
        self.obs = sim_cfg['obs']

        self.lidar = self.load_config(ros_path + "/config/lidar.config")['lidar']
        self.actions = self.load_config(ros_path + "/config/actions.config")['actions']

        rewards_cfg = self.load_config(ros_path + "/config/rewards.config")['rewards']
        self.obs_idx_r, self.obs_idx_p, self.rewards, self.penalties = rewards_cfg['obs_index_r'], rewards_cfg['obs_index_p'], \
                                                                       rewards_cfg['rewards'], rewards_cfg['penalties']

        # Class that handles robot observations
        self.observer = Observer()
        self.observer.observe()
        # Class that handles all navigation
        self.navigator = Navigator()

        # Actions are loaded from config
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = (-np.inf, np.inf)
        
        # Parametrise the steps of the simulation so that we can penalise long solutions
        self.steps = 0
        self.max_steps = sim_cfg['sim']['max_steps']
        self._seed()

        # Calculate initial distances
        self.last_cyl_dist = self.dist_init(cyl=True)
        self.last_goal_dist = self.dist_init(cyl=False)

    def load_config(self, config):
        data = None
        with open(config) as file:
            data = json.load(file)
        return data

    def dist_init(self, cyl=True):
        ''' Calculates initial distances for the reward initialisation '''
        pose_cyl = [self.sim['target_cyl']['pos']['x'], self.sim['target_cyl']['pos']['y'], self.sim['target_cyl']['pos']['z']]
        if cyl:
            pose_robot = [self.sim['robot']['pos']['x'], self.sim['robot']['pos']['y'], self.sim['robot']['pos']['z']]
            return self.dist_xy(pose_robot, pose_cyl)
        else:
            pose_goal = [self.sim['goal']['pos']['x'], self.sim['goal']['pos']['y'], self.sim['goal']['pos']['z']]
            return self.dist_xy(pose_cyl, pose_goal)

    def at_goal(self, state):
        # Unpack state
        p_target_cyl, p_goal, _ = state
        # To be at the goal, the distance between the goal and the target cylinder must be < than the goal's radius
        r = self.sim['goal']['radius']
        dist = self.dist_xy(p_target_cyl, p_goal)
        if dist < r:
            return True
        return False

    def dist_goal(self, state):
        ''' Calculates distance from target cylinder to goal '''
        p_target_cyl, p_goal, _ = state
        return self.dist_xy(p_target_cyl, p_goal)

    def dist_cyl(self, state):
        p_target_cyl, _, p_robot = state
        return self.dist_xy(p_robot, p_target_cyl)

    def dist_xy(self, pose1, pose2):
        pose1 = np.asarray(pose1)
        pose2 = np.asarray(pose2)
        if pose1.shape == (3, ) and pose2.shape == (3, ):
            pose1 = pose1[:2]
            pose2 = pose2[:2]
        return np.sqrt(np.sum(np.square(pose1 - pose2)))

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
        ''' Model states can only be observed if we specify it in configuration '''
        if self.obs['target_cyl'] and self.obs['goal']:
            ''' Try to obtain model states from Gazebo '''
            obs = None
            while obs is None:
                try:
                    obs = rospy.wait_for_message('/gazebo_msgs/model_states', ModelState, timeout=5)
                except:
                    pass
            idx_target_cyl = obs.name.index(self.sim['target_cyl']['id'])
            pose_target_cyl = obs.pose[idx_target_cyl]

            idx_goal = obs.name.index(self.sim['goal']['id'])
            pose_goal = obs.pose[idx_goal]

            if self.obs['robot']:
                idx_robot = obs.name.index(self.sim['robot']['id'])
                pose_robot = obs.pose[idx_robot]
                
                return (pose_target_cyl, pose_goal, pose_robot)
        else:
            #TODO
            print('Not supported yet.')
            
        ''' Robot odometry estimated utilised in case we do not have access to the true pose '''
        if not self.obs['robot']:
            ''' Try to obtain odometry info from robot '''
            odom = None
            while odom is None:
                try:
                    odom = rospy.wait_for_message('/odom', Odometry, timeout=5)
                except:
                    pass
            pose_robot = odom.pose.pose.position
            ori_robot = odom.pose.pose.orientation
            
            return (pose_target_cyl, pose_goal, pose_robot)

    def reward(self, state):
        reward = 0.0

        # Target cylinder distance to goal
        dist_goal = self.dist_goal(state)
        # Distance from the robot to the target cylinder (believed by the robot)
        if self.obs['target_cyl']:
            dist_cyl = self.dist_cyl(state)
        else: 
            dist_cyl = self.observer.cyl.dist_to()

        # Dist to goal reward
        reward += (self.last_goal_dist - dist_goal) * self.rewards[self.obs_idx_r['at_goal']]
        self.last_goal_dist = dist_goal       

        # Dist to cyl reward
        reward_cyl_flag = (self.last_cyl_dist > self.sim['target_cyl']['radius'] + 0.001)
        reward += (self.last_cyl_dist - dist_cyl) * self.rewards[self.obs_idx_r['at_target_cyl']] * reward_cyl_flag
        self.last_cyl_dist = dist_cyl

        # Penalise for time step
        reward -= self.penalties[self.obs_index_p['step']]

        return reward

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

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("[LOG] /gazebo/pause_physics service call failed")

        state = self.observe()
        #state, done = self.discretize_observation(data, 5)

        done = False
        reward = self.reward(state)
        if self.observer.cyl.current_pos != [0, 0, 0]:
            if self.at_goal(state):
                reward += self.rewards[self.obs_idx_r['at_goal']]
                done = True

        self.steps += 1
        if self.steps > self.max_steps:
            done = True
        
        #return state, reward, done, {}
        return state, reward, done, self.observer.cyl.get_layout_dict()

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

        # Force new cylinder registration
        self.observer.observe()(force_ob_cyl=True)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except(rospy.ServiceException) as e:
            print("[LOG] /gazebo/pause_physics service call failed")

        #state = self.discretize_observation(data, 5)
        state = self.observe()
        return state