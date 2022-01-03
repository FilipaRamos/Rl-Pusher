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
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates, LinkStates

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
        self.obs_idx_r, self.obs_idx_p, self.rewards, self.penalties, self.r_scale_factor = rewards_cfg['obs_index_r'], rewards_cfg['obs_index_p'], \
                                                                       rewards_cfg['rewards'], rewards_cfg['penalties'], rewards_cfg['scale_factor']

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

        # Terminate early if the robot is stuck
        self.robot_stuck = 0
        self.previous_robot_pose = [0, 0, 0]

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
        _, p_target_cyl, p_goal, _, _ = state
        # To be at the goal, the distance between the goal and the target cylinder must be < than the goal's radius
        r = self.sim['goal']['radius']
        dist = self.dist_xy(p_target_cyl, p_goal)
        if dist < r:
            return True
        return False

    def dist_goal(self, state):
        ''' Calculates distance from target cylinder to goal '''
        _, p_target_cyl, p_goal, _, _ = state
        return self.dist_xy(p_target_cyl, p_goal)

    def dist_cyl(self, state):
        _, p_target_cyl, _, p_robot, _ = state
        return self.dist_xy(p_robot, p_target_cyl)

    def dist_xy(self, pose1, pose2):
        pose1 = np.asarray(pose1)
        pose2 = np.asarray(pose2)
        if pose1.shape == (3, ) and pose2.shape == (3, ):
            pose1 = pose1[:2]
            pose2 = pose2[:2]
        return np.sqrt(np.sum(np.square(pose1 - pose2)))

    def pose_to_array(self, pose):
        pose_ = [pose.position.x, pose.position.y, pose.position.z]
        return np.asarray(pose_)

    def quat_to_array(self, pose):
        pose_ = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        return np.asarray(pose_)
        
    def ori_align(self, pair):
        # Is this correct?
        import math
        import transforms3d as td3
        pos_g, pos_r = pair

        ori_diff = math.atan2(pos_r[2] - pos_g[2], pos_r[0] - pos_g[0])
        print("=-------------- Alignment = ", ori_diff)

        return abs(ori_diff)

    def observe_lidar(self, data, new_ranges):
        d_ranges = []
        mod = len(data.ranges) / new_ranges

        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if data.ranges[i] == float('Inf') or np.isinf(data.ranges[i]):
                    d_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    d_ranges.append(0)
                else:
                    d_ranges.append(int(data.ranges[i]))

        return d_ranges

    def check_pose_stuck(self, obs_pose):
        if round(obs_pose[0], 4) == round(self.previous_robot_pose[0], 4) \
            and round(obs_pose[1], 4) == round(self.previous_robot_pose[1], 4) \
            and round(obs_pose[2], 4) == round(self.previous_robot_pose[2], 4):
            return True
        return False

    def observe(self):
        ''' Observe pointcloud '''
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        pc = self.observe_lidar(data, 6)
        ''' Model states can only be observed if we specify it in configuration '''
        if self.obs['target_cyl'] and self.obs['goal']:
            ''' Try to obtain model states from Gazebo '''
            obs = None
            while obs is None or self.sim['robot']['id'] not in obs.name:
                try:
                    obs = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
                except:
                    pass
            idx_target_cyl = obs.name.index(self.sim['target_cyl']['id'])
            pose_target_cyl = self.pose_to_array(obs.pose[idx_target_cyl])

            idx_goal = obs.name.index(self.sim['goal']['id'])
            pose_goal = self.pose_to_array(obs.pose[idx_goal])

            if self.obs['robot']:
                idx_robot = obs.name.index(self.sim['robot']['id'])
                pose_robot = self.pose_to_array(obs.pose[idx_robot])
                ori_robot = self.quat_to_array(obs.pose[idx_robot])
                
                return (pc, pose_target_cyl, pose_goal, pose_robot, ori_robot)
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
            pose_robot = self.pose_to_array(odom.pose.pose)
            ori_robot = self.quat_to_array(odom.pose.pose)
            
            return (pc, pose_target_cyl, pose_goal, pose_robot, ori_robot)

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
        reward_cyl_flag = (self.last_cyl_dist > self.sim['target_cyl']['radius'] + 0.01)
        reward += (self.last_cyl_dist - dist_cyl) * self.rewards[self.obs_idx_r['at_target_cyl']] * reward_cyl_flag
        self.last_cyl_dist = dist_cyl

        # Orientation reward
        #ori = (state[2] if (dist_cyl < 0.2) else state[1], state[3])
        #reward -= 1.2 * self.ori_align(ori)

        # Penalise for time step
        reward -= self.penalties[self.obs_idx_p['step']]
        
        # Rewards are on a too small scale
        reward = reward*self.r_scale_factor
        
        # Clip
        #in_range = reward < self.reward_clip and reward > -self.reward_clip
        #if not(in_range):
        #    reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        #    print('Warning: reward was outside of range!')

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
        #elif action == self.actions['push_forward']:
        #    self.navigator.push_forward()
        #elif action == self.actions['push_left']:
        #    self.navigator.push_left()
        #elif action == self.actions['push_right']:
        #    self.navigator.push_right()
        elif action == self.actions['move_forward']:
            self.navigator.move_forward(0.35)
        elif action == self.actions['move_left']:
            self.navigator.move_left(0.15)
        elif action == self.actions['move_right']:
            self.navigator.move_right(0.15)

        # Observe before pausing since our observations depend on Gazebo clock being published
        state = self.observe()

        if self.steps > 0:
            if self.check_pose_stuck(state[3]):
                self.robot_stuck += 1
            else:
                self.robot_stuck = 0
        self.previous_robot_pose = state[3]

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("[LOG] /gazebo/pause_physics service call failed")

        #state, done = self.discretize_observation(data, 5)

        done = False
        reward = self.reward(state)
        #if self.observer.cyl.current_pos[0] != 0 and self.observer.cyl.current_pos[1] != 0 and self.observer.cyl.current_pos[2] != 0:
        if self.at_goal(state):
            reward += self.rewards[self.obs_idx_r['at_goal']]
            done = True
        if self.robot_stuck > 6:
            done = True
            self.robot_stuck = 0
            print("ROBOT STUCK...")

        self.steps += 1
        if self.steps == self.max_steps:
            done = True
            print("We are so done...")
        
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
        self.observer.observe(force_ob_cyl=True)

        # Observe before pausing since our observations depend on Gazebo clock being published
        state = self.observe()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except(rospy.ServiceException) as e:
            print("[LOG] /gazebo/pause_physics service call failed")

        #state = self.discretize_observation(data, 5)
        return state