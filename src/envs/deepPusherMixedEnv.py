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
        self.action_cfg = self.load_config(ros_path + "/config/actions_mixed.config")['action_space']
        self.actions = self.action_cfg['actions']
        self.parameters = self.action_cfg['parameters']

        rewards_cfg = self.load_config(ros_path + "/config/rewards.config")['rewards']
        self.obs_idx_r, self.obs_idx_p, self.rewards, self.penalties, self.r_scale_factor = rewards_cfg['obs_index_r'], rewards_cfg['obs_index_p'], \
                                                                       rewards_cfg['rewards'], rewards_cfg['penalties'], rewards_cfg['scale_factor']

        # Class that handles robot observations
        self.observer = Observer()
        self.observer.observe()
        # Class that handles all navigation
        self.navigator = Navigator()

        # Actions are loaded from config
        parameters_min = np.array([0, -1])
        parameters_max = np.array([1, +1])
        self.action_space = spaces.Tuple((
                        spaces.Discrete(len(self.actions)),
                        spaces.Box(parameters_min, parameters_max)
                    ))
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
        dist = dist - self.sim['target_cyl']['radius']
        if dist <= r + 0.05:
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
        
    def ori_align(self, state):
        # Is this correct?
        import math
        import transforms3d as td3
        _, pos_c, pos_g, pos_r, _ = state

        #ori_diff = math.atan2(pos_r[2] - pos_g[2], pos_r[0] - pos_g[0])
        #print("=-------------- Alignment = ", ori_diff)
        # Check slopes between robot(x1, y1) - cyl(x2, y2) / cyl - goal(x3, y3) are close, if they are, robot is aligned with both which is what we want
        # ((y1 - y2) * (x1 - x3) - (y1 - y3) * (x1 - x2)) <= 1e-9
        ori_diff = ((pos_r[1] - pos_c[1]) * (pos_r[0] - pos_g[0]) - (pos_r[1] - pos_g[1]) * (pos_r[0] - pos_c[0]))
        return abs(ori_diff)

    def transform_coordinates(self, length, height, x_coord, y_coord):
        import math
        bucket_cap_l = int(length / self.sim['length'])
        bucket_cap_h = int(height / self.sim['width'])

        x_bucket = int(x_coord)
        y_bucket = int(y_coord)
        
        x_idx = math.modf(x_coord)[0] * bucket_cap_l
        y_idx = math.modf(y_coord)[0] * bucket_cap_h

        x_img_coord = x_bucket * bucket_cap_l + int(x_idx)
        y_img_coord = y_bucket * bucket_cap_h + int(y_idx)

        return x_img_coord, y_img_coord

    def fill_obs(self, image, x, y, radius):
        radius = int(radius)
        i_x_b = (x - radius) if (x -radius) in range(0, image.shape[0] + 1) else 0
        i_x_a = (x + radius +1) if (x + radius + 1) in range(0, image.shape[0] + 1) else image.shape[0]
        i_y_b = (y - radius) if (y -radius) in range(0, image.shape[1] + 1) else 0
        i_y_a = (y + radius +1) if (y + radius + 1) in range(0, image.shape[1] + 1) else image.shape[1]
        
        i_x_b, i_x_a = int(i_x_b), int(i_x_a)
        i_y_b, i_y_a = int(i_y_b), int(i_y_a)

        image[i_x_b:i_x_a, i_y_b:i_y_a] = 1
        return image

    def generate_obs_img(self, pos_c, pos_g, pos_r):
        # Get world img sizes
        length = int(self.sim['length'] / self.obs['precision'])
        height = int(self.sim['width'] / self.obs['precision'])
        image = np.zeros((length + 1, height + 1), dtype=np.float16)

        # Calculate coordinates for the poses on the image
        x_c, y_c = self.transform_coordinates(length, height, pos_c[0], pos_c[1])
        x_g, y_g = self.transform_coordinates(length, height, pos_g[0], pos_g[1])
        x_r, y_r = self.transform_coordinates(length, height, pos_r[0], pos_r[1])
        
        # Fill in observations
        image = self.fill_obs(image, x_c, y_c, self.sim['target_cyl']['radius'] / self.obs['precision'])
        image = self.fill_obs(image, x_g, y_g, self.sim['goal']['radius'] / self.obs['precision'])
        image = self.fill_obs(image, x_r, y_r, self.sim['robot']['radius'] / self.obs['precision'])

        # rotate over y and over x - np.flip(np.flip(a, 0), 1)
        return np.flip(np.flip(image, 0), 1)

    def check_pose_stuck(self, obs_pose):
        if round(obs_pose[0], self.obs['places']) == round(self.previous_robot_pose[0], self.obs['places']) \
            and round(obs_pose[1], self.obs['places']) == round(self.previous_robot_pose[1], self.obs['places']):
            return True
        return False

    def discretise_observation(self, state):
        _, pose_target_cyl, pose_goal, pose_robot, _ = state
        return self.generate_obs_img(pose_target_cyl, pose_goal, pose_robot)

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
                    obs = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=1)
                    rospy.sleep(0.5)
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
        reward_cyl_flag = (self.last_cyl_dist > self.sim['target_cyl']['radius'] + 0.05)
        reward += (self.last_cyl_dist - dist_cyl) * self.rewards[self.obs_idx_r['at_target_cyl']] * reward_cyl_flag
        self.last_cyl_dist = dist_cyl

        # Alignment with cyl and goal reward
        reward -= self.r_scale_factor * self.ori_align(state)

        # Penalise for time step
        reward -= self.r_scale_factor * self.penalties[self.obs_idx_p['step']]
        
        # DEPRECATING for now
        # Rewards are on a too small scale
        # reward = reward*self.r_scale_factor
        
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

        vel, rot = 0, 0
        if action == self.actions['stop']:
            self.navigator.stop()
        elif action == self.actions['move_forward']:
            vel = self.parameters['max_vel'] * max(min(action.parameter, 1), 0)
            self.navigator.move_forward(vel)
        elif action == self.actions['turn']:
            rot = self.parameters['max_turn'] * max(min(action.parameter, 1), -1)
            self.navigator.turn(rot)

        # Observe before pausing since our observations depend on Gazebo clock being published
        data = self.observe()
        state = self.discretise_observation(data)
        from PIL import Image
        im = Image.fromarray((state * 255).astype(np.uint8))
        im.save("resources/state.png")
        
        if self.steps > 0:
            if self.check_pose_stuck(data[3]):
                self.robot_stuck += 1
            else:
                self.robot_stuck = 0
        self.previous_robot_pose = data[3]

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("[LOG] /gazebo/pause_physics service call failed")

        done = False
        reward = self.reward(data)
        #if self.observer.cyl.current_pos[0] != 0 and self.observer.cyl.current_pos[1] != 0 and self.observer.cyl.current_pos[2] != 0:
        if self.at_goal(data):
            #reward += self.r_scale_factor * self.rewards[self.obs_idx_r['at_goal']]
            reward += self.rewards[self.obs_idx_r['at_goal']]
            done = True
            print("[ENV] Target cylinder is at goal!")
        if self.robot_stuck > 6:
            done = True
            self.robot_stuck = 0
            #reward -= self.r_scale_factor * self.penalties[self.obs_idx_p['robot_stuck']]
            reward -= self.penalties[self.obs_idx_p['robot_stuck']]
            print("[ENV] Robot has not altered its position for 6 consecutive time steps.")

        self.steps += 1
        if self.steps == self.max_steps:
            reward -= self.penalties[self.obs_idx_p['max_steps']]
            done = True
            print("[ENV] Steps taken are over the maximum allowed threshold.")
        
        print('REWARD>', reward)

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