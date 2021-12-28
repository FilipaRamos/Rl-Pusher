#!/usr/bin/env python3
import gym
from gym import wrappers

import time
import numpy as np

from rl import qlearn
from functools import reduce

from envs import deepPusherEnv

class Cylinder():
    def __init__(self, points):
        self.points = points

    def set_points(self, points_n):
        self.points = points_n

class DeepPusher():
    '''
        DeepPusher class
        @param objectDetector - class that provides the cylinder detection
        @param state - variable that describes the behaviour that the robot must take

        State 0 - Init state where we must search for a cylinder
        State 1 - Registered a cylinder
    '''
    def __init__(self, objectDetector):
        self.state = 0
        self.frequency = 10
        self.objDet = objectDetector
        self.cylinder = Cylinder([])

    def registry_manager(self, counter):
        if counter > 0 and counter % self.frequency == 0:
            self.register_cylinder()
            self.state = 0
    
    def register_cylinder(self):
        points, cluster_size = self.objDet.find_cylinder(plot=True)
        if points.size > 3:
            # TODO: nr img points?
            if points.size >= cluster_size - 1:
                self.cylinder.set_points(points)
                print("[LOG] Registered cylinder", self.cylinder.points)
            else:
                print("[LOG] Not registering the cylinder as there were less points than expected for the cluster size.")
        else: print("[LOG] Not registering the cylinder as there were not enough points for assessment.")

    def render(self, x, render_skip=0, render_interval=50, render_episodes=10):
        if (x % render_interval == 0) and (x != 0) and (x > render_skip):
            self.env.render()
        elif ((x - render_episodes) % render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
            self.env.render(close=True)

    def setup(self, out_dir='../tmp/experiments'):
        env = gym.make('deepPusher-v0')
        self.outdir = out_dir

        self.env = gym.wrappers.Monitor(env, self.outdir, force=True)
        #self.plotter = liveplot.LivePlot(self.outdir)

        # TODO: Params
        self.last_time_steps = np.ndarray(0)
        self.qlearn = qlearn.QLearn(actions=range(self.env.action_space.n),
                    alpha=0.2, gamma=0.8, epsilon=0.9)

        self.initial_epsilon = self.qlearn.epsilon
        self.epsilon_discount = 0.9986

        self.start_time = time.time()
        self.total_episodes = 10000

        self.highest_reward = 0

    def cycle(self):
        for x in range(self.total_episodes):
            done = False
            cumulated_reward = 0
            observation = self.env.reset()

            if self.qlearn.epsilon > 0.05:
                self.qlearn.epsilon *= self.epsilon_discount
            
            self.render()
            state = ''.join(map(str, observation))

            for i in range(1500):
                # Pick action based on current state
                action = self.qlearn.chooseAction(state)

                # Execute action and receive the reward
                observation, reward, done, info = self.env.step(action)
                cumulated_reward += reward

                if self.highest_reward < cumulated_reward:
                    self.highest_reward = cumulated_reward

                nextState = ''.join(map(str, observation))
                self.qlearn.learn(state, action, reward, nextState)

                self.env._flush(force=True)

                if not(done):
                    state = nextState
                else:
                    self.last_time_steps = np.append(self.last_time_steps, [int(i + 1)])
                    break

            if x % 100 == 0:
                print(self.env)
                #self.plotter.plot(self.env)

            m, s = divmod(int(time.time() - self.start_time), 60)
            h, m = divmod(m, 60)
            print("Epoch: " + str(x + 1) + "- [alpha: " + str(round(qlearn.alpha,2)) + 
            " - gamma: " + str(round(qlearn.gamma,2)) + " - epsilon: " + str(round(qlearn.epsilon,2)) + 
            "] - Reward: " + str(cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s))

        print ("\n|" + str(self.total_episodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(self.initial_epsilon) +
                    "*" + str(self.epsilon_discount) + "|" + str(self.highest_reward) + "| PICTURE |")

        l = self.last_time_steps.tolist()
        l.sort()

        #print("Parameters: a="+str)
        print("Overall score: {:0.2f}".format(self.last_time_steps.mean()))
        print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

        self.env.close()

if __name__ == '__main__':
    from objectDetector import ObjectDetector
    det = ObjectDetector()
    dp = DeepPusher(det)
    dp.setup()
    dp.cycle()
    print("Cycle done...")
