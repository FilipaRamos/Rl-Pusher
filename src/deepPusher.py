#!/usr/bin/env python3
import gym
from gym import wrappers

import sys
import time
import rospy
import numpy as np

from rl import qlearn, deep_qlearn, utils
from functools import reduce

from envs import deepPusherEnv

class DeepPusher():
    '''
        DeepPusher class
        @param objectDetector - class that provides the cylinder detectionstep
        @param state - variable that describes the behaviour that the robot must take
    '''
    def __init__(self, algo, inference):
        self.state = 0
        self.frequency = 10
        self.algo = algo
        self.inference = inference

    def render(self, x, render_skip = 0, render_interval = 50, render_episodes = 10):
        if (x % render_interval == 0) and (x != 0) and (x > render_skip):
            self.env.render()
        elif ((x - render_episodes) % render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
            self.env.render(close=True)

    def setup_log(self):
        import os
        assert os.path.exists(self.outdir)

        p = os.path.join(self.outdir, 'qlearn.batch.txt')
        f = open(p, "w+")
        f.close()

        return p

    def save_log(self, episode, message):
        if episode == 1:
            f = open(self.log, "w")
            f.write(str(message))
            f.write('\n')
        else:
            f = open(self.log, "a")
            f.write(str(message))
            f.write('\n')
        f.close()

    def clear_log(self, log):
        # Clear contents
        open(self.log, "w").close()
        with open(self.log, 'a') as f:
            for act in log:
                f.write("%s\n" % act)

    def setup(self, out_dir='./tmp/experiments'):
        env = gym.make('deepPusher-v0')
        self.outdir = out_dir

        self.env = gym.wrappers.Monitor(env, self.outdir, force=True)

        # TODO: Params
        self.last_time_steps = np.ndarray(0)

        if self.algo == 'qlearn':
            print("[ LOG] deepPusher has chosen algorithm qlearn")
            self.qlearn = qlearn.QLearn(actions=range(self.env.action_space.n),
                    alpha=0.2, gamma=0.8, epsilon=0.9)
            self.setup_qlearn()
            ''' Inference mode '''
            if self.inference is not None:
                self.qlearn.loadTable(self.inference)
                self.qlearn_inference()

        elif self.algo == 'deep-qlearn':
            print("[ LOG] deepPusher has chosen algorithm deep-qlearn")
            self.qlearn = qlearn.DeepQLearn(actions=range(self.env.action_space[0].n))
            self.setup_deep_learn()

        self.start_time = time.time()
        self.total_episodes = 50

        self.highest_reward = 0
        self.prev_cumulated_reward = 0
        
        self.log = self.setup_log()
        print("[ LOG] deepPusher setup done!")

    def setup_qlearn(self):
        self.initial_epsilon = self.qlearn.epsilon
        self.epsilon_discount = 0.9986

    def setup_deep_qlearn(self):
        cfg = {
            'epsilon_start': 1,
            'epsilon_final': 0.1,
            'epsilon_decay': 5000,
            'memory_cap': 5000,
            'batch_size': 64,
            'gamma': 0.99,
            'q_network_lr': 0.0001,
            'parameter_lr': 0.0002,
            'q_net_units': [128, 128],
            'parameter_units': [128, 128]
        }
        self.memory = deep_qlearn.Memory(cfg['memory_cap'])

    def cycle(self):
        if self.algo == 'qlearn':
            self.cycle_qlearn()
        elif self.algo == 'deep_qlearn':
            self.cycle_deep_qlearn()

    def cycle_qlearn(self):
        print("[ LOG] Initiating cycle!")

        for x in range(self.total_episodes):
            cumulated_reward = 0
            action_log = []

            observation = self.env.reset()
            if x % 20 == 0:
                # Spawn random cyl and goal
                self.env.spawn_random(cyl=True, goal=True)

            if self.qlearn.epsilon > 0.05:
                self.qlearn.epsilon *= self.epsilon_discount
            
            state = ''.join(map(str, observation))

            steps = 0
            while True:
                # Pick action based on current state
                action = self.qlearn.chooseAction(state)

                # Execute action and receive the reward
                observation, reward, done, info = self.env.step(action)
                cumulated_reward += reward

                if self.highest_reward < cumulated_reward:
                    self.highest_reward = cumulated_reward

                nextState = ''.join(map(str, observation))
                self.qlearn.learn(state, nextState, action, reward)

                self.env._flush(force=True)
                steps += 1
                action_log.append(action)

                if not(done):
                    state = nextState
                else:
                    self.last_time_steps = np.append(self.last_time_steps, [int(steps + 1)])
                    break
            
            self.qlearn.saveQTable()
            m, s = divmod(int(time.time() - self.start_time), 60)
            h, m = divmod(m, 60)
            
            print("Epoch: " + str(x + 1) + "- [alpha: " + str(round(self.qlearn.alpha,2)) + 
            " - gamma: " + str(round(self.qlearn.gamma,2)) + " - epsilon: " + str(round(self.qlearn.epsilon,2)) + 
            "] - Reward: " + str(cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s))

            if cumulated_reward > self.prev_cumulated_reward:
                self.clear_log(action_log)
            self.prev_cumulated_reward = cumulated_reward

        print ("\n| " + str(self.total_episodes) + " | " + str(self.qlearn.alpha) + " | " + str(self.qlearn.gamma) + " | " + str(self.initial_epsilon) +
                    " * " + str(self.epsilon_discount) + " | " + str(self.highest_reward) + "| QLEARN |")
        l = self.last_time_steps.tolist()
        l.sort()

        print("Overall score: {:0.2f}".format(self.last_time_steps.mean()))
        print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
        print('[ LOG] Initializing another run with random spawns.')

        self.env.close()

    def cycle_deep_qlearn(self):
        from itertools import count
        from collections import deque
        import matplotlib as plt
        print("[ LOG] Initiating cycle!")
        cumulated_reward = 0
        action_log = []

        recent_scores = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        recent_losses = deque(maxlen=100)
        graph = []

        for x in range(self.total_episodes):
            state = self.env.reset()

            score, i, epsilon = 0, 0, 0
            for i in count():
                epsilon = utils.get_epsilon(self.cfg, self.step)
                action = self.agent(state, epsilon)
                next_state, reward, done, _ = self.env.step(action.get())
                self.memory.push(deep_qlearn.Experience(state, action, reward, next_state, done))

                self.step += 1
                score += reward
                state = next_state

                if len(self.memory.buffer) > self.batch_size:
                    loss = self.learn()
                    recent_losses.append(loss)

                if self.step % 100 == 0:
                    graph.append(np.mean(recent_scores))

                if done:
                    if reward > 0:
                        print('Yay !')
                    break

            if self.step > self.total_episodes:
                break

            recent_lengths.append(i)
            recent_scores.append(score)
            print(self.step, x, np.mean(recent_scores), np.mean(recent_lengths), np.mean(recent_losses), epsilon)
        plt.plot(graph)
        plt.show()

    def qlearn_inference(self):
        cumulated_reward = 0
        action_log = []

        observation = self.env.reset()
        state = ''.join(map(str, observation))

        steps = 0
        start = time.time()
        while True:
            # Pick action based on current state
            action, q = self.qlearn.chooseActionInf(state, return_q=True)

            # Execute action and receive the reward
            observation, reward, done, info = self.env.step(action)
            cumulated_reward += reward

            nextState = ''.join(map(str, observation))
            self.env._flush(force=True)

            steps += 1
            action_log.append(action)
            if not(done):
                state = nextState
            else:
                self.last_time_steps = np.append(self.last_time_steps, [int(steps + 1)])
                break
        end = time.time()
        print("[ LOG] Robot done...!")
        print('[ RESULT] Robot took {} seconds | {} steps | with cumulated reward {}'.format(end - start, steps, cumulated_reward))

        rospy.sleep(30.)
        self.env.close()

if __name__ == '__main__':
    '''
    Main DeepPusher Cycle
    @params: [algorithm] (qlearn/deep-qlearn) [qtable_filename] 
    '''
    # Sleep in order to give Gazebo time to setup
    rospy.sleep(2.0)
    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            dp = DeepPusher(sys.argv[1], sys.argv[2])
        else:
            dp = DeepPusher(sys.argv[1], None)
    else:
        dp = DeepPusher('qlearn', None)
    dp.setup()
    dp.cycle()
    print("[ LOG] Reached the end... Closing!")