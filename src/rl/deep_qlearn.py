import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import random
import numpy as np
from typing import Optional, List, Tuple
from itertools import count
from more_itertools import pairwise
from collections import deque, namedtuple

import matplotlib.pyplot as plt

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ActionTuple:
    def __init__(self, action, params):
        self.action = action
        self.params = params

    def get(self):
        return (self.action, self.params)

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        return Experience(*zip(*experiences))

    def __len__(self):
        return len(self.buffer)

class DeepQLearn:
    def __init__(self, cfg, actions, params):
        self.actions = actions
        self.params = params

        self.cfg = cfg

class ParameterNetwork(nn.Module):
    def __init__(self, state_space, parameters_space, hidden_units: Optional[list] = None):
        super(ParameterNetwork, self).__init__()

        hidden_units = hidden_units or [128, 128]
        units = [state_space.shape[0]] + hidden_units
        self.layers = nn.ModuleList()
        for i, o in pairwise(units):
            self.layers.append(nn.Linear(i, o))

        self.parameter_outputs = nn.ModuleList()
        for low, high in parameters_space.low, parameters_space.high:
            self.parameter_outputs.append(ParameterOutput(units[-1], 1, low=low, high=high))

    def forward(self, x, action_id: Optional[int] = None) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        if action_id:
            return self.parameter_outputs[action_id](x)
        output = []
        for parameter_output in self.parameter_outputs:
            output.append(parameter_output(x))
        return torch.cat(output, 1)

class ParameterOutput(nn.Module):
    def __init__(self, in_features, out_features, low=-1, high=1):
        super(ParameterOutput, self).__init__()
        self.low = low
        self.high = high
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.softmax(self.linear(x))
        return (self.high - self.low) * x + self.low

class QNetwork(nn.Module):
    def __init__(self, state_space, action_space, parameters_space, hidden_units: Optional[list] = None):
        super(QNetwork, self).__init__()

        hidden_units = hidden_units or [128, 128]
        units = [state_space.shape[0] + parameters_space.shape[0]] + hidden_units
        self.layers = nn.ModuleList()
        for i, o in pairwise(units):
            self.layers.append(nn.Linear(i, o))
        self.fc_value = nn.Linear(hidden_units[-1], action_space.n)
        self.fc_advantage = nn.Linear(hidden_units[-1], action_space.n)

    def forward(self, state, parameter) -> torch.Tensor:
        x = torch.cat((state, parameter), 1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class Agent:
    def __init__(self, q_network: QNetwork, parameter_network: ParameterNetwork, action_space, parameters_space):
        self.q_network = q_network
        self.parameter_network = parameter_network
        self.actions = range(action_space.n)
        self.parameters_space = parameters_space

    def __call__(self, observation_raw, epsilon=0.):
        if random.random() > epsilon:
            with torch.no_grad():
                observation = torch.tensor(observation_raw, dtype=torch.float32).unsqueeze(0)
                parameters = self.parameter_network.forward(observation)
                action = self.q_network.forward(observation, parameters).max(1)[1].item()
                return Action(action, parameters.tolist()[0])
        else:
            action = random.choice(self.actions)
            parameters = np.random.uniform(self.parameters_space.low, self.parameters_space.high)
            return Action(action, list(parameters))


class DeepQLearning:
    def __init__(self, agent: Agent, q_network_optimizer, parameter_network_optimizer, memory_capacity, batch_size):
        self.step = 0
        self.agent = agent
        self.batch_size = batch_size
        self.memory = Memory(memory_capacity)
        self.q_network_optimizer = q_network_optimizer
        self.parameter_network_optimizer = parameter_network_optimizer

    def learn(self):
        batch = self.memory.sample(self.batch_size)

        done = torch.FloatTensor(batch.done)
        reward = torch.FloatTensor(batch.reward)
        state = torch.FloatTensor(np.float32(batch.state))
        next_state = torch.FloatTensor(np.float32(batch.next_state))

        action = torch.LongTensor([action.id for action in batch.action])
        parameters = torch.LongTensor(np.squeeze([action.parameters for action in batch.action]))

        q_values = self.agent.q_network.forward(state, parameters)

        next_parameters = self.agent.parameter_network.forward(next_state)
        next_q_values = self.agent.q_network.forward(next_state, next_parameters)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + GAMMA * next_q_value * (1 - done)
        q_network_loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.q_network_optimizer.zero_grad()
        q_network_loss.backward()
        self.q_network_optimizer.step()

        predicted_parameters = self.agent.parameter_network.forward(state)
        parameter_network_loss = - torch.sum(self.agent.q_network.forward(state, predicted_parameters))

        self.parameter_network_optimizer.zero_grad()
        parameter_network_loss.backward()
        self.parameter_network_optimizer.step()

        return q_network_loss.item()


if __name__ == '__main__':
    q_network = QNetwork(OBSERVATION_SPACE, ACTION_SPACE, PARAMETERS_SPACE, Q_NETWORK_UNITS)
    q_network_optimizer = optim.Adam(q_network.parameters(), Q_NETWORK_LR)

    parameter_network = ParameterNetwork(OBSERVATION_SPACE, PARAMETERS_SPACE, PARAMETER_NETWORK_UNITS)
    parameter_network_optimizer = optim.Adam(parameter_network.parameters(), PARAMETER_NETWORK_LR)

    agent = Agent(q_network, parameter_network, ACTION_SPACE, PARAMETERS_SPACE)
    trainer = DeepQLearning(agent, q_network_optimizer, parameter_network_optimizer, MEMORY_CAPACITY, BATCH_SIZE)
    trainer.train(100000)