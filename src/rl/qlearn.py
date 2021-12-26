import random

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        # Q(s, a) += alpha * (reward(s, a) + max(Q(s') - Q(s, a)))
        old_value = self.q.get((state, action), None)
        if old_value is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = old_value + self.alpha * (value - old_value)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))

            # random values to all actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # When there are many state-action max values, select a random one
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:
            return action, q
        return action

    def learn(self, s1, s2, a1, reward):
        maxQ_ = max([self.getQ(s2, a) for a in self.actions])
        self.learnQ(s1, a1, reward, reward + self.gamma * maxQ_)