import numpy as np


class Agent:

    def __init__(self, env, alpha=0.1, epsilon=0.1, decay=0.999, gamma=0.99):
        self.alpha = alpha
        self.width = env.width
        self.epsilon = epsilon
        self.decay = decay
        self.gamma = gamma
        self.Q_value = np.ones((self.width, self.width, 4)) * 0.5

    def choose_action(self, state):
        self.epsilon = self.epsilon * self.decay
        if np.random.rand() < self.epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(self.Q_value[state % self.width][state // self.width])
        return action

    def get_q(self, state):
        return self.Q_value[state % self.width][state // self.width]

    def update_policy(self, state, action, reward, next_s, next_a, done):
        if done:
            delta = reward - self.get_q(state)[action]
        else:
            delta = reward + self.gamma * self.get_q(next_s)[next_a] - self.get_q(state)[action]
        self.Q_value[state % self.width][state // self.width][action] += self.alpha * delta
        return delta ** 2

