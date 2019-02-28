import os
import sys
import logging
import gym
import numpy as np
from model import Model


class QLearningModel(Model):
    def __init__(self, **kwargs):
        super(QLearningModel, self).__init__(**kwargs)
        self._limit_min = kwargs.get('limit_min', -5.0)
        self._limit_max = kwargs.get('limit_max', 5.0)
        self._n_bins = kwargs.get('n_bins', 50)
        self._action_bins = None
        self._state_bins = None
        self._epsilon = kwargs.get('epsilon', 0.1)
        self._lr = kwargs.get('lr', 0.5)

        self._model_file = os.path.join(self._model_path, 'model.npz')
        logging.info("QLearningModel, epsilon[%.3f], lr[%.3f], gamma[%.3f]",
                     self._epsilon, self._lr, self._gamma)

    def init(self):
        if isinstance(self._state_space, gym.spaces.Box):
            self._state_bins = self.get_bins(self._state_space.low,
                                             self._state_space.high)
            self._state_scales = [
                self._n_bins**i for i in range(len(self._state_bins))
            ]
            self._state_size = self._n_bins**len(self._state_bins)
        else:
            self._state_size = self._state_space.n

        if isinstance(self._action_space, gym.spaces.Box):
            self._action_bins = self.get_bins(self._action_space.low,
                                              self._action_space.high)
            self._action_size = self._n_bins**len(self._action_size)
        else:
            self._action_size = self._action_space.n

        self._q_table = np.random.uniform(
            low=0, high=0.5, size=[self._state_size, self._action_size])

        logging.info("Init QTable[%s], state_size[%d], action_size[%d]",
                     self._q_table.shape, self._state_size, self._action_size)

    def get_bins(self, low, high):
        low = [
            self._limit_min if v < self._limit_min else v
            for v in low.flatten()
        ]
        high = [
            self._limit_max if v > self._limit_max else v
            for v in high.flatten()
        ]
        bins = [np.linspace(l, h, self._n_bins - 1) for l, h in zip(low, high)]
        return bins

    def get_state_index(self, state):
        if self._state_bins is not None:
            state_digits = [
                np.digitize(s, bin, True)
                for s, bin in zip(state.flatten(), self._state_bins)
            ]
            state_index = sum(
                digit * scale
                for digit, scale in zip(state_digits, self._state_scales))
        else:
            state_index = state
        return state_index

    def update(self, state, action, reward, state_next, done):
        state_index = self.get_state_index(state)
        state_next_index = self.get_state_index(state_next)

        q = self._q_table[state_index, action]
        if done:
            max_q_next = 0.0
        else:
            max_q_next = np.amax(self._q_table[state_next_index, :])
        loss = (reward + self._gamma * max_q_next - q)

        self._q_table[state_index, action] = q + self._lr * loss
        logging.debug(
            "Update Q_table[%d, %d], origin[%.3f], loss[%.3f], "
            "state_next[%d], reward[%d]", state_index, action, q, loss,
            state_next_index, reward)

    def select_action(self, state):
        if self._epsilon > np.random.rand():
            action = self._action_space.sample()
        else:
            action = self.select_model_action(state)
        return action

    def select_model_action(self, state):
        state_index = self.get_state_index(state)
        action = np.argmax(self._q_table[state_index, :])
        return action

    def save(self):
        logging.info("Save QAgent model to path[%s]", self._model_file)
        np.savez(self._model_file, q_table=self._q_table)

    def load(self):
        result = np.load(self._model_file)
        self._q_table = result['q_table']
        logging.info("Load q_table[%s] from path[%s]", self._q_table.shape,
                     self._model_file)


