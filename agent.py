import os
import sys
import logging
import gym
import numpy as np
from model import get_model


class Agent(object):
    def __init__(self, env=None, **kwargs):
        self._env = env
        self._action_space = None
        self._state_space = None
        if env is not None:
            self._action_space = env.action_space
            self._state_space = env.observation_space

        self._model_name = kwargs.get('model_name', 'QLearning')
        self._model = get_model(
            self._model_name,
            action_space=self._action_space,
            state_space=self._state_space,
            **kwargs)

    def init(self):
        self._model.init()

    def select_action(self, state):
        return self._model.select_action(state)

    def update(self, state, action, reward, state_next, done):
        self._model.update(state, action, reward, state_next, done)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        logging.info("Save Agent to path[%s]", path)
        self._model.save(path)

    def load(self, path):
        logging.info("Load Agent from path[%s]", path)
        self._model.load(path)
