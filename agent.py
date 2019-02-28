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

        self._data_path = kwargs.get('data_path')
        self._model_name = kwargs.pop('model_name', 'QLearning')
        self._model_path = os.path.join(self._data_path,
                                        'model_{}'.format(self._model_name))
        self._model = get_model(
            self._model_name,
            model_path=self._model_path,
            action_space=self._action_space,
            state_space=self._state_space,
            **kwargs)

    def init(self):
        self._model.init()

    def select_action(self, state):
        return self._model.select_action(state)

    def update(self, state, action, reward, state_next, done):
        self._model.update(state, action, reward, state_next, done)

    def save(self):
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)
        logging.info("Save Agent Model to path[%s]", self._model_path)
        self._model.save()

    def load(self):
        if os.path.exists(self._model_path):
            logging.info("Load Agent Model from path[%s]", self._model_path)
            self._model.load()
