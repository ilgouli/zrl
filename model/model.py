import os
import sys
import logging
import numpy as np


class Model(object):
    def __init__(self, **kwargs):
        self._gamma = kwargs.get('gamma', 0.95)
        self._action_space = kwargs.get('action_space')
        self._state_space = kwargs.get('state_space')

    def init(self):
        pass

    def select_action(self, state):
        return self._action_space.sample()

    def update(self, state, action, reward, state_next, done):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

