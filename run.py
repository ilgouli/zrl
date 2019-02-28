import os
import sys
import logging
import click
import gym
from agent import *


class Runner(object):
    def __init__(self, **kwargs):
        self._env_name = kwargs.get('env_name')
        self._max_iters = kwargs.get('max_iters')
        self._max_episodes = kwargs.get('max_episodes')
        self._max_steps = kwargs.get('max_steps')
        self._render = kwargs.get('render')
        self._bad_end = kwargs.get('bad_end')
        self._data_path = os.path.join('./data', self._env_name)
        self._kwargs = kwargs

        self._show = kwargs.get('show')
        if self._show:
            self._render = 1
            self._max_episodes = 1
            self._kwargs['epsilon'] = 0

    def init(self):
        self._env = gym.make(self._env_name)
        self._agent = Agent(
            self._env, data_path=self._data_path, **self._kwargs)
        self._agent.init()
        self._agent.load()

    def run(self):
        self.init()
        for n_iter in range(self._max_iters):
            self.run_one_iter(n_iter)

    def run_one_iter(self, n_iter=0):
        logging.info("Start iteration[%d]", n_iter)
        for n_epi in range(self._max_episodes):
            reward, steps = self.run_one_episode(n_epi)

        if not self._show:
            self._agent.save()

    def run_one_episode(self, n_epi=0):
        state = self._env.reset()
        reward = -1

        for step in range(self._max_steps):
            action = self._agent.select_action(state)
            state_next, reward, done, info = self._env.step(action)
            if self._bad_end and done:
                reward = 0.0
            self._agent.update(state, action, reward, state_next, done)
            state = state_next

            if self._render:
                self._env.render()
            if done:
                break

        logging.info("episode[%d], steps[%d], last reward[%d]", n_epi, step,
                     reward)
        return reward, step


############ run
@click.command()
@click.option('--env_name', default='CartPole-v0')
@click.option('--max_iters', default=1)
@click.option('--max_episodes', default=10)
@click.option('--max_steps', default=200)
@click.option('--render', default=0)
@click.option('--show', default=0)
@click.option('--bad_end', default=1)
@click.option('--model_name', default='QLearning')
def main(**kwargs):
    runner = Runner(**kwargs)
    runner.run()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")
    main()
