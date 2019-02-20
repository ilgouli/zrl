import os
import sys
import logging
import click
import gym
from agent import *


class Runner(object):
    def __init__(self, **kwargs):
        self._env_name = kwargs.get('env_name')
        self._max_episodes = kwargs.get('max_episodes')
        self._max_steps = kwargs.get('max_steps')
        self._render = kwargs.get('render')
        self._bad_end = kwargs.get('bad_end')
        self._path = os.path.join('./data', '{}_agent'.format(self._env_name))
        self._kwargs = kwargs

        self._show = kwargs.get('show')
        if self._show:
            self._render = 1
            self._max_episodes = 1
            self._kwargs['epsilon'] = 0

    def init(self):
        self._env = gym.make(self._env_name)
        self._agent = Agent(
            self._env,
            limit_min=-5,
            limit_max=5,
            n_bins=50,
            **self._kwargs)
        self._agent.init()
        if os.path.exists(self._path):
            self._agent.load(self._path)

    def run(self):
        self.init()
        for i in range(self._max_episodes):
            reward, steps = self.run_episode()
            logging.info("episode[%d], steps[%d], last reward[%d]", i, steps,
                         reward)

        if not self._show:
            self._agent.save(self._path)

    def run_episode(self):
        state = self._env.reset()
        reward = -1

        for i in range(self._max_steps):
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
        return reward, i

    def show(self):
        pass


############ run
@click.command()
@click.option('--env_name', default='CartPole-v0')
@click.option('--max_episodes', default=100)
@click.option('--max_steps', default=200)
@click.option('--render', default=0)
@click.option('--show', default=0)
@click.option('--bad_end', default=1)
def main(**kwargs):
    runner = Runner(**kwargs)
    runner.run()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")
    main()
