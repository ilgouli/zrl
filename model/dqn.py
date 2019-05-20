import os
import sys
import logging
from glob import glob
import gym
import numpy as np
import tensorflow as tf
import gin.tf
from model import Model
from memory import Memory, Transition
from net import get_net


class DQNModel(Model):
    def __init__(self, **kwargs):
        super(DQNModel, self).__init__(**kwargs)
        assert isinstance(self._action_space, gym.spaces.Discrete)
        assert isinstance(self._state_space, gym.spaces.Box)
        self._action_size = self._action_space.n

        self._batch_size = kwargs.get('batch_size', 50)
        self._lr = kwargs.get('lr', 0.001)
        self._optimizer = kwargs.get('optimizer', 'SGD')
        self._net_name = kwargs.get('net_name', 'mlp')
        self._layer_size = [3]
        self._epsilon = kwargs.get('epsilon', 0.1)
        self._target_update_period = kwargs.get('target_update_period', 5)

        self._memory = Memory(**kwargs)
        self._ckpt_path = os.path.join(self._model_path, 'model.ckpt')
        self._sum_path = os.path.join(self._model_path, 'summary')
        logging.info("DQNModel, gamma[%.3f], net[%s]", self._gamma,
                     self._net_name)
        logging.info("DQNModel Train, optimizer[%s], batch_size[%d], lr[%.3f]",
                     self._optimizer, self._batch_size, self._lr)

    def init(self):
        self.init_net_input()
        self.build_qnet()
        self.build_train_op()
        self.build_sync_ops()
        self.build_summary()

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self._saver = tf.train.Saver(max_to_keep=5)
        self._writer = tf.summary.FileWriter(self._sum_path, self._sess.graph)

    def init_net_input(self):
        state_shape = (None, ) + self._state_space.shape
        self._state = tf.placeholder(
            tf.float32, shape=state_shape, name='state')
        self._state_next = tf.placeholder(
            tf.float32, shape=state_shape, name='state_next')
        self._action = tf.placeholder(tf.int32, shape=(None, ), name='action')
        self._action_next = tf.placeholder(
            tf.int32, shape=(None, ), name='action_next')
        self._reward = tf.placeholder(
            tf.float32, shape=(None, ), name='reward')
        self._done = tf.placeholder(tf.float32, shape=(None, ), name='done')

    def build_qnet(self):
        net_func = get_net(self._net_name)
        hidden_units = self._layer_size + [self._action_size]
        self._q = net_func(
            self._state, hidden_units=hidden_units, name='Online')
        self._q_next = net_func(
            self._state_next, hidden_units=hidden_units, name='Target')

    def build_train_op(self):
        self._action_select_by_q = tf.argmax(
            self._q, axis=-1, name='action_select')

        batch_indices = tf.constant(
            np.arange(self._batch_size), dtype=tf.int32)
        self._action_indices = tf.stack([batch_indices, self._action], axis=1)
        self._q_select = tf.gather_nd(self._q, self._action_indices)

        self._max_q_next = tf.reduce_max(
            self._q_next, axis=-1, name="max_q_next")
        self._y = self._reward + (
            1 - self._done) * self._gamma * self._max_q_next
        self._loss = tf.nn.l2_loss(
            self._q_select - tf.stop_gradient(self._y), name='loss_mse')

        self._step = tf.train.get_or_create_global_step()
        self._train_op = tf.contrib.layers.optimize_loss(
            loss=self._loss,
            learning_rate=self._lr,
            optimizer=self._optimizer,
            summaries=[
                'loss', 'gradient_norm', 'global_gradient_norm',
                'learning_rate'
            ],
            global_step=self._step)
        #print_op = tf.print('max_q_next: ', self._max_q_next,
        #        'y: ', self._y, 'q: ', self._q)

    def build_sync_ops(self):
        self._sync_qt_ops = []
        trainables_online = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Online')
        trainables_target = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target')
        for (w_online, w_target) in zip(trainables_online, trainables_target):
            self._sync_qt_ops.append(
                w_target.assign(w_online, use_locking=True))

    def build_summary(self):
        with tf.variable_scope('summary'):
            avg_q = tf.reduce_mean(self._q, 0)
            for idx in range(self._action_size):
                tf.summary.histogram('q/%s' % idx, avg_q[idx])

            tf.summary.histogram("batch/y", self._y)
            tf.summary.histogram("batch/q_select", self._q_select)
            #tf.summary.scalar("loss", self._loss)
            self._sum_all = tf.summary.merge_all()

    def update(self, state, action, reward, state_next, done):
        record = Transition(state, action, reward, state_next, done)
        self._memory.add(record)
        if len(self._memory) >= self._batch_size:
            self.train_step()

    def train_step(self):
        batch = self._memory.sample_batch(self._batch_size)
        feed_dict = {
            self._state: batch['state'],
            self._state_next: batch['state_next'],
            self._action: batch['action'],
            self._reward: batch['reward'],
            self._done: batch['done'],
        }
        _, sum_all, step = self._sess.run(
            [self._train_op, self._sum_all, self._step], feed_dict=feed_dict)
        self._writer.add_summary(sum_all, step)

        if step % self._target_update_period == 0:
            logging.debug("Update Target params, step[%d], period[%d]", step,
                          self._target_update_period)
            self._sess.run(self._sync_qt_ops)

    def select_action(self, state):
        if self._epsilon > np.random.rand():
            action = self._action_space.sample()
        else:
            action = self.select_model_action(state)
        return action

    def select_model_action(self, state):
        feed_dict = {self._state: [state]}
        action = self._sess.run(self._action_select_by_q, feed_dict=feed_dict)
        action = action[0]
        return action

    def save(self):
        self._saver.save(self._sess, self._ckpt_path)

    def load(self):
        if len(glob(self._ckpt_path + '*')) > 0:
            logging.info("Load checkpoint from path[%s]", self._ckpt_path)
            self._saver.restore(self._sess, self._ckpt_path)
