import os
import sys
import logging
import tensorflow as tf
import gin.tf


@gin.configurable(blacklist=['inputs', 'name', 'reuse'])
def mlp(inputs,
        hidden_units=[32, 8],
        name='mlp',
        reuse=False,
        scale_l2=0.0,
        act_fn=tf.nn.relu):
    regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l2=scale_l2)

    net = inputs
    with tf.variable_scope(name, reuse=reuse):
        for i, n_units in enumerate(hidden_units):
            with tf.variable_scope("hiddenlayer_%d" % i) as hl_scope:
                net = tf.layers.dense(
                    net,
                    n_units,
                    activation=act_fn,
                    #kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    #kernel_regularizer=regularizer,
                )
    return net


def get_net(net_name):
    net_map = {
        'mlp': mlp,
    }
    net_func = net_map.get(net_name, mlp)
    return net_func
