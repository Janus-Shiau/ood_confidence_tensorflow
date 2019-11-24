'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
Define some simple operators for Tensorflow.
'''
import tensorflow as tf


def relu1(features, name='relu1'):
    with tf.variable_scope(name):
        features = tf.nn.relu(features)
        features = tf.clip_by_value(features,0 , 1)

    return features


def relu(features):
    return tf.nn.relu(features)


def safe_log(tensor, epsilon=1e-4):
    with tf.name_scope('safe_log'):
        tensor = tf.maximum(tensor, epsilon)
        tensor = tf.log(tensor)

    return tensor
