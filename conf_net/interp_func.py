'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
Define some simple interpolation functions.
'''
import tensorflow as tf


def linear(x, y, ratio):
    with tf.variable_scope('linear_interp'):
        x_ = x * ratio + y * (1 - ratio)

    return x_
