'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
Define some simple loss functions.
'''
import tensorflow as tf

from conf_net.ops import safe_log


def neg_likelihoold_loss(pred, gt, name='neg_likelihood'):
    with tf.name_scope(name):
        loss = -tf.reduce_sum((safe_log(pred) * gt), axis=-1)

    return loss
