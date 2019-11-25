'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
The implementation of confidence for the outputs of given neural networks.

    https://arxiv.org/abs/1802.04865

"Learning Confidence for Out-of-Distribution Detection in Neural Networks"
Terrance DeVries, Graham W. Taylor.
arXiv preprint.
'''

import tensorflow as tf

from conf_net import interp_func, metrics, ops


class BaseConfNet():
    """ Confidence for the outputs of given neural networks.
        This implemention is based on:
        
            https://arxiv.org/abs/1802.04865

        "Learning Confidence for Out-of-Distribution Detection in Neural Networks"
        Terrance DeVries, Graham W. Taylor
    """
    def __init__(self, lambda1=1.0, cell_nums=[32], name='conf_net',
        half_batch=False, use_budget=False, use_sigmoide=True, use_log=True):
        """ [Args]
                target_layer: (not implemented now) [str]
                output: (not implemented now) [tf.Tensor]
                lambda1: the lambda value for the confidence loss. [float]
                cell_nums: list of cell numbers for the confidence sub-net. [list]
                half_batch: True for hinting on half of the batch, otherwise all the batch. [bool]
                use_budget: (not implemented now) True for tunnig lambda automatically with budget. [bool]
                use_sigmoide: True for use sigmoid at the end of confidence sub-net, otherwise relu1. [bool]
                use_log: True for use log likelihood for calculate the confidence loss, otherwise L1. [bool]
        """
        self._name         = name
        self._cell_nums    = cell_nums
        self._lambda1      = lambda1
        self._half_batch   = half_batch
        self._use_budget   = use_budget
        self._use_sigmoide = use_sigmoide
        self._use_log      = use_log

        self._built        = False


    def forward(self, x, reuse=False, name=None):
        """ Forward hidden features to the confidence sub-net.
            [Inputs]
                x: the hidden features forward to confidence sub-net [tf.Tensor]
            [Args]
                reuse: True for reuse the variables [str]
            [Return]
                confidence value 
        """
        name = self._name if name is None else name
        with tf.variable_scope(name, reuse=reuse):
            self.conf = self._forward(x)

        self._built = True

        return self.conf


    def hinting(self, output, gt, conf=None, func=None, name='hinting'):
        """ Apply hinting to the output with interpolation function
            [Inputs]
                output: the prediction of the network [tf.Tensor]
                gt: the ground truth label of the target to apply loss [tf.Tensor]
            [Args]
                conf: is not provided, use the saved confidence in the class [tf.Tensor]
                func: the function for interpolation [func]
                name: the scope name for hinting [str]
            [Return]
                interpolated prediction
        """
        conf = self.conf if conf is None else conf
        with tf.variable_scope(name):
            y = self._interpolate(x_p=output, x_gt=gt, conf=conf, func=func)
            if self._half_batch:
                bz = output.get_shape().as_list()[0]
                y = tf.concat(y[:bz//2,...], output[bz//2:,...], axis=0)      
        
        return y


    def calculate_loss_conf(self, conf=None, name='conf_loss'):
        """ Calculate the confidence loss
            [Args]
                conf: is not provided, use the saved confidence in the class [tf.Tensor]
                name: the scope name for hinting [str]
            [Return]
                The weighted confidence loss
        """
        conf = self.conf if conf is None else conf
        scope = tf.variable_scope if self._use_budget else tf.name_scope
        
        with scope(name):
            if self._use_log:
                conf_loss = tf.reduce_mean(-tf.log(conf))
            else:
                conf_loss = tf.reduce_mean(conf)

            if self._use_budget:
                raise NotImplementedError
            else:
                conf_loss = tf.multiply(self._lambda1, conf_loss)


        return conf_loss

    
    def calculate_conf_correlation(self, pred, label, conf=None, thres=0.6, metric_type='softmax', name='metric_correlation'):
        """ Calculate the correlation between low confidence and false prediction """
        conf = self.conf if conf is None else conf

        with tf.name_scope(name):
            metric = self._correlation_metric(pred, label, conf, thres, metric_type)

        return metric


    def _forward(self, x):
        """ Core process of network construction. 
            The function is seperated for better inherited modifying and graph visualization.
            [Unputs]
                x: the hidden features forward to confidence sub-net [tf.Tensor]
            [Return]
                confidence value 
        """
        activation = tf.nn.sigmoid if self._use_sigmoide else ops.relu1

        for cell_num in self._cell_nums:
            x = tf.layers.dense(x, units=cell_num, activation=activation)

        x = tf.layers.dense(x, units=1, activation=activation)

        return x


    def _interpolate(self, x_p, x_gt, conf, func=None):
        """ The interpolation function for prediction and label. 
            [Inputs]
                x_p: the prediction [tf.Tensor]
                x_gt: the gound truth label [tf.Tensor]
                conf: the confidence [tf.Tensor]
            [Args]
                func: the function for interpolation [func]
            [Returns]
                Interpolated prediction
        """
        if func is None:
            func = interp_func.linear

        y = func(x_p, x_gt, ratio=conf)

        return y


    def _correlation_metric(self, pred, label, conf, thres=0.6, metric_type='softmax'):
        if metric_type == 'softmax':
            metric = metrics.softmax_correlation(pred, label, conf, thres)
        else:
            raise NotImplementedError


        return metric
    



    def _budget(self):
        """ The budget for automatically tunning confidence lambda. """
        raise NotImplementedError
