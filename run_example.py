'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
The simple example running on MNIST to testing the appoach of this paper:

    https://arxiv.org/abs/1802.04865

"Learning Confidence for Out-of-Distribution Detection in Neural Networks"
Terrance DeVries, Graham W. Taylor.
arXiv preprint.


The code is modified from

    https://github.com/aymericdamien/TensorFlow-Examples

I keep all the setting the same, just make it O.O. style.
'''
# pylint: disable=E1101
import numpy as np
import tensorflow as tf
from easydict import EasyDict

from conf_net import ops, visualize
from conf_net.base_conf import BaseConfNet
from conf_net.losses import neg_likelihoold_loss

WITH_CONF = True
LAMBDA    = 0.06
THRESHOLD = 0.1

class Trainer:
    """ Simple Tensorflow runner for running experimental example.
    """
    def __init__(self):
        self._paras = self._get_paras()
        self._build()

    
    def _get_paras(self):
        """ Define hyperparemeters
            [Returns]
                paras: Easydict (class like dictionary)
        """
        paras = {
            "learning_rate": 0.0002,
            "num_steps": 10000,
            "batch_size": 128,
            "display_step": 500,

            "n_hidden_1": 256,  # 1st layer number of neurons
            "n_hidden_2": 256,  # 2nd layer number of neurons
            "num_input": 784,   # MNIST data input (img shape: 28*28)
            "num_classes": 10   # MNIST total classes (0-9 digits)
        }

        return EasyDict(paras)


    def _get_data_gen(self):
        """ Load dataset 
            [Returns]
                mnist: tf.Dataset of MNIST, subsets of (train, validation, test) are provided.
        """
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        return mnist
    

    def _build(self):
        """ Build up the training graph.
        """
        # tf Graph input
        with tf.name_scope("placeholders"):
            self.x = tf.placeholder("float", [None, self._paras.num_input])
            self.y = tf.placeholder("float", [None, self._paras.num_classes])

        with tf.variable_scope("model"):
            hidden = tf.layers.dense(self.x, self._paras.n_hidden_1, activation=None)
            self.hidden = tf.layers.dense(hidden, self._paras.n_hidden_2, activation=None)
            
            self.logits = tf.layers.dense(self.hidden, self._paras.num_classes, activation=None)
            self.output = tf.nn.softmax(self.logits)

        if WITH_CONF:
            self._build_confidence_network()

        self._build_optimize()


    def _build_optimize(self):
        """ Build up the optimization graph.
        """
        # Define loss and optimizer
        output = self.conf_admin.hinting(self.output, self.y) if WITH_CONF else self.output
        with tf.name_scope("optimize"):
            self.loss_op = []
            loss_ce = tf.reduce_mean(neg_likelihoold_loss(pred=output, gt=self.y))
            self.loss_op.append(loss_ce)

            if WITH_CONF:
                loss_conf = self.conf_admin.calculate_loss_conf()
                self.loss_op.append(loss_conf)

            optimizer = tf.train.AdamOptimizer(learning_rate=self._paras.learning_rate) 
            self.train_op = optimizer.minimize(tf.reduce_sum(self.loss_op))

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
            if WITH_CONF:
                self.correlation = self.conf_admin.calculate_conf_correlation(
                    self.output, self.y, thres=THRESHOLD)

        self.init_op = tf.global_variables_initializer()


    def _build_confidence_network(self):
        """ Build up confidence branch.
        """
        self.conf_admin = BaseConfNet(lambda1=LAMBDA, cell_nums=[256])
        self.confidence = self.conf_admin.forward(self.hidden)


    def _train_step(self, sess, step=0):
        """ Define each of train step.
        """
        batch_x, batch_y = self.dataset.train.next_batch(self._paras.batch_size)

        metric_ops = (self.accuracy, self.correlation) if WITH_CONF else (self.accuracy)
        _, loss, metrics = sess.run([self.train_op, self.loss_op, metric_ops] , feed_dict={self.x: batch_x, self.y: batch_y})

        ### Logging to the terminal ###
        if step % self._paras.display_step == 0 or step == 1:
            log_msg  = "[Step {:d}] CE Loss = {:.4f}".format(step, loss[0])
            log_msg  = log_msg + ", Conf Loss = {:.4f}".format(loss[1]) if WITH_CONF else log_msg
            log_msg += ", Training Accuracy = {:.3f}".format(metrics[0])
            log_msg += ", (f1, precision, recall) = ({:.3f}, {:.3f}, {:.3f})".format(
                metrics[1][0], metrics[1][1], metrics[1][2])

            print (log_msg)
            

    def run(self):
        """ Run both training and testing.
        """
        self.dataset = self._get_data_gen()

        with tf.Session() as sess:
            ### Initialization ###
            sess.run(self.init_op)

            ### Run training ###
            for step in range(1, self._paras.num_steps+1):
                self._train_step(sess, step)

            print("Optimization Finished!")

            ### Testing for MNIST test images ###
            test_ops = [self.accuracy, self.correlation, (self.confidence, self.output, self.y)] if WITH_CONF else [self.accuracy]
            results = sess.run(test_ops,
                feed_dict={
                    self.x: self.dataset.test.images,
                    self.y: self.dataset.test.labels})
            
            ### Accuracy ###
            print("Testing Accuracy: {}".format(results[0]))

            if WITH_CONF:
                f1_score, precision, recall = results[1]
                conf, pred, gt = results[2]

                ### Confidence Analysis ###
                pred = np.argmax(pred, axis=-1)
                gt   = np.argmax(gt, axis=-1)

                correct_idx = np.where(np.abs(pred-gt) == 0) 
                wrong_idx   = np.where(np.abs(pred-gt) > 0)
                
                mean_conf_correct = np.mean(conf[correct_idx])
                mean_conf_wrong = np.mean(conf[wrong_idx])

                print ("Confidence (correct, wrong) = ({:2f}, {:2f})".format(mean_conf_correct, mean_conf_wrong))
                print ("Confidence (f1_score, precision, recall) = ({:2f}, {:2f}, {:2f})".format(f1_score, precision, recall))

                visualize.draw_confidence_histogram(conf[correct_idx], conf[wrong_idx])


            ### Saving graph to Tensorboard
            print ("Saving graph to {}".format("./logs"))
            writer = tf.summary.FileWriter(logdir='./logs', graph=tf.get_default_graph())
            writer.flush()



if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
