import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

# load the MNIST data both for training and testing
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


class NetworkManager:
    def __init__(self, epochs, batch_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.biases_values = 0.1

        self.accuracy_value = False

        self.mnist = mnist_data.read_data_sets("MNIST_data/", one_hot=True)

        # each digit will be flattened into 784-dimensional vector containing float32 values
        # the size is [None, _] because batch size can be of any length
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        # input will be a tensor
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1], name='image')

        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],

                # initialize biases and weights
                biases_initializer=tf.constant_initializer(self.biases_values),
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        ):
            # define layers
            self.net = slim.conv2d(self.x_image, 16, [5, 5], scope='conv_1')

            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool_1')
            self.net = slim.conv2d(self.net, 32, [5, 5], scope='conv_2')
            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool_2')
            self.net = slim.conv2d(self.net, 64, [5, 5], scope='conv_3')
            self.net = slim.max_pool2d(self.net, [2, 2], scope='pool_3')
            self.net = slim.flatten(self.net)
            self.net = slim.fully_connected(self.net, 256, scope='fc_1')

            # activation function of the last(output) layer will be set later
            self.net = slim.fully_connected(self.net, 10, activation_fn=None, scope='fc2')

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.net, labels=self.y), name='cross_entropy')  # loss is a tensor
        self.loss_scalar = tf.summary.scalar('loss', self.loss)

        with tf.name_scope('accuracy'):
            self.prediction_tensor = tf.equal(tf.argmax(self.net, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction_tensor, tf.float32))
        self.accuracy_scalar = tf.summary.scalar('accuracy', self.accuracy)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.init = tf.global_variables_initializer()

        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=self.config)

    def train(self, cal_accuracy=True):
        self.sess.run(self.init)

        for e in range(self.epochs):

            # get next batch, fill the placeholders, and run the network's training operation
            batch_xs, batch_ys = self.mnist.train.next_batch(self.batch_size)
            _, loss_scalar_value = self.sess.run([self.train_op, self.loss_scalar], feed_dict={
                self.x: batch_xs, self.y: batch_ys,
            })

            # calculate accuracy of the network on a subset of testing dataset,
            # and print it to console
            if e % 200 == 0:
                test_xs, test_ys = self.mnist.test.next_batch(128)
                acc, acc_scalar_value = self.sess.run([self.accuracy, self.accuracy_scalar], feed_dict={
                    self.x: test_xs, self.y: test_ys
                })
                print("[%4d] Accuracy: %5.2f %%" % (e, acc * 100))

        if cal_accuracy:
            self.calculate_accuracy()

    def calculate_accuracy(self):
        # calculate accuracy of the neural network on the entire testing dataset
        self.accuracy_value = self.sess.run(self.accuracy, feed_dict={
            self.x: self.mnist.test.images,
            self.y: self.mnist.test.labels,
        })

    def predict(self, digit):
        digit = digit[np.newaxis, ...]

        # run the neural network to predict the digit and
        # apply 'softmax' to the output layer
        prediction = self.sess.run(tf.nn.softmax(self.net), feed_dict={self.x: digit})

        # get the answer
        answer = np.argmax(prediction)

        return prediction[0].tolist(), answer

    def get_accuracy(self):
        if not self.accuracy_value:
            self.calculate_accuracy()
        return self.accuracy_value * 100
