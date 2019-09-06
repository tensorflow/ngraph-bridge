# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
    nGraph TensorFlow Tensorboard NGraph/Stats test

    Tests TensorBoard integration
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import shutil

import sys
import tempfile
import getpass
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import ngraph_bridge
from common import NgraphTest


def deepnn(x):
    '''deepnn builds the graph for a deep net for classifying digits.'''

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10], "W_fc2")
        b_fc2 = bias_variable([10])

        # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv, tf.placeholder(tf.float32)


def conv2d(x, W):
    '''conv2d returns a 2d convolution layer with full stride.'''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    '''max_pool_2x2 downsamples a feature map by 2X.'''
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, name):
    '''weight_variable generates a weight variable of a given shape.'''
    weight_var = tf.get_variable(name, shape)
    return weight_var


def bias_variable(shape):
    '''bias_variable generates a bias variable of a given shape.'''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class TestTensorBoardNGraphStats(NgraphTest):
    '''
        Add TF scope names to various modules of the network (loss, accuracy, etc.), 
        and verify that subdirectories were created based off these scope names.
    '''
    @pytest.mark.skipif(
        not ngraph_bridge.is_grappler_enabled(), reason="Only for Grappler")
    def test_train_mnist_cnn(self):
        os.environ['NGRAPH_TF_TB_LOGDIR'] = './test'

        # Config
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            inter_op_parallelism_threads=1)
        # Enable the custom optimizer using the rewriter config options
        config = ngraph_bridge.update_config(config)

        # Import data
        mnist = input_data.read_data_sets(
            '/tmp/' + getpass.getuser() + 'tensorflow/mnist/input_data',
            one_hot=True)

        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        y_conv, keep_prob = deepnn(x)

        # guard loss/optimizer/accuracy with TF scopes
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_, logits=y_conv)
            cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope("adam_optimizer"):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)

            accuracy = tf.reduce_mean(correct_prediction)

        tf.summary.scalar('Training accuracy', accuracy)
        tf.summary.scalar('Loss function', cross_entropy)

        # save graph per environment variable location
        graph_location = os.environ['NGRAPH_TF_TB_LOGDIR']

        if (os.path.isdir(graph_location)):
            shutil.rmtree(graph_location)

        print('Saving graph to: %s' % graph_location)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            loss_values = []

            for i in range(300):
                batch = mnist.train.next_batch(100, shuffle=False)
                if i % 10 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0],
                        y_: batch[1],
                        keep_prob: 1.0
                    })

                _, summary, loss = sess.run([train_step, merged, cross_entropy],
                                            feed_dict={
                                                x: batch[0],
                                                y_: batch[1],
                                                keep_prob: 0.5
                                            })
                loss_values.append(loss)
                train_writer.add_summary(summary, i)

            # check test accuracy on 100 images
            num_test_images = 100
            x_test = mnist.test.images[:num_test_images]
            y_test = mnist.test.labels[:num_test_images]

            test_accuracy = accuracy.eval(feed_dict={
                x: x_test,
                y_: y_test,
                keep_prob: 1.0
            })

            saver.save(sess, './mnist_trained/')

        dirs = [o for o in os.listdir(graph_location)]

        os.environ.pop('NGRAPH_TF_TB_LOGDIR', None)

        # assert that dirs were created based on name of TF scope (initialized above)
        assert len(
            dirs) == 8  # 7 dirs, 1 TF event file generated by this script
        assert "ngraph0_init" in dirs
        assert "ngraph1_accuracy" in dirs
        assert "ngraph2_loss" in dirs
        assert "ngraph3_save" in dirs
        assert "stats0_init" in dirs
        assert "stats1_accuracy" in dirs
        assert "stats2_loss" in dirs

