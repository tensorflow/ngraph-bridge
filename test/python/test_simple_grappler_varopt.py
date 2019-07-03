# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""nGraph TensorFlow bridge sign operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import numpy as np
from common import NgraphTest


class TestSignOperations(NgraphTest):

    def test_sign_1d(self):
        dim1 = 3
        dim2 = 4
        a = tf.placeholder(tf.float32, shape=(dim1, dim2), name='a')
        x = tf.get_variable('x', [dim1, dim2], initializer=tf.zeros_initializer)
        b = tf.placeholder(tf.float32, shape=(dim1, dim2), name='y')
        c = a * x
        axpy = c + b
        train_step = x.assign(axpy)
        with tf.control_dependencies([train_step]):
            train_op = tf.no_op('train_op')

        def run_test(sess):
            sess.run(tf.global_variables_initializer())
            return sess.run(
                train_op,
                feed_dict={
                    a: np.ones((dim1, dim2)),
                    b: np.ones((dim1, dim2))
                })

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test))
